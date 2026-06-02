import math
import sys
import time
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from core.config import MCTSConfig
from core.interfaces import IBoard, IGame, INeuralNetWrapper

# Constants
EPS: Final[float] = 1e-8  # Small constant to prevent division by zero

# F3 virtual loss. When a batch descent traverses edge (s, a), the edge is
# temporarily credited with VIRTUAL_LOSS extra visits each worth a value of -1
# (i.e. W -= VIRTUAL_LOSS, N += VIRTUAL_LOSS). This depresses the edge's UCB so
# the next descent in the same batch is steered down a different path, keeping
# the K collected leaves diversified. The loss is removed once the batch is
# backpropped. AlphaGo Zero (Silver et al. 2017) used n_vl = 3.
VIRTUAL_LOSS: Final[int] = 3

# Type aliases for improved readability
StateKey: TypeAlias = bytes  # Hashable key uniquely identifying a game state
Action: TypeAlias = int  # Integer index into the action space
StateAction: TypeAlias = tuple[StateKey, Action]  # (state, action) pair
PolicyVector: TypeAlias = NDArray[np.float64]  # Probability distribution over actions
ValidMoves: TypeAlias = NDArray[np.bool_]  # Binary mask of legal moves


@dataclass(frozen=True)
class MCTSMoveStats:
    """Per-move profiling snapshot. Only collected when profiling_level="detailed"."""

    move_number: int
    num_sims: int
    search_time_s: float
    inference_time_s: float
    valid_moves_time_s: float
    game_ended_time_s: float
    num_leaf_expansions: int
    num_valid_moves: int


@dataclass(frozen=True)
class MCTSEpisodeStats:
    """Accumulated profiling statistics for an MCTS episode (one game).

    MCTS is recreated per episode, so these counters naturally cover
    one complete game without needing a reset mechanism.
    """

    # Standard fields (always populated)
    num_moves: int  # Number of get_action_prob() calls (moves played)
    total_sims: int  # Total individual search() calls across all moves
    total_search_time_s: float  # Wall time spent in simulation loops
    total_inference_time_s: float  # Wall time spent in nnet.predict() calls
    num_leaf_expansions: int  # New leaf nodes added to the search tree
    tree_size: int  # Number of unique states in the tree (len(state_visits))

    # Detailed fields (populated when profiling_level="detailed")
    total_valid_moves_time_s: float = 0.0
    total_game_ended_time_s: float = 0.0
    num_valid_moves_calls: int = 0
    num_game_ended_calls: int = 0
    tree_memory_bytes: int = 0  # Approximate memory used by MCTS tree dictionaries
    move_stats: tuple[MCTSMoveStats, ...] = ()

    # Diagnostic — mean entropy (nats) of the raw MCTS visit-count distribution
    # across moves in this episode. Computed on the pre-temperature
    # ``visit_counts / sum(visit_counts)`` so it reflects search confidence,
    # not how the temperature sampled.
    mean_policy_entropy: float = 0.0


@dataclass(frozen=True)
class _Descent:
    """Result of one root-to-leaf descent within a batch.

    ``needs_eval`` distinguishes an unexpanded leaf (``board`` set, awaiting a
    batched network evaluation) from a terminal leaf (``value`` set to the cached
    game-ended value, no network call). ``path`` is the list of traversed
    ``(state, action)`` edges, used for backprop and virtual-loss removal.
    """

    key: StateKey
    board: IBoard | None
    path: list[StateAction]
    value: float | None
    needs_eval: bool


class MCTS:
    """
    Monte Carlo Tree Search implementation for game playing.

    This class implements the MCTS algorithm with neural network guidance,
    following the AlphaZero approach. The search is guided by:
    1. Prior probabilities from a policy network
    2. Value estimates from a value network
    3. Visit counts for exploration/exploitation balance

    The search tree is implicitly stored through dictionaries mapping:
    - States to their neural network policy predictions
    - State-action pairs to their Q-values and visit counts
    - States to their total visit counts

    The tree is built incrementally through repeated simulations, each consisting of:
    1. Selection: Choose actions recursively using UCB formula
    2. Expansion: Add a new leaf node to the tree
    3. Evaluation: Use neural network to evaluate the leaf
    4. Backpropagation: Update statistics for all visited nodes
    """

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, config: MCTSConfig) -> None:
        self.game = game
        self.nnet = nnet
        self.config = config
        self._detailed = config.profiling_level == "detailed"
        self._profiling = config.profiling_level != "none"

        # Tree statistics
        self.q_values: dict[StateAction, float] = {}
        self.visit_counts: dict[StateAction, int] = {}
        self.state_visits: dict[StateKey, int] = {}
        self.policy_priors: dict[StateKey, PolicyVector] = {}
        self.game_ended_cache: dict[StateKey, float] = {}
        self.valid_moves_cache: dict[StateKey, ValidMoves] = {}

        # F3 virtual-loss counters. Maps (state, action) -> in-flight virtual
        # visit count during a batch. Always empty between get_action_prob
        # calls (every increment is matched by a decrement after backprop).
        self.virtual_visits: dict[StateAction, int] = {}

        # Standard profiling counters
        self._total_search_time_s: float = 0.0
        self._total_inference_time_s: float = 0.0
        self._total_sims: int = 0
        self._num_leaf_expansions: int = 0
        self._num_moves: int = 0

        # Detailed profiling counters
        self._total_valid_moves_time_s: float = 0.0
        self._total_game_ended_time_s: float = 0.0
        self._num_valid_moves_calls: int = 0
        self._num_game_ended_calls: int = 0
        self._move_stats: list[MCTSMoveStats] = []
        self._policy_entropies: list[float] = []

    # -- Public methods --------------------------------------------------------

    def get_action_prob(
        self, canonical_board: IBoard, temp: float = 1, add_root_noise: bool = False,
    ) -> list[float]:
        """Get action probabilities for the current board state.

        Performs multiple MCTS simulations and returns a probability distribution
        over possible actions based on the visit counts of each action.

        Args:
            canonical_board: Root position (current player's perspective).
            temp: Temperature for the final visit-count → probability sampling.
            add_root_noise: If True (self-play only), mix Dirichlet exploration
                noise into the root priors before searching — gated by
                ``MCTSConfig.dirichlet_epsilon`` (0 = no-op). Arena/Elo leave it
                False so evaluation play is noise-free.
        """
        self._num_moves += 1

        if add_root_noise and self.config.dirichlet_epsilon > 0:
            self._apply_root_dirichlet_noise(canonical_board)

        # Snapshot counters before this move (for per-move delta)
        if self._detailed:
            pre_sims = self._total_sims
            pre_inference = self._total_inference_time_s
            pre_valid_moves = self._total_valid_moves_time_s
            pre_game_ended = self._total_game_ended_time_s
            pre_leaf = self._num_leaf_expansions

        # Run simulations in batches of K (F3). K=1 (default) collects one leaf
        # per batched NN call, reproducing the pre-F3 single-leaf search exactly
        # — virtual loss is a no-op at K=1 and the selection/backprop arithmetic
        # is unchanged. K>1 collects K virtual-loss-diversified leaves per outer
        # step and evaluates them in a single predict_batch call.
        if self._profiling:
            sim_start = time.perf_counter()

        batch_size = max(1, self.config.mcts_batch_size)
        sims_remaining = self.config.num_mcts_sims
        while sims_remaining > 0:
            k = min(batch_size, sims_remaining)
            self._simulate_batch(canonical_board, k)
            self._total_sims += k
            sims_remaining -= k

        if self._profiling:
            self._total_search_time_s += time.perf_counter() - sim_start

        # Record per-move stats
        if self._detailed:
            s = self.game.state_key(canonical_board)
            num_valid = int(self.valid_moves_cache[s].sum()) if s in self.valid_moves_cache else 0
            self._move_stats.append(MCTSMoveStats(
                move_number=self._num_moves,
                num_sims=self._total_sims - pre_sims,
                search_time_s=(time.perf_counter() - sim_start) if self._profiling else 0.0,
                inference_time_s=self._total_inference_time_s - pre_inference,
                valid_moves_time_s=self._total_valid_moves_time_s - pre_valid_moves,
                game_ended_time_s=self._total_game_ended_time_s - pre_game_ended,
                num_leaf_expansions=self._num_leaf_expansions - pre_leaf,
                num_valid_moves=num_valid,
            ))

        # Extract visit counts for all actions
        s = self.game.state_key(canonical_board)
        counts = [self.visit_counts.get((s, a), 0) for a in range(self.game.get_action_size())]

        # Record search-confidence entropy on the *raw* visit distribution
        # (before temperature sampling). Temperature controls how we *sample*
        # from this distribution; the distribution itself is what MCTS settled on.
        raw_total = float(sum(counts))
        if raw_total > 0:
            raw_probs = np.array(counts, dtype=float) / raw_total
            nonzero = raw_probs[raw_probs > 0]
            self._policy_entropies.append(float(-np.sum(nonzero * np.log(nonzero))))

        # Handle temperature=0 case (deterministic best action)
        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        # Apply temperature and normalise to get probabilities
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        return [x / counts_sum for x in counts]

    def search(self, canonical_board: IBoard) -> float:
        """Perform one MCTS simulation (selection → expansion → evaluation → backprop).

        Retained as a public single-simulation entry point. Internally it runs
        a batch of size 1 through the same machinery ``get_action_prob`` uses,
        and returns the value backpropagated to the root (negated leaf value),
        matching the pre-F3 recursive ``search`` return convention.
        """
        return self._simulate_batch(canonical_board, 1)[0]

    # -- F3 batched search internals -------------------------------------------

    def _simulate_batch(self, root_board: IBoard, k: int) -> list[float]:
        """Run ``k`` simulations as one batch.

        Performs ``k`` virtual-loss-diversified descents from the root, a single
        batched neural-net evaluation of the freshly reached (unexpanded) leaves,
        then ``k`` backpropagations. Terminal leaves need no network call. The
        per-descent root values are returned in descent order (``search`` uses
        the first; ``get_action_prob`` discards them).

        At ``k == 1`` this is bit-for-bit equivalent to the pre-F3 recursive
        search: a single descent applies virtual loss that affects no other
        descent and is removed immediately, so selection and backprop are
        identical to the original arithmetic.
        """
        descents = [self._descend_to_leaf(root_board) for _ in range(k)]

        # Collect the distinct unexpanded leaves needing a network evaluation.
        # Deduplicating means the GPU sees each unique position once even when
        # several descents collide on the same leaf (common early-game); the
        # shared result is reused by every colliding descent's backprop.
        eval_keys: list[StateKey] = []
        eval_boards: list[IBoard] = []
        seen: set[StateKey] = set()
        for descent in descents:
            if descent.needs_eval and descent.key not in seen:
                seen.add(descent.key)
                eval_keys.append(descent.key)
                eval_boards.append(descent.board)

        values_by_key: dict[StateKey, float] = {}
        if eval_boards:
            if self._profiling:
                infer_start = time.perf_counter()
            priors_list, values_list = self.nnet.predict_batch(eval_boards)
            if self._profiling:
                self._total_inference_time_s += time.perf_counter() - infer_start
            for key, board, priors, value in zip(
                eval_keys, eval_boards, priors_list, values_list, strict=True,
            ):
                values_by_key[key] = value
                if key not in self.policy_priors:
                    self._expand_leaf(board, key, priors)

        # Backprop every descent in descent order (deterministic), removing the
        # virtual loss it deposited along the way.
        root_values: list[float] = []
        for descent in descents:
            leaf_value = descent.value if not descent.needs_eval else values_by_key[descent.key]
            root_values.append(self._backprop(descent.path, leaf_value))
            self._remove_virtual_loss(descent.path)
        return root_values

    def _descend_to_leaf(self, root_board: IBoard) -> _Descent:
        """Descend from the root picking the UCB-best action (with virtual loss)
        at each expanded node until reaching a terminal or unexpanded leaf.

        Applies virtual loss to each traversed ``(state, action)`` edge as it
        descends, so a subsequent descent in the same batch is biased toward a
        different path. Returns the leaf plus the path taken so the caller can
        backprop and later undo the virtual loss.
        """
        board = root_board
        path: list[StateAction] = []
        while True:
            s = self.game.state_key(board)

            # Game-ended check (cached per state).
            if s not in self.game_ended_cache:
                if self._detailed:
                    t0 = time.perf_counter()
                self.game_ended_cache[s] = self.game.get_game_ended(board, 1)
                if self._detailed:
                    self._total_game_ended_time_s += time.perf_counter() - t0
                    self._num_game_ended_calls += 1
            if self.game_ended_cache[s] != 0:
                # Terminal leaf — value is known, no network evaluation needed.
                return _Descent(key=s, board=None, path=path,
                                value=self.game_ended_cache[s], needs_eval=False)

            # Unexpanded leaf — defer evaluation to the batched NN call.
            if s not in self.policy_priors:
                return _Descent(key=s, board=board, path=path,
                                value=None, needs_eval=True)

            # Internal node — pick a child and descend, depositing virtual loss.
            a = self._select_action(s)
            self.virtual_visits[(s, a)] = self.virtual_visits.get((s, a), 0) + VIRTUAL_LOSS
            path.append((s, a))
            next_s, next_player = self.game.get_next_state(board, 1, a)
            board = self.game.get_canonical_form(next_s, next_player)

    def _select_action(self, s: StateKey) -> Action:
        """Return the UCB-best valid action at expanded state ``s``.

        When an edge carries no virtual loss (always the case at K=1) the score
        uses the exact pre-F3 expression so the choice is bit-identical. When an
        edge is in-flight (virtual visits > 0) it is scored as if it had
        ``virtual_visits`` extra visits each returning value -1, depressing its
        UCB so parallel descents diversify.
        """
        valids = self.valid_moves_cache[s]
        priors = self.policy_priors[s]
        state_visits = self.state_visits[s]
        cpuct = self.config.cpuct
        cur_best = -float('inf')
        best_act = -1

        for a in np.where(valids)[0]:
            virtual = self.virtual_visits.get((s, a), 0)
            if (s, a) in self.q_values:
                n = self.visit_counts[(s, a)]
                if virtual == 0:
                    u = (self.q_values[(s, a)] +
                         cpuct * priors[a] * math.sqrt(state_visits) / (1 + n))
                else:
                    effective_n = n + virtual
                    effective_q = (n * self.q_values[(s, a)] - virtual) / effective_n
                    u = (effective_q +
                         cpuct * priors[a] * math.sqrt(state_visits) / (1 + effective_n))
            else:
                if virtual == 0:
                    u = cpuct * priors[a] * math.sqrt(state_visits + EPS)
                else:
                    # No real visits yet, only in-flight: effective Q is -1.
                    u = (-1.0 +
                         cpuct * priors[a] * math.sqrt(state_visits) / (1 + virtual))

            if u > cur_best:
                cur_best = u
                best_act = a

        return best_act

    def _expand_leaf(self, board: IBoard, s: StateKey, priors: PolicyVector) -> None:
        """Expand a freshly evaluated leaf: mask its priors to legal moves,
        renormalise, and initialise its node statistics. Mirrors the pre-F3
        leaf-handling exactly (only the NN call is hoisted out into the batch)."""
        if self._detailed:
            vm_start = time.perf_counter()
        valids = self.game.valid_move_masking(board, 1)
        if self._detailed:
            self._total_valid_moves_time_s += time.perf_counter() - vm_start
            self._num_valid_moves_calls += 1

        masked = priors * valids
        sum_priors = np.sum(masked)
        if sum_priors > 0:
            masked = masked / sum_priors
        else:
            logger.error("All valid moves were masked, using uniform distribution.")
            masked = valids.astype(float)
            masked = masked / np.sum(masked)

        self.policy_priors[s] = masked
        self.valid_moves_cache[s] = valids
        self.state_visits[s] = 0
        self._num_leaf_expansions += 1

    def _backprop(self, path: list[StateAction], leaf_value: float) -> float:
        """Propagate ``leaf_value`` (leaf-perspective) back up ``path``, flipping
        sign at each ply. Updates Q-values, visit counts and state visits with
        the same running-mean arithmetic as the pre-F3 recursive unwind. Returns
        the value as seen above the root (the negated leaf value for an empty
        path), matching the old ``search`` return value."""
        value = -leaf_value
        for s, a in reversed(path):
            if (s, a) in self.q_values:
                self.q_values[(s, a)] = (
                    (self.visit_counts[(s, a)] * self.q_values[(s, a)] + value)
                    / (self.visit_counts[(s, a)] + 1)
                )
                self.visit_counts[(s, a)] += 1
            else:
                self.q_values[(s, a)] = value
                self.visit_counts[(s, a)] = 1
            self.state_visits[s] += 1
            value = -value
        return value

    def _remove_virtual_loss(self, path: list[StateAction]) -> None:
        """Undo the virtual loss a descent deposited along ``path``. Every
        increment in ``_descend_to_leaf`` is matched here, so ``virtual_visits``
        returns to empty once a batch is fully backpropped."""
        for s, a in path:
            remaining = self.virtual_visits.get((s, a), 0) - VIRTUAL_LOSS
            if remaining > 0:
                self.virtual_visits[(s, a)] = remaining
            else:
                self.virtual_visits.pop((s, a), None)

    def _apply_root_dirichlet_noise(self, canonical_board: IBoard) -> None:
        """Mix Dirichlet noise into the root's priors (AlphaZero exploration).

        Ensures the root is expanded (its priors set), then replaces them over
        the legal moves with ``(1-ε)·prior + ε·Dir(α)``. Self-play only, gated by
        ``dirichlet_epsilon > 0`` at the call site. Applied once per move, before
        the simulations, so every descent this move sees the noised priors. Uses
        the episode's seeded RNG, so it stays reproducible.
        """
        s = self.game.state_key(canonical_board)

        # A terminal root has no policy to perturb.
        if s not in self.game_ended_cache:
            self.game_ended_cache[s] = self.game.get_game_ended(canonical_board, 1)
        if self.game_ended_cache[s] != 0:
            return

        # Ensure the root is expanded so its priors exist (mirrors the leaf path:
        # NN eval → mask → store). Cheap one-off per move; the sims then reuse it.
        if s not in self.policy_priors:
            if self._profiling:
                infer_start = time.perf_counter()
            priors, _ = self.nnet.predict(canonical_board)
            if self._profiling:
                self._total_inference_time_s += time.perf_counter() - infer_start
            self._expand_leaf(canonical_board, s, priors)

        epsilon = self.config.dirichlet_epsilon
        legal = np.where(self.valid_moves_cache[s])[0]
        if len(legal) == 0:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal))
        priors = self.policy_priors[s].copy()
        priors[legal] = (1 - epsilon) * priors[legal] + epsilon * noise
        self.policy_priors[s] = priors

    def _estimate_tree_memory_bytes(self) -> int:
        """Estimate memory used by MCTS tree dictionaries.

        Counts the approximate size of all cached state data: Q-values,
        visit counts, policy priors, valid moves masks, and game-ended cache.
        """
        total = 0
        # Dict overhead + entries
        total += sys.getsizeof(self.q_values)
        total += sys.getsizeof(self.visit_counts)
        total += sys.getsizeof(self.state_visits)
        total += sys.getsizeof(self.game_ended_cache)
        # Policy priors: each is a numpy array (action_size floats)
        for arr in self.policy_priors.values():
            total += arr.nbytes
        total += sys.getsizeof(self.policy_priors)
        # Valid moves cache: each is a numpy array (action_size bools)
        for arr in self.valid_moves_cache.values():
            total += arr.nbytes
        total += sys.getsizeof(self.valid_moves_cache)
        return total

    def get_episode_stats(self) -> MCTSEpisodeStats:
        """Return accumulated profiling statistics for this MCTS instance."""
        mean_entropy = float(np.mean(self._policy_entropies)) if self._policy_entropies else 0.0
        return MCTSEpisodeStats(
            num_moves=self._num_moves,
            total_sims=self._total_sims,
            total_search_time_s=self._total_search_time_s,
            total_inference_time_s=self._total_inference_time_s,
            num_leaf_expansions=self._num_leaf_expansions,
            tree_size=len(self.state_visits),
            total_valid_moves_time_s=self._total_valid_moves_time_s,
            total_game_ended_time_s=self._total_game_ended_time_s,
            num_valid_moves_calls=self._num_valid_moves_calls,
            num_game_ended_calls=self._num_game_ended_calls,
            tree_memory_bytes=self._estimate_tree_memory_bytes(),
            move_stats=tuple(self._move_stats),
            mean_policy_entropy=mean_entropy,
        )
