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

# Virtual loss. When a batch descent traverses edge (s, a), the edge is
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


@dataclass
class _Node:
    """Per-expanded-state search statistics, all aligned to ``acts``.

    Replaces the old global ``(state, action)`` dicts (``q_values``,
    ``visit_counts``, ``virtual_visits``) and per-state dicts (``state_visits``,
    ``policy_priors``, ``valid_moves_cache``): each expanded state owns one
    record whose per-edge arrays are indexed by position in ``acts`` (the
    ascending legal action ids). ``_select_action`` reads these arrays directly,
    eliminating the per-edge ``dict.get`` gather that was ~32% of self-play time
    (see ``docs/plans/numba-hot-path.md`` N3). Not frozen — ``n``/``q``/
    ``virtual`` are mutated in place during backprop and ``n_total`` is bumped.
    """

    acts: NDArray[np.int32]      # ascending legal action ids
    priors: PolicyVector         # priors aligned to acts (float32; float64 after Dirichlet)
    n: NDArray[np.int32]         # visit counts per edge (0 == unvisited)
    q: NDArray[np.float64]       # running-mean Q per edge
    virtual: NDArray[np.int32]   # in-flight virtual loss per edge (0 between batches)
    n_total: int = 0             # total visits through this node (old state_visits[s])


@dataclass(frozen=True)
class _Descent:
    """Result of one root-to-leaf descent within a batch.

    ``needs_eval`` distinguishes an unexpanded leaf (``board`` set, awaiting a
    batched network evaluation) from a terminal leaf (``value`` set to the cached
    game-ended value, no network call). ``path`` is the list of traversed
    ``(node, edge_index)`` pairs — ``edge_index`` is the position into
    ``node.acts`` of the chosen action — used for backprop and virtual-loss
    removal without re-keying a dict.
    """

    key: StateKey
    board: IBoard | None
    path: list[tuple[_Node, int]]
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

        # Tree statistics. One :class:`_Node` per expanded state holds all the
        # per-edge stats (N/Q/virtual) and per-state priors/acts/total, stored
        # SPARSE — aligned to the node's legal-move array, never a dense
        # action_size vector (the dense versions were 99.6% of tree RAM, see
        # docs/plans/mcts-memory-reduction.md). This replaces the old global
        # ``(state, action)`` dicts; the read-only ``q_values`` / ``visit_counts``
        # / ``state_visits`` / ``policy_priors`` / ``valid_moves_cache`` /
        # ``virtual_visits`` properties below reconstruct the old views for
        # external consumers during the N3→N4 migration.
        self.nodes: dict[StateKey, _Node] = {}
        self.game_ended_cache: dict[StateKey, float] = {}

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

    # -- Read-only compatibility views -----------------------------------------
    # Reconstruct the pre-N3 dict surfaces from ``self.nodes`` so external
    # readers (core/players.py, scripts, tests) keep working during the
    # migration. These are READ-ONLY — N4 routes consumers onto dedicated
    # accessors and deletes them. Each rebuilds on access; only used off the
    # hot path. An edge is "present" iff visited (``n > 0``), matching the old
    # dicts where ``.get((s, a), default)`` returned the default for unvisited.

    @property
    def visit_counts(self) -> dict[StateAction, int]:
        return {(s, int(node.acts[i])): int(node.n[i])
                for s, node in self.nodes.items()
                for i in range(node.acts.shape[0]) if node.n[i] > 0}

    @property
    def q_values(self) -> dict[StateAction, float]:
        return {(s, int(node.acts[i])): float(node.q[i])
                for s, node in self.nodes.items()
                for i in range(node.acts.shape[0]) if node.n[i] > 0}

    @property
    def virtual_visits(self) -> dict[StateAction, int]:
        return {(s, int(node.acts[i])): int(node.virtual[i])
                for s, node in self.nodes.items()
                for i in range(node.acts.shape[0]) if node.virtual[i] != 0}

    @property
    def state_visits(self) -> dict[StateKey, int]:
        return {s: node.n_total for s, node in self.nodes.items()}

    @property
    def policy_priors(self) -> dict[StateKey, PolicyVector]:
        return {s: node.priors for s, node in self.nodes.items()}

    @property
    def valid_moves_cache(self) -> dict[StateKey, NDArray[np.int32]]:
        return {s: node.acts for s, node in self.nodes.items()}

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

        # Run simulations in batches of K. K=1 (default) collects one leaf
        # per batched NN call, reproducing single-leaf search exactly — virtual
        # loss is a no-op at K=1 and the selection/backprop arithmetic is
        # unchanged. K>1 collects K virtual-loss-diversified leaves per outer
        # step and evaluates them in a single predict_batch call.
        if self._profiling:
            sim_start = time.perf_counter()

        batch_size = max(1, self.config.mcts_batch_size)
        sims_remaining = self._move_sim_budget(canonical_board)
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
            num_valid = len(self.nodes[s].acts) if s in self.nodes else 0
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

        # Extract visit counts for all actions — the dense root distribution.
        # Same integer values as the old ``visit_counts.get((s, a), 0)`` probe
        # over the whole action space, but scattered from the root node's arrays
        # (no per-action dict lookup); ``.tolist()`` restores the plain-int list
        # the temperature/entropy code below expects.
        counts = self.root_visit_counts(canonical_board).tolist()

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

    def root_visit_counts(self, board: IBoard) -> NDArray[np.int64]:
        """Dense visit-count vector over the full action space for ``board``'s root.

        ``counts[a]`` is the MCTS visit count of action ``a`` at this root — 0 for
        unvisited or illegal actions and for an unexpanded root. The single source
        of the raw visit distribution, shared by :meth:`get_action_prob` and
        :class:`core.players.NetworkPlayer`; replaces the old
        ``[visit_counts.get((s, a), 0) for a in range(action_size)]`` probe.
        """
        counts = np.zeros(self.game.get_action_size(), dtype=np.int64)
        root = self.nodes.get(self.game.state_key(board))
        if root is not None:
            counts[root.acts] = root.n
        return counts

    def num_states(self) -> int:
        """Number of expanded states in the search tree (old ``len(state_visits)``)."""
        return len(self.nodes)

    def num_edges(self) -> int:
        """Number of visited edges (N > 0) across the tree (old ``len(q_values)``)."""
        return sum(int(np.count_nonzero(node.n)) for node in self.nodes.values())

    def _move_sim_budget(self, canonical_board: IBoard) -> int:
        """Simulations to spend on this move (IDEAS.md I1).

        ``"flat"`` (default) returns ``num_mcts_sims`` unchanged. ``"branching"``
        scales with the root's legal-move count, clamped to
        ``[sims_min, num_mcts_sims]`` — thinning the wasteful endgame (few moves)
        while capping the expensive opening at ``num_mcts_sims``. The branching
        count is read from the cache when the root is already expanded (the
        self-play case, where Dirichlet noise expands it), else computed with one
        move-gen call (no network).
        """
        cfg = self.config
        if cfg.sim_schedule != "branching":
            return cfg.num_mcts_sims
        s = self.game.state_key(canonical_board)
        if s in self.nodes:
            branching = len(self.nodes[s].acts)
        else:
            branching = int(np.count_nonzero(self.game.valid_move_masking(canonical_board, 1)))
        scaled = round(cfg.sim_branching_scale * branching)
        return max(cfg.sims_min, min(cfg.num_mcts_sims, scaled))

    def search(self, canonical_board: IBoard) -> float:
        """Perform one MCTS simulation (selection → expansion → evaluation → backprop).

        Retained as a public single-simulation entry point. Internally it runs
        a batch of size 1 through the same machinery ``get_action_prob`` uses,
        and returns the value backpropagated to the root (negated leaf value),
        matching the original recursive ``search`` return convention.
        """
        return self._simulate_batch(canonical_board, 1)[0]

    # -- Batched search internals ----------------------------------------------

    def _simulate_batch(self, root_board: IBoard, k: int) -> list[float]:
        """Run ``k`` simulations as one batch.

        Performs ``k`` virtual-loss-diversified descents from the root, a single
        batched neural-net evaluation of the freshly reached (unexpanded) leaves,
        then ``k`` backpropagations. Terminal leaves need no network call. The
        per-descent root values are returned in descent order (``search`` uses
        the first; ``get_action_prob`` discards them).

        At ``k == 1`` this is bit-for-bit equivalent to the original recursive
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
                if key not in self.nodes:
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

        Applies virtual loss to each traversed edge as it descends (by index into
        the node's ``virtual`` array), so a subsequent descent in the same batch
        is biased toward a different path. Returns the leaf plus the path taken
        (a list of ``(node, edge_index)``) so the caller can backprop and later
        undo the virtual loss.
        """
        board = root_board
        path: list[tuple[_Node, int]] = []
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
            node = self.nodes.get(s)
            if node is None:
                return _Descent(key=s, board=board, path=path,
                                value=None, needs_eval=True)

            # Internal node — pick a child and descend, depositing virtual loss.
            idx = self._select_action(node)
            a = int(node.acts[idx])
            node.virtual[idx] += VIRTUAL_LOSS
            path.append((node, idx))
            next_s, next_player = self.game.get_next_state(board, 1, a)
            board = self.game.get_canonical_form(next_s, next_player)

    def _select_action(self, node: _Node) -> int:
        """Return the index (into ``node.acts``) of the UCB-best valid action.

        Vectorised PUCT over the node's aligned per-edge arrays — no per-edge
        ``dict.get`` gather (that gather was ~32% of self-play; see
        ``docs/plans/numba-hot-path.md`` N3). Bit-identical to the previous
        gather-then-score version: same operation order, same float64 promotion,
        same sqrt values, same ``argmax`` first-max tie-break (``acts`` is
        ascending). When no edge of this node carries virtual loss (always the
        case at K=1) the score uses the exact original expression; when some do
        (K>1, mid-batch) the in-flight edges are scored as if they had ``virtual``
        extra visits each returning value -1, depressing their UCB so parallel
        descents diversify.
        """
        # ``priors`` is float32 (NN output); promote to float64 so the arithmetic
        # matches the original scalar loop, where ``cpuct * prior`` promoted the
        # float32 scalar to float64 (numpy 2.x NEP-50 would otherwise keep the
        # array float32, diverging from the golden).
        priors = node.priors.astype(np.float64, copy=False)  # aligned to acts
        state_visits = node.n_total
        cpuct = self.config.cpuct

        # n == 0 marks an unvisited edge: a visited edge always has n >= 1.
        n = node.n.astype(np.float64)
        q = node.q  # float64, aligned to acts; read-only here (not mutated)

        # sqrt terms are loop-invariant; visited edges use sqrt(N), first-visit
        # edges use sqrt(N + EPS). Both are correctly-rounded doubles == np.sqrt.
        sqrt_sv = math.sqrt(state_visits)
        sqrt_sv_eps = math.sqrt(state_visits + EPS)

        visited = n > 0
        if node.virtual.any():
            # Slow path: some of this node's edges carry in-flight virtual loss
            # (K>1 only). Scoping the check to this node (vs the old global "any
            # virtual loss anywhere") is bit-identical: when ``v`` is all zero the
            # ``np.where`` arms reduce exactly to the fast-path expressions.
            v = node.virtual.astype(np.float64)
            eff_n = n + v
            # Both ``np.where`` arms are evaluated for every edge, so the unused
            # arm divides by zero on unvisited/no-virtual lanes (eff_n == 0).
            # Those results are discarded by ``np.where``; silence the warning.
            with np.errstate(invalid="ignore", divide="ignore"):
                # ``v == 0`` visited must use plain q, not (n*q - 0)/n (not bit-equal).
                base = np.where(v == 0, q, (n * q - v) / eff_n)
                u_visited = base + cpuct * priors * sqrt_sv / (1 + eff_n)
                u_unvisited = np.where(
                    v == 0,
                    cpuct * priors * sqrt_sv_eps,
                    -1.0 + cpuct * priors * sqrt_sv / (1 + v),
                )
        else:
            # Fast path: no virtual loss on this node (always true at K=1).
            u_visited = q + cpuct * priors * sqrt_sv / (1 + n)
            u_unvisited = cpuct * priors * sqrt_sv_eps

        scores = np.where(visited, u_visited, u_unvisited)
        return int(np.argmax(scores))

    def _expand_leaf(self, board: IBoard, s: StateKey, priors: PolicyVector) -> None:
        """Expand a freshly evaluated leaf: mask its priors to legal moves,
        renormalise, and initialise its node statistics. Mirrors the original
        leaf-handling exactly (only the NN call is hoisted out into the batch)."""
        if self._detailed:
            vm_start = time.perf_counter()
        valids = self.game.valid_move_masking(board, 1)
        if self._detailed:
            self._total_valid_moves_time_s += time.perf_counter() - vm_start
            self._num_valid_moves_calls += 1

        # Store sparse: the legal action ids (ascending) and the priors at those
        # ids only, aligned. ``np.where`` is ascending, matching the order
        # ``_select_action`` iterates, so the values + order are identical to the
        # old dense ``priors * valids`` path — just without the zeros.
        acts = np.where(valids)[0].astype(np.int32)
        masked = priors[acts]
        sum_priors = np.sum(masked)
        if sum_priors > 0:
            masked = masked / sum_priors
        else:
            logger.error("All valid moves were masked, using uniform distribution.")
            masked = np.ones(len(acts), dtype=float) / len(acts)

        m = len(acts)
        self.nodes[s] = _Node(
            acts=acts,
            priors=masked,
            n=np.zeros(m, dtype=np.int32),
            q=np.zeros(m, dtype=np.float64),
            virtual=np.zeros(m, dtype=np.int32),
            n_total=0,
        )
        self._num_leaf_expansions += 1

    def _backprop(self, path: list[tuple[_Node, int]], leaf_value: float) -> float:
        """Propagate ``leaf_value`` (leaf-perspective) back up ``path``, flipping
        sign at each ply. Updates each edge's Q/N and the node's total with the
        same running-mean arithmetic as the original recursive unwind — now by
        index into the node's arrays instead of a ``(state, action)`` dict.
        ``n[idx] == 0`` is the first-visit sentinel (equivalent to the edge being
        absent from the old ``q_values`` dict). Returns the value as seen above
        the root (the negated leaf value for an empty path), matching the old
        ``search`` return value."""
        value = -leaf_value
        for node, idx in reversed(path):
            if node.n[idx] != 0:
                node.q[idx] = (
                    (node.n[idx] * node.q[idx] + value)
                    / (node.n[idx] + 1)
                )
                node.n[idx] += 1
            else:
                node.q[idx] = value
                node.n[idx] = 1
            node.n_total += 1
            value = -value
        return value

    def _remove_virtual_loss(self, path: list[tuple[_Node, int]]) -> None:
        """Undo the virtual loss a descent deposited along ``path``. Every
        increment in ``_descend_to_leaf`` is matched here, so each node's
        ``virtual`` array returns to all-zero once a batch is fully backpropped."""
        for node, idx in path:
            node.virtual[idx] -= VIRTUAL_LOSS

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
        if s not in self.nodes:
            if self._profiling:
                infer_start = time.perf_counter()
            priors, _ = self.nnet.predict(canonical_board)
            if self._profiling:
                self._total_inference_time_s += time.perf_counter() - infer_start
            self._expand_leaf(canonical_board, s, priors)

        node = self.nodes[s]
        epsilon = self.config.dirichlet_epsilon
        legal = node.acts  # ascending legal ids; node.priors is aligned to it
        if len(legal) == 0:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal))
        # Sparse priors are already aligned to ``legal``, so perturb element-wise
        # (the dense path fancy-indexed ``priors[legal]`` — same values, same order).
        priors = node.priors.copy()
        priors = (1 - epsilon) * priors + epsilon * noise
        node.priors = priors

    def _estimate_tree_memory_bytes(self) -> int:
        """Estimate memory used by the MCTS search tree.

        Counts the per-node aligned arrays (acts/priors/n/q/virtual) plus the
        ``nodes`` and ``game_ended_cache`` dict overhead. Approximate — used only
        for the profiling report.
        """
        total = sys.getsizeof(self.nodes) + sys.getsizeof(self.game_ended_cache)
        for node in self.nodes.values():
            total += (node.acts.nbytes + node.priors.nbytes + node.n.nbytes
                      + node.q.nbytes + node.virtual.nbytes)
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
            tree_size=len(self.nodes),
            total_valid_moves_time_s=self._total_valid_moves_time_s,
            total_game_ended_time_s=self._total_game_ended_time_s,
            num_valid_moves_calls=self._num_valid_moves_calls,
            num_game_ended_calls=self._num_game_ended_calls,
            tree_memory_bytes=self._estimate_tree_memory_bytes(),
            move_stats=tuple(self._move_stats),
            mean_policy_entropy=mean_entropy,
        )
