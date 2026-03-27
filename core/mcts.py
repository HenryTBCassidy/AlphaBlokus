import math
import sys
import time
from dataclasses import dataclass, field
from typing import Final, TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from core.config import MCTSConfig
from core.interfaces import IBoard, IGame, INeuralNetWrapper

# Constants
EPS: Final[float] = 1e-8  # Small constant to prevent division by zero

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

    # -- Public methods --------------------------------------------------------

    def get_action_prob(self, canonical_board: IBoard, temp: float = 1) -> list[float]:
        """Get action probabilities for the current board state.

        Performs multiple MCTS simulations and returns a probability distribution
        over possible actions based on the visit counts of each action.
        """
        self._num_moves += 1

        # Snapshot counters before this move (for per-move delta)
        if self._detailed:
            pre_sims = self._total_sims
            pre_inference = self._total_inference_time_s
            pre_valid_moves = self._total_valid_moves_time_s
            pre_game_ended = self._total_game_ended_time_s
            pre_leaf = self._num_leaf_expansions

        # Run simulations
        if self._profiling:
            sim_start = time.perf_counter()

        for _ in range(self.config.num_mcts_sims):
            self.search(canonical_board)
            self._total_sims += 1

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
        """Perform one iteration of MCTS (selection → expansion → evaluation → backprop)."""
        s = self.game.state_key(canonical_board)

        # PHASE 1: Check if game has ended
        if s not in self.game_ended_cache:
            if self._detailed:
                t0 = time.perf_counter()
            self.game_ended_cache[s] = self.game.get_game_ended(canonical_board, 1)
            if self._detailed:
                self._total_game_ended_time_s += time.perf_counter() - t0
                self._num_game_ended_calls += 1
        if self.game_ended_cache[s] != 0:
            return -self.game_ended_cache[s]

        # PHASE 2: Leaf node - evaluate position with neural network
        if s not in self.policy_priors:
            if self._profiling:
                infer_start = time.perf_counter()
            self.policy_priors[s], v = self.nnet.predict(canonical_board)
            if self._profiling:
                self._total_inference_time_s += time.perf_counter() - infer_start
            self._num_leaf_expansions += 1

            if self._detailed:
                vm_start = time.perf_counter()
            valids = self.game.valid_move_masking(canonical_board, 1)
            if self._detailed:
                self._total_valid_moves_time_s += time.perf_counter() - vm_start
                self._num_valid_moves_calls += 1

            # Mask invalid moves and renormalise probabilities
            self.policy_priors[s] = self.policy_priors[s] * valids
            sum_priors = np.sum(self.policy_priors[s])

            if sum_priors > 0:
                self.policy_priors[s] /= sum_priors
            else:
                logger.error("All valid moves were masked, using uniform distribution.")
                self.policy_priors[s] = valids.astype(float)
                self.policy_priors[s] /= np.sum(self.policy_priors[s])

            # Initialize node statistics
            self.valid_moves_cache[s] = valids
            self.state_visits[s] = 0
            return -v

        # PHASE 3: Choose action according to UCB formula
        valids = self.valid_moves_cache[s]
        cur_best = -float('inf')
        best_act = -1

        for a in np.where(valids)[0]:
                if (s, a) in self.q_values:
                    u = (self.q_values[(s, a)] +
                         self.config.cpuct * self.policy_priors[s][a] *
                         math.sqrt(self.state_visits[s]) / (1 + self.visit_counts[(s, a)]))
                else:
                    u = (self.config.cpuct * self.policy_priors[s][a] *
                         math.sqrt(self.state_visits[s] + EPS))

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # PHASE 4: Recursively evaluate chosen action
        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        # Update node statistics
        if (s, a) in self.q_values:
            self.q_values[(s, a)] = ((self.visit_counts[(s, a)] * self.q_values[(s, a)] + v) /
                                    (self.visit_counts[(s, a)] + 1))
            self.visit_counts[(s, a)] += 1
        else:
            self.q_values[(s, a)] = v
            self.visit_counts[(s, a)] = 1

        self.state_visits[s] += 1
        return -v

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
        )
