"""Profiling dataclasses for MCTS episodes.

Separate from the search algorithm so consumers that only need the
stats shape (the jax backend, reporting) do not import the search
machinery itself.
"""

from __future__ import annotations

from dataclasses import dataclass


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
