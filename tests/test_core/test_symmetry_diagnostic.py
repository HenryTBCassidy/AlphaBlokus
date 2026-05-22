"""End-to-end test for the symmetry-diagnostic helpers.

Exercises ``core.symmetry_diagnostic`` against a real (tiny, untrained)
TicTacToe network. We don't assert specific KL values — they depend on
random init — but we do assert that the result has the right shape and
that an artificially symmetric net produces zero divergence.
"""
from __future__ import annotations

import numpy as np

from core.config import RunConfig
from core.interfaces import IBoard, IGame
from core.symmetry_diagnostic import (
    build_diagnostic_positions,
    compute_symmetry_diagnostic,
)
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


def test_build_diagnostic_positions_deterministic_and_sized(ttt_game: TicTacToeGame) -> None:
    a = build_diagnostic_positions(ttt_game, n=3, seed=2026)
    b = build_diagnostic_positions(ttt_game, n=3, seed=2026)
    assert len(a) == 3 == len(b)
    # Same seed → same positions
    for board_a, board_b in zip(a, b, strict=True):
        assert np.array_equal(board_a.as_2d, board_b.as_2d)


def test_build_diagnostic_positions_first_is_empty_start_board(
    ttt_game: TicTacToeGame,
) -> None:
    """Index 0 is always the empty starting board — acts as a sanity check
    in the diagnostic (the start board is its own transpose, so KL must
    be ~0 there regardless of net behaviour).
    """
    positions = build_diagnostic_positions(ttt_game, n=5, seed=0)
    assert (positions[0].as_2d == 0).all(), (
        "first reference position should be the empty initial board"
    )


def test_build_diagnostic_positions_phases_are_mixed(
    ttt_game: TicTacToeGame,
) -> None:
    """Across the reference set, positions should span multiple game
    phases — i.e. placed-piece count should vary. The implementation
    samples rollout lengths from two ranges (3–22 early/mid, 23–32 late),
    so over 30 positions we expect clear variance in cells-filled.
    """
    positions = build_diagnostic_positions(ttt_game, n=30, seed=0)
    cells_filled = [int(np.abs(p.as_2d).sum()) for p in positions]
    distinct_depths = len(set(cells_filled))
    assert distinct_depths >= 2, (
        f"expected positions to span multiple game phases, got only "
        f"{distinct_depths} distinct fill counts"
    )


def test_build_diagnostic_positions_handles_small_n(ttt_game: TicTacToeGame) -> None:
    """n=1 returns just the start board. n=0 returns empty list."""
    one = build_diagnostic_positions(ttt_game, n=1, seed=0)
    assert len(one) == 1
    assert (one[0].as_2d == 0).all()

    zero = build_diagnostic_positions(ttt_game, n=0, seed=0)
    assert zero == []


def test_diagnostic_result_shape(
    ttt_game: TicTacToeGame, test_config: RunConfig,
) -> None:
    """Result has one KL per non-identity symmetry and a real-valued mean."""
    nnet = NNetWrapper(ttt_game, test_config)
    boards = build_diagnostic_positions(ttt_game, n=2, seed=0)
    result = compute_symmetry_diagnostic(nnet, ttt_game, boards[0], position_index=0)
    # TTT has 8 symmetries from D4 (with the codebase's enumeration), so 7
    # non-identity entries.
    assert len(result.kl_divergences) == 7
    assert isinstance(result.mean_kl, float)
    # KL is non-negative by definition.
    assert all(kl >= -1e-9 for kl in result.kl_divergences)


def test_perfectly_symmetric_net_scores_zero_kl(
    ttt_game: TicTacToeGame, test_config: RunConfig,
) -> None:
    """A network whose policy is exactly the uniform distribution is
    trivially equivariant — KL must be (very close to) zero on every
    symmetry. Sanity check that the diagnostic doesn't generate spurious
    non-zero values on a known-symmetric model.
    """

    class _UniformNet:
        """Stand-in INeuralNetWrapper that returns a uniform policy."""

        def predict(self, board: IBoard) -> tuple[np.ndarray, float]:
            n = ttt_game.get_action_size()
            return np.full(n, 1.0 / n, dtype=np.float32), 0.0

    boards = build_diagnostic_positions(ttt_game, n=1, seed=42)
    result = compute_symmetry_diagnostic(_UniformNet(), ttt_game, boards[0])
    assert all(kl < 1e-6 for kl in result.kl_divergences), (
        f"uniform-policy net should be exactly equivariant, got {result.kl_divergences}"
    )
    assert result.mean_kl < 1e-6
