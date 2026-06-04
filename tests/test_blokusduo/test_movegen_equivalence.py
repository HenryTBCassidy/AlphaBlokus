"""Equivalence tests between move-gen implementations.

This test compares the *current* ``valid_move_masking`` against the
new table-driven implementation across a large set of stratified random
positions. If they ever disagree, the new implementation is buggy.

Right now there is no new implementation yet. So this test compares the
current implementation against itself, which is trivially true.
**That's deliberate**: the infrastructure needs to exist and the cache
needs to be live *before* the new implementation is written, so we have
a working oracle from day one.

When the new generator lands, replace ``new_implementation`` below with
a call into it. The rest of the test machinery stays the same.

Fixtures live at ``tests/fixtures/blokus_duo_positions/dev_5000.npz``
(5,000 stratified positions). See
``tests/fixtures/blokus_positions.py`` for the generator, cache format,
and stratification rationale.

The 50,000-position gauntlet (``gauntlet_50000.npz``) is a separate,
slow test marked with ``pytest.mark.slow`` that runs once before
switch-over rather than on every pytest invocation.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from games.blokusduo.game import BlokusDuoGame
from tests.fixtures.blokus_positions import (
    PAD_ACTION,
    load_cache,
    replay_to_board_and_player,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from games.blokusduo.board import BlokusDuoBoard


# ---------------------------------------------------------------------------
# Fixture cache paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "blokus_duo_positions"
DEV_CACHE = _FIXTURE_DIR / "dev_5000.npz"
GAUNTLET_CACHE = _FIXTURE_DIR / "gauntlet_50000.npz"


# ---------------------------------------------------------------------------
# The current and "new" move-gen implementations
# ---------------------------------------------------------------------------

def current_valid_moves(game: BlokusDuoGame, board: BlokusDuoBoard, player: int) -> NDArray:
    """The current array-based move-gen — the reference we're checking against."""
    return game.valid_move_masking(board, player)


def new_valid_moves(game: BlokusDuoGame, board: BlokusDuoBoard, player: int) -> NDArray:
    """The precomputed-table-driven move generator.

    Returns a bool mask of legal actions (shape ``(17837,)``). When this
    test passes, it means the new implementation agrees with the old
    one on every position in the cache. When it fails, the helper
    :func:`_describe_mismatch` formats the discrepancy.
    """
    from games.blokusduo.movegen_runtime import get_default_generator
    mask_bool = get_default_generator().valid_move_mask(game, board, player)
    # Coerce to numerical mask matching the current impl's output dtype
    # (game.valid_move_masking returns float for historical reasons).
    return mask_bool.astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_action_set(mask: NDArray) -> frozenset[int]:
    """Convert a binary action mask to a hashable set of action IDs."""
    return frozenset(int(a) for a in np.where(mask > 0)[0])


def _describe_mismatch(
    sequence: NDArray,
    player: int,
    current_set: frozenset[int],
    new_set: frozenset[int],
) -> str:
    """Build a human-readable failure message for a single-position mismatch."""
    only_current = current_set - new_set
    only_new = new_set - current_set
    real_actions = [int(a) for a in sequence if int(a) != PAD_ACTION]
    return (
        "Move-gen disagreement.\n"
        f"  Position reached by action sequence: {real_actions}\n"
        f"  Player to move: {player}\n"
        f"  Only in current impl (false negatives in new): "
        f"{sorted(only_current)[:10]}{' …' if len(only_current) > 10 else ''}\n"
        f"  Only in new impl (false positives):           "
        f"{sorted(only_new)[:10]}{' …' if len(only_new) > 10 else ''}\n"
        f"  Sizes: current={len(current_set)}, new={len(new_set)}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blokus_game_module() -> BlokusDuoGame:
    """Module-scoped game instance — pieces.json parsing isn't free."""
    return BlokusDuoGame(
        pieces_config_path=Path(__file__).resolve().parent.parent.parent
        / "games" / "blokusduo" / "pieces.json",
    )


def test_dev_cache_exists() -> None:
    """The dev cache must be committed alongside this test.

    Regenerate via:
        uv run python -c "from pathlib import Path; \\
            from games.blokusduo.game import BlokusDuoGame; \\
            from tests.fixtures.blokus_positions import build_cache; \\
            build_cache(BlokusDuoGame(pieces_config_path=Path('games/blokusduo/pieces.json')), \\
                        n=5_000, seed=42, label='dev_5000')"
    """
    assert DEV_CACHE.exists(), (
        f"Dev cache missing at {DEV_CACHE}. Regenerate per the docstring."
    )


def test_dev_cache_shape(blokus_game_module: BlokusDuoGame) -> None:
    """Sanity-check the cache's shape and stratification.

    Catches accidental regenerations with wrong settings — e.g. someone
    rebuilds the cache with seed=0 or n=100. Test fails early so they
    notice before the equivalence test gives misleading results.
    """
    actions, n_moves = load_cache(DEV_CACHE)
    assert len(n_moves) == 5_000, f"Expected 5,000 positions, got {len(n_moves)}"

    # Stratification: at least a few positions in each phase bucket.
    assert (n_moves == 0).sum() >= 100, "Too few empty-board positions"
    assert ((n_moves >= 1) & (n_moves <= 8)).sum() >= 1_000, "Too few early-game positions"
    assert ((n_moves >= 9) & (n_moves <= 18)).sum() >= 1_000, "Too few mid-game positions"
    assert n_moves.max() >= 20, "No late-game positions; stratification broken"


def test_movegen_equivalence_dev_cache(blokus_game_module: BlokusDuoGame) -> None:
    """Compare current vs new implementation on every cached dev position.

    Currently trivially passes (both call the current impl). Becomes
    a real test the moment the new implementation is swapped in.

    Runs in ~10 seconds against the 5,000-position cache because
    replay is fast and we're not generating positions on the fly.
    """
    actions_array, n_moves_array = load_cache(DEV_CACHE)
    mismatches: list[str] = []

    for i in range(len(n_moves_array)):
        n_moves = int(n_moves_array[i])
        sequence = actions_array[i, :n_moves]

        # Replay independently for each implementation. Doing two
        # separate replays — rather than computing the board once and
        # passing it to both impls — defends against either impl
        # mutating the board (it shouldn't, but the equivalence test
        # is the place to be paranoid).
        board_current, player_current = replay_to_board_and_player(
            blokus_game_module, sequence,
        )
        board_new, player_new = replay_to_board_and_player(
            blokus_game_module, sequence,
        )
        assert player_current == player_new, (
            f"Player-to-move drift at position {i}: {player_current} vs {player_new}"
        )

        current_mask = current_valid_moves(
            blokus_game_module, board_current, player_current,
        )
        new_mask = new_valid_moves(
            blokus_game_module, board_new, player_new,
        )

        current_set = _valid_action_set(current_mask)
        new_set = _valid_action_set(new_mask)
        if current_set != new_set:
            mismatches.append(
                f"Position {i} (after {n_moves} moves):\n"
                + _describe_mismatch(sequence, player_current, current_set, new_set),
            )
            if len(mismatches) >= 5:
                break  # don't spam the test output with hundreds of failures

    if mismatches:
        report = "\n\n".join(mismatches)
        pytest.fail(
            f"Move-gen equivalence broken on {len(mismatches)} position(s):\n\n{report}",
        )


@pytest.mark.slow
def test_movegen_equivalence_gauntlet(blokus_game_module: BlokusDuoGame) -> None:
    """The 50,000-position gauntlet — runs only before the switch-over.

    Skipped automatically if the gauntlet cache hasn't been built. To
    build, run::

        uv run python -c "from pathlib import Path; \\
            from games.blokusduo.game import BlokusDuoGame; \\
            from tests.fixtures.blokus_positions import build_cache; \\
            build_cache(BlokusDuoGame(pieces_config_path=Path('games/blokusduo/pieces.json')), \\
                        n=50_000, seed=42, label='gauntlet_50000')"

    Expected build wall-clock: ~20 min.
    """
    if not GAUNTLET_CACHE.exists():
        pytest.skip(
            f"Gauntlet cache not built at {GAUNTLET_CACHE} — run the "
            "build command in the docstring before flipping the F2 default."
        )

    actions_array, n_moves_array = load_cache(GAUNTLET_CACHE)
    n_positions = len(n_moves_array)

    for i in range(n_positions):
        n_moves = int(n_moves_array[i])
        sequence = actions_array[i, :n_moves]
        board_current, player_current = replay_to_board_and_player(
            blokus_game_module, sequence,
        )
        board_new, player_new = replay_to_board_and_player(
            blokus_game_module, sequence,
        )
        current_mask = current_valid_moves(blokus_game_module, board_current, player_current)
        new_mask = new_valid_moves(blokus_game_module, board_new, player_new)
        current_set = _valid_action_set(current_mask)
        new_set = _valid_action_set(new_mask)
        if current_set != new_set:
            pytest.fail(
                f"Gauntlet mismatch at position {i}:\n"
                + _describe_mismatch(sequence, player_current, current_set, new_set),
            )
