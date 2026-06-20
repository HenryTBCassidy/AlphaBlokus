"""Round-trip + ground-truth tests for the Pentobi move translation (H1).

The translation layer is the error-prone heart of the Pentobi harness — a swapped
coordinate or a wrong shape-match silently corrupts every benchmark game. Two guards:

1. **Ground truth:** ``action_to_cells`` must list exactly the cells that
   ``BlokusDuoBoard.with_piece`` actually fills (the real placement logic).
2. **Round-trip identity:** index → Pentobi string → index recovers the original action
   for every orientation × valid anchor (this exercises ``cells_to_action`` in reverse).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from games.blokusduo.board import Action, CoordinateIndexDecoder
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pentobi_translation import PASS, PentobiMoveTranslator

_PIECES = Path(__file__).resolve().parent.parent.parent / "games" / "blokusduo" / "pieces.json"


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=_PIECES)


@pytest.fixture(scope="module")
def tr(game: BlokusDuoGame) -> PentobiMoveTranslator:
    return PentobiMoveTranslator(game)


def test_shape_lookup_is_bijection(tr: PentobiMoveTranslator, game: BlokusDuoGame) -> None:
    """Every piece-orientation has a distinct normalised shape (else cells→Action is
    ambiguous). 91 orientations → 91 shape keys."""
    assert len(tr._shape_to_po) == game.piece_manager.num_entries == 91  # noqa: SLF001


def test_coordinate_roundtrip_and_known_values(tr: PentobiMoveTranslator, game: BlokusDuoGame) -> None:
    assert tr.coord_to_pentobi(0, 0) == "a1"
    assert tr.coord_to_pentobi(13, 13) == "n14"
    assert tr.coord_to_pentobi(4, 8) == "e9"
    assert tr.pentobi_to_coord("e9") == (4, 8)
    n = game.board_size
    for x in range(n):
        for y in range(n):
            assert tr.pentobi_to_coord(tr.coord_to_pentobi(x, y)) == (x, y)


def test_pass_roundtrips(tr: PentobiMoveTranslator, game: BlokusDuoGame) -> None:
    pass_idx = game.action_codec.pass_action_index
    assert tr.action_index_to_pentobi(pass_idx) == PASS
    assert tr.pentobi_to_action_index(PASS) == pass_idx
    assert tr.pentobi_to_action_index("PASS") == pass_idx  # case-insensitive


def test_action_translation_matches_placement_and_roundtrips(
    tr: PentobiMoveTranslator, game: BlokusDuoGame,
) -> None:
    """For every orientation × every anchor that actually fits on an empty board:
    (1) the translated cells equal the cells ``with_piece`` fills, and
    (2) index → Pentobi → index is the identity."""
    n = game.board_size
    decoder = CoordinateIndexDecoder(n)
    empty = game.initialise_board()
    tested = 0
    for piece_id, orientation in game.piece_manager.all_piece_id_basis_orientations():
        for y in range(n):
            for x in range(n):
                action = Action(piece_id, orientation, x, y)
                try:
                    placed = empty.with_piece(action, 1)
                except (IndexError, RuntimeError):
                    continue  # piece doesn't fit at this anchor — skip

                # (1) ground truth: cells with_piece actually filled
                occupied = np.argwhere(placed.as_2d != 0)
                gt = {
                    tr.coord_to_pentobi(*decoder.to_coordinate((int(r), int(c))))
                    for r, c in occupied
                }
                assert set(tr.action_to_cells(action).split(",")) == gt, (
                    f"cells mismatch for {action}"
                )

                # (2) index round-trip (exercises cells_to_action)
                idx = game.action_codec.encode(action)
                assert tr.pentobi_to_action_index(tr.action_index_to_pentobi(idx)) == idx, (
                    f"round-trip failed for {action}"
                )
                tested += 1
    assert tested > 2000, f"suspiciously few placements tested ({tested})"


def test_render_board_smoke(tr: PentobiMoveTranslator, game: BlokusDuoGame) -> None:
    """Renderer produces the a–n / 1–14 grid with the placed piece marked."""
    board = game.initialise_board().with_piece(Action(1, next(iter(
        game.piece_manager.pieces[1].basis_orientations)), 4, 4), 1)
    out = tr.render_board(board)
    assert out.splitlines()[0].strip().startswith("a b c")
    assert "W" in out  # the placed white piece shows up
    assert len(out.splitlines()) == game.board_size + 1  # header + N rows
