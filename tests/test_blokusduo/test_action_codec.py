import pytest

from games.blokusduo.board import Action, ActionCodec
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.pieces import PieceManager


@pytest.fixture(scope="session")
def codec(piece_manager: PieceManager) -> ActionCodec:
    return ActionCodec(board_size=14, piece_manager=piece_manager)


def test_action_size(codec: ActionCodec):
    assert codec.action_size == 17837


def test_pass_action_index(codec: ActionCodec):
    assert codec.pass_action_index == 17836


def test_is_pass(codec: ActionCodec):
    assert codec.is_pass(17836)
    assert not codec.is_pass(0)
    assert not codec.is_pass(17835)


def test_decode_pass_raises(codec: ActionCodec):
    with pytest.raises(ValueError, match="pass action"):
        codec.decode(17836)


def test_encode_decode_roundtrip(codec: ActionCodec, piece_manager: PieceManager):
    """Every valid action should survive encode → decode."""
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            action = Action(piece_id, orientation, x_coordinate=0, y_coordinate=0)
            assert codec.decode(codec.encode(action)) == action

            action = Action(piece_id, orientation, x_coordinate=13, y_coordinate=13)
            assert codec.decode(codec.encode(action)) == action


def test_encode_boundary_values(codec: ActionCodec, piece_manager: PieceManager):
    """First and last non-pass indices should be at the expected positions."""
    first_piece_id, first_orientation = piece_manager.get_piece_orientation(0)
    last_piece_id, last_orientation = piece_manager.get_piece_orientation(90)

    # (x=0, y=0, orientation_id=0) → index 0
    action_min = Action(first_piece_id, first_orientation, 0, 0)
    assert codec.encode(action_min) == 0

    # (x=13, y=13, orientation_id=90) → index 17835
    action_max = Action(last_piece_id, last_orientation, 13, 13)
    assert codec.encode(action_max) == 17835


def test_all_indices_unique(codec: ActionCodec, piece_manager: PieceManager):
    """Spot-check: encoding different actions at the same position gives different indices."""
    actions_at_origin = []
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            actions_at_origin.append(codec.encode(Action(piece_id, orientation, 0, 0)))

    assert len(actions_at_origin) == len(set(actions_at_origin)) == 91


def test_codec_matches_game_action_size(blokus_game: BlokusDuoGame):
    """ActionCodec.action_size should agree with BlokusDuoGame.get_action_size()."""
    assert blokus_game.action_codec.action_size == blokus_game.get_action_size()
