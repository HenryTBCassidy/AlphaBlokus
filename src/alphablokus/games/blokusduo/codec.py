"""Action and coordinate codecs for the 17,837-wide Blokus Duo action space.

Two coordinate systems coexist: board coordinates (bottom-left origin, Blokus
notation) and array indices (top-left origin, numpy). ``CoordinateIndexDecoder``
converts between them; ``ActionCodec`` maps ``Action`` dataclasses to flat
action indices (0-17,836, pass last) and back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.pieces import Orientation, PieceManager

Coordinate: TypeAlias = tuple[int, int]  # (x, y) coordinate pair, bottom-left origin
Index: TypeAlias = tuple[int, int]  # (length_idx, width_idx) pair, top-left origin


class CoordinateIndexDecoder:
    """
    Converts between board coordinates and array indices.

    The game uses two coordinate systems:
    1. Board coordinates (x, y): Origin at bottom-left, x increases right, y increases up
    2. Array indices (length_idx, width_idx): Origin at top-left, length_idx increases down, width_idx increases right
    """

    def __init__(self, board_size: int) -> None:
        self.board_size = board_size

    def to_idx(self, coordinate: Coordinate) -> Index:
        """Convert board coordinates to array indices."""
        length_idx = self.board_size - 1 - coordinate[1]
        width_idx = coordinate[0]
        return length_idx, width_idx

    def to_coordinate(self, idx: Index) -> Coordinate:
        """Convert array indices to board coordinates."""
        x = idx[1]
        y = self.board_size - 1 - idx[0]
        return x, y


@dataclass(frozen=True)
class Action:
    """
    Represents a move in the game.

    An action consists of:
    - piece_id: Identifier of the piece to place (1-21)
    - orientation: How the piece should be oriented
    - x_coordinate: X coordinate of the placement (bottom-left origin)
    - y_coordinate: Y coordinate of the placement (bottom-left origin)
    """

    piece_id: int
    orientation: Orientation
    x_coordinate: int
    y_coordinate: int


class ActionCodec:
    """
    Encodes Action instances to flat integer indices (0–17,836) and back.

    Encoding scheme:
        index = y * (board_size * num_orientations) + x * num_orientations + orientation_id

    Where x, y are board coordinates (bottom-left origin) and orientation_id
    is the 0-based contiguous ID from OrientationCodec. The pass action occupies
    the last index (board_size² × num_orientations).
    """

    def __init__(self, board_size: int, piece_manager: PieceManager) -> None:
        self._board_size = board_size
        self._num_orientations = piece_manager.num_entries
        self._piece_manager = piece_manager
        self.pass_action_index = board_size * board_size * self._num_orientations
        self.action_size = self.pass_action_index + 1

    def encode(self, action: Action) -> int:
        """Convert an Action to a flat action index."""
        orientation_id = self._piece_manager.get_piece_orientation_id((action.piece_id, action.orientation))
        return (
            action.y_coordinate * (self._board_size * self._num_orientations)
            + action.x_coordinate * self._num_orientations
            + orientation_id
        )

    def decode(self, index: int) -> Action:
        """Convert a flat action index to an Action.

        Raises:
            ValueError: If index is the pass action (use is_pass() to check first).
        """
        if index == self.pass_action_index:
            raise ValueError("Cannot decode pass action index to an Action.")
        orientation_id = index % self._num_orientations
        remaining = index // self._num_orientations
        x = remaining % self._board_size
        y = remaining // self._board_size
        piece_id, orientation = self._piece_manager.get_piece_orientation(orientation_id)
        return Action(piece_id, orientation, x, y)

    def encode_from_components(
        self,
        *,
        piece_id: int,
        orientation_id: int,
        x: int,
        y: int,
    ) -> int:
        """Encode from primitive components without constructing an Action.

        Equivalent to ``self.encode(Action(piece_id, orientation, x, y))`` but
        skips the ``Action`` construction and the piece→orientation_id lookup —
        the hot path for table building, where the orientation id is already
        decoded.
        """
        del piece_id  # used only for symmetry of the call site
        return y * (self._board_size * self._num_orientations) + x * self._num_orientations + orientation_id

    def is_pass(self, index: int) -> bool:
        """Check if an action index represents the pass action."""
        return index == self.pass_action_index
