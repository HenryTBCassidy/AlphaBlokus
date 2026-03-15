from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from core.interfaces import IBoard
from games.blokusduo.pieces import Orientation, PieceManager

# Type aliases for improved readability
BoardArray: TypeAlias = NDArray[np.int8]  # Game board as numpy array
PlayerSide: TypeAlias = int  # 1 for White, -1 for Black
Coordinate: TypeAlias = tuple[int, int]  # (x, y) coordinate pair
Index: TypeAlias = tuple[int, int]  # (length_idx, width_idx) pair
ActionDict: TypeAlias = dict[int, list['Action']]  # Maps player side to list of valid actions
PlacementDict: TypeAlias = dict[Coordinate, dict[int, 'Action']]  # Maps coordinates to piece actions


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


class BlokusDuoBoard(IBoard):
    """
    Immutable game board for Blokus Duo. Implements the IBoard protocol.

    The board is a frozen snapshot of game state. New states are created via
    ``with_piece()`` — there are no mutation methods.

    Internal representation:
        ``_piece_placement_board``: int8 14x14 array where each cell holds:
            +piece_id  for white pieces (player 1)
            -piece_id  for black pieces (player -1)
            0          for empty squares

    Derived views:
        ``as_2d``:  np.sign(_piece_placement_board) -> +1/-1/0
        ``as_multi_channel(player)``: 44-channel float32 tensor

    Channel layout for as_multi_channel (44 x 14 x 14):
        Channels  0-20:  Current player's 21 pieces (binary per-piece planes)
        Channels 21-41:  Opponent's 21 pieces
        Channel  42:     Aggregate current player (union of 0-20)
        Channel  43:     Aggregate opponent (union of 21-41)
    """

    N: int = 14  # Board size

    def __init__(
        self,
        piece_manager: PieceManager,
        initial_actions: ActionDict,
        coordinate_index_decoder: CoordinateIndexDecoder,
    ) -> None:
        """Create a fresh empty board."""
        self._piece_placement_board = np.zeros((self.N, self.N), dtype=np.int8)
        self._white_piece_ids_remaining: frozenset[int] = frozenset(range(1, 22))
        self._black_piece_ids_remaining: frozenset[int] = frozenset(range(1, 22))
        self._white_last_piece_played: int | None = None
        self._black_last_piece_played: int | None = None
        self._white_placement_points: PlacementDict = {}
        self._black_placement_points: PlacementDict = {}

        # Shared immutable refs (never copied)
        self._piece_manager = piece_manager
        self._initial_actions = initial_actions
        self._coordinate_index_decoder = coordinate_index_decoder

    # -- IBoard protocol (public) -----------------------------------------------

    @property
    def as_2d(self) -> NDArray:
        """Flat H x W game state board (+1/-1/0). Derived from placement board."""
        return np.sign(self._piece_placement_board).astype(np.int8)

    def as_multi_channel(self, current_player: int) -> NDArray:
        """Build 44-channel tensor for neural net input.

        Args:
            current_player: Whose turn it is (1 or -1).

        Returns:
            NDArray of shape (44, 14, 14) with float32 dtype.
        """
        ppb = self._piece_placement_board
        rep = np.zeros((44, self.N, self.N), dtype=np.float32)

        for piece_id in range(1, 22):
            rep[piece_id - 1] = (ppb == piece_id * current_player)
            rep[21 + piece_id - 1] = (ppb == -piece_id * current_player)

        rep[42] = (ppb * current_player > 0)   # Aggregate: current player
        rep[43] = (ppb * current_player < 0)    # Aggregate: opponent
        return rep

    @property
    def num_channels(self) -> int:
        """Number of channels in as_multi_channel output."""
        return 44

    @property
    def state_key(self) -> bytes:
        """Hashable key that uniquely identifies this board state.

        Uses the raw bytes of the signed placement board — 196 bytes,
        fully deterministic, encodes which piece is where for both players.
        """
        return self._piece_placement_board.tobytes()

    def canonical(self, player: int) -> BlokusDuoBoard:
        """Return the board in canonical form (player 1 perspective).

        When player is 1, returns self (already canonical).
        When player is -1, flips piece signs and swaps all per-player state.
        """
        if player == 1:
            return self
        return BlokusDuoBoard._from_state(
            piece_placement_board=(-self._piece_placement_board).astype(np.int8),
            white_remaining=self._black_piece_ids_remaining,
            black_remaining=self._white_piece_ids_remaining,
            white_last=self._black_last_piece_played,
            black_last=self._white_last_piece_played,
            white_points=self._black_placement_points,
            black_points=self._white_placement_points,
            piece_manager=self._piece_manager,
            initial_actions=self._initial_actions,
            coordinate_index_decoder=self._coordinate_index_decoder,
        )

    # -- Piece placement (public, immutable) ------------------------------------

    def with_piece(self, action: Action, player_side: PlayerSide) -> BlokusDuoBoard:
        """Return a new board with the given piece placed.

        Args:
            action: The move to execute.
            player_side: Player identifier (1 for White, -1 for Black).

        Returns:
            A new BlokusDuoBoard with the piece placed.

        Raises:
            IndexError: If the piece doesn't fit on the board.
            RuntimeError: If trying to place on an occupied square.
        """
        piece_orientation = self._piece_manager.get_piece_orientation_array(
            action.piece_id, action.orientation)
        length_idx, width_idx = self._coordinate_index_decoder.to_idx(
            coordinate=(action.x_coordinate, action.y_coordinate))

        # Create new placement board with piece applied
        new_ppb = self._piece_placement_board.copy()
        p_len, p_wid = piece_orientation.shape

        for i in range(p_len):
            for j in range(p_wid):
                if piece_orientation[i, j] != 0:
                    ri, ci = length_idx + i, width_idx + j
                    if ri > 13 or ci > 13:
                        raise IndexError(
                            f"Piece cannot be inserted at this index, it does not fit on the board! "
                            f"\n{self._piece_insertion_error_message(piece_orientation, action.x_coordinate, action.y_coordinate)}")
                    if new_ppb[ri, ci] != 0:
                        raise RuntimeError(
                            f"Trying to insert piece into board over existing piece! \n"
                            f"{self._piece_insertion_error_message(piece_orientation, action.x_coordinate, action.y_coordinate)}")
                    new_ppb[ri, ci] = np.int8(action.piece_id * player_side)

        # Update remaining pieces
        if player_side == 1:
            new_white_remaining = self._white_piece_ids_remaining - {action.piece_id}
            new_black_remaining = self._black_piece_ids_remaining
            new_white_last = action.piece_id
            new_black_last = self._black_last_piece_played
        else:
            new_white_remaining = self._white_piece_ids_remaining
            new_black_remaining = self._black_piece_ids_remaining - {action.piece_id}
            new_white_last = self._white_last_piece_played
            new_black_last = action.piece_id

        # Copy and update placement caches
        new_white_points: PlacementDict = dict(self._white_placement_points)
        new_black_points: PlacementDict = dict(self._black_placement_points)
        board_2d = np.sign(new_ppb).astype(np.int8)
        BlokusDuoBoard._update_placement_points(
            new_white_points, 1, (length_idx, width_idx), piece_orientation, board_2d)
        BlokusDuoBoard._update_placement_points(
            new_black_points, -1, (length_idx, width_idx), piece_orientation, board_2d)

        return BlokusDuoBoard._from_state(
            piece_placement_board=new_ppb,
            white_remaining=new_white_remaining,
            black_remaining=new_black_remaining,
            white_last=new_white_last,
            black_last=new_black_last,
            white_points=new_white_points,
            black_points=new_black_points,
            piece_manager=self._piece_manager,
            initial_actions=self._initial_actions,
            coordinate_index_decoder=self._coordinate_index_decoder,
        )

    # -- Game state queries (public) --------------------------------------------

    def valid_moves(self, player_side: PlayerSide) -> list[Action]:
        """Get all valid moves for the current player."""
        # TODO: Implement move generation logic
        return []

    def game_ended(self) -> bool:
        """Check if the game has ended.

        The game ends when:
        1. A player has no pieces remaining, or
        2. Neither player has any legal moves
        """
        raise NotImplementedError()

    def game_result(self, player: PlayerSide) -> float:
        """Get the game result from a player's perspective.

        Returns:
            1.0 = player won, -1.0 = player lost, 1e-4 = draw.
        """
        if not self.game_ended():
            raise RuntimeError("Trying to calculate the result of a game that hasn't ended!")

        white_score = self._calculate_score(
            self._white_piece_ids_remaining,
            self._white_last_piece_played,
            self._piece_manager,
        )
        black_score = self._calculate_score(
            self._black_piece_ids_remaining,
            self._black_last_piece_played,
            self._piece_manager,
        )

        if white_score == black_score:
            return 1e-4

        white_win = white_score > black_score
        if white_win == (player == 1):
            return 1
        else:
            return -1

    # -- Private helpers --------------------------------------------------------

    @classmethod
    def _from_state(
        cls,
        *,
        piece_placement_board: NDArray[np.int8],
        white_remaining: frozenset[int],
        black_remaining: frozenset[int],
        white_last: int | None,
        black_last: int | None,
        white_points: PlacementDict,
        black_points: PlacementDict,
        piece_manager: PieceManager,
        initial_actions: ActionDict,
        coordinate_index_decoder: CoordinateIndexDecoder,
    ) -> BlokusDuoBoard:
        """Construct board from existing state (internal use)."""
        board = object.__new__(cls)
        board._piece_placement_board = piece_placement_board
        board._white_piece_ids_remaining = white_remaining
        board._black_piece_ids_remaining = black_remaining
        board._white_last_piece_played = white_last
        board._black_last_piece_played = black_last
        board._white_placement_points = white_points
        board._black_placement_points = black_points
        board._piece_manager = piece_manager
        board._initial_actions = initial_actions
        board._coordinate_index_decoder = coordinate_index_decoder
        return board

    def _remaining_ids(self, player_side: PlayerSide) -> frozenset[int]:
        """Get remaining piece IDs for the given player."""
        return self._white_piece_ids_remaining if player_side == 1 else self._black_piece_ids_remaining

    def _last_piece_played(self, player_side: PlayerSide) -> int | None:
        """Get the last piece ID played by the given player."""
        return self._white_last_piece_played if player_side == 1 else self._black_last_piece_played

    def _placement_points(self, player_side: PlayerSide) -> PlacementDict:
        """Get the placement points cache for the given player."""
        return self._white_placement_points if player_side == 1 else self._black_placement_points

    @staticmethod
    def _no_sides(i: int, j: int, side: PlayerSide, board: BoardArray) -> bool:
        """Check that a square does not touch any sides of friendly pieces."""
        if i > 0 and board[i - 1, j] == side:
            return False
        if j > 0 and board[i, j - 1] == side:
            return False
        if j + 1 < 14 and board[i, j + 1] == side:
            return False
        if i + 1 < 14 and board[i + 1, j] == side:
            return False
        return True

    @staticmethod
    def _at_least_one_corner(i: int, j: int, side: PlayerSide, board: BoardArray) -> bool:
        """Check that a square has at least one friendly piece on a diagonal."""
        if i > 0 and j > 0 and board[i - 1, j - 1] == side:
            return True
        if i > 0 and j + 1 < 14 and board[i - 1, j + 1] == side:
            return True
        if i + 1 < 14 and j > 0 and board[i + 1, j - 1] == side:
            return True
        if i + 1 < 14 and j + 1 < 14 and board[i + 1, j + 1] == side:
            return True
        return False

    @staticmethod
    def _valid_placement(i: int, j: int, side: PlayerSide, board: BoardArray) -> bool:
        """Check if a square can be validly placed on (empty + corner rule + no-sides rule)."""
        if board[i, j] != 0:
            return False
        return (BlokusDuoBoard._at_least_one_corner(i, j, side, board)
                and BlokusDuoBoard._no_sides(i, j, side, board))

    @staticmethod
    def _update_placement_points(
        points: PlacementDict,
        side: PlayerSide,
        insertion_index: Index,
        piece_orientation: NDArray,
        board_2d: BoardArray,
    ) -> None:
        """Update a player's placement cache after a piece is placed.

        Mutates ``points`` in place — caller is responsible for passing a
        fresh copy when constructing immutable boards.
        """
        p_l, p_w = piece_orientation.shape
        length_range = (max(0, insertion_index[0] - 1), min(insertion_index[0] + p_l + 1, 14))
        width_range = (max(0, insertion_index[1] - 1), min(insertion_index[1] + p_w + 1, 14))

        for i in range(length_range[0], length_range[1]):
            for j in range(width_range[0], width_range[1]):
                if (i, j) in points:
                    if not BlokusDuoBoard._valid_placement(i, j, side, board_2d):
                        del points[(i, j)]
                else:
                    if BlokusDuoBoard._valid_placement(i, j, side, board_2d):
                        points[(i, j)] = {}  # TODO: populate moves when move generation is implemented

    def _piece_insertion_error_message(
        self,
        piece_orientation: NDArray,
        x_coordinate: int,
        y_coordinate: int,
    ) -> str:
        return (f"Board: \n {self.as_2d}, \n "
                f"Piece: \n {piece_orientation}, \n"
                f"At ({x_coordinate}, {y_coordinate})")

    @staticmethod
    def _calculate_score(
        remaining: frozenset[int],
        last_played: int | None,
        piece_manager: PieceManager,
    ) -> int:
        """Calculate the score for a player.

        Scoring rules:
        1. +15 points for placing all pieces
        2. +5 bonus points for placing the smallest piece last
        3. Negative points for remaining pieces (sum of squares)
        """
        if len(remaining) == 0:
            score = 15
            if last_played == 1:  # Smallest piece
                score += 5
        else:
            score = -int(np.sum([
                piece_manager.pieces[pid].identity.sum() for pid in remaining
            ]))
        return score
