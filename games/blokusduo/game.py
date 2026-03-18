from collections.abc import Generator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from games.blokusduo.board import (
    Action,
    ActionCodec,
    ActionDict,
    BoardArray,
    BlokusDuoBoard,
    Coordinate,
    CoordinateIndexDecoder,
    PlayerSide,
)
from games.blokusduo.pieces import pieces_loader, Orientation, PieceManager
from core.interfaces import IGame


class BlokusDuoGame(IGame):
    """
    Rules engine and action space for Blokus Duo.

    The game is played on a 14x14 board where:
    - White starts in the bottom-right corner
    - Black starts in the top-left corner
    - Players take turns placing pieces
    - Pieces must touch corners of friendly pieces
    - Pieces cannot touch sides of friendly pieces
    """

    def __init__(self, pieces_config_path: Path) -> None:
        super().__init__()
        self.piece_manager: PieceManager = pieces_loader(pieces_config_path)
        self.board_size: int = BlokusDuoBoard.N
        self.num_orientations: int = self.piece_manager.num_entries
        self.white_start = (9, 9)
        self.black_start = (4, 4)
        self._coordinate_index_decoder = CoordinateIndexDecoder(self.board_size)
        self.action_codec = ActionCodec(self.board_size, self.piece_manager)
        self.initial_actions: ActionDict = self._calculate_and_cache_initial_actions()

    # -- Public methods (IGame protocol, in call order) -------------------------

    def initialise_board(self) -> BlokusDuoBoard:
        """Create a new game board."""
        return BlokusDuoBoard(self.piece_manager, self.initial_actions, self._coordinate_index_decoder)

    def get_board_size(self) -> tuple[int, int]:
        """Get the dimensions of the game board."""
        return self.board_size, self.board_size

    def get_action_size(self) -> int:
        """Get the total number of possible actions.

        This includes all possible piece placements (board_size² × num_orientations)
        plus one pass action.
        """
        return (self.board_size ** 2) * self.num_orientations + 1

    def get_next_state(
        self,
        board: BlokusDuoBoard,
        player: PlayerSide,
        action: int
    ) -> tuple[BlokusDuoBoard, PlayerSide]:
        """Apply an action to the current board state.

        Args:
            board: Current board state.
            player: Current player (1 for White, -1 for Black).
            action: Flat action index (0–17,836).

        Returns:
            Tuple of (new board state, next player).
        """
        if self.action_codec.is_pass(action):
            return board, -player
        decoded = self.action_codec.decode(action)
        return board.with_piece(action=decoded, player_side=player), -player

    def valid_move_masking(self, board: BlokusDuoBoard, player: PlayerSide) -> NDArray:
        """Generate a binary mask of valid moves for the current board state.

        Args:
            board: Current board state.
            player: Current player (1 for White, -1 for Black).

        Returns:
            Binary vector of length get_action_size() where 1 = legal move.
        """
        moves = self._valid_moves(board, player)
        mask = np.zeros(self.get_action_size())
        for action in moves:
            mask[self.action_codec.encode(action)] = 1
        if not moves:
            mask[self.action_codec.pass_action_index] = 1
        return mask

    def get_game_ended(self, board: BlokusDuoBoard, player: PlayerSide) -> float:
        """Check if the game has ended and return the result.

        The game ends when neither player has any legal moves. The winner is
        determined by score (sum of placed piece squares, with bonuses for
        placing all pieces / finishing with the monomino).

        Returns:
            0.0 = game continues, 1.0 = player won,
            -1.0 = player lost, 1e-4 = draw.
        """
        white_has_moves = len(self._valid_moves(board, 1)) > 0
        black_has_moves = len(self._valid_moves(board, -1)) > 0
        if white_has_moves or black_has_moves:
            return 0

        white_score = self._calculate_score(board, 1)
        black_score = self._calculate_score(board, -1)

        if white_score == black_score:
            return 1e-4

        white_win = white_score > black_score
        if white_win == (player == 1):
            return 1
        return -1

    def get_canonical_form(self, board: BlokusDuoBoard, player: PlayerSide) -> BlokusDuoBoard:
        """Convert the board to canonical form (player 1 perspective).

        Delegates to board.canonical(player), which swaps player perspectives
        so that as_multi_channel(1) always shows the current player's pieces
        in channels 0-20.
        """
        return board.canonical(player)

    def get_symmetries(self, board: BlokusDuoBoard, pi: NDArray) -> list[tuple[BlokusDuoBoard, NDArray]]:
        """Generate all symmetric forms of the current board state and policy vector.

        The board has the following symmetries:
        1. Identity: No transformation
        2. Rot180: 180-degree rotation (with player swap)
        3. Flip90: Vertical flip + 90-degree rotation
        4. Flip270: Vertical flip + 270-degree rotation (with player swap)
        """
        raise NotImplementedError()

    def state_key(self, board: BlokusDuoBoard) -> bytes:
        """Return a hashable key that uniquely identifies the board state."""
        return board.state_key

    # -- Private methods (callees before callers) ---------------------------------

    @staticmethod
    def _all_cells_valid(
        piece_array: NDArray,
        ins_i: int,
        ins_j: int,
        player: PlayerSide,
        board_2d: BoardArray,
    ) -> bool:
        """Check that every filled cell of the piece at the given insertion point is valid.

        A cell is valid if it is empty and not side-adjacent to a friendly piece.
        The corner check is not needed — the caller guarantees at least one cell
        lands on a placement point (which already satisfies the diagonal rule).
        """
        p_len, p_wid = piece_array.shape
        for i in range(p_len):
            for j in range(p_wid):
                if piece_array[i, j] == 0:
                    continue
                ri, rj = ins_i + i, ins_j + j
                if board_2d[ri, rj] != 0:
                    return False
                if not BlokusDuoBoard._no_sides(ri, rj, player, board_2d):
                    return False
        return True

    def _placements_at_point(
        self,
        piece_id: int,
        orientation: Orientation,
        target_i: int,
        target_j: int,
        player: PlayerSide | None = None,
        board_2d: BoardArray | None = None,
    ) -> Generator[Action, None, None]:
        """Yield all Actions that place a filled cell of the piece on a target point.

        Slides the piece orientation over the target point, trying each filled cell
        as the anchor. For each candidate, checks bounds and optionally validates
        that all cells are empty and not side-adjacent to a friendly piece.

        Args:
            piece_id: Piece to place.
            orientation: Orientation of the piece.
            target_i: Target array row index.
            target_j: Target array column index.
            player: If provided (with board_2d), validate cell emptiness and side adjacency.
            board_2d: Board state for validation. Required if player is provided.
        """
        piece_array = self.piece_manager.get_piece_orientation_array(piece_id, orientation)
        p_len, p_wid = piece_array.shape
        n = self.board_size

        for di in range(p_len):
            for dj in range(p_wid):
                if piece_array[di, dj] == 0:
                    continue

                ins_i = target_i - di
                ins_j = target_j - dj

                if ins_i < 0 or ins_j < 0 or ins_i + p_len > n or ins_j + p_wid > n:
                    continue

                if player is not None and board_2d is not None:
                    if not self._all_cells_valid(piece_array, ins_i, ins_j, player, board_2d):
                        continue

                coord = self._coordinate_index_decoder.to_coordinate((ins_i, ins_j))
                yield Action(piece_id, orientation, coord[0], coord[1])

    def _valid_moves(self, board: BlokusDuoBoard, player: PlayerSide) -> list[Action]:
        """Generate all legal moves for a player.

        Algorithm:
        1. If the player has placed no pieces, return cached initial actions.
        2. Otherwise, iterate over placement points (diagonal corners of friendly pieces).
           For each point, try fitting each remaining piece orientation such that one of
           its filled cells lands on that point. Validate that all cells are on-board,
           empty, and not side-adjacent to a friendly piece.

        Returns:
            List of legal Actions (deduplicated).
        """
        if len(board.remaining_piece_ids(player)) == 21:
            return self.initial_actions[player]

        board_2d = board.as_2d
        points = board.placement_points(player)
        remaining = board.remaining_piece_ids(player)
        moves: set[Action] = set()

        for (pi, pj) in points:
            for piece_id in remaining:
                piece = self.piece_manager.pieces[piece_id]
                for orientation in piece.basis_orientations:
                    moves.update(self._placements_at_point(
                        piece_id, orientation, pi, pj,
                        player=player, board_2d=board_2d,
                    ))

        return list(moves)

    def _valid_moves(self, board: BlokusDuoBoard, player: PlayerSide) -> list[Action]:
        """Generate all legal moves for a player.

        Algorithm:
        1. If the player has placed no pieces, return cached initial actions.
        2. Otherwise, iterate over placement points (diagonal corners of friendly pieces).
           For each point, try fitting each remaining piece orientation such that one of
           its filled cells lands on that point. Validate that all cells are on-board,
           empty, and not side-adjacent to a friendly piece.

        Returns:
            List of legal Actions (deduplicated).
        """
        if len(board.remaining_piece_ids(player)) == 21:
            return self.initial_actions[player]

        board_2d = board.as_2d
        points = board.placement_points(player)
        remaining = board.remaining_piece_ids(player)
        moves: set[Action] = set()

        for (pi, pj) in points:
            for piece_id in remaining:
                piece = self.piece_manager.pieces[piece_id]
                for orientation in piece.basis_orientations:
                    moves.update(self._placements_at_point(
                        piece_id, orientation, pi, pj,
                        player=player, board_2d=board_2d,
                    ))

        return list(moves)

    def _calculate_and_cache_initial_actions(self) -> ActionDict:
        """Calculate and cache all possible initial moves for both players."""
        start_moves: ActionDict = {1: [], -1: []}
        for piece_id, orientation in self.piece_manager.all_piece_id_basis_orientations():
            start_moves[1].extend(
                self._placements_at_point(piece_id, orientation, *self.white_start))
            start_moves[-1].extend(
                self._placements_at_point(piece_id, orientation, *self.black_start))
        return start_moves

    def _calculate_score(self, board: BlokusDuoBoard, player: PlayerSide) -> int:
        """Calculate the score for a player.

        Scoring rules:
        1. +15 points for placing all pieces
        2. +5 bonus for placing the monomino (piece 1) last
        3. Negative points for remaining pieces (sum of squares in each piece)
        """
        remaining = board.remaining_piece_ids(player)
        last_played = board.last_piece_played(player)

        if len(remaining) == 0:
            score = 15
            if last_played == 1:  # Monomino
                score += 5
        else:
            score = -int(np.sum([
                self.piece_manager.pieces[pid].identity.sum() for pid in remaining
            ]))
        return score
