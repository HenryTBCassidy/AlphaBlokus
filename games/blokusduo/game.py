from collections.abc import Generator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from games.blokusduo.board import (
    Action,
    ActionCodec,
    ActionDict,
    BlokusDuoBoard,
    Coordinate,
    CoordinateIndexDecoder,
    PlayerSide,
)
from games.blokusduo.pieces import pieces_loader, Orientation, PieceManager
from core.interfaces import IGame


class BlokusDuoGame(IGame):
    """
    Implementation of the Blokus Duo game for the AlphaZero framework.

    This class implements the IGame interface, providing:
    1. Game state management
    2. Move validation and execution
    3. Game state conversion for neural network input
    4. Symmetry handling for training data augmentation

    The game is played on a 14x14 board where:
    - White starts in the bottom-right corner
    - Black starts in the top-left corner
    - Players take turns placing pieces
    - Pieces must touch corners of friendly pieces
    - Pieces cannot touch sides of friendly pieces
    """

    def __init__(self, pieces_config_path: Path) -> None:
        """
        Initialize a new Blokus Duo game.

        Args:
            pieces_config_path: Path to the configuration file defining game pieces
        """
        super().__init__()
        self.piece_manager: PieceManager = pieces_loader(pieces_config_path)
        self.board_size: int = BlokusDuoBoard.N
        self.num_planes: int = self.piece_manager.num_entries
        # Canonically will have black start in top left and white in bottom right
        self.white_start = (9, 9)
        self.black_start = (4, 4)
        self._coordinate_index_decoder = CoordinateIndexDecoder(self.board_size)
        self.action_codec = ActionCodec(self.board_size, self.piece_manager)

        # Calculate all possible starting moves on game init and cache to save time
        self.initial_actions: ActionDict = self._calculate_and_cache_initial_actions()

    def calculate_legal_initial_actions_for_piece_orientation(
        self,
        piece_id: int,
        orientation: Orientation,
        start: Coordinate
    ) -> Generator[Action, None, None]:
        """
        Generate all legal initial actions for a piece orientation.

        Args:
            piece_id: Identifier of the piece
            orientation: Orientation of the piece
            start: Starting position for the piece

        Yields:
            Action: Each valid initial placement of the piece
        """
        piece_orientation = self.piece_manager.get_piece_orientation_array(piece_id, orientation)
        length, width = piece_orientation.shape
        for i in range(length):
            for j in range(width):
                if piece_orientation[(i, j)] == 1:
                    coordinate = self._coordinate_index_decoder.to_coordinate((start[0] - i, start[1] - j))
                    yield Action(piece_id, orientation, coordinate[0], coordinate[1])

    def _calculate_and_cache_initial_actions(self) -> ActionDict:
        """
        Calculate and cache all possible initial moves for both players.

        Returns:
            ActionDict: Dictionary mapping player sides to their valid initial moves
        """
        start_moves: ActionDict = {1: [], -1: []}
        for piece_id, orientation in self.piece_manager.all_piece_id_basis_orientations():
            start_moves[1].extend(self.calculate_legal_initial_actions_for_piece_orientation(
                piece_id, orientation, self.white_start))
            start_moves[-1].extend(self.calculate_legal_initial_actions_for_piece_orientation(
                piece_id, orientation, self.black_start))
        return start_moves

    def initialise_board(self) -> BlokusDuoBoard:
        """
        Create a new game board.

        Returns:
            BlokusDuoBoard: A fresh board ready for play
        """
        return BlokusDuoBoard(self.piece_manager, self.initial_actions, self._coordinate_index_decoder)

    def get_board_size(self) -> tuple[int, int]:
        """
        Get the dimensions of the game board.

        Returns:
            Tuple[int, int]: (rows, cols) of the board
        """
        return self.board_size, self.board_size

    def get_action_size(self) -> int:
        """
        Get the total number of possible actions in the game.

        This includes:
        - All possible piece placements (board_size^2 * num_planes)
        - Plus one for the "do nothing" action

        Returns:
            int: Total number of possible actions
        """
        return (self.board_size ** 2) * self.num_planes + 1

    def get_next_state(
        self,
        board: BlokusDuoBoard,
        player: PlayerSide,
        action: int
    ) -> tuple[BlokusDuoBoard, PlayerSide]:
        """
        Apply an action to the current board state.

        Args:
            board: Current board state
            player: Current player (1 for White, -1 for Black)
            action: Flat action index (0–17,836). Use action_codec to convert.

        Returns:
            Tuple containing:
                - BlokusDuoBoard: Updated board state
                - PlayerSide: Next player to move (-player)
        """
        if self.action_codec.is_pass(action):
            return board, -player
        decoded = self.action_codec.decode(action)
        return board.with_piece(action=decoded, player_side=player), -player

    def valid_move_masking(self, board: BlokusDuoBoard, player: PlayerSide) -> NDArray:
        """
        Generate a mask of valid moves for the current board state.

        Args:
            board: Current board state
            player: Current player (1 for White, -1 for Black)

        Returns:
            NDArray: Binary vector of length get_action_size() where 1 = legal move.
        """
        # TODO: Implement. Need to decide the right return type (binary mask over
        #       the flat action space, or something richer). For now the interface
        #       returns NDArray which is a flat binary vector — same as TicTacToe.
        raise NotImplementedError()

    def get_game_ended(self, board: BlokusDuoBoard, player: PlayerSide) -> float:
        """
        Check if the game has ended and return the result.

        The game ends when neither player has any legal moves. The winner is
        determined by score (sum of placed piece squares, with bonuses for
        placing all pieces / finishing with the monomino).

        Args:
            board: Current board state
            player: Current player (1 for White, -1 for Black)

        Returns:
            float: 0.0 = game continues, 1.0 = player won,
                   -1.0 = player lost, 1e-4 = draw
        """
        # Game continues if either player has remaining moves
        # TODO: valid_moves is a stub — this will work once move generation is implemented
        white_has_moves = len(self.valid_moves(board, 1)) > 0
        black_has_moves = len(self.valid_moves(board, -1)) > 0
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

    def valid_moves(self, board: BlokusDuoBoard, player: PlayerSide) -> list[Action]:
        """Generate all legal moves for a player.

        TODO: Implement the full move generation algorithm. Currently returns [].
        """
        return []

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

    def get_canonical_form(self, board: BlokusDuoBoard, player: PlayerSide) -> BlokusDuoBoard:
        """
        Convert the board to its canonical form.

        Delegates to board.canonical(player), which swaps player
        perspectives so that as_multi_channel(1) always shows the
        current player's pieces in channels 0-20.

        Args:
            board: Current board state
            player: Current player (1 for White, -1 for Black)

        Returns:
            BlokusDuoBoard: Board state in canonical form
        """
        return board.canonical(player)

    def get_symmetries(self, board: BlokusDuoBoard, pi: NDArray) -> list[tuple[BlokusDuoBoard, NDArray]]:
        """
        Generate all symmetric forms of the current board state and policy vector.

        The board has the following symmetries:
        1. Identity: No transformation
        2. Rot180: 180-degree rotation (with player swap)
        3. Flip90: Vertical flip + 90-degree rotation
        4. Flip270: Vertical flip + 270-degree rotation (with player swap)

        Args:
            board: Current board state
            pi: Policy vector of size self.get_action_size()

        Returns:
            List[Tuple[BlokusDuoBoard, NDArray]]: List of (board, pi) pairs for each symmetry
        """
        raise NotImplementedError()

    def state_key(self, board: BlokusDuoBoard) -> bytes:
        """Return a hashable key that uniquely identifies the board state.

        Used by MCTS as a dictionary key for state lookups.

        Args:
            board: Current board state

        Returns:
            bytes: Raw byte representation of the board array
        """
        return board.state_key
