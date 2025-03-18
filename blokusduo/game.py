from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, Generator, TypeAlias, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from blokusduo.pieces import pieces_loader, Piece, Orientation, PieceManager, BidirectionalDict
from core.interfaces import IGame

# Type aliases for improved readability
BoardArray: TypeAlias = NDArray[np.int64]  # Game board as numpy array
PlayerSide: TypeAlias = int  # 1 for White, -1 for Black
Coordinate: TypeAlias = Tuple[int, int]  # (x, y) coordinate pair
Index: TypeAlias = Tuple[int, int]  # (length_idx, width_idx) pair
ActionDict: TypeAlias = Dict[int, List['Action']]  # Maps player side to list of valid actions
PlacementDict: TypeAlias = Dict[Coordinate, Dict[int, 'Action']]  # Maps coordinates to piece actions


def no_sides(i: int, j: int, side: PlayerSide, board: BoardArray) -> bool:
    """
    Check if a point on the board does not touch any sides of friendly pieces.

    Args:
        i: Row index in the board grid
        j: Column index in the board grid
        side: Player identifier (1 for White, -1 for Black)
        board: Game board as a numpy array where:
              1 represents White pieces
              -1 represents Black pieces
              0 represents empty squares

    Returns:
        bool: True if the point does not touch any sides of friendly pieces
    """
    above = False
    left = False
    right = False
    below = False

    if i - 1 > 0:
        above = board[i - 1, j] == side
    if j - 1 > 0:
        left = board[i, j - 1] == side
    if j + 1 < 14:
        right = board[i, j + 1] == side
    if i + 1 < 14:
        below = board[i + 1, j] == side

    return not above and not left and not right and not below


def at_least_one_corner(i: int, j: int, side: PlayerSide, board: BoardArray) -> bool:
    """
    Check if a point on the board has at least one friendly piece in a corner position.

    Args:
        i: Row index in the board grid
        j: Column index in the board grid
        side: Player identifier (1 for White, -1 for Black)
        board: Game board as a numpy array where:
              1 represents White pieces
              -1 represents Black pieces
              0 represents empty squares

    Returns:
        bool: True if the point has at least one friendly piece in a corner position
    """
    if (i - 1 > 0) and (j - 1 > 0):
        if board[i - 1, j - 1] == side: return True

    if (i - 1 > 0) and (j + 1 < 14):
        if board[i - 1, j + 1] == side: return True

    if (i + 1 < 14) and (j - 1 > 0):
        if board[i + 1, j - 1] == side: return True

    if (i + 1 > 0) and (j + 1 < 14):
        if board[i + 1, j + 1] == side: return True

    return False


def valid_placement(i: int, j: int, side: PlayerSide, board: BoardArray) -> bool:
    """
    Check if a point can be validly placed on the Blokus board.

    A placement is valid if:
    1. The square is empty
    2. The square has at least one friendly piece in a corner position
    3. The square does not touch any sides of friendly pieces

    Args:
        i: Row index in the board grid
        j: Column index in the board grid
        side: Player identifier (1 for White, -1 for Black)
        board: Game board as a numpy array where:
              1 represents White pieces
              -1 represents Black pieces
              0 represents empty squares

    Returns:
        bool: True if the point can be validly placed
    """
    if board[i, j] != 0:  # Square is occupied
        return False
    return at_least_one_corner(i, j, side, board) and no_sides(i, j, side, board)


# TODO: Make this available to all games & change framework to use it? Using +/- 1 everywhere is a bit pants
class Side(StrEnum):
    """
    Enumeration of player sides in the game.
    
    This enum standardizes the representation of players:
    - White: represented as 1 in the game logic
    - Black: represented as -1 in the game logic
    
    The to_int() method converts the enum value to its corresponding integer representation.
    """
    White = "White"
    Black = "Black"

    def to_int(self) -> PlayerSide:
        """
        Convert the side to its integer representation.

        Returns:
            PlayerSide: 1 for White, -1 for Black
        """
        match self.value:
            case "White":
                return 1
            case "Black":
                return -1


class CoordinateIndexDecoder:
    """
    Converts between board coordinates and array indices.
    
    The game uses two coordinate systems:
    1. Board coordinates (x, y): Origin at bottom-left, x increases right, y increases up
    2. Array indices (length_idx, width_idx): Origin at top-left, length_idx increases down, width_idx increases right
    
    This class handles the conversion between these two systems.
    """

    def __init__(self, board_size: int) -> None:
        """
        Initialize the decoder with the board size.

        Args:
            board_size: Size of the square game board
        """
        self.board_size = board_size

    def to_idx(self, coordinate: Coordinate) -> Index:
        """
        Convert board coordinates to array indices.

        Args:
            coordinate: (x, y) coordinate pair where:
                       x increases right from bottom-left
                       y increases up from bottom-left

        Returns:
            Index: (length_idx, width_idx) pair where:
                  length_idx increases down from top-left
                  width_idx increases right from top-left
        """
        length_idx = self.board_size - 1 - coordinate[1]
        width_idx = coordinate[0]
        return length_idx, width_idx

    def to_coordinate(self, idx: Index) -> Coordinate:
        """
        Convert array indices to board coordinates.

        Args:
            idx: (length_idx, width_idx) pair where:
                 length_idx increases down from top-left
                 width_idx increases right from top-left

        Returns:
            Coordinate: (x, y) coordinate pair where:
                       x increases right from bottom-left
                       y increases up from bottom-left
        """
        x = idx[1]
        y = self.board_size - 1 - idx[0]
        return x, y


# TODO: Generalise the project to use some inheritance of a more general Action later
@dataclass(frozen=True)
class Action:
    """
    Represents a move in the game.
    
    An action consists of:
    - piece_id: Identifier of the piece to place
    - orientation: How the piece should be oriented
    - x_coordinate: X coordinate of the placement (bottom-left origin)
    - y_coordinate: Y coordinate of the placement (bottom-left origin)
    """
    piece_id: int
    orientation: Orientation
    x_coordinate: int
    y_coordinate: int


class Player:
    """
    Represents a player in the game.
    
    This class manages the state and valid moves for a player:
    1. Tracks remaining pieces and played pieces
    2. Maintains a cache of valid placement points
    3. Updates the cache when pieces are placed
    4. Provides scoring information
    
    The caching system is designed to be efficient by:
    - Only checking corners for valid placements
    - Maintaining a list of potential placement points
    - Updating the cache incrementally when pieces are placed
    """

    def __init__(self, board_size: int, side: Side, pieces: Dict[int, Piece]) -> None:
        """
        Initialize a new player.

        Args:
            board_size: Size of the square game board
            side: Player's side (White or Black)
            pieces: Dictionary mapping piece IDs to Piece objects
        """
        self.side = side
        self.pieces_remaining = deepcopy(pieces)

        # TODO: Don't forget you need this for the encoded region and scoring
        #       Can potentially just change this to last id played since we are already updating the encoded region
        self.piece_ids_played: deque[int] = deque()

        # Encode information about which pieces white and black have played by adding these arrays to the White and
        # Black sections of the board (after reshaping).
        # Could generalise this by calculating how big the array needs to be based on n and the number of pieces.
        self.pieces_played_encoded_region = np.zeros(2 * board_size, dtype=int)


        # Dictionary of placement point -> piece_id -> action
        self.potential_placement_points: PlacementDict = dict()

        # TODO: Maybe it makes more sense to have the corners to place to be on each player?

    def update_potential_placement_points(
        self,
        insertion_index: Index,
        piece_orientation: BoardArray,
        board_array: BoardArray
    ) -> None:
        """
        Update the cache of valid placement points after a piece is placed.
        """
        # TODO: Not sure which order is more efficient yet but
        #       1. Need to check surrounding rectangle around piece to see what new corners were added
        #          (bear in mind they could already be in the dictionary)
        #       2. Also check if the piece was laid on a potential placement point as obviously this removes it
        # Cap iteration range to make sure we don't go out of index bounds
        # Iterating + 1 unit around where the piece was placed
        p_l, p_w = piece_orientation.shape
        length_range = (max(0, insertion_index[0] - 1), min(insertion_index[0] + p_l + 1, 13))
        width_range = (max(0, insertion_index[1] - 1), min(insertion_index[1] + p_w + 1, 13))

        for i in range(length_range[0], length_range[1]):
            for j in range(width_range[0], width_range[1]):
                if (i, j) in self.potential_placement_points:
                    if not valid_placement(i, j, self.side.to_int(), board_array):
                        del self.potential_placement_points[(i, j)]
                else:
                    if valid_placement(i, j, self.side.to_int(), board_array):
                        self.potential_placement_points[(i, j)] = self.populate_moves_for_new_placement(board_array)

    def populate_moves_for_new_placement(self, board_array: BoardArray) -> Dict[int, Action]:
        """
        Generate all valid moves for a new placement point.

        Args:
            board_array: Current state of the game board

        Returns:
            Dict[int, Action]: Dictionary mapping piece IDs to valid actions
        """
        moves: Dict[int, Action] = {}
        for piece_id, piece in self.pieces_remaining.items():
            for orientation in piece.basis_orientations:
                # TODO: Implement move generation logic
                pass
        return moves


class BlokusDuoBoard:
    """
    Manages the state of a Blokus Duo game.
    
    This class handles:
    1. Board state representation and manipulation
    2. Piece placement validation and execution
    3. Game state queries (valid moves, game end, scoring)
    4. Conversion between different board representations
    
    The board is represented internally as a numpy array where:
    - 1 represents White pieces
    - -1 represents Black pieces
    - 0 represents empty squares
    """

    def __init__(
        self,
        piece_manager: PieceManager,
        initial_actions: ActionDict,
        coordinate_index_decoder: CoordinateIndexDecoder
    ) -> None:
        """
        Initialize a new game board.

        Args:
            piece_manager: Manager for game pieces and their orientations
            initial_actions: Pre-calculated valid moves for each player
            coordinate_index_decoder: Converts between coordinate systems
        """
        self.n = 14  # Board size
        self.piece_manager = piece_manager
        self.initial_actions = initial_actions
        self._coordinate_index_decoder = coordinate_index_decoder

        self._white_player = Player(self.n, Side("White"), piece_manager.pieces)
        self._black_player = Player(self.n, Side("Black"), piece_manager.pieces)
        self._players = [self._white_player, self._black_player]

        self.as_array = np.zeros(self.n ** 2, dtype=int).reshape(self.n, self.n)

    def _get_player(self, player_side: PlayerSide) -> Player:
        """
        Get the Player object for a given player side.

        Args:
            player_side: Player identifier (1 for White, -1 for Black)

        Returns:
            Player: The corresponding Player object

        Raises:
            ValueError: If player_side is not 1 or -1
        """
        match player_side:
            case 1:
                return self._white_player
            case -1:
                return self._black_player
            case _:
                raise ValueError(f"Cannot decode player {player_side}, must be +/- 1!")

    @property
    def net_representation(self) -> BoardArray:
        """
        Get the board state in the format expected by the neural network.

        The network representation includes:
        1. Black's played pieces region (14x2)
        2. Main board state (14x14)
        3. White's played pieces region (14x2)

        Returns:
            BoardArray: Combined board state of shape (14, 18)
        """
        return np.concatenate(
            (self._black_player.pieces_played_encoded_region.reshape(self.n, 2),
             self.as_array, self._white_player.pieces_played_encoded_region.reshape(self.n, 2)), axis=1
        )

    def _piece_insertion_error_message(
        self,
        sided_piece_orientation: BoardArray,
        x_coordinate: int,
        y_coordinate: int
    ) -> str:
        """
        Generate an error message for piece insertion failures.

        Args:
            sided_piece_orientation: The piece to be placed (with correct player sign)
            x_coordinate: X coordinate of the attempted placement
            y_coordinate: Y coordinate of the attempted placement

        Returns:
            str: Formatted error message with board and piece state
        """
        return (f"Board: \n {self.as_array}, \n "
                f"Piece: \n {sided_piece_orientation}, \n"
                f"At ({x_coordinate}, {y_coordinate})")

    def insert_piece(self, action: Action, player_side: PlayerSide) -> None:
        """
        Place a piece on the board.

        This method:
        1. Validates the piece placement
        2. Updates the board state
        3. Updates player state (pieces remaining, played pieces)
        4. Updates valid move caches for both players

        Args:
            action: The move to execute
            player_side: Player identifier (1 for White, -1 for Black)

        Raises:
            IndexError: If the piece doesn't fit on the board
            RuntimeError: If trying to place on an occupied square
        """
        piece_orientation = self.piece_manager.get_piece_orientation_array(action.piece_id, action.orientation)
        sided_piece_orientation = piece_orientation * player_side
        length_idx, width_idx = self._coordinate_index_decoder.to_idx(
            coordinate=(action.x_coordinate, action.y_coordinate))

        piece_length, piece_width = sided_piece_orientation.shape
        for i in range(piece_length):
            for j in range(piece_width):
                if length_idx + i > 13 or width_idx + j > 13:
                    raise IndexError(f"Piece cannot be inserted at this index, it does not fit on the board! "
                                   f"\n{self._piece_insertion_error_message(sided_piece_orientation, action.x_coordinate, action.y_coordinate)}")
                if self.as_array[length_idx + i, width_idx + j] != 0:
                    raise RuntimeError(f"Trying to insert piece into board over existing piece! \n"
                                     f"{self._piece_insertion_error_message(sided_piece_orientation, action.x_coordinate, action.y_coordinate)}")

                self.as_array[length_idx + i, width_idx + j] = sided_piece_orientation[i, j]

        player = self._get_player(player_side=player_side)
        player.pieces_played_encoded_region[action.piece_id - 1] = player_side
        player.piece_ids_played.append(action.piece_id)
        del player.pieces_remaining[action.piece_id]

        for p in self._players:
            p.update_potential_placement_points(
                insertion_index=(length_idx, width_idx),
                piece_orientation=piece_orientation,
                board_array=self.as_array
            )

    def valid_moves(self, player_side: PlayerSide) -> List[Action]:
        """
        Get all valid moves for the current player.

        Args:
            player_side: Player identifier (1 for White, -1 for Black)

        Returns:
            List[Action]: List of valid moves
        """
        player = self._get_player(player_side=player_side)
        # TODO: Implement move generation logic
        return []

    def game_ended(self) -> bool:
        """
        Check if the game has ended.

        The game ends when:
        1. A player has no pieces remaining, or
        2. Neither player has any legal moves

        Returns:
            bool: True if the game has ended
        """
        raise NotImplementedError()

    @staticmethod
    def _calculate_score(player: Player) -> int:
        """
        Calculate the score for a player.

        Scoring rules:
        1. +15 points for placing all pieces
        2. +5 bonus points for placing the smallest piece last
        3. Negative points for remaining pieces (sum of squares)

        Args:
            player: The player to score

        Returns:
            int: The player's score
        """
        if len(player.pieces_remaining) == 0:
            score = 15
            last_played_piece_id = player.piece_ids_played[-1]
            if last_played_piece_id == 1:  # Is the smallest piece
                score += 5
        else:
            # Scoring is negative in the rules (doesn't matter for the model)
            score = -np.sum([v.identity.sum() for k, v in player.pieces_remaining.items()])

        return score

    def game_result(self, player: PlayerSide) -> float:
        """
        Get the game result from a player's perspective.

        Args:
            player: Player identifier (1 for White, -1 for Black)

        Returns:
            float: Game result where:
                  1.0 = player won
                  -1.0 = player lost
                  1e-4 = draw

        Raises:
            RuntimeError: If called before the game has ended
        """
        if not self.game_ended():
            raise RuntimeError("Trying to calculate the result of a game that hasn't ended!")

        white_score = self._calculate_score(self._white_player)
        black_score = self._calculate_score(self._black_player)

        white_win = white_score > black_score

        if white_score == black_score:  # Draw
            return 1e-4

        if white_win == (player == 1):  # White won & we are White OR white didn't win and we are not White
            return 1
        else:  # If game isn't over, we didn't win & we didn't draw => we lost
            return -1


# TODO: Need to rethink the structure of this whole class
#       Not sure we should be passing around the board object, feels better to let it be private state
#       Let's get it to work first then refactor!
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
        self.board_size = 14  # Board size
        self.num_planes = 91  # Number of distinct Piece-BasisOrientation combinations
        self.piece_manager: PieceManager = pieces_loader(pieces_config_path)
        self.board: Optional[BlokusDuoBoard] = None

        # Canonically will have black start in top left and white in bottom right
        # Black encoded region will be on the left and White encoded region on the right of the board
        self.white_start = (9, 9)
        self.black_start = (4, 4)
        self._coordinate_index_decoder = CoordinateIndexDecoder(14)

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
        self.board = BlokusDuoBoard(self.piece_manager, self.initial_actions, self._coordinate_index_decoder)
        return self.board

    def get_board_size(self) -> Tuple[int, int]:
        """
        Get the dimensions of the game board.

        Returns:
            Tuple[int, int]: (width, height) of the board
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
        canonical_board: BoardArray,
        player: PlayerSide,
        action: Action
    ) -> Tuple[BoardArray, PlayerSide]:
        """
        Apply an action to the current board state.

        Args:
            canonical_board: Current board state in canonical form
            player: Current player (1 for White, -1 for Black)
            action: Action to apply

        Returns:
            Tuple containing:
                - BoardArray: New board state in canonical form
                - PlayerSide: Next player to move (-player)
        """
        self.board.insert_piece(action=action, player_side=player)
        return self.board.net_representation, -player

    def valid_move_masking(self, board: BoardArray, player: PlayerSide) -> BoardArray:
        """
        Generate a mask of valid moves for the current board state.

        Args:
            board: Current board state in canonical form
            player: Current player (1 for White, -1 for Black)

        Returns:
            BoardArray: Binary mask where 1 indicates valid moves
        """
        valid_actions = self.board.valid_moves(player)
        raise NotImplementedError()

    def get_game_ended(self, board: BlokusDuoBoard, player: PlayerSide) -> float:
        """
        Check if the game has ended and return the result.

        Args:
            board: Current board state
            player: Current player (1 for White, -1 for Black)

        Returns:
            float: Game result where:
                  0.0 = game continues
                  1.0 = player won
                  -1.0 = player lost
                  1e-4 = draw

        Raises:
            RuntimeError: If called before a game has started
        """
        if not board:
            raise RuntimeError("Trying to figure if we have ended a game before one was started!")
        if board.game_ended():
            return board.game_result(player)
        else:
            return 0

    def get_canonical_form(self, board: BlokusDuoBoard, player: PlayerSide) -> BoardArray:
        """
        Convert the board to its canonical form.

        The canonical form should be independent of player perspective.
        For this game, we:
        1. Keep the board orientation fixed
        2. Multiply the board by the player value to get their perspective
        3. Include the encoded regions for played pieces

        Args:
            board: Current board state
            player: Current player (1 for White, -1 for Black)

        Returns:
            BoardArray: Board state in canonical form
        """
        net_representation = board.net_representation
        return player * net_representation

    def get_symmetries(self, board: BoardArray, pi: BoardArray) -> List[Tuple[BoardArray, BoardArray]]:
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
            List[Tuple[BoardArray, BoardArray]]: List of (board, pi) pairs for each symmetry
        """
        raise NotImplementedError()

    def string_representation(self, board: BoardArray) -> str:
        """
        Convert the board state to a string representation.

        This is used by MCTS for hashing board states.

        Args:
            board: Current board state

        Returns:
            str: String representation of the board state
        """
        return board.tobytes()
