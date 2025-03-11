from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, Generator

import numpy as np
from nptyping import NDArray

from blokusduo.pieces import pieces_loader, Piece, Orientation, PieceManager, BidirectionalDict
from core.interfaces import IGame


# TODO: Seems to me like there are two important but distinct board objects.
#       One is the regular board which represents the state of the game to the players.
#       One is the board form that is passed into the neural net aka the 'Canonical Board'
#       Potentially we should make this distinction more concrete so we know which Board object we have at any time.


def no_sides(i: int, j: int, side: int, board: NDArray) -> bool:
    """
    Returns True for a point in the board grid with index i, j, and side +/- 1, if the point does not touch the side of
     a friendly piece.
    :param i: length index
    :param j: width index
    :param side: int +/- 1 of the player in consideration
    :param board: numpy array of 1, 0, -1
    :return:
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


def at_least_one_corner(i: int, j: int, side: int, board: NDArray) -> bool:
    """
    Returns True for a point in the board grid with index i, j, and side +/- 1, if the point has at least one friendly
    piece in a corner position.
    :param i: length index
    :param j: width index
    :param side: int +/- 1 of the player in consideration
    :param board: numpy array of 1, 0, -1
    :return:
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


def valid_placement(i: int, j: int, side: int, board: NDArray) -> bool:
    """
    Returns True for a point if it can validly be placed on a blokus board.
    :param i: length index
    :param j: width index
    :param side: int +/- 1 of the player in consideration
    :param board: numpy array of 1, 0, -1
    :return:
    """
    if board[i, j] != 0:  # Square is occupied
        return False
    return at_least_one_corner(i, j, side, board) and no_sides(i, j, side, board)


# TODO: Make this available to all games & change framework to use it? Using +/- 1 everywhere is a bit pants
class Side(StrEnum):
    White = "White"
    Black = "Black"

    def to_int(self):
        match self.value:
            case "White":
                return 1
            case "Black":
                return -1


class CoordinateIndexDecoder:
    """
    Used to turn (x,y) coordinates into length_idx and width_idx values and vice versa
    Easier to keep this logic in one place.
    """

    def __init__(self, board_size: int):
        self.board_size = board_size

    def to_idx(self, coordinate: tuple[int, int]) -> tuple[int, int]:
        length_idx = self.board_size - 1 - coordinate[1]
        width_idx = coordinate[0]
        return length_idx, width_idx

    def to_coordinate(self, idx: tuple[int, int]) -> tuple[int, int]:
        x = idx[1]
        y = self.board_size - 1 - idx[0]
        return x, y


# TODO: Generalise the project to use some inheritance of a more general Action later
@dataclass(frozen=True)
class Action:
    piece_id: int
    orientation: Orientation
    x_coordinate: int
    y_coordinate: int


class Player:
    # TODO: Update Docstring
    """
    It is super inefficient to check the whole board every time you want to make a move (and also not human-like).
    We can only place pieces touching an existing corner, so if we keep track of where the corners are on the board, we
    save a lot of computation.
    We keep track for each colour.
    Also keep track of which pieces you have already tried to place at a corner (once you cannot place a piece at any
    point in a game it is never placeable at that point).
    """

    def __init__(self, board_size: int, side: Side, pieces: dict[int, Piece]):
        self.side = side

        # TODO: Maybe this can just be a set of ids used tbh
        self.pieces_remaining = deepcopy(pieces)

        # TODO: Don't forget you need this for the encoded region and scoring
        #       Can potentially just change this to last id played since we are already updating the encoded region
        self.piece_ids_played: deque[int] = deque()

        # Encode information about which pieces white and black have played by adding these arrays to the White and
        # Black sections of the board (after reshaping).
        # Could generalise this by calculating how big the array needs to be based on n and the number of pieces.
        self.pieces_played_encoded_region = np.zeros(2 * board_size, dtype=int)


        # Dictionary of placemwnt point -> piece_id -> action
        self.potential_placement_points: dict[tuple, dict[int, Action]] = dict()

        # TODO: Maybe it makes more sense to have the corners to place to be on each player?

    def update_potential_placement_points(
            self,
            insertion_index: tuple[int, int],
            piece_orientation: NDArray,
            board_array: NDArray

    ):
        # TODO: Not sure which order is more efficient yet but
        #       1. Need to check surrounding rectangle around piece to see what new corners were added
        #          (bear in mind they could already be in the dictionary)
        #       2. Also check if the piece was laid on a potential placement point as obviously this removes it
        # Cap iteration range to make sure we don't go out of index bounds
        # Iterating + 1 unit around where the piece was placed
        p_l, p_w = piece_orientation.shape
        length_range = (max(0, insertion_index[0] - 1), min(insertion_index[0] + p_l + 1, 13))
        width_range = (max(0, insertion_index[1] - 1), min(insertion_index[1] + p_w + 1, 13))

        # Faster to check if key is in dict first then check if placement point is valid
        # If in dict and not valid remove it, stops us checking if a corner is valid if we know about it
        for i in range(length_range[0], length_range[1]):
            for j in range(width_range[0], width_range[1]):
                if (i, j) in self.potential_placement_points:  # Already think this point is potentially placeable
                    if not valid_placement(i, j, self.side.to_int(), board_array):
                        del self.potential_placement_points[(i, j)]
                else:  # We have no record of this point as being potentially placeable
                    if valid_placement(i, j, self.side.to_int(), board_array):
                        self.potential_placement_points[(i, j)] = self.populate_moves_for_new_placement(board_array)

    def populate_moves_for_new_placement(self, board_array: NDArray):
        for piece in self.pieces_remaining.values():
            for orientation in piece.basis_orientations:


class BlokusDuoBoard:
    """
    Helper class for managing a game of BlokusDuo.
    One of these should be initialised everytime we start a game of BlokusDuo.
    """

    def __init__(
            self,
            piece_manager: PieceManager,
            initial_actions: dict[int, list[Action]],
            coordinate_index_decoder: CoordinateIndexDecoder
    ):
        """
        Called when a game of BlokusDuo is initiated
        :param piece_manager (PieceManager)
        :param initial_actions: Calculated initial actions for White and Black
        :param coordinate_index_decoder: Converts between coordinates and indices
        """
        self.n = 14
        self.piece_manager = piece_manager
        self.initial_actions = initial_actions
        self._coordinate_index_decoder = coordinate_index_decoder

        self._white_player = Player(self.n, Side("White"), piece_manager.pieces)
        self._black_player = Player(self.n, Side("Black"), piece_manager.pieces)
        self._players = [self._white_player, self._black_player]

        self.as_array = np.zeros(self.n ** 2, dtype=int).reshape(self.n, self.n)

    def _get_player(self, player_side: int) -> Player:
        """
        Take an int, either +1 for White, -1 for Black and returns the corresponding Player
        :param player_side:
        :return:
        """
        match player_side:
            case 1:
                return self._white_player
            case -1:
                return self._black_player
            case _:
                raise ValueError(f"Cannot decode player {player_side}, must be +/- 1!")

    @property
    def net_representation(self) -> NDArray:
        """

        :return: npt.NDArray: Board with additional regions containing information about which pieces have been played.
                              Dimensions (14 * 2) + (14 * 14) + (14 * 2) = (14 * 18)
                              Is type int.
        """
        return np.concatenate(
            (self._black_player.pieces_played_encoded_region.reshape(self.n, 2),
             self.as_array, self._white_player.pieces_played_encoded_region.reshape(self.n, 2)), axis=1
        )

    def _piece_insertion_error_message(self, sided_piece_orientation: NDArray, x_coordinate: int, y_coordinate: int
                                       ) -> str:
        return (f"Board: \n {self.as_array}, \n "
                f"Piece: \n {sided_piece_orientation}, \n"
                f"At ({x_coordinate}, {y_coordinate})")

    def insert_piece(
            self,
            action: Action,
            player_side: int
    ) -> None:
        """
        Take an action and player side and insert this piece-orientation into the board for said player.
        Checks to see if the piece fits on the board, that it is not being laid on an existing piece and that the piece
        placement is valid within the placement rules of Blokus.
        Updates the Player classes to account for having made this piece insertion including updating cached information
        about possible placement squares for future turns.
        :param action: (Action) Holds piece_id, orientation, x and y coordinates for move (origin is bottom left square)
        :param player_side: (int) 1 for White and -1 for Black
        :return:
        """
        # TODO: Faster to remove validation checks, leave them in as a safeguard for now
        # Remember the origin is the bottom left corner of the board
        # We insert the piece with its top left corner in the x, y coordinate given
        piece_orientation = self.piece_manager.get_piece_orientation_array(action.piece_id, action.orientation)
        sided_piece_orientation = piece_orientation * player_side  # Make correct colour
        length_idx, width_idx = self._coordinate_index_decoder.to_idx(
            coordinate=(action.x_coordinate, action.y_coordinate))

        piece_length, piece_width = sided_piece_orientation.shape
        for i in range(piece_length):
            for j in range(piece_width):
                if length_idx + i > 13 or width_idx + j > 13:
                    raise IndexError(f"Piece cannot be inserted at this index, it does not fit on the board! "
                                     f"\n{self._piece_insertion_error_message(
                                         sided_piece_orientation, action.x_coordinate, action.y_coordinate)}")
                if self.as_array[length_idx + i, width_idx + j] != 0:
                    raise RuntimeError(f"Trying to insert piece into board over existing piece! \n"
                                       f"{self._piece_insertion_error_message(
                                           sided_piece_orientation, action.x_coordinate, action.y_coordinate)}")

                self.as_array[length_idx + i, width_idx + j] = sided_piece_orientation[i, j]

        player = self._get_player(player_side=player_side)
        player.pieces_played_encoded_region[action.piece_id - 1] = player_side
        player.piece_ids_played.append(action.piece_id)
        del player.pieces_remaining[action.piece_id]

        for p in self._players:
            p.update_potential_placement_points(insertion_index=(length_idx, width_idx),
                                                piece_orientation=piece_orientation,
                                                board_array=self.as_array
                                                )

    def valid_moves(self, player_side: int) -> list[Action]:
        # TODO: Need to think about the return form for this function, could be list of actions or NDArray (for net)
        #       ALGORITHM:
        #       If we have just started the game (aka cached corners == 0), output is valid_start_moves
        #       (injected by Game class on init)
        #       Else after we make a move, we cache the (x,y) coordinates around the corners of a piece
        #       Maybe do this in a multi threaded way!?
        player = self._get_player(player_side=player_side)

        # if len(player.potential_placement_points) == 0:
        #     return self.initial_actions[player_side]
        # else:
        #     for point in player.potential_placement_points:
        #         for piece_id, piece in player.pieces_remaining.items():
        #             for orientation in piece.basis_orientations:
        #                 if (piece_id, orientation) in player.p

    def game_ended(self) -> bool:
        """

        :return: (bool) Whether game has concluded

        If no pieces => Ended.
        Either player has no legal moves => Ended.

        i.e. If no_legal_moves_white AND no_legal_moves_black
             END
        """

        raise NotImplementedError()

    @staticmethod
    def _calculate_score(player: Player) -> int:
        """

        :param player (Player): Either White or Black player
        :return: score (int)
        """

        if len(player.pieces_remaining) == 0:
            score = 15
            last_played_piece_id = player.piece_ids_played[-1]
            if last_played_piece_id == 1:  # Is the smallest piece
                score += 5
        else:
            # Scoring is apparently negative in the rules (doesn't matter for the model)
            score = -np.sum([v.identity.sum() for k, v in player.pieces_remaining.items()])

        return score

    def game_result(self, player: int) -> float:
        """

        :param player: Either 1 or -1 for White or Black respectively
        :return: 1 for player win, 1e-4 for draw, -1 for loss

        Go
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
    def __init__(self, pieces_config_path: Path):
        super().__init__()
        self.board_size = 14  # Board size
        self.num_planes = 91  # This is the number of distinct Piece-BasisOrientation combinations
        self.piece_manager: PieceManager = pieces_loader(pieces_config_path)
        self.board: Optional[BlokusDuoBoard] = None

        # Canonically will have black start in top left and white in bottom right
        # Black encoded region will be on the left and White encoded region on the right of the board
        self.white_start = (9, 9)
        self.black_start = (4, 4)
        self._coordinate_index_decoder = CoordinateIndexDecoder(14)

        # Calculate all possible starting moves on game init and cache to save time, pass cache to board object
        # everytime we start a new game. Dictionary indexed by player (1 White, -1 Black)
        self.initial_actions: dict[int, list[Action]] = self._calculate_and_cache_initial_actions()

    def calculate_legal_initial_actions_for_piece_orientation(
            self,
            piece_id: int,
            orientation: Orientation,
            start: tuple[int, int]
    ) -> Generator[Action, None, None]:
        # Think about this as iterating over the piece, moving the board the other way
        # Want to return actions => piece_id, orientation, x_coord, y_coord
        piece_orientation = self.piece_manager.get_piece_orientation_array(piece_id, orientation)
        length, width = piece_orientation.shape
        for i in range(length):
            for j in range(width):
                if piece_orientation[(i, j)] == 1:
                    coordinate = self._coordinate_index_decoder.to_coordinate((start[0] - i, start[1] - j))
                    yield Action(piece_id, orientation, coordinate[0], coordinate[1])

    def _calculate_and_cache_initial_actions(self) -> dict[int, list[Action]]:
        # TODO: Figure out what data to store this as
        # TODO: Perhaps we should build the net pi policy masking here too
        # Dict of player int (1 for White, -1 for Black) into list of all actions
        start_moves: dict[int, list[Action]] = {1: [], -1: []}
        for piece_id, orientation in self.piece_manager.all_piece_id_basis_orientations():
            start_moves[1] += self.calculate_legal_initial_actions_for_piece_orientation(piece_id, orientation,
                                                                                         self.white_start)
            start_moves[-1] += self.calculate_legal_initial_actions_for_piece_orientation(piece_id, orientation,
                                                                                          self.black_start)
        return start_moves

    def initialise_board(self) -> BlokusDuoBoard:
        """
        We are only allowed to call this when we are starting a game of BlokusDuo.
        :return: BlokusDuoBoard
        """
        # TODO: Pass in the initial moves here.
        # Initialise a clean new BlokusDuo board.
        self.board = BlokusDuoBoard(self.piece_manager, self.initial_actions, self._coordinate_index_decoder)
        return self.board

    # TODO: Can I remove this?
    def get_board_size(self) -> tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.board_size, self.board_size

    def get_action_size(self) -> int:
        """
        Returns:
            Number of all possible actions.
            There are n * n (x,y) coordinates per piece
            There are num_planes = 91 unique rotations & flippings of each piece
            +1 for a do nothing action
        """
        raise (self.board_size ** 2) * self.num_planes + 1

    # TODO: Think it makes the most sense to not have to pass the net into this function but we'll leave it in the
    #       signature before we refactor.
    #       Having looked at MCTS, this needs to return an updated Board object, could even be a new board entirely.
    def get_next_state(
            self,
            canonical_board: NDArray,
            player: int,
            action: Action
    ) -> tuple[NDArray, int]:
        """

        :param canonical_board: Board in a form the net can accept
        :param player: (int), either 1 for White or -1 for Black
        :param action: (Action) Contains piece id, basis orientation, x and y coordinate.
                       NB: The origin for the x, y coordinate is the bottom left corner of the board
        :return: canonical_form, next_player: Return the board in the net representation and the negative of the player.
        """
        # TODO: Potentially return a board not board in net representation here
        self.board.insert_piece(action=action, player_side=player)

        return self.board.net_representation, -player

    # TODO: Figure out which version of the board this is / is it needed
    # TODO: I think maybe this method should live on the board object
    def valid_move_masking(self, board, player: int) -> NDArray:
        """

        :param board: Board in canonical format.
        :param player: (int) Either 1 for White or -1 for Black
        :return: NDArray of length action_size with either 1 or 0.
                 Used to mask the outputs of neural net.
        """
        valid_actions = self.board.valid_moves(player)
        raise NotImplementedError()

    # TODO: The way this was implemented is hacky and I don't like it
    # I think game_ended should maybe live on the board?
    def get_game_ended(self, board: BlokusDuoBoard, player: int) -> float:
        """
        Input:
            board (BlokusDuoBoard): current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if not board:
            raise RuntimeError(f"Trying to figure if we have ended a game before one was started!")
        if board.game_ended():
            return board.game_result(player)
        else:
            return 0

    def get_canonical_form(self, board: BlokusDuoBoard, player: int) -> NDArray:
        """
        Input:
            board (BlokusDuoBoard): Current board
            player (int): Current player (1 or -1)

        Returns:
            canonical_board: Returns canonical form of board. The canonical form should be independent of player e.g.
                             in chess, the canonical form is from the pov of white. When the player is black, we can
                             invert the colors and return the board.

        In addition to returning the board in canonical form, this is also the form that will be fed into the net.
        Hence, we must add on the extra columns containing information about which pieces have been played
        # TODO: We might need to rotate and flip the board here
        """
        net_representation = board.net_representation
        return player * net_representation

    def get_symmetries(self, board: NDArray, pi: NDArray):
        """
        Input:
            board (npt.NDArray): Current board, 14 * 14
            pi (npt.NDArray): Policy vector of size self.get_action_size()

        Returns:
            symmetry_forms: List of [(board,pi)] where each tuple is a symmetrical form of the board and the
                            corresponding pi vector. This is used when training the neural network from examples.

        The start positions for this game mode are off-centre:

                [0,  0,  0,  0]   Shrunk board representation for illustrative purposes
                [0, -1,  0,  0]    1 is White start position
                [0,  0,  1,  0]   -1 is Black start position
                [0,  0,  0,  0]

        The board has rotational symmetry 180 subject to black and white swapping
        Flipping the board about the vertical axis and rotating anticlockwise by 90 degrees is also valid
        Further rotating the flipped configuration 180 degrees is valid but again black and white must swap
        Therefore the symmetries are Identity, Rot180, Flip90, Flip270

        Identity => Black and White are unchanged
        Rot90 => Black and White swap positions
        Flip90 => Black and White unchanged
        Flip270 => Black and White swap
        """

        raise NotImplementedError

    def string_representation(self, board: NDArray):
        """
        Input:
            board: current board

        Returns:
            board_string: a quick conversion of board to a string format.
                          Required by MCTS for hashing.
        """
        raise board.tobytes()
