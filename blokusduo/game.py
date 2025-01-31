from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, Generator

import numpy as np
from nptyping import NDArray

from blokusduo.pieces import pieces_loader, Piece, Orientation, PieceManager
from core.interfaces import IGame


# TODO: Seems to me like there are two important but distinct board objects.
#       One is the regular board which represents the state of the game to the players.
#       One is the board form that is passed into the neural net aka the 'Canonical Board'
#       Potentially we should make this distinction more concrete so we know which Board object we have at any time.


# TODO: Make this available to all games & change framework to use it? Using +/- 1 everywhere is a bit pants
class Side(StrEnum):
    White = "White"
    Black = "Black"


class CoordinateIndexDecoder:
    """
    Used to turn (x,y) coordinates into length_idx and width_idx values and vice versa
    Easier to keep this logic in one place.
    """

    @staticmethod
    def to_idx(board_size: int, coordinate: tuple[int, int]) -> tuple[int, int]:
        length_idx = board_size - 1 - coordinate[1]
        width_idx = coordinate[0]
        return length_idx, width_idx

    @staticmethod
    def to_coordinate(board_size: int, idx: tuple[int, int]) -> tuple[int, int]:
        x = idx[1]
        y = board_size - 1 - idx[0]
        return x, y


# TODO: Generalise the project to use some inheritance of an Action later
@dataclass
class Action:
    piece_id: int
    orientation: Orientation
    x_coordinate: int
    y_coordinate: int


class Player:
    def __init__(self, side: Side, pieces: dict[int, Piece]):
        self.side = side

        # TODO: Maybe this can just be a set of ids used tbh
        self.pieces_remaining = deepcopy(pieces)

        # TODO: Don't forget you need this for the encoded region and scoring
        #       Can potentially just change this to last id played since we are already updating the encoded region
        self.piece_ids_played: deque[int] = deque()

        # TODO: Maybe it makes more sense to have the corners to place to be on each player?


class MoveCache:
    """
    It is super inefficient to check the whole board every time you want to make a move (and also not human-like).
    We can only place pieces touching an existing corner, so if we keep track of where the corners are on the board, we
    save a lot of computation.
    We keep track for each colour.
    Also keep track of which pieces you have already tried to place at a corner (once you cannot place a piece at any
    point in a game it is never placeable at that point).
    """

    def __init__(self):
        # TODO: Needs to be able to split by side (or not if the corners live on the player)
        #       Lookup comes from a corner coordinate, (x, y)
        #       Probably is dict[Side, dict[coordinate, list_of_tried_ids]
        #       Needs to have a cleanup method
        # TODO: Hate this
        self._inner = {Side("White"): {}, Side("Black"): {}}


class BlokusDuoBoard:
    """
    Helper class for managing a game of BlokusDuo.
    One of these should be initialised everytime we start a game of BlokusDuo.
    """

    def __init__(self, piece_manager: PieceManager):
        """
        Called when a game of BlokusDuo is initiated
        :param piece_manager (dict[int, Piece])
        """
        self.n = 14
        self.piece_manager = piece_manager

        # Canonically will have black start in top left and white in bottom right
        # Black encoded region will be on the left and White encoded region on the right of the board
        self.black_start = (4, 4)
        self.white_start = (9, 9)

        # We can only place pieces touching an existing corner, keep track of where the corners are on the board.
        # Also track which pieces we have already tried for a corner.
        # TODO: FIX self._move_cache =

        self._white_player = Player(Side("White"), piece_manager.pieces)
        self._black_player = Player(Side("Black"), piece_manager.pieces)

        # Encode information about which pieces white and black have played by adding these arrays to the White and
        # Black sections of the board (after reshaping).
        # Could generalise this by calculating how big the array needs to be based on n and the number of pieces.
        self._white_pieces_played_encoded_region = np.zeros(2 * self.n, dtype=int)
        self._black_pieces_played_encoded_region = np.zeros(2 * self.n, dtype=int)

        self._board_representation = np.zeros(self.n ** 2, dtype=int).reshape(self.n, self.n)

    @property
    def net_representation(self) -> NDArray:
        """

        :return: npt.NDArray: Board with additional regions containing information about which pieces have been played.
                              Dimensions (14 * 2) + (14 * 14) + (14 * 2) = (14 * 18)
                              Is type int.
        """
        return np.concatenate(
            (self._black_pieces_played_encoded_region.reshape(self.n, 2),
             self._board_representation, self._white_pieces_played_encoded_region.reshape(self.n, 2)), axis=1
        )

    def insert_piece(
            self,
            action: Action,
            player: int
    ) -> None:
        """

        :param action: (Action) Holds piece_id, orientation, x and y coordinates for move (origin is bottom left square)
        :param player: (int) 1 for White and -1 for Black
        :return:
        """
        # TODO: Maybe check for index out of bound exceptions? Should have been passed a valid action but you never know
        # Remember the origin is the bottom left corner of the board
        # We insert the piece with its top left corner in the x, y coordinate given

        piece_orientation = self.piece_manager.get_piece_orientation_array(action.piece_id, action.orientation)
        sided_piece_orientation = piece_orientation * player  # Make correct colour
        length_idx, width_idx = CoordinateIndexDecoder.to_idx(board_size=self.n,
                                                              coordinate=(action.x_coordinate, action.y_coordinate))

        piece_length, piece_width = sided_piece_orientation.shape
        for i in range(piece_length):
            for j in range(piece_width):
                self._board_representation[length_idx + i, width_idx + j] = sided_piece_orientation[i, j]

        if player == 1:
            self._white_pieces_played_encoded_region[action.piece_id - 1] = 1
            self._white_player.piece_ids_played.append(action.piece_id)
            del self._white_player.pieces_remaining[action.piece_id]

        if player == -1:
            self._black_pieces_played_encoded_region[action.piece_id - 1] = -1
            self._black_player.piece_ids_played.append(action.piece_id)
            del self._black_player.pieces_remaining[action.piece_id]

        # TODO: WHEN WE INSERT A PIECE WE NEED TO UPDATE POSSIBLE CORNERS FOR WHITE/BLACK TO PLAY
        #       This includes both adding new corners for selection and removing some
        #       Don't forget that a move Black makes could change White's corners and vice versa
        # raise NotImplementedError()

    def valid_moves(self, player: int) -> NDArray:
        # TODO: THIS IS HARD, need to be efficient!
        #       ALGORITHM:
        #       If we have just started the game (aka cached corners == 0), output is valid_start_moves
        #       (injected by Game class on init)
        #       Else after we make a move, we cache the (x,y) coordinates around the corners of a piece

        if player == 1:
            _player = self._white_player
        elif player == -1:
            _player = self._black_player

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

        # Calculate all possible starting moves on game init and cache to save time, pass cache to board object
        # everytime we start a new game. Dictionary indexed by player (1 White, -1 Black)
        self.initial_actions: dict[int, list[Action]] = self._calculate_and_cache_initial_actions()

    def calculate_legal_actions_for_piece_orientation(
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
                    coordinate = CoordinateIndexDecoder.to_coordinate(self.board_size, (start[0] - i, start[1] - j))
                    yield Action(piece_id, orientation, coordinate[0], coordinate[1])

    def _calculate_and_cache_initial_actions(self) -> dict[int, list[Action]]:
        # TODO: Figure out what data to store this as
        # TODO: Perhaps we should build the net pi policy masking here too
        # Initialise a board just for this calculation
        board = BlokusDuoBoard(self.piece_manager)
        # Dict of player int (1 for White, -1 for Black) into list of all actions
        start_moves: dict[int, list[Action]] = {1: [], -1: []}
        for piece_id, orientation in self.piece_manager.all_piece_id_basis_orientations():
            start_moves[1] += self.calculate_legal_actions_for_piece_orientation(piece_id, orientation,
                                                                                 board.white_start)
            start_moves[-1] += self.calculate_legal_actions_for_piece_orientation(piece_id, orientation,
                                                                                  board.black_start)
        return start_moves

    def initialise_board(self) -> BlokusDuoBoard:
        """
        We are only allowed to call this when we are starting a game of BlokusDuo.
        :return: BlokusDuoBoard
        """
        # TODO: Pass in the initial moves here.
        # Initialise a clean new BlokusDuo board.
        self.board = BlokusDuoBoard(self.piece_manager)
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

        self.board.insert_piece(action=action, player=player)

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
