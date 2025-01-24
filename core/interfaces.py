class IGame:
    """
    This class specifies the base Game interface. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.
    """

    def __init__(self):
        pass

    def get_init_board(self):
        """
        Returns:
            A representation of the board (ideally this is the form
            that will be the input to your neural network)
        """
        raise NotImplementedError

    def get_board_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        raise NotImplementedError

    def get_action_size(self):
        """
        Returns:
            Number of all possible actions
        """
        raise NotImplementedError

    def get_next_state(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            next_board: board after applying action
            next_player: player who plays in the next turn (should be -player)
        """
        raise NotImplementedError

    def get_valid_moves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            valid_moves: Binary vector of length self.getActionSize(), 1 for
                         moves that are valid from the current board and player,
                         0 for invalid moves
        """
        raise NotImplementedError

    def get_game_ended(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        raise NotImplementedError

    def get_canonical_form(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonical_board: Returns canonical form of board. The canonical form
                             should be independent of player. For e.g. in chess,
                             the canonical form can be chosen to be from the pov
                             of white. When the player is white, we can return
                             board as is. When the player is black, we can invert
                             the colors and return the board.
        """
        raise NotImplementedError

    def get_symmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmetry_forms: a list of [(board,pi)] where each tuple is a symmetrical
                            form of the board and the corresponding pi vector. This
                            is used when training the neural network from examples.
        """
        raise NotImplementedError

    def string_representation(self, board):
        """
        Input:
            board: current board

        Returns:
            board_string: a quick conversion of board to a string format.
                          Required by MCTS for hashing.
        """
        raise NotImplementedError


class INeuralNetWrapper:
    """
    This class specifies the base NeuralNet interface. To define your own neural
    network, subclass this interface and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.
    """

    def __init__(self, game, run_config):
        pass

    def train(self, examples, iteration: int):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples have the
                      board in its canonical form.
            generation: int. Cycle number in training. Used for logging.
        """
        raise NotImplementedError

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.get_action_size
            v: a float in [-1,1] that gives the value of the current board
        """
        raise NotImplementedError

    def save_checkpoint(self, filename: str):
        """
        Saves the current neural network (with its parameters) in
        the net folder directory set in config with the filename specified
        """
        raise NotImplementedError

    def load_checkpoint(self, filename: str):
        """
        Loads parameters of the neural network from the net folder set in config and with filename specified
        """
        raise NotImplementedError
