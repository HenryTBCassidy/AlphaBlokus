class NeuralNet:
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
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
                      the given board, and v is its value. The examples has
                      board in its canonical form.
            iteration: int. Cycle number in training. Used for logging.
        """
        raise NotImplementedError

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
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
