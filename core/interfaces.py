from typing import Tuple, List, Union, Protocol, TYPE_CHECKING, TypeAlias
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.config import RunConfig

# Type aliases for commonly used types
TrainingExample: TypeAlias = Tuple[NDArray, NDArray, float]  # (board_state, policy_vector, value)
PolicyValue: TypeAlias = Tuple[NDArray, float]  # (policy_vector, value_prediction)

class IGame(Protocol):
    """
    Base interface for implementing two-player, adversarial, turn-based games.
    
    This class defines the contract that game implementations must follow.
    Player identifiers are standardised as:
        - Player 1: 1
        - Player 2: -1
        
    All board states are represented as numpy arrays with appropriate dimensions
    for the specific game being implemented.
    """

    def __init__(self) -> None:
        pass

    def initialise_board(self) -> NDArray:
        """
        Create and return the initial game board state.

        Returns:
            NDArray: A representation of the board (typically the input format for the neural network)
        """
        raise NotImplementedError

    def get_board_size(self) -> Tuple[int, int]:
        """
        Get the dimensions of the game board.

        Returns:
            Tuple[int, int]: A tuple of (width, height) representing board dimensions
        """
        raise NotImplementedError

    def get_action_size(self) -> int:
        """
        Get the total number of possible actions in the game.

        Returns:
            int: Number of all possible actions (moves) available in the game
        """
        raise NotImplementedError

    def get_next_state(self, board: NDArray, player: int, action: int) -> Tuple[NDArray, int]:
        """
        Apply an action to the current board state and return the resulting state.

        Args:
            board: Current board state as a numpy array
            player: Current player (1 or -1)
            action: Action to be applied (index into the action space)

        Returns:
            Tuple containing:
                - NDArray: Next board state after applying the action
                - int: Next player to play (-player)
        """
        raise NotImplementedError

    def valid_move_masking(self, board: NDArray, player: int) -> NDArray:
        """
        Generate a mask of valid moves for the current board state.

        Args:
            board: Current board state as a numpy array
            player: Current player (1 or -1)

        Returns:
            NDArray: Binary vector of length self.get_action_size(), where:
                    1 indicates a valid move
                    0 indicates an invalid move
        """
        raise NotImplementedError

    def get_game_ended(self, board: NDArray, player: int) -> float:
        """
        Check if the game has ended and return the result.

        Args:
            board: Current board state as a numpy array
            player: Current player (1 or -1)

        Returns:
            float: Game result where:
                  0.0 = game continues
                  1.0 = player won
                  -1.0 = player lost
                  small non-zero value = draw (e.g. 1e-4)
        """
        raise NotImplementedError

    def get_canonical_form(self, board: NDArray, player: int) -> NDArray:
        """
        Convert the board to its canonical form, which should be independent of player perspective.

        The canonical form should provide a standardised view of the board state,
        regardless of which player is currently playing. For example, in chess,
        the canonical form might always be from white's perspective.

        Args:
            board: Current board state as a numpy array
            player: Current player (1 or -1)

        Returns:
            NDArray: Board in canonical form
        """
        raise NotImplementedError

    def get_symmetries(self, board: NDArray, pi: NDArray) -> List[Tuple[NDArray, NDArray]]:
        """
        Generate all symmetric forms of the current board state and policy vector.

        This method should return all valid symmetrical representations of the current
        game state, along with appropriately transformed policy vectors. This is used
        to augment training data.

        Args:
            board: Current board state as a numpy array
            pi: Policy vector of size self.get_action_size()

        Returns:
            List[Tuple[NDArray, NDArray]]: List of (board, pi) tuples where each tuple
                                         represents a symmetrical form of the input
        """
        raise NotImplementedError

    def string_representation(self, board: NDArray) -> str:
        """
        Convert the board state to a string representation.

        This method is primarily used for hashing board states in MCTS.
        The string representation should uniquely identify the board state.

        Args:
            board: Current board state as a numpy array

        Returns:
            str: A unique string representation of the board state
        """
        raise NotImplementedError


class INeuralNetWrapper(Protocol):
    """
    Base interface for neural network wrappers used in game-playing models.
    
    This interface defines the contract for neural networks that can be used
    with the game-playing framework. It handles:
    - Training on self-play game data
    - Making predictions for game states
    - Model persistence (saving/loading checkpoints)
    
    The neural network is expected to output both:
    - A policy vector (probabilities for each possible move)
    - A value prediction (estimated game outcome from current position)
    """

    def __init__(self, game: IGame, args: 'RunConfig') -> None:
        """
        Initialise the neural network wrapper.

        Args:
            game: Game instance that this network will learn to play
            args: Configuration parameters for the network and training process
        """
        pass

    def train(self, examples: List[TrainingExample], generation: int) -> None:
        """
        Train the neural network on a batch of examples from self-play games.

        Args:
            examples: List of training examples, each containing:
                     - board_state (NDArray): The game state in canonical form
                     - policy_vector (NDArray): Target move probabilities
                     - value (float): Game outcome from this position (-1 to 1)
            generation: Current iteration in the self-play training cycle
        """
        raise NotImplementedError

    def predict(self, board: NDArray) -> PolicyValue:
        """
        Make a prediction for a given board state.

        The network outputs both a policy vector (move probabilities) and
        a value prediction (estimated game outcome) for the input state.

        Args:
            board: The game board in its canonical form

        Returns:
            Tuple containing:
                - NDArray: Policy vector (probabilities for each possible move)
                - float: Value prediction (-1 to 1) for the current board state
        """
        raise NotImplementedError

    def save_checkpoint(self, filename: str) -> None:
        """
        Save the current model state to a checkpoint file.

        This should save all necessary information to restore the model later,
        including:
        - Network architecture
        - Model weights
        - Training state (if needed)
        - Optimizer state (if needed)

        Args:
            filename: Path to save the checkpoint file
        """
        raise NotImplementedError

    def load_checkpoint(self, filename: str) -> None:
        """
        Load a model state from a checkpoint file.

        This should restore the model to exactly the same state it was in
        when saved, including:
        - Network architecture
        - Model weights
        - Training state (if saved)
        - Optimizer state (if saved)

        Args:
            filename: Path to the checkpoint file to load

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            ValueError: If the checkpoint file is invalid or incompatible
        """
        raise NotImplementedError
