from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from core.config import RunConfig
from core.storage import MetricsCollector

# Type aliases for commonly used types
TrainingExample: TypeAlias = tuple[NDArray, NDArray, float]  # (board_state, policy_vector, value)
PolicyValue: TypeAlias = tuple[NDArray, float]  # (policy_vector, value_prediction)


class IBoard(Protocol):
    """Immutable snapshot of a game state.

    A board knows its own geometry (dimensions, channels) and contents
    (which pieces are where), but not the rules of the game. It provides:

    - State encoding for neural net input (``as_multi_channel``)
    - A flat game-state view (``as_2d``)
    - A hashable identity (``state_key``)
    - Canonical form for MCTS (``canonical``)
    - Per-player state accessors (remaining pieces, placement caches, etc.)

    Game rules (legal moves, win detection, scoring) live on IGame, not here.
    """

    @property
    def as_2d(self) -> NDArray:
        """Flat H×W game state board (±1/0). Used by game logic."""
        ...

    def as_multi_channel(self, current_player: int) -> NDArray:
        """Build multi-channel tensor for neural net input.

        The tensor is in canonical form: the current player's data is always
        in the first channel group, the opponent's in the second.

        Args:
            current_player: Whose turn it is (1 or -1).

        Returns:
            NDArray of shape ``(C, H, W)`` with float32 dtype.
        """
        ...

    @property
    def num_channels(self) -> int:
        """Number of channels in the ``as_multi_channel`` output."""
        ...

    @property
    def state_key(self) -> bytes:
        """Hashable key that uniquely identifies this board state.

        Used by MCTS as a dictionary key for state lookups.
        """
        ...

    def canonical(self, player: int) -> IBoard:
        """Return the board in canonical form (player 1 perspective).

        When player is 1, returns self (or a copy — implementation decides).
        When player is -1, flips pieces so the current player's stones are +1.

        Args:
            player: Current player (1 or -1).
        """
        ...


class IGame(Protocol):
    """Rules engine and action space for a two-player, adversarial, turn-based game.

    The game knows:
    - How to create a board (``initialise_board``)
    - Board dimensions and action space size
    - What moves are legal (``valid_move_masking``)
    - How to apply moves (``get_next_state``)
    - Whether the game is over and who won (``get_game_ended``)
    - Symmetries for data augmentation (``get_symmetries``)

    The game does NOT hold mutable state — it operates on IBoard snapshots
    passed as arguments. Player identifiers: 1 (player 1), -1 (player 2).
    """

    def initialise_board(self) -> IBoard:
        """Create and return the initial game board state."""
        ...

    def get_board_size(self) -> tuple[int, int]:
        """Get the dimensions of the game board.

        Returns:
            A tuple of (rows, cols) representing board dimensions.
        """
        ...

    def get_action_size(self) -> int:
        """Get the total number of possible actions in the game."""
        ...

    def get_next_state(self, board: IBoard, player: int, action: int) -> tuple[IBoard, int]:
        """Apply an action to the current board state and return the resulting state.

        Args:
            board: Current board state.
            player: Current player (1 or -1).
            action: Action to be applied (index into the action space).

        Returns:
            Tuple of (next board state, next player).
        """
        ...

    def valid_move_masking(self, board: IBoard, player: int) -> NDArray:
        """Generate a binary mask of valid moves for the current board state.

        Args:
            board: Current board state.
            player: Current player (1 or -1).

        Returns:
            Binary vector of length ``get_action_size()``.
        """
        ...

    def get_game_ended(self, board: IBoard, player: int) -> float:
        """Check if the game has ended and return the result.

        Args:
            board: Current board state.
            player: Current player (1 or -1).

        Returns:
            0.0 = game continues, 1.0 = player won, -1.0 = player lost,
            small non-zero value = draw (e.g. 1e-4).
        """
        ...

    def get_canonical_form(self, board: IBoard, player: int) -> IBoard:
        """Convert the board to canonical form (player 1 perspective).

        Args:
            board: Current board state.
            player: Current player (1 or -1).

        Returns:
            Board in canonical form.
        """
        ...

    def get_symmetries(self, board: IBoard, pi: NDArray) -> list[tuple[IBoard, NDArray]]:
        """Generate all symmetric forms of the board state and policy vector.

        Used to augment training data.

        Args:
            board: Current board state.
            pi: Policy vector of size ``get_action_size()``.

        Returns:
            List of (board, pi) tuples representing symmetrical forms.
        """
        ...

    def state_key(self, board: IBoard) -> bytes:
        """Return a hashable key that uniquely identifies the board state.

        Used by MCTS as a dictionary key for state lookups.
        """
        ...


class INeuralNetWrapper(Protocol):
    """Interface for neural network wrappers used in game-playing models.

    Handles training on self-play data, making predictions for game states,
    and model persistence (saving/loading checkpoints).

    The neural network outputs both a policy vector (probabilities for each
    possible move) and a value prediction (estimated game outcome).
    """

    def train(
        self,
        examples: list[TrainingExample],
        generation: int,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Train the neural network on a batch of examples from self-play games.

        Args:
            examples: List of (board_state, policy_vector, value) tuples.
            generation: Current iteration in the self-play training cycle.
            metrics: Optional metrics collector for recording training loss data.
        """
        ...

    def predict(self, board: IBoard) -> PolicyValue:
        """Make a prediction for a given board state.

        Args:
            board: Board object in canonical form (player 1 perspective).

        Returns:
            Tuple of (policy vector, value prediction).
        """
        ...

    def save_checkpoint(self, filename: str) -> None:
        """Save the current model state to a checkpoint file."""
        ...

    def load_checkpoint(self, filename: str) -> None:
        """Load a model state from a checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        ...
