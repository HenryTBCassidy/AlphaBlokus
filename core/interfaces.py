from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def to_compact(self) -> NDArray:
        """Return the minimal compact array this board can be re-encoded from.

        This is the canonical (player-1 perspective) int8 representation that
        ``IGame.encode_compact`` rebuilds the full ``as_multi_channel(1)`` planes
        from. Storing this instead of the dense multi-channel tensor lets the
        training buffer hold positions at a fraction of the memory cost and
        re-encode lazily at batch time.

        Returns:
            A compact int8 NDArray (e.g. the 14×14 placement board for Blokus,
            the 3×3 grid for TicTacToe).
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

    def encode_compact(self, compact: NDArray) -> NDArray:
        """Rebuild the full neural-net input planes from a compact board array.

        Inverse of ``IBoard.to_compact``: takes the compact int8 array produced
        by ``board.to_compact()`` and returns the dense ``(C, H, W)`` float32
        tensor. The result must be exactly equal to ``board.as_multi_channel(1)``
        for the board the compact array came from.

        Args:
            compact: Compact int8 array from ``IBoard.to_compact``.

        Returns:
            NDArray of shape ``(C, H, W)`` with float32 dtype.
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
        """Train the neural network with full passes over the replay buffer.

        Args:
            examples: The whole replay buffer flattened to (board, policy, value)
                tuples — every position is trained on ``epochs`` times.
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

    def predict_batch(self, boards: Sequence[IBoard]) -> tuple[list[NDArray], list[float]]:
        """Run the network on N boards in a single forward pass.

        Equivalent to ``[self.predict(b) for b in boards]`` but executes the
        forward pass once with batch dimension ``len(boards)``. This is the
        core primitive behind batched MCTS inference — the speedup comes
        entirely from collapsing N batch-of-1 GPU calls into one batch-of-N
        call.

        Args:
            boards: Board objects in canonical form (player 1 perspective).

        Returns:
            Tuple of ``(policies, values)`` — lists of length ``len(boards)``,
            each policy a length-``action_size`` array, each value a float
            in ``[-1, 1]``.
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
