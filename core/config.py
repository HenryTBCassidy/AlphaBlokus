from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Global constants
LOGGER_NAME: Final[str] = "Alpha"


@dataclass(frozen=True)
class MCTSConfig:
    """
    Configuration parameters for Monte Carlo Tree Search (MCTS).
    
    MCTS is used to select moves during self-play by building a search tree
    and evaluating positions using the neural network. The search process is
    controlled by these parameters.
    """
    num_mcts_sims: int  # Number of MCTS simulations per move
    cpuct: float  # Exploration constant in the PUCT formula (typically between 1 and 4)


@dataclass(frozen=True)
class NetConfig:
    """
    Configuration parameters for the neural network.
    
    These parameters control both the architecture of the neural network
    and its training process. The network uses a residual architecture
    with convolutional layers for board state processing.
    """
    learning_rate: float  # Learning rate for the optimizer (TODO: Put this on a schedule)
    dropout: float  # Dropout probability for regularisation (0 to 1)
    epochs: int  # Number of training epochs per generation
    batch_size: int  # Number of positions per training batch
    cuda: bool  # Whether to use CUDA for GPU acceleration
    num_channels: int  # Number of channels in convolutional layers (power of 2)
    num_residual_blocks: int  # Number of residual blocks in the network


@dataclass(frozen=True)
class RunConfig:
    """
    Configuration parameters for a complete training run.
    
    This class holds all parameters needed to configure a training session,
    including hyperparameters for:
    - Self-play game generation
    - Neural network training
    - Model evaluation
    - Data storage and logging
    
    The training process consists of repeated cycles of:
    1. Self-play game generation using MCTS + current neural network
    2. Training the neural network on the generated games
    3. Evaluating the new network against the previous version
    """
    # Training process parameters
    run_name: str  # Unique identifier for this training run
    num_generations: int  # Number of complete self-play -> train -> evaluate cycles
    num_eps: int  # Number of complete self-play games per generation
    temp_threshold: int  # Move number after which temperature is set to ~0
    update_threshold: float  # Win rate required for new network to be accepted (0 to 1)
    max_queue_length: int  # Maximum number of game positions to keep in memory
    num_arena_matches: int  # Number of evaluation games between old/new networks
    max_generations_lookback: int  # Maximum number of past generations to train on
    
    # Model and file management
    root_directory: Path  # Root directory for all output files
    load_model: bool  # Whether to load a pre-existing model
    
    # Component configurations
    mcts_config: MCTSConfig  # Monte Carlo Tree Search parameters
    net_config: NetConfig  # Neural network parameters

    @property
    def run_directory(self) -> Path:
        """Base directory for all files related to this training run."""
        return self.root_directory / self.run_name

    @property
    def log_directory(self) -> Path:
        """Directory for log files tracking training progress and errors."""
        return self.run_directory / "Logs"

    @property
    def timings_directory(self) -> Path:
        """Directory for timing data of various training components."""
        return self.run_directory / "Timings"

    @property
    def self_play_history_directory(self) -> Path:
        """Directory for storing self-play game data used for training."""
        return self.run_directory / "SelfPlayHistory"

    @property
    def net_directory(self) -> Path:
        """Directory for neural network checkpoints from each generation."""
        return self.run_directory / "Nets"

    @property
    def training_data_directory(self) -> Path:
        """Directory for processed training data and metrics."""
        return self.run_directory / "TrainingData"

    @property
    def arena_data_directory(self) -> Path:
        """Directory for evaluation game results between network versions."""
        return self.run_directory / "ArenaData"

    @property
    def report_directory(self) -> Path:
        """Directory for generated training progress reports and visualisations."""
        return self.run_directory / "Reporting"
