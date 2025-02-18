from dataclasses import dataclass
from pathlib import Path

LOGGER_NAME: str = "Alpha"


@dataclass(frozen=True)
class MCTSConfig:
    num_mcts_sims: int  # Number of games moves for MCTS to simulate.
    cpuct: int  # Something to do with how nodes are selected TODO: Understand


@dataclass(frozen=True)
class NetConfig:
    learning_rate: float  # TODO: Put this on a schedule
    dropout: float
    epochs: int
    batch_size: int
    cuda: bool
    num_channels: int
    num_residual_blocks: int


# TODO: Add args for whether or not to log out data
@dataclass(frozen=True)
class RunConfig:
    run_name: str  # Used to distinguish between data from different run_configurations
    num_generations: int  # Number of times the algorithm completes the self play -> train -> upgade net cycle
    num_eps: int  # Number of complete self-play games to simulate during each generation
    temp_threshold: int  # TODO: Annotate
    update_threshold: float  # New neural net will be accepted if this threshold or more of games are won.
    max_queue_length: int  # Num board states remembered per generation (greater than num_moves bc of symmetry calc).
    num_arena_matches: int  # Number of games to play during arena play to determine if new net will be accepted.
    root_directory: Path  # Folder name of root where program run_configurations
    load_model: bool  # TODO: Annotate
    max_generations_lookback: int  # Maximum number of generations remembered during training i.e. buffer size
    mcts_config: MCTSConfig  # Class holding parameters used in Monte Carlo Tree Search
    net_config: NetConfig  # Class holding useful variables to parameterise neural net

    @property
    def run_directory(self): return self.root_directory / self.run_name

    @property
    def log_directory(self): return self.run_directory / "Logs"

    @property
    def timings_directory(self): return self.run_directory / "Timings"

    @property
    def self_play_history_directory(self): return self.run_directory / "SelfPlayHistory"

    @property
    def net_directory(self): return self.run_directory / "Nets"

    @property
    def training_data_directory(self): return self.run_directory / "TrainingData"

    @property
    def arena_data_directory(self): return self.run_directory / "ArenaData"

    @property
    def report_directory(self): return self.run_directory / "Reporting"
