import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dataclass_wizard import fromdict


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
    profiling_level: str = "standard"  # "none", "standard" (episode aggregates), "detailed" (per-move breakdown)

    # F3 (batched inference): number of leaf evaluations collected per MCTS
    # outer step before a single batched ``predict_batch`` call. ``1`` (default)
    # keeps the existing one-sim-per-NN-call path bit-for-bit identical to the
    # pre-F3 recursive search — the batched codepath still runs but with batch
    # size 1, so virtual loss is a no-op and selection/backprop arithmetic is
    # unchanged. Values > 1 collect K diversified leaves per step (via virtual
    # loss) and evaluate them in one GPU call, trading a slight search-quality
    # approximation for far better GPU utilisation.
    mcts_batch_size: int = 1


@dataclass(frozen=True)
class NetConfig:
    """
    Configuration parameters for the neural network.
    
    These parameters control both the architecture of the neural network
    and its training process. The network uses a residual architecture
    with convolutional layers for board state processing.
    """
    learning_rate: float  # Learning rate for the optimizer
    dropout: float  # Dropout probability for regularisation (0 to 1)
    epochs: int  # Number of training epochs per generation
    batch_size: int  # Number of positions per training batch
    cuda: bool  # Whether to use CUDA for GPU acceleration
    num_filters: int  # Number of convolutional filters per layer (power of 2)
    num_residual_blocks: int  # Number of residual blocks in the network
    lr_scheduler: str | None = None  # LR schedule: None = constant, "cosine" = CosineAnnealingLR

    # F4 (conv policy head): "fc" = the original fully-connected policy head
    # (a single Linear(2·cells → action_size), ~95% of the net's params);
    # "conv" = fully-convolutional head (1×1 conv to per-orientation logit planes
    # + a small pass head), ~1200× fewer params in that layer and a stronger
    # board-game inductive bias. Default "fc" keeps existing checkpoints/behaviour;
    # the two heads have incompatible state_dicts (loading across them raises).
    # Blokus only — TicTacToe ignores this. See docs/plans/archive/conv-policy-head.md.
    # Default "conv" since F4 (2026-06-02): correctness proven (read-out matches
    # ActionCodec), trains cleanly, ~21× fewer params / ~19× smaller checkpoints.
    # Set "fc" to restore the original fully-connected head (e.g. to load an
    # old FC checkpoint — the two head state_dicts are incompatible).
    policy_head: Literal["fc", "conv"] = "conv"


@dataclass(frozen=True)
class WandbConfig:
    """Configuration parameters for Weights & Biases logging.

    Optional. When absent from ``RunConfig`` (or set to ``None``), no W&B run
    is initialised and the training pipeline behaves exactly as before. When
    present, ``MetricsCollector`` mirrors its existing ``log_*`` calls to W&B
    in addition to the parquet writes used by the HTML report.
    """
    project: str  # W&B project name (e.g. "alphablokus-poc")
    entity: str | None = None  # W&B team/user; None uses the default for the logged-in account
    tags: list[str] = field(default_factory=list)  # Free-text tags surfaced in the W&B UI
    mode: Literal["online", "offline", "disabled"] = "online"  # Network mode for the W&B client


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
    # Game selection
    game: str  # Game to train on: "tictactoe" or "blokusduo"

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

    # Optional reporting backends
    wandb: WandbConfig | None = None  # If set, mirror metrics to Weights & Biases

    # Elo evaluation: number of games per generation to play vs the frozen
    # gen-0 baseline. 0 disables Elo tracking entirely. The default of 50 is
    # always-on by intention — Elo vs gen-0 is the headline "is the model
    # actually getting stronger?" curve for AlphaZero-style work.
    elo_games_per_gen: int = 50

    # Anchor rating shown for the gen-0 baseline. Display-only — the
    # underlying Elo difference math is unchanged. There's no universal
    # convention here:
    #
    # - AlphaGo Zero / AlphaZero papers anchor random nets at 0 Elo and let
    #   the curve climb monotonically. Works for them because at their scale
    #   it never dips below.
    # - USCF / Chess.com default unrated players to 1200; Lichess uses 1500.
    #   These imply "random plays at average human" which is misleading.
    #
    # 400 sits in the middle: above 0 (so a trained net that briefly learns
    # something worse-than-random has room to dip without going negative —
    # this can happen with early-gen overfitting on noisy MCTS targets),
    # but low enough to read as "weak baseline." Matches the scholastic /
    # kids'-tournament starting range; gives the converged model room to
    # climb to ~800-1200+ at full training.
    elo_baseline_rating: int = 400

    # TTT-specific: games per generation to play vs a perfect-play minimax
    # opponent. Only used when ``game == "tictactoe"``. 0 disables.
    minimax_games_per_gen: int = 20

    # Per-generation symmetry diagnostic: i.e. the number of randomly-
    # generated reference board positions on which we test the network's
    # asymmetric-preference (whether mirroring the board flips its raw
    # policy in the equivalent way). 0 disables the diagnostic. Same
    # seeded set of positions is used every generation so the metric is
    # cross-gen comparable. See ``core/symmetry_diagnostic.py`` for the
    # phase-distribution (heavy on early/mid game, lighter on late game).
    # Default 100 is essentially free in compute terms (~200 forward
    # passes per gen on Blokus, sub-second) while giving a stable point
    # estimate that isn't dominated by single-position variance.
    symmetry_diagnostic_positions: int = 100

    # Global RNG seed for numpy, torch, MCTS tie-breaks and the eval-set
    # sampler. Set to a fixed value to make a run bit-for-bit reproducible;
    # ``None`` skips seeding entirely (non-deterministic — only useful if you
    # want a single run to have stochastic warm-up). Two runs with the same
    # seed + same config + same hardware will produce identical metrics.
    seed: int | None = 42

    # F1 (parallel self-play): number of worker processes used for the
    # self-play / arena / Elo phases. ``1`` (default) keeps the existing
    # single-process behaviour bit-for-bit identical to before F1 landed —
    # the parallel codepath is not taken at all. Values > 1 spawn a
    # ``ProcessPoolExecutor`` with that many workers; each holds its own
    # copy of the network in its own CUDA context. Determinism is
    # preserved per-episode via a seed derived from
    # ``(seed, generation, episode_idx)``, so set membership of training
    # examples matches the serial path regardless of worker count.
    num_parallel_workers: int = 1

    # F2 (precomputed-move-list move generator): if True, BlokusDuoGame
    # routes ``valid_move_masking`` through the F2 implementation in
    # :mod:`games.blokusduo.movegen_runtime`. Default False to preserve
    # the existing array-based path. Produces bit-identical training
    # trajectories at the same seed (verified by
    # ``tests/test_blokusduo/test_movegen_determinism.py``). Only
    # ``BlokusDuoGame`` consults this flag; TTT ignores it.
    use_optimised_movegen: bool = False

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

    @property
    def self_play_profiling_directory(self) -> Path:
        """Directory for per-episode MCTS profiling data (sims, timing, tree size)."""
        return self.run_directory / "SelfPlayProfiling"

    @property
    def resource_usage_directory(self) -> Path:
        """Directory for process and GPU memory usage snapshots."""
        return self.run_directory / "ResourceUsage"

    @property
    def training_throughput_directory(self) -> Path:
        """Directory for per-epoch training throughput metrics."""
        return self.run_directory / "TrainingThroughput"

    @property
    def eval_set_directory(self) -> Path:
        """Directory holding the frozen held-out positions used for per-epoch
        network-entropy evaluation. Built once after the first generation's
        self-play and reused for every subsequent epoch."""
        return self.run_directory / "EvalSet"

    @property
    def training_entropy_directory(self) -> Path:
        """Directory for per-epoch network policy entropy on the held-out eval set."""
        return self.run_directory / "TrainingEntropy"

    @property
    def policy_accuracy_directory(self) -> Path:
        """Directory for per-epoch network top-1 / top-5 policy accuracy on the eval set."""
        return self.run_directory / "PolicyAccuracy"

    @property
    def value_calibration_directory(self) -> Path:
        """Directory for per-epoch network value-head reliability buckets on the eval set."""
        return self.run_directory / "ValueCalibration"

    @property
    def elo_ratings_directory(self) -> Path:
        """Directory for per-generation Elo rating measured against the frozen gen-0 baseline."""
        return self.run_directory / "EloRatings"

    @property
    def minimax_results_directory(self) -> Path:
        """Directory for per-generation results vs a perfect-play minimax opponent (TTT only)."""
        return self.run_directory / "MinimaxResults"

    @property
    def arena_replays_directory(self) -> Path:
        """Directory for recorded arena games (move sequences + top-K policies per move)."""
        return self.run_directory / "ArenaReplays"

    @property
    def symmetry_diagnostic_directory(self) -> Path:
        """Directory for per-generation policy-symmetry diagnostic results.

        Stores KL divergences between the network's raw policy on reference
        positions and the same policy reconstructed from the symmetric
        variant via ``game.get_symmetries``. See
        ``core/symmetry_diagnostic.py``.
        """
        return self.run_directory / "SymmetryDiagnostic"


def load_args(config_path: str | Path) -> RunConfig:
    """
    Load run configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        RunConfig: Configuration object for the run
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        args_json = json.load(f)

    return fromdict(RunConfig, args_json)
