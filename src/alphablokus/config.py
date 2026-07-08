from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dataclass_wizard import fromdict
from loguru import logger

# Maps a game id (RunConfig.game) to the folder runs are grouped under inside
# ``<root>/runs/``. Keeps the output root tidy (temp/runs/blokus/…,
# temp/runs/tictactoe/…) rather than a flat pile. Unknown games → "other".
_GAME_GROUPS: dict[str, str] = {"blokusduo": "blokus", "tictactoe": "tictactoe"}

# Named net-size recipes: the budget-vs-strength knob for cloud runs
# (docs/plans/cloud-scale-training.md C5). A JSON ``net_config`` may say
# ``"preset": "large"`` instead of spelling out filters/blocks; explicit
# ``num_filters``/``num_residual_blocks`` keys always win over the preset.
# VRAM is not the constraint for this workload (44×14×14 activations are
# tiny) — the ceiling is what the run budget can train to usefulness.
#   small  = today's production net; medium = run3's "bignet";
#   large/xl = cloud-scale candidates (xl ≈ AlphaGo Zero's 256 filters at
#   14×14 depth-scaled). Throughput per size: scripts/benchmarks/cloud_calibration.py.
NET_PRESETS: dict[str, dict[str, int]] = {
    "small": {"num_filters": 64, "num_residual_blocks": 4},
    "medium": {"num_filters": 128, "num_residual_blocks": 8},
    "large": {"num_filters": 192, "num_residual_blocks": 12},
    "xl": {"num_filters": 256, "num_residual_blocks": 16},
}


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration parameters for Monte Carlo Tree Search (MCTS).

    MCTS selects moves during self-play by building a search tree and
    evaluating positions with the neural network; these parameters control
    that search.
    """

    num_mcts_sims: int  # Number of MCTS simulations per move
    cpuct: float  # Exploration constant in the PUCT formula (typically between 1 and 4)
    profiling_level: str = "standard"  # "none", "standard" (episode aggregates), "detailed" (per-move breakdown)

    # Dirichlet root-exploration noise (AlphaZero). During self-play only, the
    # root node's priors are mixed with Dirichlet noise:
    # ``P(s,a) = (1-ε)·p_a + ε·η_a``, ``η ~ Dir(α)`` over the legal moves. This
    # guarantees exploration of moves the (possibly-wrong, early-training) policy
    # rates poorly. ``dirichlet_epsilon = 0`` (default) disables it entirely — the
    # search is then bit-identical to pre-noise behaviour, so arena/Elo (which
    # never request noise) and existing tests are unaffected. AlphaZero used
    # ε=0.25; α≈0.03 suits Blokus's ~400 early legal moves (Go-like).
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 0.03

    # Number of leaf evaluations collected per MCTS outer step before a single
    # batched ``predict_batch`` call. ``1`` (default) keeps the
    # one-sim-per-NN-call path bit-for-bit identical to recursive search — the
    # batched codepath still runs but with batch size 1, so virtual loss is a
    # no-op and selection/backprop arithmetic is unchanged. Values > 1 collect
    # K diversified leaves per step (via virtual loss) and evaluate them in one
    # GPU call, trading a slight search-quality approximation for far better
    # GPU utilisation.
    mcts_batch_size: int = 1

    # Adaptive per-move simulation budget (IDEAS.md I1). ``"flat"`` (default)
    # spends ``num_mcts_sims`` every move — bit-identical to pre-taper behaviour.
    # ``"branching"`` scales the budget with the root's legal-move count:
    # ``sims = clamp(round(sim_branching_scale * branching), sims_min, num_mcts_sims)``
    # — so ``num_mcts_sims`` becomes the early-game cap and ``sims_min`` the
    # endgame floor. Blokus branching swings ~450 (opening) → <10 (endgame), so a
    # flat count is thin in the opening and wasteful in the endgame.
    sim_schedule: str = "flat"
    sims_min: int = 1
    sim_branching_scale: float = 1.0

    # Search policy — jax backend only (the python backend is PUCT-only and
    # warns if this is set to "gumbel"). "puct" = classic AlphaZero (Dirichlet
    # root noise + PUCT selection); "gumbel" = mctx's Gumbel AlphaZero
    # (Sequential Halving root, no Dirichlet/temperature, policy target =
    # completed-Q improved policy). Gumbel achieves equal-or-better policy
    # improvement at far fewer sims (n≈16–64) — the plan's G10 lever. This is a
    # deliberate behavioural change, opt-in and validated by its own A/B run.
    search_policy: Literal["puct", "gumbel"] = "puct"
    gumbel_max_considered: int = 16  # root actions Sequential Halving considers


@dataclass(frozen=True)
class JaxSelfPlayConfig:
    """Knobs for the GPU-native jax self-play backend (``selfplay_backend: "jax"``).

    Search hyperparameters (sims, cpuct, Dirichlet noise) come from
    ``MCTSConfig`` — same source of truth as the python backend; these are the
    jax-only execution knobs. Defaults chosen from the G4/G7 box sweeps
    (docs/plans/archive/jax-selfplay-pipeline.md).
    """

    batch_size: int = 256  # parallel game slots searched in lockstep
    # Compact per-node action space. mctx's per-sim tree traffic scales with
    # top_k (measured: K=128 is ~5x slower than K=64 at 128f×8b), while search
    # quality at K=64 still beats the python K=16 virtual-loss yardstick — see
    # the G4/G7 notes in docs/plans/archive/jax-selfplay-pipeline.md.
    top_k: int = 64
    dtype: str = "bfloat16"  # net inference dtype: "bfloat16" or "float32"
    wave_plies: int = 32  # scan horizon between host-side harvests

    # Fraction of VRAM XLA may claim (XLA_PYTHON_CLIENT_MEM_FRACTION). 0.4
    # suits an 8 GB card shared with torch (the box); raise on bigger cloud
    # cards so jax search isn't needlessly capped. An explicit env var always
    # wins over this value; torch/jax coexistence (PREALLOCATE=false) is
    # unaffected. Applied at the backend entry point, before the process's
    # first ``import jax``.
    xla_mem_fraction: float = 0.4


@dataclass(frozen=True)
class TrainingPerfConfig:
    """Opt-in performance knobs for the torch **training** loop.

    Everything here defaults to "off" = today's behaviour, so existing configs
    (and the Mac CPU dev path) are bit-identical unless a run opts in. CUDA-only
    knobs are inert on CPU — setting them in a CPU config is harmless. Sized for
    a single fast cloud GPU where the unmodernised fp32 single-threaded loop
    would leave the card starved (docs/plans/cloud-scale-training.md C2/C3).
    """

    # Mixed-precision autocast for the training forward+loss. "bf16" is the
    # right choice on Ampere+ (no GradScaler needed, fp32-like dynamic range);
    # "fp16" is the fallback for older cards and runs under a GradScaler.
    # "off" (default) trains in fp32 exactly as before. CUDA only.
    autocast_dtype: Literal["off", "bf16", "fp16"] = "off"

    # TF32 matmul/conv (torch.set_float32_matmul_precision("high") +
    # cudnn.allow_tf32). Free ~2x matmul throughput on Ampere+ at negligible
    # precision cost for this workload. CUDA only.
    tf32: bool = False

    # cudnn autotuner. Safe and profitable here: conv shapes are fixed
    # (44×14×14 boards, fixed batch size). CUDA only.
    cudnn_benchmark: bool = False

    # channels_last memory format for the conv net + training batches —
    # enables tensor-core-friendly NHWC kernels. CUDA only.
    channels_last: bool = False

    # torch.compile on the net. Guarded: compile failure logs a warning and
    # falls back to eager, and checkpoints are always saved from the original
    # (uncompiled) module so they stay interchangeable.
    compile: bool = False

    # DataLoader parallelism. 0 (default) loads in-process exactly as before.
    # >0 moves the per-item work — densifying 17,837-length policies and encoding
    # compact boards to (44, 14, 14) planes — into worker processes, which is
    # what keeps a fast GPU fed. Portable (CPU or CUDA).
    dataloader_workers: int = 0
    pin_memory: bool = False  # page-locked host buffers (enables true async H2D copies)
    persistent_workers: bool = False  # keep workers alive across epochs (skip respawn cost)
    prefetch_factor: int = 2  # batches each worker keeps ready (used only when workers > 0)

    # multiprocessing start method for the DataLoader's worker processes (used
    # only when ``dataloader_workers > 0``). The default **fork** deadlocks here:
    # self-play (JAX) and training (torch) share one process, so JAX's threads
    # are live when the loader forks workers — that is what killed the
    # pin-memory thread at gen 59 of blokus_cloud_60. "forkserver" (default)
    # forks workers from a clean helper process that never loaded JAX; "spawn"
    # cold-starts a fresh interpreter (heavier, always available); "fork"
    # restores the old behaviour. An unavailable method falls back to "spawn".
    # Inert when ``dataloader_workers == 0`` (the Mac/CPU default), so default
    # behaviour there is unchanged. See docs/plans/archive/harden-long-runs.md H1.
    dataloader_context: Literal["forkserver", "spawn", "fork"] = "forkserver"

    # Per-batch metric cadence. 1 (default) = today's behaviour: a CUDA sync
    # (.item()) and a metrics row every batch. N>1 accumulates losses on-device
    # and syncs/logs once every N batches (the logged row carries the mean of
    # the window), keeping the hot loop free of forced syncs.
    log_every_batches: int = 1


@dataclass(frozen=True)
class TournamentConfig:
    """Knobs for the post-hoc pool BayesElo tournament (``scripts/tournament_elo.py``).

    The tournament plays a *sparse but connected* round-robin among a finished
    run's saved checkpoints and fits one consistent Elo per checkpoint, giving
    the rising strength curve the frozen-baseline metric can't (it saturates —
    see ``evaluation/rating.py``). Nothing here touches the training loop; it's
    read only by the standalone tournament tool. Plan:
    docs/plans/archive/pool-based-elo.md.
    """

    # Games each checkpoint pair plays. Arena rounds this down to even and swaps
    # colours at halftime, so >= 2. More games = tighter ratings, more compute.
    games_per_pairing: int = 30

    # Each checkpoint plays the checkpoints these many generations behind it.
    # Exponential spacing keeps the comparison graph connected at O(K·log K)
    # pairs instead of a full O(K²) round-robin (60 gens → ~300 pairs, not 1770).
    back_ref_offsets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    # Always also pair every checkpoint with gen-0 and the final generation.
    # Guarantees connectivity and ties the whole field to the shared anchor.
    include_first_last: bool = True

    # BayesElo regularisation: virtual draws vs a fixed R=0 anchor. Keeps an
    # undefeated / winless checkpoint's rating finite.
    prior_games: float = 2.0

    # Elo assigned to the gen-0 anchor checkpoint after fitting (display gauge).
    anchor_rating: float = 0.0

    # Subsample the checkpoint list to cap cost (e.g. take every ⌈K/max⌉-th
    # generation). None = use every saved checkpoint.
    max_checkpoints: int | None = None

    # MCTS simulations per move for the tournament games. Deliberately explicit
    # and low (32) rather than inherited from the heavy training ``mcts_config``:
    # ranking is robust to weak play, so this keeps a full end-of-run tournament
    # to ~30–60 min instead of hours. See ``pool-elo-methodology.md``.
    num_mcts_sims: int = 32

    # Run the pool tournament automatically at end-of-run (normal completion),
    # so the report includes the rigorous pool-Elo curve without a manual step.
    # Default False preserves current behaviour (run it by hand via
    # ``scripts/tournament_elo.py``); enable in cloud/production configs.
    run_at_end: bool = False


@dataclass(frozen=True)
class NetConfig:
    """Configuration parameters for the neural network.

    Controls both the architecture (a residual network with convolutional
    layers for board-state processing) and its training process.
    """

    learning_rate: float  # Learning rate for the optimizer
    dropout: float  # Dropout probability for regularisation (0 to 1)
    epochs: int  # Number of full passes over the replay buffer per generation
    batch_size: int  # Number of positions per training batch
    cuda: bool  # Whether to use CUDA for GPU acceleration
    num_filters: int  # Number of convolutional filters per layer (power of 2)
    num_residual_blocks: int  # Number of residual blocks in the network
    # LR schedule: None / "constant" = constant at ``learning_rate``,
    # "cosine" = CosineAnnealingLR (floored by ``lr_eta_min``),
    # "step" = MultiStepLR (see ``lr_milestones`` / ``lr_gamma``).
    lr_scheduler: str | None = None

    # Floor the cosine schedule at this learning rate (``CosineAnnealingLR``'s
    # ``eta_min``). 0.0 (default) preserves the original behaviour exactly —
    # the schedule anneals all the way to ~0, which strangled the last quarter
    # of ``blokus_cloud_60`` (LR ≤ 1e-4 from ~gen 48, 2.7e-6 by gen 58: late
    # arena rejections + value-loss rise while policy loss was still falling —
    # see docs/research/blokus-cloud-60-analysis.md §3). A non-zero floor (e.g.
    # 1e-4, 10% of a 1e-3 peak) keeps the optimiser moving to the run's end.
    # Only consulted when ``lr_scheduler == "cosine"``.
    lr_eta_min: float = 0.0

    # Milestones for the ``"step"`` scheduler (MultiStepLR), in **generations**:
    # at each listed generation the LR is multiplied by ``lr_gamma``. Converted
    # to scheduler steps via ``epochs`` (mirroring the cosine ``T_max``
    # convention). Empty () is invalid for ``"step"`` (raises). Ignored by every
    # other scheduler. e.g. ``[20]`` with ``lr_gamma=0.3`` steps 1e-3 → 3e-4 at
    # generation 20.
    lr_milestones: tuple[int, ...] = ()

    # Decay factor applied at each ``lr_milestones`` entry for the ``"step"``
    # scheduler. Only consulted when ``lr_scheduler == "step"``.
    lr_gamma: float = 0.1

    # Half-precision (fp16) inference. When True AND running on CUDA, the
    # forward pass in predict/predict_batch runs under torch.autocast(fp16) —
    # faster on GPUs with Tensor Cores (e.g. the 3060 Ti), inference-only so no
    # gradient-stability concerns. No effect on CPU. Default False; outputs are
    # cast back to float32 so downstream code is unaffected either way.
    fp16_inference: bool = False

    # Policy head architecture. "fc" = the original fully-connected policy head
    # (a single Linear(2·cells → action_size), ~95% of the net's params);
    # "conv" = fully-convolutional head (1×1 conv to per-orientation logit planes
    # + a small pass head), ~1200× fewer params in that layer and a stronger
    # board-game inductive bias. The two heads have incompatible state_dicts
    # (loading across them raises). Blokus only — TicTacToe ignores this.
    # Default "conv" (since 2026-06-02): correctness proven (read-out matches
    # ActionCodec), trains cleanly, ~21× fewer params / ~19× smaller checkpoints.
    # Set "fc" to restore the original fully-connected head (e.g. to load an
    # old FC checkpoint — the two head state_dicts are incompatible).
    policy_head: Literal["fc", "conv"] = "conv"

    # Opt-in training-loop performance knobs (autocast, TF32, channels_last,
    # torch.compile, DataLoader workers, metric-sync cadence). Every field
    # defaults to "off" = current behaviour; see ``TrainingPerfConfig``.
    perf: TrainingPerfConfig = field(default_factory=TrainingPerfConfig)

    # Record of the ``NET_PRESETS`` name this config was built from, if any.
    # Resolution happens in ``load_args`` (the preset fills
    # ``num_filters``/``num_residual_blocks`` unless the JSON sets them
    # explicitly); this field is informational so run artefacts show intent.
    preset: str | None = None


@dataclass(frozen=True)
class ObjectStoreConfig:
    """S3-compatible object storage for run artefacts (opt-in).

    When present on ``RunConfig``, checkpoints/metrics/reports sync to the
    bucket after every completed generation and ``--resume`` can rebuild the
    local run directory from it — so a terminated cloud instance loses at most
    its in-flight generation. Absent (the default) means pure local-FS
    behaviour. Works against any S3-compatible endpoint (AWS S3, Cloudflare
    R2, MinIO, a neocloud's store); credentials come from the standard
    ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` env chain, never from run
    JSON. Requires the ``s3`` extra (``uv sync --extra s3``).
    """

    bucket: str  # bucket name
    prefix: str | None = None  # key prefix; None = mirror the local layout (runs/<group>/<run_name>)
    endpoint_url: str | None = None  # None = AWS S3; else any S3-compatible endpoint URL
    region: str | None = None  # region name, where the endpoint needs one


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
    """Configuration parameters for a complete training run.

    Holds all parameters for a training session — self-play generation,
    neural network training, model evaluation, and data storage/logging.

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
    num_eps: int  # Number of complete self-play games per generation (fresh games F)
    temp_threshold: int  # Move number after which temperature is set to ~0
    update_threshold: float  # Win rate required for new network to be accepted (0 to 1)
    # Number of evaluation games between old/new networks. Doubles as the
    # rolling arena-derived Elo sample size (that metric reuses these games), so
    # very low values make the Elo noisier — 100 is comfortable, ≤40 is coarse.
    num_arena_matches: int

    # Model and file management
    root_directory: Path  # Root directory for all output files
    load_model: bool  # Whether to load a pre-existing model

    # Component configurations
    mcts_config: MCTSConfig  # Monte Carlo Tree Search parameters
    net_config: NetConfig  # Neural network parameters

    # Rolling replay buffer (replaces the old generation-window machinery).
    #
    # The buffer holds the last ``replay_buffer_games`` *games* worth of
    # positions; oldest games auto-evict. This is the staleness knob **B** — the
    # oldest game is ``B / num_eps`` net-versions old — independent of how fast
    # the buffer turns over (``num_eps``, the fresh games F). Sizing in games
    # (not generations or positions) is invariant to ``num_eps`` and to
    # positions-per-game variance, matching how AlphaZero/MuZero/KataGo describe
    # buffers. Default 5000 reproduces run2's healthy regime (≈5 gens × 1000 eps).
    #
    # Training uses **all** the data: ``net_config.epochs`` full passes over the
    # whole buffer each generation. Reuse is therefore the *emergent* quantity
    # ``epochs × (B / num_eps)`` — logged, not a knob. e.g. ``B=5000``,
    # ``num_eps=1000``, ``epochs=1`` ⇒ reuse 5 (run2's regime).
    replay_buffer_games: int = 5000

    # Which engine generates self-play games. ``"python"`` is the original
    # CPU-worker path (serial or ``num_parallel_workers``-way parallel);
    # ``"jax"`` is the GPU-native batched pipeline (games/blokusduo/jax +
    # games/blokusduo/jax — Blokus only, requires the ``jax``/``jax-cuda`` extra).
    # Arena/Elo/Pentobi evaluation always uses the python path regardless.
    # Plan: docs/plans/archive/jax-selfplay-pipeline.md.
    selfplay_backend: Literal["python", "jax"] = "python"

    # Execution knobs for the jax backend; ignored by the python backend.
    jax_selfplay: JaxSelfPlayConfig = field(default_factory=JaxSelfPlayConfig)

    # Post-hoc pool BayesElo tournament knobs; read only by
    # ``scripts/tournament_elo.py``, never by the training loop.
    tournament: TournamentConfig = field(default_factory=TournamentConfig)

    # Optional reporting backends
    wandb: WandbConfig | None = None  # If set, mirror metrics to Weights & Biases

    # Optional S3-compatible artefact sync + remote resume; None (default) =
    # local filesystem only. See ``ObjectStoreConfig``.
    object_store: ObjectStoreConfig | None = None

    # Starting anchor for the rolling arena-derived Elo curve: the run's
    # gen-0 net is pinned here and every later generation's Elo is chained off
    # it (see ``docs/plans/archive/arena-derived-elo.md``). Also the display
    # anchor for the gen-0 checkpoint in the pool tournament. Display-only — the
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

    # Opening-diversification for arena/Elo eval games. ``arena_opening_temp``
    # is the play temperature applied to the first ``arena_opening_moves`` of a
    # player's own plies (then it reverts to deterministic argmax); it samples
    # from the MCTS visit distribution, so it picks among moves search already
    # rated well rather than random blunders. Both default to 0 = today's exact
    # behaviour (fully deterministic per (seed, colour)). >0 injects opening
    # diversity so near-equal nets don't split *exactly* 50/50 by colour — the
    # v3 gate-resolution problem (14/19 arena rejections scored exactly 50-50,
    # see docs/research/xl-training-scaleup.md addendum). Applied symmetrically
    # to *both* arena players (plan S1 option 1 — diversify the gate too), so it
    # is fair. Production candidate: ~1.0 for ~6 plies, but that flip is gated on
    # the S3 validation control (docs/plans/p0-instrument-and-dataloader.md);
    # keep at 0 until S3 passes.
    arena_opening_temp: float = 0.0
    arena_opening_moves: int = 0

    # TTT-specific: games per generation to play vs a perfect-play minimax
    # opponent. Only used when ``game == "tictactoe"``. 0 disables.
    minimax_games_per_gen: int = 20

    # Per-generation symmetry diagnostic: i.e. the number of randomly-
    # generated reference board positions on which we test the network's
    # asymmetric-preference (whether mirroring the board flips its raw
    # policy in the equivalent way). 0 disables the diagnostic. Same
    # seeded set of positions is used every generation so the metric is
    # cross-gen comparable. See ``evaluation/symmetry.py`` for the
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

    # Number of worker processes used for the self-play / arena / Elo phases.
    # ``1`` (default) keeps single-process behaviour bit-for-bit identical —
    # the parallel codepath is not taken at all. Values > 1 spawn a
    # ``ProcessPoolExecutor`` with that many workers; each holds its own
    # copy of the network in its own CUDA context. Determinism is
    # preserved per-episode via a seed derived from
    # ``(seed, generation, episode_idx)``, so set membership of training
    # examples matches the serial path regardless of worker count.
    num_parallel_workers: int = 1

    # Device for the game-playing pool workers (self-play / arena / Elo).
    # Default False = CPU-only workers: they skip the CUDA context entirely
    # (~0.5 GB each instead of ~2.5 GB), so we can run ~one-per-core without the
    # per-worker GPU-stack duplication that caps worker count and OOM'd a run.
    # The main process keeps ``net_config.cuda`` for the training step. Set True
    # to put the workers' net on the GPU (the pre-S1 behaviour / benchmark
    # baseline). Independent of ``inference_server``: that routes inference to a
    # central GPU process instead (a separate way to keep workers light) — the
    # two are alternative inference strategies, both supported. See
    # ``docs/plans/archive/lean-self-play-workers.md``.
    worker_cuda: bool = False

    # multiprocessing start method for the worker pool. "auto" (default) =
    # ``forkserver`` on Linux/WSL, ``spawn`` elsewhere (e.g. the macOS test box).
    # ``spawn`` cold-starts a fresh interpreter per worker — N simultaneous torch
    # re-imports (~629 MB each) burst the host's resources and wedge WSL at ~16
    # workers. ``forkserver`` imports torch once in a warm helper and forks
    # workers from it (no re-import burst, copy-on-write shared pages), which both
    # lifts the worker ceiling and cuts memory. Explicit "spawn"/"forkserver"/
    # "fork" override the auto choice (e.g. for tests). See
    # ``docs/plans/archive/lean-self-play-workers.md``.
    worker_start_method: str = "auto"

    # Precomputed-move-list move generator: if True, BlokusDuoGame routes
    # ``valid_move_masking`` through the implementation in
    # :mod:`games.blokusduo.movegen_runtime`. Default False to preserve
    # the existing array-based path. Produces bit-identical training
    # trajectories at the same seed (verified by
    # ``tests/games/blokusduo/test_movegen_determinism.py``). Only
    # ``BlokusDuoGame`` consults this flag; TTT ignores it.
    use_optimised_movegen: bool = False

    # Cross-worker inference server: if True, a single server process owns
    # the GPU net and batches MCTS leaf evaluations across *all* workers into
    # large forward passes (vs per-worker batching), recovering GPU
    # under-utilisation. Default False keeps the existing per-worker path
    # bit-for-bit identical. Only takes effect when ``num_parallel_workers > 1``
    # and ``net_config.cuda`` is True. Results are unaffected (inference is
    # per-row independent in eval mode) — verified by the server-on vs
    # server-off equivalence tests.
    inference_server: bool = False
    # Max positions per server GPU batch. 0 = auto (num_parallel_workers ×
    # mcts_config.mcts_batch_size, i.e. every worker's full leaf batch at once).
    server_max_batch: int = 0
    # Max time the server's first queued request waits before flushing an
    # under-full batch. The size-OR-timeout rule self-corrects, so this is a
    # safe backstop rather than a tuned knife-edge.
    server_max_wait_ms: float = 5.0

    @property
    def run_directory(self) -> Path:
        """Base directory for all files related to this training run.

        Runs are grouped by game under ``<root>/runs/<group>/`` so the output
        root (e.g. ``temp/``) stays organised instead of becoming a flat dump
        of every run — a TicTacToe run lands in ``<root>/runs/tictactoe/<name>``,
        a Blokus run in ``<root>/runs/blokus/<name>``. Every other directory
        property below hangs off this, so the whole run tree moves with it.
        Unknown games fall back to ``other``.
        """
        group = _GAME_GROUPS.get(self.game, "other")
        return self.root_directory / "runs" / group / self.run_name

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
    def learning_rate_directory(self) -> Path:
        """Directory for the per-generation optimizer learning rate.

        Records ``optimizer.param_groups[0]["lr"]`` — the LR the epoch actually
        trained at — so any LR-schedule experiment is reviewable. Absent for
        runs predating LR logging, in which case the report omits the section.
        """
        return self.run_directory / "LearningRate"

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
    def rolling_elo_directory(self) -> Path:
        """Directory for the per-generation rolling arena-derived Elo.

        Holds the non-saturating live strength metric (candidate rated against
        the current arena incumbent, benchmark rolled forward on acceptance).
        Written by :meth:`MetricsCollector.log_rolling_elo`; rendered as the
        report's rolling-Elo curve. Absent for runs predating this metric, in
        which case the report omits the section. See
        ``docs/plans/archive/arena-derived-elo.md``.
        """
        return self.run_directory / "RollingElo"

    @property
    def tournament_directory(self) -> Path:
        """Directory for post-hoc pool BayesElo tournament results.

        Written by ``scripts/tournament_elo.py`` (ratings parquet + raw W/L/D
        JSON); rendered as the report's pool-Elo curve. Absent for older runs.
        """
        return self.run_directory / "Tournament"

    @property
    def minimax_results_directory(self) -> Path:
        """Directory for per-generation results vs a perfect-play minimax opponent (TTT only)."""
        return self.run_directory / "MinimaxResults"

    @property
    def arena_replays_directory(self) -> Path:
        """Directory for recorded arena games (move sequences + top-K policies per move)."""
        return self.run_directory / "ArenaReplays"

    @property
    def pentobi_ladder_directory(self) -> Path:
        """Directory for Pentobi ladder benchmark results (JSON per benchmark run).

        Written by ``scripts/pentobi_benchmark.py``; rendered as the report's
        "Pentobi Ladder" section. Empty/absent = section omitted.
        """
        return self.run_directory / "PentobiLadder"

    @property
    def symmetry_diagnostic_directory(self) -> Path:
        """Directory for per-generation policy-symmetry diagnostic results.

        Stores KL divergences between the network's raw policy on reference
        positions and the same policy reconstructed from the symmetric
        variant via ``game.get_symmetries``. See
        ``evaluation/symmetry.py``.
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

    if "elo_games_per_gen" in args_json:
        logger.warning(
            "Config {} sets 'elo_games_per_gen', which was retired with the "
            "per-generation gen-0 Elo eval (replaced by the rolling arena-derived "
            "Elo — docs/plans/archive/arena-derived-elo.md). The key is ignored.",
            config_path,
        )
        args_json.pop("elo_games_per_gen")

    _resolve_net_preset(args_json)
    return fromdict(RunConfig, args_json)


def _resolve_net_preset(args_json: dict) -> None:
    """Fill ``net_config`` size fields from a named preset, in place.

    Explicit ``num_filters``/``num_residual_blocks`` keys in the JSON win over
    the preset's values, so a preset is a starting point, not a straitjacket.
    """
    net_json = args_json.get("net_config")
    if not isinstance(net_json, dict):
        return
    preset_name = net_json.get("preset")
    if preset_name is None:
        return
    if preset_name not in NET_PRESETS:
        raise ValueError(f"Unknown net preset {preset_name!r}. Expected one of {sorted(NET_PRESETS)}.")
    for key, value in NET_PRESETS[preset_name].items():
        net_json.setdefault(key, value)
