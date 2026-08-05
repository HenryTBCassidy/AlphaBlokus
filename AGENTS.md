# AlphaBlokus — AGENTS.md

> Canonical agent instructions. `CLAUDE.md` is a symlink to this file — edit here only.

## What is this project?

AlphaZero implementation for Blokus Duo. Self-play reinforcement learning on a 14x14 board with 21 polyomino pieces per player. The goal is to beat Pentobi (strongest open-source Blokus AI) at its maximum difficulty, level 9. Current best net holds a winning record through level 4 (2026-07 ladder: 80/75/60/55/45/20% at levels 1–6, 40 games/level).

**Current state:** The full pipeline is built, optimised, and running scaled Blokus training on the home GPU (native Ubuntu). The code is an installable `src/alphablokus` package with CI (ruff + format check, strict mypy, base + jax test jobs). Scaled runs (15–30+ generations, ~1,000 games/gen) have executed; the Pentobi benchmark harness is built (`games/blokusduo/pentobi/`, `scripts/pentobi_benchmark.py`). The phase-by-phase history lives in `docs/plans/archive/` (the README links the highlights; regenerate its ladder chart via `scripts/plot_pentobi_ladder.py`).

**JAX self-play backend (2026-07):** Blokus self-play generation can run GPU-native — rules as int8 matmuls, mctx search over a top-K compact action space, inference-only jnp net bridged from the torch checkpoint each generation (`games/blokusduo/jax/`). Selected per run via `RunConfig.selfplay_backend: "python" | "jax"`, reached through `registry.resolve_jax_selfplay_backend`; production Blokus configs use jax + Gumbel search (`search_policy: "gumbel"`, n=64 — validated at strength parity and ~3.5–12× wall-clock in `docs/research/jax-pipeline-ab.md`). The python path is unchanged, remains the dataclass default, and always drives arena/Elo/Pentobi evaluation. Requires `uv sync --extra jax` (Mac CPU) or `--extra jax-cuda` (box).

## Current focus

In-flight work lives in `docs/plans/` (top-level = in-flight, `archive/` = done). In flight: [`pentobi-distillation.md`](docs/plans/pentobi-distillation.md) — the curriculum fix chosen by the capacity probe (plateau is not net size): a diverse Pentobi **L9** expert corpus (generator + validated pilot landed; **deterministic stratified opening keys**, schema in `docs/07-DATA-STORAGE.md`), then SL distillation, then RL beyond the teacher. Next step: run D5 stage-1 generation (~13k L9 games) on the box. Recently landed: [`post-regression-recovery.md`](docs/plans/archive/post-regression-recovery.md) (2026-07-23) — AdamW weight-decay default-on + external keep-best-by-ladder + drift circuit-breaker; its box capacity probe returned a tie (`xl` no better than `large`), which chose distillation over an `xl` run; [`arena-derived-elo.md`](docs/plans/archive/arena-derived-elo.md) (2026-07-06) — replaced the saturating per-gen "Elo vs frozen gen-0" eval with a rolling arena-derived Elo (reuses arena games, non-saturating) + auto-runs the pooled BayesElo tournament at end-of-run (`tournament.run_at_end`); [`pool-based-elo.md`](docs/plans/archive/pool-based-elo.md) (2026-07-05) — post-hoc pool BayesElo tournament (`scripts/tournament_elo.py`) that rates a run's checkpoints against each other into a non-saturating strength curve (E8/E9 deferred; methodology in [`docs/research/pool-elo-methodology.md`](docs/research/pool-elo-methodology.md)); [`cloud-scale-training.md`](docs/plans/archive/cloud-scale-training.md) (PR #34) — single rented-GPU runs at a ~£100 budget (uv CUDA container, S3-compatible sync + remote `--resume`, opt-in training-perf knobs, net-size presets, cost-calibration tool, Pentobi ladder in the report; recommendation in [`docs/research/cloud-training-recommendation.md`](docs/research/cloud-training-recommendation.md)); [`oom-hardening.md`](docs/plans/archive/oom-hardening.md) (2026-07-04) — RAM verification at scale is pending the next box run. Candidate ideas queue in `docs/IDEAS.md`.

## Commands

```bash
uv sync --extra dev                              # Install dependencies + tooling (pytest, ruff, mypy)
uv sync --extra jax                              # + JAX self-play backend (CPU; use --extra jax-cuda on the box)
uv run pytest                                    # Run tests
uv run pytest -m "not slow"                      # Skip integration tests
uv run pytest tests/games/blokusduo/             # Blokus tests only
uv run pytest tests/search/ tests/training/      # Framework subsets (tests mirror src/alphablokus/)
uv run alphablokus --config run_configurations/test_run.json   # Run training from some configuration
uv run alphablokus --config <cfg> --report-only  # Re-render the HTML report, no training
uv run alphablokus --config <cfg> --resume       # Continue a crashed/stopped run
uv run ruff check . && uv run ruff format --check src tests scripts   # Lint (matches CI)
uv run mypy                                      # Typecheck (strict: disallow_untyped_defs)
```

Scripts run as modules with the package installed — no `PYTHONPATH` needed: `uv run python scripts/pentobi_benchmark.py`, `uv run python scripts/benchmarks/benchmark.py`.

## Architecture

Game-agnostic framework (`src/alphablokus/`) with pluggable game implementations (`src/alphablokus/games/`).

**Package map** (each subpackage is a pipeline phase or shared machinery):

- `cli.py` / `config.py` / `interfaces.py` / `registry.py` — entry point, frozen-dataclass config, protocols, composition root
- `search/` — MCTS (PUCT, Dirichlet noise, batched inference + virtual loss) + its stats dataclasses
- `selfplay/` — the episode loop (`episode.py`) + backend dispatch serial/parallel/jax (`generate.py`)
- `parallel/` — worker pool (self-play/arena/Elo) + the (unadopted-by-default) inference server
- `training/` — `Coach` generation loop, `ReplayBuffer`, eval set, memory diagnostics
- `evaluation/` — arena, players, acceptance rule, Elo, symmetry diagnostic
- `storage/` — `MetricsCollector` (hive parquet + W&B mirror), `SelfPlayStore`, sparse policy codec
- `games/` — `tictactoe/` (reference impl + minimax oracle), `blokusduo/` (board/codec/game/pieces + `movegen/`, `pentobi/`, `jax/`, `nn/`)
- `reporting/` — self-contained HTML run report (JSON payload + hand-rolled SVG charts, arena replay browser; no CDN/build step)
- `testing/` — shipped test utilities (position caches)

**Core protocols** (in `alphablokus/interfaces.py`):
- `IBoard` — immutable state snapshot: geometry, encoding (`as_multi_channel`), `state_key`, `canonical`
- `IGame[TBoard]` — rules engine + action space: legal moves, game-over detection, symmetries, board factory (generic over its board type)
- `IPolicyValuePredictor` — the inference surface MCTS actually needs (`predict`, `predict_batch`)
- `INeuralNetWrapper` — extends it with `train`, `save_checkpoint`, `load_checkpoint`
- `IOracle` — perfect-play evaluation hooks (TicTacToe minimax)

**The registry rule:** `registry.py` is the **only** framework module allowed to name concrete game classes or backends (`instantiate_game`, `instantiate_game_and_network`, `resolve_jax_selfplay_backend`, `resolve_oracle`). Everything else depends on the protocols. Adding a game = implement the protocols under `games/<name>/` + register it there; never import `games.*` from framework code.

**Two game implementations:**
- `games/tictactoe/` — complete reference implementation. 2-channel encoding (2x3x3), 10 actions.
- `games/blokusduo/` — the target game, complete. 44-channel encoding (44x14x14), 17,837 actions.

**Key design decisions:**
- Boards are immutable (use `with_piece()`/`with_move()` to get new board)
- 44-channel encoding: 21 binary planes per player (one per piece type showing where it sits) + 2 aggregate planes. Piece inventory is implicit (all-zero plane = unplayed).
- Canonical form via channel reordering in `as_multi_channel(current_player)` — current player's planes always first
- `state_key` = `_piece_placement_board.tobytes()` (196 bytes, signed int8, +piece_id=White, -piece_id=Black)
- Action space: 14x14 grid positions x 91 piece-orientations + 1 pass = 17,837

## Conventions

Follow `docs/guides/STYLE-GUIDE.md` for all code (it also documents the project layout and tooling contract). Key points:
- Full type annotations on every function signature — machine-enforced (`mypy` with `disallow_untyped_defs`, run in CI)
- `ruff` lint + `ruff format` (CI checks both; line length 120, Python 3.11+)
- Google-style docstrings on public classes/methods
- `loguru` for logging (no `print()`), `{}` placeholders
- Frozen dataclasses for config/DTOs
- Protocol interfaces with explicit subclassing
- `from __future__ import annotations` instead of quoted type refs
- No mocks for game logic tests — use real objects
- `time.perf_counter()` for timing, `pathlib.Path` for filesystem

Follow `docs/guides/PLAN-FORMAT.md` when creating implementation plans.

## Gotchas

1. **Move generation is done; don't rewrite it.** Algorithm is documented inline and in `docs/plans/archive/blokus-valid-move-algorithm.md`. Further speedups (Cython, bitboard) were considered and deliberately set aside — candidate work goes through `docs/IDEAS.md`.
2. **Action space is huge (17,837).** MCTS iterates only valid moves (`np.where(valids)[0]`).
3. **Orientation IDs are 0-based (0–90).** `OrientationCodec` in `games/blokusduo/pieces.py` handles `(piece_id, orientation) ↔ int`. `ActionCodec` in `games/blokusduo/codec.py` handles the full `Action ↔ int` (0–17,836) mapping.
4. **Coordinate systems:** Board = bottom-left origin (Blokus notation). Arrays = top-left origin (numpy). `CoordinateIndexDecoder` handles conversion.
5. **Board sizes use class constants.** `BlokusDuoBoard.N = 14`, `Board.N = 3` (TicTacToe). Never hardcode board dimensions as literals.
6. **The jax backend is Blokus-only and inference-only.** `selfplay_backend: "jax"` raises for TicTacToe; training stays in torch (weights bridged per generation). Don't compare internal Elo curves across runs — each is anchored to its own gen-0 net (see `docs/research/jax-pipeline-ab.md` §3.2).
7. **Device selection is a simple `cuda: bool` flag** in `RunConfig.net_config` (`alphablokus/config.py` `NetConfig.cuda`, used in `games/base_wrapper.py`). No MPS auto-detection. On the Mac always set `cuda: false`; on the home PC set `cuda: true`.
8. **`pieces.json` has one accessor.** Resolve it via `alphablokus.games.blokusduo.pieces.default_pieces_path()` — never a repo- or CWD-relative path.
9. **Elo is now a two-tier scheme (2026-07-06).** The live per-generation metric is the **rolling arena-derived Elo** (`Coach._record_rolling_elo` → `RollingElo/`): candidate rated against the current arena incumbent, benchmark rolled forward on acceptance, non-saturating, zero extra games. The old per-gen "Elo vs frozen gen-0" eval and `elo_games_per_gen` are **gone** — but `Nets/elo_baseline.pth.tar` is still saved at gen 0 (unconditionally) purely as the pooled-tournament anchor. The rigorous curve is the end-of-run pooled BayesElo tournament (`evaluation/tournament_run.py`, auto-run via `tournament.run_at_end`). Cross-run splicing goes through `Nets/elo_anchor.json`. Plan: `docs/plans/archive/arena-derived-elo.md`.
10. **The arena gate can be colour-pinned; two config knobs fix it (2026-07-16).** In Blokus Duo ~96% of decisive deterministic games are won by White, so an unpaired arena between near-equal nets is pinned to ~0.50 and a 0.55 threshold is unreachable (this froze `blokus_search_harder` at 0/17). `paired_arena: true` plays `num_arena_matches/2` colour-swapped **pairs** that share one sampled opening prefix (`Arena.play_games_paired`), cancelling first-mover advantage; `gate_mode: "threshold" | "regression_guard" | "always"` (in `evaluation/acceptance.py::is_accepted`) replaces the improvement-filter with a regression-guard (`guard_floor`, default 0.48). **Both default to today's behaviour** (`paired_arena=False`, `gate_mode="threshold"`) — only the rerun config opts in. White/black-win split is now logged per gen in `ArenaData`, and the report red-flags exact-0.500 / sub-binomial scores. Plan: `docs/plans/fix-arena-colour-pinning.md`; root cause: `docs/research/plateau-investigation.md`.
11. **Weight decay is on by default for every run (2026-07-22).** The optimizer is `AdamW` with `NetConfig.weight_decay = 1e-4` (`games/base_wrapper.py`); this deliberately changes all existing configs' behaviour — training without it let `blokus_paired_gate_rerun` drift a converged net L4 → L3 (`docs/research/regression-and-next-steps.md` §1.3). There is exactly one parameter group (the optimizer is built from a bare `nnet.parameters()`), so the decay reaches every tensor including norms and biases. `load_checkpoint` re-asserts the configured decay so pre-change checkpoints can't silently disable it on `--resume`. Set `"weight_decay": 0.0` only to reproduce an old run bit-for-bit.
12. **A warm continuation may not start at the from-scratch peak LR (2026-08-04).** `load_model: true` builds a *fresh* optimizer and a *fresh* LR schedule (`cli.py`), so `learning_rate` is applied at full strength to an already-converged net — the setting behind every ≤0 continuation in the run ledger. `validate_active_path_knobs` refuses `load_model: true` with `learning_rate > 2.5e-4`; the long-run recipe is `lr_scheduler: "cosine"` with `learning_rate: 2.5e-4` and `lr_eta_min: 1e-4`. Set `net_config.allow_high_warm_start_lr: true` only for a run that is deliberately sweeping the rate (the `lr_ab_*` arms).
13. **Configs may not set knobs their own search path ignores (2026-08-04).** `dataclass_wizard` silently drops unknown JSON keys, so a stale knob is invisible rather than inert-but-visible. `config.validate_active_path_knobs` (called from `load_args`) raises when a **Gumbel** config sets `mcts_config.dirichlet_epsilon`, `mcts_config.dirichlet_alpha` or `temp_threshold` — all three are unreachable under Gumbel (`jax/search.py` takes `root_log_pi = log_pi`; `jax/actors.py` never builds the temperature branch). `temp_threshold` is now `int | None`, required only by the python/PUCT path, which reaches it via `RunConfig.sampling_temp_threshold`.
14. **A run refuses to start unless its config is committed (2026-08-04).** `provenance.check_config_is_committed` compares the config file against git HEAD and exits if it is modified or untracked; `--allow-uncommitted-config` overrides it and the override is recorded. Every launch also writes `<run>/run_provenance.json` (code commit + dirty flag, config-commit state, and a SHA-256 manifest of the config, donor checkpoint and eval set) alongside `config.resolved.json`. This exists because one run's committed config was edited five days *after* the run and now describes a run that never happened.
15. **The eval set is genuinely held out, refreshed, reproducible and game-clustered — all four matter (2026-08-04).** This one instrument is what every internal health signal is read through, and it had four independent defects. (a) **Held out:** positions are sampled at *game* granularity and their source games are withheld from `ReplayBuffer.flat_examples` via content fingerprints (`EvalSet.source_fingerprints`, persisted so exclusion survives `--resume`). Previously the set was sampled from the training buffer and then trained on for `replay_buffer_games / num_eps` generations at `epochs` passes each, so every "held-out" per-epoch diagnostic was in-sample early in a run and silently changed meaning as those positions aged out — that is the direct explanation for a run reporting eval top-1 ~0.99 while its real strength fell. **`_ensure_eval_set` must run before the buffer is flattened.** An on-disk eval set with no fingerprints is rebuilt, never reused. (b) **Refreshed** every `eval_set_rebuild_every` generations (0 = the old build-once behaviour), with each metric carrying its vintage, because different vintages measure different positions. (c) **Reproducible:** the unseeded `random.shuffle` in the replay buffer is gone (the Coach seeds numpy and torch, never Python's `random`), so a fixed seed now yields a fixed eval set. (d) **Game-clustered:** `alphablokus.bootstrap.game_cluster_bootstrap` over `EvalSet.source_game_ids` is the only sanctioned way to put an interval on an eval-set statistic — position-level intervals are about `sqrt(positions per game)` too narrow (verified in `tests/test_bootstrap.py`: cluster coverage 0.94 vs nominal 0.95, position-level 0.52). `MAX_EVAL_POSITIONS_PER_GAME = 2` spreads a set over as many lineages as possible, since interval width depends on the number of games, not positions.
17. **Never write `config.seed or 0` (2026-08-04).** It makes an explicit seed of 0 indistinguishable from unseeded, which silently collapses two arms of a multi-seed sweep into one. Use `0 if config.seed is None else config.seed`.
16. **The arena is telemetry, not a promotion signal, once `gate_mode: "always"` is set.** Promotion is `select_best` over Pentobi ladder results and the catastrophe stop is the drift circuit-breaker; both are armed by `ladder_check_every` (0 disables) and consume results produced out-of-process by `scripts/mini_ladder.py`. With the gate off, `best.pth.tar` is merely the latest candidate — the run's product is `Nets/best_by_ladder.json`. `arena_crash_floor` keeps the arena useful as a crash detector (a score far below the instrument's measured 0.485–0.530 floor means something broke, not that the candidate is slightly worse). Reference recipe: `run_configurations/blokus_pilot_b6.json`.

## Documentation

```
docs/
├── 01-BACKGROUND.md         # Why Blokus, competitive landscape, key decisions
├── 02-ALGORITHMS.md         # MCTS, self-play, arena, move generation, jax/Gumbel backend
├── 03-NEURAL-NETWORKS.md    # ResNet, board encoding, loss functions
├── 04-BLOKUS-DUO.md         # Rules, all 21 pieces, 91 orientations
├── 05-EVALUATION.md         # Metrics, Elo, Pentobi benchmarking
├── 06-INTERFACES.md         # Pentobi GTP adapter + translation layer (as built), UI scoping
├── 07-DATA-STORAGE.md       # Parquet format, metrics tables, checkpoints
├── 08-TRAINING-ESTIMATES.md # Pre-optimisation cost model (superseded; kept for methodology)
├── 09-COMPUTE-OPTIONS.md    # Local + cloud hardware, cost per run, phasing
├── IDEAS.md                 # Register of candidate avenues not yet committed (distinct from plans/)
├── guides/
│   ├── STYLE-GUIDE.md       # Code conventions + project layout (ALWAYS reference before writing code)
│   ├── PLAN-FORMAT.md       # How to write implementation plans
│   ├── REMOTE-TRAINING.md   # Runbook for running training on the home box over SSH
│   └── AI-CONTEXT.md        # Extended context, architecture rationale, gotchas
├── research/                # Deep investigations (jax A/B, profiling, Pentobi internals, …)
└── plans/                   # Top-level = in-flight (none currently); archive/ = completed, kept for context
    └── archive/                       # ~40 completed plans — the project's full history, e.g.:
        ├── oom-hardening.md           #   Sparse on-disk policies + OOM guardrails (O1–O9)
        ├── refactor-repo-architecture.md #  src/ package restructure
        ├── full-cycle-optimisation.md #   Master optimisation tracker (F1–F5; ~14× vs serial)
        ├── jax-selfplay-pipeline.md   #   GPU-native self-play backend + Gumbel search
        ├── replay-buffer-refactor.md  #   Rolling game-sized buffer + compact board storage
        ├── resumable-runs.md          #   --resume from the last completed generation
        ├── pentobi-harness.md         #   GTP client + translation + benchmark runner
        ├── linux-migration.md         #   Box moved from Windows/WSL2 to native Ubuntu
        └── ...                        #   (move-gen, batched inference, conv head, reporting, …)
```

## Things NOT to do

- Don't rewrite the core framework (MCTS, Coach, Arena) — it's validated on TicTacToe
- Don't over-engineer move generation — correct first, fast second
- Don't import `games.*` from framework code — `registry.py` is the one composition root
- Don't add 4-player Blokus support yet — Duo must beat Pentobi level 9 first
- Don't switch architectures (no Transformer/ViT) — ResNet is proven for AlphaZero
- Don't pad options for learning — if one approach is clearly best, just recommend it
- Don't add unnecessary complexity, abstractions, or features that weren't asked for
