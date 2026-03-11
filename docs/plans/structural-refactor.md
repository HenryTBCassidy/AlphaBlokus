# AlphaBlokus — Structural Refactor

Everything in this doc is about **how the code is organised, not what it does**. The companion doc ([`bug-fixes.md`](bug-fixes.md)) covers correctness issues. The goal: clean up structure, naming, tooling, and infrastructure so that the codebase is consistent, testable, and easy to extend — then fix bugs on a solid foundation.

---

## Checklist

Items ordered to minimise merge pain. Each row ~ one commit. Section numbers below match these IDs.

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| S1 | Create `pyproject.toml` + uv setup | 30 min | High | ✅ |
| S2 | Reorganise into `games/` directory | 45 min | High | ✅ |
| S3 | Fix JSON config booleans | 10 min | Medium | ✅ |
| S4 | Extract BaseNNetWrapper | 1 hour | High | ✅ |
| S5 | Switch to loguru | 30 min | High | ✅ |
| S6 | Clean up / delete utils.py | 15 min | Medium | ✅ |
| S7 | Naming pass + import modernisation | 1.5 hours | Medium | ✅ |
| S8 | Build MetricsCollector | 1.5 hours | High | ✅ |
| S9 | Add CLI to main.py, fix module-level side effects | 30 min | Medium | ✅ |
| S10 | Data storage → parquet everywhere | 1 hour | Medium | ✅ |
| S11 | Set up pytest + initial test suite | 3–4 hours | High | |
| S12 | Profiling instrumentation (nice-to-have) | 2.5 hours | Low | |

**Estimated total (excl. nice-to-have): ~11 hours across 11 commits**

---

## S1. Package Management — Move to uv

### Current state

No `pyproject.toml`, no `requirements.txt`, no `environment.yml`. Dependencies are implicit (inferred from imports). There is no formal package management at all.

### Recommendation: uv

**Use uv.** It's fast, it's modern, and Henry prefers it.

The historical argument for conda was that ML packages (numpy, scipy) shipped with MKL-optimized BLAS through conda-forge, while pip gave you slower OpenBLAS builds. This is largely outdated:

- PyTorch pip wheels ship with CUDA support and optimized BLAS — no conda needed
- NumPy's pip wheels now default to OpenBLAS 0.3.x which is competitive with MKL for the array sizes we use (14×14 boards, not 10,000×10,000 matrices)
- The heavy lifting is on GPU (PyTorch), where the BLAS backend is irrelevant
- uv's dependency resolution is ~100× faster than conda's
- `pyproject.toml` is the Python packaging standard — better tooling support everywhere

**Action items:**

| Task | Notes |
|------|-------|
| Create `pyproject.toml` with project metadata and dependencies | Include `[project.optional-dependencies]` for `dev` (pytest, ruff, etc.) and `gpu` (CUDA) |
| Create `uv.lock` | Pin exact versions for reproducibility |
| Add `.python-version` file | Pin Python 3.11+ |
| Remove `nptyping` dependency | Replace with `numpy.typing` (see S7) |
| Add `ruff` for linting/formatting | Fast, opinionated, replaces flake8+black+isort |

**Estimated effort:** ~30 minutes

---

## S2. Project Structure — Introduce `games/` Directory

### Current layout

```
AlphaBlokus/
├── core/              ← framework (good)
├── blokusduo/         ← game implementation
├── tictactoe/         ← game implementation
├── run_configurations/
├── main.py
├── utils.py
├── reporting.py
└── ...
```

### Proposed layout

```
AlphaBlokus/
├── core/
│   ├── __init__.py
│   ├── interfaces.py      ← IGame, INeuralNetWrapper protocols
│   ├── config.py           ← RunConfig, MCTSConfig, NetConfig
│   ├── arena.py
│   ├── coach.py
│   ├── mcts.py
│   └── metrics.py          ← NEW: MetricsCollector (see S8)
├── games/
│   ├── __init__.py
│   ├── base_wrapper.py     ← NEW: shared NNetWrapper base (see S4)
│   ├── types.py            ← NEW: shared type aliases
│   ├── blokusduo/
│   │   ├── __init__.py
│   │   ├── game.py
│   │   ├── pieces.py
│   │   ├── pieces.json
│   │   └── neuralnets/
│   │       ├── __init__.py
│   │       ├── net.py
│   │       └── wrapper.py
│   └── tictactoe/
│       ├── __init__.py
│       ├── board.py
│       ├── game.py
│       └── neuralnets/
│           ├── __init__.py
│           ├── net.py
│           └── wrapper.py
├── run_configurations/
│   ├── full_run.json
│   └── test_run.json
├── tests/                   ← NEW (see S11)
│   ├── conftest.py
│   ├── test_core/
│   ├── test_blokusduo/
│   └── test_tictactoe/
├── notebooks/               ← move eval.ipynb here
│   └── eval.ipynb
├── main.py
├── utils.py                 ← stripped down (see S6)
├── reporting.py
├── pyproject.toml           ← NEW (see S1)
└── STYLE-GUIDE.md
```

### Key changes

1. **`games/` directory** — groups all game implementations under one folder. Cleaner on GitHub, makes the pattern obvious when adding future games (Connect4, Othello, etc.)
2. **`games/base_wrapper.py` + `games/types.py`** — shared base classes and type aliases used across games, at the `games/` root level rather than a separate subfolder.
3. **`tests/`** — mirrors the source structure
4. **`notebooks/`** — moves `eval.ipynb` out of the project root
5. **`core/metrics.py`** — new metrics collection module (see S8)

### What NOT to move

- `core/` stays at the top level — it's the framework, not a game
- `run_configurations/` stays — it's config, not code
- `main.py` stays at root — it's the entry point
- `reporting.py` stays — it's framework-level reporting

**Estimated effort:** ~45 minutes (mostly updating imports)

---

## S3. Fix JSON Config Booleans

### Current state

JSON config files store booleans as strings (`"False"` instead of `false`). `dataclass_wizard.fromdict()` handles native JSON booleans correctly, so these string values are fragile and unnecessary.

### Fix

Change `"False"` to `false` and `"True"` to `true` in all config files under `run_configurations/`.

**Estimated effort:** ~10 minutes

---

## S4. Extract BaseNNetWrapper

### Current state

`blokusduo/neuralnets/wrapper.py` and `tictactoe/neuralnets/wrapper.py` are ~85% identical. Methods that are **copy-pasted verbatim** between the two:

- `train()` (~50 lines)
- `predict()` (~20 lines)
- `loss_pi()` / `loss_v()` (~10 lines each)
- `_log_data()` (~15 lines)
- `collect_training_data()` (~20 lines)
- `save_checkpoint()` / `load_checkpoint()` (~10 lines each)
- `TrainingDataLoggable` dataclass (identical in both files)

### Fix

Extract a `BaseNNetWrapper` in `games/base_wrapper.py` that implements all shared methods. Game-specific wrappers only override what differs (network construction, board shape handling).

**Estimated effort:** ~1 hour

---

## S5. Switch to Loguru

### Current state

- Python stdlib `logging` with a JSON config file (`logging_config.json`)
- Centralised logger name `"Alpha"` via `LOGGER_NAME` constant
- `coloredlogs.install()` at module level in `main.py`
- Inconsistent usage: some files call `logging.info()` (root logger) instead of `log.info()` (named logger)
- Several `print()` calls that should be logging

### Recommendation: loguru for application logging

**Use loguru.** It eliminates all the boilerplate that makes stdlib logging painful:

```python
# Before (stdlib)
import logging
from core.config import LOGGER_NAME
log = logging.getLogger(LOGGER_NAME)
log.info(f"Starting generation {gen}")

# After (loguru)
from loguru import logger
logger.info(f"Starting generation {gen}")
```

No handler setup, no JSON config file, no `getLogger` boilerplate, coloured output by default, structured logging built in. Delete `logging_config.json` and `coloredlogs` dependency.

**Configuration in main.py:**

```python
from loguru import logger
import sys

# Remove default handler, add custom one
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(
    "runs/{run_name}/logs/alphablokus_{time}.log",
    rotation="10 MB",
    retention=3,
    level="DEBUG",
)
```

**Estimated effort:** ~30 minutes

---

## S6. Clean Up / Delete utils.py

### Current state

`utils.py` contains three unrelated things:
1. `AverageMeter` — training utility → move to `core/` or `games/`
2. `load_args()` — config loading → move to `core/config.py`
3. `setup_logging()` — logging setup → delete (loguru replaces this)

### Fix

Move the useful parts, delete the rest. After this, `utils.py` can probably be deleted entirely.

**Estimated effort:** ~15 minutes

---

## S7. Naming Pass + Import Modernisation

A comprehensive pass across the codebase to make naming consistent and modernise imports. This is the most diffuse change — touches almost every file.

### MCTS dictionary names

| Current | Meaning | Rename to |
|---------|---------|-----------|
| `Qsa` | Q-values per (state, action) | `q_values` |
| `Nsa` | Visit counts per (state, action) | `visit_counts` |
| `Ns` | Visit counts per state | `state_visits` |
| `Ps` | Policy priors per state | `policy_priors` |
| `Es` | End-state cache | `game_ended_cache` |
| `Vs` | Valid moves mask | `valid_moves_cache` |

`Vs` is particularly bad — `V` universally means "value" in RL. Using it for "valid moves" will confuse anyone reading the code.

**Note:** the single-letter math notation (`Q(s,a)`, `N(s,a)`, etc.) is conventional in RL literature. There's an argument for keeping it as-is since anyone familiar with AlphaZero would recognise it. **Decision needed:** keep math notation or use descriptive names?

### Config parameter naming

Currently three different names for the same concept:

| File | Parameter name |
|------|---------------|
| `main.py` | `args` |
| `core/coach.py` | `run_config` |
| `core/mcts.py` | `config` (but takes `MCTSConfig`, fine) |
| Wrapper files | `args` |

**Standardise on `config`** — it's shorter than `run_config` and more descriptive than `args`. The MCTS `config` parameter is already correct (it takes `MCTSConfig`, a different type).

### Other naming fixes

| Current | Issue | Fix |
|---------|-------|-----|
| `CoordinateIndexDecoder` | Unclear direction | `CoordConverter` or `BoardCoordTransform` |
| `pieces_played_encoded_region` | Too long | `piece_tracker` or `played_bitmap` |
| `nnet` / `net` | Used interchangeably | Standardise on `net` for the PyTorch module, `wrapper` for the NNetWrapper |
| `board_x` / `board_y` | x usually = width, but here x = height | Use `board_rows` / `board_cols` |
| Arena `players` list trick | `[p2, None, p1]` indexed by `cur_player+1` | Use dict: `{1: p1, -1: p2}` |

### File and module naming

Current naming is already good (snake_case, descriptive). No changes needed except:
- `neuralnets/` → consider `nn/` for brevity? Or keep — it's explicit.

### Import modernisation

Three inconsistencies to clean up in the same pass:

1. **`nptyping` → `numpy.typing`** — `blokusduo/pieces.py` uses `from nptyping import NDArray`. Every other file uses `from numpy.typing import NDArray`. They're different libraries. `nptyping` is poorly maintained. Standardise on `numpy.typing`.

2. **`Tuple`/`List` → `tuple`/`list`** — Some files import `from typing import Tuple, List` but the code uses lowercase `tuple[...]` (Python 3.10+ syntax). Drop the typing imports, use builtins everywhere. Since we're targeting 3.11+ (StrEnum), this is safe.

3. **`os.path` → `pathlib.Path`** — `core/coach.py` uses `os.path.join()` and `os.path.isfile()`. Both wrapper files use `os.listdir()`. Convert to `Path` methods for consistency with the rest of the codebase.

**Estimated effort:** ~1.5 hours

---

## S8. Build MetricsCollector

### Why not use the logging system?

Application logging (loguru) is for human-readable operational messages. Training metrics (loss curves, MCTS stats, arena results) are structured numerical data that gets queried and plotted. Different tools for different jobs.

### Design

```python
@dataclass
class MetricsCollector:
    """Collects metrics from all components during a training run.

    Components call collector.log_* methods during execution.
    At generation boundaries, call flush() to write to parquet.
    """

    def log_training(self, generation: int, epoch: int,
                     pi_loss: float, v_loss: float, lr: float) -> None: ...

    def log_mcts(self, generation: int, game: int,
                 sims_per_sec: float, tree_nodes: int,
                 avg_branching: float) -> None: ...

    def log_arena(self, generation: int,
                  wins: int, losses: int, draws: int) -> None: ...

    def flush(self, output_dir: Path) -> None:
        """Write all collected metrics to parquet files."""
```

### Why this approach over alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **MetricsCollector (recommended)** | Simple, explicit, type-safe, easy to test | Components need a reference to the collector |
| Event stream / pub-sub | Decoupled, flexible | Over-engineered for our scale |
| Write-to-log-and-parse-later | Zero coupling | Fragile parsing, hard to query |
| Each component writes own parquet | No shared state | Inconsistent schemas, no consolidation |

Each component (`Coach`, `MCTS`, `Arena`, `NNetWrapper`) gets a reference to the collector and calls the appropriate `log_*` method. The collector batches everything in memory and flushes to parquet at generation boundaries. The reporting module reads the parquet files to generate dashboards.

This also solves the current two-stage pickle → parquet pipeline. Training data goes straight to parquet via the collector.

**Estimated effort:** ~1.5 hours

---

## S9. CLI & Entry Point Cleanup

### Current state

- Config loaded from JSON via `load_args("test_run.json")` in `main.py` line 16
- Filename is **hardcoded** — changing the config requires editing source code
- `load_args()` runs at **module level** — importing `main` triggers config loading
- No CLI argument parsing
- No `__main__.py` — can't run as `python -m alphablokus`

### Fix

1. **Add CLI via argparse or click** — pass config file as an argument:
   ```bash
   python main.py --config run_configurations/full_run.json
   ```

2. **Move config loading into `if __name__ == "__main__"` block** — prevent module-level side effects.

3. **Consider TOML for config** — since we're adopting `pyproject.toml` anyway, TOML is more natural for config files than JSON (supports comments, cleaner syntax). Not essential — JSON works fine.

4. **Keep `dataclass_wizard.fromdict()` for deserialization** — it's a clean pattern. Frozen dataclasses for config are excellent and should be preserved.

**Estimated effort:** ~30 minutes

---

## S10. Data Storage — Parquet Everywhere

### Current state

| Data | Format | Issues |
|------|--------|--------|
| Self-play games | Pickle | Slow, fragile across Python versions |
| Training metrics | Pickle → Parquet (two-stage) | Unnecessary middle step |
| Arena results | Individual Parquet per generation | Never consolidated |
| Checkpoints | PyTorch `.pth.tar` | Missing optimizer state (bug fix doc) |
| Timing | Single Parquet | Appended across generations ✓ |
| Reports | HTML + Plotly | Good ✓ |

### Fix

1. **Self-play history → Parquet directly** via the MetricsCollector. No more pickle.
2. **Training metrics → Parquet directly** — the two-stage pipeline was unnecessary.
3. **Arena data → consolidated Parquet** — append to a single file like timing data.
4. **Add schema versioning** — include a `schema_version` column in all parquet files so we can handle format changes gracefully.

Don't migrate to HDF5 — parquet is simpler, columnar, and we already use it. Keep things consistent.

**Estimated effort:** ~1 hour (mostly handled by the MetricsCollector work in S8)

---

## S11. Testing

### Current state

Zero tests. No `tests/` directory, no test files, no test framework configured.

### Recommended framework

**pytest** with `conftest.py` fixtures. No test classes — pytest's function-based style with fixtures is cleaner and less boilerplate.

### Directory structure

```
tests/
├── conftest.py              ← shared fixtures (configs, boards, etc.)
├── test_core/
│   ├── test_mcts.py         ← MCTS produces valid moves on TicTacToe
│   ├── test_arena.py        ← Arena runs a complete game
│   └── test_coach.py        ← Coach orchestration basics
├── test_blokusduo/
│   ├── test_game.py         ← Placement validation, boundary checks
│   ├── test_pieces.py       ← Piece loading, orientation mapping
│   └── test_action_encoding.py ← Encode/decode roundtrip
└── test_tictactoe/
    └── test_game.py         ← Smoke test: TTT still works after refactor
```

### What to test (priority order)

| Priority | What | Why |
|----------|------|-----|
| **Must** | Piece placement validation | Core game logic, boundary bugs known |
| **Must** | Action encoding roundtrip | Bridge between NN and game — can't be wrong |
| **Must** | MCTS produces valid moves (TicTacToe) | Regression gate after any framework changes |
| **Must** | Full TicTacToe game simulation | End-to-end smoke test |
| **Should** | Config loading from JSON | Catches boolean string issues, missing fields |
| **Should** | Board coordinate conversion | Two coordinate systems, easy to get wrong |
| **Should** | Metrics collector output | Verify parquet schema is correct |
| **Could** | Neural net forward pass shapes | Catches tensor dimension mismatches |
| **Could** | Checkpoint save/load roundtrip | Verify model + optimizer state preserved |

### Testing philosophy

- **No mocks for game logic.** Use real boards, real pieces, real MCTS. The game objects are cheap to create.
- **Fixtures for configs.** Shared `test_config` fixture with small values (2 MCTS sims, 1 epoch, etc.)
- **Parameterized tests for piece placements.** Each piece × orientation × edge case = many combinations, `@pytest.mark.parametrize` handles this cleanly.
- **One integration test** that runs a complete 1-generation TicTacToe training loop. Slow (~30s) but catches framework-level breaks. Mark with `@pytest.mark.slow`.

**Estimated effort:** ~3-4 hours for initial suite

---

## S12. Profiling & Observability (Nice-to-have)

### Lightweight instrumentation (always on)

- Wrap MCTS search, neural net inference, and training epochs with `time.perf_counter()`
- Log per-generation summary stats (mean, p50, p95, max) via MetricsCollector → Parquet
- Add `torch.cuda.max_memory_allocated()` / `torch.mps.current_allocated_memory()` after training

### Deep profiling (on demand)

- `cProfile` + `snakeviz` for CPU profiling of a single self-play game
- `torch.profiler` for GPU kernel-level analysis
- `tracemalloc` for Python memory tracking during MCTS
- `py-spy` for sampling profiler (no code changes)

### Profiling report

Extend the existing Plotly HTML reporting to include:
- Time breakdown pie chart per generation (self-play / training / arena / I/O)
- MCTS drill-down: inference vs tree traversal vs move generation
- Memory timeline: peak RSS and GPU memory across generations
- Throughput trends: games/hour and examples/second over time
- Bottleneck flags: auto-highlight any metric >2× slower than previous generation

**Estimated effort:** ~2.5 hours

---

## Note: Compute Infrastructure

Unchanged from the original architecture review — see [`04-COMPETITIVE-LANDSCAPE.md`](../reference/04-COMPETITIVE-LANDSCAPE.md) for external context.

**Tiered approach:**
- **Tier 1: Mac M4 Pro** — development, debugging, small test runs (MPS backend)
- **Tier 2: RTX 3080 Ti PC** — full training runs (CUDA, most mature backend)
- **Tier 3: AWS/cloud** — overflow if PC isn't enough (g5.xlarge spot ~8-10/day)

No action item — this informs Phase 3 decisions.
