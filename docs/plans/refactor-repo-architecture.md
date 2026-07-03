# Repo Architecture Refactor — src/ Layout, Core Decomposition, Productionisation

This plan restructures the whole repository into a modern, installable `src/alphablokus` package with real submodules, completes type coverage, adds CI + type checking, strips comment noise, and refreshes all documentation — **without rewriting any validated logic** (MCTS, Coach, Arena, move generation, and game rules are relocated and clarified, never re-derived). It is grounded in a four-way code audit (core/, games+reporting, scripts+tests+configs, docs) performed 2026-07-03. Companion docs: [PLAN-FORMAT](../guides/PLAN-FORMAT.md), [STYLE-GUIDE](../guides/STYLE-GUIDE.md). Branch per phase off `main`: `refactor/repo-architecture-p<N>`; one PR per phase; the full test suite must be green at every phase boundary (and at every commit within Phases 1–4, since each row is an atomic move + import sweep).

> **Sequencing vs `docs/plans/oom-hardening.md` — decided 2026-07-03: refactor first, OOM execution deferred.** The original draft required O1–O9 to complete first, but the box is unavailable for the verification that plan needs (peak-RSS at 10k games/gen), so implementing it now would ship unverified memory-critical format changes. Instead: R1 commits the OOM plan file (tracked, visibly in-flight but deferred), the refactor proceeds, and R18 re-maps the OOM plan's file/line citations onto the new layout so it stays executable the moment the box returns. The interim mitigation (`num_eps ≤ 8000`) stays in force until O1–O2 land. Upside: O1–O3/O6 will be implemented against their cleaner post-refactor homes (`training/replay_buffer.py`, `storage/selfplay_store.py`).

**Execution routing:** every row is tagged **MECH** (mechanical: `git mv`, renames, import sweeps, config — safe for any subagent) or **JUDGE** (requires design judgement: module boundaries, API shape, prose — route to a stronger agent or review closely). Estimated total effort: ~45–55 h.

---

## Checklist

| # | Phase | Item | Type | Effort | Priority | Done |
|---|-------|------|------|--------|----------|------|
| R1 | 0 | Commit `docs/plans/oom-hardening.md` (currently untracked) with a deferred-execution banner cross-referencing this plan | MECH | 15 min | High | ✅ |
| R2 | 0 | Repo hygiene: prune stale worktrees, delete merged branches, remove `scripts/.DS_Store`, extend `.gitignore` | MECH | 20 min | Medium | ✅ |
| R3 | 0 | Delete `notebooks/` (audit confirms no unique content) | MECH | 10 min | Medium | ✅ |
| R4 | 0 | Add GitHub Actions CI (ruff + pytest `-m "not slow"`, base + jax jobs) against the *current* layout | MECH | 1 h | High | ✅ |
| R5 | 0 | Add mypy at a lenient baseline; wire into CI | JUDGE | 1.5 h | High | ✅ |
| R6 | 1 | Move `core/`, `games/`, `reporting/` → `src/alphablokus/`; add `[build-system]`; repo-wide import rewrite (src, tests, scripts) | MECH | 2.5 h | High | |
| R7 | 1 | `main.py` → `alphablokus/cli.py` + `[project.scripts]` console entry; delete root `main.py`; update every documented command | MECH | 45 min | High | |
| R8 | 1 | Single `pieces.json` accessor via `importlib.resources`; kill the four divergent load paths | MECH | 1 h | High | |
| R9 | 2 | `search/`: move `mcts.py`; extract profiling dataclasses to `search/stats.py` | MECH | 1 h | High | |
| R10 | 2 | `storage/`: split `storage.py` → `metrics.py` + `selfplay_store.py`; move `sparse_policy.py`; single `ProcessedExample` home | MECH | 1.5 h | High | |
| R11 | 2 | `evaluation/`: move `arena.py`, `players.py`, `acceptance.py`, `symmetry_diagnostic.py`→`symmetry.py`; dedupe `Player` alias; fix stale docstring + no-op ternary | MECH | 1 h | High | |
| R12 | 2 | `parallel/`: `parallel_self_play.py`→`pool.py`; move `inference_server.py`/`inference_channel.py`; co-locate server lifecycle helpers | MECH | 1.5 h | High | |
| R13 | 2 | `selfplay/`: `self_play.py`→`episode.py`; extract backend dispatch from `Coach` into `selfplay/generate.py` | JUDGE | 1.5 h | High | |
| R14 | 2 | Consolidate `core/jaxplay/` + `games/blokusduo/jaxenv/` → `games/blokusduo/jax/`; rename colliding `GameRecord` | MECH | 1.5 h | High | |
| R15 | 2 | `registry.py`: rename `game_factory.py`; concentrate ALL core→games coupling (game, net, jax backend, TTT oracle) | JUDGE | 1.5 h | High | |
| R16 | 2 | `training/`: move `coach.py`; extract `replay_buffer.py` + `diagnostics.py` (memory snapshots) | JUDGE | 2 h | High | |
| R17 | 2 | Extract `training/eval_set.py`, `evaluation/elo.py`; TTT-specific eval → `games/tictactoe/oracle.py` | JUDGE | 2 h | High | |
| R18 | 2 | Delete emptied `core/`; final import sweep; green-suite checkpoint | MECH | 30 min | High | |
| R19 | 3 | Extract `games/blokusduo/codec.py` from `board.py`; absorb the `ActionCodec.encode_from_components` monkey-patch | JUDGE | 1.5 h | High | |
| R20 | 3 | `games/blokusduo/movegen/` subpackage (`runtime.py`, `tables.py`, table-build CLI) | MECH | 45 min | Medium | |
| R21 | 3 | `games/blokusduo/pentobi/` subpackage (`gtp.py`, `player.py`, `translation.py`) | MECH | 30 min | Medium | |
| R22 | 3 | Rename `neuralnets/` → `nn/` in both games | MECH | 20 min | Low | |
| R23 | 3 | Split `reporting/training.py` (1,824 lines) → `report.py` + `charts.py` + `arena_replays.py` | JUDGE | 2 h | High | |
| R24 | 3 | Remove reporting's private reach-ins; add the small public accessors it needs on board/game | JUDGE | 1 h | Medium | |
| R25 | 4 | Restructure `tests/` to mirror `src/alphablokus/`; move cross-test helper imports into conftest/fixtures | MECH | 1.5 h | High | |
| R26 | 4 | `tests/fixtures/blokus_positions.py` → `alphablokus/testing/positions.py` (it is imported by 3 scripts + 7 test files) | MECH | 45 min | Medium | |
| R27 | 5 | Typing-gap sweep: missing `from __future__ import annotations` (9 files) + every unannotated signature | MECH | 2 h | High | |
| R28 | 5 | Raise mypy to `disallow_untyped_defs` globally; fix fallout | JUDGE | 2.5 h | High | |
| R29 | 5 | Adopt `ruff format` in one dedicated commit; enable format check in CI | MECH | 45 min | Medium | |
| R30 | 5 | Comment-noise pass: delete restating comments, keep rationale; deduplicate docstrings | JUDGE | 2 h | Medium | |
| R31 | 5 | Dead-code removals + fix `INeuralNetWrapper.train` protocol drift | MECH | 1 h | Medium | |
| R32 | 6 | Reorganise `scripts/` into operational / `benchmarks/` / `profiling/`; delete the two superseded baked-data report scripts; drop `PYTHONPATH` incantations | JUDGE | 1.5 h | Medium | |
| R33 | 6 | Sort `run_configurations/` into current vs `archive/`; sweep references | MECH | 45 min | Low | |
| R34 | 7 | Mechanical docs path sweep for the new layout (README tree, CLAUDE/AGENTS, 02/03/04/06/07, AI-CONTEXT) | MECH | 1.5 h | High | |
| R35 | 7 | README rewrite: status, JAX/Gumbel backend, CI badge, new layout tree | JUDGE | 2 h | High | |
| R36 | 7 | Make AGENTS.md canonical with CLAUDE.md as symlink; refresh Critical path, gotchas, doc tree | JUDGE | 1.5 h | High | |
| R37 | 7 | Rewrite `docs/guides/AI-CONTEXT.md` (five factually wrong claims) | JUDGE | 1 h | High | |
| R38 | 7 | Freshness fixes to numbered docs: 02, 03, 06, 07, 08, 09 | JUDGE | 2 h | Medium | |
| R39 | 7 | Guides + IDEAS refresh: STYLE-GUIDE (layout/tooling section), REMOTE-TRAINING (new commands), IDEAS.md (I1/I3 partially shipped), broken-link sweep | JUDGE | 1.5 h | Medium | |
| R40 | 8 | Full verification: complete suite incl. `slow` + jax extra; end-to-end `test_run.json` and a jax CPU config; render both reports; load a pre-refactor checkpoint | MECH | 1 h | High | |
| R41 | 8 | Box validation: quick GPU run (python + jax gumbel), `pentobi_benchmark` sanity, `fetch_run_reports.sh` | MECH | 1 h | High | |
| R42 | 8 | Archive this plan | MECH | 10 min | High | |

**Phase boundaries (suite green, PR merged):** Phase 0 = safety net on the old layout · 1 = the package exists · 2 = `core/` fully dissolved · 3 = games/reporting internal shape · 4 = tests mirror source · 5 = typing/style complete · 6 = scripts/configs · 7 = docs · 8 = verification + close.

---

## Design decisions

Each decision below is committed, not open. Rejected alternatives are recorded so the reasoning survives.

### D1. `src/` layout, installable `alphablokus` package, hatchling

Adopt the PyPA-recommended `src/` layout with one import package, `alphablokus`, made installable via PEP 621 metadata plus `[build-system]` (hatchling). `uv sync` then installs the project editable, so `tests/` and `scripts/` import the real installed package instead of relying on the repo root happening to be on `sys.path` (the audit confirmed that is the *only* thing making imports work today — there is no editable install). Entry point becomes a console script (`uv run alphablokus --config …`).

- **Rejected: status quo (flat top-level packages).** Works only when CWD is the repo root; `reporting/` already carries a CWD-relative `pieces.json` bug because of it. Reads as script-soup, not a library.
- **Rejected: flat package at root without `src/` (the flax/jax style).** Defensible for huge libraries, but `src/` is the current packaging-guide recommendation, prevents accidentally importing the uninstalled tree, and signals deliberate packaging craft — the point of a portfolio repo.
- **Rejected: `uv_build` backend.** Newer and fine, but hatchling is the widely recognised default; recognisability wins for a repo meant to be read by strangers.

### D2. Core decomposition: hybrid pipeline/concern split

`core/` dissolves into subpackages named for the pipeline where the code *is* phase-shaped, with shared machinery extracted beside them:

- `selfplay/`, `training/`, `evaluation/` — the three phases, readable as the AlphaZero loop.
- `search/` — MCTS. **Not** filed under self-play because arena/Elo also use it, via `NetworkPlayer` (`players.py:93`). A pure phase split would orphan it.
- `parallel/` — the worker pool serves both self-play episodes *and* arena/Elo games (`run_two_player_games_parallel`); it is cross-phase infrastructure, like search.
- `storage/` — `MetricsCollector` and `SelfPlayStore` are two independent subsystems currently sharing one 1,319-line file; every phase writes to them.
- `config.py`, `interfaces.py`, `registry.py`, `cli.py` at package root — small, cross-cutting, and the first things a reader should find.

**Future-proofing check (continuous generations):** moving to continuous training later changes the *orchestration* (Coach's loop) and the buffer, both of which live inside `training/` after this plan (`coach.py`, `replay_buffer.py`). The layout survives; only files within `training/` change. A stricter phase split (e.g. burying MCTS or the buffer inside a phase) would age badly; this hybrid doesn't.

- **Rejected: pure three-phase split.** Orphans MCTS, storage, the worker pool, and config — they'd end up in an `common/`-style dumping ground, which is the current problem with a different name.
- **Rejected: pure concern split (flat modules, better names).** That is roughly today's layout; it hides the pipeline, which is exactly what Henry says he can't hold in his head.

### D3. Backends: both first-class; all JAX code lives with the game

The python engine is **not legacy** and will not be quarantined: it is the framework itself — it drives arena, Elo, and Pentobi evaluation on every run regardless of `selfplay_backend`, it is the only engine for TicTacToe, and it is the parity oracle the JAX pipeline is validated against (`docs/research/jax-pipeline-ab.md`). The JAX backend, conversely, is Blokus-only and inference-only by design (`core/jaxplay/backend.py` hard-raises for other games). The honest structure follows the facts:

- `core/jaxplay/` (driver: `backend.py`, `actors.py`, `harvest.py`) + `games/blokusduo/jaxenv/` (env: kernels, tables, net, search, checkpoint, bridge) merge into **`games/blokusduo/jax/`** — nine flat modules, one home. This *removes* today's layering violation where "game-agnostic" core imports `games.blokusduo.jaxenv`.
- Coach reaches the backend through one seam (`selfplay/generate.py` → `registry.py`), same as it reaches concrete games.
- Naming note: a subpackage called `jax` does not shadow the `jax` library — Python 3 absolute imports always resolve `import jax` to the top-level package. Rejected keeping the name `jaxenv` because the directory now also holds the driver, not just the env.

- **Rejected: "JAX default, python quarantined as legacy" (the starting intuition).** Factually wrong about the system: evaluation and TicTacToe have no JAX path, and the parity gates need the python engine forever. Quarantining it would misdocument the architecture to exactly the readers we're trying to impress.
- **Rejected: a top-level `backends/` package with python and jax side by side.** The python "backend" *is* the framework (MCTS + episode loop used by evaluation too); the jax backend is game-specific. A generic `backends/` would either recreate the core→blokusduo violation or force a premature multi-game backend protocol nobody needs yet.

### D4. Interfaces: keep Protocol + explicit subclassing (no change of idiom)

The repo's existing pattern — `typing.Protocol` interfaces that implementations explicitly inherit (`class TicTacToeGame(IGame)`) — is both idiomatic modern Python *and* reads like C# interfaces, and it is already codified in STYLE-GUIDE.md with sound reasoning (static enforcement + structural fallback). Strong modern Python codebases use exactly this where an interface has multiple implementations; ABCs remain only where shared behaviour exists (`BaseNNetWrapper(INeuralNetWrapper, ABC)` stays). What we fix instead is *drift*: `INeuralNetWrapper.train` omits the `eval_set` kwarg Coach actually passes (R31), and `inference_server.py` uses `@runtime_checkable` against the style guide's explicit prohibition — align it or document the exception at the use site (R31).

### D5. One composition root: `registry.py`

Today core→games coupling leaks from three places: `game_factory.py` (by design), `coach.py`'s lazy TicTacToe-minimax imports (lines 683, 842–843), and `core/jaxplay` (whole package). After this plan there is exactly **one** module allowed to name concrete game classes: `alphablokus/registry.py` (renamed from `game_factory.py`, extended to resolve the jax self-play backend and the TTT oracle hooks). Everything else in the framework depends only on the protocols. This makes "game-agnostic core" true rather than aspirational, and gives future games a single registration point.

### D6. Tests mirror source; shared position fixtures become a package module

`tests/` reshapes to mirror `src/alphablokus/` one-to-one (a stated non-negotiable). Two audit findings force a design choice: five jaxenv tests import `DEV_CACHE_PATH` from another test's conftest, one test imports helpers from two other *test modules*, and — worse — three scripts import `tests.fixtures.blokus_positions`. Production-adjacent tooling importing from `tests/` is the wrong direction. The position-cache utility moves to `alphablokus/testing/positions.py` (the `numpy.testing`/`pandas.testing` idiom: shipped test utilities); the `dev_5000.npz` cache file stays in `tests/fixtures/` (checked-in data, not package data) with the path passed in / defaulted via one constant.

### D7. Tooling: mypy, ruff format, GitHub Actions

- **mypy** (not pyright, not ty): the "everything typed" requirement needs an enforcer, and mypy's `disallow_untyped_defs` maps to it exactly; it is the recognised default in ML codebases and its gradual knobs fit a staged rollout (lenient baseline in R5 → strict-ish in R28). Rejected pyright-strict (fights numpy/torch too hard for the payoff here) and Astral's `ty` (not yet mature enough to gate a portfolio repo on).
- **ruff format** adopted in one dedicated, no-logic commit (R29) so `git blame` damage is contained and CI can enforce formatting forever after.
- **CI**: ubuntu-latest; jobs = lint (ruff check + format check), typecheck (mypy), test (uv sync, `pytest -m "not slow"`), test-jax (`--extra jax`, CPU). Pentobi tests already self-skip when the binary is absent. CI lands *first* (R4, on the old layout) so every subsequent move is verified by the same gate.

### D8. Docs & plans housekeeping

- **AGENTS.md becomes canonical; CLAUDE.md becomes a symlink to it.** They are near-duplicate forks today and AGENTS.md is the staler one; two files guarantee future divergence. AGENTS.md is the emerging cross-tool standard; Claude Code follows the symlink.
- **`docs/plans/` keeps its two-state lifecycle (loose = in-flight, `archive/` = done). No `future/` directory** — rejected from the starting intuitions because `docs/IDEAS.md` already *is* the candidate-work register with a working promotion flow (I4 → replay-buffer-refactor proves it), and an empty third directory is structure for its own sake.
- **`notebooks/` is deleted** (R3): the audit walked all 155 cells of `eval.ipynb` — every idea in it is realised in `game.py`/`movegen_*`/`reporting/` or captured in `docs/research/pentobi/`; its API references are dead (`insert_piece`, `board.as_array`, pre-`games.` package paths). Git history preserves it.

### D9. Sequencing against in-flight work

Decided 2026-07-03 (see banner): the refactor runs first because the box is unavailable for OOM verification either way, and the OOM work lands more cleanly in the refactored structure. The cost — the OOM plan's line-number citations go stale — is paid deliberately and repaired in R18, where every citation is re-mapped to its new file/method (method names, not line numbers, so they cannot rot again). `SelfPlayStore.load_window` (which OOM O2 planned to delete as dead) is deleted here in R31 instead, with the OOM plan updated to match.

---

## Target layout — before → after

```
BEFORE                                    AFTER
AlphaBlokus/                              AlphaBlokus/
├── main.py                               ├── src/alphablokus/
├── core/                                 │   ├── __init__.py
│   ├── interfaces.py                     │   ├── cli.py                  ← main.py
│   ├── config.py                         │   ├── config.py
│   ├── mcts.py                           │   ├── interfaces.py
│   ├── coach.py                          │   ├── registry.py             ← game_factory.py (+ backend/oracle resolution)
│   ├── self_play.py                      │   ├── search/
│   ├── parallel_self_play.py             │   │   ├── mcts.py
│   ├── arena.py                          │   │   └── stats.py            ← MCTSMoveStats/EpisodeStats + memory estimation
│   ├── players.py                        │   ├── selfplay/
│   ├── acceptance.py                     │   │   ├── episode.py          ← self_play.py
│   ├── symmetry_diagnostic.py            │   │   └── generate.py         ← backend dispatch (serial/parallel/jax), from Coach
│   ├── storage.py                        │   ├── parallel/
│   ├── sparse_policy.py                  │   │   ├── pool.py             ← parallel_self_play.py
│   ├── inference_channel.py              │   │   ├── inference_server.py
│   ├── inference_server.py               │   │   └── inference_channel.py
│   ├── game_factory.py                   │   ├── training/
│   └── jaxplay/                          │   │   ├── coach.py            (slimmed: the generation loop)
│       ├── backend.py                    │   │   ├── replay_buffer.py    ← buffer deque + save/load/resume, from Coach
│       ├── actors.py                     │   │   ├── eval_set.py         ← _ensure_eval_set, from Coach
│       └── harvest.py                    │   │   └── diagnostics.py      ← MemorySnapshot helpers, from Coach
├── games/                                │   ├── evaluation/
│   ├── base_wrapper.py                   │   │   ├── arena.py
│   ├── tictactoe/                        │   │   ├── players.py
│   │   ├── board.py / game.py            │   │   ├── acceptance.py
│   │   ├── minimax.py                    │   │   ├── elo.py              ← _compute_elo, from Coach
│   │   └── neuralnets/                   │   │   └── symmetry.py         ← symmetry_diagnostic.py
│   └── blokusduo/                        │   ├── storage/
│       ├── board.py  (4 concerns)        │   │   ├── metrics.py          ← MetricsCollector, EvalSet, CycleStage
│       ├── game.py                       │   │   ├── selfplay_store.py   ← SelfPlayStore
│       ├── pieces.py / pieces.json       │   │   └── sparse_policy.py
│       ├── movegen_runtime.py            │   ├── games/
│       ├── movegen_tables.py             │   │   ├── base_wrapper.py
│       ├── pentobi_gtp.py                │   │   ├── tictactoe/
│       ├── pentobi_player.py             │   │   │   ├── board.py / game.py / minimax.py
│       ├── pentobi_translation.py        │   │   │   ├── oracle.py       ← TTT-specific eval, from Coach
│       ├── neuralnets/                   │   │   │   └── nn/             ← neuralnets/
│       └── jaxenv/                       │   │   └── blokusduo/
│           ├── kernels.py / tables.py    │   │       ├── board.py        (board state + encoding only)
│           ├── net.py / search.py        │   │       ├── codec.py        ← Action, ActionCodec (monkey-patch absorbed),
│           ├── checkpoint.py / bridge.py │   │       │                      CoordinateIndexDecoder, from board.py
├── reporting/                            │   │       ├── game.py / pieces.py / pieces.json
│   ├── display.py                        │   │       ├── movegen/        ← movegen_runtime.py, movegen_tables.py
│   ├── display_tictactoe.py              │   │       ├── pentobi/        ← pentobi_{gtp,player,translation}.py
│   ├── display_blokusduo.py              │   │       ├── jax/            ← jaxenv/* + core/jaxplay/* (9 modules)
│   ├── mcts_profiling.py                 │   │       └── nn/             ← neuralnets/
│   └── training.py  (1,824 lines)        │   ├── reporting/
├── notebooks/eval.ipynb                  │   │   ├── report.py           ← orchestrator + metrics loading
├── scripts/          (26 flat files)     │   │   ├── charts.py           ← the ~15 _make_* plotly builders
├── run_configurations/ (31 flat JSONs)   │   │   ├── arena_replays.py    ← embedded HTML/CSS/JS template + builder
├── tests/                                │   │   ├── display{,_tictactoe,_blokusduo}.py
│   ├── test_core/                        │   │   └── mcts_profiling.py
│   ├── test_blokusduo/                   │   └── testing/
│   ├── test_tictactoe/                   │       └── positions.py        ← tests/fixtures/blokus_positions.py
│   ├── test_games/                       ├── tests/                      (mirrors src/alphablokus/)
│   └── test_integration/                 │   ├── search/ selfplay/ parallel/ training/ evaluation/ storage/
├── docs/                                 │   ├── games/{tictactoe,blokusduo}/  reporting/  integration/
└── pyproject.toml (no build-system)      │   └── fixtures/ (dev_5000.npz stays)
                                          ├── scripts/                    (operational top-level)
                                          │   ├── benchmarks/  profiling/
                                          ├── run_configurations/ (+ archive/)
                                          ├── docs/  (refreshed; AGENTS.md canonical, CLAUDE.md → symlink)
                                          ├── .github/workflows/ci.yml
                                          └── pyproject.toml (hatchling, console script, mypy, package data)
```

---

## Risks and how each phase verifies itself

| Risk | Why it's real | Mitigation / verification |
|---|---|---|
| Silent breakage from string-based imports | Would survive an import sweep unnoticed | Audit found **zero** dynamic imports, zero mock-patch strings in the whole repo; game/backend dispatch is by config string (`"blokusduo"`), not module path. Moves therefore fail **loudly** at import time — and CI (R4) runs on every commit. |
| JAX path breaks with no GPU to test on | Mac is CPU-only | The jaxenv parity/gumbel/search tests run on CPU (jax extra) and gate every phase in CI; R41 additionally runs a real jax gumbel config on the box. |
| Torch checkpoints stop loading after moves | Pickled module paths would break | Checkpoints store only `state_dict` + optimizer/scheduler tensors (`base_wrapper.py:457–478`) — no class paths. R40 explicitly loads a pre-refactor checkpoint from `temp/` as proof. |
| Multiprocessing workers (spawn/forkserver) resolve worker functions by module path | Renames change those paths | Paths are resolved at import inside each fresh worker process; within any commit the names are consistent. `test_parallel_self_play.py` + the slow integration tests cover it at each boundary. |
| Pentobi evaluation silently regresses | Binary lives outside the repo (`$PENTOBI_GTP_PATH` / `~/code/pentobi/...`) | Path-resolution logic is untouched; pentobi tests self-skip without the binary locally but run wherever it exists; R41 runs `pentobi_benchmark.py` for real. |
| Resume / parquet artifacts from old runs stop reading | On-disk formats must outlive the refactor | No schema changes anywhere in this plan; `test_resume.py` + `test_self_play_parquet.py` run at every boundary; R40 renders a report from an existing run directory via `--report-only`. |
| Collision with OOM-hardening work | Same files, exact line-number citations | OOM execution deferred (box unavailable); its citations re-mapped to the new layout in R18 (D9). |
| Coach decomposition changes behaviour | It's the validated orchestrator | R13/R16/R17 are *method relocations only* — bodies move verbatim into collaborators; no control flow is redesigned. The slow integration tests (`test_training_loop.py`, `test_jax_training_loop.py`, `test_resume.py`) are the acceptance gate for each of those rows. |
| Docs drift while code moves under them | Long plan, many phases | All prose/doc work is deferred to Phase 7, after the tree is final; R34 is a mechanical path sweep with a grep checklist. |

---

## R1. Commit the OOM-hardening plan

`docs/plans/oom-hardening.md` is untracked (`??` in git status) — invisible to clones and to the plans-lifecycle invariant. Commit it with a status banner recording the sequencing decision (D9): execution deferred until the box is available; the repo-architecture refactor runs first and will re-map this plan's citations (R18); interim mitigation `num_eps ≤ 8000` remains in force.

## R2. Repo hygiene

- `git worktree prune` (the `/private/tmp/.../run3` worktree is already flagged prunable); remove `.claude/worktrees/numba-hot-path` — its branch merged into main via PR #15 and the tree is clean.
- Delete merged local branches (`chore/run3-overnight`, `feat/jax-pipeline-plan`, `feat/jax-spike`, `feat/pentobi-eval-improvements`) and their remotes; check `--no-merged` before touching `feat/pentobi-harness` / `worktree-numba-hot-path`.
- `git rm --cached scripts/.DS_Store` if tracked (it's present on disk); add `.DS_Store` to `.gitignore` if missing.

## R3. Delete notebooks/

`git rm -r notebooks/`. Justification recorded in D8: the audit walked all 155 cells; every algorithm idea is implemented (move generation in `game.py` + `movegen_*`, superseded report prototype in `reporting/training.py`, human-play covered by `core/arena.py` + `scripts/replay.py`), and its code references long-dead APIs. Update the CLAUDE.md/AGENTS.md gotcha that points at it (fully handled again in R36, but don't leave a dangling reference in the meantime).

## R4. CI on the current layout

`.github/workflows/ci.yml`, four jobs on ubuntu-latest with uv caching:

1. **lint** — `uv run ruff check .`
2. **test** — `uv sync --dev` then `uv run pytest -m "not slow"`
3. **test-jax** — `uv sync --extra jax --dev` then `uv run pytest tests/test_blokusduo tests/test_core -m "not slow"` (picks up the jaxenv/jaxplay suites; they self-skip only where `dev_5000.npz` is missing — it's checked in, so they run)
4. **typecheck** — placeholder until R5 lands, then mypy.

Landing this *before* any move means every subsequent commit in this plan is verified by the same gate a reviewer sees. Add the badge to README in R35.

## R5. mypy baseline

Add `mypy` to the dev group and a `[tool.mypy]` block: `python_version = "3.11"`, `check_untyped_defs = true`, `warn_unused_ignores = true`, `ignore_missing_imports` scoped per-module for the untyped deps (`mctx`, `numba`, `wandb`, `plotly`, `dataclass_wizard`). Goal for this row is a **green lenient baseline** (fix trivial errors, `# type: ignore[<code>]` with justification for the hard ones) — strictness ratchets in R28. Wire into the CI typecheck job.

## R6. src/ move + build system

The one intentionally large commit; everything is mechanical and CI-gated:

1. `git mv core src/alphablokus_tmp_core` style moves: `core/` → `src/alphablokus/core/`, `games/` → `src/alphablokus/games/`, `reporting/` → `src/alphablokus/reporting/` (names unchanged in this row — `core/` dissolves in Phase 2, keeping this diff pure-move).
2. Add to pyproject: `[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"`; `[tool.hatch.build.targets.wheel] packages = ["src/alphablokus"]` (hatchling ships `pieces.json` as package data automatically).
3. Repo-wide import rewrite: `from core.` → `from alphablokus.core.`, `from games.` → `from alphablokus.games.`, `from reporting` → `from alphablokus.reporting` — in src, tests, scripts, and `main.py`.
4. `uv sync` (installs the project editable); `uv run pytest -m "not slow"` green.

Note: `core/jaxplay/__init__.py` and `games/blokusduo/jaxenv/__init__.py` set `XLA_PYTHON_CLIENT_PREALLOCATE` at import — verify the env-guard still fires before any jax import after the move (the jax CI job covers this).

## R7. Console entry point

`main.py` moves to `src/alphablokus/cli.py` (contents unchanged apart from imports); add `[project.scripts] alphablokus = "alphablokus.cli:main"`. Delete root `main.py` — the documented invocation everywhere becomes `uv run alphablokus --config run_configurations/test_run.json`. Sweep the command in: README, CLAUDE/AGENTS, `docs/guides/REMOTE-TRAINING.md`, `docs/08-TRAINING-ESTIMATES.md`, `scripts/benchmark_selfplay_backends.py` (spawns via subprocess), and any run-config docstrings. No back-compat shim: the only consumers are Henry's own runbooks, which are updated in the same commit.

## R8. One pieces.json accessor

Add `blokusduo_pieces_path() -> Path` (or a loader returning `PieceManager`) in `games/blokusduo/pieces.py` using `importlib.resources.files("alphablokus.games.blokusduo")`. Replace the four divergent resolutions found by the audit: `game_factory._REPO_ROOT / "games" / …` (breaks under src/), the CWD-relative `Path("games/blokusduo/pieces.json")` in `reporting/display_blokusduo.py:35` and `reporting/training.py:809` (broken today whenever CWD ≠ repo root), the `Path(__file__)`-relative forms in `movegen_*`, and every hard-coded path in scripts/tests (`tests/conftest.py` `pieces_path` fixture included).

## R9. search/

`git mv` `core/mcts.py` → `alphablokus/search/mcts.py`. Extract `MCTSMoveStats`, `MCTSEpisodeStats`, `_estimate_tree_memory_bytes`, and `get_episode_stats`'s dataclass-assembly into `search/stats.py` — the search algorithm and its instrumentation are separable concerns (the jax backend imports *only* `MCTSEpisodeStats`, today dragging in the whole search module). `MCTS` keeps its counters and a thin `get_episode_stats` that builds the dataclass. Importers to update: coach, pool, players, jax backend, `reporting/mcts_profiling.py`, 6 scripts, 8 tests. Algorithm code moves verbatim.

## R10. storage/

Split `core/storage.py` (1,319 lines, two unrelated subsystems) at the existing class boundary: `storage/metrics.py` (`MetricsCollector`, `EvalSet`, `CycleStage`, `_dataclass_to_jsonable`) and `storage/selfplay_store.py` (`SelfPlayStore`). Move `core/sparse_policy.py` → `storage/sparse_policy.py`. Resolve the **duplicate `ProcessedExample`**: the sparse-policy shape in `self_play.py` is the live contract; the dense variant in `storage.py` is drift — `selfplay/episode.py` (R13) becomes the single home, storage imports it. Class bodies move verbatim; the 17 `log_*` methods are *not* redesigned here (any table-driving idea goes to IDEAS.md — out of scope per the no-rewrite rule).

## R11. evaluation/

`git mv`: `arena.py`, `players.py`, `acceptance.py` → `evaluation/`; `symmetry_diagnostic.py` → `evaluation/symmetry.py`. While touching them (all facts from the audit):

- Delete the duplicate `Player` alias — `players.py` owns it; `arena.py` imports it.
- Fix `players.py`'s docstring advertising `MinimaxTicTacToePlayer`/`HumanPlayer` that don't live there.
- `arena.py:118`: `[] if record else []` → `[]`.
- Deduplicate the double local `from core.mcts import MCTS` (lines 84, 131) into one deferred import with a cycle-explaining comment (the cycle itself is addressed by the split — check whether it still exists once `search/` is separate; if not, hoist to a top import).

## R12. parallel/

`parallel_self_play.py` → `parallel/pool.py`; `inference_server.py` and `inference_channel.py` move alongside. Relocate the inference-server lifecycle block currently embedded in the pool (`_server_enabled`, `_resolve_server_batch`, `_run_inference_server`, `_worker_init_self_play_server`, the spawn/teardown block at old lines 461–517) into `parallel/inference_server.py`, co-locating the server with its lifecycle — the pool keeps only the "if server mode, use these init/task fns" branch. Fix `_make_worker_context`'s missing return annotation (`multiprocessing.context.BaseContext`) while moving it.

## R13. selfplay/ + backend dispatch extraction  **(JUDGE)**

`self_play.py` → `selfplay/episode.py` (home of `ProcessedExample` and `GameExamples` aliases per R10). Then the one genuinely shape-changing extraction in core: Coach's three `_run_self_play_*` methods (coach.py:445–512) move, bodies verbatim, into `selfplay/generate.py` as free functions plus a single dispatcher:

```python
def generate_games(config, game, nnet, generation) -> tuple[list[GameExamples], list[MCTSEpisodeStats]]:
    if config.selfplay_backend == "jax": ...      # lazy import via registry (R15)
    elif config.num_parallel_workers > 1: ...     # parallel.pool
    else: ...                                     # in-process MCTS + play_self_play_episode
```

Coach's phase 1 becomes one call. This preserves the existing contract exactly (all three paths already converge on identical return types — that was the seam's design). The Gumbel-requires-jax guard stays in `Coach.__init__`/config validation. Gate: `test_selfplay_backend_config.py`, `test_jaxplay_backend.py`, and the slow integration tests.

## R14. games/blokusduo/jax/

Merge per D3: `git mv` `games/blokusduo/jaxenv/*` and `core/jaxplay/{backend,actors,harvest}.py` into `games/blokusduo/jax/` (nine modules: `kernels, tables, net, search, checkpoint, bridge, actors, harvest, backend`). Fold the two package `__init__`s' XLA-preallocate guards into one. Rename `harvest.py`'s `GameRecord` → `HarvestedGame` (it name-collides with `evaluation/arena.GameRecord` and is a different dataclass). Update the lazy imports in coach/generate, `scripts/benchmark_jax_env.py`, `scripts/validate_jax_search.py`, `scripts/benchmark_selfplay_backends.py`, and the `test_jaxenv_*`/`test_jaxplay_backend` suites. Gate: the full jax CI job.

## R15. registry.py  **(JUDGE)**

Rename `game_factory.py` → `alphablokus/registry.py` and make it the *only* module that names concrete game code (D5):

- Existing `instantiate_game` / `instantiate_game_and_network` (unchanged behaviour; pieces path now via R8's accessor).
- `resolve_jax_selfplay_backend(config)` — returns `games.blokusduo.jax.backend.generate_self_play_games` for `"blokusduo"`, raises otherwise (moves the guard that lives at `backend.py:45–47` up to the seam); `selfplay/generate.py` calls this instead of importing blokusduo directly.
- `resolve_oracle(config)` — returns the TicTacToe minimax oracle hooks for `"tictactoe"`, `None` otherwise, replacing Coach's inline `games.tictactoe.*` lazy imports (consumed in R17).

Keep it boring: a `match config.game` per function, no plugin machinery. Document at the top that this is the composition root and the one sanctioned core→games dependency.

## R16. training/coach.py + replay_buffer.py + diagnostics.py  **(JUDGE)**

`git mv` `coach.py` → `training/coach.py`, then two verbatim-body extractions:

- **`training/replay_buffer.py`** — a `ReplayBuffer` owning the `deque(maxlen=replay_buffer_games)` plus the persistence trio (`save_self_play_history`, `load_self_play_history`, `load_self_play_history_for_resume`) and their `SelfPlayStore` interplay. Coach holds one and delegates. (These are the exact functions OOM O1–O3/O6 will have just rewritten — move whatever they became, verbatim.)
- **`training/diagnostics.py`** — `MemorySnapshot`, `_get_memory_snapshot` (module-level free functions today, coach.py:72–101).

`read_progress_marker`/`_write_progress_marker` stay with Coach (loop state, used by `cli.py`). Delete the dead `skip_first_self_play` flag (always `False`, guards an unreachable branch at old line 255 — "reserved for a future warm-start path" goes to IDEAS.md if wanted). Gate: `test_resume.py`, `test_full_pass_training.py`, slow integration suite.

## R17. eval_set.py, elo.py, tictactoe/oracle.py  **(JUDGE)**

Final Coach slimming, all verbatim moves:

- `_ensure_eval_set` (~110 lines) → `training/eval_set.py`.
- `_compute_elo` (module-level, coach.py:32) → `evaluation/elo.py`.
- The TicTacToe-specific pair — `_evaluate_minimax_tictactoe`, `_minimax_targets_for_eval_set` — → `games/tictactoe/oracle.py`, reached via `registry.resolve_oracle` (R15). This deletes the last `games.*` import from the framework outside the registry.

Coach lands at roughly 450–550 lines: the generation loop, phase dispatch, arena/Elo orchestration methods (thin), acceptance, and marker I/O — readable top to bottom, which is the point of the whole plan. Also fix the untyped `stats` param on `_log_self_play_stats` (coach.py:425 → `MCTSEpisodeStats`).

## R18. Delete core/, re-map OOM citations, checkpoint

`core/` should now contain only `__init__.py` (empty) and nothing else — `git rm` it. `grep -rn "alphablokus\.core\|from core\." src tests scripts docs` must return only docs hits (fixed in Phase 7). Then re-map every file/line citation in `docs/plans/oom-hardening.md` to the new layout (D9): cite file + method name (e.g. `training/replay_buffer.py::save_self_play_history`), never line numbers. Full non-slow suite + both CI jax/base jobs green; merge the Phase 2 PR.

## R19. blokusduo codec.py  **(JUDGE)**

`board.py` (551 lines) currently mixes four concerns. Extract `codec.py`: `CoordinateIndexDecoder` (lines 21–43), `Action` + `ActionCodec` (46–110). While doing so, **absorb the monkey-patch**: `movegen_tables.py:636–654` bolts `encode_from_components` onto `ActionCodec` at import time — make it a real method on the class and delete the patch (behaviour identical; the audit confirms it's pure index arithmetic). `board.py` keeps `BlokusDuoBoard` and `encode_planes_from_placement` (board representation). `game.py`, movegen, pentobi translation, reporting, and the codec tests update their imports. Gate: `test_action_codec.py`, `test_action_encoding.py`, `test_movegen_tables.py`.

## R20. movegen/ subpackage

`movegen_runtime.py` → `movegen/runtime.py`, `movegen_tables.py` → `movegen/tables.py`; `movegen/__init__.py` re-exports `F2MoveGenerator` and the table types. The table-build CLI `main()` inside `tables.py` stays runnable as `python -m alphablokus.games.blokusduo.movegen.tables`. Move-generation logic untouched (Gotcha 1).

## R21. pentobi/ subpackage

`pentobi_gtp.py` → `pentobi/gtp.py`, `pentobi_player.py` → `pentobi/player.py`, `pentobi_translation.py` → `pentobi/translation.py`. `find_pentobi_gtp`'s `$PENTOBI_GTP_PATH` / default-path behaviour is untouched. Fix the stray Cyrillic typo (`генmoves`, player.py:83) while there. Update `scripts/pentobi_benchmark.py`, `scripts/diagnose_pentobi_losses.py`, and the three pentobi test files.

## R22. neuralnets/ → nn/

`git mv` both games' `neuralnets/` → `nn/` (matches the `torch.nn` idiom; `nn` is on the style guide's universal-abbreviation list). Import sweep: wrappers, registry, checkpoint bridge (`jax/checkpoint.py` references the torch net only via `state_dict` keys — no code change), tests.

## R23. Split reporting/training.py  **(JUDGE)**

1,824 lines → three modules along the seams the audit identified:

- `reporting/charts.py` — the ~15 `_make_*` plotly figure builders (pure functions: dataframe in, figure out).
- `reporting/arena_replays.py` — `_ARENA_REPLAYS_STANDALONE_TEMPLATE` (~200 lines of embedded HTML/CSS/JS, old lines 840–1033), the `_CSS` blob, and the replay-page builder.
- `reporting/report.py` — `_load_metrics`, KPI computation, `_instantiate_game` (replace with `registry.instantiate_game` — and add its missing return annotation), and the `create_html_report` orchestrator. `reporting/__init__.py` keeps re-exporting `create_html_report` so `cli.py` is untouched.

Chart/template content moves verbatim — this row is scissors, not a redesign. Gate: `test_metrics.py` + rendering a report from an existing run dir (`--report-only`).

## R24. Reporting public accessors  **(JUDGE)**

`display_blokusduo.py` reaches into `game._valid_moves`, `game._coordinate_index_decoder`, and `board._piece_placement_board`. Add the minimal public surface on the game/board (e.g. a read-only `placement_grid` property on `BlokusDuoBoard`; `coordinate_decoder` property on the game — names finalised in review) and switch reporting to it. This is the only row that *adds* API; keep it to exactly what reporting consumes.

## R25. Tests mirror source

`git mv` test files into a tree mirroring `src/alphablokus/`: `tests/test_core/*` fans out to `tests/{search,selfplay,parallel,training,evaluation,storage}/`, `tests/test_blokusduo/` → `tests/games/blokusduo/` (with `jax/`, `pentobi/`, `movegen/` subdirs matching source), `tests/test_tictactoe/` + `tests/test_games/` → `tests/games/tictactoe/`, `tests/test_integration/` → `tests/integration/`. Fix the audit's cross-test imports: `_run_config`/`_config` helpers (imported by `test_jaxenv_gumbel.py` from two other test modules) become fixtures in the relevant conftest; `DEV_CACHE_PATH` moves to a shared conftest (or R26's module). Keep `test_<module>.py` naming mirroring source modules. Update pyproject `testpaths` markers if needed; suite must collect the same test count before/after (assert via `pytest --collect-only -q | wc -l`).

## R26. alphablokus/testing/positions.py

Move `tests/fixtures/blokus_positions.py` → `src/alphablokus/testing/positions.py` (D6). The `dev_5000.npz` cache stays at `tests/fixtures/blokus_duo_positions/` — the module takes the path as a parameter with that as the documented default for in-repo use. Update the 7 test files and 3 scripts (`benchmark_jax_env.py`, `benchmark_movegen.py`, `validate_jax_search.py`) that import it; scripts no longer import from `tests.*`.

## R27. Typing-gap sweep

Add `from __future__ import annotations` to the 9 files missing it (`config.py`, `storage/` descendants of old storage.py, `games/blokusduo/{game,pieces}.py`, both `nn/net.py`, both `nn/wrapper.py`, `games/tictactoe/game.py`) and fix every concrete gap the audit catalogued:

- `games/blokusduo/nn/net.py`: `calc_conv2d_output` (zero annotations), `__init__`/`forward` unannotated; same pattern in `games/tictactoe/nn/net.py` (torch `forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`).
- `games/base_wrapper.py`: `_inference_autocast` return type; `_compute_eval_set_diagnostics` bare `dict` → a typed `dict[str, float]` (or small dataclass if the keys are fixed — reviewer's call).
- `games/blokusduo/jax/actors.py`: `make_actor` return annotation (name the pair with a `NamedTuple` or `tuple[...]` alias); `params` params typed as a `Params: TypeAlias` for the pytree.
- `parallel/pool.py`: `counter: object` and channel `ctx`/`stop_event` — use `multiprocessing.context.BaseContext`, `synchronize.Event`, `sharedctypes.Synchronized[int]` (typeshed has these; where it genuinely can't be typed, a narrow documented alias beats `object`).
- `training/coach.py` `stats` param (done in R17 if not already).
- `storage.py`'s quoted `"GameRecord"` refs → real imports under `TYPE_CHECKING` now that `from __future__` is in.

## R28. mypy strict-ish  **(JUDGE)**

Ratchet `[tool.mypy]` to `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`, `no_implicit_optional = true` globally; delete the R5 per-module `ignore_errors` debt list and fix the fallout. One design decision lives here (surfaced by the R5 baseline's 14 `[override]` errors): concrete games narrow Protocol parameter types (`TicTacToeGame.get_next_state(board: Board)` vs `IGame`'s `IBoard`). Resolve by making the protocols generic (`IGame(Protocol[TBoard])`, the mathematically clean fix) or by keeping `IBoard` signatures with internal casts — decide when the errors are in front of you, favouring generics if the churn is contained. Where third-party untypedness forces it, prefer local `# type: ignore[<code>]  # <reason>` over module-wide overrides. From here on, "everything is typed" is machine-enforced, not aspirational. CI typecheck job flips to the strict config.

## R29. ruff format

One commit: enable `ruff format`, run it repo-wide, add `ruff format --check` to the CI lint job. No logic changes ride along (enforced by review: the diff must be whitespace/quotes/wrapping only). Line length stays 120.

## R30. Comment-noise pass  **(JUDGE)**

The audit's finding: rationale comments dominate and should stay; genuine restatement is concentrated in a few places. Remove/rewrite per module:

- Both `nn/net.py` files: shape-restating inline comments (`# batch_size * 17837 (which is 14 x 14 x 91 + 1)` etc.) — where the shape is non-obvious, move it into the method docstring once; delete the per-line echoes. Convert the reST-style `:param:` docstring in tictactoe net to Google style.
- `training/coach.py`: phase-banner comments (`# PHASE 1: …` above a method call that says the same) and the class-docstring/`learn`-docstring duplication of the three-step loop — keep one authoritative description.
- General rule applied everywhere touched: a comment may state a constraint or a why; if it narrates the what, the fix is a better name or extraction, not the comment. Rationale comments citing `docs/plans/*` / parity contracts are explicitly preserved.

## R31. Dead code + protocol drift

- Delete `Coach.skip_first_self_play` if it survived R16.
- Delete `SelfPlayStore.load_window` (test-only today; OOM O2 listed it for deletion — update that plan's row to "done in refactor R31").
- Delete the deprecated `render_top_k_moves_html` stubs in reporting (`display.py:78` marks them `[Deprecated]`).
- Fix `INeuralNetWrapper.train` to include the `eval_set` kwarg Coach actually passes (interfaces.py vs coach.py:305–308) so the Protocol matches reality.
- `inference_server.py`'s `@runtime_checkable` vs the style guide's prohibition: remove it if no isinstance check depends on it; otherwise keep and document the exception at the use site.

## R32. scripts/ reorganisation  **(JUDGE)**

Target: a reader opening `scripts/` sees operational tools, not 26 undifferentiated files.

- **Top level (operational):** `arena_run.py`, `arena_two_checkpoints.py`, `pentobi_benchmark.py`, `play_ttt.py`, `replay.py`, `setup_wandb_workspace.py`, `validate_jax_search.py`, `validate_inference_server.py`, `fetch_run_reports.sh`, `run_benchmark.sh`.
- **`scripts/benchmarks/`:** `benchmark.py`, `benchmark_phases.py`, `benchmark_jax_env.py`, `benchmark_selfplay_backends.py`, `benchmark_movegen.py`, `benchmark_predict_batch.py`, `benchmark_inference_server.py`, `bench_parallel.py`.
- **`scripts/profiling/`:** `mcts_profiling.py`, `profile_mcts_memory.py`, `profile_self_play.py`, `move_count_analysis.py`, `count_onboard_placements.py`, `diagnose_pentobi_losses.py`, `render_symmetry_snapshot.py`.
- **Delete** `profile_report.py` and `optimisation_progress_report.py`: both render reports from numbers hard-coded at measurement time (2026-06-05 baselines; parquets "no longer exist locally" per their own docstrings). Their outputs' story lives in `docs/plans/archive/full-cycle-optimisation.md` and `docs/research/profiling-report.md`; git history keeps the scripts.
- With the package installed (R6), delete every `PYTHONPATH=$PWD` / `PYTHONPATH=.` incantation from script docstrings — plain `uv run python scripts/benchmarks/benchmark.py` now works from anywhere in the repo. Fix the scratch-checkpoint writers (`benchmark_phases.py`, `bench_parallel.py`, `benchmark_inference_server.py`) to write into `temp/` instead of CWD while touching them.

## R33. run_configurations/ sort

Create `run_configurations/archive/` and move the superseded configs: the Windows/WSL-era (`blokus_pc_first/second`, `blokus_run1_taper`, `blokus_run2_bignet*`), the settled-experiment set (`bench_workers_*`, `profile_baseline*`, `blokus_linux_15`, `blokus_linux16_15`), and finished A/B arms (`ab_python_10` — keep `ab_jax_10`/`ab_gumbel_10` only if still referenced by docs, else archive all three). Keep current: `test_run`, `blokus_quicktest`, `blokus_mac_test`, `smoke_test*` (rename to `pipeline_check*`? — optional, per the no-"smoke-test" convention), `blokus_3gen`, `blokus_validation`, `blokus_scaled*`, `blokus_run3_overnight`, `blokus_jax_gumbel_30`, `blokus_gumbel_overnight`, `full_run`, `ttt_*`. Sweep references: `run_benchmark.sh` default (`profile_baseline.json` → point at a live config or move the default), `cli.py` default, README/docs examples.

## R34. Docs path sweep

Mechanical, grep-driven; the audit's sweep list: **README** (layout tree + prose), **CLAUDE/AGENTS** (protocol paths, `core/config.py:35` stale line-cite, jax paths, doc tree), **02-ALGORITHMS** (`core/{coach,mcts,self_play,arena,acceptance}.py`, `games/blokusduo/movegen_*`, `game.py`), **03-NEURAL-NETWORKS** (net source-file table cells), **04-BLOKUS-DUO** (`blokusduo/pieces.py`), **06-INTERFACES** (`games/blokusduo/*`), **07-DATA-STORAGE** (source-reference table + `Coach line NNN` cites — replace line numbers with anchors/method names so they can't rot again), **AI-CONTEXT** (light). Safe-list (no code paths): 01, 05, 08, 09, IDEAS, research/README, REMOTE-TRAINING — verify with `grep -rn "core/\|games/\|reporting/" docs/` at the end. Prose rewrites belong to R35–R38; this row is paths only.

## R35. README rewrite  **(JUDGE)**

The portfolio front door; currently a full backend behind reality. Fix:

- **Status**: Phase 6 text says the first scaled run is blocked on memory + WSL reliability — both resolved (resumable runs shipped, box on native Ubuntu, OOM work has its own completed plan by now). Roadmap checkbox "Sparse policy storage" is stale/contradictory (per audit: described as fixed *and* unchecked; supersede with the oom-hardening outcome).
- **Add the JAX backend as a headline phase**: GPU-native rules as int8 matmuls, mctx Gumbel search at n=64, strength-parity A/B, ~3.5–12× wall-clock (numbers from `docs/research/jax-pipeline-ab.md`) — this is the single most impressive engineering artefact in the repo and README doesn't mention it.
- Pentobi harness exists (`pentobi/` subpackage + `pentobi_benchmark.py`) — "not yet built" is false.
- New repo-layout tree (from D-section here), `uv run alphablokus` commands, CI badge, "no plan currently in flight" line corrected.

## R36. AGENTS.md canonical, CLAUDE.md symlink  **(JUDGE)**

Merge the two near-duplicates into one canonical AGENTS.md (starting from CLAUDE.md, the fresher fork: it has the JAX section and gotcha the AGENTS.md fork lacks), then `ln -s AGENTS.md CLAUDE.md` (git tracks the symlink). Content refresh: delete the fully obsolete "Critical path" section (every item done or contradicted — W&B done, symmetries done, `main.py` loading works) and replace with a live "Current focus" pointing at `docs/plans/`; fix the broken `move-gen-further-optimisation.md` reference (Gotcha 1); update gotcha paths to the new tree; regenerate the docs tree listing; remove the notebook gotcha (R3); update commands to `uv run alphablokus`.

## R37. AI-CONTEXT.md rewrite  **(JUDGE)**

Five audited falsehoods to purge: "no parallel MCTS yet" (F1 shipped), "move generation is the blocker" (done a year ago), "policy head FC layer is ~9M params" (conv head default, ~47K), "piece-orientation IDs have gaps" (OrientationCodec is contiguous 0–90 — contradicts 04-BLOKUS-DUO and code), "no optimizer state in checkpoints" (saved since `base_wrapper.py:457`), plus the pointer to `docs/plans/bug-fixes.md` as the "active bug list" (archived). Rewrite as a current extended-context doc: architecture rationale, the python↔jax parity contract, where to start reading (suggest: `cli.py` → `training/coach.py` → `selfplay/generate.py`).

## R38. Numbered-docs freshness  **(JUDGE)**

- **02-ALGORITHMS**: add a section on the JAX self-play backend + Gumbel search (it materially changes the policy target and root selection — an algorithms doc without it is incomplete); fix the broken `move-gen-further-optimisation.md` link (line 182).
- **03-NEURAL-NETWORKS**: add a short note on the inference-only jnp net + torch→jax checkpoint bridge; unify the draw-value statement (`1e-4` vs "~0").
- **06-INTERFACES**: the Pentobi GTP adapter + translation layer are **built** — update from scoping-doc to as-built + remaining scope (UI).
- **07-DATA-STORAGE**: fix the internal "6 datasets" vs ~13 actual hive dirs contradiction; add a note that the jax path's harvester writes the same self-play format.
- **08-TRAINING-ESTIMATES**: add a clear superseded banner and a pointer to the measured jax numbers in `docs/research/jax-pipeline-ab.md` (full re-measure is out of scope — say so).
- **09-COMPUTE-OPTIONS**: box is native Ubuntu, not WSL2; note GPU-native self-play changes the workload shape (CPU cores no longer the self-play cap on the jax path).

## R39. Guides + IDEAS + links  **(JUDGE)**

- **STYLE-GUIDE.md**: add a "Project layout" section (src/ package map, where new modules go, the registry rule from D5) and a "Tooling" section (mypy strictness contract, ruff format, CI expectations). Refresh `Last updated`.
- **REMOTE-TRAINING.md**: command updates (`uv run alphablokus`), any path changes.
- **IDEAS.md**: I1 (adaptive sim budget) and I3 (lean workers) are partially shipped per the audit — annotate what landed and what remains; add the warm-start note from R16 if desired.
- Repo-wide link check: every relative link in docs/ resolves (a `grep`-based pass or a one-off link-check script; fix `lean-self-play-workers.md`'s missing title heading while there).

## R40. Full verification

On the Mac: `uv run pytest` (full, incl. slow) with base extras, then with `--extra jax`; `uv run alphablokus --config run_configurations/test_run.json` end-to-end; a short jax CPU config (e.g. a 1-gen variant of `ab_jax_10`) end-to-end; `--report-only` against an **existing pre-refactor run directory** in `temp/` (proves parquet + report compatibility); load a pre-refactor checkpoint through `load_checkpoint` (proves checkpoint compatibility). Record results in the PR description.

## R41. Box validation

On gpu-linux: pull, `uv sync --extra jax-cuda`, quick GPU configs — one python-backend (`blokus_quicktest`) and one jax gumbel (`blokus_jax_gumbel_30` trimmed) — then `scripts/pentobi_benchmark.py` at a low level for a handful of games (the binary lives on the box), and `scripts/fetch_run_reports.sh` to confirm the operational loop end-to-end. This is the only place the CUDA path and Pentobi integration are truly exercised.

## R42. Archive this plan

Every row ✅ or `Deferred` with a reason → `git mv` to `docs/plans/archive/` in the closing commit, per PLAN-FORMAT. Add a "Scope additions" section first if execution picked up anything beyond these rows.
