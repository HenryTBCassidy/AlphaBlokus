# Pool-based BayesElo rating — fix the saturating gen-0 Elo curve

**What this covers.** Our current strength metric rates the live net against a *single frozen gen-0
baseline* over `elo_games_per_gen` games. Once the net wins ~100% of those games the number
saturates at the mathematical cap (~+1200 Elo) and can no longer distinguish, say, generation 41
from 43 — the curve flatlines even while the net keeps improving. DeepMind never did this: they
computed Elo from games *among a pool of players/checkpoints* and fit the ratings with **BayesElo**
(Bayesian logistic regression), so the curve keeps rising until genuine convergence. See
[`../research/deepmind-run-configs.md`](../../research/deepmind-run-configs.md) and the AlphaGo Zero /
AlphaZero papers (Elo "computed from evaluation games between different players", BayesElo,
`c_elo = 1/400`).

This plan delivers that in two parts:

- **Part A (E1–E7) — the primary deliverable: a post-hoc pool tournament + BayesElo curve.** It reads
  the per-generation checkpoints a run *already* saves (`Nets/accepted_<N>.pth.tar`), plays a sparse
  round-robin among them, fits BayesElo, and renders a proper rising Elo curve. It requires **no
  retraining** and can be run on any finished run's checkpoints (including the one training now). This
  is the exact DeepMind methodology and the thing to build first.
- **Part B (E8–E9) — optional, lower priority: an in-loop sliding-reference metric** so *future* runs
  don't saturate live, plus an optional external Pentobi anchor for an absolute scale.

**Prerequisites:** none for Part A — it operates on saved checkpoints. Part B touches the training
loop and must preserve resume behaviour.

**Ground truth for the current code** (verified file:line anchors the executor will edit):
- `src/alphablokus/evaluation/elo.py:8` — `compute_elo(wins, losses, draws) -> (elo_diff, raw_rate)`;
  saturation comes from clamping the score rate to `[0.001, 0.999]` → `elo_diff` caps at
  `±400·log10(0.999/0.001) ≈ ±1200`. **No dedicated test exists.**
- `src/alphablokus/training/coach.py:148-160` — frozen gen-0 baseline created/saved as
  `Nets/elo_baseline.pth.tar`, reloaded into `self.elo_baseline_net`.
- `src/alphablokus/training/coach.py:484-519` — `_evaluate_elo_vs_baseline(generation)`; plays
  `elo_games_per_gen` games, calls `compute_elo`, then `self.metrics.log_elo(...)`.
- `src/alphablokus/training/coach.py:521-569` — serial (`Arena`) and parallel
  (`run_two_player_games_parallel`) game runners returning `(new_wins, baseline_wins, draws)`.
- `src/alphablokus/evaluation/arena.py:188` — `Arena.play_games(num, ...) -> (p1_wins, p2_wins, draws, records)` (the reusable W/L/D engine; rounds `num` down to even, swaps colours at halftime).
- `src/alphablokus/parallel/pool.py:553` — `run_two_player_games_parallel(config, generation, checkpoint_a_path, checkpoint_b_path, num_games, num_workers, *, phase, record=False, ...) -> (a_wins, b_wins, draws, records)`. Checkpoint paths are filenames relative to `config.net_directory`; `phase=PHASE_ELO` (pool.py:95); `num_games >= 2`.
- Checkpoints (all in `RunConfig.net_directory`): `accepted_<N>.pth.tar` + `best.pth.tar` (coach.py:292-293), `rejected_<N>.pth.tar` (coach.py:296), `latest.pth.tar` (coach.py:343), `elo_baseline.pth.tar` (coach.py:154).
- `src/alphablokus/games/base_wrapper.py:643` `save_checkpoint(filename)` / `:662` `load_checkpoint(filename)` — `load_checkpoint` resolves `net_directory / filename`, so passing an **absolute path** overrides the folder (used by `scripts/arena_two_checkpoints.py:92`).
- `src/alphablokus/registry.py:45` `instantiate_game_and_network(config) -> (IGame, INeuralNetWrapper)` — the composition-root entry point; returns a fresh random-init wrapper, caller then `load_checkpoint(...)`.
- Storage: `src/alphablokus/storage/metrics.py:640` `log_elo(...)` writes rows to `EloRatings/generation=N/elo.parquet` (flush at metrics.py:982-990); config dir property `RunConfig.elo_ratings_directory` (config.py:510).
- Reporting: `src/alphablokus/reporting/charts.py:552` `make_elo_plot(elo_data, arena_data)` (title hardcoded "Elo Rating vs Frozen Gen-0 Baseline", charts.py:660); wired in `report.py:254` (load), `:286` (build), `:348-349` (render).
- Config: `RunConfig` frozen dataclass (config.py:251); `elo_games_per_gen: int = 50` (config.py:321), `elo_baseline_rating: int = 400` (config.py:339). Nested config blocks use `field(default_factory=...)` (e.g. config.py:308) and are auto-hydrated by `dataclass_wizard.fromdict` in `load_args` (config.py:545-560) — a new nested block needs no custom parsing.
- Script bootstrap pattern: `scripts/arena_run.py` / `scripts/arena_two_checkpoints.py`, invoked `uv run python -m scripts.<name>`.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| E1 | Unit-test the existing `compute_elo` (lock behaviour + document the ~1200 saturation) | 30 min | High | ✅ |
| E2 | `evaluation/rating.py`: self-contained BayesElo (Bradley–Terry MM) estimator + tests | 2.5 h | High | ✅ |
| E3 | `TournamentConfig` nested dataclass + `tournament_directory` property in config.py | 45 min | High | ✅ |
| E4 | Sparse pairing-schedule generator (consecutive + exponential back-refs) + connectivity test | 1 h | High | ✅ |
| E5 | `scripts/tournament_elo.py`: enumerate checkpoints → run pairings → W/L/D matrix → BayesElo → write parquet/JSON | 3 h | High | ✅ |
| E6 | Reporting: `make_tournament_elo_plot` + wire into `report.py`; relabel old chart "(saturates)" | 1.5 h | High | ✅ |
| E7 | Docs: methodology note in `docs/research/`, update `docs/05-EVALUATION.md`, report caveat | 45 min | Medium | ✅ |
| E8 | *(Optional)* In-loop sliding-reference Elo in coach (rate gen N vs recent net, accumulate) | 3 h | Low | Deferred |
| E9 | *(Optional)* Pentobi as an anchored external player for an absolute scale | 2 h | Low | Deferred |

> Part A = E1–E7 (do these; they fully deliver the DeepMind-style curve on existing checkpoints).
> Part B = E8–E9 (forward-looking polish; safe to defer). Mark deferred rows `Deferred` with a note.
>
> **E8/E9 deferred (2026-07-05).** Part A (E1–E7) is complete and fully delivers the stated goal — a
> non-saturating DeepMind-style pool BayesElo curve computed post-hoc from any finished run's saved
> checkpoints, with no retraining. E8 (in-loop sliding-reference Elo) and E9 (Pentobi as an anchored
> external player) are forward-looking polish that touch the training loop / require the Pentobi
> adapter running during the tournament; they add no capability Part A lacks for the current goal.
> Both are cleanly re-openable from this archived plan when a *live* non-saturating number or an
> *absolute* cross-run scale is wanted. E9's hook already exists: `fit_bayeselo`'s `anchor`/
> `anchor_rating` accept any pool player, so folding Pentobi in is additive.

---

## E1. Unit-test the existing `compute_elo`

**Current state.** `evaluation/elo.py:8` is a pure 22-line function with **no test**. Before replacing
how it's *used*, pin its behaviour so nothing regresses (the function itself stays — it's still fine
for a single pairwise comparison; what changes is that we stop feeding it a fixed weak anchor).

**Fix.** Add `tests/evaluation/test_elo.py` (real values, no mocks — matches the style guide):
- `compute_elo(1,0,0)` and `(20,0,0)` both return `elo_diff ≈ +1199.8` → **demonstrate the saturation**
  (add a comment: this is exactly why the vs-gen-0 curve flatlines; the pool tournament fixes it).
- `compute_elo(0,0,0) == (0.0, 0.0)` (the `total==0` guard).
- Symmetry: `compute_elo(w,l,d)[0] == -compute_elo(l,w,d)[0]`.
- A mid-range check: `compute_elo(3,1,0)` → rate 0.75 → `400·log10(0.75/0.25) ≈ +190.8`.

**Effort:** 30 min.

---

## E2. `evaluation/rating.py` — self-contained BayesElo estimator

**Why.** Pool ratings need a maximum-likelihood fit over a whole W/L/D matrix, not pairwise
`compute_elo`. Implement the standard **Bradley–Terry model fit by Minorization–Maximization (MM)**
(Hunter 2004; the algorithm behind Rémi Coulom's BayesElo). No new dependency — pure numpy (already a
dep; scipy also available).

**Model.** For players with Elo ratings `R_i`, define `γ_i = 10^(R_i / 400)`. Probability i beats j:
`P(i > j) = γ_i / (γ_i + γ_j)`. Draws are counted as half a win to each side (standard BayesElo
handling; do **not** model draws explicitly — overkill here).

**Inputs.** A symmetric-ish tally: for each unordered pair `(i, j)`, the number of games `n_ij` and
i's score `s_ij` (wins + 0.5·draws, from i's perspective). Equivalently pass a `wins[i][j]` matrix
(count of games i won against j) and a `draws[i][j]` matrix.

**MM update** (iterate to convergence). Let `W_i = Σ_j (wins_ij + 0.5·draws_ij)` be i's total score.
Repeat until `max|Δln γ| < 1e-6` (cap ~1000 iters):
```
for each player i:
    denom_i = Σ_j  n_ij / (γ_i + γ_j)      # n_ij = total games between i and j
    γ_i ← W_i / denom_i
normalise: divide all γ by their geometric mean (fixes the scale/gauge freedom)
```
Then `R_i = 400 · log10(γ_i)`.

**Regularisation (the "Bayes" in BayesElo) — required.** A checkpoint that goes 100% or 0% against all
its opponents pushes γ → ∞/0 and the fit diverges (early gens can get swept; the strongest net may
never lose). Add a **virtual prior game**: give every player one virtual draw against a fixed
rating-anchor pseudo-player at γ=1 (R=0). Concretely, add `prior_games` (default 2.0, configurable)
split as virtual draws vs the anchor into `W_i` and `denom_i`. This keeps every rating finite and is
mathematically the MAP estimate under a weak prior.

**Anchoring.** After the fit, shift all ratings so a chosen anchor checkpoint (default: gen-0) sits at
`anchor_rating` (default 0; or reuse `elo_baseline_rating=400` for continuity with the old chart).
This makes the curve comparable across a single run. (Cross-run comparability still requires a shared
external anchor — see E9.)

**API** (`src/alphablokus/evaluation/rating.py`):
```python
@dataclass(frozen=True)
class RatingResult:
    ratings: dict[str, float]        # player id -> Elo
    iterations: int
    converged: bool

def fit_bayeselo(
    players: list[str],
    wins: dict[tuple[str, str], int],   # (i, j) -> games i beat j
    draws: dict[tuple[str, str], int],  # (i, j) -> drawn games (store once per unordered pair)
    *,
    prior_games: float = 2.0,
    anchor: str | None = None,
    anchor_rating: float = 0.0,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> RatingResult: ...
```

**Tests** (`tests/evaluation/test_rating.py`, no mocks):
- **Recovery:** synthesise games from known ratings (e.g. players at 0/100/200/300 Elo, sample
  outcomes deterministically via a seeded rng or use expected win counts), fit, assert recovered
  ratings match the ground-truth *ordering* and pairwise gaps to within ~30 Elo after anchoring.
- **No saturation:** an undefeated top player and a winless bottom player both get **finite** ratings
  (contrast with `compute_elo`'s ±1200 clamp — this is the whole point).
- **Anchoring:** `anchor="gen0"` puts gen0 exactly at `anchor_rating`.
- **Determinism:** same input → same output.

**Effort:** 2.5 h.

---

## E3. `TournamentConfig` + `tournament_directory`

**Fix.** Add a frozen nested dataclass to `config.py` (mirroring `JaxSelfPlayConfig` at config.py:89 /
`TrainingPerfConfig` at config.py:206) and hang it off `RunConfig` with `field(default_factory=...)`
(like config.py:308). No custom parsing needed — `dataclass_wizard.fromdict` (config.py:560) hydrates
it from a nested JSON block automatically.
```python
@dataclass(frozen=True)
class TournamentConfig:
    games_per_pairing: int = 30       # games each checkpoint pair plays (>=2; rounded even by Arena)
    back_ref_offsets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)  # who each gen plays (generations back)
    include_first_last: bool = True   # always pair every node with gen-0 and the final gen
    prior_games: float = 2.0          # BayesElo regularisation
    anchor_rating: float = 0.0        # gen-0 pinned here
    max_checkpoints: int | None = None  # subsample (e.g. every 2nd gen) to cap cost; None = all
```
Add a directory property next to `elo_ratings_directory` (config.py:510):
```python
@property
def tournament_directory(self) -> Path:
    return self.run_directory / "Tournament"
```
Add a field on `RunConfig`: `tournament: TournamentConfig = field(default_factory=TournamentConfig)`.

**Test:** extend `tests/test_config.py` — a JSON with a `"tournament": {...}` block round-trips into
`TournamentConfig`; defaults apply when the block is absent.

**Effort:** 45 min.

---

## E4. Sparse pairing-schedule generator

**Why.** A full round-robin over K checkpoints is O(K²) pairs (60 gens → 1,770 pairs) — far too many
games, since each pairing plays MCTS games via the arena (much slower than jax self-play). Instead
build a **sparse but connected** schedule: each checkpoint plays its neighbours at exponentially
spaced offsets, which keeps the comparison graph connected (a requirement for a well-conditioned
BayesElo fit) at O(K·log K) pairs.

**Fix.** A pure function (put it in `evaluation/rating.py` or a small `evaluation/tournament.py`):
```python
def build_pairings(
    checkpoint_ids: list[str],           # ordered by generation, e.g. ["gen0","gen1",...,"gen59"]
    back_ref_offsets: tuple[int, ...],
    include_first_last: bool,
) -> list[tuple[str, str]]:
    # for each i, pair (i, i-off) for off in offsets if in range; dedupe unordered pairs;
    # if include_first_last, also pair every node with checkpoint_ids[0] and [-1].
```
For K=60 with offsets (1,2,4,8,16,32): ~6 pairs/node → ~300–360 unordered pairs. At
`games_per_pairing=30` that's ~10k games — minutes on the 5090, longer on CPU; both configurable.

**Tests** (`tests/evaluation/test_tournament.py`):
- Every checkpoint appears in ≥1 pairing (no isolated nodes).
- The pairing graph is **connected** (BFS/union-find over the pairs) — this is what makes the
  BayesElo fit well-posed.
- No duplicate unordered pairs; no self-pairs.

**Effort:** 1 h.

---

## E5. `scripts/tournament_elo.py` — the post-hoc tournament tool

**This is the primary deliverable.** A standalone script (run `uv run python -m scripts.tournament_elo
--config <run.json> [--run-dir <path>]`) that turns a finished run's saved checkpoints into a proper
Elo curve. Template it on `scripts/arena_run.py` / `scripts/arena_two_checkpoints.py`.

**Steps inside the tool:**
1. `config = load_args(args.config)`.
2. Enumerate checkpoints in `config.net_directory`: glob `accepted_*.pth.tar`, sort by the integer
   generation, prepend `elo_baseline.pth.tar` as the "gen0" anchor node (it *is* the gen-0 net). Apply
   `tournament.max_checkpoints` subsampling if set (e.g. take every ⌈K/max⌉-th). Result: an ordered
   list of `(player_id, absolute_checkpoint_path)`.
3. `build_pairings(...)` (E4).
4. For each pairing, play `games_per_pairing` games between the two checkpoints:
   - **Reuse the existing engines.** If `config.num_parallel_workers > 1`, call
     `run_two_player_games_parallel(config, generation=0, checkpoint_a_path=<a>, checkpoint_b_path=<b>,
     num_games=games_per_pairing, num_workers=config.num_parallel_workers, phase=PHASE_ELO,
     record=False)` (pool.py:553). Note it takes checkpoint **filenames relative to
     `config.net_directory`** — pass the basenames (they all live there). Otherwise build two
     `NetworkPlayer`s (temp=0) via `registry.instantiate_game_and_network` + `load_checkpoint(abs path)`
     and use `Arena(pa, pb, game).play_games(games_per_pairing)` (arena.py:188).
   - Record `(a_wins, b_wins, draws)` into `wins[(a,b)] += a_wins`, `wins[(b,a)] += b_wins`,
     `draws[(a,b)] += draws`.
5. `fit_bayeselo(players, wins, draws, prior_games=..., anchor="gen0", anchor_rating=...)` (E2).
6. **Write results** to `config.tournament_directory`:
   - `tournament_ratings.parquet`: columns `generation, rating, n_games, n_pairings` (one row per
     checkpoint; `generation` parsed from the id, gen0 = 0).
   - `tournament_raw.json`: the pairing results (the W/L/D matrix) so the fit can be re-run/audited
     without replaying games.
7. Log a summary table (loguru) and the final rating of the last generation.

**Cost guardrail.** `log()` the total number of games to be played and the estimated time before
starting; honour `--dry-run` to print the schedule + game count and exit. Never silently play a
huge number of games.

**Test** (`tests/test_tournament_elo.py`, TicTacToe, real objects): create 2–3 tiny nets (train a few
steps or just random-init distinct seeds), save them as `accepted_1/2/3.pth.tar` in a tmp
`net_directory`, run the tool's core (factor the logic into an importable `run_tournament(config)` so
the test doesn't shell out), assert it produces a ratings dict covering all checkpoints with finite
values and writes the parquet. Keep `games_per_pairing` tiny (e.g. 2) for speed.

**Effort:** 3 h.

---

## E6. Reporting — render the pool curve

**Current state.** `charts.py:552` `make_elo_plot` plots the saturating `elo_rating` column; title is
hardcoded "Elo Rating vs Frozen Gen-0 Baseline" (charts.py:660). `report.py:254/286/348` load/build/
render it.

**Fix.**
1. Add `make_tournament_elo_plot(tournament_data: pd.DataFrame)` to `charts.py` — line chart of
   `rating` vs `generation` from `tournament_ratings.parquet`; hover shows `n_games`. Title:
   "Pool Elo (BayesElo tournament)".
2. In `report.py`: load `tournament_data = _load_metrics(config.tournament_directory)` guarded for the
   file's absence (older runs won't have it); build + render the new figure **above** the old one.
3. Relabel the old chart title to "Elo vs Frozen Gen-0 Baseline (saturates once net ≫ gen-0)" so a
   reader understands why it flatlines and looks at the pool curve instead. Keep it — it's still a
   fine early-training signal.

**Test:** extend the reporting smoke test (or add `tests/reporting/test_tournament_chart.py`) — build
the figure from a small synthetic `tournament_ratings.parquet` and assert it returns a Plotly figure
with the expected number of traces. (`reporting/` is currently thinly tested; a render-from-fixture
test also closes that gap.)

**Effort:** 1.5 h.

---

## E7. Docs

- Add `docs/research/pool-elo-methodology.md`: one page on why vs-gen-0 saturates (the ±1200 clamp),
  how BayesElo/MM works, the sparse-pairing rationale, and how this matches DeepMind (cite
  `deepmind-run-configs.md` and the papers). This doubles as the "I measured strength the way DeepMind
  did" writeup material.
- Update `docs/05-EVALUATION.md` to describe the pool tournament as the canonical strength curve and
  the vs-gen-0 number as an early-training-only signal.
- Note in the report/README that the pool Elo is the metric to read.

**Effort:** 45 min.

---

## E8. *(Optional, Low)* In-loop sliding-reference Elo

**Why optional.** E1–E7 already produce the DeepMind-style curve post-hoc from checkpoints, which is
enough for the stated goal. This item only helps if you want a *non-saturating live* number during
training rather than after.

**Fix.** In `coach.py`, alongside (or replacing) `_evaluate_elo_vs_baseline`, rate generation N against
a **recent** reference instead of frozen gen-0 — e.g. the accepted net from `R` generations ago
(config: `elo_reference_lag`, default 5), or the previous accepted net. Play `elo_games_per_gen`
games, `compute_elo` gives the *delta* vs that reference, and you accumulate an absolute running rating
`R_N = R_ref + delta`. Log it as a new series (`log_elo` already stores `elo_diff`; add a
`rolling_rating` column or a parallel `log_rolling_elo`). Because consecutive nets are close in
strength, the win rate stays in the resolvable band and the accumulated curve keeps rising.

**Resume care.** The running rating must be reconstructable on `--resume` — persist the per-generation
`(reference_id, delta, rolling_rating)` (it's already in the EloRatings parquet if you add the column),
and rebuild the accumulator from the parquet on resume rather than from memory. Add a test to
`tests/integration/test_resume.py` asserting the rolling rating continues correctly across a resume.

**Caveat to document:** accumulated chained Elo drifts (errors compound); the E5 post-hoc BayesElo fit
over the full pool is the authoritative curve. The live chained number is a monitoring convenience.

**Effort:** 3 h.

---

## E9. *(Optional, Low)* Pentobi as an external anchor

**Why optional.** Gives the pool curve an *absolute* meaning (comparable across runs and to "real"
strength) instead of a within-run relative scale.

**Fix.** Add Pentobi (at one or more fixed levels) as extra player node(s) in the tournament: play a
sample of checkpoints against Pentobi via the existing GTP harness
(`scripts/pentobi_benchmark.py` / `games/blokusduo/pentobi/`), fold those W/L/D into the same matrix,
and **anchor the BayesElo fit on Pentobi-level-K** at its known/assumed rating instead of on gen-0.
This is how DeepMind grounded their scale on human/AlphaGo-Fan references. Heavier because it needs the
Pentobi adapter running during the tournament, so it's separate from the core (E5) which stays
self-contained.

**Effort:** 2 h.

---

## Notes for the executing agent

- **Style contract:** full type annotations (mypy `--strict`), `ruff` lint + format, frozen dataclasses,
  loguru (`{}` placeholders, no `print`), Google docstrings, `from __future__ import annotations`, real
  objects in tests (no mocks). CI runs ruff + format check + mypy + base + jax test jobs — keep it green.
- **Registry rule:** only `registry.py` names concrete game/net classes. The tool should go through
  `registry.instantiate_game_and_network` (or reuse `scripts/arena_run.py`'s loader), not import
  `games.*` directly.
- **One commit per checklist row**; tick the Done column the moment each row lands (don't batch-tick).
- **Validate on the real artefacts:** once `blokus_cloud_60` finishes, run `scripts/tournament_elo.py`
  on its `Nets/accepted_*.pth.tar` and confirm the curve rises where the vs-gen-0 curve flatlined —
  that's the acceptance test for the whole plan.
- **Archive on completion:** when every row is ✅ or `Deferred`, `git mv` this file to
  `docs/plans/archive/` in the same commit (per PLAN-FORMAT.md).
