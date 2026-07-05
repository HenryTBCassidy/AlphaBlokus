# Pool BayesElo — measuring strength the way DeepMind did

**TL;DR.** Rating every generation against a *single frozen gen-0 baseline* saturates: once the net
wins ~100% of those games the number pins at the mathematical ceiling (~±1200 Elo) and can no longer
tell a strong net from a much stronger one. DeepMind never did this — they computed Elo from games
*among a pool of checkpoints* and fit one consistent rating per player with **BayesElo**. This project
now does the same, post-hoc, over the checkpoints a run already saves. Tool: `scripts/tournament_elo.py`;
estimator: `src/alphablokus/evaluation/rating.py`; schedule: `src/alphablokus/evaluation/tournament.py`.

## Why vs-gen-0 saturates

`evaluation/elo.py:compute_elo` computes a pairwise Elo difference from a win/loss/draw tally:

```
score_rate = (wins + 0.5·draws) / games        # clamped to [0.001, 0.999]
elo_diff   = 400 · log10(score_rate / (1 − score_rate))
```

The clamp is necessary (a 100% score would send `log10` to +∞), but it means the largest expressible
difference is `400·log10(0.999/0.001) ≈ 1199.8`. A net that goes 50-0 against gen-0 gets the *same*
number as one that goes 1-0 — both saturate. Since a well-trained AlphaZero net beats its random
gen-0 self ~100% of the time within a few dozen generations, the vs-gen-0 curve flatlines exactly when
you most want to see continued improvement. It's a fine **early-training** signal and a terrible
late-training one.

## What DeepMind did

AlphaGo Zero and AlphaZero report Elo "computed from evaluation games between different players" and
fit with BayesElo (Rémi Coulom's Bradley–Terry tool), with the logistic scale `c_elo = 1/400`. The
key difference from a fixed anchor: each checkpoint is compared against *nearby* checkpoints where the
win rate stays in the resolvable band (not pinned at 0/100%), and a single global fit reconciles all
those local comparisons into one consistent scale. The curve then keeps climbing until the net stops
improving. See [`deepmind-run-configs.md`](deepmind-run-configs.md) for how their run configs map onto
ours.

## The Bradley–Terry model and the MM fit

Give player *i* an Elo rating `R_i` and define `γ_i = 10^(R_i / 400)`. The model's probability that
*i* beats *j* is

```
P(i > j) = γ_i / (γ_i + γ_j)
```

which is exactly the logistic Elo expectation. Draws count as half a win to each side (standard
BayesElo handling — we don't model draws explicitly).

We fit the `γ_i` by **maximum likelihood via Minorization–Maximization** (Hunter, 2004 — the algorithm
behind BayesElo). Let `W_i` be player *i*'s total score (wins + half-draws) and `n_ij` the number of
games between *i* and *j*. The MM update iterated to convergence is:

```
for each player i:
    denom_i = Σ_j  n_ij / (γ_i + γ_j)
    γ_i ← W_i / denom_i
renormalise all γ by their geometric mean      # fixes the scale/gauge freedom
```

then `R_i = 400 · log10(γ_i)`. Our implementation does the update in a vectorised Jacobi sweep
(compute every denominator from the previous `γ`, update all together), which converges to the same
fixed point.

### The "Bayes" — a weak prior that keeps ratings finite

A checkpoint that goes 100% (or 0%) against every opponent it plays would push `γ → ∞` (or `0`) under
a pure MLE, and the fit diverges — early gens get swept, and the strongest net may never lose. BayesElo
adds a **virtual prior game**: every player plays `prior_games` virtual draws against a fixed
pseudo-player at `γ = 1` (R = 0). Concretely this adds `0.5·prior_games` to `W_i` and
`prior_games/(γ_i + 1)` to `denom_i`. That's the MAP estimate under a weak prior; it guarantees every
rating is finite. `prior_games` defaults to 2.0 (`TournamentConfig.prior_games`).

### Anchoring

After fitting, all ratings are shifted so a chosen anchor (default: the gen-0 `elo_baseline`) sits at
`anchor_rating`. This makes the curve comparable *within* a run. Cross-run comparability needs a shared
external anchor — e.g. folding Pentobi at a known level into the pool (the deferred E9 extension).

## The sparse pairing schedule

A full round-robin over K checkpoints is O(K²) pairings (60 gens → 1,770), and each pairing plays real
MCTS arena games — far slower than jax self-play. Instead we pair each checkpoint with a handful of
earlier ones at **exponentially spaced offsets** (`back_ref_offsets = (1, 2, 4, 8, 16, 32)`), plus
optionally every checkpoint with gen-0 and the final gen (`include_first_last`). This keeps the
comparison graph **connected** — the precondition for a well-conditioned BayesElo fit — at O(K·log K)
pairings (60 gens → ~300, ~6 per node). Connectivity is unit-tested via union-find in
`tests/evaluation/test_tournament.py`. At `games_per_pairing = 30` that's ~10k games: minutes on a fast
GPU, longer on CPU; all configurable via `TournamentConfig`.

## How to run it

```
uv run python -m scripts.tournament_elo --config <run.json>            # play + fit + write
uv run python -m scripts.tournament_elo --config <run.json> --dry-run  # schedule + game count only
```

Outputs land in `<run>/Tournament/`: `tournament_ratings.parquet` (one row per checkpoint:
generation, rating, n_games, n_pairings) and `tournament_raw.json` (the raw W/L/D matrix, so the fit
can be re-run or audited without replaying games). The HTML report renders the rising pool curve above
the saturating vs-gen-0 chart.

**Acceptance test for the approach:** run it on a finished run whose vs-gen-0 curve has flatlined and
confirm the pool curve keeps rising where the old one pinned at ~1200. That is the DeepMind
methodology reproduced on our own checkpoints.

## References

- Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero), 2017.
- Silver et al., *A general reinforcement learning algorithm …* (AlphaZero), 2018.
- Hunter, *MM algorithms for generalized Bradley–Terry models*, Annals of Statistics, 2004.
- Coulom, *Whole-History Rating / BayesElo*.
- [`deepmind-run-configs.md`](deepmind-run-configs.md) — DeepMind run configs mapped to ours.
