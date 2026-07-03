# JAX Self-Play Pipeline — GPU-Native Generation Behind a Backend Flag

Successor to the de-risk spike (`docs/plans/archive/jax-spike.md`, findings:
`docs/research/jax-spike-findings.md`). This plan makes GPU-native JAX self-play the
production generation path while **keeping the existing Python/torch pipeline fully runnable**
— selected by a config flag, never deleted — for redundancy, cross-checking, and benchmarking.

**Scope: speed enhancement only.** Same algorithm (AlphaZero: PUCT search, root Dirichlet,
temperature schedule, visit-count policy targets, outcome value targets), same training step,
same buffer/storage/reporting/eval machinery, same run-config surface. Bit-level identity with
the Python path is impossible (different RNG streams, fp arithmetic, batched search order), so
the fidelity contract is **same hyperparameter semantics + statistically equivalent training**,
enforced by the validation steps (G4, G8). Known, accepted semantic deltas are listed in
"Behavioural fidelity contract" below — anything not on that list appearing in A/B results is a
bug.

> **Completed 2026-07-03.** All rows ✅. Results: search-agreement in the G4 note,
> throughput in the G7 note, the three-arm A/B + flip decision in
> [`docs/research/jax-pipeline-ab.md`](../../research/jax-pipeline-ab.md). Headline: exact
> example-format compatibility; jax-PUCT trains a net that beats the python baseline
> head-to-head (60.5%); Gumbel n=64 trains at parity strength in 20 min vs python's 71 min
> (3.5×; ~12× at production net size). Production Blokus = jax + Gumbel
> (`blokus_jax_gumbel_30.json`).

Prerequisites: PR #19 (the spike) merged — this plan builds directly on
`experiments/jax_spike/` and the `jax`/`jax-cuda` extras.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| G1 | Backend seam: `selfplay_backend` config field + Coach dispatch + `PythonSelfPlayBackend` wrapping the existing path (pure refactor, zero behaviour change) | 1 day | High | ✅ |
| G2 | Promote spike code: `experiments/jax_spike/` → `games/blokusduo/jaxenv/`; tests → `tests/test_blokusduo/`; benchmark → `scripts/benchmark_jax_env.py` | 0.5 day | High | ✅ |
| G3 | Inference-only JAX net: exact port of `AlphaBlokusDuo` (plain jnp, eval-mode BN), torch→jax weight converter, forward-equivalence test, bf16 option | 2–3 days | High | ✅ |
| G4 | Search core: mctx `muzero_policy` behind a top-K action-compaction layer; PUCT/noise/temp parameter mapping; validation vs Python MCTS on fixed positions; VRAM/throughput sweep → choose K, B, S defaults | 4–5 days | High | ✅ |
| G5 | Batched actor loop (pgx auto-reset pattern): temp schedule per slot, action sampling, game harvesting, value backfill, host-side symmetry augmentation, `ProcessedExample` assembly | 3–4 days | High | ✅ |
| G6 | `JaxSelfPlayBackend`: Coach integration, per-generation torch→jax weight sync, stats reporting, 1-generation integration test | 2 days | High | ✅ |
| G7 | Backend-vs-backend throughput benchmark (games/s, sims/s, VRAM) on the box; extend the spike harness | 1 day | High | ✅ |
| G8 | A/B validation: ~10-gen training runs (python-PUCT vs jax-PUCT vs jax-Gumbel), same config; compare Elo trajectory, final-net head-to-head arena, Pentobi L1; research note | 1 day + box time | High | ✅ |
| G9 | Flip Blokus default to `selfplay_backend: "jax"`; docs (CLAUDE.md, README, REMOTE-TRAINING, training estimates) | 0.5 day | High | ✅ |
| G10 | Gumbel mode (`search_policy: "gumbel"`, opt-in): mctx `gumbel_muzero_policy`, n≈32–64 sims; validation run — the known ~6× sims lever | 1–2 days | Medium | ✅ |

Total: ~3–4 working weeks solo. G1→G6 are sequential; G7 depends on G6; G8 on G7; G10 is
independent after G6 and can be dropped from this plan without affecting the flip (G9).

---

## Design decisions (made; flagging for review)

1. **Old path stays in place, selected by config — no `legacy/` directory move.** A directory
   move churns every import, breaks `git blame`, and implies deprecation. Instead:
   `RunConfig.selfplay_backend: Literal["python", "jax"] = "python"` (flipped to `"jax"` for
   Blokus configs only in G9, after A/B validation). The Python path stays the only path for
   TicTacToe and remains importable without jax installed (lazy imports at the dispatch seam).
   Nothing is deleted anywhere in this plan.
2. **JAX is inference-only; torch remains the learner.** Training step, replay buffer, storage,
   reporting, checkpoints, resume — all unchanged. The jax side needs the net only for leaf
   evaluation, and eval-mode BatchNorm is a fixed affine transform, so the port is plain-jnp
   pytree params (no flax/haiku dependency). Weights flow torch→jax once per generation
   (~2.7M params, milliseconds). This keeps the diff small, keeps "no behavioural change" true
   for training, and keeps every checkpoint compatible with the Pentobi/arena machinery.
3. **mctx with a top-K action-compaction layer.** mctx allocates six dense `[B, S+1, A]`
   arrays (24 B per node-action; confirmed from `mctx/_src/tree.py`). At A=17,837 that is
   ~428 KB per node per game — dense 400-sim search caps B at ~24–32 on 8 GB and there is no
   built-in sparse/large-action support (maintainer declined; issues #87/#108). Fix: at each
   node expansion keep only the **top-K legal actions by prior** (K configurable, ~128–256),
   store their global action ids in the node embedding, and let mctx search the compact
   0..K-1 space; `recurrent_fn` maps compact→global before stepping the env. Memory drops
   ~70× (K=256, B=1024, S=400 ≈ 2.7 GB). At 392 sims over ≥17k actions the Python search
   visits ≪K distinct actions per node anyway, so the pruning bites only where PUCT would
   essentially never go; the root applies Dirichlet noise **before** top-K selection so noise
   can promote low-prior moves into the searched set. G4 validates all of this against the
   Python MCTS and picks K empirically (with the dense small-B configuration measured as a
   fallback).
4. **Actor loop copies pgx's pattern** (`examples/alphazero/train.py`): B parallel game slots
   inside a `lax.scan`/`fori_loop`, finished games auto-reset in place, fixed step horizon,
   trailing-truncation mask; harvested to host per generation and converted to the *exact*
   existing `ProcessedExample` format — `(compact int8 board, sparse policy, value)` per
   position, transpose-augmented (2×), grouped per game — so `Coach`, the rolling buffer,
   `SelfPlayStore`, and the lazy Dataset are untouched.
5. **Eval stays on the Python path entirely.** Arena gating, Elo, Pentobi benchmarking all use
   `NetworkPlayer`/GTP as today (confirmed independent of generation). This is deliberate
   redundancy: every generation, the jax-generated net is exercised by the Python search —
   a continuous cross-check of the weight bridge and net port.

## Behavioural fidelity contract (accepted deltas, documented up front)

| Delta | Why | Mitigation / check |
|---|---|---|
| RNG streams differ (jax PRNG vs numpy/torch) | unavoidable | seeds still control reproducibility *within* the jax path |
| No cross-move tree accumulation (Python MCTS keeps its node table across moves within a game; mctx searches each move fresh) | mctx design; subtree reuse exists only in third-party forks (mctx-az) | quantified in G4 position-level comparison; revisit only if A/B shows a strength gap |
| PUCT details: mctx uses `pb_c = pb_c_init + log((N+pb_c_base+1)/pb_c_base)` and a Q-normalising `qtransform`; ours is raw-Q, constant `cpuct`, unvisited-Q=0 | mctx API | map `pb_c_init=cpuct`, `pb_c_base` large (log term → 0), and use an identity-style qtransform to match raw-Q semantics; A/B the default qtransform as a variant in G4 |
| Search over top-K prior actions per node, not all 17,837 | tree memory (see decision 3) | K sweep + agreement metrics in G4; root noise applied pre-top-K |
| Flat sim budget (no branching taper) inside jit | dynamic per-move sim counts don't jit | taper was a CPU-side speed hack; flat S is the original algorithm. Config stays honest: jax backend reads `num_mcts_sims`, ignores `sim_schedule` with a logged warning |
| Policy target = visit distribution from batched search (values fp32/bf16) | fp + batching order | statistically equivalent; G8 A/B is the arbiter |

---

## G1. Backend seam (pure refactor)

Add `selfplay_backend: Literal["python", "jax"] = "python"` to `RunConfig` (config.py, with
the other optional fields; dataclass-wizard picks it up from JSON automatically). Introduce
`core/selfplay_backend.py`:

```python
class ISelfPlayBackend(Protocol):
    def generate(self, generation: int) -> tuple[list[GameExamples], list[MCTSEpisodeStats]]: ...
```

> **As built:** lighter than the Protocol sketch — the existing serial/parallel runners stay
> as Coach methods (they are metrics-coupled anyway); the seam is a three-way dispatch in
> `_learn_loop` plus `Coach._run_self_play_jax`, which mirrors `_run_self_play_parallel`'s
> exact contract (save checkpoint → call `core.jaxplay.backend.generate_self_play_games(config,
> generation, checkpoint_path)` → log per-game stats through a shared `_log_self_play_stats`
> helper that de-triplicates the logging). Jax import is deferred inside the branch. Config
> field lands with parse tests (`tests/test_core/test_selfplay_backend_config.py`); the
> 1-worker-vs-4-worker determinism test pins that the python path is untouched.

## G2. Promote the spike code

`experiments/jax_spike/{tables,kernels,bridge}.py` → `games/blokusduo/jaxenv/` (same file
names; module docstrings updated; `experiments/` removed **after** the move — it's a move, not
a deletion). `tests/test_jax_spike/` → `tests/test_blokusduo/test_jaxenv_{tables,parity,step_parity,scoring}.py`
(same importorskip guard). `experiments/jax_spike/benchmark.py` → `scripts/benchmark_jax_env.py`
(matching the `scripts/benchmark_*.py` convention). Fix references in the findings doc/archived
plan. `mctx` joins the `jax`/`jax-cuda` extras here (it's pure-Python over jax, one dep).

## G3. Inference-only JAX net + weight bridge

`games/blokusduo/jaxenv/net.py`: exact `AlphaBlokusDuo` forward in plain jnp — trunk
conv+BN+ReLU, N residual blocks (conv-BN-ReLU-conv-BN, skip, ReLU), value head
(1×1 conv → BN(1) → ReLU → flatten → Linear(196→F) → ReLU → Linear(F→1) → tanh), ConvPolicyHead
(1×1 conv to 91 planes + permutation gather via `build_action_permutation` + pooled pass
logit). Eval-mode BN folds to `(x - running_mean)/sqrt(running_var + eps) * γ + β` — the
converter (`checkpoint.py`) maps the torch `state_dict` (including BN running stats) to a jnp
pytree; no dropout exists in the torch net so none is needed here.

Tests: (a) forward equivalence vs `predict_encoded` on 500 dev-cache states — fp32, atol ~1e-4
on probabilities and value (BN eps and conv order make exact equality unrealistic; tolerance
pinned in the test); (b) bf16 variant within a looser tolerance; (c) permutation correctness
via a policy round-trip on a known position. Benchmark forwards/s (should reproduce the
spike's 31.2k bf16 number).

## G4. Search core — mctx + top-K compaction (the risk concentration)

`games/blokusduo/jaxenv/search.py`:

- `recurrent_fn(params, rng, compact_action, embedding)`: embedding carries
  `(GameState, topk_global_ids)`; map compact→global id, `env.step`, encode, net forward,
  compute the child's legal mask, take top-K legal priors → child embedding; return
  `RecurrentFnOutput(reward=0, discount=-1 …two-player…, prior_logits[K], value)`.
- Root preparation: legal mask → priors → **Dirichlet mix over legal actions
  (ε=`dirichlet_epsilon`, α=`dirichlet_alpha`) → then top-K** → `RootFnOutput`. mctx's own
  `dirichlet_fraction` stays 0 (we pre-noise so noise precedes pruning).
- Parameter mapping: `pb_c_init=cpuct`, `pb_c_base=1e9` (log term ≈ 0), identity-style
  `qtransform` (raw Q, unvisited→0) as default; mctx's default qtransform kept as a config
  variant. Terminal handling via discount=0 at terminal steps.
- Output: visit counts mapped back to global action ids (dense 17,837 vector for the policy
  target).

Validation (the go/no-go of this plan): on ~200 fixed dev-cache positions with the *same
converted net*, run Python MCTS (no noise, S=400, K=16 batching) vs jax search (no noise,
S=400) and report top-1 move agreement, visit-distribution KL, and value error, sweeping
K ∈ {64, 128, 256, dense-at-small-B}. Acceptance: K exists where top-1 agreement vs Python is
statistically indistinguishable from Python-vs-Python across seeds (the Python search's own
seed-to-seed agreement is the yardstick — batched search can't beat that). Also VRAM + sims/s
sweep over (B, K, S) on the box → pick defaults. Escape hatch if mctx fights back: `mctx-az`
fork or a custom fixed-shape tree — flagged for discussion before building either.

> **Result (2026-07-02, box, run3 `accepted_82` net, S=400, 200 positions,
> `scripts/validate_jax_search.py`, `temp/benchmarks/validate_jax_search.json`):** the
> acceptance gate passes at every K. Yardstick — production's own python K=16 batching vs
> the exact K=1 reference: top-1 **0.715**, overlap 0.620. Jax search vs the same reference:
> K=64 → top-1 **0.735** / overlap 0.771; K=128 → **0.830** / 0.846; K=256 → **0.870** /
> 0.922. i.e. the jax/mctx/top-K search is a strictly *better* approximation of the exact
> PUCT search than the production pipeline's own virtual-loss batching, at every tested K.
> Unit-level (small net, S=60, no batching noise): jax top-1 vs exact = 20/20 with 0.983
> overlap. **K=128 chosen as default** (0.830 at half the tree memory of K=256). Small-net
> corollary from the unit tests holds on the trained net; no mctx escape hatch needed.

## G5. Batched actor loop

`core/jaxplay/actors.py` (game-agnostic over a small `JaxEnv` protocol that
`games/blokusduo/jaxenv` implements): B game slots; per move — search (G4), temperature
per slot (`temp = move_count < temp_threshold` → sample from visit distribution, else argmax
with random tie-break, matching `self_play.py:73` and `mcts.py:253-264`), step, auto-reset
finished slots (pgx pattern), record `(compact board, dense pi, player, move_count)` per step
to a fixed-horizon on-device trace. Per harvest: transfer to host, split by game, backfill
values (`game_result * (-1)^(player != final_player)` with the 1e-4 draw convention —
`self_play.py:109-116` semantics), apply transpose augmentation host-side (numpy board
transpose + the existing `transpose_action` permutation — same 2× as `get_symmetries`),
sparsify policies, emit `GameExamples` lists. Truncation mask discards trailing unfinished
games at generation end (pgx's value-mask trick) — logged, not silent. Seeded from
`config.seed` + generation (jax fold_in; documented as a different stream from
`derive_episode_seed`).

Unit tests at tiny scale (B=4, S=8, TTT-sized horizons on the real Blokus env): games
terminate, values are antisymmetric across players, augmentation doubles examples,
example format round-trips through `SelfPlayStore`.

## G6. `JaxSelfPlayBackend` + Coach integration

Implements the G1 protocol: per generation — load current torch checkpoint → convert weights
(G3) → run actors until `num_eps` games harvested → return `(games, stats)` with
`MCTSEpisodeStats`-compatible entries (num_moves, total_sims — synthesised from the trace so
Coach logging/W&B keep working). Lazy jax import (python-backend runs remain jax-free).
Integration test: 2 generations of a tiny Blokus config end-to-end on CPU (small B/S) —
buffer fills, training step runs, checkpoint saves, resume works.

## G7. Backend-vs-backend throughput benchmark

Extend `scripts/benchmark_jax_env.py` (or a sibling `benchmark_selfplay_backends.py`): measured
games/s and sims/s for `python` (16 workers, K=16 — reproduce the N6 10.5k sims/s baseline) vs
`jax` (chosen B/K/S) at the run3 config on the box, fp32 and bf16, plus VRAM. Output: HTML +
JSON per the benchmark conventions. Expected: ~2.5–3× at S=400 PUCT (per the spike ceiling);
this is the honest number that goes in the findings note.

> **Result (2026-07-02, box, 128f×8b, S=400 flat, `scripts/benchmark_selfplay_backends.py` +
> component profile + (B,K,S) sweep): the spike's 2× mctx-overhead margin was wrong — it is
> ~15× at K=128.** Component decomposition at B=128/S=400: 400 net forwards = 1.55s; the
> entire recurrent body (step + game-end + masks + forward + top-k) = 2.13s; the full mctx
> search = 32.7s. The overhead lives in mctx's per-sim tree machinery and scales with
> **top_k** (tree-array traffic), NOT with launches: K=128→64→32 gives 1.57k→8.5k→11.6k
> sims/s, while batch size barely matters (B=128 vs 1024 vs 2048 within ~15%). Consequences,
> measured:
>
> | Config | sims/s | moves/s | games/s (est ÷28.6 plies) | vs python 11.9k sims/s / 0.91 games/s |
> |---|---|---|---|---|
> | jax PUCT S=400 K=128 B=128 | 1.3k | 3.3 | 0.10 | **0.11×** — unusable |
> | jax PUCT S=400 K=64 B=128 | 8.5k | 21 | 0.75 | 0.8× |
> | jax PUCT S=400 K=32 B=1024 | 10.8k | 27 | 0.94 | **~1× parity** |
> | jax **Gumbel-shaped** S=64 K=64 B=1024 | 20.9k | **327** | **~11.4** | **~12.5×** |
>
> So on the 3060 Ti: **PUCT-mode jax ≈ parity** with the python pipeline (while freeing all
> 16 CPU cores), and the speed win comes from **low-sim search** — exactly the plan's G10
> framing, now with hard numbers. The G8 A/B therefore runs three arms (python-PUCT,
> jax-PUCT for equivalence, jax-Gumbel n=64 for the lever). Defaults updated: `top_k=64`.
> Longer-term PUCT-mode fix (custom fixed-shape tree to kill the mctx per-sim traffic) noted
> as future work, not this plan.

## G8. A/B validation runs

Two ~15-generation runs at a `blokus_linux16_15`-style config (1000 games/gen, 64f×4b or
128f×8b — pick one and state it), identical hyperparameters, `selfplay_backend` the only
difference. Compare: internal Elo trajectory, head-to-head arena between the two final nets
(≥100 games), Pentobi L1 win rate (800 eval sims, existing harness), and per-gen wall-clock.
Acceptance to proceed to G9: jax-trained net not weaker beyond noise (head-to-head within CI
of 50%, Pentobi L1 within CI) AND ≥2× wall-clock win on generation. Write
`docs/research/jax-pipeline-ab.md` with the tables.

## G9. Flip the default + docs

Set `selfplay_backend: "jax"` in the Blokus run configs (the dataclass default stays
`"python"` so TTT and old configs are untouched); update CLAUDE.md current-state, README
status, `docs/guides/REMOTE-TRAINING.md` (jax-cuda extra in the box setup), and
`docs/08-TRAINING-ESTIMATES.md` with the measured G7/G8 numbers.

> **As applied (per the G8 verdict):** the flip is jax **+ Gumbel** — the new production
> config is `run_configurations/blokus_jax_gumbel_30.json` (128f×8b, 2000 games/gen, n=64,
> K=64, B=512, bf16); historical run configs are untouched. Docs updated: CLAUDE.md
> (current-state block, commands, gotcha #7), REMOTE-TRAINING (jax-cuda sync). README has no
> "Current Status" section to update (pre-existing stale pointer in CLAUDE.md);
> 08-TRAINING-ESTIMATES deferred to when a full gumbel production run exists to measure.

## G10. Gumbel mode (opt-in, the ~6× lever)

`mcts_config.search_policy: Literal["puct", "gumbel"] = "puct"` + `gumbel_max_considered: int`.
`gumbel_muzero_policy` with n≈32–64 sims and 16–32 considered actions (interacts neatly with
top-K compaction: K can shrink toward `max_num_considered_actions` at the root). Note the
policy target changes meaningfully (softmax of `prior_logits + completed_qvalues`, not raw
visits) — this is deliberately *outside* the "no behavioural change" contract, which is why it
is a separate, opt-in step validated by its own short A/B run before anyone trains on it. This
is the step that turns the pipeline's 3× into ~15–20× on current hardware.

---

## Risks / open questions (for review)

1. **G4 is the risk concentration** — parameter-mapping fidelity and top-K behaviour. It has
   its own measurable acceptance gate before anything downstream is built on it.
2. **VRAM on 8 GB** limits (B, K, S) until a 16 GB card arrives; G4's sweep produces the
   frontier so the 5070 Ti simply moves a config knob, not code.
3. **Gumbel in or out of this plan?** Kept as G10 (opt-in, post-flip). Cutting it doesn't
   block anything; including it is where the headline speedup lives.
4. **A/B box budget**: G8 is ~2 × (a day-ish) of box time at the 15-gen config. Cheap
   insurance for a default flip; shout if you'd rather gate on something lighter.
5. Naming: `games/blokusduo/jaxenv/` + `core/jaxplay/` mirrors the existing core/games split —
   open to better names before G2 lands.
