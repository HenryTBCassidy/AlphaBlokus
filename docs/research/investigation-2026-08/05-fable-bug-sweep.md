# Stream C bug sweep — Fable (2026-08-04)

Six items from `03-synthesis.md` §Stream C, audited in order, **executed checks
preferred over code reading**. Probes live in the session scratchpad
(`probe_c1_prng.py`, `probe_c2_topk.py`, `probe_c3_endgame.py`,
`probe_c3b_discriminate.py`, `probe_c3c_focus.py`, `probe_c5_optimizer.py`);
durable tests were committed on branch `audit/fable-bug-sweep-stream-c`
(worktree, not pushed):

- `tests/games/blokusduo/jax/test_gumbel_exact.py` — 5 new tests (4 fast, 1 `slow`)
- `tests/training/test_optimizer_continuation.py` — 2 new tests

All empirical work ran CPU fp32 with the real production modules
(`search.py`/`actors.py`/`backend.py`/`base_wrapper.py`, mctx 0.10.2). bf16
behaviour remains A4's job — none of the verdicts below covers reduced precision.

Machine-readable findings: `findings-fable.jsonl` (same directory).

---

## Verdicts at a glance

| Item | Subject | Verdict |
|---|---|---|
| 1 | PRNG key uniqueness in lockstep self-play | **CLEAN** (executed) |
| 2 | top-k vs illegal-move masking | **CLEAN** (executed, root + 1,152 child nodes) |
| 3 | completed-Q perspective per depth | **CLEAN** (executed, negamax endgames + sims ladder + independent reference) |
| 4 | replay sampling | mechanics **CLEAN**; **2 verified defects** in adjacent machinery; measurements below |
| 5 | optimizer continuation | **CLEAN** (executed, 14 checks) |
| 6 | missing small exact tests | 4 gaps found; **7 tests added** (all pass); 1 gap left open (expensive) |

**Defect count: 2 verified, both `degrades-silently`. 0 `corrupts-training`. 2 suspicions (cosmetic).**

---

## VERIFIED DEFECTS

### FABLE-04 — replay-buffer flatten order is unseeded; same-seed runs are not reproducible

- **File:** `src/alphablokus/training/replay_buffer.py:13,89` (`from random import shuffle`; `shuffle(examples)`), with `src/alphablokus/training/coach.py:161-166` seeding only `np.random` and `torch`.
- **What:** `flat_shuffled_examples()` shuffles the flattened buffer with Python's stdlib `random`, which the Coach never seeds. Executed check: two fresh processes seeded exactly as the Coach seeds (np + torch, seed 42) produce different shuffle orders. The DataLoader's carefully seeded per-generation shuffle (`_shuffle_seed(seed, generation)`, built specifically to kill the data-order confound measured at ~4× a treatment effect) permutes *indices into a nondeterministically ordered list*, so actual batch composition differs run-to-run at the same seed.
- **Why it matters:** `config.py:602-606` promises "same seed + same config + same hardware will produce identical metrics" — broken at the first training step of every run. Every same-seed A/B (B3's 3×3 sweep, B6 vs its repeats, any D-stream arm) carries reintroduced data-order variance that the loader-side fix was supposed to remove.
- **Severity:** degrades-silently (experiment interpretability; not a training-correctness bug — the shuffle is still a uniform shuffle).
- **How to verify:** `python -c "import numpy,torch; numpy.random.seed(42); torch.manual_seed(42); from random import shuffle; xs=list(range(1000)); shuffle(xs); print(xs[:8])"` twice → different output.
- **Proposed fix:** shuffle with a seeded generator, e.g. `random.Random(_shuffle_seed(config.seed, generation)).shuffle(examples)` or drop the flatten-time shuffle entirely (the seeded DataLoader shuffle already randomises batch order; flatten order only needs to be deterministic).

### FABLE-05 — the "held-out" eval set is trained on for the first ~6 generations

- **File:** `src/alphablokus/training/eval_set.py:92-95` (samples from `train_examples`), `src/alphablokus/training/coach.py:290-312` (then trains on the same flattened buffer).
- **What:** the 200 eval positions are sampled from generation 1's *training* buffer and never removed from it. They — and their transpose-augmented twins, and their ~63 same-game sibling positions each — remain in the training set until they age out of the 60k-game buffer at generation ≈ `B/F` = 6 (rerun: gens 1–6, 2 epochs each = up to 12 gradient passes over the exact "held-out" positions). Every per-epoch diagnostic advertised as held-out (`TrainingEntropy`, `PolicyAccuracy`, `ValueCalibration`, PVC) is in-sample over that window, and the metric silently changes meaning at gen ≈ 6 when the positions age out.
- **Why it matters:** these are the dashboards the runs are judged by mid-flight; early-run curves read optimistic, and the gen-6 kink is an artifact, not a training event. Magnitude is bounded (200 positions in ~3.8M) but the same-game-sibling correlation is not small — 200 positions ≈ 47 game lineages whose *other* positions are all trained on. Note `training/holdout.py` (used by the offline probes and B3) splits **by game** and is clean — this defect is specific to the Coach's in-run eval set.
- **Severity:** degrades-silently (corrupts the instrument, not the training targets).
- **How to verify:** read `eval_set.py:92-95` — `idx = rng.choice(len(train_examples), ...)`; nothing removes `idx` from the buffer. Or empirically: hash the eval-set boards and grep them out of `SelfPlayHistory/self_play_0.parquet`.
- **Proposed fix:** exclude the sampled positions' *source games* from the buffer (game-level exclusion, mirroring `split_games_holdout`), or build the eval set from games generated outside the buffer. Fold into A6, which is already rebuilding this machinery.

---

## SUSPICIONS (cosmetic — flagged, not blocking)

### FABLE-08 — the rerun trained `epochs=2`, while the plan (A3) says "Epochs stay at 1"

`config.resolved.json` of `blokus_paired_gate_rerun`: `net_config.epochs = 2`. All
20 generations show `Epoch 1/2` + `Epoch 2/2` in `rerun.log`, and the data-regime
line logs `emergent reuse ≈12.0`. Synthesis A3's "Epochs **stay** at 1" implies the
historical value was 1 — it was 2. Whoever lands A3 should know they are *halving*
the gradient passes relative to every historical run, not holding them constant.
(Not a code defect; a plan/ledger discrepancy.)

### FABLE-09 — `config.seed or 0` conflates `seed=0` with `seed=None`

`backend.py:167` and `eval_set.py:93` use `config.seed or 0`: a run explicitly
configured with `seed=0` and a run with seeding disabled (`seed=None`) derive the
same self-play RNG stream. Harmless today; a footgun for seed sweeps that include 0.

---

## Item-by-item detail

### 1. PRNG key uniqueness — CLEAN

The production chain is `PRNGKey(seed) → fold_in(generation)` (backend.py:167)
`→ split per wave` (:179) `→ split(wave_key, wave_plies)` (actors.py:137)
`→ split(step_key, 3)` (:99) `→ split(search_key, 2)` (search.py:190). Executed:

- **2,880/2,880 derived keys unique** over 3 generations × 6 waves × 32 plies × 5
  roles (search/sample/tie/noise/mctx); 40 generation root keys unique.
- **No Gumbel broadcast:** spied `jax.random.gumbel` during a traced production
  search — exactly one draw, shape `(B, K)` = (64, 16): every game row gets its
  own noise from the single shared key (mctx 0.10.2 samples
  `shape=root.prior_logits.shape`). No `(K,)` draw exists anywhere in the trace.
- **Behavioural:** 64 identical initial states under ONE key → 16 distinct chosen
  openings (= `max_considered`, the ceiling), most-common share 0.12; two
  consecutive step keys agree on only 8% of slots; same key twice → identical.
- No `pmap`/multi-device use anywhere in `src/` — the per-device axis is vacuous.

Durable regression test added: `test_gumbel_noise_is_per_slot_not_broadcast`.

### 2. top-k vs masking — CLEAN

Masking is applied *before* `top_k` in both places, via the same helper
(`search.py:194-198` root, `:176-177` children; `topk_legal` remaps padded −inf
slots to pass). Executed, by running the real `make_search` body unjitted with a
spy capturing the concrete mctx root and tree, on 24 mixed-phase positions
(legal counts 1–649 vs K=16, so displacement pressure was real):

- **Root:** all 24 windows are *exactly* the top-K of the legal-masked priors;
  `invalid_actions` handed to mctx == the non-finite slots; padded ids == pass.
- **Children:** all **1,152 expanded interior nodes** audited the same way
  (legality recomputed from each node's embedded state, priors recomputed from
  scratch) — every window exact, zero illegal ids, zero displaced legal moves.
- **Independent rules cross-check:** jax `legal_mask_batch` == python
  `valid_move_masking` on all root positions.

Caveat (A4's, restated): this was fp32. Masking order is dtype-independent — an
illegal move can never enter the window at any precision — but *which* legal
near-tied moves fill it can shift under bf16.

Durable regression test added: `test_root_window_is_topk_of_masked_priors`.

### 3. completed-Q perspective per depth — CLEAN

Analytic: `recurrent_fn` emits reward from the **parent** (mover) perspective on
terminal edges, discount 0 there and −1 otherwise, and the child's net value in
the **child's** to-move frame; mctx's backward (`leaf_value = reward +
discount·leaf_value`) and `qvalues = r + γ·v_child` then keep every node's Q in
its own mover's frame, which is what both `qtransform_completed_by_mix_value`
(gumbel, root + interior) and the custom `qtransform_raw_value` (puct) consume.
Draw sentinel: an in-tree draw is +1e-4 to the mover reaching it and −1e-4 one
ply up — matching the harvest labels.

Executed (the part the parity tests structurally cannot see):

- Random-playout endgames solved by **exhaustive negamax** (win/draw/loss per
  move), positions filtered to mixed move classes at depth ≥ 2 — exactly where a
  per-ply sign flip inverts preferences. Gumbel sims ladder over 12 such
  positions (B=16): 128 sims → 8/12 optimal, 512 → 10/12; win-vs-loss weight
  inversions 72/300 → 48/300. **Improvement with budget is the signature of a
  search limit, not a sign error** (a sign error locks in and worsens).
- Both 512-sim stragglers re-run in isolation at 512 and 2048 sims: **solved,
  losing moves at ≤0.03 target weight, the sole drawing move at 0.89–0.95**
  (e.g. pos9, depth 3: 16 losing moves vs 1 draw — the search finds the draw).
- **Independent reference:** the python full-action-space PUCT MCTS (separate
  implementation, conventions previously verified) at 512 sims *fails* one of
  those positions (puts visit share 1.0 on a losing move, 0.0 on the saving
  draw). The JAX Gumbel path outperformed the reference on the same net — the
  residual failures are random-prior blindness, not a JAX perspective bug.
- All depth-2 positions were solved at every budget tested.

Note for other streams: an unlucky Gumbel draw at 512 sims *can* leave a
depth-4 position with fully inverted win/loss weights (observed once in a
B=16 batch, resolved at a different draw and at 2048 sims). With a trained
prior this is far less likely than with the random-net priors used here, but it
is a concrete mechanism for occasional bad self-play targets at 256 sims —
data noise, not corruption.

Durable regression test added: `test_endgame_negamax_exact` (`slow`).

### 4. replay sampling — mechanics CLEAN; measurements; 2 defects above

Everything measured on the production rerun (`blokus_paired_gate_rerun`,
resolved config + `rerun.log` + `SelfPlayProfiling`, 200k episodes):

- **Actual vs configured buffer:** configured 60,000 games; actual hits exactly
  60000/60000 at gen 6 and stays (gen-6..20 log lines). ~3.78–3.85M positions
  (this **includes** the transpose twin — each board situation is stored twice).
  Warm-started rerun began with an **empty** buffer: gen 1 trained on 10k games
  (632k positions), gens 1–5 on a partially filled, younger-than-steady-state buffer.
- **There is no sampler.** Training is `epochs` full shuffled passes over the
  entire flattened buffer (`base_wrapper.train`, `flat_shuffled_examples`).
  Lifetime reuse per stored example = epochs × B/F = **2 × 6 = 12 gradient
  contributions** (log: "emergent reuse ≈12.0"); **24 per underlying position**
  counting its deterministic transpose twin. ~3,700 optimizer steps/epoch,
  ~7,400/generation at batch 1024. A game's single outcome label is seen
  ~63 × 12 ≈ **750 times** by the value head over its buffer lifetime.
- **Long-game overweighting: negligible in production.** Uniform-by-position
  does weight games by length, but measured lengths are 31.7 ± 1.6 plies
  (p10 30, p90 34, max 40): a p90 game gets 1.13× a p10 game's exposure; the
  top length-decile holds 13.1% of positions. Not a mechanism for anything.
- **Composition of a typical 1024-position minibatch:** movers ~50/50
  White/Black (alternation + even-ish lengths); phase ~uniform over ~32 plies
  (~32 positions per ply index); expected same-game sibling pairs ≈ 9 per batch
  (mild within-batch correlation). Value labels are ~50/50 ±1 *marginally* —
  the 73% White-win skew lives entirely in the conditional:
  P(label=+1 | White to move) ≈ 0.73, P(label=−1 | Black to move) ≈ 0.73.
  That conditional is exactly the colour shortcut the value head is suspected
  of learning, and it survives any amount of uniform sampling — supporting D1
  (outcome-balanced value sampling) as the right intervention point.
- **Tail truncation:** each generation drops ~970 still-open games (~9% of the
  10k kept) plus 0–1,000 overflow games at the wave boundary. Slight length
  bias in what's dropped, immaterial at σ=1.6 plies.

### 5. optimizer continuation — CLEAN

Executed against the real wrapper driving the Coach's exact sequence (14 checks,
all pass), plus existing coverage in `tests/games/test_base_wrapper.py`:

- **Moments:** one `AdamW` built at wrapper construction; moments persist and
  keep accumulating across accepted generations (no per-generation reset). On
  arena **reject**, `load_checkpoint(temp, restore_lr_schedule=False)` restores
  weights **and** Adam moments exactly to the pre-training snapshot (bitwise-
  verified) — never a weights-only revert.
- **LR schedule:** built once with `T_max = num_generations × epochs`, stepped
  once per epoch, never restarted within a run; reject keeps the clock and
  re-syncs the optimizer LR to the scheduler (verified: post-revert LR is the
  advanced value, not the stale saved one).
- **Weight decay:** live and equal to `net_config.weight_decay` (default 1e-4)
  in every param group — there is exactly **one** param group
  (`AdamW(self.nnet.parameters(), ...)`), so "all groups" is trivially true, and
  BN scales/biases are decayed too (a style deviation from AGZ practice, not a
  bug; worth knowing when comparing to published setups). A legacy checkpoint
  saved with `weight_decay=0.0` is re-asserted to config on load (verified).
- **Warm start (`load_weights`):** optimizer and scheduler left fresh — LR back
  at configured peak, clock at 0, no moments. **By design**, and it means every
  historical warm continuation restarted the schedule at peak; that is exactly
  A3's "never reset the LR to peak on a warm continuation" — the mechanism is
  confirmed, the policy change is A3's to land.
- Context confirmed for the ledger: the rerun's resolved `net_config` has **no
  `weight_decay` key at all** (predates AdamW-default-on) — it ran plain Adam
  with zero decay, constant LR 1e-3, epochs 2.

Durable tests added: `tests/training/test_optimizer_continuation.py` (moments
persist; reject reverts weights+moments together without rewinding the clock).

### 6. missing small exact tests — 4 gaps, 7 tests added, 1 left open

What existed: structural invariants (legal-only support, normalisation),
terminal-root-→-pass **under PUCT only**, PUCT-vs-python agreement at a pinned
noise floor, fp32/bf16 net parity. What was missing, per the spec:

| Gap | Status |
|---|---|
| Single-legal-action position (forced pass, game ongoing) | **Added** — `test_forced_pass_midgame_gets_all_mass` |
| Terminal position under the production (gumbel) path | **Added** — `test_terminal_root_under_gumbel_resolves_to_pass` |
| Short endgames vs exhaustive minimax | **Added** — `test_endgame_negamax_exact` (`slow`, ~2 min) |
| Gumbel vs an independent reference implementation | **Partially covered** by the negamax test + the one-off python-PUCT cross-check in the probes. A true independent Gumbel-MuZero reimplementation is real work (days, and it would mostly re-test mctx); recommend instead a box-scale pinned-position corpus test at production sims. **Left open.** |
| (bonus) root window exactness (C2) | **Added** — `test_root_window_is_topk_of_masked_priors` |
| (bonus) per-slot noise independence (C1) | **Added** — `test_gumbel_noise_is_per_slot_not_broadcast` |
| (bonus) optimizer continuation (C5) | **Added** — 2 tests |

All added tests pass (`ruff` clean, format clean). The `slow` endgame test is
deterministic on a given platform (fixed seeds throughout) but exercises a
stochastic search near its budget edge — if a future jax/mctx upgrade flakes it,
loosen to depth-2 positions before deleting anything.

---

## What I would act on first

**FABLE-04 (unseeded buffer shuffle).** It is a one-line fix, and until it lands
every experiment the plan is about to run — B3's nine LR jobs, B6 and its
repeats, any D-stream A/B — quietly carries the exact data-order confound the
project already measured at ~4× a treatment effect and believed it had fixed.
FABLE-05 matters too but its instrument (A6) is being rebuilt anyway; fold the
game-level exclusion into that rebuild.
