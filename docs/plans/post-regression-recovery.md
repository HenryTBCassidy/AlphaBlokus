# Post-regression recovery — training hygiene, external keep-best, and the capacity gate

Executes the recommendations of
[`../research/regression-and-next-steps.md`](../research/regression-and-next-steps.md) (the
`blokus_paired_gate_rerun` post-mortem, 2026-07-20). Context in one paragraph: the rerun regressed
the best net from Pentobi L4 (weighted 0.344) to L3 (0.298) because a degrading training step
(epochs 2 + constant 1e-3 + **Adam with no weight decay**) compounded freely behind a
`regression_guard 0.45` gate that the paired-arena instrument makes mathematically inert (research
§1.2 — even a ~+100-Elo-class real gap reads as 0.525). Separately, the post-mortem found the `xl`
demotion in [`../research/xl-training-scaleup.md`](../research/xl-training-scaleup.md) A4 rested on
circular evidence (research §3.1): capacity is *untested*, not refuted. This plan fixes the training
hygiene and the selection mechanism (cheap, certain), then gates the two expensive bets — an `xl`
run vs Pentobi distillation — on a free capacity probe run on the box.

Decisions already made (Henry, 2026-07-22): weight decay defaults **on** for all runs (it is a fix,
not an option — intentional behaviour change); the capacity probe is a runnable script executed on
the box as the final step before any paid run.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| P1 | Training hygiene: `optim.Adam` → `AdamW`, `NetConfig.weight_decay` default 1e-4 (on), re-asserted on checkpoint load | 1 h | High | ✅ |
| P2 | Continuation-config hygiene: `epochs: 1`, `learning_rate: 2.5e-4` in the warm-start continuation config; document the LR-for-continuation rule | 30 min | High | ✅ |
| P3 | Keep-best + drift circuit-breaker library (`evaluation/ladder_selection.py`) | 2 h | High | ✅ |
| P4 | Mini-ladder runner (`scripts/mini_ladder.py`): L3–L6 × 50 games × 400 sims per checkpoint, history + alarm | 2 h | High | ✅ |
| P5 | Author `run_configurations/blokus_xl_scratch.json` (launch gated on P8) | 30 min | High | ✅ (config; launch gated) |
| P6 | Capacity-probe script (`scripts/capacity_probe.py`): supervised `large` vs `xl` fit on a frozen buffer, game-level held-out split | 3 h | High | ✅ (script; run = P8) |
| P7 | Re-crown v3 gen-40 as project best + wire the mini-ladder runbook on the box | 1 h (box) | High | |
| P8 | **Run the capacity probe on the box — the P9-vs-P10 gate** | ~½ day box GPU, $0 | High | |
| P9 | Paid run: `xl` from scratch on a rented 5090 (launch iff P8 fires or is ambiguous) | ~$100–130, 4–5 days | Gated | |
| P10 | Pentobi distillation: L7–L9 game generation + SL fine-tune + RL continue (iff P8 shows a clear tie) | 1–2 weeks eng | Gated | |
| P11 | Colour-conditional value calibration: thread side-to-move through `ProcessedExample` → store → eval set | ~1 day | Medium | |

**Execution order + gates:** P1–P6 are this PR (code + configs, free, certain). P7–P8 are box work,
no rental. **P8's verdict is the gate**: `xl` beats `large` on held-out fit by ≥0.03 nats policy CE
(or the result is ambiguous) → P9; a clear tie → P10 (and P9 is skipped, its config kept for a
later capacity-justified moment). P11 is independent code that can land any time (e.g. while P9
trains). Nothing in P9/P10 launches without a human reading P8's JSON.

---

## P1. Training hygiene: AdamW + weight decay, default on

**Current state:** `base_wrapper.py:223` builds `optim.Adam(..., lr=...)` with **no weight decay**.
The rerun showed what that permits at a converged net: policy symmetry KL 0.64 → 1.24, value
symmetry MAE 0.10 → 0.25 over 20 accepted generations, value loss overfitting to 0.24 vs v3's
healthy 0.36–0.42 (research §1.3). AGZ/AZ used L2 c=1e-4 precisely against late-stage drift.

**Fix:** switch to `optim.AdamW` with a new `NetConfig.weight_decay: float = 1e-4`. **Default on for
every run** — an intentional behaviour change to all configs, approved 2026-07-22. Also re-assert
the configured weight decay after `optimizer.load_state_dict` in `load_checkpoint`: saved param
groups from pre-change checkpoints carry `weight_decay=0.0` and would otherwise silently disable
the fix on `--resume` / arena reject-reload.

**Success criterion:** on the next continuation run, symmetry KL and value-symmetry MAE flat or
falling across the run (they were monotonically rising in the rerun); ladder criterion lives with
the run that uses it (P9/P10).

---

## P2. Continuation-config hygiene: epochs 1, LR 2.5e-4

**Current state:** `blokus_paired_gate_rerun.json` ran `epochs: 2` at constant `1e-3` — reuse ≈12
passes/position and a from-scratch LR applied to a converged net (research §1.3).
`plateau-investigation.md` R4 ("keep 1e-3") predates the rerun's drift data and is superseded for
warm-start continuations.

**Fix:** set `epochs: 1`, `learning_rate: 2.5e-4` in the warm-start continuation config.
**The LR-for-continuation rule** (also noted in `NetConfig`): from-scratch runs keep peak 1e-3
(cosine, floored 1e-4 — v2 recipe); warm-start continuations of a converged net run constant
~2.5e-4. v3's early gains came from a donor far from the operator's fixed point — at the fixed
point, 1e-3 is all diffusion.

**Success criterion:** any future continuation run holds or improves its start ladder (the rerun
lost 4.6 pp weighted in 20 gens).

---

## P3. Keep-best + drift circuit-breaker library

**Current state:** weight-flow decisions are made by candidate-vs-incumbent arena, which cannot
rank near-equal nets in this colour-dominated game (research §1.2, §4): the strict gate froze
(0/17), the guard/always variants regressed. The only instrument that repeatedly resolved
differences the arena calls a tie is the Pentobi ladder.

**Fix:** `src/alphablokus/evaluation/ladder_selection.py` — pure, tested logic the runner (P4) and
any future orchestration call:

- `LadderPoint` (frozen dataclass): checkpoint label, generation, weighted score, level, score.
- `select_best(points)` — keep-best by weighted ladder score (tie → lowest generation: least drift).
- `detect_drift(points, drop=0.05, consecutive=2)` — trip when two consecutive evaluations sit
  ≥5 pp weighted below the best seen so far (the rerun would have tripped by ~gen 8–10, saving
  ~$20 and the regression — research §4).

**Success criterion:** unit-tested; the rerun's pooled trajectory replayed through `detect_drift`
trips before gen 10.

---

## P4. Mini-ladder runner

**Fix:** `scripts/mini_ladder.py` — wraps `scripts/pentobi_benchmark.py` (its parallel sweep +
`write_ladder_result` JSON) to ladder one or more checkpoints at **L3–L6, 50 games/level,
400 sims** (~2–3 h/checkpoint on the box), append each result to
`<run>/MiniLadder/history.json`, and report `select_best` + `detect_drift` over the history.
On a tripped alarm it writes `<run>/MiniLadder/DRIFT_ALARM` and exits 3, so a box-side loop (P7)
can stop a run / flag Henry without parsing output.

**Scope note — deliberately not built here:** the async orchestration (box daemon watching a cloud
run's synced checkpoints, auto-rollback on alarm) is operational glue, specced in P7 as a runbook
loop rather than new machinery. Don't over-build until one run has used the manual loop.

**Success criterion:** two consecutive drops in a run's mini-ladder history produce the flag file +
exit 3; keep-best output names the checkpoint the full 9-level ladder would pick (spot-check on the
rerun's checkpoints when the box is free).

---

## P5. `blokus_xl_scratch.json`

Authored per research §5's diff from `blokus_cloud_v2.json`: preset `xl` (19.07M params),
**from scratch** (`load_model: false`), Gumbel n=128/considered 32 (research §3.3: heavier sims at
top_k 64 are measured dead weight — the saving funds ~40 extra generations), cosine LR floored at
1e-4, `epochs: 1`, `weight_decay: 1e-4` explicit, paired arena at 200 games **for telemetry only**
(`gate_mode: "always"` — weight flow is continuous; selection is P4's external keep-best),
end-of-run pool tournament on. Extend past 60 gens via `--resume` while the mini-ladder climbs.

**Launch is gated on P8.** Cost when launched: ~$100–130, 4–5 days on one 5090.

**Success criterion (pre-registered):** weighted ladder > 0.344 outside CI, or L5 > 50%, by run
end; stop when the mini-ladder is flat across two consecutive evaluation windows.

---

## P6. Capacity-probe script

**Why:** the `xl` demotion rested on diagnostics measured against the net's own self-play targets —
circular, incapable of detecting capacity limits (research §3.1). The decisive, free experiment:
can `xl` fit the *same data* better than `large` out-of-sample?

**Fix:** `scripts/capacity_probe.py` — one command on the box, no training-loop involvement:

1. Load a frozen buffer of gen-40 self-play games from `SelfPlayHistory/self_play_<i>.parquet`
   (`SelfPlayStore.load_games`; if the parquets aren't on the box, regenerate ~10k games with the
   gen-40 net at n=256 first — ~1 h).
2. Split **by game** (never by position — positions within a game are correlated;
   `training/holdout.py::split_games_holdout`), default 5% held out.
3. Arms: `large` fresh, `xl` fresh, and optionally `large` warm-started from gen-40
   (`--warm-start`). Each trains full shuffled passes (`wrapper.train`, epochs=1 per pass,
   constant LR) with per-epoch held-out **policy CE + value MSE**
   (`BaseNNetWrapper.evaluate_holdout`), early-stopped on patience.
4. Reports per-arm curves + best metrics to JSON, including held-out target entropy so
   KL = CE − H is readable directly (the stochastic-target noise floor is common to both arms and
   cancels in the comparison).

Example box invocation:

```bash
uv run python scripts/capacity_probe.py \
    --config run_configurations/blokus_cloud_v2.json \
    --history-dir temp/runs/blokus/<run>/SelfPlayHistory --file-indices 16 \
    --arms large,xl --max-epochs 20 --out temp/benchmarks/capacity_probe.json
```

**Status: script written + unit-tested in this PR; the run itself is P8 (box, not done here).**

---

## P7. Re-crown v3 gen-40 + box mini-ladder runbook

Box/housekeeping, no code:

1. `blokus_cloud_v3/Nets/accepted_40.pth.tar` is the project's best net — the rerun produced
   nothing above +5.5 pool Elo and its final net laddered 0.298 (research §1.1). Copy it to a
   pinned location (e.g. `temp/best_nets/v3_gen40.pth.tar` on box + Mac) and note it in
   `README.md`'s ladder section next time the chart is regenerated.
2. Runbook loop for the next cloud run (manual, per P4's scope note): after every 5th generation
   syncs to the object store, fetch `accepted_*.pth.tar`, run `scripts/mini_ladder.py` on the box,
   stop the pod on exit 3 / `DRIFT_ALARM`, and resume from `select_best`'s checkpoint.

**Success criterion:** the next run's report names its best checkpoint by mini-ladder, not by
`best.pth.tar`.

---

## P8. Run the capacity probe (the gate)

Run P6's script on the box (GPU, free, ~½ day incl. optional data regeneration). **Verdict rule
(pre-registered, research §3.4):**

- `xl` best held-out policy CE ≤ `large`'s − 0.03 nats → **capacity binding** → launch P9.
- Gap < 0.01 nats → **clear tie** → the A4 demotion finally stands on sound evidence → P10.
- Between → ambiguous → default to P9 (config-only, and the distillation build P10 proceeds in
  parallel anyway — research §5 R4).

Also record value-MSE deltas and the warm-start arm (does gen-40 warm-start reach a lower CE than
fresh `large`? If yes, continued SL on frozen data still has headroom — relevant to P10's design).

---

## P9. Paid run: `xl` from scratch (gated)

Launch `blokus_xl_scratch.json` (P5) on a rented 5090 per the cloud runbook
(`docs/guides/REMOTE-TRAINING.md`, object-store sync verified **before** any auto-stop — see
cloud-run data-safety rule). Drive selection via P7's runbook loop. Success criterion as P5.
Budget: ~$100–130 — comparable to what the three failed `large` continuations already spent
(research §3.2).

---

## P10. Pentobi distillation (gated / built in parallel)

1–2 weeks engineering; scaffold only when reached: generate 20–50k Pentobi L7–L9 self-play games
via the box GTP harness (`games/blokusduo/pentobi/`, free CPU, days); SL fine-tune the best net
(policy → Pentobi moves, value → outcomes, LR 1e-4, AdamW); validate by full ladder; then resume
RL with an opponent pool. Rationale: the L7–L9 nets that beat us 80–90% are an information source
self-play cannot synthesise (research §5 R5); v3's real transfer concentrated at exactly those
levels. **Success criterion:** +10 pp at any of L5–L7 after SL alone.

---

## P11. Colour-conditional value calibration

**Why:** 73% of self-play outcomes are White wins; the rerun's value head overfit to 0.24 MSE and
we cannot see whether it exploits colour (plateau-investigation R8a, still undone; research §5 R7).

**Why not in this PR:** stored boards are *canonical* (current player sign-flipped —
`BlokusDuoBoard.canonical`, `board.py:152`), so absolute colour is unrecoverable from an eval-set
position; piece-count parity misattributes post-pass positions silently. The clean fix threads
side-to-move through `ProcessedExample` → jax harvest → `SelfPlayStore` schema (new column +
`policy_kind`-style marker) → `EvalSet` (optional `to_move_is_white` array) → a colour-split
`log_value_calibration`. ~1 day, touching the storage schema — a separate PR. Old eval
sets/parquets lack the field; the diagnostic degrades to the current uncoloured form on them.

**Success criterion:** ValueCalibration parquet gains a colour dimension; the report shows
White-to-move vs Black-to-move reliability curves.

---

## Not in scope

- **Search-width calibration** (top_k 128 @ n=128 feasibility — research §5 R6): queued in
  `docs/IDEAS.md` territory; becomes an arm for the run after next if pursued.
- **Async keep-best orchestration daemon / auto-rollback machinery** — P7 runs the loop manually
  first; automate only if it earns it.
- **EMA teacher / averaged incumbent** — selection is solved more directly by P3/P4 (research §4).
- **Multi-GPU** — unchanged verdict from `xl-training-scaleup.md` §4–5 (single 5090).
- **Any further warm-start self-play continuation of `large` through the current operator** —
  explicitly recommended against (research §5, bottom).
