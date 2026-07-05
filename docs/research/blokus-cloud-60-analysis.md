# blokus_cloud_60 post-run analysis — Top-1 decline verdict + next-run config

Analysis of the `blokus_cloud_60` run (2026-07, RTX 5090, ~13 h, 58 completed generations of a planned 60; crashed at gen 59 on the since-fixed DataLoader/JAX fork bug). Written 2026-07-05. All numbers below were extracted programmatically from the Plotly traces embedded in `temp/runs/blokus/blokus_cloud_60/Reporting/report.html`; nothing is eyeballed from charts.

**Run config** (`run_configurations/blokus_cloud.json`): large net (192×12, ~8M params), jax Gumbel self-play (n=64 sims, `gumbel_max_considered=16`, K=64), 10,000 games/gen, 40,000-game buffer (staleness 4 gens, emergent reuse 4× at epochs=1), batch 1024, LR 1e-3 cosine (`T_max = num_generations = 60`, stepped once per generation), gate = 40 arena games at >55%.

**Headline outcomes:** 39/58 candidates accepted (67%); Pentobi ladder gen-44 = Level 3 → gen-57 = Level 4; internal Elo pinned at the +1200 cap from ~gen 18 (metric artifact — ignored throughout as a strength signal); gen-58 candidate rejected at 20% wins (37.5% score).

---

## 1. The Top-1 decline is benign — verdict and evidence

**The trigger:** Top-1 policy agreement on the held-out eval set declines from a ~90% plateau (gens 4–27) to 63.5% at gen 58, while Top-5 stays 96–100%.

**Verdict: benign — a stale-eval-set artifact, not policy regression.** Applying the decisive test:

| Test | Result |
|---|---|
| (a) Does training policy loss plateau/rise alongside the Top-1 decline? | **No.** Policy loss falls monotonically the entire run: 2.53 (gen 1) → 0.73 (gen 17) → 0.63 (gen 28, where the Top-1 decline starts) → 0.55 (gen 44) → **0.40 (gen 58)** — still dropping briskly at the end. |
| (b) Does arena acceptance decline over the same generations as Top-1? | **No — it moves the opposite way through the bulk of the decline.** During the steepest Top-1 fall (gens 28–48, 83% → 74%), acceptance was at its *strongest*: 12 of 15 candidates accepted in gens 34–48, with scores of 55–76%. The late acceptance collapse (gens 49–58, §3) starts 20 generations after the Top-1 decline began and has a separate cause (LR annealing, §3). |

So the capacity-limit signature (loss plateaus AND acceptance falls AND Top-1 falls together) is **absent**. Four further lines of evidence:

1. **The mechanism is confirmed in code.** The eval set is frozen at generation 1: `src/alphablokus/training/eval_set.py` builds it once from gen-1 self-play (positions + gen-1's 64-sim MCTS visit-distribution targets), persists it to disk, and reloads it forever after. `evaluate_on_eval_set` in `src/alphablokus/games/base_wrapper.py` scores the *current* net's argmax against **gen-1's noisy MCTS argmax** on **gen-1-distribution positions**. The `eval_set.py` docstring itself documents this failure mode — chasing "gen-1's noisy MCTS targets … is what was making that curve dip over training" — and fixes it for TicTacToe with a minimax oracle. Blokus has no oracle, so the Blokus curve still chases gen-1 targets. A net that *surpasses* gen-1's 64-sim search is **expected** to diverge from its argmax.
2. **Top-5 at 96–100% throughout** (100% at the gen-4–27 plateau, 96% at gen 58). Gen-1's preferred move remains in the current net's top 5 essentially always — the net has re-ranked among near-equal candidate moves, not forgotten or degraded. This is exactly the signature of competitive positions with several near-equal moves.
3. **External strength anchor improved while Top-1 hit its lows.** Pentobi ladder: gen-44 (Top-1 ≈ 72%) = Level 3; gen-57 (Top-1 ≈ 68.5%) = Level 4. The net got objectively stronger over the window where the metric fell furthest.
4. **Eval-set entropy declines smoothly** (4.10 → 1.59 nats, gen 1 → 58) with no collapse — 1.59 nats ≈ 5 effective moves, a healthy sharpness for Blokus midgame branching. A regressing policy would show entropy blow-up or collapse; neither occurs.

**Caveat carried forward:** the value-calibration plot shares the same confound — it compares current-net value predictions against *gen-1 game outcomes*. The compressed gen-58 calibration (predicted +0.9 → actual +0.34; predicted −0.9 → actual −0.59) partly reflects that gen-1's weak play resolved those positions differently than strong play would, so it should not be read as a straight miscalibration measure. §5 recommends fixing the eval set so these diagnostics regain meaning.

---

## 2. Run health — what the other curves say

**Policy loss: healthy, still improving.** 2.53 → 0.40 over 58 gens, monotone modulo small blips, and its steepest late-run segment (0.53 → 0.40 over gens 45–58) is at the *end* of the run. No plateau ⇒ no evidence the 192×12 net is out of capacity.

**Value loss: the weak link.** Falls 0.89 → ~0.39–0.43 by gens 30–41 (minimum 0.386 at gen 41), then **rises to 0.54–0.58 across gens 51–55** before easing to 0.46 at gen 58. The late rise coincides with (i) the incumbent being frozen for 6 straight rejected generations (49–54), so the whole 4-gen buffer converged to a single net's self-play distribution, and (ii) the LR tail (§3). Structurally the value head is data-starved: ~32 positions share one game outcome (10k games/gen = only 10k independent value labels), the exact decorrelation problem AlphaGo hit (`docs/research/deepmind-run-configs.md` §AlphaGo — value net trained on one position per *distinct* game after full-game training overfit 0.37 → 0.23).

**Entropy: healthy taper.** Eval-set network entropy 4.10 → 1.59 nats; self-play move-selection entropy 2.32 → ~1.1 nats. Gradual sharpening, no collapse, no stall.

**Arena: strong through gen 48, coin-flips after.** Gens 1–48: 37/48 accepted, typical accepted scores 57–76%. Gens 49–58: **2/10 accepted** (55 at 73.75%, 57 at 67.5%); the rejected gens 49–54 scored 45–50% — i.e. candidate ≈ incumbent, not candidate-worse. Gen 58's 37.5% (20% wins) is the one genuinely bad candidate, a single-gen event on 40 games.

**Games got longer as play improved:** mean length 28.1 → 32.5 plies — denser board filling, consistent with rising strength.

**Throughput: flat and stable, no drift.** Self-play ~510 s/gen (~19.6 games/s, ~47–48k sims/s), training ~187 s/gen (~14.1k samples/s over the ~2.5M-sample buffer pass), arena ~40–48 s/gen. Total ~12.5 min/gen. Notably **self-play is 69% of wall-clock while arena is only 6%** — which prices the recommendations below.

---

## 3. The real finding: the late-run stall is the LR schedule, not capacity

The one genuinely unhealthy pattern — 8/10 rejections in gens 49–58 with candidate scores of 45–50% — lines up almost exactly with the cosine schedule's tail. `_create_scheduler` in `base_wrapper.py` sets `CosineAnnealingLR(T_max = num_generations × epochs = 60)` with the default `eta_min=0`, stepped once per generation:

| Gen | LR |
|---|---|
| 20 | 7.5e-4 |
| 40 | 2.5e-4 |
| 48 | 9.5e-5 |
| 50 | 6.7e-5 |
| 55 | 1.7e-5 |
| 58 | 2.7e-6 |

Acceptance collapses precisely where LR crosses below ~1e-4 (gen ~48). With near-zero LR the candidate barely moves from the incumbent each generation, so arena scores cluster at ~50% and a >55% gate on only 40 games (binomial SE ≈ 8 pp) rejects almost everything. Meanwhile policy loss *keeps falling* — a barely-moving net converging onto a now-static data distribution (six straight gens of self-play from the same frozen gen-48 incumbent). The two late acceptances (gens 55, 57 at 73.75%/67.5%) and the Pentobi Level 3 → 4 step show real improvement was still available; the schedule was strangling the optimiser's ability to reach it.

**Bottleneck diagnosis, in order:**
1. **LR tail** — mechanically capped late-run progress (free to fix).
2. **Improvement-operator thinness** — n=64 Gumbel sims over only the top-16 considered actions produces targets only modestly better than the raw policy once the net is decent; Blokus opening branching is ~400–500 (Go-like — `deepmind-run-configs.md` uses this to justify α=0.03), and DeepMind ran 800–1,600 sims. This is the growth-rate lever.
3. **Value-head data starvation** — 10k independent outcome labels/gen; the late value-loss rise and (confounded) calibration compression both point here.
4. **Gate noise** — 40 games ⇒ SE ~8 pp; a true-55% candidate is rejected ~half the time, and false rejections freeze the incumbent, which staled the buffer in gens 49–54.

**Not** capacity: no policy-loss plateau, healthy entropy, external strength still rising.

### Addendum (2026-07-05): the LR table above is formula-derived and overstates the anneal

The §3 LR table is computed from `0.5·lr₀·(1 + cos(π·t/60))` indexed by *generation number*. That is **not** the LR the run actually trained at. Scheduler state has been embedded in checkpoints since `e08a0a1` (2026-06-23), and the Coach's rejection path reloads `temp.pth.tar` (`coach.py:223` saves it before training; `coach.py:297` reloads it on reject), whose `load_checkpoint` (`base_wrapper.py:774-775`) restored **both** the optimizer (pre-step LR) and the scheduler position. Net effect: **every arena rejection rewound the LR schedule by one step.** The scheduler's effective clock was *cumulative accepted generations*, not generation number.

Simulating the exact save → step → reject-reload cycle against `blokus_cloud_60`'s acceptance pattern (39/58 accepted; 37/48 by gen 48; gens 49–54 a rejection streak) gives the corrected trajectory:

| Gen | §3 table (formula, no rewind) | Actual LR under reject-rewind |
|---|---|---|
| 20 | 7.5e-4 | ~8.5e-4 |
| 40 | 2.5e-4 | ~4.5e-4 |
| 48 | 9.5e-5 | ~3.5e-4 |
| 49–54 (rejection streak) | 6.7e-5 → 3.3e-5 | **pinned at ~3.2e-4** |
| 58 | 2.7e-6 | ~2.7e-4 |

**The run never trained below ~2.7e-4** — the §3 table overstates the late-run anneal by ~10–50×. The LR-tail story survives in weaker, composite form: falling LR *raised the probability of entering* a gate-hysteresis trap (a one-generation training delta at ~3.2e-4 stopped reliably clearing a noisy 55%-of-40-games gate), and the trap then *sustained itself* — each rejection rewound weights, Adam moments **and** the LR onto an increasingly single-incumbent buffer (after four rejections the 40k buffer is one net's games), so the next generation retrained from the same start at the same LR and produced a near-identical ~50% candidate. Six generations of self-play (49–54) were discarded.

Consequences for recommendation #1 above:
- **The `eta_min = 1e-4` floor barely binds** under these semantics — at a realistic acceptance rate a 60-gen run's schedule position stays well above where the floor would engage. The stopgap is nearly inert.
- **`blokus_cloud_v2` as configured would not start at 1e-3.** Its `load_model: true` full-restored the donor's optimizer LR (~2.7e-4) and scheduler position, so the warm start silently began at ~27% of peak.

Recommendation #1's mechanism is **superseded by `docs/plans/lr-scheduler-options.md`**, which fixes the reject-rewind (L3), makes warm start weights-only (L4), logs the actual LR (L2), and A/Bs constant vs floored cosine to set the production default.

---

## 4. Recommendations for the next run

Prioritised. Costs priced against the measured phase split (self-play 510 s / training 187 s / arena 44 s per gen).

| # | Lever | Change | Why |
|---|---|---|---|
| 1 | **LR schedule** | **Fix the tail.** Keep cosine but floor it: `eta_min = 1e-4` (10% of peak) — a one-line change in `_create_scheduler` — or switch to AZ-style steps (1e-3 → 1e-4 stepped, never 0; AZ annealed 0.2 → 2e-4, a 1000× range but never to zero, over 700k steps). | The single highest-leverage change and it's free. The gens 49–58 stall is mechanically explained by LR → 0; the run demonstrably had improvement left (gens 55/57 accepted strongly, Pentobi L3→L4). |
| 2 | **Gumbel sims** | **n: 64 → 128; `gumbel_max_considered`: 16 → 32.** | Better MCTS targets = a stronger improvement operator, the diagnosed growth-rate bottleneck. DeepMind: 800–1,600 sims; we're at 64 over top-16 in a ~400-branch opening. Cost: self-play ~510 → ~1,020 s/gen ⇒ ~21 min/gen total, ~21 h for 60 gens on the 5090 — affordable. |
| 3 | **Arena games** | **`num_arena_matches`: 40 → 80.** | Halves gate noise (SE 8 → 5.6 pp), reducing the false rejections that froze the incumbent and staled the buffer. Costs ~44 s/gen — the cheapest phase; doubling it is negligible. |
| 4 | **Buffer** | **`replay_buffer_games`: 40k → 60k** (staleness 4 → 6 gens; reuse 4× → 6× at epochs=1). | More distinct games per training pass stabilises the value head and dilutes single-incumbent buffer collapse during rejection streaks. AGZ's window was ~20 iterations of games; 6 is still fresh. RAM: (60k+10k) × 256 KB ≈ 18 GB buffer + 6 GB overhead — fine on a typical 5090 cloud box (≥48 GB); the O8 pre-flight check will verify. Training pass grows to ~280 s/gen. |
| 5 | **Games/gen** | **Keep 10,000.** | The value head wants more games, but the sims doubling (rec 2) already claims the wall-clock budget, and #1/#2/#4 attack the same value-instability symptom more cheaply. Raise to 15k only if, after this run, value loss still rises late while policy loss falls. |
| 6 | **Net size** | **Keep `large` (192×12).** | No capacity signal: policy loss falling monotonically at run end, entropy healthy, Top-1 decline shown benign. `xl` (256×16) would ~2× training and inference cost for a ceiling we haven't reached. Per `deepmind-run-configs.md` §5 ("bigger nets learn slower per generation but raise the ceiling — judge them late"), move to `xl` when a healthy-LR run shows policy loss flattening or the Pentobi ladder stalls across two runs. |
| 7 | **Everything else** | Keep: α=0.03 (right for Blokus branching per the AZ scaling rule), gate threshold 0.55, epochs 1, batch 1024, K=64, `wave_plies` 32. | No evidence against any of them in this run. |

**Recommended next-run config** (delta from `blokus_cloud.json`):

```jsonc
{
  "num_generations": 60,
  "num_eps": 10000,
  "replay_buffer_games": 60000,          // was 40000
  "num_arena_matches": 80,               // was 40
  "mcts_config": {
    "num_mcts_sims": 128,                // was 64
    "gumbel_max_considered": 32          // was 16
  },
  "net_config": {
    "preset": "large",
    "learning_rate": 0.001,
    "lr_scheduler": "cosine"             // + eta_min = 1e-4 (one-line code change in _create_scheduler)
  }
}
```

Estimated ~21 min/gen ⇒ ~21 h for 60 generations on the 5090.

**Single highest-leverage change:** the LR floor. It costs nothing, and the data says the last ~15 generations of this run — a quarter of the compute — were spent training at learning rates between 1e-4 and 3e-6.

---

## 5. Metric follow-ups (not training config)

- **Refresh or supplement the eval set.** The frozen gen-1 eval set has done its job of exposing this confusion once; Top-1-vs-gen-1 should not be read as a strength signal again. Options: (a) rebuild the eval set every N generations and plot agreement per eval-set version, (b) additionally score agreement against *current-net MCTS* on the fixed positions (measures net-vs-own-search gap — the actually interesting quantity), or at minimum (c) relabel the chart to "agreement vs gen-1 targets". Same fix rehabilitates the value-calibration plot, which currently compares current values against gen-1 outcomes.
- **Raise or remove the internal Elo cap** (+1200 vs frozen gen-0 saturated from ~gen 18 — 40 of 58 generations produced no signal). Anchoring periodically to a mid-run frozen net, or relying on the Pentobi ladder as the primary external measure, both work.
