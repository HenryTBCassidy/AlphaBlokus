# Box queue

Workers append jobs here. The [GPU-runner](gpu-runner.md) drains it top-down within priority, and
moves finished entries to the Done log at the bottom with a pointer to
[`box-results.md`](box-results.md).

**There is one GPU.** Jobs run one at a time. A job without a **kill condition** gets bounced.

## Format — copy this block

```
### <id> — <one-line description>
- **Requested by:** <scope>            - **Priority:** critical | high | medium | low
- **Plan item:** <plan doc> <item id>
- **Question it answers:** <one sentence — what changes depending on the outcome>
- **Command:** <exact command, including the config and every flag>
- **Expected cost:** <wall-clock, workers, RAM>
- **Kill condition:** <what result means stop rather than continue>
- **Prerequisites:** <other queue ids, or none>
```

---

## Pending

### F12 — is 0-for-50 as second mover real, or a colour-handling bug?
- **Requested by:** benchmark — **Priority:** **critical** — it gates how F9 is read
- **Plan item:** `docs/plans/fair-pentobi-benchmark.md` F9 follow-up
- **Question it answers:** F9 shows our net winning 0 of 50 games as second mover against level 9,
  and 30 of 50 as first mover. Either the net genuinely cannot play from behind — which would be the
  strongest evidence yet for the value-head hypothesis — or the fair-fight path mishandles colour.
  Everything downstream depends on which.
- **Command:** `scripts/pentobi_benchmark.py --config run_configurations/blokus_cloud_v3_eval.json
  --net accepted_40.pth.tar --levels 1,2 --games 20 --sims 400 --nobook --condition colour-check
  --workers 4 --seed 23` — then read the per-colour split in the payload.
- **Expected cost:** ~15 min. Trivial.
- **Kill condition:** if the net also wins ~0 as second mover against **level 1**, stop and treat it
  as a bug — a net that beats level 9 as White cannot legitimately lose to level 1 as Black.
- **Prerequisites:** none.

### F10 — search-scaling slope at levels 8 and 9
- **Requested by:** benchmark — **Priority:** medium
- **Plan item:** `docs/plans/fair-pentobi-benchmark.md` F10
- **Question it answers:** how many Elo does *our* net gain per doubling of its own search? Decides
  whether more search is a route to level 9 at all, and how far F9's single point can be extrapolated.
- **Command:** `scripts/pentobi_benchmark.py --config run_configurations/blokus_cloud_v3_eval.json
  --net accepted_40.pth.tar --levels 8,9 --games 100 --sims {400,1600,6400} --book
  --condition search-scaling --workers 6 --seed 11` (three runs, one per sim count)
- **Expected cost:** ~10–14 h total at 6 workers. RAM ~22 GB.
- **Kill condition:** if the 400→6,400 slope is under ~10 Elo per doubling, search is not the lever —
  stop and do not add a 25,600 arm.
- **Prerequisites:** none. **Note:** opening diversity shrinks as sims rise, so record distinct
  ply-8 positions per arm; the noise floor grows along the treatment axis.

### F11 — what is Pentobi's opening book worth?
- **Requested by:** benchmark — **Priority:** medium
- **Plan item:** `docs/plans/fair-pentobi-benchmark.md` F11 (= `future/pentobi-corpus-v2.md` V11)
- **Question it answers:** converts every historical book-free number onto the "as shipped" scale.
  Without it, pre- and post-2026-08-05 results sit on two scales with an unknown offset.
- **Command:** `twogtp` book-on L9 vs book-off L9, **two colour-swapped batches of 100** (twogtp has
  no colour-alternation flag, and Duo's first mover takes ~75% of decisive games).
- **Expected cost:** ~4 h, 6 engines, no GPU.
- **Kill condition:** none needed — any result is informative.
- **Prerequisites:** none.

### M5 — bfloat16 vs float32 self-play A/B
- **Requested by:** benchmark — **Priority:** low, but it has slipped three times
- **Plan item:** `docs/plans/ROADMAP.md` M5
- **Question it answers:** production self-play runs bf16 while every correctness test runs fp32
  CPU. Top-64 selection over 17,837 near-tied logits has never been checked under the numerics we
  actually ship.
- **Command:** `scripts/validate_jax_search.py --dtype bfloat16` and `--dtype float32`, same
  checkpoint and positions.
- **Expected cost:** hours.
- **Kill condition (pre-registered):** ≥99% top-64 overlap **and** target-distribution KL below the
  noise floor closes the question. Anything worse and self-play moves to fp32 or a mixed policy.
- **Prerequisites:** none.

---

## Done

### F9 — the fair fight: gen-40 vs Pentobi L9 at equal thinking time ✅ 2026-08-11
100 games, 4,096 sims, book on (probe confirmed engaged), 6 workers, 7,407 s.
Raw numbers in [`box-results.md`](box-results.md#f9). **Awaiting Analyst.**

### F8 — time-parity calibration ✅ 2026-08-11
Parity at ~4,100 sims. Recorded in `docs/plans/fair-pentobi-benchmark.md` F8.

### F2 — Pentobi L7 vs L9 head-to-head ✅ 2026-08-10
200 colour-balanced games. L9 scores 0.710. Recorded in the plan.

### F3 — Pentobi's realised effort per move ✅ 2026-08-10
195 vs 14 CPU-s/game = 13.9× (table implies 25×). Recorded in the plan.
