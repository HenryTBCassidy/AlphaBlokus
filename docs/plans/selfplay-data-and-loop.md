# Self-play data and the training loop

Owns the data path — what a self-play game records — and whether the training loop improves the net
at all. **Everything that produces or consumes self-play data goes through this plan**, so no two
sessions fight over the same schema.

**Scope owner:** `data-loop`. **Blocks [`value-head.md`](value-head.md)**, which needs D1's schema
change before it can start.

---

## Checklist

| # | Item | Role | Effort | Priority | Done |
|---|---|---|---|---|---|
| D1 | Record whose turn it is on every stored position | `W` | 1–2 d | **Critical** — window closing | |
| D2 | A generate-only entry point (frozen weights, no training) | `W` | ½ d | **Critical** | |
| D3 | Generate a ~5,000-game dataset with the current best net | `W→R` | 80 min box | High | |
| D4 | Learning-rate sweep on the frozen dataset | `W→R→A` | 9 short jobs | High | |
| D5 | **The pilot** — 15–20 generations with a healthy training step | `W→R→A` | 2–3 d box | **Critical** | |
| D6 | Games-per-generation arm | `W→R→A` | 2–3 d box | Low | |

---

## D1. Record whose turn it is — and do it before D3

`ProcessedExample` is `tuple[compact_board, (indices, probs), value]`. Nothing records which side was
to move. Thread it through `selfplay/episode.py` → `storage/selfplay_store.py` →
`training/replay_buffer.py` → the dataset.

**Why this is first and why it is urgent.** Both historical replay buffers were deleted, so there is
**no legacy data to migrate** — the schema change is nearly free right now. The window shuts the
moment D3 generates data: after that, the one dataset everything downstream depends on would lack the
column and we would have to regenerate or work around it.

**Do not use the inference shortcut.** `infer_mover_colour()` exists and derives colour from piece
parity, but its own docstring notes the parity "breaks and the colour is genuinely unrecoverable" once
a player has passed — and passing happens in the **endgame**, precisely where the value signal is
strongest. Thread the real value.

Threading `player` through also removes the piece-parity workaround in the colour-value diagnostic.

## D2. A generate-only entry point

There is currently no way to produce self-play games without also training on them — every path calls
`generate_games` and then trains. That single gap blocks three separate things: D4's sweep has no
dataset, there is no genuinely held-out eval set, and the question of whether the value head
*degraded* or was merely graded against another net's games cannot be settled.

Frozen weights, no gradient updates, no acceptance gate.

## D3. Generate the dataset

~5,000 games with the current best net, frozen. ~80 minutes on the box, £0. **Must land after D1** so
the dataset carries the colour column from birth.

Also produces the fresh, genuinely held-out eval set that PR #69's machinery can now build — the old
one was sampled from the training buffer and never removed from it, so every historical "internal
signals looked healthy" reading is uninformative rather than reassuring.

## D4. Learning-rate sweep

AdamW at 1e-4 / 3e-4 / 1e-3 × 3 seeds, one frozen dataset, same starting checkpoint. Judged on
**held-out games** — never training loss, the metric that lied throughout the last run.

**Judged against (pre-registered):** if all three rates land within the 0.05-nat held-out CE noise
floor, the learning rate is not the lever. It stays as a cheap hedge but stops being treated as a fix.

## D5. The pilot

15–20 generations on the box, warm from the best net, changing **only the training step**: the rate
from D4, AdamW weight decay 1e-4, one epoch, acceptance gate off, keep-best-by-ladder every 5
generations, drift breaker armed. Buffer held at 60k so it is not a confound. Ladder at generations
0 / 5 / 10 / 15, pre-registered.

**Why it matters more than it sounds.** Every stalled run in this project's history has a
since-diagnosed defect — two were frozen by a gate demanding something measurably impossible, another
trained with settings the project's own post-mortem calls toxic. **A healthy training operator has
never once been pointed at the current net.** Until that baseline exists, every fancier experiment is
uninterpretable: a fix layered on a loop that does not improve tells you nothing.

**Judged against (pre-registered):** the pilot succeeds if its best checkpoint beats the warm-start
checkpoint by **≥0.031 weighted ladder score, both measured under the same scoring convention and
level range**. Expressed as a difference on purpose — the old absolute bar silently got easier when
draw scoring changed, and the measurement has now changed once.

Config already exists: `run_configurations/blokus_pilot_b6.json`.

## D6. Games-per-generation arm

2,500–5,000 games per generation at the chosen rate with a proportionally smaller buffer. 10,000 has
never been varied in the project's history, and the operator improves per *iteration* — so smaller
generations buy 2–4× more improvement iterations per GPU-hour.

**Judged against:** if the ladder slope per GPU-hour is no better than D5's, keep 10,000 and close it.
