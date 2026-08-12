# Live scopes

One Worker session per scope. Ownership is **exclusive** — it is what stops two sessions fighting over
one PR. The Integrator keeps this current.

`src/alphablokus/config.py` is the one **shared** file; flag it in the PR title when you touch it.

Last reconciled: 2026-08-12.

---

| Scope | Plan | Owns these paths | Depends on |
|---|---|---|---|
| **`instruments`** | [`../docs/plans/evaluation-instruments.md`](../docs/plans/evaluation-instruments.md) | `scripts/pentobi_benchmark.py`, `scripts/measure_move_times.py`, `scripts/mini_ladder.py`, `scripts/width_shadow_probe.py`, `src/alphablokus/games/blokusduo/pentobi/**`, `src/alphablokus/evaluation/ladder_*.py`, `docs/10-EVALUATION-SPEC.md` | nothing |
| **`data-loop`** | [`../docs/plans/selfplay-data-and-loop.md`](../docs/plans/selfplay-data-and-loop.md) | `src/alphablokus/selfplay/**`, `src/alphablokus/storage/selfplay_store.py`, `src/alphablokus/training/**`, `src/alphablokus/cli.py` | nothing |
| **`value-head`** | [`../docs/plans/value-head.md`](../docs/plans/value-head.md) | `src/alphablokus/evaluation/colour_value.py`, the value-loss path in `src/alphablokus/games/base_wrapper.py` | **`data-loop` D1** |
| **`network`** | [`../docs/plans/network-experiments.md`](../docs/plans/network-experiments.md) | `src/alphablokus/games/blokusduo/nn/**`, `scripts/distill_sl.py`, `scripts/capacity_probe.py` | nothing |
| **`corpus`** | [`../docs/plans/corpus-scale-up.md`](../docs/plans/corpus-scale-up.md) | `src/alphablokus/games/blokusduo/pentobi/corpus*.py`, `store.py`, `move_values.py` | nothing |

## Two ownership collisions to watch

**`games/base_wrapper.py` is shared between `data-loop` and `value-head`.** The former owns the
training loop and dataset plumbing; the latter owns the value-loss term inside it. They must not both
have an open PR touching it — `value-head` is blocked on `data-loop` D1 anyway, so sequence them.

**`games/blokusduo/pentobi/` is shared between `instruments` and `corpus`.** `instruments` owns the
GTP client and the player (`gtp.py`, `player.py`, `book.py`); `corpus` owns the corpus builders
(`corpus*.py`, `store.py`, `move_values.py`). Same directory, disjoint files.

## Running order

**Three scopes are mutually independent** — `instruments`, `network`, `corpus` — and `data-loop` runs
alongside them. `value-head` waits on `data-loop` D1.

All five compete for **one GPU**, so [`box-queue.md`](box-queue.md) is the real constraint, not the
number of open sessions. Expect two or three Workers to be productive at once; more will simply queue.
