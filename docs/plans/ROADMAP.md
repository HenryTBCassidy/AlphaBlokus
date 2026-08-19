# Roadmap — what is in flight, what it is called, and where it stands

The index for `docs/plans/`. Read this first; it tells you which plans are live, what each
workstream is for, and what the next decision is. Every plan doc is a checklist; this is the
map over them.

**Folder convention.** `docs/plans/` holds only what is **actively being worked**.
`docs/plans/future/` holds plans that are real but parked — deliberately not started, with the
reason recorded in each. `docs/plans/archive/` holds finished work, kept for context. A plan moves
out of the top level the moment nobody is working it; if you find something at the top level that
nobody is touching, move it rather than leaving it to rot.

Last updated: 2026-08-10.

---

## Naming

**Workstreams have names, not letters.** An earlier triage (2026-08-04) split the work into
"Stream A/B/C/D", which made every status update unreadable. Those letters are retired. The
table below decodes them so older documents and commit messages still parse, but **do not use
them in new writing**.

| Old label | Name to use | What it was for |
|---|---|---|
| Stream A | **Instruments** | Repair the measurement tools before running anything else |
| Stream B | **Free measurements** | Box experiments that cost £0 and gate all spending |
| Stream C | **Bug sweep** | Two independent audits of the training/self-play code |
| Stream D | **Value head** | Candidate fixes for the leading hypothesis about why training stalled |
| F1–F11 | **Benchmark integrity** | Make the Pentobi comparison fair and self-describing |

Item IDs use a **unique prefix per plan** so an ID is unambiguous project-wide — `I` instruments,
`D` data-loop, `H` value-head, `E` network, `V` corpus. See
[`../guides/PLAN-FORMAT.md`](../guides/PLAN-FORMAT.md). IDs in archived plans (F1, N6, S7…) stay as
they were written.

---

## The five active plans

Each is owned by one **scope**, executed by the roles in [`../../agents/README.md`](../../agents/README.md).
Every checklist row names the roles that carry it — see the Role column in
[`../guides/PLAN-FORMAT.md`](../guides/PLAN-FORMAT.md).

| Plan | Scope | What it answers | Depends on |
|---|---|---|---|
| [`evaluation-instruments.md`](evaluation-instruments.md) | `instruments` | How strong are we, and is the measurement honest? | nothing |
| [`selfplay-data-and-loop.md`](selfplay-data-and-loop.md) | `data-loop` | Does the training loop improve the net at all? | nothing |
| [`value-head.md`](value-head.md) | `value-head` | Why can the net not play from behind? | `data-loop` D1 |
| [`network-experiments.md`](network-experiments.md) | `network` | Do auxiliary targets help, and is the trunk the constraint? | nothing |
| [`corpus-scale-up.md`](corpus-scale-up.md) | `corpus` | Does more data help? (**measured: yes**) | nothing |

**Three can run at once** — `instruments`, `network` and `corpus` are mutually independent; `data-loop`
runs alongside them and unblocks `value-head`. All five compete for one GPU, so the queue is the real
constraint, not the number of sessions.

### The one hard dependency

`data-loop` **D1** — recording whose turn it is on every stored position — must land before **D3**
generates any data. Both replay buffers were deleted, so the schema change is nearly free right now and
that window shuts the moment new data exists. `value-head` cannot start until D1 lands.

### The tension worth arbitrating

`corpus-scale-up` carries the only *measured* priority claim in the project: we are data-limited, and
"generate more games" outranks all technique work. Everything in `network-experiments` and `value-head`
is technique work. The resolution taken here is to run corpus generation as the **background stream** —
it is a 3-day box job needing little attention — rather than to pause the technique work behind it.

---

## Status at a glance

| Workstream | State |
|---|---|
| Instruments (was Stream A) | ✅ Done — PR #69 |
| Bug sweep (was Stream C) | ✅ Done — PR #70 |
| Benchmark integrity (F1–F9) | ✅ Done — [`archive/fair-pentobi-benchmark.md`](archive/fair-pentobi-benchmark.md); remainder is `instruments` I2/I3 |
| The five plans above | 🔄 Reorganised 2026-08-12, none started |

## What we learned in the 2026-08 investigation

The full record is in [`../research/investigation-2026-08/`](../research/investigation-2026-08/)
— read [`06-handoff.md`](../research/investigation-2026-08/06-handoff.md) then
[`03-synthesis.md`](../research/investigation-2026-08/03-synthesis.md). It lived in gitignored
`temp/` until 2026-08-10, which is why it was unfindable; treat
[`00-briefing.md`](../research/investigation-2026-08/00-briefing.md) as **superseded** by
[`00-briefing-erratum.md`](../research/investigation-2026-08/00-briefing-erratum.md).

The load-bearing conclusions, with what has changed since:

1. **The plateau is not capacity.** A box probe found `xl` no better than `large`. Net size is
   not the constraint.
2. **The value head has no demonstrable skill beyond guessing from whose turn it is.** Its
   output tracks mover colour at 0.76–0.81 while outcomes track it at only 0.52–0.62. Both
   production checkpoints' confidence intervals bracket zero skill. This is the leading
   hypothesis and the reason the **Value head** workstream exists.
3. **A healthy training operator has never been pointed at the current net.** Every stalled run
   since has a since-diagnosed defect. That is what the pilot tests.
4. **Pentobi's difficulty levels are a hardcoded simulation count** —
   `counts_duo = {3, 21, 77, 213, 861, 7280, 221867, 1109339, 5546695}` in
   `libpentobi_mcts/Player.cpp`. The level 6→7 step multiplies its search by **30×** while every
   other step is 3–8×, so the ladder is not an evenly spaced difficulty scale and reading it as
   one was misleading.
5. **Pentobi's opening book had never been active** — its binary looks for book files beside
   itself and the build directory held none, while the engine still reported `use_book 1`. Every
   Pentobi number in the project's history faced a book-free opponent. Fixed 2026-08-05 by
   symlinking the books; the strength this is worth is unmeasured (V11).
6. **Level 9 is genuinely ~156 Elo stronger than level 7** (measured 2026-08-10, 200
   colour-balanced games, engine vs engine, pooled score 0.710, CI [0.647, 0.773]). An earlier
   claim that Pentobi *saturates* above level 7 was wrong — it came from three of our own ladder
   cells whose intervals are ±87 Elo each. Realised effort is 13.9× between those levels, against
   a tabled 25×, so quote realised effort whenever a ratio is load-bearing.
7. **The fair fight is measured (2026-08-11).** At equal thinking time, with Pentobi's book on, our
   best net scores **0.315** at level 9 (CI [0.232, 0.411]) — so we lose, and it is resolved. Two
   refinements to what was previously written here: 10× more search moved us from 0.22 to 0.315
   against a *stronger* opponent, so **"search is not the lever" was too strong** — it is real but
   insufficient. And the split by colour is the actual finding: **0.630 as first mover (we beat level
   9), 0.000 as second mover across 50 games.** A colour-handling defect has not yet been ruled out.
8. **Previously written here, now superseded by (7):** Every net-vs-Pentobi
   number the project owns was taken at **400 simulations for us against Pentobi's own budget**,
   with Pentobi thinking ~12× longer per move at level 9. On that footing the best net is ~280 Elo
   below level 9 once first-mover advantage is corrected for. Estimating from the ~3.6 doublings of
   thinking time involved, equal time would buy ~70–145 Elo — which would leave a gap, and is the
   basis for "search is not the lever; the network is the constraint". **That estimate has never
   been tested. F9 is the experiment that tests it and it has not been run.** Until it has, treat
   the ranking of the value head above further search work as well-founded but not proven.

Conclusion (6) plus (7) is why **Value head** is now ranked above further measurement.

---

## The next decision

Merge order and dependencies live in the five-plan table above. The immediate sequence:

1. **`data-loop` D1** — record whose turn it is on every stored position. The schema window is open
   only while both replay buffers are absent; it shuts the moment D3 generates data.
2. **`instruments` I1** — extend the `--condition` enum, which blocks I2.
3. **Drain the box queue** — I2, I3, I4 are queued and the box is idle.
4. **`value-head` H1** — pure analysis, no code, and it decides whether the second-mover collapse is a
   *data* problem or a *learning* problem. Those imply completely different work.

An honest outcome that remains on the table: if the pilot (D5) comes back within noise, and then
comes back within noise again with the value-head fixes layered on, the remaining levers are
architectural. At that point scaling the goal down to level 6–7, or stopping, is a legitimate result
rather than a failure — and should be called plainly rather than rationalised into another run.
