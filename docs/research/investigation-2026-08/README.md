# The 2026-08 plateau investigation

Why the training loop stopped improving, investigated 2026-08-03/04 by two independent agents
working from the same briefing, then reconciled. This is the record, not a plan — the work it
produced is tracked in [`../../plans/ROADMAP.md`](../../plans/ROADMAP.md).

**These files lived in gitignored `temp/` until 2026-08-10.** The whole investigation — the plan,
both bug-sweep audits, every correction — existed on one laptop for a week, and a returning reader
could not find any of it. That is why it is here now.

## Read in this order

| File | What it is |
|---|---|
| [`06-handoff.md`](06-handoff.md) | **Start here.** Status, open decisions, operational notes |
| [`03-synthesis.md`](03-synthesis.md) | The action plan the two reports were reconciled into |
| [`00-briefing-erratum.md`](00-briefing-erratum.md) | Corrections to the briefing. **Where this and the briefing disagree, this wins** |
| [`04-codex-bug-sweep.md`](04-codex-bug-sweep.md), [`05-fable-bug-sweep.md`](05-fable-bug-sweep.md) | Two independent audits, same JSONL schema so they diff |
| [`findings-codex.jsonl`](findings-codex.jsonl), [`findings-fable.jsonl`](findings-fable.jsonl) | The audits as data |
| [`stream-b-results.md`](stream-b-results.md) | Box measurements as they landed |
| [`probes-fable/`](probes-fable/) | Executed probe scripts from the audit |

## Health warnings

**[`00-briefing.md`](00-briefing.md) is a known-faulty source.** It is roughly 95% checkable and
checks out, but its defects propagated into three paid runs' worth of "next steps". Prefer the
erratum. Specifically: the 96.3% white-win figure is unreproducible (its raw data is gone), one
run's generation count is wrong, several figures are quoted to more precision than the instrument
supports, and two interpretive rankings are backwards.

**Two claims marked `[V, corrected 2026-08-04]` in `03-synthesis.md` overturn earlier
conclusions**, including one asserted as verified and wrong: the eval set's "~47 game lineages"
could not be reproduced, and per-position statistics on it are approximately correct after all.

**Superseded by later measurement (2026-08-10):**

- The reading that Pentobi *saturates* above level 9 is **wrong**. Engine-vs-engine, level 9 beats
  level 7 by 0.710 over 200 colour-balanced games (+156 Elo). The earlier flatness came from three
  of our own ladder cells whose intervals are ±87 Elo each — and this document's own plan had
  already called those cells meaningless.
- Elo figures computed by inverting a pooled score with a plain logistic are **understated**.
  Halving games between colours flattens the curve, so the true gap at level 9 is ~280 Elo, not
  ~220. Use `alphablokus.evaluation.ladder_elo`.
- "Stream A/B/C/D" labels are retired; the roadmap decodes them.

## What survived, and is still the basis for current work

- The plateau is **not** net capacity — a box probe found `xl` no better than `large`.
- The **value head has no demonstrable skill beyond guessing from whose turn it is**: its output
  tracks mover colour at 0.76–0.81 while outcomes track it at only 0.52–0.62, and both production
  checkpoints' intervals bracket zero. This is the leading hypothesis.
- **No healthy training operator has ever been pointed at the current net** — every stalled run
  since has a since-diagnosed defect.
- The **eval set was never actually held out**: its positions were sampled from the training buffer
  and never removed, no source game id was recorded, and the draw was not reproducible at a fixed
  seed. So every "internal signals looked healthy" observation in the briefing should be read as
  *uninformative*. Fixed in PR #69.
