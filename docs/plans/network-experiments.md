# Network experiments — auxiliary targets, then the trunk

Three auxiliary heads are built, default off, and **effectively unmeasured** — the one experiment that
was meant to judge them was invalid. Until that is fixed, the most expensive architectural change in
the backlog is gated on a question nobody has answered.

**Scope owner:** `network`. Independent of the data and value-head streams, so it can run in parallel
with either.

Predecessors: [`archive/supervised-network-improvements.md`](archive/supervised-network-improvements.md),
[`archive/score-auxiliary-target.md`](archive/score-auxiliary-target.md). Evidence:
[`../research/alphazero-technique-review.md`](../research/alphazero-technique-review.md).

---

## Why this plan exists: the measurement was invalid

The four-arm comparison of the auxiliary heads was run **directly through `distill_sl.py` rather than
the A/B harness**, while the corpus was still being written. The plan's own caveat:

> *"each arm globbed a different number of shards — 11,804 scored rows for one arm and 13,165 for
> another. **The arms sat different exams.**"*
>
> *"The weight-0 arm — which changes nothing that can affect the policy — moved further from the
> baseline than the real arm did."*

A control that should have been identical moved *more* than the treatment. So the comparison measured
nothing, three heads remain undecided, and **N8's gate — "do global pooling only once the auxiliary
targets show the trunk is the constraint" — has never been evaluated.**

There is a second, separate finding from the same run that *does* hold and outranks everything here:
the data-fraction curve showed **we are data-limited, decisively** (doubling the corpus buys 0.18–0.35
nats of held-out CE with no flattening), and concluded *"generate more games outranks all of them"*.
Read every result below against that. A marginal gain here is not evidence the technique is weak.

---

## Checklist

| # | Item | Role | Effort | Priority | Done |
|---|---|---|---|---|---|
| E1 | Freeze a corpus snapshot so arms sit the same exam | `W` | ½ d | **Critical** — everything below depends on it | |
| E2 | Re-run the auxiliary-head arms through the harness | `W→R→A` | ½ d + box | **Critical** | |
| E3 | Decide each head: keep, drop or retune — recorded with numbers | `A` | — | High | |
| E4 | Global pooling in the trunk | `W→R→A` | 3–4 d + box | Medium — **gated on E3** | |

---

## E1. Freeze a corpus snapshot

The invalid experiment's root cause was that the corpus grew *while the arms ran*, so each arm globbed
a different number of shards. Pin an explicit shard list, assert every arm loads an identical row
count, and fail loudly if not.

This is cheap and it is the difference between an experiment and a coincidence. Do it first.

**Include the weight-0 control arm** — a head at loss weight 0 cannot affect the policy, so it must
land on the baseline. Last time it drifted further than the treatment did, which is exactly how the
invalidity was caught. Keep it as the harness's own smoke test.

## E2. Re-run the auxiliary-head arms properly

Through `scripts/capacity_probe.py`-style harness discipline: same shards, same seeds, same epochs, one
metric set, one command. Arms:

| Arm | What it adds | Published evidence |
|---|---|---|
| baseline | — | |
| weight-0 control | a head, contributing nothing | must equal baseline |
| **score / margin** | predict the final score margin | KataGo ablates score+ownership jointly at **1.65×** |
| **ownership** | per-cell final board owner — ~196 labels per position instead of 1 | same ablation; the margin is the sum of this map |
| **opponent reply** | predict the reply distribution | KataGo **1.30×**, and its provenance is *supervised* (Darkforest) |

**Why these are the right attack on the value head's problem.** Every position in a game carries the
same outcome label, so the value head cannot tell an open opening from a decided endgame. Ownership and
margin supply exactly that within-game discrimination — and unlike a teacher's evaluation, they are
**facts about the final position**.

**Judged against (pre-registered):** held-out policy CE and value skill, with the 0.05-nat CE noise
floor applied. A head that moves nothing beyond the floor is dropped, not retained "just in case".

## E3. Decide each head

Three heads have been built and none decided; the project now has a habit of accumulating unmeasured
optional machinery. Each head gets a written verdict with its numbers: **keep and default on, drop and
delete, or retune with a stated reason.**

This row is the gate for E4. If no auxiliary target moves anything, enriching the trunk further is
unlikely to either — and E4 is expensive enough that the gate is worth respecting.

## E4. Global pooling in the trunk

Layers that summarise the whole board and feed that summary back to every square. A plain convolution
stack only sees local neighbourhoods, so board-wide facts — space remaining, piece inventory, tempo
parity, phase — are inferred slowly and badly. Our trunk has no global path at all.

KataGo ablates this at **1.60×**, its second-largest factor; Lc0 has shipped the equivalent in every
network since T60.

**Why it is last.** The net exists in three places — torch, the jax bridge, the ONNX export — so a
trunk change touches all three plus the parity tests, and **it breaks warm-starting**, meaning it
cannot be evaluated as a continuation of the current net. Only worth it once E3 says the trunk is
actually the constraint.
