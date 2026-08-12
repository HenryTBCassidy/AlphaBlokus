# Evaluation instruments

Keeps the measuring tools honest and answers "how strong are we, really". Nothing here changes the
network — that is deliberate, so a measurement can never be confounded by the thing it measures.

**Scope owner:** `instruments`. **Box-heavy and otherwise standalone** — this plan depends on nothing
and nothing depends on it, so it can run alongside any other stream.

Predecessor: [`archive/fair-pentobi-benchmark.md`](archive/fair-pentobi-benchmark.md) — F1–F9 landed
there, including the fair-fight result and the parity calibration. Read it before touching anything
here. Spec: [`../10-EVALUATION-SPEC.md`](../10-EVALUATION-SPEC.md).

---

## Checklist

| # | Item | Role | Effort | Priority | Done |
|---|---|---|---|---|---|
| I1 | Extend `--condition` beyond two values | `W` | 1 h | **High** — blocks I2 | |
| I2 | Search-scaling slope: does *our* search convert into strength? | `W→R→A` | 10–14 h box | **High** | |
| I3 | What Pentobi's opening book is worth in Elo | `R→A` | 4 h box, no GPU | Medium | |
| I4 | bfloat16 vs float32 self-play A/B | `R→A` | hours | Medium | |
| I5 | Rebase and smoke-test the width probe, then run it | `W→R→A` | 15–30 h box | Low | |

---

## I1. Extend `--condition` beyond two values

`--condition` accepts only `ladder` and `fair-fight`, but the queue already needs `search-scaling`,
`colour-check` and `book-delta`. The mechanism exists to stop incomparable scales being pooled, and
with two labels genuinely different experiments are forced to share a directory — which is the exact
thing it was built to prevent. The colour-check run had to be mislabelled `fair-fight` for this reason.

Touch `CONDITION_DIRNAMES` in `scripts/pentobi_benchmark.py`. **`is_longitudinal()` in
`evaluation/ladder_selection.py` must keep excluding everything except `ladder`** — that filter is
what stops a one-off comparison driving promotion or tripping the drift breaker.

## I2. Search-scaling slope

400 / 1,600 / 6,400 simulations at levels 8 and 9, 100 games each, book on.

**The question.** The fair fight showed 10× more search moving us from 0.22 to 0.315 at level 9.
Does that continue, or flatten? It decides whether search is a route to level 9 at all, and how far
the single fair-fight point can be extrapolated.

**Judged against (pre-registered):** if the 400→6,400 slope is under ~10 Elo per doubling, search is
not the lever — stop, and do not add a 25,600 arm.

**The trap.** Opening diversity comes only from sampling the net's first four plies from the visit
distribution, and that distribution *sharpens* as simulations rise. So a 6,400-sim arm may play far
fewer distinct games than a 400-sim arm — the noise floor grows along the treatment axis. Record
distinct ply-8 positions per arm or the comparison is not sound.

## I3. What the opening book is worth

Book-on level 9 versus book-off level 9, engine against engine, **two colour-swapped batches of 100**
— `twogtp` has no colour-alternation flag and Duo's first mover takes ~75% of decisive games, so an
unswapped run measures colour rather than the book.

**Why it matters beyond curiosity:** the book was inactive for the project's entire history until
2026-08-05. Every pre-2026-08-05 Pentobi number and every post- number therefore sit on two scales
with an unknown offset. This measures the offset. Until it exists, the two cannot be compared.

## I4. bfloat16 vs float32 self-play

Production self-play runs bf16 while every correctness test runs fp32 on CPU. Top-64 selection over
17,837 near-tied logits has never been checked under the numerics we actually ship.

`scripts/validate_jax_search.py --dtype bfloat16` and `--dtype float32`, same checkpoint and
positions. **Judged against (pre-registered):** ≥99% top-64 overlap *and* target-distribution KL
below the noise floor closes the question. Anything worse and self-play moves to fp32 or a mixed
policy.

This has slipped three times. It is short and free.

## I5. The width probe

`scripts/width_shadow_probe.py` is on main, reviewed, and its two defects fixed — including one that
would have made it measure completed-Q from a *different* Gumbel draw than the one that chose the
action, silently answering the width question wrongly.

It has still **never been run**. It needs a CPU smoke test first, then the real run at spec settings.
Deliberately last: it informs a later run, is on nobody's critical path, and costs 15–30 box hours.
