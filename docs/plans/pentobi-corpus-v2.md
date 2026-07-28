# Pentobi distillation corpus v2 — strong openings + soft targets

The v1 expert corpus ([`pentobi-distillation.md`](pentobi-distillation.md) D1–D5) generated cleanly,
validated perfectly, and **failed the D8 ladder gate**: the distilled nets (`96x6` weighted 0.088,
`256x16` similar) came out far below v3 gen-40 (0.344) — ~55% at Pentobi L1, ~0% at L6+. The sizing
sweep ruled out capacity (18× params bought +1.8 pp top-1). This plan replaces the corpus
*generator*, not the training half: D6/D7's dataloader and SL trainer stay, D8's gate stays, and the
v1 shards stay on disk as a mid-game supplement. Everything below is Blokus Duo / `pentobi-gtp` only.
Companion documents: [`corpus-search-space-store.md`](corpus-search-space-store.md) (the persistent
search-space DAG + allocation-plan store this plan builds on) and
[`../research/corpus-generation-literature.md`](../research/corpus-generation-literature.md) (what
AlphaGo/KataGo/Lc0/Stockfish-NNUE/ChessBench and the imitation-learning literature say about
engine-generated corpora — several defaults below cite it).

**Two root causes, one fix each:**

| v1 defect | v2 fix |
|---|---|
| Openings were a **uniform-random 4-ply prefix** over all ~414 legal first moves — mostly junk, off-distribution, and **never harvested**. The net saw zero opening signal and trained only on "recover from a bad start" positions. | Openings come from a **budget-proportional allocation over Pentobi's own search distribution** (a persistent, position-keyed DAG whose depth *emerges* from the game budget), and **every opening position is harvested**. |
| Policy target was the **single played move, one-hot** — Pentobi's rich move preferences were computed on every ply and thrown away. | Harvest the **full `move_values` distribution** at every ply as a soft policy target. |

**Strategic thesis (Henry).** Our edge over Pentobi is opening/positional pattern recognition;
Pentobi's edge is endgame calculation. The route to beating it is *better openings* — so the corpus
must carry strong **and varied** openings, labelled from ply 1. And because the goal is to *surpass*
the teacher, not match it: allocation follows **Pentobi's own opinion** (visit share, flattened),
so that playing its preferred lines out tells us where that opinion was wrong; the independent
evaluations we have collected are **measurement and validation only, never a generator input**
(Henry's ruling, 2026-07-28). The preference-vs-outcome disagreement the corpus captures is the
signal that lets a later net exceed the teacher; pruning or "correcting" it at generation time
would be self-defeating.

> **⚠ Colour convention.** **White = player 1 = the first mover** (`board.py`, `+piece_id`,
> `episode.py current_player=1`). GTP `b` = our White, GTP `w` = our Black. The Pentobi book SGF uses
> Go's convention (`B` listed first = first player), so **book `B` = our White**. Do not say "Black
> goes first" anywhere in this work.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| V1 | GTP layer: `reg_genmove` / `move_values` + `MoveValues` parser (strip `[PIECE]`, signed values) + fixture tests; make `--nobook` an explicit `PentobiGtp` flag | 2 h | High | ✅ |
| V2 | **Confidently-wrong base-rate probe** (top-8 children of ~30 allocated nodes independently evaluated) + residual engine probes (argmax-vs-`genmove`, pass/terminal edge cases, drive-pattern overhead) | 4 h box | High | |
| V3 | Search-space store: execute [`corpus-search-space-store.md`](corpus-search-space-store.md) S1–S6 (SQLite DAG + allocation plans + playout registry + export + coverage) | 1.5 days | High | ✅ |
| V4 | Phase A — `plan`: budget-proportional allocation (`w ∝ p^(1/T)`, split floor R), emergent depth, search-on-demand mapping, mirror-pair merging, book-line floors | 1 day | High | ✅ |
| V5 | Phase B — `generate`: fulfilment-driven scheduling against the active plan, prefix replay, harvest **every** ply, full-strength continuations | ½ day | High | ✅ |
| V6 | Schema v2 (games shards + `export-opening` parquet, plan provenance in footers), validator, `docs/07-DATA-STORAGE.md` | ½ day | High | ✅ |
| V7 | CLI + diagnostics: v2 subcommands, plan-fulfilment/coverage report, opening-vs-midgame row ratio, target-entropy / duplicate-position metrics | 3 h | High | ✅ |
| V8 | `link` pass: aggregate playout outcomes up the DAG into `outcome_mean`/`outcome_count` | 2 h | Medium | ✅ |
| V9 | Trainer: soft-target load path, target temperature τ, **opening-subtree holdout split** (fixes a latent leak), opening-value target choice, source mix weights | 1 day | High | |
| V10 | L9 pilot on the box (plan at B=1,000 + ~200 games): validate, measure, freeze knobs | 4 h box | High | |
| V11 | **Book-strength measurement**: enable the opening book, verify engagement, book-on L9 vs book-off L9, spell out benchmark consequences | 4 h box | Medium | |
| V12 | Stage-1 v2 corpus generation on the box — **(B = 10,000, T = 2, R = 2)**, ~3-day run — + `corpus_wrapup.py` to R2 (verify-before-done; the store DB syncs with the shards) | 3 days box | High | |
| V13 | Breadth-vs-replication ablation as a **stage-1 subset experiment** (zero extra generation; informs the top-up shape) | 1 day box GPU | Medium | |
| V14 | SL re-fit on v2 (warm + scratch, τ sweep, opening-value arm, opening mix-weight arm, ± v1 mid-game mix; report top-1 **and top-3** agreement) | 1 day box GPU | High | |
| V15 | D8 ladder gate re-run vs v3 gen-40 — **the gate for everything below** | ½ day box | High | |
| V16 | (gated on V15) Phase 2: net-in-the-loop opening proposal + pentomino enumeration filtered by `move_values` | TBD | Medium | |

**Gate:** V15 is D8's criterion unchanged — **+10 pp at any of L5–L7 after SL alone**, mini-ladder
L1–L9 × 50 games × 400 sims against the v3 gen-40 baseline. If v2 does not move the ladder, the
distillation thesis (not just the generator) is what's wrong, and Phase 3 RL spend stays blocked.

---

## What `move_values` actually gives us (measured 2026-07-27/28, box, L9)

`pentobi_gtp/GtpEngine.cpp::cmd_move_values` dumps the **root children of the last search**, sorted
strongest-first, one per line:

```
= 1019897 1019901.0 0.709 [F]f8,e9,f9,g9,e10
   119575  119578.0 0.703 [X]f9,e10,f10,g10,f11
     ...
  <visits>  <value_count>  <value>  [PIECE]<cells>
```

Raw evidence lives in `local/probes/` (`all_cands.txt`, `all_eval.txt`, `cands.txt` +
`cand_eval2.txt`, `branch_1..8.txt`, `reply_cands.txt` + `reply_eval.txt`, `allocation_sim.py`,
plus the scripts that produced each). Measured facts that drive the design:

1. **It is free.** Pentobi builds this tree on every search anyway; `move_values` just prints it.
2. **`reg_genmove` populates it** (search without playing) and **`undo` exists** — so the drive
   pattern is `reg_genmove <c>` → `move_values` → `play <c> <our choice>`. We can harvest the expert
   distribution at a position and then play a *different* move, which is the whole point.
3. **The move string carries a piece-name prefix** (`[F]`, `[L5]`, `[T5]`…) that
   `PentobiMoveTranslator.cells_to_action` will not parse. Strip everything through `]`.
4. **Coverage is partial, and now exactly quantified: 315 root children vs 414 legal first moves.**
   After mirror canonicalisation the 414 legal moves are **212 distinct positions** (10
   self-symmetric); Pentobi's 315 searched children collapse to **160 of those 212** — so **52
   distinct first positions are entirely outside its search**, never labelled and never played.
   Our soft target's support is a *subset* of the legal set (validator asserts `support ⊆ legal`,
   never equality), and the 52-position hole is a **recorded coverage gap** that only V16's
   net-in-the-loop phase can close.
5. **Value semantics.** Values are win-probability-like for the side to move, roughly [0, 1] but not
   bounded (they go negative deep in games); the first mover's best opening evaluates ≈ 0.72 and
   the reply position ≈ 0.28–0.31 for the second player — approximately complementary. Values are
   only meaningful for **visited** children: unvisited ones report `value_count 3.0` and the
   prior's value (often exactly `0.500`). A position's backed-up value must be read as the **top
   child's value** from `move_values`; GTP `get_value` is useless — it returns a constant `0`
   because Pentobi never updates the root node's own value.
6. **Visit distributions are hyper-concentrated at every ply, not just the opening.** One full L9
   self-play walk, plies 1–14 (`--threads 1`, `--nobook`):

| Ply | children | root visits | top-1 visit share | effective moves `exp(H)` | k@90% | top-16 mass | top-32 mass | strong set at `visits ≥ 1% of top` | search time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 315 | 1.62 M | 0.524 | 7.4 | 8 | 0.937 | 0.967 | 10 | 29.6 |
| 2 | 315 | 1.22 M | 0.915 | 1.7 | 1 | 0.977 | 0.983 | 2 | 23.5 |
| 3 | 422 | 3.08 M | 0.618 | 4.1 | 4 | 0.965 | 0.987 | 6 | 22.5 |
| 4 | 363 | 2.20 M | 0.953 | 1.4 | 1 | 0.992 | 0.997 | 2 | 6.2 |
| 5 | 394 | 2.41 M | 0.969 | 1.3 | 1 | 0.995 | 0.997 | 1 | 6.9 |
| 6 | 390 | 2.40 M | 0.978 | 1.2 | 1 | 0.994 | 0.997 | 1 | 21.7 |
| 7 | 383 | 3.20 M | 0.661 | 3.9 | 4 | 0.980 | 0.996 | 7 | 11.5 |
| 8–14 | 276–386 | 2.7–4.7 M | 0.40–0.98 | 1.1–6.3 | 1–6 | ≥0.969 | ≥0.986 | 1–12 | 5.4–15.3 |

   *(Search times come from a second walk — a different game — so read that column as indicative
   per-ply cost.)* The game is **lumpy**: junctions (plies 1, 3, 7: ~4–7 effective moves) separated
   by corridors (plies 2, 4–6: 1.2–1.7) — the shape that kills fixed-depth expansion (V4).
   `store_k = 32` remains ample for *training-row* targets (top-32 ≥ 96.7% of visits everywhere).
7. **The noise floor of an independent evaluation is σ ≈ 0.014.** 60 first moves were evaluated in
   two independent runs (`cands.txt`/`cand_eval2.txt` vs `all_eval.txt`): mean difference +0.0002,
   stdev of differences 0.0197 ⇒ single-measurement σ ≈ 0.014; test-retest r = 0.939; pairwise
   ordering agreement 88.5%. **Resolution limit for everything below: gaps ≳ 0.03 are real,
   ≲ 0.02 are noise** (a two-measurement comparison has σ ≈ 0.02).
8. **The ply-1 strength curve is smooth — there is no strong set to find.** All 315 first moves
   independently evaluated (`all_eval.txt`: play the move, fresh L9 search of the reply, White's
   value = 1 − its top-child value): monotone decay **0.722 → 0.293**, largest interior gap 0.024
   (below the noise floor). Moves within 0.05 / 0.075 / 0.10 / 0.15 / 0.20 of the best:
   **3 / 15 / 32 / 91 / 176**. Any "opening fan width" is therefore a *chosen strength tolerance*,
   not a property of the game — which is part of why width is now set by budget allocation (V4),
   not by a filter.
9. **Visits are a screen, not a ranking — and the two signals disagree in a structured way.**
   Restating the earlier top-60 finding at the proper resolution: visits order two randomly chosen
   moves correctly ~74.5% of the time (Kendall τ ≈ 0.49) against ~88.5% for the evaluation's own
   test-retest — so visit rank carries real but much weaker order information, and the mid-pack
   rank statistics quoted in the first pass ("only 4 of the true top-10 are in the visit top-10")
   are **noise-dominated**; the robust claims are the large-gap ones. Structurally, at **ply 1
   Pentobi is essentially right where it looked**: its top-8 moves absorb ~90% of visits and all
   evaluate inside the genuine top ~15; the tail is not *mis*-ranked so much as **unranked** (~8
   opinions and ~307 non-opinions). The exploitable ply-1 phenomenon is **under-rating in the
   unsearched tail**: e.g. visit rank 250 → true rank 21 (0.638), visit rank 275 → true rank 30
   (0.626), and one of the three moves within 0.05 of best sits at visit rank **178**. Of the 32
   moves within 0.10 of best, **18 lie outside the visit top-32** (ranks up to 274).
10. **At ply 2, Pentobi is confidently wrong — the key new measurement.** All 315 replies to each
    of two strong openings independently evaluated (`reply_eval.txt`, judge = a fresh L9 search
    one ply deeper):
    - After `f8,e9,f9,g9,e10`: **92.9% of visits** on a reply that is **true rank 35 of 315**
      (0.239 vs best 0.311 — a 0.072 gap, ~5σ). 8 replies sit within 0.03 of the best.
    - After `f8,f9,g9,e10,f10`: **73.9% of visits** on true rank 7 (0.287 vs 0.343 — 0.056, ~4σ).
    - The mirror-draw artefact is ruled out (`avoid_symmetric_draw` declining a mirror move):
      neither the diagonal mirror nor the 180° rotation of White's move is even legal for Black.
    - **But the shortlist is right even where the mass is wrong**: the independently-best reply
      sits at **visit rank 2** and **visit rank 5** respectively — inside the top handful, inside
      any stored target support. Pentobi's error is *mass allocation among its own top
      candidates*, not candidate selection.
    - Caveats: n = 2 positions, and the judge is Pentobi one ply deeper — this shows
      self-inconsistency, not ground truth. The base rate across depths is V2's probe.
11. **Branching, measured across 8 independent strong lines** (`branch_1..8.txt`, candidates at
    ≥ 1% of the top child's visits): median **3 / 7 / 4 / 4 / 2** at depths 2–6, effective moves
    2.2–3.6. Replaces the earlier single-walk extrapolation.
12. **Per-ply search cost is modest and games dominate everything.** Plies 1–3 ≈ 25 s each at L9
    single-threaded, later plies 5–22 s; a full ~30-ply game ≈ 300–400 s CPU; v1 measured
    **176 games/h at 12 workers**. Under the allocation plan (V4), mapping the opening space is
    ~1,600 searches ≈ **1 box-hour** while 10,000 games ≈ **58 box-hours** — the map is a rounding
    error and there is no tree-vs-games budget tension at all.
13. **The opening book has never been active.** `pentobi_gtp/Main.cpp` sets
    `books_dir = application_dir_path` — the build directory, which holds no `.blksgf` files. Every
    Pentobi result we have (the whole L1–L9 benchmark ladder included) is **book-free** play. The
    book's 44 curated lines are ours to inject deliberately; and enabling the book would *break
    harvesting* (a book hit returns a move with no search tree — empty `move_values`), so the
    corpus engine must run `--nobook` explicitly. The benchmark-side consequence is V11.
14. **L9 is the engine's ceiling.** `Player::max_supported_level = 9`, hard-clamped. Duo simulation
    budgets per level are `{3, 21, 77, 213, 861, 7280, 221867, 1109339, 5546695}`, scaled by
    `0.6 · 0.7 · exp(0.1 · move_index)` — later moves get *more* simulations. GTP `param` exposes
    no simulation-count override; threads change wall-clock, not the budget.

---

## Does imitating Pentobi's distributions inherit its errors?

Fact 10 puts a sharp question under the whole plan: if 92.9% of the target mass at a measured node
sits on a ~5σ-suboptimal move, does a policy head trained on that distribution inherit the error?
The honest answer, quantified on the measured data:

**Yes, for the raw prior's argmax at affected nodes — and no training temperature can fix that.**
Flattening is order-preserving: the target's argmax stays the wrong move at any τ. Measured on the
stored top-32 target at the two evaluated nodes, τ barely moves the target's expected regret and
can *worsen* it (branch_1: 0.071 → 0.064 as τ goes 1 → 3; branch_2: 0.057 → **0.084**, because
tail dilution outweighs the mass shifted onto the good reply). So τ must not be sold as the fix —
its real function is *confidence softening*, which lowers how much prior mass search has to
overcome at play time.

**But the correction is present in the corpus, three ways:**

1. **The soft target contains the right answer where one-hot didn't.** The independently-best
   reply sits at visit rank 2 and 5 — well inside the stored support with non-trivial mass. v1's
   one-hot carried literally zero signal that an alternative existed; v2's target teaches the
   correct *shortlist* even where the mass is misallocated.
2. **Allocation flattening routes real games through the alternatives** (V4): at T = 2 the 92.9% /
   73.9% concentrations become 23.6% / 20.3% of *games*, so the value head sees L9-vs-L9 outcomes
   from both the over-weighted reply and its rivals. This is the distinction that resolves the
   question: **allocation-T diversifies the evidence; training-τ only reshapes confidence.** The
   outcome labels — not the policy target — carry the correction.
3. **At play time, search overrides a misordered prior when the value head is honest** — the
   standard AlphaZero mechanism, and the literature's view of visit distributions supports leaning
   on it: [Grill et al. 2020](https://arxiv.org/abs/2007.12509) show the visit-count distribution
   is only an approximation to the search's own policy-improvement target, and KataGo prunes its
   visit targets before training (see the literature note §8). A visit distribution is where the
   search *spent effort*, not a calibrated statement of move quality.

**Consequences adopted in this plan:** (a) held-out **top-1-vs-Pentobi measures agreement, not
strength** — a net could score lower top-1 by disagreeing exactly where Pentobi is wrong — so V14
reports top-3 agreement alongside it and the verdict weight stays on the ladder; (b) V2 measures
the **base rate**: how often is the visit argmax ≳ 0.04 below the best of the top-8 across
allocated nodes and depths — two nodes is an anecdote, thirty is a rate; (c) **value-informed
target reshaping is deliberately rejected for stage 1**: within-search child values are
visit-confounded (fact 5), independent evaluations exist only for mapped internal nodes, Henry's
ruling keeps evaluations out of the generator, and reshaping would blur the clean
imitate-the-preferences / correct-by-outcomes separation. If V2's base rate comes back high *and*
V15 fails with diagnostics pointing at opening play, target reshaping from stored
`edge_disagreement` data is the first gated follow-up to consider — as a training-time arm,
never a regeneration.

**Does this explain v1's failure? No** — v1 trained on the *same* argmax moves (as one-hots), so
teacher error cannot explain v1 landing far *below* the teacher-agreement ceiling; junk unharvested
openings + one-hot targets remain the diagnosis. What fact 10 does bound is **v2's upside**: pure
imitation inherits L9's opening mass-allocation errors, which is consistent with the strategic
thesis — the surpassing has to come from the value pathway and Phase 3, not from SL alone
(ChessBench found the same ceiling at industrial scale — literature note §6).

---

## V1. GTP layer

`pentobi/gtp.py` gains `move_values()` alongside the existing `reg_genmove()`, plus
a parser in a new `pentobi/move_values.py`:

```python
@dataclass(frozen=True)
class MoveValueEntry:
    visits: int
    value: float          # side-to-move perspective; NOT bounded to [0, 1]
    cells: str            # piece-name prefix already stripped

@dataclass(frozen=True)
class MoveValues:
    entries: tuple[MoveValueEntry, ...]   # visit-descending, as the engine emits them
    total_visits: int
```

Parsing rules, all from the measured output: split each line into three numeric fields plus the
move; the move is everything after the first `]` (a line with no `[` is still accepted); values may
be negative; an **empty response is legal** (no search tree — e.g. a forced pass or a book hit) and
yields an empty `MoveValues`, which the caller handles rather than crashing.

Also in this row: `PentobiGtp` gains an explicit `nobook: bool` flag (passing `--nobook`). Today
book-free play is an *accident* of the `books_dir` build-path bug (fact 13); if a future Pentobi
rebuild fixed that, every harvest would silently start returning empty `move_values` on book hits.
The corpus path sets it always; the benchmark path decides per V11.

Tests are fixture-driven on real captured output (checked into `tests/.../data/move_values_l9.txt`,
source `local/probes/mv_deep.txt`) — no engine binary on CI, matching every other Pentobi test in
the repo.

**Effort:** 2 h.

## V2. Confidently-wrong base-rate probe + residual engine probes (box)

The most important open measurement of this pass: **how often is fact 10 true?** Protocol, on the
box via `gpu-anywhere` (extends `local/probes/reply_eval.sh`):

- Draw ~**30 internal nodes** from the V10 pilot plan, stratified across depths 2–6 weighted by
  budget share (so the sample matches where the corpus actually spends games).
- For each, independently evaluate its **top-8-by-visits children** (play the child, fresh L9
  search, read the top-child value): ~240 searches ≈ 2 box-hours at 10 workers.
- Report: (a) the fraction of nodes where the visit argmax evaluates **> 0.04 below** the best of
  the eight (≈ 2.8σ against the two-measurement noise floor, fact 7); (b) the mean stored-target
  regret vs the best-of-8; (c) the visit rank of the best-of-8 (is the shortlist property of
  fact 10 general?); (d) the same numbers split junction vs corridor nodes.

This decides whether the imitation-error block's "consequences adopted" are sufficient (low rate),
or whether the gated target-reshaping follow-up gets scoped after V15 (high rate). The evaluations
stay out of the generator either way (Henry's ruling).

Residual smaller probes in the same session: whether `argmax(visits)` equals what `genmove` plays
over ~20 sampled positions (a mismatch is recorded either way via V6's `top_action`); what
`reg_genmove` / `move_values` return at forced passes and in the last plies (the parser must
accept an empty response); and the drive-pattern overhead of `reg_genmove` + `play <non-best>` vs
`genmove` (subtree-reuse loss) over ~20 plies. *(The first pass's planned ply-1 straggler check
and depth-2/3 misranking probes are superseded — facts 8–10 measured both properly.)*

Output: a short table appended to this section; V4's defaults are confirmed or revised against it.

**Effort:** 4 h box CPU.

## V3. Search-space store

Execute [`corpus-search-space-store.md`](corpus-search-space-store.md) S1–S6: the position-keyed
SQLite DAG (symmetry-canonical node keys — **decided**, with mirror-pair weight merging at
symmetric nodes; full child lists as queryable edges; content-derived seeds), the **allocation
plans** (`plans`/`plan_nodes`: every plan's per-node game targets stored, planned-vs-actual
queryable), the playout registry with structural game identity `(board_key, replica)`, the
`export-opening` materialisation and the coverage report. The store is the interface between
phase A and phase B and what makes every later extension — bigger budget, different T/R, more of
anything — an incremental re-plan that cannot collide with existing work.

**Effort:** 1.5 days (S1–S6).

## V4. Phase A — `plan`: budget-proportional allocation with emergent depth

**Depth is an output, not an input** (Henry's ruling). The game is lumpy — junctions of 4–7
effective moves at plies 1/3/7 separated by near-forced corridors (fact 6) — so a fixed-depth tree
spends equal effort where one ply buys dozens of distinct strong starts and where it buys nothing.
Instead the whole **game budget** is handed to the root and split recursively:

**The allocation rule.** At a node holding budget `b`:

1. If `b < 2R`, the node is a **playout start**: it keeps its games (integerised by
   largest-remainder rounding so the total budget is conserved exactly).
2. Otherwise take the node's children from its stored search (**all** of them, mirror-pair-merged
   when the node's position is symmetric — in practice the root, where 310 of 315 children merge
   into 160 canonical positions, fact 4 / store D-b), weight them
   **`w ∝ visit_share^(1/T)`** — Pentobi's opinion, flattened; never the independent evaluations
   (Henry's ruling) — drop children whose renormalised share of `b` is < R, and renormalise over
   the survivors. If none survive, the node is a playout start.
3. Recurse into each surviving child. A child not yet in the store is **searched on demand**
   (content-derived seed) — mapping exactly as far as the plan needs and no further.

*Implementation clarification (2026-07-28, V3):* rule 2 needs **two** survivors, not one. With a
single survivor the child inherits the parent's whole budget undiminished and splits again on the
same terms, so the walk follows one forced line to the end of the game and never terminates (the
`allocation_sim.py` numbers above avoided this only via its `depth_cap`). Requiring two survivors
caps each child at `b − R`, which bounds the descent, and costs nothing: splitting into one piece
moves the same games one ply deeper for no extra coverage.

Total games are conserved; depth, opening count, and the effective width of the first-move fan all
**emerge** from `(budget, T, R)`. The plan — every node's budget share and planned games — is
persisted (`plan_nodes`), reproducible (a pure function of DAG content + parameters, store D-e),
and is what phase B fulfils. The 44 book lines (`opening_books/book_duo.blksgf`, ~17 first moves;
parser `local/probes/parse_book.py`) are **force-mapped** with a floor of R games at each line's
terminal node, `source = 'book'`, regardless of weight — the engine itself stays `--nobook`
throughout (fact 13).

**Simulated on the real measured distributions** (`local/probes/allocation_sim.py`: ply 1 from the
full root search, depths 2–6 sampled from the 8 walked lines), 10,000 games at R = 2:

| 1/T | openings | nodes to search | 1st moves covered | median depth | max games on one opening |
|---|---|---|---|---|---|
| 1 (raw) | 608 | 1,063 | 84 | 7 | 1,426 |
| 1/1.5 | 1,398 | 2,354 | 230 | 7 | 100 |
| **1/2 (sqrt)** | **1,211** | **1,566** | **279** | **4** | **32** |
| 1/2.5 | 868 | 878 | 293 | 4 | 35 |
| 1/3 | 600 | 470 | 293 | 3 | 46 |

Note the non-monotonicity: spreading wider starves every branch of the budget needed to go deeper,
so coverage and depth trade off; √-flattening sits near the sweet spot structurally. **Cost:**
mapping ≈ 1,600 searches ≈ 1 box-hour; the games are 58 box-hours — the earlier tree-vs-games
budget tension does not exist (fact 12). *(Two known simplifications of the sim, for the
implementer: its leaf "depth" counts the split level, one above the prefix ply count; and depths
2–6 reuse the 8 walked lines' distributions independently of the parent — real branching is
node-specific. Neither changes the shape; V10 measures the real plan.)*

**What the flattening buys and what it costs — quantified at ply 1**, the one ply where the true
strength of all 315 candidates is known (fact 8; scratchpad analysis of `all_cands` × `all_eval`,
10 k budget, keep-filter ≥ 2):

| 1/T | first moves kept | mean eval of allocated games | mass > 0.05 below best | > 0.10 | > 0.15 | > 0.20 |
|---|---|---|---|---|---|---|
| 1 | 84 | 0.686 | 0.365 | 0.102 | 0.025 | 0.005 |
| 1/1.5 | 230 | 0.641 | 0.659 | 0.350 | 0.164 | 0.060 |
| **1/2** | **279** | **0.608** | **0.816** | **0.545** | **0.312** | **0.133** |
| 1/2.5 | 293 | 0.589 | 0.887 | 0.655 | 0.410 | 0.189 |

Read both ways, honestly. **The tail is where the under-rated openings live**: 18 of the 32
genuinely strong first moves (within 0.10) sit outside the visit top-32, one of the top three sits
at visit rank 178 — raw allocation (T = 1, 84 moves kept) would never play roughly half the strong
set, and mining exactly these under-rated lines is the strategic thesis. **And the tail is where
the waste is**: at T = 2 over half the ply-1 game mass lands on first moves more than 0.10 below
best — coverage and waste are inseparable under visit-only weights, because Pentobi's tail is
unranked (fact 9), and no visit-based rule can pick out the rank-178 gem without also playing its
weak neighbours. Two mitigations are built in: the "waste" is far above v1's uniform-random junk
(mean eval ≈ 0.61 vs v1 sampling the 0.29-tail uniformly), and weak-but-plausible starts are
precisely what gives the **value head outcome variance** — v1's D3 measured 96% White wins from
balanced strong starts, i.e. near-constant value labels; unbalanced starts are where the value
signal comes from. A depth-split T (tighter at ply 1, √ deeper) is the recorded first alternative
if V10's pilot opening-quality histogram looks too weak — it stays within the visits-only ruling —
but the default is one global knob.

**Defaults: T = 2, R = 2, B = 10,000** (V12). All three are plan parameters, not schema — a later
re-plan at different values composes with everything already generated (store D-e).

**Effort:** 1 day.

## V5. Phase B — `generate`

**Start from a planned node, then Pentobi at full strength to the end, harvesting every ply.**

**The phase-A / phase-B contract, with the plan as the interface:** `plan` writes the store
(nodes/edges/plans); `generate` reads the active plan and writes only game shards + the playout
registry — it never mutates a node, edge, or plan. The DAG and plans can grow later without
touching existing games, and games can be added without re-searching any opening.

- **Scheduling is fulfilment-driven** (store D-e): jobs `(node, next replica)` ordered by actual ÷
  planned ascending, then `hash64(board_key)` — every node's fulfilment rises together, a
  truncated run is an even proportional slice of the plan at any point, and nodes added by a
  re-plan are covered first automatically. "Generate 5 k more games" can never reproduce a game we
  already hold: identity is `(board_key, replica)` with content-derived engine seeds.
- **The prefix is replayed into the engine with `play`, not searched** — the witness-path plies'
  targets already live in the DAG. The start position itself *is* searched in-game
  (`reg_genmove` → harvest → `play argmax`), once per replica — repeated harvests of the same
  start under different seeds, which doubles as free test-retest data for noise-floor monitoring.
- **Every subsequent ply is harvested**: `reg_genmove` → `move_values` → store → `play argmax`.
- **Continuations are full-strength only — no temperature, no move sampling.** Temperature would
  weaken the very play being distilled; D3 measured per-game seed variation as sufficient for
  continuation diversity (24 seed-only games from one start: 24/24 unique, 91.7% distinct
  positions); and any *deliberate* deviation is strictly better expressed as allocation breadth —
  where it gets a full harvested label and a plan entry — than as an unlabelled in-game deviation.
  If V13's ablation favours replication, the lever is R, not temperature.
- **Labels per row:** the soft policy (V6), the outcome from the side to move, the signed margin,
  and `search_value` — the top child's value from the same `move_values` response (fact 5; never
  GTP `get_value`).
- Desync guards from v1 are unchanged and non-negotiable: every engine move legality-checked
  against our rules engine, engine `final_score` cross-checked against
  `BlokusDuoGame.final_scores`, `--noresign`, hard ply cap. Each game's replayed prefix is
  additionally validated against the start node's witness path.

Cost per game: ~300–400 s of search minus the prefix plies the DAG absorbs → **roughly 170 games/h
at 12 workers**, comparable to v1's measured 176.

**Effort:** ½ day.

## V6. Schema v2

Two datasets under one corpus directory (plus the store DB), both `dataset_kind =
"pentobi_distill_v2"`:

**`opening/opening_{NNNNN}.parquet`** — one row per searched DAG node, materialised by
`export-opening` from the store (the DB is the source of truth; the export is regenerable and
stamped with `dag_hash`).

| Column | Type | Description |
|---|---|---|
| `board` | bytes | node key-frame compact board (side-to-move), as v1 |
| `policy_indices` / `policy_values` | bytes | **soft target**: top-32 children by visits, normalised to sum 1 (`policy_kind` stays `sparse_v1`, so every downstream densify path is unchanged) |
| `child_values` | bytes | float32, aligned to `policy_indices` — Pentobi's per-child value |
| `tail_mass` | float32 | visit mass dropped by the top-32 truncation (measured ≈ 0.036 at ply 1, 0.017 at ply 2) |
| `search_value` | float32 | Pentobi's backed-up value for the side to move = the top child's value |
| `depth` | int32 | plies from the empty board |
| `reach_weight` | float32 | product of ancestor visit shares (summed over DAG parents) |
| `budget_share` / `planned_games` | float32 / int32 | the active plan's allocation at this node |
| `node_id` / `parent_id` | int64 | graph structure (`parent_id` = first witness parent) |
| `player` | int8 | side to move |
| `outcome_mean` / `outcome_count` | float32 / int32 | filled by V8's `link` pass (count 0 until then) |

**`games/corpus_{NNNNN}.parquet`** — one row per harvested game ply: v1's schema plus the soft
target. v1's columns all survive with identical meaning (`value`, `margin`, `player`, `game_id`,
`ply`, `action`):

- `policy_indices` / `policy_values` become the **soft** distribution (v1 stored `[action]`/`[1.0]`).
- `action` stays the move actually played; a new `top_action` records `argmax(visits)` — equal on
  continuation plies by construction, and the pair makes any mismatch visible.
- `child_values`, `tail_mass`, `search_value` as above.
- Footer `games_meta` gains, per game: the start node's `board_key` (hex), `replica`,
  `engine_seed`, `witness_actions`, plus the `dag_hash` and the generating plan's
  `(plan_id, budget, temperature, min_replicas)` — shards are self-describing and the playout
  registry is reconcilable from footers alone (store D-e).

**Validator** (`validate_shard`) changes: replay is unchanged, but the one-hot assertion is
replaced by — target sums to 1 (±1e-5), support ⊆ the position's legal set, `action ∈ support`,
`top_action == argmax(policy_values)`, and every stored board reachable by replaying
`witness prefix + played actions`. Opening rows validate against a replay of their witness path
(plus the key-frame transpose when flagged).

**Row-mix note (opening vs midgame).** At the stage-1 shape a game harvests ~26 rows from median
start depth ~4 onward, so the corpus is ~260 k midgame game-rows against only ~1.6 k opening-node
rows — positions at depths 1–3 exist *only* in the opening dataset and are ~0.6% of rows by count.
Since openings are the stated strategic edge, this ratio is tracked by V7's report and corrected
at training time by V9's source mix weights (an opening row must not be a 1-in-160,000 sampling
event); the weight is a V14 arm.

Size: ~32 × 12 B ≈ 400 B/row vs v1's ~100 B — a ~300 k-row stage-1 corpus is ~120 MB. Fine for R2
and box RAM.

**Effort:** ½ day. Update [`../07-DATA-STORAGE.md`](../07-DATA-STORAGE.md) in the same commit.

## V7. CLI + diagnostics

The v2 subcommands ship as **`scripts/pentobi_corpus_v2.py`** (`plan`, `generate`, `export-opening`,
`link`, `coverage`, `analyze`, `validate`) rather than as new subcommands on the v1 script: the two
generators give `generate`/`validate`/`analyze` incompatible meanings and arguments, and v1's shards
stay on disk (and its CLI usable) as the mid-game supplement. `analyze` reports what v2 is actually
claiming:

- the store's coverage report (store D-f): plan fulfilment, mapping debt, nodes/starts by
  emergent depth, distinct first moves / canonical first positions (vs 414 / 212, and the
  52-position gap of fact 4), play-mass coverage, planned-games distribution;
- **row mix:** opening rows vs game rows by depth bucket (the V6 ratio);
- **target quality:** mean target entropy and mean effective-move count per ply bucket (the direct
  measure of "did we keep more than a one-hot"), plus mean `tail_mass`;
- **duplicate-position rate** across the game shards, raw and mirror-collapsed (v1 measured 0% at
  prefix-4; shared strong openings will raise it);
- outcome/margin distributions and the White/Black split — expected to be *less* White-skewed than
  D3's 96% precisely because flattened allocation plays unbalanced starts (V4).

**Effort:** 3 h.

## V8. `link` pass

One recursive aggregation in the store (S5): for each DAG node, the outcomes of playouts started
in its subtree land in `outcome_mean` / `outcome_count` (sign-adjusted to the node's side to
move). Opening rows then carry an *empirical* value label from real L9 continuations alongside
Pentobi's `search_value`; V9 chooses between them. Honest caveat, stored in the doc not the data:
an interior node's `outcome_mean` averages continuations from *imposed* prefixes under the
allocation's mixture, not Pentobi's own play from that node. With fulfilment scheduling, interior
nodes near the root aggregate hundreds of games; deep starts have only their own replicas
(≥ R = 2), which is why V9 has a blend.

**Effort:** 2 h. **Priority:** Medium — the corpus is trainable without it (fall back to
`search_value` everywhere).

## V9. Trainer: soft targets, subtree holdout, value-target choice

**No training-code change is needed for the loss.** `BaseNNetWrapper.loss_pi` is already
`F.kl_div(outputs, targets)` against a full distribution — v1's one-hot was a degenerate case. The
work is in the dataloader:

- `CorpusGameRows` carries the stored `(indices, values)` per position instead of just `action`;
  `build_training_examples` uses them directly rather than calling `smooth_policy`.
- **Target temperature τ** applied at load (`p^(1/τ)` renormalised over the stored support), with
  its role stated honestly per the imitation-error block: τ softens confidence (helping play-time
  search override misallocated mass) but is order-preserving and does **not** reduce — can even
  increase — the target's expected regret at nodes like fact 10's. The corpus stores τ=1
  normalised visits, so retuning never requires regeneration.
- **Legal-set floor ε** kept as an option but defaulting to **0**.
- **Holdout splits by opening subtree, not by game — this fixes a latent leak.** v1's game-level
  split (`split_games_holdout`) was valid because every v1 game had a unique opening; v2
  deliberately gives many games a shared opening, so game-level splitting would put near-identical
  early positions (and literally identical opening rows) on both sides of the boundary and report
  a falsely good held-out score. The split unit is the **canonical ply-1 position** (the
  mirror-canonicalised first move of the game's witness path — well-defined for every game and
  every opening row regardless of the depth its start sits at, since every witness path passes
  through exactly one root child). Hold out ~5% of units, sampled stratified by planned game
  mass; **all** games and **all** opening rows whose witness path starts in a held-out unit are
  excluded from training. Residual midgame-transposition leakage across the boundary is measured
  (duplicate-position rate between train and holdout) and reported, not assumed away.
- **Opening-row value target** — `--opening-value {outcome, search, blend}`, default **blend**:
  `v = (n·outcome_mean + k·(2·search_value − 1)) / (n + k)` with `k = 5` — equal to the teacher's
  opinion at n = 0 and the empirical outcome as n grows. The `2v − 1` rescaling is approximate
  (fact 5) — which is exactly why outcomes dominate as n grows and why V14 carries a
  pure-`outcome` arm. Game rows keep the pure outcome, unchanged.
- **Value-loss decorrelation option** (off by default): ~26 rows of a game share one outcome —
  AlphaGo's value net memorised outcomes exactly this way (train/test MSE 0.19/0.37) until they
  sampled one position per game. If the per-colour calibration diagnostic shows memorisation, the
  lever is a per-position value-loss weight of `1/game_size` (or value-head position subsampling)
  at batch build — a trainer flag, never a corpus change.
- **No auxiliary disagreement target, no disagreement-weighted sampling — deliberately.** The
  "learn where Pentobi is wrong" objective is served by the *data*: flattened allocation (coverage
  of the misjudged alternatives), honest outcome-grounded value labels (the correction signal,
  reaching MCTS through the value head), and the queryable `edge_disagreement` view (V16's seeding
  input). Training on the disagreement directly has no precedent in any analogue we found, would
  distort the imitation distribution away from the expert's state distribution, and adds a knob
  with no measurement behind it. Revisit only via the V2-probe + V15-failure route defined in the
  imitation-error block.
- **Mix weights** over the three sources: opening rows, v2 game rows, and the v1 13 k corpus as a
  mid-game supplement. Defaults for the first run: opening rows upweighted to ~5% of sampled
  examples (see V6's row-mix note) / games 1.0 / v1 0.0, with the opening weight and the v1 mix as
  explicit V14 arms.
- Symmetry augmentation is unchanged — transposing an arbitrary support through `transpose_action`
  already works. Assert support ⊆ legal at load; a violation means corpus/rules-engine desync.

**Effort:** 1 day.

## V10. L9 pilot (box)

`plan` at **B = 1,000, T = 2, R = 2** (~120 openings, ~160 mapping searches ≈ 10 min), then ~200
games by fulfilment order; `validate` (every row, both datasets), `analyze`, `coverage`, and
measure — mapping wall-clock and node count vs the simulation, games/h, positions/game, target
entropy per ply, duplicate-position rate, White/Black split, drive-pattern overhead in situ, and
the **opening-quality histogram** (allocated game mass by ply-1 independent eval, reusing
`all_eval.txt` — the direct check on V4's waste table that decides whether the depth-split-T
alternative is needed). **Deliverables: confirmed (T, R) and the V12 run sized from measured
rates.** Nothing scales until this table exists.

**Effort:** 4 h box CPU.

## V11. Book-strength measurement

Our entire L1–L9 benchmark ladder has been measured against **book-free** Pentobi (fact 13). If
book-enabled Pentobi is stronger, that is the opponent "beat Pentobi level 9" must mean — otherwise
we are grading against a weakened engine. Independent of corpus generation (the corpus engine stays
`--nobook` regardless, because harvesting requires a search tree); this is purely about what we
*evaluate* against.

- **Enable + verify:** copy (or symlink) `book_duo.blksgf` from `~/code/pentobi/opening_books/`
  into the build directory next to the binary (that *is* `books_dir` as built), or patch a
  `--books-dir` option. Verify engagement with the clean detector: a book hit returns a move whose
  following `move_values` is **empty** (and returns near-instantly at L9).
- **Measure:** book-enabled L9 vs book-free L9, colour-balanced, **200 games** (95% CI ≈ ±7 pp;
  the book only touches the first ~4 plies, so expect a modest effect). ~3–4 box-hours at 12
  workers. Report win rate and margin distribution.
- **Consequences if book-on is meaningfully stronger (≳ 5 pp):** `pentobi/player.py`,
  `scripts/pentobi_benchmark.py` and `scripts/mini_ladder.py` gain a book option **defaulting on
  for evaluation**; the V15/D8 gate is defined against book-enabled L-levels; the 2026-07 ladder
  results (80/75/60/55/45/20% at L1–L6) are re-labelled a *book-free baseline* rather than the
  headline, and the README chart gets the annotation. If the difference is within noise, record
  that and keep book-free as the benchmark (it is also what every historical number means).

**Effort:** 4 h box. **Priority:** Medium — it changes goalposts, not the corpus; run it before
V15 interprets anything.

## V12. Stage-1 v2 corpus generation (box)

**The recommended first run: `plan --budget 10000 --temperature 2 --min-replicas 2`, then
`generate` for a 3-day box window.** Mapping ≈ 1,600 searches ≈ 1 box-hour; 10,000 games ≈ 58
box-hours ≈ 2.5 days at 12 workers; expected shape per the simulation: ~1,200 distinct openings,
~279 first moves covered, median start depth ~4, 2–32 games per start (mean ≈ 8). Fulfilment
scheduling makes the run **budget-bounded rather than count-bounded**: stop it at the window's end
and the corpus is an even proportional slice of the plan; restart and it continues exactly where
it left off. Then `scripts/corpus_wrapup.py` — validate + analyze + rsync to the laptop mirror +
**sync to R2 with the verify-before-done gate** — with the **store DB included in the sync**: the
DB is the map of what the shards mean, and nothing is declared done until every file is confirmed
present in the bucket.

Rented CPU stays gated on V15, exactly as D5's two-stage rule; a top-up is a re-plan at a larger
budget (store D-e), which deepens where budgets now split and never invalidates a game.

**Effort:** 3 days box wall-clock, £0.

## V13. Breadth-vs-replication ablation (stage-1 subset experiment)

At a fixed position count, many openings once each or fewer openings several times? The literature
leans breadth with a small per-opening threshold
([`../research/corpus-generation-literature.md`](../research/corpus-generation-literature.md) §7),
and stage 1's allocation already sits between the extremes (mean ≈ 8 games/start). Because every
start holds ≥ 2 and up to ~32 replicas, the ablation needs **zero extra generation** — it is two
training subsets of the stage-1 corpus:

- **Arm A (breadth):** 1 game from each of ~800 starts.
- **Arm B (replication):** 4 games from each of ~200 of those starts (position counts matched).
- **Probe set:** games from held-out opening subtrees (the V9 split units), common to both arms.
- Train identical SL configs; compare probe-set top-1/top-3-vs-Pentobi + policy CE (primary),
  value MSE / per-colour calibration (replication's claimed advantage — check it directly), and a
  mini-ladder L1–L4 × 50 sanity check.
- **Decision rule:** |Δ top-1| ≥ 1 pp → the winner sets the *top-up* shape after V15 (breadth ⇒
  bigger budget at same R; replication ⇒ raise R). A tie ⇒ breadth (cheaper per novel position,
  literature prior).

**Effort:** 1 day box GPU. **Priority:** Medium — it informs the top-up, not stage 1 itself.

## V14. SL re-fit

Re-run `scripts/distill_sl.py` on the v2 corpus: `warm` (v3 gen-40) and `scratch` arms as before,
a short **τ sweep** (1.0 / 1.5 / 2.0 — read through the imitation-error block's lens: expect τ to
trade top-1 agreement against search-corrigibility, not to fix ordering), an **opening-value arm**
(pure `outcome` vs the default blend), an **opening mix-weight arm** (V6 row-mix note), and one
arm with the v1 corpus mixed in at ~25% as a mid-game supplement. Read held-out top-1 **and
top-3**-vs-Pentobi and per-colour value calibration on the subtree-level holdout; net size from
the [distillation-net-sizing](../research/distillation-net-sizing.md) provisional pick, re-checked
against the larger corpus per D8's caveat.

**Effort:** 1 day box GPU.

## V15. D8 ladder gate

Unchanged from [`pentobi-distillation.md`](pentobi-distillation.md) D8: `scripts/mini_ladder.py`,
L1–L9 × 50 games × 400 sims, the chosen net plus the v3 gen-40 baseline, **gate = +10 pp at any of
L5–L7 after SL alone** — against book-enabled or book-free Pentobi per V11's verdict. If v2 clears
it, Phase 3 (D9–D11) unblocks and the corpus scales via a re-plan top-up. If it fails a second
time, the conclusion is about distillation itself, not the generator, and the next move is
diagnosis — starting from the V2 base-rate probe's verdict on teacher error — not a v3 corpus.

**Effort:** ½ day box.

## V16. Phase 2 — beyond Pentobi's own opening taste (gated on V15)

Only once SL distillation demonstrably works:

- **Net-in-the-loop expert iteration.** The distilled net — which, unlike v3, will have opening
  breadth — proposes openings Pentobi *undervalues* (the store's `edge_disagreement` view is the
  seeding query; fact 9's under-rated tail and fact 4's 52 never-searched first positions are the
  known hunting grounds); those are inserted as `source = 'net'` DAG nodes, searched and played
  out at L9, retrain, repeat. This is the mechanism for exceeding the teacher — with production
  precedent: Stockfish's training mix explicitly includes "self-play data from openings it usually
  gets wrong" (literature note §5).
- **Systematic pentomino-opening enumeration filtered through `move_values`** — every legal
  first-move placement scored by an L9 search, so breadth is bounded by the game, not by Pentobi's
  root move generator (closing fact 4's gap).

Scoped properly when V15 passes.

---

## Resolved decisions (2026-07-28, second pass)

1. **Depth is an output: budget-proportional allocation** `(B, T, R)` with emergent depth replaces
   fixed depth + per-depth pool caps (Henry's ruling 1; V4).
2. **Allocation weights come from Pentobi's visit shares only** — flattened, mirror-pair-merged —
   never from the independent evaluations, which are measurement and validation (Henry's
   ruling 2). The corpus tests the teacher's opinion by playing it out.
3. **Symmetry-canonical node keys: decided** (measured: 414 → 212 first positions; the engine
   itself searches both members of 310 mirror pairs at the root).
4. **The imitation-error question is answered by separation of channels**: allocation-T
   diversifies the evidence, outcome labels carry the correction, training-τ only softens
   confidence; no value-informed target reshaping in stage 1 (gated follow-up defined in the
   imitation-error block); top-3 agreement reported alongside top-1.
5. **Holdout is by opening subtree** (canonical ply-1 unit), fixing the latent game-level leak.
6. **Stage-1 shape: B = 10,000, T = 2, R = 2** — one 3-day box window, mapping cost negligible.
7. **No temperature anywhere in generation**; deliberate deviations are allocation breadth, not
   in-game sampling. Opening value target = count-shrunk blend (V9), pure-outcome arm in V14.

## Remaining open questions

1. **The confidently-wrong base rate** (V2) — the most important open measurement: two positions
   say Pentobi misallocates mass among its own candidates by 4–5σ; thirty positions across depths
   say whether that is the norm, and V15's diagnosis depends on knowing it.
2. **Global T = 2 vs depth-split T** — quantified trade-off in V4's ply-1 table; decided by V10's
   pilot opening-quality histogram, not by taste.
3. **Breadth vs replication for the top-up** — V13's subset ablation; default breadth.
4. **Is v1 worth mixing in at all?** One V14 arm at 25%; drop it if it doesn't help.
