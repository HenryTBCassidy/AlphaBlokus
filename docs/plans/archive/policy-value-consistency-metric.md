# Policy–Value Consistency (PVC) metric

Add a per-generation diagnostic that measures how well the **policy head** and the **value head**
agree, via a one-ply lookahead on the frozen eval set. Pure instrumentation — it does **not**
change training. Motivated by the v3 post-mortem ([`../../research/xl-training-scaleup.md`](../../research/xl-training-scaleup.md)):
v3's policy kept improving internally while external strength plateaued, and we had no signal that
*decomposed* the two heads. PVC gives that decomposition — are the heads mutually consistent, or is
one lagging?

**The idea.** For an eval position `s` (current player to move) and a candidate move `a` leading to
`s'`, the one-ply value of `a` is the negamax `Q₁(s,a) = −V(s')` (after your move it's the
opponent's turn, so their value negates to yours). A well-trained net should have its policy
`π(·|s)` broadly agree with the ranking `Q₁(s,·)`.

**The crucial caveat (bake into the chart caption, or it will be misread).** Perfect agreement is
NOT expected and disagreement is often *correct*: the policy is trained on the MCTS *visit*
distribution, which reflects **multi-ply** search, while `Q₁` is only **one ply** of value. A move
that looks weak one-ply but is best after deep search — the policy rightly likes it, `Q₁` doesn't.
So read PVC as a **trend**: rising early (both heads improving + becoming consistent) → plateau
*below* 100% (the residual ≈ how much deeper the policy sees than one-ply value). A **late drop or a
persistently low level** is the red flag — value head lagging (can't evaluate the states the policy
leads to) or policy chasing lines the value head doesn't support (the v3 decoupling).

Companion docs: [`../../guides/PLAN-FORMAT.md`](../../guides/PLAN-FORMAT.md),
[`../../guides/STYLE-GUIDE.md`](../../guides/STYLE-GUIDE.md).

---

## The metric, precisely (defaults chosen; §"Design choices" flags them for review)

Per eval-set position `s` (current player to move):
1. Get `π(·|s)` from the net; take the **top-K = 8** legal moves by policy probability (bounds cost;
   focuses on where the policy has mass).
2. For each candidate `a`: build child `s'` (reuse the compact-board→playable rebuild already in
   `_compute_mcts_agreement`), then `Q₁(a) = −V(s')` — **except** if `s'` is terminal, use the true
   game result (from the mover's perspective). Batch all children through `predict_batch`.
3. Two agreement measures over the K candidates:
   - **argmax-match**: `1` if `argmax_a π(a) == argmax_a Q₁(a)`, else `0`.
   - **Spearman rank correlation** between `π(a)` and `Q₁(a)` across the K candidates.

Aggregate over the eval set → **`pvc_argmax_match`** (fraction) and **`pvc_spearman`** (mean). Both
logged per generation.

**Optional value-symmetry sub-metric** (your reflection idea): `value_symmetry_mae =
mean|V(s) − V(reflect(s))|` over the eval set — should sit near 0; a rising value informs us the
value head isn't respecting the order-2 symmetry. (Policy symmetry is already tracked as a KL in the
existing symmetry diagnostic.)

**Cost:** ~`|eval set| × K` extra child evals per gen (~100 × 8 = ~800), one batched `predict_batch`
— well under a second on GPU. Negligible.

---

## Checklist

| # | Item | Effort | Priority | Files | Done |
|---|------|--------|----------|-------|------|
| S1 | Compute PVC in the eval-set diagnostics (`_compute_policy_value_consistency`) | 2 h | High | `games/base_wrapper.py` | ✅ |
| S2 | Metrics schema + wire into the per-gen eval block | 1 h | High | `storage/metrics.py`, `games/base_wrapper.py` | ✅ |
| S3 | Report chart (PVC over generations, with the caveat caption) | 1 h | Medium | `reporting/charts.py`, `reporting/report.py` | ✅ |
| S4 | Optional `value_symmetry_mae` sub-metric | 45 min | Low | `games/base_wrapper.py`, `storage/metrics.py` | ✅ |
| S5 | Tests + docs | 1.5 h | High | `tests/`, `docs/05-EVALUATION.md` | ✅ |

---

## S1. Compute PVC in the eval-set diagnostics

**Current state.** `_compute_eval_set_diagnostics` (`base_wrapper.py:587`) computes top1/top5/entropy
+ value calibration from the eval set; `_compute_mcts_agreement` (`:562`) already rebuilds *playable*
boards from the compact int8 positions (`compact_boards.npy`) to search them. Both run inside
`train()` at `:553-570` when an `eval_set` is passed.

**Target.** Add `_compute_policy_value_consistency(eval_set) -> dict` returning `pvc_argmax_match`
and `pvc_spearman`:
- Rebuild the playable board per position (reuse the `_compute_mcts_agreement` rebuild path — factor
  it into a shared helper if cleaner).
- For each board: `predict` → policy; take top-K=8 legal moves by prob (mask illegal via
  `self.game` valid-moves, as MCTS does); apply each via the board's `with_move`/`with_piece` to get
  children; **batch** all children across all positions into one `predict_batch`; take `−value`
  (negamax) per child, substituting the true result for terminal children.
- Compute argmax-match + Spearman per position; return the means.
- Correctness risks to nail (and unit-test in S5): (a) the **negamax sign** — value is
  current-player-perspective, so child value must be negated; (b) **terminal children** use the game
  result, not `V`; (c) **pass** moves and positions with `< 2` legal moves (Spearman undefined on
  <2 items — skip or define as NaN and exclude from the mean); (d) canonical/perspective consistency
  between `s` and `s'` encodings.

---

## S2. Metrics schema + wire into the per-gen eval block

**Target.**
- Add `metrics.log_policy_value_consistency(generation, pvc_argmax_match, pvc_spearman,
  eval_set_size)` mirroring `log_policy_accuracy` (`storage/metrics.py`): a parquet record
  (`PolicyValueConsistency/`) + W&B keys `pvc/argmax_match`, `pvc/spearman` (step_metric
  `generation`).
- Call `_compute_policy_value_consistency` from the eval block in `train()` (`base_wrapper.py:553`),
  alongside the existing entropy/accuracy logging.

---

## S3. Report chart

**Target.** Add `make_policy_value_consistency_plot` in `reporting/charts.py` (mirror
`make_elo_plot`): two lines over generations (`pvc_argmax_match`, `pvc_spearman`), wired into the
report page (`reporting/report.py`). **Caption must state the caveat** — "a healthy net plateaus
below 100%; watch for a late drop or persistently low level (value/policy imbalance), not for
100%." Without that caption the chart will be misread.

---

## S4. Optional `value_symmetry_mae` sub-metric

**Target.** In the same eval pass, compute `mean|V(s) − V(reflect(s))|` using the game's order-2
symmetry (`get_symmetries`). Log it (`pvc/value_symmetry_mae`) and add to the symmetry diagnostic
chart (policy symmetry KL is already there). Low priority — do only if S1–S3 land cleanly.

**As built.** Included. Compute is `BaseNNetWrapper._compute_value_symmetry_mae` (a static
`_value_symmetry_mae` core so it's testable with a scripted predictor); the identity variant is
excluded game-agnostically by `state_key` equality rather than assuming its list position. Logged as
an optional `value_symmetry_mae` column on the PVC record + W&B `pvc/value_symmetry_mae`. **Charted on
the PVC plot's secondary y-axis** (not the symmetry-KL chart as the plan suggested): the PVC chart
already reads the PVC parquet, so this keeps the sub-metric self-contained on one figure instead of
cross-wiring the PVC table into the policy-symmetry chart. The value MAE and the policy-symmetry KL
answer the same "is the net respecting the symmetry?" question for the two heads respectively.

---

## S5. Tests + docs

- **Unit tests** (real objects, no mocks): (a) negamax sign — a constructed position where the
  one-ply-best move is known, assert `Q₁` ranks it top; (b) terminal-child handling — a move that
  wins uses result `+1`, not `V`; (c) Spearman skips positions with `<2` candidates; (d) a
  hand-built case where policy and value fully agree → `argmax_match == 1`, `spearman == 1`, and one
  where they're reversed → `spearman == −1`.
- **Docs:** `docs/05-EVALUATION.md` — add PVC (definition, the one-ply-vs-multi-ply caveat, how to
  read it against the value-starvation / decoupling findings). Note it's diagnostic only.

---

## Design choices flagged for review (defaults chosen; change on sign-off)

1. **Candidate set = top-K=8 policy moves** (vs all legal). Top-K is cheap and focuses on the
   policy's mass; all-legal is more complete but costs ~10-50× more child evals in Blokus's wide
   move space. Default top-8.
2. **Two measures (argmax-match + Spearman).** argmax-match = "same best move"; Spearman = "same
   ranking." Keeping both; drop one if the chart is noisy.
3. **Include `value_symmetry_mae` (S4)?** Cheap and answers your reflection idea; defaulted to
   optional/low-priority so it doesn't hold up the core metric.
4. **Interpretation is a trend, not a target.** Documented in S3's caption + S5 docs so nobody reads
   "not 100%" as failure.

## Not in scope
Does not change training, search, or the loss — pure reporting instrumentation. Slots into the next
run's metrics; composes with the arena-derived-Elo work.
