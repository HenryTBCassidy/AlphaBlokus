# Score head — an auxiliary training target for the value body

Adds a third output to the Blokus net predicting the **final score margin**, trained
alongside the existing policy and value heads and **never read at play time**. The aim is
not to play for points; it is to give the shared body a richer signal to learn from, and
to break a measured shortcut in the value head.

**Why now.** Measured on 400 real v2 corpus games (`docs/research/corpus-quality-principles.md`):

| Positions where… | win | loss | draw |
|---|---|---|---|
| White to move | **79.0%** | 5.1% | 15.9% |
| Black to move | 5.6% | **77.5%** | 16.9% |

A predictor that sees **only whose turn it is** and never looks at the board gets 78.3% of
outcomes right at **0.304 MSE** (against 0.836 for always predicting a draw). Blokus Duo's
first-player advantage is severe and has no komi to offset it, so "did they win?" is
largely answerable from piece parity. The margin is not: v1 measured margins from −43 to
+88, mean +12.4, median 3. Two White wins of +3 and +40 are the same value label and very
different positions.

**Locked design decisions** (settled before writing this — do not re-litigate):

- **Auxiliary, never primary.** The value head keeps predicting the game result and search
  keeps choosing moves by it. AlphaGo's habit of playing slack moves in won games is
  correct behaviour for maximising win probability; a net that maximised margin would take
  risks to win bigger and lose games it had won. KataGo keeps win-rate primary and adds
  only a small score bonus in its utility — we add **no** bonus at all.
- **Not read at inference.** `predict`/`predict_batch` return `(pi, v)` exactly as today, so
  MCTS, the arena, the Pentobi harness, the jax bridge and the ONNX export are untouched.
  The head cannot influence play because nothing consults it when playing.
- **Off by default.** `NetConfig.score_head = False` reproduces today's net bit-for-bit.
  This is an experiment with an A/B, not a new default.
- **The data already exists.** Every v2 corpus row stores `margin` from the side to move.
  No regeneration, and v1 corpora (which also store it) work too.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| S1 | `NetConfig` flags: `score_head`, `score_loss_weight`, `score_scale` (all inert by default) | 1 h | High | ✅ |
| S2 | The head itself in `nn/net.py`; `forward` returns a 3-tuple, `predict`/`predict_batch` still return `(pi, v)` | 2 h | High | ✅ |
| S3 | Checkpoint compatibility both ways — warm-start an old net into a score-head net and vice versa | 2 h | High | ✅ |
| S4 | `BaseNNetWrapper.train` takes optional per-example scores and adds the weighted loss term | 3 h | High | ✅ |
| S5 | Thread `margin` out of the corpus loaders as the score target | 2 h | High | |
| S6 | `distill_sl.py` wiring + report score MSE and the change in value skill | 2 h | High | |
| S7 | **A/B on the box**: identical SL fits with and without the head | ½ day box GPU | High | |
| S8 | Decision + default: keep, drop, or retune — recorded with the numbers | 1 h | High | |

**Gate (S8):** keep the head only if it improves **value skill** (`1 − mse / colour_only_mse`,
the diagnostic added in PR #61) *without* hurting held-out policy agreement. A score head
that improves nothing is complexity for its own sake and gets deleted.

---

## S1. Config flags

Three fields on `NetConfig`, all inert unless `score_head` is on:

- `score_head: bool = False` — build the third head at all.
- `score_loss_weight: float = 0.15` — how much the score term contributes. Deliberately
  small: the body is being *nudged* to see more, not retargeted. KataGo's score weight is
  likewise a fraction of the value weight.
- `score_scale: float = 25.0` — see S2 for what this does and why 25.

Frozen dataclass, so this is three fields plus their docstrings. Existing configs are
unaffected and existing checkpoints stay loadable (S3).

## S2. The head

A near-copy of the value head — same input, same shape, one number out:

```
1×1 conv → BatchNorm → ReLU → Flatten → Linear(→ num_filters) → ReLU → Linear(→ 1) → Tanh
```

**Target scaling.** Raw margins run roughly −88…+88 with a median around 3, so a raw-margin
MSE would be dominated by a handful of blowouts and would fight the value head for the
body's capacity. The target is therefore `tanh(margin / score_scale)`, and the head ends in
`tanh` to match. At `score_scale = 25` a 3-point win maps to 0.12, 10 points to 0.38, 25 to
0.76, and 60+ saturates near 1 — so the resolution sits where the mass is (small margins)
and lopsided games stop pulling. This mirrors how the value head is bounded and keeps both
targets on the same scale, so `score_loss_weight` means what it says.

**Interface.** `forward` returns `(log_pi, value, score)`; when the head is off, `score` is
`None`. `predict`/`predict_batch` — the `IPolicyValuePredictor` surface MCTS uses — keep
returning `(pi, v)` and simply drop the third element. Nothing downstream changes, and
**no code path consults the score when choosing a move**.

> **As built — the arity varies instead of the third element being `None`.** A `None` in a
> module's output makes it untraceable (`torch.jit.trace`: *"Only tensors, lists, tuples of
> tensors, or dictionary of tensors can be output from traced functions"*), which would
> break `scripts/export_web_assets.py`'s ONNX export **even with the head off** — the one
> thing S2 promises not to touch. So `forward` returns a 2-tuple with the head off (byte
> for byte today's output) and a 3-tuple with it on. Call sites unpack through the single
> helper `BaseNNetWrapper._split_net_outputs`.
>
> The head is also constructed **after** the policy head, not before, so that at a fixed
> seed the trunk, value head and policy head initialise identically with the head on and
> off — the S7 arms then differ by the head alone rather than by a shifted RNG stream.

## S3. Checkpoint compatibility

Two directions, both needed and both easy to get silently wrong:

- **Old checkpoint → score-head net.** The distillation warm-start arm loads v3 gen-40,
  which has no score-head weights. A strict `load_state_dict` raises. `load_weights` must
  load non-strictly, leave the new head at its initialisation, and **log which tensors were
  missing** so a genuinely mismatched checkpoint is still loud.
- **Score-head checkpoint → plain net.** Extra unexpected keys must be ignored, so a net
  trained with the head can still be evaluated, exported to ONNX, or used by the jax bridge.

Test both directions explicitly: a real save/load round trip each way, asserting the shared
body's weights are byte-identical afterwards.

> **As built.** `alphablokus/training/checkpoint_compat.py::load_state_dict_compat` is the
> one implementation, used by `load_checkpoint`, `load_weights` and the five scripts that
> loaded a raw `state_dict` themselves. Tolerance is **scoped to the `score_head.` prefix**:
> any other missing or unexpected tensor still raises, so the existing fc-vs-conv
> policy-head guard survives. Cross-architecture warm starts must go through
> `load_weights`, not `load_checkpoint` — the latter also restores optimizer state, whose
> param groups genuinely do not match across a head change.

## S4. The training loss

Today: `total = policy_loss + value_loss`. With the head on:

```
total = policy_loss + value_loss + score_loss_weight × score_loss
```

`score_loss` is MSE against the scaled target, same form as the value loss.

`train()` gains `scores: Sequence[float] | None = None`, index-aligned with `examples`. It
stays optional so self-play, the replay buffer and every existing caller are untouched —
`ProcessedExample` does **not** change shape, which is what keeps this out of the self-play
pipeline entirely. If the head is on and scores are absent, the term is skipped and a
warning is logged once (silently training a head on nothing is exactly the sort of thing
this project has been bitten by).

Log the score loss alongside `pi_loss`/`v_loss` in `TrainingData` so it appears in the
report without special-casing.

## S5. Margins out of the loaders

`CorpusGameRows` already carries per-position rows; add `margins` alongside `values`, read
from the `margin` column that both v1 and v2 shards store. `build_training_examples`
returns them index-aligned with the examples (including the symmetry twin, which shares its
original's margin — transposing a board does not change the score).

Opening rows have no single margin (a DAG node has many games through it); use the mean
margin of the playouts beneath it where `link` has computed one, and skip the score term for
the rest rather than inventing a number.

## S6. Trainer wiring

`distill_sl.py` passes the margins through and reports, per epoch:

- score MSE on the held-out set;
- **value skill** (`1 − value_mse / colour_only_value_mse`) — already added in PR #61 — which
  is the number this whole plan exists to move.

Both into the run JSON so the S7 arms are comparable after the fact.

## S7. The A/B

Two SL fits on the finished stage-1 corpus, identical but for the head: same data, same
seed, same net size, same schedule. Compare on the held-out set:

| Metric | What it tells us |
|---|---|
| **value skill** | the point of the exercise — is the value head reading the board yet? |
| top-1 / top-3 agreement with Pentobi | did the policy head pay for it? |
| value MSE, per colour | is the improvement real or a re-shuffle between colours |
| score MSE | is the head learning anything at all |

Also worth a look while the arms are running: whether the score-head arm reaches a given
policy accuracy in fewer epochs. The published claim for auxiliary targets is faster
learning as much as better endpoints.

## S8. Decision

Record the numbers and pick one, in the plan and in the config default:

- **Keep** (flip `score_head` on for the RL phase) if value skill improves materially with
  policy agreement flat or better.
- **Retune** if score MSE is near zero (the head has an easy job — `score_scale` too small)
  or barely moves (too large, or the weight is too low).
- **Drop** if nothing improves. An auxiliary target that does not help is dead weight in
  every future run, and this project already has a habit of accumulating unused paths.

---

## Not in scope

- **No score bonus in search.** The utility MCTS maximises is unchanged. Revisit only with
  evidence, and never before the A/B.
- **No win/draw/loss head.** Our 22% draw rate is a genuine argument for one — a single
  number cannot separate "certain draw" from "coin flip", which is why Lc0 moved to WDL —
  but that is a different change with a different blast radius, and mixing the two would
  make the A/B unreadable. Recorded in `docs/IDEAS.md` instead.
- **No self-play changes.** `ProcessedExample` keeps its shape; the score target reaches the
  trainer through a separate optional argument, so the self-play path never sees it.
