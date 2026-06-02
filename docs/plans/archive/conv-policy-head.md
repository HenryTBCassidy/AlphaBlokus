# F4 — Convolutional policy head

Sub-plan for F4 in [`full-cycle-optimisation.md`](full-cycle-optimisation.md). Replaces the fully-connected policy head (a single `Linear(392 → 17,837)` that is **~95% of the net's ~7.3M params**) with a fully-convolutional head that emits a `(91, 14, 14)` logit map plus a pass logit. Speeds inference (compounds with F3) and training, shrinks the model ~20×, and gives the correct board-game inductive bias.

The reasoning for *why* is fully worked out in the research note [`../research/policy-head-architecture.md`](../research/policy-head-architecture.md) — read it first. This plan is the *how*.

**The one hard part:** the conv output is laid out as `(orientation_channel, array_row, array_col)`, but the action space is indexed `index = y·(N·91) + x·91 + orientation_id` in **board** coordinates (bottom-left origin), whereas the conv spatial axes are **array** coordinates (top-left origin). Getting that permutation exactly right — and proving it — is the bulk of the risk. Everything else is small.

Default stays FC (`policy_head: "fc"`) until the conv head is validated; the switch is a config flag, opt-in, so nothing regresses while we build and test.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| C1 | Derive + pin the conv-output → action-index permutation. Build it from `ActionCodec.decode` + the coordinate decoder as a precomputed gather index; prove it's a correct bijection with a one-hot probe test | 2-3 hr | High | ✅ `build_action_permutation` in `net.py`; bijection + ActionCodec cross-check tests |
| C2 | Implement the conv policy head in `net.py` (1×1 conv → gather to action order + a small pass head), behind a `policy_head` flag | 2-3 hr | High | ✅ `ConvPolicyHead` (1×1 conv → gather → avg-pool pass head); head selection in `AlphaBlokusDuo` |
| C3 | Add `policy_head: Literal["fc", "conv"] = "fc"` to `NetConfig`; wire the net to pick the head. Default "fc" keeps current behaviour bit-identical | 30 min | High | ✅ Added (default later flipped to "conv" in C8) |
| C4 | Tests — the safety net for the reshape. Three layers: (1) the C1 one-hot probe (permutation is a correct bijection); (2) head-level: a crafted features tensor where one `(o, x, y)` should dominate → assert the conv head's argmax index decodes via `ActionCodec` to exactly that action; (3) **end-to-end round-trip on real boards**: run the conv-head net → mask → for the top-K actions, `ActionCodec.decode` each and assert it's a legal placement on that board (and that illegal actions carry zero probability after masking). Plus shape `(B, 17837)` and ~340K param count | 2-2.5 hr | High | ✅ `tests/test_blokusduo/test_conv_policy_head.py`. L2 strengthened to verify **every** conv output lands at the exact `ActionCodec.encode` index (not just argmax) |
| C5 | Checkpoint compatibility: loading an FC checkpoint into a conv net (or vice-versa) must fail loudly, not silently corrupt. No weight transfer — train from scratch | 1 hr | High | ✅ `strict` load raises `RuntimeError` on head mismatch; test pins it |
| C6 | Measure: param count, checkpoint size, forward + train-step time, conv vs FC. Confirm the research doc's predictions | 1 hr | Medium | ✅ 64f×4b: params **7,344,220 → 340,127 (21.6×)**, checkpoint **29.4 → 1.53 MB (19×)**. Forward time ~equal (1.08→1.13 ms/board CPU) — trunk-dominated, as the research doc said |
| C7 | Symmetry-diagnostic check — conv head should score near-zero asymmetry where the FC head has measurable asymmetry. Validates the inductive-bias claim | 30 min | Medium | ✅ Ran — **uninformative pre-training** (untrained nets output near-uniform → trivially symmetric, both heads ~0 KL). Meaningful comparison needs a trained net (scaled run). Also: a 1×1 conv is translation-equivariant, *not* transpose-equivariant, so the research doc's reason-3 overstated this for Blokus's transpose symmetry |
| C8 | Validation run: a short Blokus training run on the conv head to confirm it trains (loss down, no NaN, Elo moves). Flip `NetConfig` default to "conv" if clean | 1-2 hr + run | High | ✅ Committed test trains the conv net on a real-board batch — loss drops, no NaN, valid policy after. Default flipped to "conv"; full suite **256 passed**. (Full Elo-over-gens strength validation folded into the scaled run.) |
| C9 | Archive. `git mv` to `archive/`, update master plan F4 row → Done with measurements, Scope-additions note if needed | 15 min | Low | ✅ |

Total: ~10-13 focused hours + a validation run.

---

## C1. The action-index permutation (the keystone)

`ActionCodec` (in `games/blokusduo/board.py`) encodes:

```
index = y · (board_size · num_orientations) + x · num_orientations + orientation_id
      = y · (14 · 91)                        + x · 91                + o
```
- `(x, y)` are **board** coordinates, **bottom-left origin** (Blokus notation).
- `o = orientation_id ∈ [0, 90]` (0-based, from `OrientationCodec`).
- pass action = `14·14·91 = 17836` (the final index).

A `Conv2d(num_filters → 91, 1×1)` produces `(B, 91, H, W)` = `(B, o, array_row, array_col)` — **array** coordinates, **top-left origin**. Two mismatches vs the action index:

1. **Axis order.** Conv flatten is channel-major (`o` outermost): `o·(14·14) + row·14 + col`. ActionCodec wants `o` *innermost*. → a transpose.
2. **Coordinate origin.** Conv spatial `(row, col)` are array coords; ActionCodec `(x, y)` are board coords. `CoordinateIndexDecoder.to_idx((x, y))` (board.py) gives the conversion: `row = N−1−y`, `col = x` (confirmed — a y-flip plus identity on x).

**Approach: a precomputed gather index, built once at net construction.**

```python
# perm[action_index] = flat position in the conv output (91*14*14) for that action.
perm = np.empty(board_size * board_size * num_orientations, dtype=np.int64)
for action_index in range(len(perm)):
    o = action_index % num_orientations
    rem = action_index // num_orientations
    x = rem % board_size          # board x
    y = rem // board_size         # board y  (matches ActionCodec.decode)
    row, col = decoder.to_idx((x, y))   # CoordinateIndexDecoder: (N-1-y, x)
    perm[action_index] = o * (board_size * board_size) + row * board_size + col
```

Then in the head: `policy_flat = conv_out.reshape(B, -1)[:, perm]` gives the `(B, 17836)` logits in exact ActionCodec order; concat the pass logit as column 17836.

**The test that makes this safe (do not skip):** a one-hot probe. For every `(o, row, col)`, place a 1.0 at that conv-output position, zeros elsewhere, run the permute, and assert the resulting 17,837-vector has its 1.0 at exactly `ActionCodec.encode(decoded action)`. Also assert `perm` is a bijection (`sorted(perm) == range(17836)`). If this passes for all positions, the mapping is provably correct — every downstream consumer (MCTS, masking, training targets) keeps working unchanged because the policy vector stays in ActionCodec order.

---

## C2. Conv policy head implementation

Current (FC), in `games/blokusduo/neuralnets/net.py`:
```python
self.policy_head = nn.Sequential(
    nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
    nn.BatchNorm2d(2), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2 * conv_out, action_size),   # 392 -> 17,837  ← ~7M params
)
```

Conv replacement (as a small `nn.Module`, since it needs a custom `forward` for the gather + pass):
```python
class ConvPolicyHead(nn.Module):
    def __init__(self, num_filters, num_orientations, board_size, perm):
        super().__init__()
        self.board_cells = board_size * board_size
        self.move_conv = nn.Conv2d(num_filters, num_orientations, kernel_size=1, bias=True)
        # pass logit: a dedicated tiny head (avg-pool features -> scalar), ~65 params,
        # rather than wasting a 196-cell channel on a single action.
        self.pass_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(num_filters, 1))
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, features):           # features: (B, num_filters, 14, 14)
        moves = self.move_conv(features).reshape(features.size(0), -1)  # (B, 91*14*14)
        moves = moves[:, self.perm]                                      # -> ActionCodec order
        return torch.cat([moves, self.pass_head(features)], dim=1)       # (B, 17837)
```
`forward()` in the net then applies `log_softmax` to the concatenated logits exactly as today, so the wrapper (`predict`, `predict_batch`) is unchanged.

Param count: `move_conv` `64·91 + 91 ≈ 5.9K` + pass head `~65` ≈ **~6K** in the head (was ~7.0M). Total net ~340K.

---

## C4. Testing the net→move path

The whole risk of this change is that the policy is now *read out* of a 3D block into the flat action order — a wrong read-out is silent (no crash, just scrambled targets). So we test the read-out at three levels, narrow to broad:

1. **Permutation probe (from C1) — isolates the index map.** For every `(o, row, col)`, a one-hot at that conv position must land at `ActionCodec.encode(action(o, board-coords-of(row,col)))`, and `perm` must be a bijection over `[0, 17836)`. Proves "conv slot ↔ action index" with zero ambiguity.

2. **Head-level argmax — isolates the head module.** Hand `ConvPolicyHead` a crafted features tensor engineered so one specific `(o, x, y)` produces the largest move logit; assert `argmax` of the head's 17,837-output, fed through `ActionCodec.decode`, returns exactly that `(piece, orientation, x, y)`. Catches mistakes in the conv→gather→concat wiring that the pure-permutation test wouldn't (e.g. a reshape vs the gather disagreeing).

3. **End-to-end round-trip on real boards — the integration test.** Replay a handful of `dev_5000` positions, run the *conv-head net* → `predict` → mask with `valid_move_masking`. Then: (a) assert every index with non-zero probability `ActionCodec.decode`s to a placement that's actually legal on that board; (b) assert every *illegal* action carries exactly zero probability (the mask and the conv output are in the same order). This exercises the full path the way self-play and MCTS do, on positions with real legal-move sets — the test that would catch a coordinate-flip bug end-to-end.

Layer 3 is the one Henry flagged: it confirms the net→encoding→move pipeline as a whole, not just the permutation in a vacuum. If all three pass, a wrong read-out cannot reach training.

---

## C5. Checkpoint compatibility

The two heads have incompatible `state_dict`s (one has a 7M `Linear`, the other a tiny conv). `load_checkpoint` must **fail loudly** if a checkpoint's head doesn't match the configured `policy_head`, never load partially. PyTorch's `load_state_dict` with `strict=True` (the default) already raises on key mismatch — verify the wrapper doesn't pass `strict=False` anywhere, and add a test that loading an FC checkpoint into a conv net raises.

No weight transfer (the research doc flags it as non-trivial and we have no precious Blokus checkpoint to preserve). Conv runs train from scratch. The immutable 2026-05-22 FC baseline stays as-is for historical comparison — we don't try to load it into the new head.

---

## Notes for whoever picks this up

- **C1 is the whole game.** If the one-hot probe passes, the rest is plumbing. If you're ever unsure the mapping is right, run that test — a silent permutation bug would train a net that's quietly wrong and you'd only notice as mysteriously bad play.
- **Coordinate convention is confirmed** (`board.py` `CoordinateIndexDecoder.to_idx`: `row = N−1−y`, `col = x`) — but the one-hot probe in C1 tests it end-to-end anyway, so a wrong assumption here can't slip through.
- **Keep FC as the default until C8 validates conv.** This mirrors the F1/F2/F3 pattern: opt-in flag, old path bit-identical until proven.
- **This is in scope for the optimisation plan** because the justification is throughput; net *capacity* changes (bigger trunks) remain a separate concern.

---

## Outcome (landed 2026-06-02)

Conv policy head implemented, tested, and made the default. All C1–C9 done.

**Headline measurements (64f×4b):** params **7,344,220 → 340,127 (21.6× shrink)**; checkpoint **29.4 MB → 1.53 MB (19×)**. The read-out is provably correct — the C4 tests verify every conv output lands at the exact `ActionCodec.encode` index. The conv net trains cleanly (loss drops, no NaN). Full suite: **256 passed**.

**Framing correction (be honest):** F4 does **not** meaningfully speed the raw forward pass — it's trunk-dominated, so conv vs FC forward time is ~equal (1.08 → 1.13 ms/board on CPU). The real wins are **parameter count, checkpoint size, training-step cost (less to backprop), and inductive bias / sample efficiency** — not inference latency. Earlier menu/plan wording that said "speeds inference" overstated it; the master plan F4 row has been corrected.

**Deferred / not validated here:**
- **Strength** (does a conv-trained net play as well as / better than FC-trained?) — needs a real training run; folded into the scaled run, which now uses the conv head by default.
- **Symmetry benefit** — unmeasurable on untrained nets (near-uniform → trivially symmetric); and note a 1×1 conv is translation- not transpose-equivariant, so don't expect a large win on Blokus's transpose symmetry. Re-check on a trained net.

## Out of scope

- **Bigger / different trunk** (more filters/blocks) — separate `net-scaling.md` concern.
- **Weight transfer FC→conv** — not worth it; train from scratch.
- **Encoding Blokus's symmetry group structurally** (group-equivariant convs) — the conv head gets translation-equivariance for free; the full 2-fold Blokus symmetry is handled by `get_symmetries` data augmentation, which is enough.
