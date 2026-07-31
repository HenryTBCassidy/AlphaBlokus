"""The ownership and opponent-reply heads, end to end (plan N4/N5).

The score head's own tests live in ``test_score_head.py``; this file covers the two
heads added on top of its machinery, plus the contracts that only exist once there is
**more than one** auxiliary head:

1. **Off by default is bit-for-bit**, per head and for all of them together.
2. **Append-only construction order.** Turning on a later head must leave every earlier
   head's weights byte-identical at a fixed seed, or a one-head-at-a-time A/B is
   measuring a shifted RNG stream instead of the head.
3. **No positional unpacking.** With the score head off and the ownership head on,
   ``forward``'s element 2 is the *ownership* map — every consumer must resolve the
   layout through the net's ``aux_head_names``.
4. **Never read at play time**, whatever combination is built.
5. **Checkpoints cross the boundary both ways**, with a genuinely mismatched checkpoint
   still raising.
6. **Each head's own loss term is what drives it** — measured against a zero-weight arm,
   because a head reading the shared trunk improves anyway as the trunk does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from alphablokus.aux_heads import AUX_HEAD_NAMES, aux_key
from alphablokus.games.blokusduo.nn.net import OWNERSHIP_CLASSES
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from tests.conftest import RecordingMetrics
from tests.games.blokusduo.nn.aux_helpers import SEED, build_net, net_config, run_config, train_once
from tests.games.blokusduo.nn.aux_helpers import examples as build_examples

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame

CELLS = 14 * 14

# One arbitrary but fixed per-cell ownership map per test position, in the position's own
# canonical frame: mostly "mine", a band of "theirs", the rest unowned.
_OWNERSHIP = np.zeros((14, 14), dtype=np.int8)
_OWNERSHIP[:5, :] = 1
_OWNERSHIP[5:9, :] = -1


def _reply(action: int) -> tuple[np.ndarray, np.ndarray]:
    """A one-hot sparse reply target on ``action``."""
    return np.array([action], dtype=np.int32), np.array([1.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. Off by default, and 2. append-only construction order
# --------------------------------------------------------------------------- #


def test_the_new_heads_are_off_by_default() -> None:
    assert net_config().ownership_head is False
    assert net_config().reply_head is False
    assert net_config().ownership_loss_weight == 0.15
    assert net_config().reply_loss_weight == 0.15


def test_with_every_head_off_the_net_is_the_pre_change_two_tuple(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    net = build_net(blokus_game, blokus_board, net_config())

    assert net.aux_head_names == ()
    assert not [key for key in net.state_dict() if key.startswith(("score_head", "ownership_head", "reply_head"))]

    net.eval()
    with torch.no_grad():
        outputs = net(torch.zeros(2, blokus_board.num_channels, 14, 14))
    assert len(outputs) == 2, "with every head off the forward output must be the pre-change 2-tuple"


@pytest.mark.parametrize("head", ["ownership_head", "reply_head"])
def test_turning_one_head_on_leaves_every_other_parameter_byte_identical(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, head: str
) -> None:
    """The property the whole A/B rests on: only the new head's tensors are new."""
    off = build_net(blokus_game, blokus_board, net_config())
    on = build_net(blokus_game, blokus_board, net_config(**{head: True}))

    off_state, on_state = off.state_dict(), on.state_dict()
    for key, tensor in off_state.items():
        assert tensor.numpy().tobytes() == on_state[key].numpy().tobytes(), f"{key} changed"

    extra = sorted(key for key in on_state if key not in off_state)
    assert extra and all(key.startswith(f"{head}.") for key in extra)

    off.eval()
    on.eval()
    planes = torch.zeros(2, blokus_board.num_channels, 14, 14)
    with torch.no_grad():
        off_out, on_out = off(planes), on(planes)
    assert torch.equal(off_out[0], on_out[0])
    assert torch.equal(off_out[1], on_out[1])


def test_a_later_head_never_perturbs_an_earlier_one(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard) -> None:
    """Heads are appended, so each is initialised from the same RNG position every time.

    Walked cumulatively — score, then score+ownership, then all three — because the
    failure this guards against is exactly an *insertion*: putting a new head anywhere
    but last silently re-randomises every head after it, and then an A/B that flips one
    flag is comparing two different networks.
    """
    previous: dict[str, bytes] = {}
    enabled: dict[str, bool] = {}
    for name in AUX_HEAD_NAMES:
        enabled[name] = True
        net = build_net(blokus_game, blokus_board, net_config(**enabled))
        state = {key: tensor.numpy().tobytes() for key, tensor in net.state_dict().items()}
        for key, value in previous.items():
            assert state[key] == value, f"{key} was re-randomised by building {name}"
        previous = state
    assert set(enabled) == set(AUX_HEAD_NAMES)


def test_aux_head_names_match_the_forward_output_order(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    """``forward`` appends built heads in ``AUX_HEAD_NAMES`` order and says so."""
    net = build_net(blokus_game, blokus_board, net_config(score_head=True, ownership_head=True, reply_head=True))
    assert net.aux_head_names == AUX_HEAD_NAMES

    net.eval()
    with torch.no_grad():
        outputs = net(torch.zeros(2, blokus_board.num_channels, 14, 14))

    assert len(outputs) == 2 + len(AUX_HEAD_NAMES)
    assert outputs[2].shape == (2, 1)  # score
    assert outputs[3].shape == (2, OWNERSHIP_CLASSES, 14, 14)  # ownership
    assert outputs[4].shape == (2, blokus_game.get_action_size())  # reply
    # The reply head emits log-probabilities, like the main policy head.
    assert torch.allclose(outputs[4].exp().sum(dim=1), torch.ones(2), atol=1e-5)


def test_head_parameter_costs_at_the_production_net_size(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    """Both heads must be a rounding error on the trunk at the 192x12 ``large`` preset.

    The ownership head is one 1x1 convolution; the reply head is a second copy of the
    (already tiny) conv policy head. Neither may approach the cost of a residual block,
    or the A/B is confounded by capacity.
    """
    large = net_config(num_filters=192, num_residual_blocks=12)
    base = sum(p.numel() for p in build_net(blokus_game, blokus_board, large).parameters())

    for head in ("ownership_head", "reply_head"):
        net = build_net(blokus_game, blokus_board, net_config(num_filters=192, num_residual_blocks=12, **{head: True}))
        grown = sum(p.numel() for p in net.parameters())
        assert 0 < grown - base < 0.01 * base, f"{head} costs {(grown - base) / base:.2%} of the net"


# --------------------------------------------------------------------------- #
# 3. Layout resolution, not positional unpacking
# --------------------------------------------------------------------------- #


def test_the_output_split_is_by_name_not_position(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """With score off and ownership on, element 2 is ownership — reading it as the
    score is the exact bug a fixed-position unpack would introduce, and it would be
    silent (both are just tensors)."""
    wrapper = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(ownership_head=True)))
    planes = torch.zeros(2, blokus_board.num_channels, 14, 14)

    with torch.no_grad():
        outputs = wrapper._split_net_outputs(wrapper.nnet(planes))

    assert set(outputs.aux) == {"ownership"}
    assert outputs.aux["ownership"].shape == (2, OWNERSHIP_CLASSES, 14, 14)


def test_aux_key_maps_head_names_to_the_config_and_metric_spelling() -> None:
    assert [aux_key(name) for name in AUX_HEAD_NAMES] == ["score", "ownership", "reply"]
    for name in AUX_HEAD_NAMES:
        assert hasattr(net_config(), name)
        assert hasattr(net_config(), f"{aux_key(name)}_loss_weight")


# --------------------------------------------------------------------------- #
# 4. Never read at play time
# --------------------------------------------------------------------------- #


def test_the_play_surface_still_returns_policy_and_value_only(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """``predict``/``predict_batch``/``predict_encoded`` are what search, the arena and
    the Pentobi harness call — they must be untouched with every head built."""
    config = net_config(score_head=True, ownership_head=True, reply_head=True)
    wrapper = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, config))

    assert all(wrapper.has_aux_head(aux_key(name)) for name in AUX_HEAD_NAMES)
    assert len(wrapper.predict(blokus_board)) == 2
    assert len(wrapper.predict_batch([blokus_board, blokus_board])) == 2
    planes = blokus_board.as_multi_channel(1)[np.newaxis, ...]
    assert len(wrapper.predict_encoded(planes)) == 2

    policies, values, aux = wrapper.predict_encoded_aux(planes)
    assert set(aux) == {"score", "ownership", "reply"}
    assert aux["ownership"].shape == (1, OWNERSHIP_CLASSES, 14, 14)
    assert aux["reply"].shape == (1, blokus_game.get_action_size())
    # Class probabilities per cell, so the metric code can take logs directly.
    assert np.allclose(aux["ownership"].sum(axis=1), 1.0, atol=1e-5)
    # The diagnostics surface must agree with the play surface on the first two outputs.
    play_policies, play_values = wrapper.predict_encoded(planes)
    assert np.array_equal(policies, play_policies)
    assert np.array_equal(values, play_values)


def test_a_headless_net_reports_no_auxiliary_outputs(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    wrapper = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config()))

    assert not any(wrapper.has_aux_head(aux_key(name)) for name in AUX_HEAD_NAMES)
    assert wrapper.predict_encoded_aux(blokus_board.as_multi_channel(1)[np.newaxis, ...])[2] == {}


# --------------------------------------------------------------------------- #
# 5. Checkpoint compatibility, both directions
# --------------------------------------------------------------------------- #


def _shared_bytes(wrapper: BlokusDuoNNetWrapper) -> dict[str, bytes]:
    """Every non-auxiliary-head tensor of a net, as raw bytes."""
    return {
        key: tensor.detach().cpu().numpy().tobytes()
        for key, tensor in wrapper.nnet.state_dict().items()
        if not key.startswith(("score_head.", "ownership_head.", "reply_head."))
    }


@pytest.mark.parametrize("head", ["ownership_head", "reply_head"])
def test_an_older_checkpoint_warm_starts_a_net_with_the_new_head(
    blokus_game: BlokusDuoGame, tmp_path: Path, head: str
) -> None:
    """Donor without the head → net with it: loads, and the head stays fresh."""
    donor = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config()))
    donor.save_checkpoint("plain.pth.tar")

    target = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(**{head: True})))
    module = getattr(target.nnet, head)
    fresh = {key: value.detach().clone() for key, value in module.state_dict().items()}

    target.load_weights("plain.pth.tar")

    assert _shared_bytes(target) == _shared_bytes(donor)
    for key, value in module.state_dict().items():
        assert torch.equal(value, fresh[key]), f"{head} tensor {key} was overwritten"


@pytest.mark.parametrize("head", ["ownership_head", "reply_head"])
def test_a_new_head_checkpoint_loads_into_a_plain_net(blokus_game: BlokusDuoGame, tmp_path: Path, head: str) -> None:
    """The direction that keeps evaluation, ONNX export and the jax bridge working."""
    donor = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(**{head: True})))
    donor.save_checkpoint("with_head.pth.tar")

    target = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config()))
    target.load_weights("with_head.pth.tar")

    assert _shared_bytes(target) == _shared_bytes(donor)
    assert getattr(target.nnet, head) is None


def test_tolerance_does_not_extend_to_a_different_net_size(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """A mismatched filter count is a wrong checkpoint, head or no head."""
    donor = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(num_filters=8, ownership_head=True)))
    donor.save_checkpoint("small.pth.tar")

    target = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(num_filters=16, ownership_head=True)))

    with pytest.raises(RuntimeError):
        target.load_weights("small.pth.tar")


def test_tolerance_does_not_extend_to_a_missing_trunk(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """Dropping a real tensor must still raise even while a head is tolerated."""
    donor = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config()))
    donor.save_checkpoint("plain.pth.tar")

    path = run_config(tmp_path, net_config()).net_directory / "plain.pth.tar"
    checkpoint = torch.load(path, map_location="cpu")
    dropped = next(key for key in checkpoint["state_dict"] if key.startswith("conv_block"))
    del checkpoint["state_dict"][dropped]
    torch.save(checkpoint, path)

    target = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(ownership_head=True)))
    with pytest.raises(RuntimeError, match="does not match this network architecture"):
        target.load_weights("plain.pth.tar")


# --------------------------------------------------------------------------- #
# 6. The ownership loss
# --------------------------------------------------------------------------- #


def test_ownership_loss_masks_cells_rather_than_scoring_them() -> None:
    """``-1`` labels must contribute neither error nor gradient.

    Exact check: masking half the cells gives *identically* the loss of the unmasked
    half alone, and the masked rows receive a zero gradient.
    """
    logits = torch.zeros(2, OWNERSHIP_CLASSES, 2, 2, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 0], [-1, -1, -1, -1]])

    loss = BlokusDuoNNetWrapper.loss_ownership(targets, logits)
    loss.backward()

    # Uniform logits ⇒ every unmasked cell costs ln 3, and the masked row is excluded.
    assert loss.item() == pytest.approx(float(np.log(OWNERSHIP_CLASSES)))
    assert logits.grad is not None
    assert torch.equal(logits.grad[1], torch.zeros(OWNERSHIP_CLASSES, 2, 2))
    assert torch.isfinite(logits.grad).all()


def test_a_fully_masked_ownership_batch_is_a_zero_term() -> None:
    """No final board anywhere in the batch must give 0, not NaN from a 0/0 mean."""
    logits = torch.zeros(2, OWNERSHIP_CLASSES, 2, 2, requires_grad=True)
    loss = BlokusDuoNNetWrapper.loss_ownership(torch.full((2, 4), -1), logits)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits))


def test_ownership_head_on_adds_its_weighted_term_to_the_total(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    examples = build_examples(blokus_game, blokus_board, 8)
    config = net_config(ownership_head=True)

    _, metrics = train_once(blokus_game, config, tmp_path, examples, ownership_targets=[_OWNERSHIP] * len(examples))

    assert metrics.rows
    for row in metrics.rows:
        assert row["ownership_loss"] is not None and row["ownership_loss"] > 0.0
        expected = row["pi_loss"] + row["v_loss"] + config.ownership_loss_weight * row["ownership_loss"]
        assert row["total_loss"] == pytest.approx(expected, rel=1e-5)


def test_ownership_head_off_ignores_the_targets(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    examples = build_examples(blokus_game, blokus_board, 8)

    _, metrics = train_once(
        blokus_game, net_config(), tmp_path, examples, ownership_targets=[_OWNERSHIP] * len(examples)
    )

    assert metrics.rows
    for row in metrics.rows:
        assert "ownership_loss" not in row or row["ownership_loss"] is None
        assert row["total_loss"] == pytest.approx(row["pi_loss"] + row["v_loss"], rel=1e-6)


def test_ownership_head_on_without_targets_warns(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """Silently training a head on nothing is the failure mode this must not have."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        examples = build_examples(blokus_game, blokus_board, 8)
        _, metrics = train_once(blokus_game, net_config(ownership_head=True), tmp_path, examples)
    finally:
        logger.remove(sink_id)

    assert any("ownership_head is on but train() got no ownership targets" in message for message in messages)
    for row in metrics.rows:
        assert row["total_loss"] == pytest.approx(row["pi_loss"] + row["v_loss"], rel=1e-6)


def test_misaligned_ownership_targets_are_rejected(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    examples = build_examples(blokus_game, blokus_board, 4)
    wrapper = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path, net_config(ownership_head=True)))

    with pytest.raises(ValueError, match="index-aligned"):
        wrapper.train(examples, generation=1, ownership_targets=[_OWNERSHIP, _OWNERSHIP])


def test_the_ownership_term_is_what_drives_the_ownership_head(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """Gradients reach the head *because of the ownership term*, not by accident.

    Asserting only that the loss falls is not enough — the head reads the shared trunk,
    which keeps improving under the policy and value losses, so the loss drifts down at
    weight 0 too. The comparison is against a zero-weight arm: identical data, identical
    seed, identical everything except whether the term contributes.
    """
    examples = build_examples(blokus_game, blokus_board, 8)
    targets = [_OWNERSHIP] * len(examples)

    def final_loss(weight: float) -> float:
        config = net_config(ownership_head=True, learning_rate=1e-2, ownership_loss_weight=weight)
        torch.manual_seed(SEED)
        wrapper = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path / f"w{weight}", config))
        metrics = RecordingMetrics()
        torch.manual_seed(SEED + 1)
        for generation in range(1, 9):
            wrapper.train(examples, generation=generation, metrics=metrics, ownership_targets=targets)
        tail = [row["ownership_loss"] for row in metrics.rows[-4:]]
        return sum(tail) / len(tail)

    driven = final_loss(0.15)
    drifting = final_loss(0.0)
    assert driven < drifting, (
        f"the ownership term made no difference: {driven:.4f} with it, {drifting:.4f} without — "
        "the head is being carried by the shared trunk rather than trained"
    )


# --------------------------------------------------------------------------- #
# 7. The reply loss
# --------------------------------------------------------------------------- #


def test_reply_loss_equals_the_policy_loss_when_nothing_is_masked() -> None:
    """Same target shape, same KL: the reply head is a policy head on another target."""
    log_probs = torch.log_softmax(torch.tensor([[0.2, 0.5, 0.3], [1.0, 0.0, -1.0]]), dim=1)
    targets = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])

    assert BlokusDuoNNetWrapper.loss_reply(targets, log_probs).item() == pytest.approx(
        BlokusDuoNNetWrapper.loss_pi(targets, log_probs).item()
    )


def test_reply_loss_masks_the_all_zero_rows() -> None:
    """A position with no next ply is an all-zero target: no error, no gradient.

    Exact check: masking one of two rows gives *identically* the loss of the other row
    alone — the mask must change the denominator too, not just the numerator.
    """
    log_probs = torch.log_softmax(torch.zeros(2, 3), dim=1).requires_grad_(True)
    masked = torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.0, 0.0]])
    only_real = torch.tensor([[0.5, 0.25, 0.25]])

    loss = BlokusDuoNNetWrapper.loss_reply(masked, log_probs)
    loss.backward()

    alone = BlokusDuoNNetWrapper.loss_reply(only_real, torch.log_softmax(torch.zeros(1, 3), dim=1))
    assert loss.item() == pytest.approx(alone.item())
    assert log_probs.grad is not None
    assert torch.equal(log_probs.grad[1], torch.zeros(3))
    assert torch.isfinite(log_probs.grad).all()


def test_a_non_finite_reply_row_neither_contributes_nor_poisons_the_batch() -> None:
    """Regression: masking *after* the arithmetic leaves ``NaN x 0 = NaN``.

    The score loss already zeroes its targets before the subtraction for exactly this
    reason. The reply loss masked on ``sum > 0``, which reads ``False`` for a ``NaN`` row
    and so looked safe — but the row's KL was still computed and still ``NaN``, and one
    such row takes the whole batch's loss and gradient with it.
    """
    log_probs = torch.log_softmax(torch.zeros(2, 3), dim=1).requires_grad_(True)
    poisoned = torch.tensor([[0.5, 0.25, 0.25], [float("nan"), 0.0, 0.0]])
    only_real = torch.tensor([[0.5, 0.25, 0.25]])

    loss = BlokusDuoNNetWrapper.loss_reply(poisoned, log_probs)
    loss.backward()

    alone = BlokusDuoNNetWrapper.loss_reply(only_real, torch.log_softmax(torch.zeros(1, 3), dim=1))
    assert loss.item() == pytest.approx(alone.item())
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()
    assert torch.equal(log_probs.grad[1], torch.zeros(3))


def test_a_fully_masked_reply_batch_is_a_zero_term() -> None:
    log_probs = torch.log_softmax(torch.zeros(2, 3), dim=1).requires_grad_(True)
    loss = BlokusDuoNNetWrapper.loss_reply(torch.zeros(2, 3), log_probs)
    loss.backward()

    assert loss.item() == 0.0
    assert log_probs.grad is not None and torch.equal(log_probs.grad, torch.zeros_like(log_probs))


def test_reply_head_on_adds_its_weighted_term_to_the_total(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    examples = build_examples(blokus_game, blokus_board, 8)
    config = net_config(reply_head=True)
    # Each position's reply target is the *next* position's move, and the last is masked.
    replies = [example[1] for example in examples[1:]] + [None]

    _, metrics = train_once(blokus_game, config, tmp_path, examples, reply_targets=replies)

    assert metrics.rows
    for row in metrics.rows:
        assert row["reply_loss"] is not None and row["reply_loss"] > 0.0
        expected = row["pi_loss"] + row["v_loss"] + config.reply_loss_weight * row["reply_loss"]
        assert row["total_loss"] == pytest.approx(expected, rel=1e-5)


def test_reply_head_off_ignores_the_targets(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    examples = build_examples(blokus_game, blokus_board, 8)
    replies = [example[1] for example in examples[1:]] + [None]

    _, metrics = train_once(blokus_game, net_config(), tmp_path, examples, reply_targets=replies)

    assert metrics.rows
    for row in metrics.rows:
        assert "reply_loss" not in row or row["reply_loss"] is None
        assert row["total_loss"] == pytest.approx(row["pi_loss"] + row["v_loss"], rel=1e-6)


def test_reply_head_on_without_targets_warns(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        examples = build_examples(blokus_game, blokus_board, 8)
        train_once(blokus_game, net_config(reply_head=True), tmp_path, examples)
    finally:
        logger.remove(sink_id)

    assert any("reply_head is on but train() got no reply targets" in message for message in messages)


def test_the_reply_term_is_what_drives_the_reply_head(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """As for ownership: compare against a zero-weight arm, not against nothing."""
    examples = build_examples(blokus_game, blokus_board, 8)
    replies = [example[1] for example in examples[1:]] + [None]

    def final_loss(weight: float) -> float:
        config = net_config(reply_head=True, learning_rate=1e-2, reply_loss_weight=weight)
        torch.manual_seed(SEED)
        wrapper = BlokusDuoNNetWrapper(blokus_game, run_config(tmp_path / f"w{weight}", config))
        metrics = RecordingMetrics()
        torch.manual_seed(SEED + 1)
        for generation in range(1, 9):
            wrapper.train(examples, generation=generation, metrics=metrics, reply_targets=replies)
        tail = [row["reply_loss"] for row in metrics.rows[-4:]]
        return sum(tail) / len(tail)

    driven = final_loss(0.15)
    drifting = final_loss(0.0)
    assert driven < drifting, (
        f"the reply term made no difference: {driven:.4f} with it, {drifting:.4f} without — "
        "the head is being carried by the shared trunk rather than trained"
    )


# --------------------------------------------------------------------------- #
# 8. The DataLoader-side target sources
# --------------------------------------------------------------------------- #


def test_the_target_sources_pickle_and_append_in_head_order(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """The wrapper ships to spawn/forkserver DataLoader workers, so it must pickle.

    Also pins the item layout: ``(board, pi, value)`` then one tensor per head in
    ``AUX_HEAD_NAMES`` order — the order ``train()`` zips the batch back up in.
    """
    import pickle

    from alphablokus.games.base_wrapper import (
        _AuxTargetDataset,
        _OwnershipTargetSource,
        _ReplyTargetSource,
        _ScoreTargetSource,
    )
    from alphablokus.training.memmap_dataset import MemmapPolicyDataset

    examples = build_examples(blokus_game, blokus_board, 4)
    action_size = blokus_game.get_action_size()
    base = MemmapPolicyDataset.build(examples, action_size, blokus_game.encode_compact, tmp_path / "memmap")
    sources = {
        "score": _ScoreTargetSource(np.array([0.1, np.nan, -0.3, 0.4], dtype=np.float32)),
        "ownership": _OwnershipTargetSource([_OWNERSHIP, None, _OWNERSHIP, -_OWNERSHIP], CELLS),
        "reply": _ReplyTargetSource([_reply(3), _reply(7), None, _reply(11)], action_size),
    }

    dataset = pickle.loads(pickle.dumps(_AuxTargetDataset(base, sources, len(examples))))

    assert len(dataset) == 4
    for index in range(4):
        board, pi, value, score, ownership, reply = dataset[index]
        assert board.shape == base[index][0].shape
        assert torch.equal(pi, base[index][1])
        assert value == base[index][2]
        assert ownership.shape == (CELLS,)
        assert reply.shape == (action_size,)

    # Ownership: {-1, 0, +1} shifts to class indices, and a missing map is all -1.
    assert torch.equal(dataset[0][4], torch.from_numpy(_OWNERSHIP.reshape(-1).astype(np.int64) + 1))
    assert torch.equal(dataset[1][4], torch.full((CELLS,), -1, dtype=torch.int64))
    assert torch.equal(dataset[3][4], torch.from_numpy((-_OWNERSHIP).reshape(-1).astype(np.int64) + 1))
    # Reply: a sparse target densifies, and a missing one is the all-zero sentinel.
    assert dataset[0][5][3] == pytest.approx(1.0) and dataset[0][5].sum() == pytest.approx(1.0)
    assert dataset[2][5].sum() == 0.0
    # Score: NaN is the missing sentinel.
    assert torch.isnan(dataset[1][3])
