"""The auxiliary score head, end to end (plan ``docs/plans/score-auxiliary-target.md``).

Four contracts, in the order they can go wrong:

1. **Off by default is bit-for-bit.** With ``score_head=False`` the module, its state
   dict, its parameter count and its forward output are the pre-change net's.
2. **Never read at inference.** ``predict`` / ``predict_batch`` / ``predict_encoded`` —
   the surface search, the arena and the Pentobi harness use — keep returning
   ``(pi, v)``. Only the explicit diagnostics call exposes the score.
3. **Checkpoints cross the boundary both ways**, with the shared body byte-identical
   afterwards and any *other* mismatch still raising.
4. **The loss term is real when on and absent when off**, and masks positions with no
   margin instead of inventing one.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from alphablokus.config import MCTSConfig, NetConfig, RunConfig
from alphablokus.games.blokusduo.nn.net import AlphaBlokusDuo
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from alphablokus.training.score_target import scale_margin
from tests.conftest import RecordingMetrics

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame

SEED = 20260730


def _net_config(**overrides: object) -> NetConfig:
    """A tiny CPU net config; ``overrides`` flips the score-head knobs."""
    base = NetConfig(
        learning_rate=5e-3,
        dropout=0.0,
        epochs=1,
        batch_size=4,
        cuda=False,
        num_filters=16,
        num_residual_blocks=1,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]  # kwargs are field values


def _run_config(tmp_path: Path, net_config: NetConfig) -> RunConfig:
    return RunConfig(
        game="blokusduo",
        run_name="score_head_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=5,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=tmp_path,
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0),
        net_config=net_config,
    )


def _build_net(game: BlokusDuoGame, board: BlokusDuoBoard, net_config: NetConfig) -> AlphaBlokusDuo:
    """Seeded construction, so two nets differ only where the architecture does."""
    rows, cols = game.get_board_size()
    torch.manual_seed(SEED)
    return AlphaBlokusDuo(
        board_rows=rows,
        board_cols=cols,
        action_size=game.get_action_size(),
        num_input_channels=board.num_channels,
        config=net_config,
    )


# --------------------------------------------------------------------------- #
# 1. Off by default
# --------------------------------------------------------------------------- #


def test_score_head_is_off_by_default() -> None:
    assert _net_config().score_head is False
    assert _net_config().score_loss_weight == 0.15
    assert _net_config().score_scale == 25.0


def test_head_off_builds_no_head_and_returns_the_two_tuple(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    net = _build_net(blokus_game, blokus_board, _net_config())

    assert net.score_head is None
    assert not [key for key in net.state_dict() if key.startswith("score_head")]

    net.eval()
    with torch.no_grad():
        outputs = net(torch.zeros(2, blokus_board.num_channels, 14, 14))
    assert len(outputs) == 2, "with the head off the forward output must be the pre-change 2-tuple"


def test_turning_the_head_on_leaves_every_other_parameter_untouched(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    """The strongest form of "off is bit-for-bit", and of "the A/B differs by the head".

    At the same seed the trunk, value head and policy head must initialise *identically*
    with the head on and off — which is why the score head is constructed last, after
    every other head, rather than in the middle of the RNG stream.
    """
    off = _build_net(blokus_game, blokus_board, _net_config())
    on = _build_net(blokus_game, blokus_board, _net_config(score_head=True))

    off_state, on_state = off.state_dict(), on.state_dict()
    for key, tensor in off_state.items():
        assert tensor.numpy().tobytes() == on_state[key].numpy().tobytes(), f"{key} changed"

    extra = sorted(key for key in on_state if key not in off_state)
    assert extra and all(key.startswith("score_head.") for key in extra)

    off.eval()
    on.eval()
    planes = torch.zeros(2, blokus_board.num_channels, 14, 14)
    with torch.no_grad():
        off_out, on_out = off(planes), on(planes)
    assert torch.equal(off_out[0], on_out[0])
    assert torch.equal(off_out[1], on_out[1])


def test_head_costs_about_half_a_percent_at_the_production_net_size(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    """The plan's cost claim, at the ``large`` preset the runs actually use (192x12)."""
    large = _net_config(num_filters=192, num_residual_blocks=12)
    off = _build_net(blokus_game, blokus_board, large)
    on = _build_net(blokus_game, blokus_board, replace(large, score_head=True))

    off_params = sum(p.numel() for p in off.parameters())
    on_params = sum(p.numel() for p in on.parameters())

    assert 0.0 < (on_params - off_params) / off_params < 0.01


# --------------------------------------------------------------------------- #
# 2. Never read at inference
# --------------------------------------------------------------------------- #


def test_head_on_returns_a_bounded_third_output(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard) -> None:
    net = _build_net(blokus_game, blokus_board, _net_config(score_head=True))

    net.eval()
    with torch.no_grad():
        outputs = net(torch.zeros(3, blokus_board.num_channels, 14, 14))

    assert len(outputs) == 3
    score = outputs[2]
    assert score.shape == (3, 1)
    assert bool(((score > -1.0) & (score < 1.0)).all()), "the head is tanh-bounded like its target"


def test_inference_surface_never_exposes_the_score(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """``predict``/``predict_batch``/``predict_encoded`` are what search calls."""
    wrapper = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config(score_head=True)))

    assert wrapper.has_score_head()
    assert len(wrapper.predict(blokus_board)) == 2
    assert len(wrapper.predict_batch([blokus_board, blokus_board])) == 2
    planes = blokus_board.as_multi_channel(1)[np.newaxis, ...]
    assert len(wrapper.predict_encoded(planes)) == 2

    policies, values, scores = wrapper.predict_encoded_with_score(planes)
    assert scores is not None and scores.shape == (1,)
    # The diagnostics surface must agree with the play surface on the first two outputs.
    play_policies, play_values = wrapper.predict_encoded(planes)
    assert np.array_equal(policies, play_policies)
    assert np.array_equal(values, play_values)


def test_a_headless_net_reports_no_score(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    wrapper = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config()))

    assert not wrapper.has_score_head()
    assert wrapper.predict_encoded_with_score(blokus_board.as_multi_channel(1)[np.newaxis, ...])[2] is None


# --------------------------------------------------------------------------- #
# 3. Checkpoint compatibility, both directions
# --------------------------------------------------------------------------- #


def _body_bytes(wrapper: BlokusDuoNNetWrapper) -> dict[str, bytes]:
    """Every non-score-head tensor of a net, as raw bytes."""
    return {
        key: tensor.detach().cpu().numpy().tobytes()
        for key, tensor in wrapper.nnet.state_dict().items()
        if not key.startswith("score_head.")
    }


def test_old_checkpoint_warm_starts_a_score_head_net(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """v3-style donor (no score weights) → score-head net: loads, head left fresh."""
    donor = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config()))
    donor.save_checkpoint("plain.pth.tar")

    target = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config(score_head=True)))
    assert target.nnet.score_head is not None
    fresh_head = {k: v.detach().clone() for k, v in target.nnet.score_head.state_dict().items()}

    target.load_weights("plain.pth.tar")

    assert _body_bytes(target) == _body_bytes(donor)
    for key, value in target.nnet.score_head.state_dict().items():
        assert torch.equal(value, fresh_head[key]), f"score head tensor {key} was overwritten"


def test_score_head_checkpoint_loads_into_a_plain_net(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """The direction that keeps evaluation, ONNX export and the jax bridge working."""
    donor = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config(score_head=True)))
    donor.save_checkpoint("scored.pth.tar")

    target = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config()))
    target.load_weights("scored.pth.tar")

    assert _body_bytes(target) == _body_bytes(donor)
    assert target.nnet.score_head is None


def test_a_genuinely_wrong_checkpoint_still_raises(blokus_game: BlokusDuoGame, tmp_path: Path) -> None:
    """Tolerance is scoped to the score head: an fc/conv head swap must stay loud."""
    donor = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config(policy_head="fc")))
    donor.save_checkpoint("fc.pth.tar")

    target = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config(policy_head="conv")))

    with pytest.raises(RuntimeError, match="does not match this network architecture"):
        target.load_weights("fc.pth.tar")


# --------------------------------------------------------------------------- #
# 4. The loss term
# --------------------------------------------------------------------------- #


def _examples(blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, count: int) -> list:
    """``count`` trivially-distinct training examples in the stored (sparse) shape."""
    examples = []
    board = blokus_board
    player = 1
    for i in range(count):
        legal = np.flatnonzero(blokus_game.valid_move_masking(board, player))
        action = int(legal[i % len(legal)])
        indices = np.array([action], dtype=np.int32)
        values = np.array([1.0], dtype=np.float32)
        compact = np.asarray(board.to_compact(), dtype=np.int8)
        examples.append((compact, (indices, values), float((-1) ** i)))
        board, player = blokus_game.get_next_state(board, player, action)
        board = blokus_game.get_canonical_form(board, player)
        player = 1
    return examples


def _train_once(
    blokus_game: BlokusDuoGame,
    net_config: NetConfig,
    tmp_path: Path,
    examples: list,
    score_margins: list[float | None] | None,
) -> tuple[BlokusDuoNNetWrapper, RecordingMetrics]:
    torch.manual_seed(SEED)
    wrapper = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, net_config))
    metrics = RecordingMetrics()
    torch.manual_seed(SEED + 1)
    wrapper.train(examples, generation=1, metrics=metrics, score_margins=score_margins)
    return wrapper, metrics


def test_head_off_ignores_margins_and_logs_no_score_loss(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """With the head off the total is exactly ``pi + v`` and no score column appears."""
    examples = _examples(blokus_game, blokus_board, 8)
    margins: list[float | None] = [3.0, -7.0, 0.0, 40.0, -2.0, 12.0, 5.0, -20.0]

    _, metrics = _train_once(blokus_game, _net_config(), tmp_path, examples, margins)

    assert metrics.rows
    for row in metrics.rows:
        assert row["score_loss"] is None
        assert row["total_loss"] == pytest.approx(row["pi_loss"] + row["v_loss"], rel=1e-6)


def test_head_on_adds_the_weighted_score_term_to_the_total(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """With the head on the total is ``pi + v + w·score`` and the term is non-trivial."""
    examples = _examples(blokus_game, blokus_board, 8)
    margins: list[float | None] = [3.0, -7.0, 0.0, 40.0, -2.0, 12.0, 5.0, -20.0]
    net_config = _net_config(score_head=True)

    _, metrics = _train_once(blokus_game, net_config, tmp_path, examples, margins)

    assert metrics.rows
    for row in metrics.rows:
        assert row["score_loss"] is not None and row["score_loss"] > 0.0
        expected = row["pi_loss"] + row["v_loss"] + net_config.score_loss_weight * row["score_loss"]
        assert row["total_loss"] == pytest.approx(expected, rel=1e-5)
        assert row["total_loss"] > row["pi_loss"] + row["v_loss"]


def test_head_on_without_margins_trains_the_old_total_and_warns(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """Silently training a head on nothing is the failure mode this must not have."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        examples = _examples(blokus_game, blokus_board, 8)
        _, metrics = _train_once(blokus_game, _net_config(score_head=True), tmp_path, examples, None)
    finally:
        logger.remove(sink_id)

    assert any("score_head is on but train() got no score_margins" in message for message in messages)
    for row in metrics.rows:
        assert row["score_loss"] is None
        assert row["total_loss"] == pytest.approx(row["pi_loss"] + row["v_loss"], rel=1e-6)


def test_zero_weight_keeps_the_total_unchanged_but_still_reports_the_head(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    examples = _examples(blokus_game, blokus_board, 8)
    margins: list[float | None] = [3.0, -7.0, 0.0, 40.0, -2.0, 12.0, 5.0, -20.0]

    _, metrics = _train_once(
        blokus_game, _net_config(score_head=True, score_loss_weight=0.0), tmp_path, examples, margins
    )

    for row in metrics.rows:
        assert row["score_loss"] is not None
        assert row["total_loss"] == pytest.approx(row["pi_loss"] + row["v_loss"], rel=1e-6)


def test_the_head_actually_learns_its_target(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """Gradients reach the head: repeated passes over a fittable set drive its MSE down."""
    examples = _examples(blokus_game, blokus_board, 8)
    margins: list[float | None] = [3.0, -7.0, 0.0, 40.0, -2.0, 12.0, 5.0, -20.0]
    net_config = _net_config(score_head=True, learning_rate=1e-2)

    torch.manual_seed(SEED)
    wrapper = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, net_config))
    metrics = RecordingMetrics()
    torch.manual_seed(SEED + 1)
    for generation in range(1, 9):
        wrapper.train(examples, generation=generation, metrics=metrics, score_margins=margins)

    first = metrics.rows[0]["score_loss"]
    last = metrics.rows[-1]["score_loss"]
    assert last < first

    # And the head's own predictions now track the scaled targets' sign more often
    # than not — proof it is regressing the margin, not emitting a constant.
    planes = np.stack([blokus_game.encode_compact(board) for board, _pi, _value in examples])
    _, _, scores = wrapper.predict_encoded_with_score(planes)
    assert scores is not None
    targets = np.array([scale_margin(m, net_config.score_scale) for m in margins])  # type: ignore[arg-type]
    assert float(np.corrcoef(scores, targets)[0, 1]) > 0.0


def test_masked_positions_contribute_nothing_rather_than_a_zero_target() -> None:
    """v2 opening rows have no single margin; ``None`` must mean absent, not ``0.0``.

    Exact check: masking two of four rows gives *identically* the loss of the two real
    rows alone, and a different number from the "``None`` means 0.0" reading — which
    would quietly teach the head that every opening is a dead-level game.
    """
    outputs = torch.tensor([[0.1], [0.9], [-0.9], [0.3]])
    masked_targets = torch.tensor([0.5, float("nan"), -0.25, float("nan")])
    zeroed_targets = torch.tensor([0.5, 0.0, -0.25, 0.0])
    subset = BlokusDuoNNetWrapper.loss_score(torch.tensor([0.5, -0.25]), torch.tensor([[0.1], [-0.9]]))

    masked = BlokusDuoNNetWrapper.loss_score(masked_targets, outputs)

    assert masked.item() == pytest.approx(subset.item(), rel=1e-9)
    assert masked.item() != pytest.approx(BlokusDuoNNetWrapper.loss_score(zeroed_targets, outputs).item())


def test_a_partly_masked_batch_still_trains(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """End to end: a mixed batch (some rows without a margin) yields a finite term."""
    examples = _examples(blokus_game, blokus_board, 4)

    _, metrics = _train_once(
        blokus_game, _net_config(score_head=True, batch_size=4), tmp_path, examples, [40.0, None, -30.0, None]
    )

    for row in metrics.rows:
        assert row["score_loss"] is not None and row["score_loss"] > 0.0
        assert np.isfinite(row["total_loss"])


def test_all_margins_missing_contributes_a_zero_term(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """A fully-masked batch must give 0, not NaN from a 0/0 mean."""
    examples = _examples(blokus_game, blokus_board, 4)

    _, metrics = _train_once(
        blokus_game, _net_config(score_head=True, batch_size=4), tmp_path, examples, [None, None, None, None]
    )

    for row in metrics.rows:
        assert row["score_loss"] == 0.0
        assert np.isfinite(row["total_loss"])


def test_misaligned_margins_are_rejected(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """Silently training on other positions' margins would show up in no metric."""
    examples = _examples(blokus_game, blokus_board, 4)
    wrapper = BlokusDuoNNetWrapper(blokus_game, _run_config(tmp_path, _net_config(score_head=True)))

    with pytest.raises(ValueError, match="index-aligned"):
        wrapper.train(examples, generation=1, score_margins=[1.0, 2.0])


def test_loss_score_masks_and_averages_over_real_targets_only() -> None:
    """The loss primitive on its own: NaN targets neither contribute nor poison."""
    targets = torch.tensor([0.5, float("nan"), -0.5, float("nan")])
    outputs = torch.tensor([[0.0], [0.9], [0.0], [-0.9]], requires_grad=True)

    loss = BlokusDuoNNetWrapper.loss_score(targets, outputs)
    loss.backward()

    assert loss.item() == pytest.approx(0.25)  # mean of 0.5² and 0.5²
    assert outputs.grad is not None
    assert torch.equal(outputs.grad[[1, 3]], torch.zeros(2, 1)), "masked rows must get no gradient"
    assert torch.isfinite(outputs.grad).all()


# --------------------------------------------------------------------------- #
# 5. Downstream consumers of a score-head state dict
# --------------------------------------------------------------------------- #


def test_the_jax_bridge_converts_a_score_head_state_dict(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard
) -> None:
    """The jax self-play net reads named tensors, so the extra head must be inert.

    ``convert_state_dict`` needs numpy only (torch is lazy there), so this runs without
    the jax extra installed.
    """
    from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict

    net_config = _net_config(score_head=True, num_residual_blocks=1)
    net = _build_net(blokus_game, blokus_board, net_config)

    params = convert_state_dict(net.state_dict(), net_config.num_residual_blocks)

    assert set(params) == {"trunk", "blocks", "value", "policy", "perm"}
    assert len(params["blocks"]) == net_config.num_residual_blocks


def test_scored_dataset_appends_the_target_to_the_memmap_dataset(
    blokus_game: BlokusDuoGame, blokus_board: BlokusDuoBoard, tmp_path: Path
) -> None:
    """The DataLoader-worker path is a *wrapped* memmap dataset, not a second one.

    Also checks the wrapper pickles: ``dataloader_workers > 0`` ships the dataset to
    spawn/forkserver workers, and an unpicklable wrapper would only fail on the box.
    """
    import pickle

    from alphablokus.games.base_wrapper import _ScoredDataset
    from alphablokus.training.memmap_dataset import MemmapPolicyDataset

    examples = _examples(blokus_game, blokus_board, 4)
    base = MemmapPolicyDataset.build(
        examples, blokus_game.get_action_size(), blokus_game.encode_compact, tmp_path / "memmap"
    )
    targets = np.array([0.1, np.nan, -0.3, 0.4], dtype=np.float32)

    scored = pickle.loads(pickle.dumps(_ScoredDataset(base, targets)))

    assert len(scored) == 4
    for index in range(4):
        board, pi, value, score = scored[index]
        assert board.shape == base[index][0].shape
        assert torch.equal(pi, base[index][1])
        assert value == base[index][2]
        if np.isnan(targets[index]):
            assert torch.isnan(score)
        else:
            assert score.item() == pytest.approx(float(targets[index]))
