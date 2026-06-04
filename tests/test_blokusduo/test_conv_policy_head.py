"""Conv-policy-head tests — the safety net for the policy read-out.

The conv head emits a ``(num_orientations, rows, cols)`` block and *reads it out*
into the flat 17,837-long action vector that MCTS / masking / training expect. A
wrong read-out is silent — no crash, just every action index pointing at the
wrong move. These tests prove the read-out against the **real** ``ActionCodec``
(the production source of truth), at three levels:

1. The permutation itself is a correct bijection matching ``ActionCodec``.
2. The head places *every* conv output at the exact ``ActionCodec.encode`` index.
3. End-to-end on real boards: the conv-head net produces a well-formed, maskable
   policy whose legal support decodes to real placements.

Levels 1-2 are what catch an axis-order or coordinate-flip bug; level 3 is the
integration sanity (shapes, masking alignment, no crash).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from core.config import MCTSConfig, NetConfig, RunConfig
from games.blokusduo.board import Action, ActionCodec, CoordinateIndexDecoder
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.net import ConvPolicyHead, build_action_permutation
from games.blokusduo.neuralnets.wrapper import NNetWrapper
from tests.fixtures.blokus_positions import load_cache, replay_to_board_and_player

if TYPE_CHECKING:
    from games.blokusduo.pieces import PieceManager

_PIECES = Path(__file__).resolve().parent.parent.parent / "games" / "blokusduo" / "pieces.json"
_DEV_CACHE = Path(__file__).resolve().parent.parent / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"
_N = 14  # Blokus Duo board size


def _wrapper(
    tmp_path: Path, policy_head: str, *, filters: int = 16, epochs: int = 1,
) -> tuple[BlokusDuoGame, NNetWrapper]:
    game = BlokusDuoGame(pieces_config_path=_PIECES)
    net_cfg = NetConfig(
        learning_rate=1e-3, dropout=0.3, epochs=epochs, batch_size=4, cuda=False,
        num_filters=filters, num_residual_blocks=1, policy_head=policy_head,
    )
    run_cfg = RunConfig(
        game="blokusduo", run_name="test_conv", num_generations=1, num_eps=2,
        temp_threshold=5, update_threshold=0.55, max_queue_length=10, num_arena_matches=2,
        max_generations_lookback=1, root_directory=tmp_path, load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0), net_config=net_cfg,
    )
    return game, NNetWrapper(game, run_cfg)


# ---------------------------------------------------------------------------
# Level 1 — the permutation
# ---------------------------------------------------------------------------

def test_permutation_is_bijection(piece_manager: PieceManager) -> None:
    """perm must be a permutation of [0, num_moves) — no slot dropped or doubled."""
    num_orientations = piece_manager.num_entries
    perm = build_action_permutation(_N, _N, num_orientations)
    assert perm.shape == (_N * _N * num_orientations,)
    np.testing.assert_array_equal(np.sort(perm), np.arange(perm.size))


def test_permutation_matches_action_codec(piece_manager: PieceManager) -> None:
    """Each perm entry must decode to the same (orientation, x, y) the real
    ActionCodec assigns to that action index — catches axis-order/flip bugs."""
    num_orientations = piece_manager.num_entries
    codec = ActionCodec(_N, piece_manager)
    decoder = CoordinateIndexDecoder(_N)
    perm = build_action_permutation(_N, _N, num_orientations)
    cells = _N * _N

    for action_index in range(perm.size):
        slot = int(perm[action_index])
        o = slot // cells
        rem = slot % cells
        row, col = divmod(rem, _N)
        x, y = decoder.to_coordinate((row, col))
        action = codec.decode(action_index)
        expected_o = piece_manager.get_piece_orientation_id((action.piece_id, action.orientation))
        assert o == expected_o, f"orientation mismatch at action {action_index}"
        assert (x, y) == (action.x_coordinate, action.y_coordinate), (
            f"coordinate mismatch at action {action_index}: perm says ({x},{y}), "
            f"ActionCodec says ({action.x_coordinate},{action.y_coordinate})")


# ---------------------------------------------------------------------------
# Level 2 — the head places every output at the right ActionCodec index
# ---------------------------------------------------------------------------

def test_conv_head_places_every_logit_at_action_codec_index(piece_manager: PieceManager) -> None:
    """The decisive read-out test: build a ConvPolicyHead, and assert its
    17,837-output equals the conv block reordered so output[ActionCodec index] ==
    move_conv[orientation, row, col] for *every* (orientation, row, col), with
    the pass logit at the pass index. Pure reordering => exact equality."""
    torch.manual_seed(0)
    num_orientations = piece_manager.num_entries
    head = ConvPolicyHead(num_filters=8, num_orientations=num_orientations,
                          board_rows=_N, board_cols=_N)
    head.eval()
    codec = ActionCodec(_N, piece_manager)
    decoder = CoordinateIndexDecoder(_N)

    features = torch.randn(1, 8, _N, _N)
    with torch.no_grad():
        moves = head.move_conv(features)[0].numpy()        # (O, rows, cols)
        pass_logit = float(head.pass_head(features)[0, 0])
        out = head(features)[0].numpy()                     # (action_size,)

    expected = np.empty(codec.action_size, dtype=out.dtype)
    for o in range(num_orientations):
        piece_id, orientation = piece_manager.get_piece_orientation(o)
        for row in range(_N):
            for col in range(_N):
                x, y = decoder.to_coordinate((row, col))
                idx = codec.encode(Action(piece_id, orientation, x, y))
                expected[idx] = moves[o, row, col]
    expected[codec.pass_action_index] = pass_logit

    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# Net-level: shape, normalisation, param count
# ---------------------------------------------------------------------------

def test_conv_net_policy_shape_and_normalises(tmp_path: Path) -> None:
    game, nnet = _wrapper(tmp_path, "conv")
    board = game.get_canonical_form(game.initialise_board(), 1)
    policy, value = nnet.predict(board)
    assert policy.shape == (game.get_action_size(),)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert -1.0 <= float(np.asarray(value).reshape(-1)[0]) <= 1.0


def test_conv_head_slashes_param_count(tmp_path: Path) -> None:
    """The whole point: the conv head removes the ~7M-param FC layer."""
    _, fc = _wrapper(tmp_path, "fc", filters=64)
    _, conv = _wrapper(tmp_path, "conv", filters=64)
    fc_params = sum(p.numel() for p in fc.nnet.parameters())
    conv_params = sum(p.numel() for p in conv.nnet.parameters())
    assert conv_params < fc_params / 10, (
        f"expected conv net far smaller; got conv={conv_params:,} fc={fc_params:,}")


# ---------------------------------------------------------------------------
# Level 3 — end-to-end on real boards
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DEV_CACHE.exists(), reason="dev_5000 fixture not present")
def test_conv_net_masked_policy_decodes_to_legal_moves(
    tmp_path: Path, piece_manager: PieceManager,
) -> None:
    """On real positions: the conv-head policy, masked to legal moves, keeps mass
    only on legal actions, and legal actions decode to placements the game
    accepts. Integration sanity for the full net→mask→decode pipeline."""
    game, nnet = _wrapper(tmp_path, "conv")
    codec = ActionCodec(_N, piece_manager)
    actions_array, n_moves_array = load_cache(_DEV_CACHE)

    indices = np.linspace(0, len(n_moves_array) - 1, num=4, dtype=int)
    for idx in indices:
        n_moves = int(n_moves_array[idx])
        board, player = replay_to_board_and_player(game, actions_array[idx, :n_moves])
        canonical = game.get_canonical_form(board, player)
        policy, _ = nnet.predict(canonical)
        valids = np.asarray(game.valid_move_masking(canonical, 1))

        masked = policy * valids
        # All probability mass sits on legal actions; the dimensions line up.
        assert float(masked[valids == 0].sum()) == 0.0
        assert masked.sum() > 0, "no legal move had any probability"

        # A sample of legal non-pass actions must decode to placements the game
        # accepts (applying them must not raise, coords in range).
        legal = [int(a) for a in np.where(valids > 0)[0] if a != codec.pass_action_index]
        for action_index in legal[:: max(1, len(legal) // 25)]:  # ~25 spread across the set
            action = codec.decode(action_index)
            game.get_next_state(canonical, 1, action_index)  # must not raise
            assert 0 <= action.x_coordinate < _N and 0 <= action.y_coordinate < _N


# ---------------------------------------------------------------------------
# checkpoint compatibility must fail loudly across head types
# ---------------------------------------------------------------------------

def test_loading_fc_checkpoint_into_conv_net_raises(tmp_path: Path) -> None:
    """The two heads have incompatible state_dicts. Loading an FC checkpoint
    into a conv net (or vice-versa) must raise, never load partially — strict
    load is the guard against silently running on a half-loaded net."""
    _, fc = _wrapper(tmp_path, "fc")
    fc.save_checkpoint("fc_head.pth.tar")

    _, conv = _wrapper(tmp_path, "conv")  # same run_name => same net_directory
    with pytest.raises(RuntimeError):
        conv.load_checkpoint("fc_head.pth.tar")


# ---------------------------------------------------------------------------
# the conv head actually trains (gradients flow, loss drops, no NaN)
# ---------------------------------------------------------------------------

def test_conv_net_trains_and_loss_drops(tmp_path: Path) -> None:
    """Train the conv-head net on a small batch of real positions with
    legal-move-masked targets. Confirms gradients flow end-to-end through the
    conv head + gather + pass head: the net's policy must move toward the
    targets (loss drops) and stay finite (no NaN), with valid output after."""
    game, nnet = _wrapper(tmp_path, "conv", filters=16, epochs=8)

    # Build ~16 examples: real boards, legal-move-masked uniform policy targets,
    # alternating value targets. A learnable signal the net should fit.
    rng = np.random.default_rng(0)
    board = game.initialise_board()
    player = 1
    examples = []
    for i in range(16):
        canonical = game.get_canonical_form(board, player)
        valids = np.asarray(game.valid_move_masking(canonical, 1), dtype=np.float64)
        if valids.sum() == 0:
            break
        target_pi = valids / valids.sum()
        examples.append((canonical.as_multi_channel(1), target_pi, float((-1) ** i) * 0.5))
        action = int(rng.choice(np.where(valids > 0)[0]))
        board, player = game.get_next_state(board, player, action)

    boards_t = torch.tensor(np.array([e[0] for e in examples]), dtype=torch.float32)
    targets_pi = torch.tensor(np.array([e[1] for e in examples]), dtype=torch.float32)

    def batch_pi_loss() -> float:
        nnet.nnet.eval()
        with torch.no_grad():
            log_pi, _ = nnet.nnet(boards_t)
        return float(nnet.loss_pi(targets_pi, log_pi).item())

    loss_before = batch_pi_loss()
    nnet.train(examples, generation=0)  # uses config epochs; must not raise/NaN
    loss_after = batch_pi_loss()

    assert np.isfinite(loss_before) and np.isfinite(loss_after)
    assert loss_after < loss_before, f"policy loss did not drop: {loss_before} -> {loss_after}"

    # Net still produces a valid policy after training.
    policy, value = nnet.predict(game.get_canonical_form(game.initialise_board(), 1))
    assert np.all(np.isfinite(policy)) and abs(float(policy.sum()) - 1.0) < 1e-5
    assert np.isfinite(float(np.asarray(value).reshape(-1)[0]))
