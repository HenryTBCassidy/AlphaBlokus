"""Batched MCTS inference safety-net tests.

Two invariants are pinned here, both established *before* the iterative
leaf-collection refactor so we can prove the refactor didn't change
behaviour:

1. ``predict_batch(boards)`` is numerically equivalent to N separate
   ``predict(board)`` calls. The whole batched-inference speedup rests on
   this — collapsing N batch-of-1 forward passes into one batch-of-N pass
   must not change the numbers (modulo float noise). Pinned on TicTacToe
   and on real Blokus positions replayed from the dev fixture cache.

2. MCTS is deterministic given a fixed network: the same network + same seed +
   same sim count produces identical visit counts. This is the contract the
   refactor must preserve at ``K=1`` (the batched path with batch size 1 has
   to stay bit-identical to today's recursive search).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from core.config import MCTSConfig, NetConfig, RunConfig
from core.mcts import MCTS
from games.blokusduo.game import BlokusDuoGame
from games.tictactoe.game import TicTacToeGame
from tests.fixtures.blokus_positions import load_cache, replay_to_board_and_player

if TYPE_CHECKING:
    from games.base_wrapper import BaseNNetWrapper

_PIECES_PATH = Path(__file__).resolve().parent.parent.parent / "games" / "blokusduo" / "pieces.json"
_DEV_CACHE = Path(__file__).resolve().parent.parent / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"


# ---------------------------------------------------------------------------
# Wrapper builders — small CPU nets, untrained (random weights are fine; we
# only test that batched == serial, not that the net is any good).
# ---------------------------------------------------------------------------

def _ttt_wrapper(tmp_path: Path) -> tuple[TicTacToeGame, BaseNNetWrapper]:
    from games.tictactoe.neuralnets.wrapper import NNetWrapper

    game = TicTacToeGame()
    config = _run_config(tmp_path, game="tictactoe", num_filters=32, blocks=1)
    return game, NNetWrapper(game, config)


def _blokus_wrapper(tmp_path: Path) -> tuple[BlokusDuoGame, BaseNNetWrapper]:
    from games.blokusduo.neuralnets.wrapper import NNetWrapper

    game = BlokusDuoGame(pieces_config_path=_PIECES_PATH)
    config = _run_config(tmp_path, game="blokusduo", num_filters=16, blocks=1)
    return game, NNetWrapper(game, config)


def _run_config(tmp_path: Path, *, game: str, num_filters: int, blocks: int) -> RunConfig:
    net_config = NetConfig(
        learning_rate=1e-3, dropout=0.3, epochs=1, batch_size=4, cuda=False,
        num_filters=num_filters, num_residual_blocks=blocks,
    )
    return RunConfig(
        game=game, run_name="test_batched", num_generations=1, num_eps=2,
        temp_threshold=5, update_threshold=0.55, max_queue_length=10,
        num_arena_matches=2, max_generations_lookback=1, root_directory=tmp_path,
        load_model=False, mcts_config=MCTSConfig(num_mcts_sims=8, cpuct=1.0),
        net_config=net_config,
    )


def _assert_batch_matches_serial(nnet, boards) -> None:
    """predict_batch(boards) must equal [predict(b) for b in boards]."""
    serial = [nnet.predict(b) for b in boards]
    policies, values = nnet.predict_batch(boards)

    assert len(policies) == len(boards)
    assert len(values) == len(boards)
    for i, (serial_policy, serial_value) in enumerate(serial):
        serial_value = float(np.asarray(serial_value).reshape(-1)[0])
        np.testing.assert_allclose(serial_policy, policies[i], atol=1e-6, rtol=0)
        assert abs(serial_value - values[i]) < 1e-6, (
            f"value mismatch at {i}: serial={serial_value}, batched={values[i]}"
        )
        assert isinstance(values[i], float)


# ---------------------------------------------------------------------------
# predict_batch equivalence — TicTacToe
# ---------------------------------------------------------------------------

def test_predict_batch_matches_serial_ttt(tmp_path: Path) -> None:
    game, nnet = _ttt_wrapper(tmp_path)
    empty = game.initialise_board()
    boards = [empty]
    # A spread of mid-game positions reached by distinct opening moves.
    for opening in (0, 2, 4, 6, 8):
        board, player = game.get_next_state(empty, 1, opening)
        boards.append(game.get_canonical_form(board, player))
    _assert_batch_matches_serial(nnet, boards)


def test_predict_batch_single_board_ttt(tmp_path: Path) -> None:
    """A length-1 batch must equal a plain predict() — the K=1 base case."""
    game, nnet = _ttt_wrapper(tmp_path)
    board = game.initialise_board()
    _assert_batch_matches_serial(nnet, [board])


def test_predict_batch_order_independence_ttt(tmp_path: Path) -> None:
    """Each board's result must depend only on the board, not its batch position."""
    game, nnet = _ttt_wrapper(tmp_path)
    empty = game.initialise_board()
    boards = []
    for opening in (0, 1, 3, 5, 7):
        board, player = game.get_next_state(empty, 1, opening)
        boards.append(game.get_canonical_form(board, player))

    forward_pols, forward_vals = nnet.predict_batch(boards)
    reversed_pols, reversed_vals = nnet.predict_batch(list(reversed(boards)))

    n = len(boards)
    for i in range(n):
        np.testing.assert_allclose(forward_pols[i], reversed_pols[n - 1 - i], atol=1e-6, rtol=0)
        assert abs(forward_vals[i] - reversed_vals[n - 1 - i]) < 1e-6


# ---------------------------------------------------------------------------
# predict_batch equivalence — Blokus (real positions from the dev fixture)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DEV_CACHE.exists(), reason="dev_5000 fixture not present")
def test_predict_batch_matches_serial_blokus(tmp_path: Path) -> None:
    game, nnet = _blokus_wrapper(tmp_path)
    actions_array, n_moves_array = load_cache(_DEV_CACHE)

    # A handful of positions spanning early/mid/late game, evenly spaced
    # through the cache so we're not just testing near-empty boards.
    boards = []
    indices = np.linspace(0, len(n_moves_array) - 1, num=8, dtype=int)
    for idx in indices:
        n_moves = int(n_moves_array[idx])
        sequence = actions_array[idx, :n_moves]
        board, player = replay_to_board_and_player(game, sequence)
        boards.append(game.get_canonical_form(board, player))

    _assert_batch_matches_serial(nnet, boards)


# ---------------------------------------------------------------------------
# MCTS determinism contract — the invariant the batched-inference refactor must preserve
# ---------------------------------------------------------------------------

# Golden visit counts + policy captured from the recursive MCTS that
# predates the iterative refactor, with torch.manual_seed(0) for the
# network weights, np.random.seed(123) before each search, num_mcts_sims=50,
# cpuct=1.0, on a 32-filter / 1-block TTT net. The default mcts_batch_size=1
# path must reproduce these byte-for-byte — that is the proof the iterative
# leaf-collection refactor did not change behaviour.
_GOLDEN_TTT = {
    "empty": {
        "openings": [],
        "visit_counts": [6, 5, 6, 5, 6, 5, 5, 5, 6, 0],
        "probs": [0.1224489796, 0.1020408163, 0.1224489796, 0.1020408163,
                  0.1224489796, 0.1020408163, 0.1020408163, 0.1020408163,
                  0.1224489796, 0.0],
    },
    "after_0": {
        "openings": [0],
        "visit_counts": [0, 6, 6, 6, 6, 6, 6, 6, 7, 0],
        "probs": [0.0, 0.1224489796, 0.1224489796, 0.1224489796, 0.1224489796,
                  0.1224489796, 0.1224489796, 0.1224489796, 0.1428571429, 0.0],
    },
    "after_0_4": {
        "openings": [0, 4],
        "visit_counts": [0, 7, 7, 7, 0, 7, 7, 7, 7, 0],
        "probs": [0.0, 0.1428571429, 0.1428571429, 0.1428571429, 0.0,
                  0.1428571429, 0.1428571429, 0.1428571429, 0.1428571429, 0.0],
    },
}


@pytest.mark.parametrize("label", list(_GOLDEN_TTT))
def test_k1_matches_pre_f3_golden(tmp_path: Path, label: str) -> None:
    """K=1 batched MCTS reproduces the older recursive search bit-for-bit.

    Compares against visit counts and policies frozen from the recursive
    implementation. Any divergence means the iterative refactor changed
    selection or backprop — a regression to fix, not a test to relax.
    """
    import torch

    case = _GOLDEN_TTT[label]
    torch.manual_seed(0)
    np.random.seed(0)
    game, nnet = _ttt_wrapper(tmp_path)  # 32f/1b net — matches the golden capture
    config = MCTSConfig(num_mcts_sims=50, cpuct=1.0)  # mcts_batch_size defaults to 1

    np.random.seed(123)
    mcts = MCTS(game, nnet, config)
    board = game.initialise_board()
    player = 1
    for move in case["openings"]:
        board, player = game.get_next_state(board, player, move)
    canonical = game.get_canonical_form(board, player)
    probs = mcts.get_action_prob(canonical, temp=1)

    s = game.state_key(canonical)
    visit_counts = [mcts.visit_counts.get((s, a), 0) for a in range(game.get_action_size())]
    assert visit_counts == case["visit_counts"], (
        f"[{label}] visit counts diverged from pre-F3 golden:\n"
        f"  expected {case['visit_counts']}\n  got      {visit_counts}"
    )
    np.testing.assert_allclose(probs, case["probs"], atol=1e-9, rtol=0)


def test_mcts_visit_counts_deterministic_ttt(tmp_path: Path) -> None:
    """Same network + same seed + same sims => identical visit counts.

    Both MCTS instances share one (untrained, random-weight) network, so the
    priors are identical; any divergence in visit counts would mean the search
    itself is non-deterministic. The K=1 batched path must keep this property.
    """
    game, nnet = _ttt_wrapper(tmp_path)
    config = MCTSConfig(num_mcts_sims=25, cpuct=1.0)
    board = game.get_canonical_form(game.initialise_board(), 1)

    def run() -> dict:
        np.random.seed(1234)
        mcts = MCTS(game, nnet, config)
        mcts.get_action_prob(board, temp=1)
        return dict(mcts.visit_counts)

    first = run()
    second = run()
    assert first == second, "MCTS visit counts diverged across identical runs"
    assert sum(first.values()) > 0, "search recorded no visits"


# ---------------------------------------------------------------------------
# Batched (K>1) search — determinism and self-consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [4, 8, 16])
def test_batched_search_deterministic_ttt(tmp_path: Path, batch_size: int) -> None:
    """Same network + same seed + same K => identical visit counts.

    The K>1 path must be reproducible: backprop runs in a fixed descent order
    and virtual loss is fully unwound each batch, so two identical runs cannot
    diverge. This is the determinism contract for batched self-play.
    """
    game, nnet = _ttt_wrapper(tmp_path)
    config = MCTSConfig(num_mcts_sims=48, cpuct=1.0, mcts_batch_size=batch_size)
    board = game.get_canonical_form(game.initialise_board(), 1)

    def run() -> dict:
        np.random.seed(99)
        mcts = MCTS(game, nnet, config)
        mcts.get_action_prob(board, temp=1)
        # Virtual loss must net to zero once a batch is fully backpropped.
        assert not mcts.virtual_visits, "virtual loss left dangling after search"
        return dict(mcts.visit_counts)

    assert run() == run(), f"K={batch_size} batched search diverged across runs"


@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.skipif(not _DEV_CACHE.exists(), reason="dev_5000 fixture not present")
def test_batched_search_self_consistent_blokus(tmp_path: Path, batch_size: int) -> None:
    """On real Blokus positions, batched MCTS yields a well-formed policy.

    For each K, the returned policy must: sum to 1, put probability only on
    legal moves, and total exactly ``num_mcts_sims`` visits across the root's
    edges minus the first cold-start batch that lands on the (then unexpanded)
    root. Confirms the iterative machinery is correct on the 44-channel net and
    the huge action space — not just on TTT.
    """
    game, nnet = _blokus_wrapper(tmp_path)
    actions_array, n_moves_array = load_cache(_DEV_CACHE)

    # A mid-game position with plenty of legal moves.
    idx = len(n_moves_array) // 3
    n_moves = int(n_moves_array[idx])
    sequence = actions_array[idx, :n_moves]
    board, player = replay_to_board_and_player(game, sequence)
    canonical = game.get_canonical_form(board, player)

    num_sims = 32
    config = MCTSConfig(num_mcts_sims=num_sims, cpuct=1.0, mcts_batch_size=batch_size)
    np.random.seed(5)
    mcts = MCTS(game, nnet, config)
    probs = np.asarray(mcts.get_action_prob(canonical, temp=1))

    assert not mcts.virtual_visits, "virtual loss left dangling after search"
    assert abs(probs.sum() - 1.0) < 1e-9, f"policy sums to {probs.sum()}, expected 1.0"

    valids = game.valid_move_masking(canonical, 1)
    illegal_mass = float(probs[np.asarray(valids) == 0].sum())
    assert illegal_mass == 0.0, f"{illegal_mass} probability mass on illegal moves"

    # Total root visits = num_sims minus the first cold-start batch of size
    # min(K, num_sims) whose descents all land on the still-unexpanded root.
    s = game.state_key(canonical)
    root_visits = sum(mcts.visit_counts.get((s, a), 0) for a in range(game.get_action_size()))
    expected = num_sims - min(batch_size, num_sims)
    assert root_visits == expected, (
        f"K={batch_size}: root visits {root_visits}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# fp16 inference flag — must be a clean no-op on CPU
# ---------------------------------------------------------------------------

def test_fp16_inference_flag_noop_on_cpu(tmp_path: Path) -> None:
    """fp16 autocast only engages on CUDA; on CPU the flag must be a no-op so
    the flag is safe to leave set. Confirms plumbing without needing a GPU."""
    import torch

    game = TicTacToeGame()
    board = game.get_canonical_form(game.initialise_board(), 1)

    def predict_with(fp16: bool):
        torch.manual_seed(0)
        net_config = NetConfig(
            learning_rate=1e-3, dropout=0.3, epochs=1, batch_size=4, cuda=False,
            num_filters=32, num_residual_blocks=1, fp16_inference=fp16,
        )
        run_config = RunConfig(
            game="tictactoe", run_name="t", num_generations=1, num_eps=2, temp_threshold=5,
            update_threshold=0.55, max_queue_length=10, num_arena_matches=2,
            max_generations_lookback=1, root_directory=tmp_path, load_model=False,
            mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0), net_config=net_config,
        )
        from games.tictactoe.neuralnets.wrapper import NNetWrapper
        return NNetWrapper(game, run_config).predict(board)

    pol_off, val_off = predict_with(False)
    pol_on, val_on = predict_with(True)
    np.testing.assert_array_equal(pol_off, pol_on)
    assert float(np.asarray(val_off).reshape(-1)[0]) == float(np.asarray(val_on).reshape(-1)[0])
