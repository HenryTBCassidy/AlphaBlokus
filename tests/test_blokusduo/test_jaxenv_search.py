"""G4: the mctx/top-K search behaves like the python MCTS.

Two layers:

1. Structural invariants that must hold exactly (visits only on legal actions,
   distributions normalised, terminal roots resolve to pass, noise stays legal).
2. Agreement with ``alphablokus.search.mcts.MCTS`` on mid-game dev-cache positions using the
   *same* (converted) small random net, no noise, same sims: top-1 move match
   rate and visit-distribution overlap. Bounds are set from the measured
   python-vs-python noise floor (K=1 vs K=16 virtual-loss batching — the same
   algorithm under a different batching approximation), asserted in
   ``test_python_noise_floor_yardstick`` so the yardstick itself is pinned.

The box-scale version of (2) — real trained net, S=400, 200 positions — is
``scripts/validate_jax_search.py`` (results in the plan's G4 notes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_blokusduo.conftest import DEV_CACHE_PATH

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")
pytest.importorskip("mctx")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from alphablokus.config import MCTSConfig, NetConfig, RunConfig  # noqa: E402
from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board  # noqa: E402
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device  # noqa: E402
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels  # noqa: E402
from alphablokus.games.blokusduo.jax.search import SearchConfig, dense_policy, make_search  # noqa: E402
from alphablokus.games.blokusduo.jax.tables import build_jax_tables  # noqa: E402
from alphablokus.search.mcts import MCTS  # noqa: E402

N_POSITIONS = 20
SIMS = 60


def _run_config(tmp_path, num_filters: int = 16, blocks: int = 1) -> RunConfig:
    return RunConfig(
        game="blokusduo", run_name="test_jax_search", num_generations=1, num_eps=1,
        temp_threshold=5, update_threshold=0.55, num_arena_matches=2,
        root_directory=tmp_path, load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=SIMS, cpuct=2.5),
        net_config=NetConfig(
            learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=4, cuda=False,
            num_filters=num_filters, num_residual_blocks=blocks,
        ),
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory, blokus_game_module: BlokusDuoGame):
    """Shared: one small random torch net + its jax conversion + mid-game states."""
    from tests.fixtures.blokus_positions import iter_cached_positions

    torch.manual_seed(3)
    game = blokus_game_module
    game.enable_optimised_movegen()
    from alphablokus.games.blokusduo.neuralnets.wrapper import NNetWrapper

    nnet = NNetWrapper(game, _run_config(tmp_path_factory.mktemp("jax_search")))
    params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
    kernels = make_kernels(build_jax_tables(game))

    boards, players = [], []
    for _, (board, player, seq) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        if 6 <= len(seq) <= 20:  # mid-game: interesting branching, no first-move path
            boards.append(board)
            players.append(player)
        if len(boards) >= N_POSITIONS:
            break
    states = GameState(*(
        np.stack(rows) for rows in zip(
            *(numpy_state_from_board(b, p) for b, p in zip(boards, players, strict=True)), strict=True
        )
    ))
    return game, nnet, params, kernels, boards, players, states


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_structural_invariants(setup) -> None:
    game, _nnet, params, kernels, _boards, players, states = setup
    search = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5))
    result = search(params, jax.random.PRNGKey(0), states)

    weights = np.asarray(result.action_weights)
    ids = np.asarray(result.topk_ids)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert (weights >= 0).all()
    np.testing.assert_allclose(
        np.asarray(result.visit_counts).sum(axis=1), SIMS, atol=0,
        err_msg="root visit counts must sum to num_simulations",
    )

    dense = np.asarray(dense_policy(result.action_weights, result.topk_ids, kernels.action_size))
    masks = np.asarray(kernels.legal_mask_batch(states))
    assert not np.any((dense > 0) & ~masks), "search must never visit an illegal action"
    assert ids.min() >= 0 and ids.max() < kernels.action_size


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_terminal_root_resolves_to_pass(setup) -> None:
    _game, _nnet, params, kernels, *_ = setup
    # A finished game: only pass is legal for either player.
    state = kernels.initial_state()
    empty = jnp.zeros_like(state.remaining)
    state = state._replace(remaining=empty)  # no pieces left -> no placements
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)), state)

    search = make_search(kernels, SearchConfig(num_simulations=8, top_k=16))
    result = search(params, jax.random.PRNGKey(1), batch)
    dense = np.asarray(dense_policy(result.action_weights, result.topk_ids, kernels.action_size))
    assert (dense[:, kernels.pass_index] == 1.0).all()


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_root_noise_stays_legal_and_perturbs(setup) -> None:
    _game, _nnet, params, kernels, _boards, _players, states = setup
    noisy = make_search(
        kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5, dirichlet_epsilon=0.25)
    )
    clean = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5))
    noisy_result = noisy(params, jax.random.PRNGKey(2), states)
    clean_result = clean(params, jax.random.PRNGKey(2), states)

    dense = np.asarray(dense_policy(noisy_result.action_weights, noisy_result.topk_ids, kernels.action_size))
    masks = np.asarray(kernels.legal_mask_batch(states))
    assert not np.any((dense > 0) & ~masks)
    clean_dense = np.asarray(dense_policy(clean_result.action_weights, clean_result.topk_ids, kernels.action_size))
    assert np.abs(dense - clean_dense).max() > 0, "noise should change at least one visit distribution"


def _python_visit_distribution(game, nnet, board, player, sims: int, batch_size: int) -> np.ndarray:
    config = MCTSConfig(num_mcts_sims=sims, cpuct=2.5, mcts_batch_size=batch_size)
    mcts = MCTS(game, nnet, config)
    canonical = game.get_canonical_form(board, player)
    return np.asarray(mcts.get_action_prob(canonical, temp=1))


@pytest.fixture(scope="module")
def python_distributions(setup):
    game, nnet, _params, _kernels, boards, players, _states = setup
    k1 = [_python_visit_distribution(game, nnet, b, p, SIMS, 1) for b, p in zip(boards, players, strict=True)]
    k16 = [_python_visit_distribution(game, nnet, b, p, SIMS, 16) for b, p in zip(boards, players, strict=True)]
    return np.stack(k1), np.stack(k16)


def _top1_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a.argmax(axis=1) == b.argmax(axis=1)))


def _mean_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Σ min(p, q) — 1.0 iff identical distributions."""
    return float(np.minimum(a, b).sum(axis=1).mean())


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_python_noise_floor_yardstick(python_distributions) -> None:
    """Pin the yardstick: K=1 vs K=16 python search on the same positions."""
    k1, k16 = python_distributions
    assert _top1_agreement(k1, k16) >= 0.7
    assert _mean_overlap(k1, k16) >= 0.6


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_agreement_with_python_mcts(setup, python_distributions) -> None:
    """jax-vs-python agreement must reach the python-vs-python noise floor."""
    _game, _nnet, params, kernels, _boards, _players, states = setup
    k1, k16 = python_distributions
    search = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5))
    result = search(params, jax.random.PRNGKey(0), states)
    jax_dense = np.asarray(dense_policy(result.action_weights, result.topk_ids, kernels.action_size))

    floor_top1 = min(_top1_agreement(k1, k16), 0.95)
    floor_overlap = min(_mean_overlap(k1, k16), 0.95)
    top1_vs_k1 = _top1_agreement(jax_dense, k1)
    overlap_vs_k1 = _mean_overlap(jax_dense, k1)
    assert top1_vs_k1 >= floor_top1 - 0.10, (
        f"jax-vs-python top-1 {top1_vs_k1:.2f} below python noise floor {floor_top1:.2f}"
    )
    assert overlap_vs_k1 >= floor_overlap - 0.10, (
        f"jax-vs-python overlap {overlap_vs_k1:.2f} below python noise floor {floor_overlap:.2f}"
    )
