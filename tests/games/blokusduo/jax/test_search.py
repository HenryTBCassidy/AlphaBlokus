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

from tests.games.blokusduo.conftest import DEV_CACHE_PATH
from tests.games.blokusduo.jax.conftest import PARITY_DTYPES, make_search_config

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

jax = pytest.importorskip("jax")
pytest.importorskip("mctx")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from alphablokus.config import MCTSConfig  # noqa: E402
from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board  # noqa: E402
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device  # noqa: E402
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels  # noqa: E402
from alphablokus.games.blokusduo.jax.search import SearchConfig, dense_policy, make_search  # noqa: E402
from alphablokus.games.blokusduo.jax.tables import build_jax_tables  # noqa: E402
from alphablokus.search.mcts import MCTS  # noqa: E402

N_POSITIONS = 20
SIMS = 60


@pytest.fixture(scope="module")
def setup(tmp_path_factory, blokus_game_module: BlokusDuoGame):
    """Shared: one small random torch net + its jax conversion + mid-game states."""
    from alphablokus.testing.positions import iter_cached_positions

    torch.manual_seed(3)
    game = blokus_game_module
    game.enable_optimised_movegen()
    from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper

    nnet = NNetWrapper(game, make_search_config(tmp_path_factory.mktemp("jax_search")))
    params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
    kernels = make_kernels(build_jax_tables(game))

    boards, players = [], []
    for _, (board, player, seq) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        if 6 <= len(seq) <= 20:  # mid-game: interesting branching, no first-move path
            boards.append(board)
            players.append(player)
        if len(boards) >= N_POSITIONS:
            break
    states = GameState(
        *(
            np.stack(rows)
            for rows in zip(*(numpy_state_from_board(b, p) for b, p in zip(boards, players, strict=True)), strict=True)
        )
    )
    return game, nnet, params, kernels, boards, players, states


@pytest.fixture(scope="module")
def params_by_dtype(setup) -> dict[str, object]:
    """The same net's parameters converted to each parity dtype.

    Both halves of the dtype choice have to move together: ``SearchConfig.dtype``
    sets the *input encoding* dtype while the *parameter* dtype comes from
    ``params_to_device``. Setting only one raises a dtype mismatch inside the
    conv — production sets both from ``jax_selfplay.dtype`` (``jax/backend.py``),
    so the tests must too.
    """
    _game, nnet, _params, _kernels, *_ = setup
    state_dict = nnet.nnet.state_dict()
    return {
        dtype: params_to_device(convert_state_dict(state_dict, num_residual_blocks=1), dtype=dtype)
        for dtype in PARITY_DTYPES
    }


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
@pytest.mark.parametrize("dtype", PARITY_DTYPES)
def test_structural_invariants(setup, params_by_dtype, dtype: str) -> None:
    """Structural invariants must hold in the dtype production actually runs."""
    game, _nnet, _params, kernels, _boards, players, states = setup
    params = params_by_dtype[dtype]
    search = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5, dtype=dtype))
    result = search(params, jax.random.PRNGKey(0), states)

    weights = np.asarray(result.action_weights)
    ids = np.asarray(result.topk_ids)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert (weights >= 0).all()
    np.testing.assert_allclose(
        np.asarray(result.visit_counts).sum(axis=1),
        SIMS,
        atol=0,
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
    noisy = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5, dirichlet_epsilon=0.25))
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
@pytest.mark.parametrize("dtype", PARITY_DTYPES)
def test_agreement_with_python_mcts(setup, python_distributions, params_by_dtype, dtype: str) -> None:
    """jax-vs-python agreement must reach the python-vs-python noise floor.

    Run in both dtypes. The python oracle is fp32 in both cases — it is the
    reference — so the bf16 leg measures "does reduced-precision search still
    agree with the tested path", which is the question production depends on and
    which nothing checked before. bf16 gets extra slack because rounding
    legitimately reorders near-tied logits in the top-k window.
    """
    _game, _nnet, _params, kernels, _boards, _players, states = setup
    k1, k16 = python_distributions
    params = params_by_dtype[dtype]
    search = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=128, cpuct=2.5, dtype=dtype))
    result = search(params, jax.random.PRNGKey(0), states)
    jax_dense = np.asarray(dense_policy(result.action_weights, result.topk_ids, kernels.action_size))

    slack = 0.10 if dtype == "float32" else 0.20
    floor_top1 = min(_top1_agreement(k1, k16), 0.95)
    floor_overlap = min(_mean_overlap(k1, k16), 0.95)
    top1_vs_k1 = _top1_agreement(jax_dense, k1)
    overlap_vs_k1 = _mean_overlap(jax_dense, k1)
    assert top1_vs_k1 >= floor_top1 - slack, (
        f"{dtype} jax-vs-python top-1 {top1_vs_k1:.2f} below python noise floor {floor_top1:.2f}"
    )
    assert overlap_vs_k1 >= floor_overlap - slack, (
        f"{dtype} jax-vs-python overlap {overlap_vs_k1:.2f} below python noise floor {floor_overlap:.2f}"
    )


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_bf16_top_k_selection_agrees_with_fp32(setup, params_by_dtype) -> None:
    """The specific bf16 risk: which actions survive the top-k truncation.

    The jax search truncates to the prior's top-``k`` of 17,837 actions at the
    root *and* at every child (``jax/search.py`` ``topk_legal``), and
    ``lax.top_k`` breaks ties by lowest index. Under bf16 the logits are rounded
    before that comparison, so a near-tie can reshuffle which actions are
    searchable at all — and an action outside the window can never be searched,
    never enter a training target, and never be corrected.

    This pins how much churn there is. Measured here: **97.6%** of the top-64
    window is shared between fp32 and bf16 on this net, i.e. bf16 changes which
    actions are searchable for roughly 1 action in 40.

    Two caveats on reading that number. It is a small random net, whose 17,837
    logits are near-uniform — the worst case for tie reordering, so a trained net
    with a concentrated prior should churn less. And it is *below* the ">99.5%
    selection agreement" figure the plan proposes as the threshold for closing the
    bf16 question, so the question should be settled on a trained net at
    production width rather than on this test.

    The bound below is deliberately loose: its job is to fail if the churn ever
    becomes large, not to re-assert the measurement.
    """
    _game, _nnet, _params, kernels, _boards, _players, states = setup
    fp32 = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=64, cpuct=2.5, dtype="float32"))
    bf16 = make_search(kernels, SearchConfig(num_simulations=SIMS, top_k=64, cpuct=2.5, dtype="bfloat16"))

    fp32_ids = np.asarray(fp32(params_by_dtype["float32"], jax.random.PRNGKey(0), states).topk_ids)
    bf16_ids = np.asarray(bf16(params_by_dtype["bfloat16"], jax.random.PRNGKey(0), states).topk_ids)

    overlaps = [
        len(set(fp32_row.tolist()) & set(bf16_row.tolist())) / len(fp32_row)
        for fp32_row, bf16_row in zip(fp32_ids, bf16_ids, strict=True)
    ]
    mean_overlap = float(np.mean(overlaps))
    assert mean_overlap >= 0.90, (
        f"bf16 top-{fp32_ids.shape[1]} window shares only {mean_overlap:.1%} of its actions with fp32 — "
        "reduced precision is materially changing which actions are searchable"
    )
