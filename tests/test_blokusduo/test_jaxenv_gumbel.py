"""G10: Gumbel search mode — structural invariants and backend integration.

Gumbel deliberately changes the policy-improvement operator (completed-Q
targets, Sequential Halving action choice), so there is no python oracle to
agree with; these tests pin what must hold regardless: legal-only support,
normalised targets, legal chosen actions, and a working end-to-end backend
generation at tiny scale.
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

from alphablokus.games.blokusduo.jax.bridge import numpy_state_from_board  # noqa: E402
from alphablokus.games.blokusduo.jax.checkpoint import convert_state_dict, params_to_device  # noqa: E402
from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels  # noqa: E402
from alphablokus.games.blokusduo.jax.search import SearchConfig, dense_policy, make_search  # noqa: E402
from alphablokus.games.blokusduo.jax.tables import build_jax_tables  # noqa: E402
from alphablokus.games.blokusduo.pieces import default_pieces_path  # noqa: E402
from tests.test_blokusduo.test_jaxenv_search import _run_config  # noqa: E402

N_POSITIONS = 12
SIMS = 32


@pytest.fixture(scope="module")
def setup(tmp_path_factory, blokus_game_module: BlokusDuoGame):
    from tests.fixtures.blokus_positions import iter_cached_positions

    torch.manual_seed(5)
    game = blokus_game_module
    from alphablokus.games.blokusduo.neuralnets.wrapper import NNetWrapper

    nnet = NNetWrapper(game, _run_config(tmp_path_factory.mktemp("gumbel")))
    params = params_to_device(convert_state_dict(nnet.nnet.state_dict(), num_residual_blocks=1))
    kernels = make_kernels(build_jax_tables(game))

    boards, players = [], []
    for _, (board, player, seq) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        if 6 <= len(seq) <= 20:
            boards.append(board)
            players.append(player)
        if len(boards) >= N_POSITIONS:
            break
    states = GameState(*(
        np.stack(rows) for rows in zip(
            *(numpy_state_from_board(b, p) for b, p in zip(boards, players, strict=True)), strict=True
        )
    ))
    return params, kernels, states


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_gumbel_structural_invariants(setup) -> None:
    params, kernels, states = setup
    search = make_search(kernels, SearchConfig(
        num_simulations=SIMS, top_k=64, policy="gumbel", gumbel_max_considered=16,
    ))
    result = search(params, jax.random.PRNGKey(0), states)

    weights = np.asarray(result.action_weights)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-4)
    assert (weights >= 0).all()

    dense = np.asarray(dense_policy(result.action_weights, result.topk_ids, kernels.action_size))
    masks = np.asarray(kernels.legal_mask_batch(states))
    assert not np.any((dense > 1e-8) & ~masks), "gumbel target mass on an illegal action"

    chosen = np.asarray(result.chosen_global)
    batch = np.arange(len(chosen))
    assert masks[batch, chosen].all(), "gumbel chose an illegal action"


@pytest.mark.skipif(not DEV_CACHE_PATH.exists(), reason="dev_5000 cache not built")
def test_gumbel_backend_generates_games(tmp_path) -> None:
    import dataclasses

    from alphablokus.core.config import JaxSelfPlayConfig, MCTSConfig
    from alphablokus.games.blokusduo.jax.backend import generate_self_play_games
    from alphablokus.games.blokusduo.neuralnets.wrapper import NNetWrapper
    from tests.test_core.test_jaxplay_backend import _config

    torch.manual_seed(6)
    config = _config(tmp_path)
    config = dataclasses.replace(
        config,
        mcts_config=MCTSConfig(
            num_mcts_sims=16, cpuct=2.5, search_policy="gumbel", gumbel_max_considered=8,
        ),
        jax_selfplay=JaxSelfPlayConfig(batch_size=2, top_k=32, dtype="float32", wave_plies=16),
        num_eps=2,
    )

    from alphablokus.games.blokusduo.game import BlokusDuoGame

    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    NNetWrapper(game, config).save_checkpoint(filename="init.pth.tar")
    games, stats = generate_self_play_games(config, generation=1, checkpoint_path="init.pth.tar")
    assert len(games) == 2
    assert all(len(g) >= 2 and len(g) % 2 == 0 for g in games)


def test_gumbel_python_backend_rejected(tmp_path) -> None:
    import dataclasses

    from alphablokus.core.coach import Coach
    from alphablokus.core.config import MCTSConfig
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.neuralnets.wrapper import NNetWrapper
    from tests.test_core.test_jaxplay_backend import _config

    config = dataclasses.replace(
        _config(tmp_path),
        selfplay_backend="python",
        mcts_config=MCTSConfig(num_mcts_sims=8, cpuct=2.5, search_policy="gumbel"),
    )
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    with pytest.raises(ValueError, match="gumbel"):
        Coach(game, NNetWrapper(game, config), config)
