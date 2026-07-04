"""G5/G6: the jax self-play backend produces Coach-compatible games.

End-to-end at tiny scale on CPU (small net, 8 sims, 2 game slots): the
backend must return exactly ``num_eps`` games whose examples are
bit-compatible with what ``play_self_play_episode`` stores — canonical compact
int8 boards, sparse float32 policies over legal actions summing to 1,
outcome-backfilled values with the draw-sign convention, transpose-augmented
pairs — plus schema-compatible per-game stats, deterministically at a fixed
seed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mctx")
pytest.importorskip("torch")

from alphablokus.games.blokusduo.game import BlokusDuoGame  # noqa: E402
from alphablokus.games.blokusduo.jax.backend import generate_self_play_games  # noqa: E402
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.search.stats import MCTSEpisodeStats  # noqa: E402
from alphablokus.storage.sparse_policy import densify  # noqa: E402
from tests.games.blokusduo.jax.conftest import make_backend_config  # noqa: E402

NUM_EPS = 3
SIMS = 8


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One backend invocation shared by all assertions (it's the slow part)."""

    import torch

    from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper

    torch.manual_seed(11)
    tmp_path = tmp_path_factory.mktemp("jaxplay")
    config = make_backend_config(tmp_path)
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    nnet = NNetWrapper(game, config)
    nnet.save_checkpoint(filename="init.pth.tar")

    games, stats = generate_self_play_games(config, generation=1, checkpoint_path="init.pth.tar")
    return config, game, games, stats


def test_returns_exactly_num_eps_games(generated) -> None:
    _config_, _game, games, stats = generated
    assert len(games) == NUM_EPS
    assert len(stats) == NUM_EPS
    assert all(len(game_examples) >= 2 for game_examples in games)


def test_example_format_matches_python_path(generated) -> None:
    config, game, games, _stats = generated
    action_size = game.get_action_size()
    for game_examples in games:
        assert len(game_examples) % 2 == 0, "transpose augmentation must double examples"
        for board, (indices, values), value in game_examples:
            assert board.shape == (14, 14) and board.dtype == np.int8
            assert indices.dtype == np.int32 and values.dtype == np.float32
            assert np.all(np.diff(indices) > 0), "sparsify stores ascending unique indices"
            assert value in (1.0, -1.0, 1e-4, -1e-4)
            dense = densify(indices, values, action_size)
            np.testing.assert_allclose(dense.sum(), 1.0, atol=1e-5)


def test_policies_are_legal_on_their_boards(generated) -> None:
    """Harvest bookkeeping check: each stored policy's support is legal on its
    own stored board, reconstructed from the canonical compact form (inventory
    derived: a piece is unplayed iff absent from the board), masked by the
    parity-proven jax kernels.

    The canonical form deliberately does not record which physical colour is
    to move, and the two colours have different first-move start squares (the
    canonical frame's start is (9,9) when the real mover was Black — the
    engine swaps ``initial_actions`` in ``board.canonical``). So positions
    where the mover has a full inventory are checked against the union of the
    two mover interpretations; everywhere else the mask is unambiguous.
    """
    import jax.numpy as jnp

    from alphablokus.games.blokusduo.jax.kernels import GameState, make_kernels
    from alphablokus.games.blokusduo.jax.tables import build_jax_tables

    _config_, game, games, _stats = generated
    kernels = make_kernels(build_jax_tables(game))

    def mask_for(ppb: np.ndarray, mover: int) -> np.ndarray:
        remaining = np.zeros((2, 22), dtype=np.bool_)
        for piece_id in range(1, 22):
            remaining[0, piece_id] = not np.any(ppb == piece_id)
            remaining[1, piece_id] = not np.any(ppb == -piece_id)
        state = GameState(
            ppb=jnp.asarray(ppb),
            remaining=jnp.asarray(remaining),
            last_piece=jnp.zeros(2, dtype=jnp.int8),
            current_player=jnp.int8(mover),
        )
        return np.asarray(kernels.legal_mask(state))

    for game_examples in games:
        for board_compact, (indices, _values), _value in game_examples[0::2]:  # identity twins
            canonical = board_compact.reshape(-1).astype(np.int8)
            mask = mask_for(canonical, 1)
            if not np.any(canonical > 0):  # mover's first move: colour ambiguous
                mask = mask | mask_for(-canonical, -1)
            assert mask[indices].all(), "policy mass on an illegal action"


def test_transpose_twins_are_consistent(generated) -> None:
    _config_, game, games, _stats = generated
    action_size = game.get_action_size()
    for game_examples in games:
        for (board_a, pi_a, value_a), (board_b, pi_b, value_b) in zip(
            game_examples[0::2], game_examples[1::2], strict=True
        ):
            assert value_a == value_b
            np.testing.assert_array_equal(board_b, board_a.T)
            dense_a = densify(*pi_a, action_size)
            dense_b = densify(*pi_b, action_size)
            np.testing.assert_allclose(dense_b, game.transpose_policy(dense_a), atol=0)


def test_values_alternate_with_players(generated) -> None:
    """Within a game (non-draw), consecutive identity positions alternate sign."""
    _config_, _game, games, _stats = generated
    for game_examples in games:
        identity_values = [value for _b, _p, value in game_examples[0::2]]
        if abs(identity_values[-1]) < 0.5:
            continue  # draw — signs follow the end-player convention instead
        for first, second in zip(identity_values, identity_values[1:], strict=False):
            assert first == -second


def test_stats_schema(generated) -> None:
    config, _game, _games, stats = generated
    for entry in stats:
        assert isinstance(entry, MCTSEpisodeStats)
        assert entry.num_moves >= 2
        assert entry.total_sims == entry.num_moves * SIMS
        assert entry.total_search_time_s > 0
        assert entry.mean_policy_entropy >= 0


def test_deterministic_at_fixed_seed(generated, tmp_path) -> None:
    """Same seed + same checkpoint => identical games (jax PRNG is stable)."""
    config, _game, games, _stats = generated
    games_again, _ = generate_self_play_games(config, generation=1, checkpoint_path="init.pth.tar")
    assert len(games_again) == len(games)
    for game_a, game_b in zip(games, games_again, strict=True):
        assert len(game_a) == len(game_b)
        for (board_a, (idx_a, val_a), value_a), (board_b, (idx_b, val_b), value_b) in zip(game_a, game_b, strict=True):
            np.testing.assert_array_equal(board_a, board_b)
            np.testing.assert_array_equal(idx_a, idx_b)
            np.testing.assert_array_equal(val_a, val_b)
            assert value_a == value_b
