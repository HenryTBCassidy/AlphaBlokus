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
        for example in game_examples:
            board, (indices, values), value = example.board, example.policy, example.value
            assert board.shape == (14, 14) and board.dtype == np.int8
            assert indices.dtype == np.int32 and values.dtype == np.float32
            assert np.all(np.diff(indices) > 0), "sparsify stores ascending unique indices"
            assert value in (1.0, -1.0, 1e-4, -1e-4)
            assert example.player in (1, -1)
            dense = densify(indices, values, action_size)
            np.testing.assert_allclose(dense.sum(), 1.0, atol=1e-5)


def test_policies_are_legal_on_their_boards(generated) -> None:
    """Harvest bookkeeping check: each stored policy's support is legal on its
    own stored board, reconstructed from the canonical compact form (inventory
    derived: a piece is unplayed iff absent from the board), masked by the
    parity-proven jax kernels.

    The canonical form does not record which physical colour is to move, and the
    two colours have different first-move start squares (the canonical frame's
    start is (9,9) when the real mover was Black — the engine swaps
    ``initial_actions`` in ``board.canonical``). The stored ``player`` now says
    which it was, so every position is checked against its own unambiguous mask;
    this used to fall back to the union of both interpretations wherever the
    mover still held a full inventory (plan D1).
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
        for example in game_examples[0::2]:  # identity twins
            canonical = example.board.reshape(-1).astype(np.int8)
            # Canonical is the absolute placement board multiplied by the mover, so
            # multiplying back recovers the real board the mask must be built from.
            absolute = (canonical * example.player).astype(np.int8)
            mask = mask_for(absolute, example.player)
            assert mask[example.policy[0]].all(), "policy mass on an illegal action"


def test_transpose_twins_are_consistent(generated) -> None:
    _config_, game, games, _stats = generated
    action_size = game.get_action_size()
    for game_examples in games:
        for row, twin in zip(game_examples[0::2], game_examples[1::2], strict=True):
            board_a, pi_a = row.board, row.policy
            pi_b = twin.policy
            assert twin.value == row.value
            # The twin is the same position from the same side's perspective.
            assert twin.player == row.player
            np.testing.assert_array_equal(twin.board, board_a.T)
            dense_a = densify(*pi_a, action_size)
            dense_b = densify(*pi_b, action_size)
            np.testing.assert_allclose(dense_b, game.transpose_policy(dense_a), atol=0)


def test_values_alternate_with_players(generated) -> None:
    """Consecutive identity positions swap side to move, and the value follows.

    Blokus Duo's jax path never passes mid-game (a blocked side ends its game), so
    the mover strictly alternates; the outcome each position is labelled with must
    flip with it. Previously only the value sign could be checked, since the mover
    was not stored — so a harvester that mislabelled *which* colour a value
    belonged to was invisible.
    """
    _config_, _game, games, _stats = generated
    for game_examples in games:
        identity = list(game_examples[0::2])
        players = [example.player for example in identity]
        assert players[0] == 1  # White opens
        assert players == [1 if index % 2 == 0 else -1 for index in range(len(players))]
        if abs(identity[-1].value) < 0.5:
            continue  # draw — signs follow the end-player convention instead
        for first, second in zip(identity, identity[1:], strict=False):
            assert first.value == -second.value


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
        for example_a, example_b in zip(game_a, game_b, strict=True):
            np.testing.assert_array_equal(example_a.board, example_b.board)
            np.testing.assert_array_equal(example_a.policy[0], example_b.policy[0])
            np.testing.assert_array_equal(example_a.policy[1], example_b.policy[1])
            assert example_a.value == example_b.value
            assert example_a.player == example_b.player
