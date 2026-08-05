"""Opening-diversification plumbing (plan P0 S1/S2/S6).

These tests pin the *mechanism* — that ``opening_temp``/``opening_moves`` reach
the players and switch the effective play temperature at the right ply — and the
*plumbing* — that the arena gate (``coach``), the pool tournament
(``tournament_run``) and the parallel worker pool (``parallel.pool``) all forward
the configured values through to :class:`NetworkPlayer`.

Real objects throughout: a real TicTacToe game + a real (tiny, untrained) net +
real MCTS. Only the ``NetworkPlayer``/``Arena`` at the *call site under test* are
swapped for recorders, so we observe the constructor arguments without playing
whole games (the play itself is exercised elsewhere).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from alphablokus.config import MCTSConfig, RunConfig
from alphablokus.evaluation.players import NetworkPlayer
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper

if TYPE_CHECKING:
    from alphablokus.games.tictactoe.game import TicTacToeGame
    from alphablokus.interfaces import IBoard


# ---------------------------------------------------------------------------
# 1. The temperature switch at the opening boundary (real MCTS).
# ---------------------------------------------------------------------------


def test_opening_temp_switches_at_boundary(ttt_game: TicTacToeGame, mcts_config: MCTSConfig) -> None:
    """First ``opening_moves`` plies use ``opening_temp``, then revert to ``temp``.

    Records the effective temperature MCTS is actually asked for on each ply, so
    the assertion is deterministic regardless of which move the sampler draws.
    """
    nnet = NNetWrapper(ttt_game, _minimal_config())
    player = NetworkPlayer(
        ttt_game,
        nnet,
        mcts_config,
        temp=0.0,
        opening_temp=1.0,
        opening_moves=2,
    )

    recorded: list[float] = []
    original = player._mcts.get_action_prob

    def _spy(board: IBoard, temp: float = 1.0, add_root_noise: bool = False) -> list[float]:
        recorded.append(temp)
        return original(board, temp=temp, add_root_noise=add_root_noise)

    player._mcts.get_action_prob = _spy  # type: ignore[method-assign]

    board = ttt_game.initialise_board()
    for _ in range(4):
        player(board)

    # Plies 0,1 (< opening_moves=2) sample at opening_temp; plies 2,3 are greedy.
    assert recorded == [1.0, 1.0, 0.0, 0.0]


def test_startgame_resets_opening_counter(ttt_game: TicTacToeGame, mcts_config: MCTSConfig) -> None:
    """``startGame`` resets the per-game ply counter so the opening reapplies."""
    nnet = NNetWrapper(ttt_game, _minimal_config())
    player = NetworkPlayer(ttt_game, nnet, mcts_config, temp=0.0, opening_temp=1.0, opening_moves=1)

    recorded: list[float] = []

    def _install_spy() -> None:
        # startGame rebuilds the MCTS instance, so re-wrap after each reset.
        original = player._mcts.get_action_prob

        def _spy(board: IBoard, temp: float = 1.0, add_root_noise: bool = False) -> list[float]:
            recorded.append(temp)
            return original(board, temp=temp, add_root_noise=add_root_noise)

        player._mcts.get_action_prob = _spy  # type: ignore[method-assign]

    _install_spy()
    board = ttt_game.initialise_board()
    player(board)  # ply 0 -> opening_temp
    player(board)  # ply 1 -> greedy
    player.startGame()  # counter back to 0 (and rebuilds the MCTS)
    _install_spy()
    player(board)  # ply 0 again -> opening_temp

    assert recorded == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# 2. The three call sites forward the configured values to NetworkPlayer.
# ---------------------------------------------------------------------------


class _RecordingPlayer:
    """Stand-in for ``NetworkPlayer`` that captures its constructor kwargs."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _RecordingPlayer.calls.append(kwargs)

    def __call__(self, board: IBoard) -> int:  # pragma: no cover - never played
        return 0

    def startGame(self) -> None:  # noqa: N802 - Arena's camelCase hook
        pass


class _StubArena:
    """Arena stand-in that plays nothing (returns a fixed draw tally)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def play_games(self, *args: Any, **kwargs: Any) -> tuple[int, int, int, list[Any]]:
        return 0, 0, 2, []

    def play_game(self, *args: Any, **kwargs: Any) -> tuple[int, None]:
        return 0, None


def test_coach_arena_forwards_opening_config(
    ttt_game: TicTacToeGame, test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Coach._run_arena_serial`` builds both arena players with the config's
    ``arena_opening_*`` values (plan S1 — symmetric, gate diversification)."""
    from alphablokus.training import coach as coach_mod

    config = dataclasses.replace(test_config, arena_opening_temp=0.7, arena_opening_moves=3)
    coach = coach_mod.Coach(ttt_game, NNetWrapper(ttt_game, config), config)

    _RecordingPlayer.calls = []
    monkeypatch.setattr(coach_mod, "NetworkPlayer", _RecordingPlayer)
    monkeypatch.setattr(coach_mod, "Arena", _StubArena)

    coach._run_arena_serial(top_k_to_record=0)

    assert len(_RecordingPlayer.calls) == 2  # prev + new player
    for call in _RecordingPlayer.calls:
        assert call["opening_temp"] == pytest.approx(0.7)
        assert call["opening_moves"] == 3


def test_tournament_pairing_forwards_opening_config(
    ttt_game: TicTacToeGame, test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tournament_run._play_pairing`` (serial) builds both checkpoint players
    with ``TournamentConfig.opening_*`` (plan S2)."""
    from pathlib import Path

    from alphablokus.evaluation import tournament_run as tr_mod

    tour = dataclasses.replace(test_config.tournament, opening_temp=0.9, opening_moves=4)
    config = dataclasses.replace(test_config, tournament=tour, num_parallel_workers=1)

    class _StubNet:
        def load_checkpoint(self, *args: Any, **kwargs: Any) -> None:
            pass

    _RecordingPlayer.calls = []
    monkeypatch.setattr(tr_mod, "NetworkPlayer", _RecordingPlayer)
    monkeypatch.setattr(tr_mod, "Arena", _StubArena)
    monkeypatch.setattr(
        tr_mod.registry,
        "instantiate_game_and_network",
        lambda cfg: (ttt_game, _StubNet()),
    )

    tr_mod._play_pairing(config, Path("a.pth.tar"), Path("b.pth.tar"), num_games=2)

    assert len(_RecordingPlayer.calls) == 2
    for call in _RecordingPlayer.calls:
        assert call["opening_temp"] == pytest.approx(0.9)
        assert call["opening_moves"] == 4


def test_pool_worker_forwards_opening_config(
    ttt_game: TicTacToeGame, test_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parallel two-player worker forwards the phase's opening schedule to
    both players (plan S1/S2 — the production path)."""
    from alphablokus.parallel import pool as pool_mod

    config = dataclasses.replace(
        test_config,
        arena_opening_temp=0.7,
        arena_opening_moves=3,
        tournament=dataclasses.replace(test_config.tournament, opening_temp=0.9, opening_moves=4),
    )

    monkeypatch.setattr(pool_mod, "_WORKER_CONFIG", config)
    monkeypatch.setattr(pool_mod, "_WORKER_GAME", ttt_game)
    monkeypatch.setattr(pool_mod, "_WORKER_NNET_A", object())
    monkeypatch.setattr(pool_mod, "_WORKER_NNET_B", object())
    monkeypatch.setattr(pool_mod, "NetworkPlayer", _RecordingPlayer)
    monkeypatch.setattr(pool_mod, "Arena", _StubArena)

    # task = (base_seed, generation, episode_idx, a_first, record, top_k, phase)
    _RecordingPlayer.calls = []
    pool_mod._worker_play_two_player_game((0, 0, 0, True, False, 0, pool_mod.PHASE_ARENA))
    assert len(_RecordingPlayer.calls) == 2
    for call in _RecordingPlayer.calls:
        assert call["opening_temp"] == pytest.approx(0.7)
        assert call["opening_moves"] == 3

    _RecordingPlayer.calls = []
    pool_mod._worker_play_two_player_game((0, 0, 0, True, False, 0, pool_mod.PHASE_ELO))
    assert len(_RecordingPlayer.calls) == 2
    for call in _RecordingPlayer.calls:
        assert call["opening_temp"] == pytest.approx(0.9)
        assert call["opening_moves"] == 4


def test_opening_schedule_defaults_are_inert() -> None:
    """With the defaults (all 0) the schedule is fully deterministic play."""
    from alphablokus.parallel.pool import PHASE_ARENA, PHASE_ELO, PHASE_SELF_PLAY, _opening_schedule_for_phase

    config = _minimal_config()
    assert _opening_schedule_for_phase(config, PHASE_ARENA) == (0.0, 0)
    assert _opening_schedule_for_phase(config, PHASE_ELO) == (0.0, 0)
    # An unrelated phase always maps to deterministic play.
    assert _opening_schedule_for_phase(config, PHASE_SELF_PLAY) == (0.0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> RunConfig:
    from pathlib import Path

    from alphablokus.config import NetConfig

    return RunConfig(
        game="tictactoe",
        run_name="opening_diversity_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=5,
        update_threshold=0.55,
        num_arena_matches=2,
        replay_buffer_games=20,
        root_directory=Path("/tmp/alphablokus-opening-test"),
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0),
        net_config=NetConfig(
            learning_rate=1e-3,
            dropout=0.3,
            epochs=1,
            batch_size=4,
            cuda=False,
            num_filters=8,
            num_residual_blocks=1,
        ),
    )


# ---------------------------------------------------------------------------
# 4. A seed makes a run reproducible — including the temp=0 tie-break.
# ---------------------------------------------------------------------------


def test_seeded_players_replay_the_same_game(ttt_game: TicTacToeGame, mcts_config: MCTSConfig) -> None:
    """Two identically-seeded players must produce identical move sequences.

    Seeding the opening sampler alone was not enough: MCTS breaks a tie on visit
    counts by drawing from numpy's *global* RNG, and with 2 simulations over 9 legal
    moves (or 400 over Blokus's 17,837) top counts tie constantly. A benchmark that
    recorded ``--seed`` while its play depended on process entropy was not
    reproducible, whatever the payload claimed.
    """
    from alphablokus.evaluation.arena import Arena

    nnet = NNetWrapper(ttt_game, _minimal_config())  # one net: a fresh one re-randomises its weights

    def moves_with(seed: int) -> tuple[int, ...]:
        player = NetworkPlayer(ttt_game, nnet, mcts_config, temp=0.0, seed=seed)
        _, record = Arena(player, player, ttt_game).play_game(record=True)
        assert record is not None
        return tuple(move.action for move in record.moves)

    assert moves_with(11) == moves_with(11)


def test_an_unseeded_player_still_uses_the_global_rng(ttt_game: TicTacToeGame, mcts_config: MCTSConfig) -> None:
    """No seed keeps the pre-existing behaviour, so self-play is unchanged."""
    nnet = NNetWrapper(ttt_game, _minimal_config())
    assert NetworkPlayer(ttt_game, nnet, mcts_config)._mcts._rng is None
