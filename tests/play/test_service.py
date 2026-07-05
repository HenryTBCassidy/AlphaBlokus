"""PlayService behaviour: history replay, legality, both search paths.

Uses TicTacToe (fast, exercises the game-agnostic path end to end with a
real net); the Blokus path shares every line of the service. The FastAPI
layer is a thin serialisation shim over this service and is exercised by the
web tier's agreement script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphablokus.config import MCTSConfig, NetConfig, RunConfig
from alphablokus.play.service import SERVER_DIFFICULTIES, PlayService
from alphablokus.registry import instantiate_game_and_network


@pytest.fixture(scope="module")
def service() -> PlayService:
    config = RunConfig(
        game="tictactoe",
        run_name="play_service_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=1,
        update_threshold=0.55,
        num_arena_matches=1,
        root_directory=Path("./temp"),
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=8, cpuct=2.5, profiling_level="none"),
        net_config=NetConfig(
            learning_rate=0.001,
            dropout=0.3,
            epochs=1,
            batch_size=8,
            cuda=False,
            num_filters=8,
            num_residual_blocks=1,
        ),
        elo_games_per_gen=0,
        minimax_games_per_gen=0,
        symmetry_diagnostic_positions=0,
        seed=42,
    )
    game, nnet = instantiate_game_and_network(config)
    return PlayService(game, nnet)


def test_legal_actions_from_empty_history(service: PlayService) -> None:
    legal = service.legal_actions([])
    assert legal == sorted(legal), "legal ids should be ascending"
    assert len(legal) == 9, "TTT opening has 9 placements"


def test_replay_advances_the_player(service: PlayService) -> None:
    legal = service.legal_actions([])
    after = service.legal_actions([legal[0]])
    assert legal[0] not in after, "occupied square must leave the legal set"
    assert len(after) == 8


@pytest.mark.parametrize("difficulty_id", ["level-1", "level-2"])
def test_best_move_is_legal(service: PlayService, difficulty_id: str) -> None:
    result = service.best_move([], difficulty_id)
    legal = service.legal_actions([])
    assert result.action in legal
    assert result.legal == legal
    assert -1.0 <= result.value <= 1.0
    assert result.elapsed_ms > 0


def test_unknown_difficulty_raises(service: PlayService) -> None:
    with pytest.raises(KeyError):
        service.best_move([], "level-99")


def test_difficulty_table_is_exposed(service: PlayService) -> None:
    assert service.difficulties == SERVER_DIFFICULTIES
    assert [level.sims for level in SERVER_DIFFICULTIES] == sorted(level.sims for level in SERVER_DIFFICULTIES)
