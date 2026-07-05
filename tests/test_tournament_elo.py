"""End-to-end test of the pool tournament tool (``scripts/tournament_elo.py``).

Real TicTacToe nets, real arena games, no mocks — just tiny (2 sims, 2 games per
pairing) so it runs fast. Exercises the whole path: enumerate saved checkpoints →
build pairings → play → fit BayesElo → write parquet + JSON.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from alphablokus.config import MCTSConfig, NetConfig, RunConfig, TournamentConfig
from alphablokus.registry import instantiate_game_and_network
from scripts.tournament_elo import run_tournament

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        game="tictactoe",
        run_name="tournament_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=5,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=tmp_path,
        load_model=False,
        num_parallel_workers=1,  # serial arena path
        mcts_config=MCTSConfig(num_mcts_sims=2, cpuct=1.0),
        net_config=NetConfig(
            learning_rate=1e-3,
            dropout=0.0,
            epochs=1,
            batch_size=4,
            cuda=False,
            num_filters=8,
            num_residual_blocks=1,
        ),
        tournament=TournamentConfig(
            games_per_pairing=2,
            back_ref_offsets=(1,),
            include_first_last=True,
        ),
    )


def _save_checkpoints(config: RunConfig, filenames: list[str]) -> None:
    """Save a distinct random-init net under each filename in net_directory."""
    for filename in filenames:
        _, nnet = instantiate_game_and_network(config)
        nnet.save_checkpoint(filename=filename)


def test_run_tournament_produces_ratings_and_files(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    _save_checkpoints(
        config,
        ["elo_baseline.pth.tar", "accepted_1.pth.tar", "accepted_2.pth.tar", "accepted_3.pth.tar"],
    )

    result = run_tournament(config)

    assert result is not None
    # Every checkpoint (gen0 anchor + 3 accepted) gets a finite rating.
    assert set(result.ratings) == {"gen0", "gen1", "gen2", "gen3"}
    assert all(pd.notna(v) and abs(v) < 1e6 for v in result.ratings.values())
    # Anchor pinned at the default rating (0.0).
    assert result.ratings["gen0"] == pytest.approx(0.0, abs=1e-9)

    # Ratings parquet exists with a row per checkpoint and the expected columns.
    ratings_path = config.tournament_directory / "tournament_ratings.parquet"
    assert ratings_path.exists()
    df = pd.read_parquet(ratings_path)
    assert set(df.columns) == {"generation", "rating", "n_games", "n_pairings"}
    assert sorted(df["generation"].tolist()) == [0, 1, 2, 3]
    assert (df["n_games"] > 0).all()

    # Raw JSON round-trips and covers the pool.
    raw_path = config.tournament_directory / "tournament_raw.json"
    assert raw_path.exists()
    raw = json.loads(raw_path.read_text())
    assert raw["players"] == ["gen0", "gen1", "gen2", "gen3"]
    assert raw["converged"] is True


def test_run_tournament_dry_run_writes_nothing(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    _save_checkpoints(config, ["elo_baseline.pth.tar", "accepted_1.pth.tar"])

    result = run_tournament(config, dry_run=True)

    assert result is None
    assert not config.tournament_directory.exists()


def test_run_tournament_needs_two_checkpoints(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    _save_checkpoints(config, ["elo_baseline.pth.tar"])

    assert run_tournament(config) is None
