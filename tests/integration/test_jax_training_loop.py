"""G6: full 1-generation Blokus training loop with ``selfplay_backend: "jax"``.

Self-play runs on the jax backend (tiny batch/sims, CPU); everything
downstream — buffer, torch training step, arena gating, Elo, storage,
metrics — is the untouched python machinery. Completing without exceptions
and producing the same artifacts as the python path is the integration
contract of the backend seam.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("mctx")

from typing import TYPE_CHECKING

from alphablokus.config import JaxSelfPlayConfig, MCTSConfig, NetConfig, RunConfig  # noqa: E402
from alphablokus.games.blokusduo.game import BlokusDuoGame  # noqa: E402
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper  # noqa: E402
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.training.coach import Coach  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
def test_one_generation_blokus_jax_backend(tmp_path: Path) -> None:
    config = RunConfig(
        game="blokusduo",
        run_name="test_jax_loop",
        num_generations=1,
        num_eps=2,
        temp_threshold=12,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=tmp_path,
        load_model=False,
        mcts_config=MCTSConfig(
            num_mcts_sims=8,
            cpuct=2.5,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.03,
        ),
        net_config=NetConfig(
            learning_rate=1e-3,
            dropout=0.0,
            epochs=1,
            batch_size=8,
            cuda=False,
            num_filters=16,
            num_residual_blocks=1,
        ),
        selfplay_backend="jax",
        jax_selfplay=JaxSelfPlayConfig(batch_size=2, top_k=32, dtype="float32", wave_plies=16),
        minimax_games_per_gen=0,
        symmetry_diagnostic_positions=0,
        use_optimised_movegen=True,
        seed=13,
    )
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    coach = Coach(game, NNetWrapper(game, config), config)
    coach.learn()

    assert any(config.training_data_directory.rglob("*.parquet")), "training parquet missing"
    assert any(config.self_play_history_directory.rglob("*.parquet")), "self-play parquet missing"
    assert list(config.net_directory.glob("*.pth.tar")), "no checkpoints written"
