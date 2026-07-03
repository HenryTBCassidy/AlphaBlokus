"""Shared config builders for the jax backend/search test suites."""
from __future__ import annotations

from typing import TYPE_CHECKING

from alphablokus.config import JaxSelfPlayConfig, MCTSConfig, NetConfig, RunConfig

if TYPE_CHECKING:
    from pathlib import Path


def make_search_config(tmp_path, num_filters: int = 16, blocks: int = 1, num_sims: int = 60) -> RunConfig:
    return RunConfig(
        game="blokusduo", run_name="test_jax_search", num_generations=1, num_eps=1,
        temp_threshold=5, update_threshold=0.55, num_arena_matches=2,
        root_directory=tmp_path, load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=num_sims, cpuct=2.5),
        net_config=NetConfig(
            learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=4, cuda=False,
            num_filters=num_filters, num_residual_blocks=blocks,
        ),
    )



def make_backend_config(tmp_path: Path, num_eps: int = 3, num_sims: int = 8) -> RunConfig:
    return RunConfig(
        game="blokusduo", run_name="test_jaxplay", num_generations=1, num_eps=num_eps,
        temp_threshold=12, update_threshold=0.55, num_arena_matches=2,
        root_directory=tmp_path, load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=num_sims, cpuct=2.5, dirichlet_epsilon=0.25, dirichlet_alpha=0.03),
        net_config=NetConfig(
            learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=8, cuda=False,
            num_filters=16, num_residual_blocks=1,
        ),
        selfplay_backend="jax",
        jax_selfplay=JaxSelfPlayConfig(batch_size=2, top_k=32, dtype="float32", wave_plies=16),
        seed=7,
    )


