"""Self-play generation: one entry point, three interchangeable backends.

``generate_games`` dispatches on the run config — GPU-native jax
(``selfplay_backend: "jax"``), the process pool (``num_parallel_workers > 1``),
or the in-process serial loop — and returns the identical
one-list-per-game contract from all three, so the Coach never cares which
backend produced the data.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tqdm import tqdm

from alphablokus.search.mcts import MCTS
from alphablokus.selfplay.episode import GameExamples, play_self_play_episode

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame, INeuralNetWrapper
    from alphablokus.search.stats import MCTSEpisodeStats

# Fixed checkpoint filename the pool / jax backends load the current net from.
WORKER_INIT_CHECKPOINT = "parallel_worker_init.pth.tar"

# Per-episode profiling sink: (episode_idx, stats) -> None.
StatsLogger = Callable[[int, "MCTSEpisodeStats"], None]

# The contract a GPU-native backend implements: (config, generation,
# checkpoint_path) -> (one list of positions per game, per-game stats).
SelfPlayBackendFn = Callable[..., "tuple[list[GameExamples], list[MCTSEpisodeStats]]"]


def generate_games(
    config: RunConfig,
    game: IGame,
    nnet: INeuralNetWrapper,
    generation: int,
    log_stats: StatsLogger,
) -> list[GameExamples]:
    """Generate one generation of self-play games with the configured backend.

    Args:
        config: Run configuration; ``selfplay_backend`` and
            ``num_parallel_workers`` select the backend.
        game: Game implementation (rules + action space).
        nnet: Current network; the pool/jax backends load it from a
            checkpoint saved here.
        generation: Current generation number (worker seed derivation).
        log_stats: Called once per completed game with
            ``(episode_idx, MCTSEpisodeStats)`` — the shared profiling
            schema across all backends.

    Returns:
        One list of positions per game, in episode order (game boundaries
        preserved so the games-sized replay buffer can evict whole games).
    """
    if config.selfplay_backend == "jax":
        return _generate_jax(config, nnet, generation, log_stats)
    if config.num_parallel_workers > 1:
        return _generate_parallel(config, nnet, generation, log_stats)
    return _generate_serial(config, game, nnet, log_stats)


def _generate_serial(
    config: RunConfig,
    game: IGame,
    nnet: INeuralNetWrapper,
    log_stats: StatsLogger,
) -> list[GameExamples]:
    """Sequential in-process loop: a fresh MCTS per episode."""
    fresh_games: list[GameExamples] = []
    for episode_idx in tqdm(range(config.num_eps), desc="Self Play"):
        mcts = MCTS(game, nnet, config.mcts_config)
        fresh_games.append(play_self_play_episode(game, mcts, config.temp_threshold))
        log_stats(episode_idx, mcts.get_episode_stats())
    return fresh_games


def _generate_parallel(
    config: RunConfig,
    nnet: INeuralNetWrapper,
    generation: int,
    log_stats: StatsLogger,
) -> list[GameExamples]:
    """Process-pool backend: workers load the net from a fixed checkpoint."""
    from alphablokus.parallel.pool import run_self_play_episodes_parallel

    nnet.save_checkpoint(filename=WORKER_INIT_CHECKPOINT)
    per_ep_examples, per_ep_stats = run_self_play_episodes_parallel(
        config=config,
        generation=generation,
        checkpoint_path=WORKER_INIT_CHECKPOINT,
        num_workers=config.num_parallel_workers,
    )

    fresh_games: list[GameExamples] = []
    for episode_idx, (examples, stats) in enumerate(zip(per_ep_examples, per_ep_stats, strict=False)):
        fresh_games.append(examples)
        log_stats(episode_idx, stats)
    return fresh_games


def _generate_jax(
    config: RunConfig,
    nnet: INeuralNetWrapper,
    generation: int,
    log_stats: StatsLogger,
) -> list[GameExamples]:
    """GPU-native batched backend, resolved per-game through the registry
    (deferred import — python-backend runs never require the ``jax`` extra)."""
    from alphablokus.registry import resolve_jax_selfplay_backend

    generate_self_play_games = resolve_jax_selfplay_backend(config)
    nnet.save_checkpoint(filename=WORKER_INIT_CHECKPOINT)
    per_game_examples, per_game_stats = generate_self_play_games(
        config=config,
        generation=generation,
        checkpoint_path=WORKER_INIT_CHECKPOINT,
    )

    fresh_games: list[GameExamples] = []
    for episode_idx, (examples, stats) in enumerate(zip(per_game_examples, per_game_stats, strict=True)):
        fresh_games.append(examples)
        log_stats(episode_idx, stats)
    return fresh_games
