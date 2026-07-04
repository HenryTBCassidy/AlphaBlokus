"""Self-play generation: one entry point, three interchangeable backends.

``generate_games`` dispatches on the run config — GPU-native jax
(``selfplay_backend: "jax"``), the process pool (``num_parallel_workers > 1``),
or the in-process serial loop — and **streams** each completed game to the
caller's ``sink`` from all three, so a whole generation is never accumulated
here (``docs/plans/oom-hardening.md`` O6). The Coach never cares which backend
produced the data.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import count
from typing import TYPE_CHECKING

from tqdm import tqdm

from alphablokus.search.mcts import MCTS
from alphablokus.selfplay.episode import GameExamples, play_self_play_episode

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame, INeuralNetWrapper
    from alphablokus.search.stats import MCTSEpisodeStats
    from alphablokus.selfplay.episode import ProcessedExample

# Fixed checkpoint filename the pool / jax backends load the current net from.
WORKER_INIT_CHECKPOINT = "parallel_worker_init.pth.tar"

# Per-episode profiling sink: (episode_idx, stats) -> None.
StatsLogger = Callable[[int, "MCTSEpisodeStats"], None]

# Per-game training-data sink: called once with each completed game's examples,
# in the order games are handed over (episode order for the serial and pool
# backends, completion order for jax).
GameSink = Callable[[GameExamples], None]

# The contract a GPU-native backend implements: (config, generation,
# checkpoint_path, sink) -> (one list of positions per game — empty when a
# sink consumed them — and per-game stats).
SelfPlayBackendFn = Callable[..., "tuple[list[GameExamples], list[MCTSEpisodeStats]]"]


def generate_games(
    config: RunConfig,
    game: IGame,
    nnet: INeuralNetWrapper,
    generation: int,
    log_stats: StatsLogger,
    sink: GameSink,
) -> None:
    """Generate one generation of self-play games with the configured backend.

    Every backend hands each completed game to ``sink`` as it finishes rather
    than returning the whole generation at once — the multi-GB per-generation
    accumulation used to coexist with the replay buffer it was about to enter.

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
        sink: Called once per completed game with its examples, in episode
            order (game boundaries preserved so the games-sized replay buffer
            can evict whole games).
    """
    if config.selfplay_backend == "jax":
        _generate_jax(config, nnet, generation, log_stats, sink)
    elif config.num_parallel_workers > 1:
        _generate_parallel(config, nnet, generation, log_stats, sink)
    else:
        _generate_serial(config, game, nnet, log_stats, sink)


def _generate_serial(
    config: RunConfig,
    game: IGame,
    nnet: INeuralNetWrapper,
    log_stats: StatsLogger,
    sink: GameSink,
) -> None:
    """Sequential in-process loop: a fresh MCTS per episode."""
    for episode_idx in tqdm(range(config.num_eps), desc="Self Play"):
        mcts = MCTS(game, nnet, config.mcts_config)
        sink(play_self_play_episode(game, mcts, config.temp_threshold))
        log_stats(episode_idx, mcts.get_episode_stats())


def _generate_parallel(
    config: RunConfig,
    nnet: INeuralNetWrapper,
    generation: int,
    log_stats: StatsLogger,
    sink: GameSink,
) -> None:
    """Process-pool backend: workers load the net from a fixed checkpoint.

    Streams via the pool orchestrator's per-result sink — ``pool.map`` yields
    results in submission order, so episode indices line up exactly as the
    old accumulate-then-relist path did.
    """
    from alphablokus.parallel.pool import run_self_play_episodes_parallel

    nnet.save_checkpoint(filename=WORKER_INIT_CHECKPOINT)

    episode_counter = count()

    def _consume(examples: list[ProcessedExample], stats: MCTSEpisodeStats) -> None:
        sink(examples)
        log_stats(next(episode_counter), stats)

    run_self_play_episodes_parallel(
        config=config,
        generation=generation,
        checkpoint_path=WORKER_INIT_CHECKPOINT,
        num_workers=config.num_parallel_workers,
        sink=_consume,
    )


def _generate_jax(
    config: RunConfig,
    nnet: INeuralNetWrapper,
    generation: int,
    log_stats: StatsLogger,
    sink: GameSink,
) -> None:
    """GPU-native batched backend, resolved per-game through the registry
    (deferred import — python-backend runs never require the ``jax`` extra).

    Examples stream to ``sink`` per completed game during the wave loop; the
    (tiny) per-game stats are returned at the end because their timing
    apportionment needs the whole generation's wall clock.
    """
    from alphablokus.registry import resolve_jax_selfplay_backend

    generate_self_play_games = resolve_jax_selfplay_backend(config)
    nnet.save_checkpoint(filename=WORKER_INIT_CHECKPOINT)
    _empty, per_game_stats = generate_self_play_games(
        config=config,
        generation=generation,
        checkpoint_path=WORKER_INIT_CHECKPOINT,
        sink=sink,
    )

    for episode_idx, stats in enumerate(per_game_stats):
        log_stats(episode_idx, stats)
