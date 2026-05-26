"""Process-pool worker for parallel self-play (F1).

Public surface:

- :func:`run_self_play_episodes_parallel` — orchestrator called by
  :class:`core.coach.Coach`. Takes a config + checkpoint path + episode
  count, spawns N workers, runs the episodes, returns
  ``(training_examples, per_episode_mcts_stats)``.
- :data:`derive_episode_seed` — pure function that maps
  ``(base_seed, generation, episode_idx)`` to a per-episode seed so the
  same triple always plays the same game regardless of which worker
  picks it up.

Each worker process:

1. Runs :func:`_worker_init` once at pool start, loading the game +
   checkpoint into module-level globals so per-task setup is cheap.
2. Runs :func:`_worker_play_episode` per task, which reseeds RNGs,
   builds a fresh MCTS, plays one episode via
   :func:`core.self_play.play_self_play_episode`, and returns examples
   plus episode-level MCTS stats.

The orchestrator uses ``ProcessPoolExecutor.map`` with chunksize=1 so
the main process gets results as they complete (for tqdm) and so a slow
game doesn't block other workers' return paths.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from core.game_factory import instantiate_game_and_network
from core.mcts import MCTS

if TYPE_CHECKING:
    from core.config import RunConfig
    from core.interfaces import IGame, INeuralNetWrapper
    from core.mcts import MCTSEpisodeStats
    from core.self_play import ProcessedExample


# Module-level state populated inside each worker process by
# :func:`_worker_init`. Lives at module level (not in a class) because
# ``multiprocessing`` initializers populate per-process globals — the
# alternative (passing the game + nnet via every task argument) would
# re-pickle them per task, which is exactly what we're trying to avoid.
_WORKER_CONFIG: RunConfig | None = None
_WORKER_GAME: IGame | None = None
_WORKER_NNET: INeuralNetWrapper | None = None


def derive_episode_seed(base_seed: int, generation: int, episode_idx: int) -> int:
    """Stable mapping from ``(base, gen, ep)`` to a 32-bit RNG seed.

    The same triple always returns the same seed; different triples
    return different seeds. Two callers with the same base seed + gen +
    episode index will run the exact same self-play game whether they
    execute it in the main process or in a worker, in any order.

    Uses ``numpy``'s ``SeedSequence`` rather than a hand-rolled hash so
    we inherit numpy's documented good statistical properties — for the
    1000s-of-episodes scale we operate at, collisions are vanishingly
    unlikely.
    """
    seq = np.random.SeedSequence([base_seed, generation, episode_idx])
    # ``generate_state`` returns ``uint32`` — coerce to plain int because
    # ``torch.manual_seed`` rejects numpy scalar types.
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _worker_init(config: RunConfig, checkpoint_path: str | None) -> None:
    """Initialiser run once when a worker process starts.

    Builds the per-process game + nnet (and loads weights from
    ``checkpoint_path`` if provided). Stashes both in module-level
    globals so every per-task call reuses them without re-construction.
    """
    global _WORKER_CONFIG, _WORKER_GAME, _WORKER_NNET
    _WORKER_CONFIG = config
    _WORKER_GAME, _WORKER_NNET = instantiate_game_and_network(config)
    if checkpoint_path is not None:
        _WORKER_NNET.load_checkpoint(filename=checkpoint_path)


def _worker_play_episode(
    seed_gen_ep: tuple[int, int, int],
) -> tuple[list[ProcessedExample], MCTSEpisodeStats]:
    """Run one self-play episode inside a worker process.

    Args:
        seed_gen_ep: ``(base_seed, generation, episode_idx)``. Re-derives
            the per-episode seed locally rather than receiving it
            already-derived, so the orchestrator's outgoing task payload
            stays human-readable in logs.

    Returns:
        ``(training_examples, mcts_episode_stats)``.
    """
    from core.self_play import play_self_play_episode

    assert _WORKER_CONFIG is not None, "_worker_init must run before _worker_play_episode"
    assert _WORKER_GAME is not None
    assert _WORKER_NNET is not None

    base_seed, generation, episode_idx = seed_gen_ep
    seed = derive_episode_seed(base_seed, generation, episode_idx)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mcts = MCTS(_WORKER_GAME, _WORKER_NNET, _WORKER_CONFIG.mcts_config)
    examples = play_self_play_episode(
        _WORKER_GAME, mcts, _WORKER_CONFIG.temp_threshold,
    )
    stats = mcts.get_episode_stats()
    return examples, stats


def run_self_play_episodes_parallel(
    config: RunConfig,
    generation: int,
    checkpoint_path: str,
    num_workers: int,
) -> tuple[list[list[ProcessedExample]], list[MCTSEpisodeStats]]:
    """Run ``config.num_eps`` self-play episodes across a worker pool.

    Args:
        config: Run config. ``config.num_eps`` controls episode count;
            ``config.seed`` (or 0 if unset) seeds the per-episode RNG
            derivation; ``config.mcts_config`` configures the MCTS each
            worker builds.
        generation: Current training generation. Mixed into per-episode
            seeds so different generations produce different games even
            at the same episode index.
        checkpoint_path: Filename (under ``config.net_directory``)
            workers should ``load_checkpoint`` from at init. Passing a
            path here is how we sync the latest weights into each
            worker — the pool is re-spawned per generation.
        num_workers: Number of worker processes. Must be ≥ 1; ``1`` is
            allowed and runs sequentially through the pool (useful for
            testing the worker module in isolation; for the actual
            single-process path :class:`core.coach.Coach` skips this
            function entirely).

    Returns:
        ``(per_episode_examples, per_episode_stats)`` — outer list is
        one entry per episode, in submission order
        ``range(config.num_eps)``. The order is preserved (not
        ``imap_unordered``) so episode-indexed metrics line up cleanly.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    base_seed = config.seed if config.seed is not None else 0
    tasks = [(base_seed, generation, ep) for ep in range(config.num_eps)]

    logger.info(
        "Spawning {} worker(s) for {} self-play episodes (gen {})",
        num_workers, config.num_eps, generation,
    )

    per_episode_examples: list[list[ProcessedExample]] = []
    per_episode_stats: list[MCTSEpisodeStats] = []
    # ``ProcessPoolExecutor`` defaults to using the current start method.
    # On macOS/Linux+CUDA, the safe choice is "spawn" — set via
    # ``mp_context`` so we don't have to rely on a global setting that
    # might collide with other code that touched ``multiprocessing``.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with (
        ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(config, checkpoint_path),
        ) as pool,
        # ``map`` preserves submission order. ``chunksize=1`` keeps each
        # task as its own unit of work so tqdm advances per episode and
        # one slow game doesn't block neighbours behind it.
        tqdm(total=len(tasks), desc=f"Self-play gen {generation}") as bar,
    ):
        for examples, stats in pool.map(_worker_play_episode, tasks, chunksize=1):
            per_episode_examples.append(examples)
            per_episode_stats.append(stats)
            bar.update(1)

    return per_episode_examples, per_episode_stats
