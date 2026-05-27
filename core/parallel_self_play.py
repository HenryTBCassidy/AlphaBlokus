"""Process-pool workers for parallel game-playing phases (F1).

Three phases of training run independent games and are all parallelised
through the same machinery here:

1. **Self-play** (P3): one network plays both sides of each game,
   producing training examples + MCTS stats.
2. **Arena** (P8): two networks (current vs previous-best) play, with
   game records returned for replay storage.
3. **Elo** (P9): two networks (current vs frozen gen-0 baseline) play,
   no records persisted.

Public surface:

- :func:`run_self_play_episodes_parallel` — self-play orchestrator.
- :func:`run_two_player_games_parallel` — shared orchestrator for arena
  + Elo. Returns ``(a_wins, b_wins, draws, records)``; the caller maps
  "a"/"b" to whichever role names it cares about.
- :data:`derive_episode_seed` — pure function mapping
  ``(base_seed, generation, episode_idx, phase)`` to a per-episode seed
  so the same input always plays the same game regardless of which
  worker picks it up.

Each worker process is initialised once at pool start (loading game +
network(s) into module-level globals) and runs many per-task functions
afterward, amortising the per-process construction cost across many
games. The orchestrator uses ``ProcessPoolExecutor.map`` with
``chunksize=1`` so the main process sees results in submission order
and one slow game doesn't block neighbours behind it.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from core.arena import Arena, GameRecord
from core.game_factory import instantiate_game_and_network
from core.mcts import MCTS
from core.players import NetworkPlayer

if TYPE_CHECKING:
    from core.config import RunConfig
    from core.interfaces import IGame, INeuralNetWrapper
    from core.mcts import MCTSEpisodeStats
    from core.self_play import ProcessedExample


# Module-level state populated inside each worker process by the pool
# initializer. Lives at module level (not in a class) because
# ``multiprocessing`` initializers populate per-process globals — the
# alternative (passing the game + nnet via every task argument) would
# re-pickle them per task, which is exactly what we're trying to avoid.
#
# Self-play workers populate ``_WORKER_NNET_A`` only and leave
# ``_WORKER_NNET_B`` as ``None``. Two-player workers (arena / Elo)
# populate both.
_WORKER_CONFIG: RunConfig | None = None
_WORKER_GAME: IGame | None = None
_WORKER_NNET_A: INeuralNetWrapper | None = None
_WORKER_NNET_B: INeuralNetWrapper | None = None


# Phase codes that get mixed into the per-episode seed so the same
# ``(base, gen, ep_idx)`` triple plays *different* games across phases.
# Numeric so they can be passed positionally to ``derive_episode_seed``.
PHASE_SELF_PLAY = 0
PHASE_ARENA = 1
PHASE_ELO = 2


def derive_episode_seed(
    base_seed: int,
    generation: int,
    episode_idx: int,
    phase: int = PHASE_SELF_PLAY,
) -> int:
    """Stable mapping from ``(base, gen, ep, phase)`` to a 32-bit RNG seed.

    The same input always returns the same seed; different inputs return
    different seeds. Two callers with the same arguments will run the
    exact same game whether they execute it in the main process or in a
    worker, in any order.

    ``phase`` defaults to ``PHASE_SELF_PLAY`` so existing self-play
    callers don't need to change. Arena / Elo orchestrators pass
    ``PHASE_ARENA`` / ``PHASE_ELO`` so episode 0 of self-play and
    episode 0 of arena get different seeds.

    Uses ``numpy``'s ``SeedSequence`` rather than a hand-rolled hash so
    we inherit numpy's documented good statistical properties — for the
    1000s-of-episodes scale we operate at, collisions are vanishingly
    unlikely.
    """
    seq = np.random.SeedSequence([base_seed, generation, episode_idx, phase])
    # ``generate_state`` returns ``uint32`` — coerce to plain int because
    # ``torch.manual_seed`` rejects numpy scalar types.
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _seed_worker_rngs(seed: int) -> None:
    """Reset every RNG a worker touches to a fixed seed.

    Called at the start of every per-task function so each task is
    deterministic from the same ``(base, gen, ep, phase)`` regardless of
    which worker picks it up.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_self_play(config: RunConfig, checkpoint_path: str | None) -> None:
    """Pool initialiser for self-play workers — single network only.

    Builds the per-process game + nnet (and loads weights from
    ``checkpoint_path`` if provided). Stashes both in module-level
    globals so every per-task call reuses them.
    """
    global _WORKER_CONFIG, _WORKER_GAME, _WORKER_NNET_A
    _WORKER_CONFIG = config
    _WORKER_GAME, _WORKER_NNET_A = instantiate_game_and_network(config)
    if checkpoint_path is not None:
        _WORKER_NNET_A.load_checkpoint(filename=checkpoint_path)


def _worker_init_two_nets(
    config: RunConfig,
    checkpoint_a_path: str,
    checkpoint_b_path: str,
) -> None:
    """Pool initialiser for two-net workers — arena + Elo phases.

    Builds the per-process game + *two* nnets, each loaded from its own
    checkpoint. The two roles ("a" = network being evaluated, "b" =
    opponent) are caller-defined; the orchestrator wires them to
    new/prev for arena or new/baseline for Elo.
    """
    global _WORKER_CONFIG, _WORKER_GAME, _WORKER_NNET_A, _WORKER_NNET_B
    _WORKER_CONFIG = config
    _WORKER_GAME, _WORKER_NNET_A = instantiate_game_and_network(config)
    _WORKER_NNET_A.load_checkpoint(filename=checkpoint_a_path)
    # Second net shares the already-constructed game; only the wrapper
    # itself is rebuilt with fresh weights.
    _WORKER_NNET_B = _WORKER_NNET_A.__class__(_WORKER_GAME, config)
    _WORKER_NNET_B.load_checkpoint(filename=checkpoint_b_path)


def _worker_play_self_play_episode(
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

    assert _WORKER_CONFIG is not None, "_worker_init_self_play must run before this task"
    assert _WORKER_GAME is not None
    assert _WORKER_NNET_A is not None

    base_seed, generation, episode_idx = seed_gen_ep
    _seed_worker_rngs(derive_episode_seed(base_seed, generation, episode_idx, PHASE_SELF_PLAY))

    mcts = MCTS(_WORKER_GAME, _WORKER_NNET_A, _WORKER_CONFIG.mcts_config)
    examples = play_self_play_episode(
        _WORKER_GAME, mcts, _WORKER_CONFIG.temp_threshold,
    )
    stats = mcts.get_episode_stats()
    return examples, stats


def _worker_play_two_player_game(
    task: tuple[int, int, int, bool, bool, int, int],
) -> tuple[int, GameRecord | None]:
    """Play one two-net game inside a worker process.

    Args:
        task: 7-tuple of ``(base_seed, generation, episode_idx, a_goes_first,
            record, top_k, phase)``.

            - ``a_goes_first``: ``True`` plays as ``Arena(a, b)``,
              ``False`` plays as ``Arena(b, a)`` (the existing
              halfway-swap convention from :meth:`Arena.play_games`).
            - ``record``: whether to capture move-by-move policies for
              replay storage. Arena uses ``True``, Elo uses ``False``.
            - ``top_k``: passed through to ``arena.play_game`` for
              recorded games; ignored when ``record`` is False.
            - ``phase``: one of ``PHASE_ARENA`` / ``PHASE_ELO`` — mixed
              into the seed so arena and Elo episodes at the same index
              don't replay the same game.

    Returns:
        ``(a_outcome, record_or_none)``.

        ``a_outcome`` is ``+1`` if A won, ``-1`` if B won, ``0`` for a
        draw — *always from A's perspective*, regardless of who went
        first. The orchestrator can sum these straight into
        ``(a_wins, b_wins, draws)`` without further bookkeeping.

        ``record_or_none`` is a :class:`GameRecord` when ``record`` was
        True, with ``outcome`` rewritten to A's perspective and
        ``player1_was_white`` reflecting whether A played as white
        (i.e. went first).
    """
    assert _WORKER_CONFIG is not None, "_worker_init_two_nets must run before this task"
    assert _WORKER_GAME is not None
    assert _WORKER_NNET_A is not None
    assert _WORKER_NNET_B is not None

    base_seed, generation, episode_idx, a_first, record, top_k, phase = task
    _seed_worker_rngs(derive_episode_seed(base_seed, generation, episode_idx, phase))

    player_a = NetworkPlayer(
        game=_WORKER_GAME, nnet=_WORKER_NNET_A,
        mcts_config=_WORKER_CONFIG.mcts_config, temp=0.0,
    )
    player_b = NetworkPlayer(
        game=_WORKER_GAME, nnet=_WORKER_NNET_B,
        mcts_config=_WORKER_CONFIG.mcts_config, temp=0.0,
    )

    if a_first:
        arena = Arena(player_a, player_b, _WORKER_GAME)
        raw_outcome, raw_record = arena.play_game(record=record, top_k=top_k)
        a_outcome = raw_outcome  # raw_outcome +1 = player1 = A won
    else:
        arena = Arena(player_b, player_a, _WORKER_GAME)
        raw_outcome, raw_record = arena.play_game(record=record, top_k=top_k)
        a_outcome = -raw_outcome  # raw_outcome +1 = player1 = B won; flip for A

    # ``int(a_outcome)`` so the caller can sum without numpy types
    # leaking through.
    a_outcome_int = int(a_outcome) if a_outcome != 0 else 0

    if raw_record is None:
        return a_outcome_int, None

    # Rewrite the record from A's perspective for downstream consumers
    # (arena replay parquet keys games by player1 = the network being
    # evaluated, which is A by our convention).
    rebased = GameRecord(
        moves=raw_record.moves,
        outcome=a_outcome_int,
        player1_was_white=a_first,
    )
    return a_outcome_int, rebased


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
            initializer=_worker_init_self_play,
            initargs=(config, checkpoint_path),
        ) as pool,
        # ``map`` preserves submission order. ``chunksize=1`` keeps each
        # task as its own unit of work so tqdm advances per episode and
        # one slow game doesn't block neighbours behind it.
        tqdm(total=len(tasks), desc=f"Self-play gen {generation}") as bar,
    ):
        for examples, stats in pool.map(_worker_play_self_play_episode, tasks, chunksize=1):
            per_episode_examples.append(examples)
            per_episode_stats.append(stats)
            bar.update(1)

    return per_episode_examples, per_episode_stats


def run_two_player_games_parallel(
    config: RunConfig,
    generation: int,
    checkpoint_a_path: str,
    checkpoint_b_path: str,
    num_games: int,
    num_workers: int,
    *,
    phase: int,
    record: bool = False,
    top_k: int = 5,
    desc: str = "Two-player games",
) -> tuple[int, int, int, list[GameRecord]]:
    """Run ``num_games`` two-net games across a worker pool.

    Used by both **arena** (``phase=PHASE_ARENA``, ``record=True``) and
    **Elo** (``phase=PHASE_ELO``, ``record=False``). Each worker loads
    both checkpoints once at pool init and plays one game per task.

    **Caller-defined role convention**: the orchestrator is agnostic
    about which network is "the new net" vs "the opponent" — it just
    plays games between two checkpoints. *The caller must pick A/B to
    match the downstream consumer's convention.* Specifically:

    - For **arena**, ``Coach._run_arena_parallel`` passes A = prev,
      B = new so the resulting records' ``outcome`` and
      ``player1_was_white`` line up with
      ``Coach._run_arena_serial``'s ``Arena(prev, new)`` convention
      that ``reporting/training._render_arena_replays`` reads.
    - For **Elo**, ``Coach._run_elo_parallel`` passes A = new,
      B = baseline so ``a_wins`` flows naturally into the ``wins``
      slot of ``_compute_elo``.

    Args:
        config: Run config — ``config.seed`` seeds episode derivation,
            ``config.mcts_config`` parameterises both players' MCTS.
        generation: Current training generation — mixed into per-game
            seeds so different generations replay different games at the
            same episode index.
        checkpoint_a_path: Filename (relative to ``config.net_directory``)
            workers load into network A. Half the games run with A
            going first; the other half with B first.
        checkpoint_b_path: Filename for network B.
        num_games: Total games to play. Halved so half the games have A
            going first and half have B going first, matching the
            existing :meth:`Arena.play_games` swap convention.
        num_workers: Number of worker processes.
        phase: ``PHASE_ARENA`` or ``PHASE_ELO`` — namespaces the seed.
        record: Whether to capture and return per-game move records.
        top_k: Top-K policies recorded per move when ``record`` is True.
        desc: Label for the tqdm progress bar.

    Returns:
        ``(a_wins, b_wins, draws, records)`` — outcomes counted from
        network A's perspective regardless of who went first per game.
        Each :class:`GameRecord` in ``records`` carries ``outcome``
        from A's perspective and ``player1_was_white`` = whether A
        went first that game. ``records`` is empty when ``record`` is
        False.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if num_games < 2:
        raise ValueError(
            f"num_games must be >= 2 (halved into A-first / B-first), got {num_games}",
        )

    base_seed = config.seed if config.seed is not None else 0
    half = num_games // 2
    # First ``half`` games: A goes first. Next ``half``: B goes first.
    # Episode indices stay 0..2*half-1 so seeds are unique per game.
    tasks = [
        (base_seed, generation, ep_idx, True, record, top_k, phase)
        for ep_idx in range(half)
    ]
    tasks += [
        (base_seed, generation, ep_idx, False, record, top_k, phase)
        for ep_idx in range(half, 2 * half)
    ]

    logger.info(
        "Spawning {} worker(s) for {} two-player games (gen {}, phase {})",
        num_workers, len(tasks), generation, phase,
    )

    a_wins = 0
    b_wins = 0
    draws = 0
    records: list[GameRecord] = []

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with (
        ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init_two_nets,
            initargs=(config, checkpoint_a_path, checkpoint_b_path),
        ) as pool,
        tqdm(total=len(tasks), desc=f"{desc} gen {generation}") as bar,
    ):
        for a_outcome, rec in pool.map(_worker_play_two_player_game, tasks, chunksize=1):
            if a_outcome > 0:
                a_wins += 1
            elif a_outcome < 0:
                b_wins += 1
            else:
                draws += 1
            if rec is not None:
                records.append(rec)
            bar.update(1)

    return a_wins, b_wins, draws, records
