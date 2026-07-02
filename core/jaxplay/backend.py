"""Coach-facing entry point for the jax self-play backend (plan step G6).

``generate_self_play_games`` mirrors ``run_self_play_episodes_parallel``'s
contract: load the freshly-saved checkpoint, generate exactly ``num_eps``
complete games, and return ``(per_game_examples, per_game_stats)`` in the
schema the Coach already consumes. Weight conversion (torch → jnp) happens
here once per generation; the compiled actor/search artefacts are cached
module-globally so later generations skip recompilation (shapes are identical
— only the params change).

Stats caveat, by design: the fused jit pipeline cannot separate inference time
from search time, so ``total_inference_time_s`` is reported as 0.0 and the
whole wave wall-clock is apportioned to ``total_search_time_s`` by each game's
share of harvested moves. Downstream reports treat these as diagnostics, not
invariants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from tqdm import tqdm

from core.mcts import MCTSEpisodeStats

if TYPE_CHECKING:
    from core.config import RunConfig
    from core.jaxplay.harvest import GameRecord
    from core.self_play import ProcessedExample

#: Compiled actor/search/env artefacts, keyed by everything that affects shapes
#: or compiled constants. Persist across generations (and Coach instances).
_ARTEFACT_CACHE: dict[tuple, dict[str, Any]] = {}


def _artefacts(config: RunConfig) -> dict[str, Any]:
    from core.game_factory import instantiate_game
    from core.jaxplay.actors import make_actor
    from core.jaxplay.harvest import TraceHarvester
    from games.blokusduo.jaxenv.kernels import make_kernels
    from games.blokusduo.jaxenv.search import SearchConfig, make_search
    from games.blokusduo.jaxenv.tables import build_jax_tables

    if config.game != "blokusduo":
        raise ValueError(
            f"selfplay_backend 'jax' supports only blokusduo (got {config.game!r}); "
            "use selfplay_backend 'python'."
        )
    jax_config = config.jax_selfplay
    mcts = config.mcts_config
    key = (
        jax_config.batch_size, jax_config.top_k, jax_config.dtype, jax_config.wave_plies,
        mcts.num_mcts_sims, mcts.cpuct, mcts.dirichlet_epsilon, mcts.dirichlet_alpha,
        config.temp_threshold,
    )
    if key not in _ARTEFACT_CACHE:
        if mcts.sim_schedule != "flat":
            logger.warning(
                "jax backend ignores sim_schedule={!r}: fixed-shape search uses a flat "
                "num_mcts_sims={} budget (see the plan's fidelity contract).",
                mcts.sim_schedule, mcts.num_mcts_sims,
            )
        game = instantiate_game(config)
        kernels = make_kernels(build_jax_tables(game))
        search = make_search(kernels, SearchConfig(
            num_simulations=mcts.num_mcts_sims,
            top_k=jax_config.top_k,
            cpuct=mcts.cpuct,
            dirichlet_epsilon=mcts.dirichlet_epsilon,
            dirichlet_alpha=mcts.dirichlet_alpha,
            dtype=jax_config.dtype,
        ))
        initial_carry, run_wave = make_actor(
            kernels, search,
            batch_size=jax_config.batch_size,
            temp_threshold=config.temp_threshold,
            wave_plies=jax_config.wave_plies,
        )
        _ARTEFACT_CACHE[key] = {
            "game": game,
            "initial_carry": initial_carry,
            "run_wave": run_wave,
            "make_harvester": lambda: TraceHarvester(game, jax_config.batch_size),
        }
    return _ARTEFACT_CACHE[key]


def _stats_for(record: GameRecord, num_sims: int, seconds_per_move: float) -> MCTSEpisodeStats:
    total_sims = record.num_moves * num_sims
    return MCTSEpisodeStats(
        num_moves=record.num_moves,
        total_sims=total_sims,
        total_search_time_s=seconds_per_move * record.num_moves,
        total_inference_time_s=0.0,  # fused with search under jit — not separable
        num_leaf_expansions=total_sims,
        tree_size=total_sims + 1,
        mean_policy_entropy=record.mean_policy_entropy,
    )


def generate_self_play_games(
    config: RunConfig, generation: int, checkpoint_path: str,
) -> tuple[list[list[ProcessedExample]], list[MCTSEpisodeStats]]:
    """Generate ``config.num_eps`` self-play games on the GPU.

    Args:
        config: Full run config (`jax_selfplay` + `mcts_config` drive search).
        generation: Current generation — folded into the RNG stream so every
            generation explores differently at a fixed ``config.seed``.
        checkpoint_path: Filename (under ``config.net_directory``) of the
            torch checkpoint holding the current network.

    Returns:
        ``(per_game_examples, per_game_stats)`` with exactly ``num_eps``
        entries each, completion-ordered.
    """
    import time

    import jax

    from games.blokusduo.jaxenv.checkpoint import convert_torch_checkpoint, params_to_device

    artefacts = _artefacts(config)
    params = params_to_device(
        convert_torch_checkpoint(
            config.net_directory / checkpoint_path, config.net_config.num_residual_blocks,
        ),
        dtype=config.jax_selfplay.dtype,
    )

    rng_key = jax.random.fold_in(jax.random.PRNGKey(config.seed or 0), generation)
    carry = artefacts["initial_carry"]()
    harvester = artefacts["make_harvester"]()
    run_wave = artefacts["run_wave"]

    records: list[GameRecord] = []
    wave_seconds = 0.0
    total_moves = 0
    with tqdm(total=config.num_eps, desc=f"Self-play gen {generation} (jax)") as progress:
        while len(records) < config.num_eps:
            rng_key, wave_key = jax.random.split(rng_key)
            wave_start = time.perf_counter()
            carry, trace = run_wave(params, wave_key, carry)
            jax.block_until_ready(carry.games.ppb)
            wave_seconds += time.perf_counter() - wave_start
            fresh_records = harvester.harvest(trace)
            total_moves += sum(record.num_moves for record in fresh_records)
            records.extend(fresh_records)
            progress.update(min(len(records), config.num_eps) - progress.n)

    overflow = len(records) - config.num_eps
    records = records[: config.num_eps]
    harvester.finalize()
    if overflow or harvester.truncated_games:
        logger.info(
            "jax self-play gen {}: kept {} games ({} overflow beyond num_eps dropped, "
            "{} truncated tail games discarded)",
            generation, len(records), overflow, harvester.truncated_games,
        )

    seconds_per_move = wave_seconds / max(total_moves, 1)
    examples = [record.examples for record in records]
    stats = [
        _stats_for(record, config.mcts_config.num_mcts_sims, seconds_per_move) for record in records
    ]
    return examples, stats
