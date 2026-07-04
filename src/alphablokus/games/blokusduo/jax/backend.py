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

from alphablokus.search.stats import MCTSEpisodeStats

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphablokus.config import RunConfig
    from alphablokus.selfplay.episode import ProcessedExample

#: Compiled actor/search/env artefacts, keyed by everything that affects shapes
#: or compiled constants. Persist across generations (and Coach instances).
_ARTEFACT_CACHE: dict[tuple, dict[str, Any]] = {}


def _artefacts(config: RunConfig) -> dict[str, Any]:
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.jax.actors import make_actor
    from alphablokus.games.blokusduo.jax.harvest import TraceHarvester
    from alphablokus.games.blokusduo.jax.kernels import make_kernels
    from alphablokus.games.blokusduo.jax.search import SearchConfig, make_search
    from alphablokus.games.blokusduo.jax.tables import build_jax_tables
    from alphablokus.registry import instantiate_game

    if config.game != "blokusduo":
        raise ValueError(
            f"selfplay_backend 'jax' supports only blokusduo (got {config.game!r}); use selfplay_backend 'python'."
        )
    jax_config = config.jax_selfplay
    mcts = config.mcts_config
    key = (
        jax_config.batch_size,
        jax_config.top_k,
        jax_config.dtype,
        jax_config.wave_plies,
        mcts.num_mcts_sims,
        mcts.cpuct,
        mcts.dirichlet_epsilon,
        mcts.dirichlet_alpha,
        mcts.search_policy,
        mcts.gumbel_max_considered,
        config.temp_threshold,
    )
    if key not in _ARTEFACT_CACHE:
        if mcts.sim_schedule != "flat":
            logger.warning(
                "jax backend ignores sim_schedule={!r}: fixed-shape search uses a flat "
                "num_mcts_sims={} budget (see the plan's fidelity contract).",
                mcts.sim_schedule,
                mcts.num_mcts_sims,
            )
        game = instantiate_game(config)
        assert isinstance(game, BlokusDuoGame)  # config.game validated above
        kernels = make_kernels(build_jax_tables(game))
        search = make_search(
            kernels,
            SearchConfig(
                num_simulations=mcts.num_mcts_sims,
                top_k=jax_config.top_k,
                cpuct=mcts.cpuct,
                dirichlet_epsilon=mcts.dirichlet_epsilon,
                dirichlet_alpha=mcts.dirichlet_alpha,
                dtype=jax_config.dtype,
                policy=mcts.search_policy,
                gumbel_max_considered=mcts.gumbel_max_considered,
            ),
        )
        initial_carry, run_wave = make_actor(
            kernels,
            search,
            batch_size=jax_config.batch_size,
            temp_threshold=config.temp_threshold,
            wave_plies=jax_config.wave_plies,
            use_search_action=mcts.search_policy == "gumbel",
        )
        _ARTEFACT_CACHE[key] = {
            "game": game,
            "initial_carry": initial_carry,
            "run_wave": run_wave,
            "make_harvester": lambda: TraceHarvester(game, jax_config.batch_size),
        }
    return _ARTEFACT_CACHE[key]


def _stats_for(num_moves: int, mean_policy_entropy: float, num_sims: int, seconds_per_move: float) -> MCTSEpisodeStats:
    total_sims = num_moves * num_sims
    return MCTSEpisodeStats(
        num_moves=num_moves,
        total_sims=total_sims,
        total_search_time_s=seconds_per_move * num_moves,
        total_inference_time_s=0.0,  # fused with search under jit — not separable
        num_leaf_expansions=total_sims,
        tree_size=total_sims + 1,
        mean_policy_entropy=mean_policy_entropy,
    )


def generate_self_play_games(
    config: RunConfig,
    generation: int,
    checkpoint_path: str,
    sink: Callable[[list[ProcessedExample]], None] | None = None,
) -> tuple[list[list[ProcessedExample]], list[MCTSEpisodeStats]]:
    """Generate ``config.num_eps`` self-play games on the GPU.

    Args:
        config: Full run config (`jax_selfplay` + `mcts_config` drive search).
        generation: Current generation — folded into the RNG stream so every
            generation explores differently at a fixed ``config.seed``.
        checkpoint_path: Filename (under ``config.net_directory``) of the
            torch checkpoint holding the current network.
        sink: Optional per-game consumer. When provided, each kept game's
            examples are handed over as soon as its final wave is harvested —
            the whole generation is never accumulated here — and the returned
            examples list is empty. Stats are still returned at the end
            because their timing apportionment needs the generation's total
            wall clock.

    Returns:
        ``(per_game_examples, per_game_stats)`` — stats always have exactly
        ``num_eps`` entries, completion-ordered; the examples list matches
        when ``sink`` is ``None`` and is empty when a ``sink`` consumed them.
    """
    import time

    # Apply the configured VRAM cap before this process's first ``import jax``
    # (XLA reads the env var once at backend init; explicit env vars win).
    from alphablokus.games.blokusduo.jax import configure_xla_mem_fraction

    configure_xla_mem_fraction(config.jax_selfplay.xla_mem_fraction)

    import jax

    from alphablokus.games.blokusduo.jax.checkpoint import convert_torch_checkpoint, params_to_device

    artefacts = _artefacts(config)
    params = params_to_device(
        convert_torch_checkpoint(
            config.net_directory / checkpoint_path,
            config.net_config.num_residual_blocks,
        ),
        dtype=config.jax_selfplay.dtype,
    )

    rng_key = jax.random.fold_in(jax.random.PRNGKey(config.seed or 0), generation)
    carry = artefacts["initial_carry"]()
    harvester = artefacts["make_harvester"]()
    run_wave = artefacts["run_wave"]

    per_game_examples: list[list[ProcessedExample]] = []
    game_meta: list[tuple[int, float]] = []  # (num_moves, mean_policy_entropy) per kept game
    overflow = 0
    wave_seconds = 0.0
    total_moves = 0
    with tqdm(total=config.num_eps, desc=f"Self-play gen {generation} (jax)") as progress:
        while len(game_meta) < config.num_eps:
            rng_key, wave_key = jax.random.split(rng_key)
            wave_start = time.perf_counter()
            carry, trace = run_wave(params, wave_key, carry)
            jax.block_until_ready(carry.games.ppb)
            wave_seconds += time.perf_counter() - wave_start
            fresh_records = harvester.harvest(trace)
            total_moves += sum(record.num_moves for record in fresh_records)
            for record in fresh_records:
                if len(game_meta) >= config.num_eps:
                    overflow += 1  # harvested beyond num_eps in the final wave — dropped
                    continue
                game_meta.append((record.num_moves, record.mean_policy_entropy))
                if sink is not None:
                    sink(record.examples)
                else:
                    per_game_examples.append(record.examples)
            progress.update(len(game_meta) - progress.n)

    harvester.finalize()
    if overflow or harvester.truncated_games:
        logger.info(
            "jax self-play gen {}: kept {} games ({} overflow beyond num_eps dropped, "
            "{} truncated tail games discarded)",
            generation,
            len(game_meta),
            overflow,
            harvester.truncated_games,
        )

    seconds_per_move = wave_seconds / max(total_moves, 1)
    stats = [
        _stats_for(num_moves, entropy, config.mcts_config.num_mcts_sims, seconds_per_move)
        for num_moves, entropy in game_meta
    ]
    return per_game_examples, stats
