"""Calibrate net-size cost on the current GPU → budget-fit table (cloud-scale C10).

Run this in the first minutes on a rented card, before committing the budget:
for each candidate net size it measures real self-play (jax backend) and real
training throughput, then prints what a £ budget buys — games/s, minutes and £
per generation, and how many generations fit — plus a recommended size.

Usage (on the rented box / in the container)::

    uv run python -m scripts.benchmarks.cloud_calibration \
        --config run_configurations/blokus_cloud_calibration.json \
        --rate-gbp-per-hour 0.70 --budget-gbp 100

    # CPU / no-jax machines: cost table from training measurements only
    uv run python -m scripts.benchmarks.cloud_calibration --config <cfg> --skip-selfplay

The config supplies everything except the sizes and money: game, search
config (Gumbel sims, batch), perf knobs, games/gen, buffer size, epochs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from loguru import logger

from alphablokus.calibration import (
    DEFAULT_POSITIONS_PER_GAME,
    NetSizeMeasurement,
    estimate_costs,
    format_markdown_table,
    measure_selfplay_games_per_second,
    measure_train_seconds_per_position,
    parse_net_sizes,
    recommend,
)
from alphablokus.config import RunConfig, load_args
from alphablokus.games.base_wrapper import count_parameters
from alphablokus.registry import instantiate_game_and_network


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure net-size throughput and fit a £ budget")
    ap.add_argument(
        "--config",
        default="run_configurations/blokus_cloud_calibration.json",
        help="Run config supplying game/search/perf/buffer settings for the measurements",
    )
    ap.add_argument("--sizes", default="small,medium,large,xl", help="Presets and/or <F>x<B> specs, comma separated")
    ap.add_argument("--rate-gbp-per-hour", type=float, required=True, help="What the card costs to rent")
    ap.add_argument("--budget-gbp", type=float, default=100.0, help="Total budget to fit (default £100)")
    ap.add_argument("--games-per-gen", type=int, default=None, help="Override config num_eps for the cost model")
    ap.add_argument("--min-generations", type=int, default=30, help="Floor a net must fit to be recommendable")
    ap.add_argument("--selfplay-games", type=int, default=64, help="Games in the measured self-play burst")
    ap.add_argument("--train-positions", type=int, default=8192, help="Synthetic buffer positions timed per size")
    ap.add_argument("--eval-overhead", type=float, default=0.15, help="Arena/Elo/report time as a fraction of core")
    ap.add_argument("--skip-selfplay", action="store_true", help="Train-only measurement (no jax required)")
    ap.add_argument("--json", dest="json_out", default=None, help="Also write results as JSON to this path")
    args = ap.parse_args()

    base_config: RunConfig = load_args(args.config)
    games_per_generation = args.games_per_gen or base_config.num_eps

    measurements: list[NetSizeMeasurement] = []
    for name, num_filters, num_blocks in parse_net_sizes(args.sizes):
        logger.info("Calibrating {} ({}f×{}b)...", name, num_filters, num_blocks)
        config = replace(
            base_config,
            net_config=replace(base_config.net_config, num_filters=num_filters, num_residual_blocks=num_blocks),
        )
        game, wrapper = instantiate_game_and_network(config)
        config.net_directory.mkdir(parents=True, exist_ok=True)

        games_per_s: float | None = None
        positions_per_game = None
        if not args.skip_selfplay:
            games_per_s, positions_per_game = measure_selfplay_games_per_second(
                config, wrapper, num_games=args.selfplay_games
            )
            logger.info("  self-play: {:.2f} games/s ({:.1f} positions/game)", games_per_s, positions_per_game)

        train_s_per_position = measure_train_seconds_per_position(wrapper, game, args.train_positions)
        logger.info("  training: {:.3f} ms/position", train_s_per_position * 1e3)

        measurements.append(
            NetSizeMeasurement(
                name=name,
                num_filters=num_filters,
                num_residual_blocks=num_blocks,
                parameters=count_parameters(wrapper.nnet),
                selfplay_games_per_s=games_per_s,
                train_seconds_per_position=train_s_per_position,
                positions_per_game=positions_per_game or DEFAULT_POSITIONS_PER_GAME,
            )
        )

    estimates = estimate_costs(
        measurements,
        games_per_generation=games_per_generation,
        replay_buffer_games=base_config.replay_buffer_games,
        epochs=base_config.net_config.epochs,
        eval_overhead_fraction=args.eval_overhead,
        rate_gbp_per_hour=args.rate_gbp_per_hour,
        budget_gbp=args.budget_gbp,
    )
    recommended = recommend(estimates, args.min_generations)

    print()
    print(
        f"Budget £{args.budget_gbp:.0f} at £{args.rate_gbp_per_hour:.2f}/h "
        f"— {games_per_generation} games/gen, buffer {base_config.replay_buffer_games} games, "
        f"epochs {base_config.net_config.epochs}"
    )
    print(format_markdown_table(estimates, recommended))
    if recommended is None:
        print(
            f"\nNo measured size fits >= {args.min_generations} generations in budget — "
            "lower --games-per-gen, raise the budget, or rent a faster card."
        )

    if args.json_out:
        payload = {
            "rate_gbp_per_hour": args.rate_gbp_per_hour,
            "budget_gbp": args.budget_gbp,
            "games_per_generation": games_per_generation,
            "estimates": [
                {
                    "name": e.measurement.name,
                    "num_filters": e.measurement.num_filters,
                    "num_residual_blocks": e.measurement.num_residual_blocks,
                    "parameters": e.measurement.parameters,
                    "selfplay_games_per_s": e.measurement.selfplay_games_per_s,
                    "train_seconds_per_position": e.measurement.train_seconds_per_position,
                    "positions_per_game": e.measurement.positions_per_game,
                    "seconds_per_generation": e.seconds_per_generation,
                    "cost_gbp_per_generation": e.cost_gbp_per_generation,
                    "generations_in_budget": e.generations_in_budget,
                    "recommended": recommended is e,
                }
                for e in estimates
            ],
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        logger.info("JSON results → {}", out)


if __name__ == "__main__":
    main()
