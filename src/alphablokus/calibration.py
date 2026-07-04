"""Cost model + measurement helpers for cloud-run calibration (cloud-scale C10).

Answers the question a rented GPU should be asked in its first fifteen minutes:
*given this card's measured self-play and training throughput, what net size and
how many generations does a £ budget actually buy?* The measurement functions
run the real code paths (the jax self-play backend, the real ``train()`` loop)
so the numbers reflect production behaviour; the cost arithmetic is pure and
unit-tested. CLI harness: ``scripts/benchmarks/cloud_calibration.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from alphablokus.config import NET_PRESETS

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame, INeuralNetWrapper
    from alphablokus.selfplay.episode import ProcessedExample

# Fallback when self-play measurement is skipped: mean positions per Blokus
# Duo self-play game (≈57 plies shared between two players' perspectives),
# consistent with the jax A/B measurements (docs/research/jax-pipeline-ab.md).
DEFAULT_POSITIONS_PER_GAME = 29.0


@dataclass(frozen=True)
class NetSizeMeasurement:
    """Measured throughput for one net size on the current GPU."""

    name: str  # preset name or "FxB" spec
    num_filters: int
    num_residual_blocks: int
    parameters: int
    selfplay_games_per_s: float | None  # None = self-play measurement skipped
    train_seconds_per_position: float  # one training pass, per buffer position
    positions_per_game: float  # mean examples one self-play game contributes


@dataclass(frozen=True)
class CostEstimate:
    """What one generation costs — and what the budget buys — at one net size."""

    measurement: NetSizeMeasurement
    selfplay_seconds_per_generation: float
    train_seconds_per_generation: float
    seconds_per_generation: float  # incl. eval overhead
    cost_gbp_per_generation: float
    generations_in_budget: int
    total_games_in_budget: int


def parse_net_sizes(spec: str) -> list[tuple[str, int, int]]:
    """Parse ``--sizes`` into ``(name, num_filters, num_residual_blocks)`` triples.

    Accepts preset names and explicit ``<filters>x<blocks>`` entries, comma
    separated — e.g. ``"small,medium,192x12"``.
    """
    sizes: list[tuple[str, int, int]] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token in NET_PRESETS:
            preset = NET_PRESETS[token]
            sizes.append((token, preset["num_filters"], preset["num_residual_blocks"]))
            continue
        filters_str, _, blocks_str = token.partition("x")
        if not (filters_str.isdigit() and blocks_str.isdigit()):
            raise ValueError(f"Bad net size {token!r}: expected a preset name {sorted(NET_PRESETS)} or '<F>x<B>'.")
        sizes.append((token, int(filters_str), int(blocks_str)))
    if not sizes:
        raise ValueError("--sizes parsed to nothing")
    return sizes


def estimate_costs(
    measurements: list[NetSizeMeasurement],
    *,
    games_per_generation: int,
    replay_buffer_games: int,
    epochs: int,
    eval_overhead_fraction: float,
    rate_gbp_per_hour: float,
    budget_gbp: float,
) -> list[CostEstimate]:
    """Turn throughput measurements into £-per-generation and budget fits.

    Training cost is modelled at the steady state (buffer full): ``epochs``
    full passes over ``replay_buffer_games × positions_per_game`` positions.
    ``eval_overhead_fraction`` covers arena/Elo/report time as a fraction of
    the self-play + train core (0.15 ≈ the historical share at 50-game arenas).
    """
    estimates: list[CostEstimate] = []
    for m in measurements:
        selfplay_s = games_per_generation / m.selfplay_games_per_s if m.selfplay_games_per_s else 0.0
        buffer_positions = replay_buffer_games * m.positions_per_game
        train_s = m.train_seconds_per_position * buffer_positions * epochs
        total_s = (selfplay_s + train_s) * (1.0 + eval_overhead_fraction)
        cost_per_gen = total_s / 3600.0 * rate_gbp_per_hour
        generations = int(budget_gbp / cost_per_gen) if cost_per_gen > 0 else 0
        estimates.append(
            CostEstimate(
                measurement=m,
                selfplay_seconds_per_generation=selfplay_s,
                train_seconds_per_generation=train_s,
                seconds_per_generation=total_s,
                cost_gbp_per_generation=cost_per_gen,
                generations_in_budget=generations,
                total_games_in_budget=generations * games_per_generation,
            )
        )
    return estimates


def recommend(estimates: list[CostEstimate], min_generations: int) -> CostEstimate | None:
    """The biggest net that still fits ``min_generations`` in the budget.

    AlphaZero-style training needs enough generations to climb regardless of
    net size (a huge net at 5 generations loses to a medium net at 40), so the
    rule is: qualify on generations first, then maximise capacity. Returns
    ``None`` when even the smallest measured net can't fit the floor —
    at which point the honest answer is "lower games/gen or raise the budget".
    """
    qualifying = [e for e in estimates if e.generations_in_budget >= min_generations]
    if not qualifying:
        return None
    return max(qualifying, key=lambda e: e.measurement.parameters)


def format_markdown_table(estimates: list[CostEstimate], recommended: CostEstimate | None) -> str:
    """Human-readable calibration summary (also dropped into run notes/PRs)."""
    lines = [
        "| net | params | self-play games/s | min/gen | £/gen | gens in budget | games total |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in estimates:
        m = e.measurement
        marker = " **⟵ recommended**" if recommended is e else ""
        games_per_s = f"{m.selfplay_games_per_s:.2f}" if m.selfplay_games_per_s else "—"
        lines.append(
            f"| {m.name} ({m.num_filters}f×{m.num_residual_blocks}b){marker} "
            f"| {m.parameters / 1e6:.1f}M "
            f"| {games_per_s} "
            f"| {e.seconds_per_generation / 60:.1f} "
            f"| {e.cost_gbp_per_generation:.2f} "
            f"| {e.generations_in_budget} "
            f"| {e.total_games_in_budget:,} |"
        )
    return "\n".join(lines)


def measure_train_seconds_per_position(
    wrapper: INeuralNetWrapper,
    game: IGame,
    num_positions: int,
) -> float:
    """Time one full training pass over a synthetic buffer via the real ``train()``.

    Uses compact boards + realistically sparse policies (the replay buffer's
    storage format), so DataLoader/encoding cost is included — that is exactly
    the loop the perf knobs (C3) are meant to keep fed.
    """
    board = game.initialise_board()
    compact = board.to_compact()
    action_size = game.get_action_size()
    rng = np.random.default_rng(0)
    examples: list[ProcessedExample] = []
    for _ in range(num_positions):
        # Sparse (indices, values) policies over ~30 legal moves — the replay
        # buffer's real storage format, so densify cost is measured too.
        support = rng.choice(action_size, size=min(30, action_size), replace=False).astype(np.int32)
        values = rng.dirichlet(np.ones(len(support))).astype(np.float32)
        examples.append((compact, (np.sort(support), values), float(rng.uniform(-1, 1))))

    start = time.perf_counter()
    wrapper.train(examples, generation=0)
    elapsed = time.perf_counter() - start
    return elapsed / num_positions


def measure_selfplay_games_per_second(
    config: RunConfig,
    wrapper: INeuralNetWrapper,
    num_games: int = 0,
    warmup: bool = True,
) -> tuple[float, float]:
    """jax-backend self-play throughput: ``(games_per_s, mean_positions_per_game)``.

    Saves the wrapper's current weights as a checkpoint and drives the real
    ``generate_self_play_games``. A warmup burst absorbs jit compilation so
    the measured burst reflects steady-state throughput (the artefact cache
    persists between calls).

    ``num_games`` defaults (0) to ``2 × jax_selfplay.batch_size``: the backend
    computes whole waves of ``batch_size`` game slots regardless of how few
    games are requested, so a burst smaller than the batch would divide full-
    wave wall-clock by a fraction of the games actually produced and badly
    undercount throughput.

    ``positions_per_game`` counts **training examples** (symmetry augmentation
    included) — the number the buffer-size cost model needs — not raw moves.
    """
    from alphablokus.registry import resolve_jax_selfplay_backend

    generate = resolve_jax_selfplay_backend(config)
    wrapper.save_checkpoint(filename="calibration.pth.tar")
    if num_games <= 0:
        num_games = 2 * config.jax_selfplay.batch_size
    burst_config = replace(config, num_eps=num_games)
    if warmup:
        generate(replace(burst_config, num_eps=max(1, num_games // 8)), 0, "calibration.pth.tar")
    start = time.perf_counter()
    per_game_examples, _stats = generate(burst_config, 1, "calibration.pth.tar")
    elapsed = time.perf_counter() - start
    positions_per_game = (
        float(np.mean([len(g) for g in per_game_examples])) if per_game_examples else DEFAULT_POSITIONS_PER_GAME
    )
    return num_games / elapsed, positions_per_game
