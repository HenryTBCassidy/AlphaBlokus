"""Build the AlphaBlokus W&B workspace via the `wandb-workspaces` SDK.

Run this once per W&B project (e.g. after the first run, or when the project
gets reset). The script creates a saved view named "AlphaBlokus default"
with the agreed section ordering and panel grouping. Re-running is safe —
the SDK upserts by view name.

Usage::

    uv run python scripts/setup_wandb_workspace.py \
        --entity henrycassidy --project alphablokus-poc

The dashboard follows the chronological order of a single training cycle:

  Run progress → Self-play → Training loss → Learning quality
  → Arena → Strength → Operational

so a glance tells you both where you are in the run and what's happening.
Each section's panels match the ``define_metric`` namespaces wired in
``core/storage.py``.
"""
from __future__ import annotations

import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws


def _line(
    title: str,
    metrics: list[str],
    *,
    x: str = "generation",
    smoothing: float | None = None,
    y_range: tuple[float | None, float | None] = (None, None),
) -> wr.LinePlot:
    """Convenience wrapper around ``wr.LinePlot`` with the common defaults
    we use across the dashboard: explicit x-axis, optional smoothing, optional
    y-range. Title is shown above the panel in the W&B UI.
    """
    kwargs: dict = {
        "title": title,
        "x": x,
        "y": metrics,
        "range_y": y_range,
    }
    if smoothing is not None:
        kwargs["smoothing_factor"] = smoothing
        kwargs["smoothing_type"] = "exponential"
    return wr.LinePlot(**kwargs)


def _bar(title: str, metrics: list[str], *, x: str = "generation") -> wr.BarPlot:  # noqa: ARG001
    """Bar chart — used for per-gen wins/losses/draws breakdowns. The ``x``
    argument is accepted for symmetry with ``_line`` but BarPlot's x-axis is
    always the (numerical) bar value, so it's ignored here.
    """
    return wr.BarPlot(title=title, metrics=metrics)


def build_sections() -> list[ws.Section]:
    """Return the full ordered list of sections for the workspace.

    Section ordering matches Henry's chronological-cycle preference:
    Run progress → Self-play → Training → Learning quality → Arena
    → Strength → Operational.
    """
    return [
        ws.Section(
            name="Run progress",
            is_open=True,
            panels=[
                _line(
                    "Generation",
                    ["progress/generation"],
                    x="progress/wall_clock_seconds",
                ),
                _line(
                    "ETA (seconds remaining)",
                    ["progress/eta_seconds"],
                    x="progress/wall_clock_seconds",
                ),
                _line(
                    "Generation fraction",
                    ["progress/generation_fraction"],
                    x="progress/wall_clock_seconds",
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Epoch within current generation",
                    ["progress/epoch"],
                    x="progress/wall_clock_seconds",
                ),
                _line(
                    "Episode within current generation",
                    ["progress/episode"],
                    x="progress/wall_clock_seconds",
                ),
                _line(
                    "Batch within current epoch",
                    ["progress/batch"],
                    x="progress/wall_clock_seconds",
                ),
            ],
        ),
        ws.Section(
            name="Self-play",
            is_open=True,
            panels=[
                _line(
                    "MCTS policy entropy (per-gen mean)",
                    ["self_play_per_gen/policy_entropy_mean"],
                ),
                _line(
                    "Mean game length",
                    ["self_play_per_gen/num_moves_mean"],
                ),
                _line(
                    "Mean MCTS tree size",
                    ["self_play_per_gen/tree_size_mean"],
                ),
                _line(
                    "MCTS sims/second",
                    ["self_play_per_gen/sims_per_second_mean"],
                ),
                _line(
                    "MCTS policy entropy (per-episode)",
                    ["self_play/policy_entropy"],
                    x="global_episode",
                    smoothing=0.6,
                ),
                _line(
                    "Game length (per-episode)",
                    ["self_play/num_moves"],
                    x="global_episode",
                    smoothing=0.6,
                ),
            ],
        ),
        ws.Section(
            name="Training loss",
            is_open=True,
            panels=[
                _line(
                    "Per-gen losses (pi / v / total)",
                    [
                        "training_per_gen/total_loss",
                        "training_per_gen/pi_loss",
                        "training_per_gen/v_loss",
                    ],
                ),
                _line(
                    "Total loss",
                    ["training_per_gen/total_loss"],
                ),
                _line(
                    "Policy loss",
                    ["training_per_gen/pi_loss"],
                ),
                _line(
                    "Value loss",
                    ["training_per_gen/v_loss"],
                ),
                _line(
                    "Per-batch loss (raw, smoothed)",
                    [
                        "training/total_loss",
                        "training/pi_loss",
                        "training/v_loss",
                    ],
                    x="global_batch",
                    smoothing=0.8,
                ),
            ],
        ),
        ws.Section(
            name="Learning quality",
            is_open=True,
            panels=[
                _line(
                    "Net top-1 agreement (vs minimax for TTT, vs MCTS for Blokus)",
                    ["training_per_gen/network_top1_accuracy"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Net top-5 agreement",
                    ["training_per_gen/network_top5_accuracy"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Value-head calibration error",
                    ["training_per_gen/value_calibration_error"],
                ),
                _line(
                    "Network policy entropy",
                    ["training_per_gen/network_policy_entropy"],
                ),
            ],
        ),
        ws.Section(
            name="Arena (new vs prev)",
            is_open=True,
            panels=[
                _line(
                    "Arena win rate",
                    ["arena/win_rate"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Acceptance rate (running)",
                    ["arena/acceptance_rate"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Accepted (per gen, 0/1)",
                    ["arena/accepted"],
                    y_range=(-0.05, 1.05),
                ),
                _bar(
                    "Arena wins / losses / draws",
                    ["arena/wins", "arena/losses", "arena/draws"],
                ),
            ],
        ),
        ws.Section(
            name="Strength (Elo + minimax)",
            is_open=True,
            panels=[
                _line(
                    "Elo rating (anchored at gen-0 baseline = 400)",
                    ["elo/rating"],
                ),
                _line(
                    "Elo difference vs baseline",
                    ["elo/diff_vs_baseline"],
                ),
                _line(
                    "Score rate vs gen-0",
                    ["elo/score_rate"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "vs minimax: win / draw / loss rate (TTT only)",
                    [
                        "minimax/win_rate",
                        "minimax/draw_rate",
                        "minimax/loss_rate",
                    ],
                    y_range=(0.0, 1.05),
                ),
                _bar(
                    "vs minimax: wins / losses / draws (TTT only)",
                    ["minimax/wins", "minimax/losses", "minimax/draws"],
                ),
            ],
        ),
        ws.Section(
            name="Operational",
            is_open=False,
            panels=[
                _line(
                    "Wall-clock per phase (seconds)",
                    [
                        "timing/SelfPlay_s",
                        "timing/Training_s",
                        "timing/Arena_s",
                        "timing/WholeCycle_s",
                    ],
                ),
                _line(
                    "Training throughput (samples/sec)",
                    ["throughput/samples_per_second"],
                ),
                _line(
                    "Self-play inference fraction",
                    ["self_play_per_gen/inference_fraction_mean"],
                    y_range=(0.0, 1.05),
                ),
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", required=True,
        help="W&B entity (user or team) that owns the project",
    )
    parser.add_argument(
        "--project", required=True,
        help="W&B project name (e.g. alphablokus-poc)",
    )
    parser.add_argument(
        "--name", default="AlphaBlokus default",
        help="Saved-view name. Re-running with the same name overwrites the view.",
    )
    args = parser.parse_args()

    workspace = ws.Workspace(
        entity=args.entity,
        project=args.project,
        name=args.name,
        sections=build_sections(),
    )
    saved = workspace.save()
    print(f"Saved view: {saved.url}")


if __name__ == "__main__":
    main()
