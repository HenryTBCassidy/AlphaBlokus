"""Build the AlphaBlokus W&B workspace via the `wandb-workspaces` SDK.

Run this once per W&B project (e.g. after the first run, or when the project
gets reset). The script creates a saved view named "AlphaBlokus default".
Re-running is safe — the SDK upserts by view name.

Usage::

    uv run python scripts/setup_wandb_workspace.py \
        --entity henrycassidy --project alphablokus-poc

Design (2026-08 cut-down). The dashboard is what gets watched *mid-flight*,
and the previous 30-panel layout buried the signal: during the
``blokus_paired_gate_rerun`` regression every prominent panel (loss ↓,
acceptance 100%, eval top-1 0.99) looked healthy while the only honest
warnings — policy symmetry KL and value symmetry MAE, computed against the
game's ground-truth invariances — sat in an unwatched parquet
(docs/research/regression-and-next-steps.md §1.5). So:

* The **first, open section** is exclusively externally-anchored health:
  symmetry KL, value symmetry MAE, self-play target entropy (the gen-17
  collapse detector) and the arena colour split (is the gate measuring
  strength, or first-mover advantage?).
* Strength/gate telemetry and training loss follow, explicitly labelled
  self-referential — useful trends, but they cannot certify progress.
* Progress (generation / ETA) stays for glanceability; everything
  operational is collapsed at the bottom.
* Cut entirely: per-episode duplicates of per-gen panels, epoch/episode/batch
  sawtooths, top-5 agreement, minimax (TTT-only), the retired frozen-gen-0
  Elo keys — clutter that nobody watched and that crowded out the warnings.

Each panel's metric matches the ``define_metric`` namespaces wired in
``storage/metrics.py``.
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
    """``wr.LinePlot`` with the dashboard's defaults: explicit x-axis,
    optional exponential smoothing, optional y-range."""
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


def build_sections() -> list[ws.Section]:
    """The full ordered section list: honest warnings first, ops last."""
    return [
        ws.Section(
            name="Is it improving? — externally anchored",
            is_open=True,
            panels=[
                _line(
                    "Policy symmetry KL (rising = drifting off game invariances)",
                    ["learning_quality/symmetry_kl_mean", "learning_quality/symmetry_kl_max"],
                ),
                _line(
                    "Value symmetry MAE (rising = value head drifting)",
                    ["pvc/value_symmetry_mae"],
                ),
                _line(
                    "Self-play target entropy (a cliff = target collapse, e.g. rerun gen 17)",
                    ["self_play_per_gen/policy_entropy_mean"],
                ),
                _line(
                    "Arena white-win share of decisive games (≥0.85 = colour-pinned gate)",
                    ["arena/white_win_rate"],
                    y_range=(0.0, 1.05),
                ),
            ],
        ),
        ws.Section(
            name="Strength & gate telemetry — self-referential",
            is_open=True,
            panels=[
                _line(
                    "Rolling arena-derived Elo (chained; trend only)",
                    ["elo/rolling"],
                ),
                _line(
                    "Arena score vs incumbent (near-equal nets pin to ~0.50)",
                    ["elo/score_rate"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Accepted (per gen, 0/1) + running acceptance rate",
                    ["arena/accepted", "arena/acceptance_rate"],
                    y_range=(-0.05, 1.05),
                ),
            ],
        ),
        ws.Section(
            name="Training",
            is_open=True,
            panels=[
                _line(
                    "Per-gen losses (total / pi / v)",
                    [
                        "training_per_gen/total_loss",
                        "training_per_gen/pi_loss",
                        "training_per_gen/v_loss",
                    ],
                ),
                _line(
                    "Per-batch loss (smoothed)",
                    [
                        "training/total_loss",
                        "training/pi_loss",
                        "training/v_loss",
                    ],
                    x="global_batch",
                    smoothing=0.8,
                ),
                _line(
                    "Learning rate actually applied",
                    ["training_per_gen/learning_rate"],
                ),
            ],
        ),
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
                    "Self-play search throughput (sims/sec, per-gen mean)",
                    ["self_play_per_gen/sims_per_second_mean"],
                ),
                _line(
                    "Self-play inference fraction",
                    ["self_play_per_gen/inference_fraction_mean"],
                    y_range=(0.0, 1.05),
                ),
                _line(
                    "Replay buffer fill + emergent reuse",
                    [
                        "training_per_gen/buffer_fill_fraction",
                        "training_per_gen/emergent_reuse",
                    ],
                ),
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        required=True,
        help="W&B entity (user or team) that owns the project",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project name (e.g. alphablokus-poc)",
    )
    parser.add_argument(
        "--name",
        default="AlphaBlokus default",
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
