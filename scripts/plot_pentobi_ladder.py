"""Render the README's Pentobi-ladder chart (light + dark variants) from benchmark results.

The numbers below come from ``scripts/pentobi_benchmark.py --sweep`` runs; update ``LADDER_RESULTS``
after a new ladder and re-run this script to refresh ``docs/assets/pentobi-ladder-*.png``.

Usage:
    uv run python scripts/plot_pentobi_ladder.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from loguru import logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class LevelResult:
    """AlphaBlokus record against one Pentobi difficulty level."""

    level: int
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games


# Best net vs Pentobi levels 1-6: 40 games/level, 400 MCTS sims (2026-07, gen ~71 net, 192x12 ResNet).
LADDER_RESULTS: tuple[LevelResult, ...] = (
    LevelResult(level=1, wins=32, losses=8, draws=0),
    LevelResult(level=2, wins=30, losses=9, draws=1),
    LevelResult(level=3, wins=24, losses=14, draws=2),
    LevelResult(level=4, wins=22, losses=17, draws=1),
    LevelResult(level=5, wins=18, losses=22, draws=0),
    LevelResult(level=6, wins=8, losses=30, draws=2),
)

_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ChartTheme:
    """Ink/mark colours for one README colour scheme (background stays transparent)."""

    name: str
    bar: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    gridline: str


THEMES: tuple[ChartTheme, ...] = (
    ChartTheme(
        name="light",
        bar="#2a78d6",
        ink_primary="#0b0b0b",
        ink_secondary="#52514e",
        ink_muted="#898781",
        gridline="#e1e0d9",
    ),
    ChartTheme(
        name="dark",
        bar="#3987e5",
        ink_primary="#ffffff",
        ink_secondary="#c3c2b7",
        ink_muted="#898781",
        gridline="#2c2c2a",
    ),
)


def wilson_interval(wins: int, games: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as (lower, upper) fractions."""
    p = wins / games
    denom = 1.0 + z**2 / games
    centre = (p + z**2 / (2 * games)) / denom
    half_width = (z / denom) * math.sqrt(p * (1 - p) / games + z**2 / (4 * games**2))
    return centre - half_width, centre + half_width


def _draw(ax: Axes, theme: ChartTheme) -> None:
    levels = [r.level for r in LADDER_RESULTS]
    rates = [100 * r.win_rate for r in LADDER_RESULTS]
    intervals = [wilson_interval(r.wins, r.games) for r in LADDER_RESULTS]
    err_low = [rate - 100 * lo for rate, (lo, _) in zip(rates, intervals, strict=True)]
    err_high = [100 * hi - rate for rate, (_, hi) in zip(rates, intervals, strict=True)]

    ax.bar(levels, rates, width=0.62, color=theme.bar, zorder=3)
    ax.errorbar(
        levels,
        rates,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor=theme.ink_secondary,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )
    for level, rate, (_, hi) in zip(levels, rates, intervals, strict=True):
        ax.annotate(
            f"{rate:.0f}%",
            (level, 100 * hi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10.5,
            color=theme.ink_primary,
            fontweight="bold",
        )

    ax.axhline(50, color=theme.ink_muted, linewidth=1.0, linestyle=(0, (4, 4)), zorder=2)
    ax.annotate(
        "50% = even",
        (7.0, 50),
        textcoords="offset points",
        xytext=(0, 5),
        va="bottom",
        ha="right",
        fontsize=9,
        color=theme.ink_secondary,
    )

    ax.set_xlim(0.45, 7.1)
    ax.set_ylim(0, 100)
    ax.set_xticks(levels)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Pentobi difficulty level (ladder runs 1–9)", fontsize=10.5, color=theme.ink_secondary)
    ax.tick_params(colors=theme.ink_muted, labelsize=10, length=0)
    ax.grid(axis="y", color=theme.gridline, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(theme.gridline)

    ax.set_title(
        "AlphaBlokus win rate vs Pentobi, by difficulty level",
        loc="left",
        fontsize=13,
        color=theme.ink_primary,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.02,
        "40 games per level · 400 MCTS simulations · error bars are 95% Wilson intervals",
        transform=ax.transAxes,
        fontsize=9.5,
        color=theme.ink_secondary,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        _draw(ax, theme)
        fig.tight_layout()
        out_path = out_dir / f"pentobi-ladder-{theme.name}.png"
        fig.savefig(out_path, transparent=True)
        plt.close(fig)
        logger.info("Wrote {}", out_path)


if __name__ == "__main__":
    main()
