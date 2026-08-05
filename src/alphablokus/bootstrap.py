"""Game-cluster bootstrap for eval-set diagnostics.

Every eval-set diagnostic in this project is computed over positions, but the
positions are not independent: they are sampled from a handful of self-play
games, and **every position in a game carries the same outcome label** (see
:func:`alphablokus.selfplay.episode.play_self_play_episode`, which stamps one
``game_result`` onto every position of the game). Symmetry augmentation then
duplicates each position, so a "200-position" eval set can hold far fewer
independent observations than its length suggests.

Treating those positions as independent draws understates every confidence
interval. The error is not small and it is not conservative: for a statistic
whose value is fixed within a game, the naive position-level interval is
narrower than the truth by roughly ``sqrt(positions per game)``. Reading a kill
criterion through such an interval produces plausible-looking wrong answers —
which is precisely how a run's dashboards can read healthy while its real
strength falls.

The fix is the standard **cluster bootstrap**: resample whole *games* with
replacement (not positions), recompute the statistic on the pooled positions of
the resampled games, and read percentiles off that distribution. The number of
games is the sample size; the positions within a game are one observation
smeared over several rows.

``tests/training/test_bootstrap.py`` verifies this against a synthetic
population whose true sampling distribution is known analytically, and
simultaneously demonstrates that the naive position-level bootstrap
under-covers on the same data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Enough resamples that the percentile endpoints are stable to ~0.003 for a
# statistic on [-1, 1], cheap enough to run every generation on 200 positions.
DEFAULT_RESAMPLES = 2_000

# A resample can leave a statistic undefined — e.g. a colour-conditional
# baseline when every drawn game happens to be White-to-move only. Those
# resamples are dropped, but if most of them fail the interval is meaningless
# and we say so rather than quietly reporting a CI over a biased subset.
MIN_VALID_FRACTION = 0.5


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a game-cluster bootstrap confidence interval.

    Attributes:
        point: The statistic evaluated once on the full sample. This is the
            number to report — the bootstrap distribution supplies the interval,
            not the estimate.
        lo: Lower confidence bound (percentile method).
        hi: Upper confidence bound.
        confidence: Nominal coverage the bounds were computed for, e.g. 0.95.
        n_games: Distinct source games — **this is the effective sample size**,
            and the reason the interval is wider than a position-level one.
        n_positions: Positions the statistic was computed over.
        n_valid_resamples: Resamples that produced a finite statistic. Compare
            against ``n_resamples`` to see how often the statistic was
            undefined; a large shortfall means the interval is fragile.
        n_resamples: Resamples attempted.
    """

    point: float
    lo: float
    hi: float
    confidence: float
    n_games: int
    n_positions: int
    n_valid_resamples: int
    n_resamples: int

    @property
    def positions_per_game(self) -> float:
        """Mean positions per source game — the clustering factor."""
        if self.n_games == 0:
            return 0.0
        return self.n_positions / self.n_games

    def as_payload(self) -> dict[str, object]:
        """Serialise for the report payload.

        Uses the report's existing interval convention: a 2-element ``ci`` list
        (see ``reporting/data.py``'s ladder payload), so the JS side can render
        it with the same helper as a Wilson interval.
        """
        return {
            "point": round(self.point, 4),
            "ci": [round(self.lo, 4), round(self.hi, 4)],
            "n_games": self.n_games,
            "n_positions": self.n_positions,
        }


def game_cluster_bootstrap(
    statistic: Callable[[NDArray], float],
    game_ids: NDArray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap ``statistic`` by resampling whole games with replacement.

    Args:
        statistic: Called with an array of **position indices** into the sample
            and returns one float. Written this way so the caller closes over
            whatever per-position arrays it needs (predictions, targets,
            colours, phases) and the bootstrap stays agnostic to them. A
            resample may legitimately leave the statistic undefined; return
            ``nan`` (or any non-finite value) and it will be dropped.
        game_ids: Per-position source game id, one entry per position. Ids are
            opaque — any hashable integer labelling works, gaps included.
        n_resamples: Number of bootstrap resamples.
        confidence: Nominal coverage, e.g. 0.95 for a 95% interval.
        seed: Seed for the resampling RNG, so a generation's reported interval
            is reproducible.

    Returns:
        The point estimate and its percentile interval.

    Raises:
        ValueError: If ``game_ids`` is not a non-empty 1-D array, if
            ``confidence`` is not in (0, 1), if ``n_resamples`` is not positive,
            or if too few resamples produced a finite statistic to form an
            interval (see :data:`MIN_VALID_FRACTION`).
    """
    ids = np.asarray(game_ids)
    if ids.ndim != 1:
        raise ValueError(f"game_ids must be 1-D, got shape {ids.shape!r}")
    if ids.size == 0:
        raise ValueError("game_ids is empty — nothing to bootstrap")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples!r}")

    # Group position indices by source game. ``np.unique`` gives a stable order,
    # so the same inputs and seed always produce the same interval.
    unique_ids = np.unique(ids)
    groups = [np.flatnonzero(ids == game_id) for game_id in unique_ids]
    n_games = len(groups)

    point = float(statistic(np.arange(ids.size)))

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        # Draw n_games games *with replacement*. A game drawn twice contributes
        # its positions twice — that is the whole point: the resampling
        # granularity is the game, so within-game correlation cannot shrink the
        # interval.
        drawn = rng.integers(0, n_games, size=n_games)
        indices = np.concatenate([groups[d] for d in drawn])
        value = float(statistic(indices))
        if np.isfinite(value):
            samples.append(value)

    if len(samples) < max(2, int(MIN_VALID_FRACTION * n_resamples)):
        raise ValueError(
            f"only {len(samples)} of {n_resamples} resamples produced a finite statistic "
            f"({n_games} games, {ids.size} positions) — the statistic is undefined too "
            "often for a bootstrap interval to mean anything. Widen the sample or report "
            "the point estimate without an interval."
        )

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(samples, [100.0 * tail, 100.0 * (1.0 - tail)])
    return BootstrapResult(
        point=point,
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        n_games=n_games,
        n_positions=int(ids.size),
        n_valid_resamples=len(samples),
        n_resamples=n_resamples,
    )
