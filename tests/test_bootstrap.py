"""Tests for the game-cluster bootstrap.

The load-bearing test here is :func:`test_cluster_bootstrap_covers_the_truth`,
which checks the interval against a synthetic population whose true sampling
distribution is known analytically. Its companion,
:func:`test_position_bootstrap_undercovers_the_same_data`, runs the naive
position-level bootstrap on identical data and shows it fails the same check —
without that second test, a coverage assertion could pass for the wrong reason
(e.g. an interval that is simply too wide).
"""

from __future__ import annotations

import numpy as np
import pytest

from alphablokus.bootstrap import BootstrapResult, game_cluster_bootstrap

# --- Synthetic population -------------------------------------------------
#
# ``n_games`` games, each contributing ``per_game`` positions that all carry the
# *same* value — the structure of a real eval set, where every position in a
# game shares the game's outcome label. Game values are i.i.d. standard normal,
# so the sample mean over the positions equals the mean over the game values:
# its standard error is ``1/sqrt(n_games)``, independent of ``per_game``.
#
# That is the fact the cluster bootstrap must reproduce and the position
# bootstrap cannot: the latter sees ``n_games * per_game`` rows and reports an
# interval about ``sqrt(per_game)`` times too narrow.

TRUE_MEAN = 0.0


def _draw_population(rng: np.random.Generator, n_games: int, per_game: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, game_ids)`` for one synthetic eval set."""
    game_values = rng.normal(loc=TRUE_MEAN, scale=1.0, size=n_games)
    values = np.repeat(game_values, per_game)
    game_ids = np.repeat(np.arange(n_games), per_game)
    return values, game_ids


def _position_bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """The WRONG bootstrap, for contrast: resample positions, ignoring games.

    Deliberately lives in the test file and not in the shipped module — its only
    purpose is to demonstrate the failure the cluster bootstrap exists to avoid.
    """
    rng = np.random.default_rng(seed)
    n = values.size
    means = values[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return float(lo), float(hi)


def test_point_estimate_is_the_statistic_on_the_full_sample() -> None:
    values = np.array([1.0, 1.0, -1.0, -1.0, 0.5, 0.5])
    game_ids = np.array([0, 0, 1, 1, 2, 2])

    result = game_cluster_bootstrap(
        lambda idx: float(values[idx].mean()),
        game_ids,
        n_resamples=50,
        seed=0,
    )

    assert result.point == pytest.approx(values.mean())
    assert result.n_games == 3
    assert result.n_positions == 6
    assert result.positions_per_game == pytest.approx(2.0)


def test_resamples_are_unions_of_whole_games() -> None:
    """Each resample must be built from whole games, never individual positions."""
    game_ids = np.array([0, 0, 0, 1, 1, 2])
    groups = {0: {0, 1, 2}, 1: {3, 4}, 2: {5}}
    seen: list[np.ndarray] = []

    def statistic(idx: np.ndarray) -> float:
        seen.append(idx)
        return 0.0

    game_cluster_bootstrap(statistic, game_ids, n_resamples=40, seed=1)

    # First call is the point estimate over every position.
    assert sorted(seen[0].tolist()) == [0, 1, 2, 3, 4, 5]
    for idx in seen[1:]:
        counts = {game: 0 for game in groups}
        for position in idx.tolist():
            for game, members in groups.items():
                if position in members:
                    counts[game] += 1
        # A game appears 0, 1, 2, ... times, and when it appears its positions
        # appear together the same number of times — never a strict subset.
        for game, members in groups.items():
            assert counts[game] % len(members) == 0
        # Exactly n_games games are drawn, so the total is a sum of whole games.
        assert sum(counts[game] // len(members) for game, members in groups.items()) == len(groups)


def test_cluster_bootstrap_covers_the_truth() -> None:
    """95% intervals must contain the true mean about 95% of the time.

    The synthetic population's sampling distribution is known: the sample mean
    has standard error ``1/sqrt(n_games)``. This checks the bootstrap recovers
    that, rather than the ``1/sqrt(n_games * per_game)`` a position-level
    resample would give.
    """
    n_trials, n_games, per_game = 200, 30, 8
    rng = np.random.default_rng(12345)
    covered = 0
    widths: list[float] = []

    for trial in range(n_trials):
        values, game_ids = _draw_population(rng, n_games, per_game)
        result = game_cluster_bootstrap(
            lambda idx, v=values: float(v[idx].mean()),
            game_ids,
            n_resamples=300,
            seed=trial,
        )
        if result.lo <= TRUE_MEAN <= result.hi:
            covered += 1
        widths.append(result.hi - result.lo)

    coverage = covered / n_trials
    # Percentile bootstrap on 30 clusters under-covers slightly (no t-correction),
    # so allow a little slack below nominal — but it must be near 0.95, and the
    # companion test shows the naive alternative lands nowhere near this band.
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f} is not ~0.95"

    # Width must track the true 95% interval, 2 * 1.96 / sqrt(n_games).
    expected_width = 2 * 1.96 / np.sqrt(n_games)
    assert np.mean(widths) == pytest.approx(expected_width, rel=0.20)


def test_position_bootstrap_undercovers_the_same_data() -> None:
    """The naive bootstrap fails the coverage check the cluster one passes.

    This is what makes the coverage test above meaningful: on identical data,
    ignoring the game clustering collapses coverage from ~95% to ~50% and the
    interval to roughly ``1/sqrt(per_game)`` of its correct width.
    """
    n_trials, n_games, per_game = 200, 30, 8
    rng = np.random.default_rng(12345)
    covered = 0
    cluster_widths: list[float] = []
    position_widths: list[float] = []

    for trial in range(n_trials):
        values, game_ids = _draw_population(rng, n_games, per_game)
        lo, hi = _position_bootstrap_ci(values, n_resamples=300, seed=trial)
        if lo <= TRUE_MEAN <= hi:
            covered += 1
        position_widths.append(hi - lo)
        cluster = game_cluster_bootstrap(
            lambda idx, v=values: float(v[idx].mean()),
            game_ids,
            n_resamples=300,
            seed=trial,
        )
        cluster_widths.append(cluster.hi - cluster.lo)

    coverage = covered / n_trials
    assert coverage < 0.75, f"position bootstrap coverage {coverage:.3f} — expected badly low"

    # The narrowing factor is ~sqrt(per_game): this is the "intervals are about
    # 2x too narrow" claim made concrete and measured rather than asserted.
    ratio = float(np.mean(cluster_widths) / np.mean(position_widths))
    assert ratio == pytest.approx(np.sqrt(per_game), rel=0.25), f"width ratio {ratio:.2f}"


def test_uncorrelated_positions_give_the_same_answer_either_way() -> None:
    """With one position per game there is no clustering, so both agree.

    Guards against the cluster bootstrap being systematically too wide: when the
    clustering is absent it must reproduce the ordinary bootstrap.
    """
    rng = np.random.default_rng(7)
    values = rng.normal(size=240)
    game_ids = np.arange(240)  # one position per game

    cluster = game_cluster_bootstrap(
        lambda idx: float(values[idx].mean()),
        game_ids,
        n_resamples=800,
        seed=3,
    )
    lo, hi = _position_bootstrap_ci(values, n_resamples=800, seed=3)

    assert (cluster.hi - cluster.lo) == pytest.approx(hi - lo, rel=0.15)


def test_non_finite_resamples_are_dropped() -> None:
    """A statistic that is undefined for some resamples still yields an interval."""
    game_ids = np.repeat(np.arange(12), 4)
    values = np.repeat(np.linspace(-1.0, 1.0, 12), 4)

    def statistic(idx: np.ndarray) -> float:
        # Undefined whenever the resample happens to miss game 0 entirely.
        if 0 not in set(game_ids[idx].tolist()):
            return float("nan")
        return float(values[idx].mean())

    result = game_cluster_bootstrap(statistic, game_ids, n_resamples=400, seed=2)

    assert result.n_valid_resamples < result.n_resamples
    assert np.isfinite(result.lo) and np.isfinite(result.hi)


def test_statistic_undefined_too_often_raises() -> None:
    game_ids = np.repeat(np.arange(8), 3)

    with pytest.raises(ValueError, match="finite statistic"):
        game_cluster_bootstrap(lambda idx: float("nan"), game_ids, n_resamples=100, seed=0)


def test_same_seed_reproduces_the_interval() -> None:
    values, game_ids = _draw_population(np.random.default_rng(0), 20, 5)

    first = game_cluster_bootstrap(lambda idx: float(values[idx].mean()), game_ids, n_resamples=200, seed=9)
    second = game_cluster_bootstrap(lambda idx: float(values[idx].mean()), game_ids, n_resamples=200, seed=9)

    assert (first.lo, first.hi) == (second.lo, second.hi)


@pytest.mark.parametrize(
    ("game_ids", "kwargs", "match"),
    [
        (np.zeros((2, 2)), {}, "must be 1-D"),
        (np.array([], dtype=int), {}, "empty"),
        (np.array([0, 1]), {"confidence": 0.0}, "confidence"),
        (np.array([0, 1]), {"confidence": 1.0}, "confidence"),
        (np.array([0, 1]), {"n_resamples": 0}, "n_resamples"),
    ],
)
def test_invalid_arguments_raise(game_ids: np.ndarray, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        game_cluster_bootstrap(lambda idx: 0.0, game_ids, **kwargs)


def test_payload_uses_the_report_interval_convention() -> None:
    result = BootstrapResult(
        point=0.102401,
        lo=-0.150456,
        hi=0.092301,
        confidence=0.95,
        n_games=47,
        n_positions=200,
        n_valid_resamples=2000,
        n_resamples=2000,
    )

    assert result.as_payload() == {
        "point": 0.1024,
        "ci": [-0.1505, 0.0923],
        "n_games": 47,
        "n_positions": 200,
    }
