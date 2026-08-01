"""The score head's target transform at the margins we actually measure.

``tanh(margin / score_scale)`` exists to spend resolution where the mass is: the real
corpora give margins from about −43 to +88 with a **median of 3**, so the transform has to
separate small margins and stop caring about blowouts. These tests pin that behaviour at
those real values rather than at abstract ones (plan
``docs/plans/score-auxiliary-target.md`` S2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphablokus.config import NetConfig
from alphablokus.training.score_target import scale_margin, scale_margins

# The shipped default; the numbers below are the plan's own worked example.
SCALE = NetConfig(
    learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=1, cuda=False, num_filters=1, num_residual_blocks=1
).score_scale


def test_default_scale_is_the_plans_25() -> None:
    assert SCALE == 25.0


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(0, 0.0), (3, 0.1194), (10, 0.3799), (25, 0.7616), (60, 0.9836), (88, 0.9982)],
)
def test_transform_matches_the_plans_worked_example(margin: int, expected: float) -> None:
    assert scale_margin(margin, SCALE) == pytest.approx(expected, abs=5e-4)


def test_small_margins_stay_separable() -> None:
    """The median margin is 3, so 1/2/3/5-point games must not collapse together.

    A raw-margin target would leave these at 1..5 against blowouts of 88; the point of
    the tanh is that they still occupy distinguishable, roughly linear ground.
    """
    small = [scale_margin(m, SCALE) for m in (1, 2, 3, 5)]
    assert small == sorted(small)
    gaps = np.diff(small)
    # Near the origin tanh is ~linear: consecutive one-point steps stay within 10% of
    # each other, so resolution is essentially uniform across the bulk of the data.
    assert gaps.min() > 0.03
    assert gaps.max() / gaps.min() < 1.1 * 2  # the 3→5 gap spans two points, hence 2x


def test_large_margins_saturate_rather_than_dominate() -> None:
    """Three points at the median move the target more than forty-five at the top.

    ``0 → 3`` (the median margin) spans ~0.12 of target while ``43 → 88`` — the measured
    extremes, 45 points apart — spans ~0.06. Under a raw-margin MSE the ratio would be
    the other way round by a factor of fifteen, and a handful of blowouts would shape a
    loss that the median-3 positions are supposed to shape.
    """
    median_step = scale_margin(3, SCALE) - scale_margin(0, SCALE)
    blowout_step = scale_margin(88, SCALE) - scale_margin(43, SCALE)

    assert blowout_step < median_step
    assert scale_margin(88, SCALE) < 1.0


def test_transform_is_odd_so_both_colours_are_treated_alike() -> None:
    for margin in (1, 3, 12, 43, 88):
        assert scale_margin(-margin, SCALE) == pytest.approx(-scale_margin(margin, SCALE))


def test_measured_range_stays_strictly_inside_the_tanh_head_range() -> None:
    """Every measured margin maps inside (-1, 1), which the ``tanh`` head can reach."""
    for margin in range(-43, 89):
        target = scale_margin(margin, SCALE)
        assert -1.0 < target < 1.0


def test_scale_margins_vectorises_and_maps_none_to_nan() -> None:
    targets = scale_margins([3, None, -10, 88], SCALE)

    assert targets.dtype == np.float32
    assert targets[0] == pytest.approx(math.tanh(3 / SCALE), abs=1e-6)
    assert np.isnan(targets[1])
    assert targets[2] == pytest.approx(math.tanh(-10 / SCALE), abs=1e-6)
    assert targets[3] == pytest.approx(math.tanh(88 / SCALE), abs=1e-6)


def test_all_none_gives_an_all_nan_array() -> None:
    assert np.isnan(scale_margins([None, None], SCALE)).all()


def test_empty_input_is_an_empty_float32_array() -> None:
    empty = scale_margins([], SCALE)
    assert empty.shape == (0,)
    assert empty.dtype == np.float32


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_scale_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="score_scale must be positive"):
        scale_margin(3, bad)
    with pytest.raises(ValueError, match="score_scale must be positive"):
        scale_margins([3], bad)


def test_score_scale_rejects_values_that_would_fail_silently() -> None:
    """A bad scale must raise, not quietly produce useless targets.

    ``<= 0`` is the obvious case and was already caught. The dangerous ones are the
    quiet failures: a huge or infinite scale maps every margin to ~0, so the head learns
    to predict zero, the diagnostics report "no skill", and the experiment's verdict
    reads "the head didn't help" rather than "the config is broken". A tiny scale
    saturates every target at exactly ±1, which a ``tanh`` head can never reach.
    """
    for bad in (0.0, -1.0, float("inf"), float("-inf"), float("nan"), 1e-6, 1e30):
        with pytest.raises(ValueError):
            scale_margin(3.0, bad)
        with pytest.raises(ValueError):
            scale_margins([3.0], bad)
    # the usable band still works, and still puts resolution on small margins
    assert scale_margin(3.0, 25.0) == pytest.approx(0.1194, abs=1e-4)
