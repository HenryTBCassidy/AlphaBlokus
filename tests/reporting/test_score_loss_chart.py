"""The loss charts pick up the auxiliary score series — and only when it exists.

Runs predating the score head (and every run with it off) write no ``score_loss``
column, so the charts must render unchanged for them; a run with the head on gets one
extra series in each of the two loss charts (plan ``docs/plans/score-auxiliary-target.md``
S4).
"""

from __future__ import annotations

import pandas as pd

from alphablokus.reporting.charts import make_loss_per_generation, make_loss_timeline

_SCORE_SERIES = "Score (auxiliary)"


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "generation": [1, 1, 2, 2],
            "epoch": [0, 0, 0, 0],
            "batch_number": [0, 1, 0, 1],
            "pi_loss": [9.0, 8.6, 8.2, 8.0],
            "v_loss": [0.9, 0.85, 0.8, 0.78],
            "total_loss": [9.9, 9.45, 9.0, 8.78],
        }
    )


def _with_score_loss() -> pd.DataFrame:
    df = _training_frame()
    df["score_loss"] = [0.60, 0.55, 0.48, 0.44]
    df["total_loss"] = df["pi_loss"] + df["v_loss"] + 0.15 * df["score_loss"]
    return df


def test_no_score_series_for_a_run_without_the_head() -> None:
    for figure in (make_loss_per_generation(_training_frame()), make_loss_timeline(_training_frame())):
        assert _SCORE_SERIES not in {trace.name for trace in figure.data}


def test_score_series_appears_when_the_column_is_present() -> None:
    per_generation = make_loss_per_generation(_with_score_loss())
    timeline = make_loss_timeline(_with_score_loss())

    assert _SCORE_SERIES in {trace.name for trace in per_generation.data}
    assert _SCORE_SERIES in {trace.name for trace in timeline.data}


def test_an_all_null_score_column_is_treated_as_absent() -> None:
    """A mixed-history run (head switched on mid-way) must not draw an empty series."""
    df = _training_frame()
    df["score_loss"] = [None, None, None, None]

    assert _SCORE_SERIES not in {trace.name for trace in make_loss_per_generation(df).data}
