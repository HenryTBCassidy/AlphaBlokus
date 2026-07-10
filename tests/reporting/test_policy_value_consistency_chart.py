"""Render-from-fixture tests for ``make_policy_value_consistency_plot``."""

from __future__ import annotations

import pandas as pd
from plotly import graph_objects as go

from alphablokus.reporting.charts import make_policy_value_consistency_plot


def _base() -> pd.DataFrame:
    """A minimal PVC table: two epochs at gen 1 (aggregated), then gens 2–3."""
    return pd.DataFrame(
        {
            "generation": [1, 1, 2, 3],
            "epoch": [0, 1, 0, 0],
            "pvc_argmax_match": [0.40, 0.50, 0.60, 0.70],
            "pvc_spearman": [0.10, 0.20, 0.50, 0.60],
        }
    )


def test_renders_two_series_and_bounded_axis() -> None:
    fig = make_policy_value_consistency_plot(_base())
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Policy–Value Consistency (one-ply lookahead)"
    names = {trace.name for trace in fig.data}
    assert "Argmax-match (policy top = Q₁ top)" in names
    assert "Spearman (π vs Q₁ ranking)" in names
    # No value-symmetry series when the column is absent, and no secondary axis.
    assert len(fig.data) == 2
    assert fig.layout.yaxis.range == (-1.05, 1.05)
    assert all(trace.yaxis in (None, "y") for trace in fig.data)


def test_epochs_aggregate_to_one_point_per_generation() -> None:
    fig = make_policy_value_consistency_plot(_base())
    argmax_trace = next(t for t in fig.data if t.name.startswith("Argmax"))
    # Gen 1's two epochs collapse to their mean (0.45); one x per generation.
    assert list(argmax_trace.x) == [1, 2, 3]
    assert argmax_trace.y[0] == 0.45


def test_value_symmetry_series_on_secondary_axis_when_present() -> None:
    df = _base()
    df["value_symmetry_mae"] = [0.02, 0.03, 0.01, 0.005]
    fig = make_policy_value_consistency_plot(df)
    value_trace = next(t for t in fig.data if "Value-symmetry" in t.name)
    assert value_trace.yaxis == "y2"
    assert fig.layout.yaxis2.title.text == "Value-symmetry MAE"
