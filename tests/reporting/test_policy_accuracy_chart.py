"""Render-from-fixture tests for ``make_policy_accuracy_plot``."""

from __future__ import annotations

import pandas as pd
from plotly import graph_objects as go

from alphablokus.reporting.charts import make_policy_accuracy_plot


def _frozen_only() -> pd.DataFrame:
    """An older-style run: only the frozen-target agreement series."""
    return pd.DataFrame(
        {
            "generation": [1, 2, 3],
            "epoch": [0, 0, 0],
            "top1_accuracy": [0.90, 0.83, 0.70],
            "top5_accuracy": [1.00, 0.99, 0.97],
        }
    )


def _with_current_net_mcts() -> pd.DataFrame:
    df = _frozen_only()
    df["mcts_top1_accuracy"] = [0.60, 0.72, 0.81]
    df["mcts_top5_accuracy"] = [0.95, 0.97, 0.99]
    return df


def test_relabelled_title_and_frozen_series_names() -> None:
    fig = make_policy_accuracy_plot(_frozen_only(), "blokusduo")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Policy Agreement: raw net vs search (held-out set)"
    names = {trace.name for trace in fig.data}
    # The frozen series is explicitly labelled as vs gen-1 targets, not "strength".
    assert "Top-1 vs gen-1 MCTS targets" in names
    assert "Top-5 vs gen-1 MCTS targets" in names
    # No current-net MCTS series when the columns are absent.
    assert len(fig.data) == 2


def test_current_net_mcts_series_rendered_when_present() -> None:
    fig = make_policy_accuracy_plot(_with_current_net_mcts(), "blokusduo")
    names = {trace.name for trace in fig.data}
    assert "Top-1 vs current-net MCTS" in names
    assert "Top-5 vs current-net MCTS" in names
    assert len(fig.data) == 4


def test_tictactoe_frozen_series_labelled_minimax() -> None:
    fig = make_policy_accuracy_plot(_frozen_only(), "tictactoe")
    names = {trace.name for trace in fig.data}
    assert "Top-1 vs minimax oracle" in names
