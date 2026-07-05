"""Render-from-fixture test for the pool Elo chart (``make_tournament_elo_plot``)."""

from __future__ import annotations

import pandas as pd
from plotly import graph_objects as go

from alphablokus.reporting.charts import make_tournament_elo_plot


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "generation": [0, 1, 2, 3],
            "rating": [0.0, 120.0, 245.0, 330.0],
            "n_games": [90, 120, 120, 90],
            "n_pairings": [3, 4, 4, 3],
        }
    )


def test_returns_figure_with_single_trace() -> None:
    fig = make_tournament_elo_plot(_fixture())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Pool Elo (BayesElo tournament)"


def test_plots_ratings_in_generation_order() -> None:
    # Shuffled input should still plot ascending by generation.
    shuffled = _fixture().sample(frac=1.0, random_state=0)
    fig = make_tournament_elo_plot(shuffled)
    trace = fig.data[0]
    assert list(trace.x) == [0, 1, 2, 3]
    assert list(trace.y) == [0.0, 120.0, 245.0, 330.0]
