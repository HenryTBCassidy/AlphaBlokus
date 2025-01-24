import logging
import time

import pandas as pd
import plotly.express as px
from pandas import DataFrame
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from core.config import LOGGER_NAME
from core.config import RunConfig

log = logging.getLogger(LOGGER_NAME)


def _make_loss_curves(df: DataFrame) -> tuple[go.Figure, go.Figure]:
    df_plot = df.sort_values(["generation", "epoch", "batch_number"]).astype(
        {"epoch": "category", "generation": "category"})
    agg_df_plot = df_plot.groupby("generation").agg({"average_loss": "last"}).reset_index()

    fig1 = px.line(
        agg_df_plot, x="generation", y="average_loss"
    )

    fig2 = px.scatter(
        df_plot, x="batch_number", y="average_loss", color="epoch", hover_data=["generation"],
        labels={"x": "Batch Number", "y": "Average Loss"}
    )

    fig1.update_layout(
        width=1200,
        height=600,
        xaxis_title="Generation",
        yaxis_title="Average Loss",
        title="Average Total Pi And V Loss per Generation",
    )

    fig2.update_traces(marker_size=2)
    fig2.update_layout(
        width=1200,
        height=600,
        yaxis_title="Average Loss",
        xaxis_title="Batch Number",
        title="Average Total Pi And V Loss As A Function Of Batch Number For Each Generation And Epoch",
    )

    return fig1, fig2


def _make_arena_data_plot(arena_data: pd.DataFrame) -> go.Figure:
    arena_data = arena_data.assign(**{
        "Percentage Losses": lambda df: 100 * df.losses / (df.wins + df.losses + df.draws),
        "Percentage Wins": lambda df: 100 * df.wins / (df.wins + df.losses + df.draws)
    }).sort_values(["generation"]).astype({"generation": "category"})

    fig = px.bar(
        arena_data, x="generation", y=["Percentage Wins", "Percentage Losses"],
        labels={"x": "Generation", "y": "Percentage Value"}
    )

    fig.update_layout(
        width=1200,
        height=600,
        yaxis_title="Percentage Value",
        xaxis_title="Generation",
        title="Win & Loss Rate Of New Net Against Predecessor",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def _make_performance_statistics_plot(timings_data: pd.DataFrame) -> go.Figure:
    cycles = timings_data.cycle_stage.unique()
    fig = make_subplots(cols=len(cycles), shared_yaxes=True, subplot_titles=cycles, x_title="Generation",
                        y_title="Time Elapsed / S")

    for i, cycle in enumerate(cycles):
        _filter = lambda df: df.cycle_stage == cycle
        fig.add_trace(
            go.Bar(
                x=timings_data[_filter]["generation"], y=timings_data[_filter]["time_elapsed"], name=cycle
            ), row=1, col=i + 1
        )

    fig.update_layout(height=600, width=1200,
                      title_text=f"Performance Statistics For run")

    return fig


def create_html_report(args: RunConfig):
    log.info(f"Writing report...")
    start = time.perf_counter()
    loss_data = pd.read_parquet(args.training_data_directory / "data.parquet")
    arena_data = pd.read_parquet(args.arena_data_directory)
    timings_data = pd.read_parquet(args.timings_directory)

    fig1, fig2 = _make_loss_curves(loss_data)
    fig3 = _make_arena_data_plot(arena_data)
    fig4 = _make_performance_statistics_plot(timings_data)

    filename = args.report_directory / "report.html"
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, 'w') as f:
        f.write(fr"""
            <html><head><title> Output Data in an HTML file \ 
            </title></head><body><h1> <u>AlphaBlokus</u> Report </h1> 
            <h2>For run: '{args.run_name}' </h2> </body></html>"""
                )
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))

    end = time.perf_counter()
    log.info(f"Wrote report in {end-start} Seconds!")
