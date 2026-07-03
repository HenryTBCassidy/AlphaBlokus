"""Reporting and visualisation: the training-run HTML report and renderers.

Submodules:
    report        — report orchestrator (``create_html_report``) + metrics loading
    charts        — plotly figure builders
    arena_replays — interactive replay viewer (template + section builder)
    display*      — per-game board renderers
    mcts_profiling — MCTS profiling report builder
"""

from alphablokus.reporting.report import create_html_report

__all__ = ["create_html_report"]
