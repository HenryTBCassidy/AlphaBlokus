"""Reporting and visualisation: the training-run HTML report and renderers.

Submodules:
    report        — report orchestrator (``create_html_report``): payload + assets → one HTML file
    data          — metric tables → JSON payload (signals, events, per-section series)
    arena_replays — compact replay payloads for the client-side replay browser
    assets/       — the report's CSS + JS (package data; no CDN, no build step)
    pentobi_ladder — ladder result JSON persistence
    display*      — per-game board renderers (scripts + ad-hoc tooling)
    mcts_profiling — MCTS profiling report builder
"""

from alphablokus.reporting.report import create_html_report

__all__ = ["create_html_report"]
