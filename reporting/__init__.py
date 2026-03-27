"""Reporting and visualisation utilities for AlphaBlokus.

Submodules:
    training — HTML report generation for training runs
    board    — Board state rendering (ASCII + HTML) for debugging and game replay
"""
from reporting.training import create_html_report

__all__ = ["create_html_report"]
