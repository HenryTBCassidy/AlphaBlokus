"""End-of-run report: a single self-contained HTML file per run.

``create_html_report`` reduces the run's metric tables to one JSON payload
(:mod:`alphablokus.reporting.data`) and embeds it in an HTML shell together
with the report's CSS and JS (shipped as package data under ``assets/``). The
page renders everything client-side — hand-rolled SVG charts, the arena replay
browser, light/dark theming — with no CDN, no frameworks and no build step, so
the file opens offline from ``file://`` and survives being copied around.

The front page answers one question: *is this run improving, or fooling
itself?* Externally-anchored signals (Pentobi ladder, pooled BayesElo,
symmetry diagnostics, target entropy) are promoted above — and visually
separated from — the self-referential training telemetry that once masked a
44-Elo regression (docs/research/regression-and-next-steps.md §1.5).
"""

from __future__ import annotations

import json
import time
from importlib import resources
from typing import TYPE_CHECKING

from loguru import logger

from alphablokus.reporting.data import build_report_payload

if TYPE_CHECKING:
    from alphablokus.config import RunConfig

_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<noscript>This report renders client-side — enable JavaScript to view it.</noscript>
<script id="report-data" type="application/json">
{payload}
</script>
<script>
{js}
</script>
</body>
</html>
"""


def _asset(name: str) -> str:
    """Read a bundled report asset (CSS/JS shipped as package data)."""
    return resources.files("alphablokus.reporting").joinpath(f"assets/{name}").read_text(encoding="utf-8")


def create_html_report(config: RunConfig) -> None:
    """Generate the run's interactive HTML report.

    Reads every metric table the run directory holds (each one optional),
    assembles the JSON payload, and writes a single self-contained
    ``Reporting/report.html``. Crash-safe by design: the caller (``cli.main``)
    already wraps this in a try/except so a rendering failure can never sink a
    finished run.

    Args:
        config: The run configuration used for this training session.
    """
    logger.info("Writing report...")
    start = time.perf_counter()

    payload = build_report_payload(config)
    # ``</`` must not appear inside an inline <script> block (it would
    # terminate the tag early); escape it inside JSON strings.
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    html = _HTML_SHELL.format(
        title=f"AlphaBlokus — {config.run_name}",
        css=_asset("report.css"),
        payload=payload_json,
        js=_asset("report.js"),
    )

    filename = config.report_directory / "report.html"
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename.write_text(html, encoding="utf-8")

    elapsed = time.perf_counter() - start
    logger.info("Wrote report to {} in {:.2f}s ({:.1f} MB)", filename, elapsed, len(html) / 1024**2)
