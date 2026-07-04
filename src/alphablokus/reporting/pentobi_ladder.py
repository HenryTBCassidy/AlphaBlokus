"""Pentobi ladder results: JSON persistence + the report section (cloud-scale C11).

``scripts/pentobi_benchmark.py`` writes one JSON per benchmark run into the
run's ``PentobiLadder/`` directory; ``create_html_report`` renders whatever it
finds there as a "Pentobi Ladder" section. Both sides share this module so the
schema can't drift. No results directory → empty section → report unchanged.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def parse_levels(spec: str) -> list[int]:
    """Parse a levels spec — ``"1-5"``, ``"1,3,9"``, or a mix (``"1-3,9"``)."""
    levels: list[int] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        start, _, end = token.partition("-")
        if end:
            levels.extend(range(int(start), int(end) + 1))
        else:
            levels.append(int(start))
    if not levels or not all(1 <= level <= 9 for level in levels):
        raise ValueError(f"Bad levels spec {spec!r}: Pentobi levels are 1-9 (e.g. '1-5' or '1,3,9').")
    return levels


def write_ladder_result(
    directory: Path,
    *,
    net: str,
    sims: int,
    games_per_level: int,
    per_level: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> Path:
    """Persist one benchmark run's ladder results for the HTML report.

    ``per_level`` rows are the benchmark's stats dicts (any ``records`` key —
    the raw game replays — is dropped; replays belong to the benchmark's own
    HTML report, not the training report).
    """
    timestamp = datetime.now(UTC)
    payload = {
        "net": net,
        "sims": sims,
        "games_per_level": games_per_level,
        "timestamp": timestamp.isoformat(),
        "levels": [{k: v for k, v in row.items() if k != "records"} for row in per_level],
        "metrics": metrics,
    }
    directory.mkdir(parents=True, exist_ok=True)
    safe_net = net.replace("/", "_")
    path = directory / f"ladder_{safe_net}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_ladder_results(directory: Path) -> list[dict[str, Any]]:
    """All ladder result payloads in ``directory``, oldest first."""
    if not directory.exists():
        return []
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("ladder_*.json"))]
    return sorted(results, key=lambda r: str(r.get("timestamp", "")))


def build_pentobi_ladder_section(directory: Path) -> str:
    """The report's "Pentobi Ladder" section, or ``""`` when no results exist."""
    results = load_ladder_results(directory)
    if not results:
        return ""

    header_levels = sorted({row["level"] for result in results for row in result["levels"]})
    head_cells = "".join(f"<th>L{level}</th>" for level in header_levels)
    body_rows = []
    for result in results:
        by_level = {row["level"]: row for row in result["levels"]}
        cells = []
        for level in header_levels:
            row = by_level.get(level)
            if row is None:
                cells.append("<td>—</td>")
                continue
            beat = row["win_rate"] > 0.5
            cells.append(
                f'<td class="{"ladder-beat" if beat else "ladder-lost"}">'
                f"{row['win_rate']:.0%} ({row['net_wins']}-{row['pentobi_wins']}-{row['draws']})</td>"
            )
        timestamp = str(result.get("timestamp", ""))[:16].replace("T", " ")
        body_rows.append(
            f"<tr><td>{result['net']}</td><td>{timestamp}</td>"
            f"<td>{result['metrics']['pentobi_level']}</td>{''.join(cells)}</tr>"
        )

    return f"""<section>
<h2>Pentobi Ladder</h2>
<p class="section-desc">
    Checkpoints benchmarked against pentobi-gtp (scripts/pentobi_benchmark.py).
    "Pentobi Level" = highest level beaten at &gt;50% win rate; cells show
    win rate (W-L-D). Green = beaten, red = not yet.
</p>
<style>
.ladder-beat {{ background: #dcfce7; }}
.ladder-lost {{ background: #fee2e2; }}
</style>
<table class="ladder-table">
<tr><th>net</th><th>when</th><th>Pentobi Level</th>{head_cells}</tr>
{"".join(body_rows)}
</table>
</section>"""
