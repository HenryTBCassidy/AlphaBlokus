"""Pentobi ladder results: JSON persistence (cloud-scale C11).

``scripts/pentobi_benchmark.py`` and ``scripts/mini_ladder.py`` write one JSON
per benchmark run into the run's ``PentobiLadder/`` directory; the report's
payload builder (``reporting.data.ladder_payload``) reads whatever it finds
there. Both sides share this module so the schema can't drift. No results
directory → no ladder section in the report.
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
