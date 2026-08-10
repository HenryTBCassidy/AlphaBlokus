"""Pentobi ladder results: JSON persistence (cloud-scale C11).

``scripts/pentobi_benchmark.py`` and ``scripts/mini_ladder.py`` write one JSON
per benchmark run into the run's ``PentobiLadder/`` directory; the report's
payload builder (``reporting.data.ladder_payload``) reads whatever it finds
there. Both sides share this module so the schema can't drift. No results
directory → no ladder section in the report.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

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
    duration_s: float | None = None,
    condition: str = "ladder",
    context: dict[str, Any] | None = None,
) -> Path:
    """Persist one benchmark run's ladder results for the HTML report.

    ``per_level`` rows are the benchmark's stats dicts (any ``records`` key —
    the raw game replays — is dropped; replays belong to the benchmark's own
    HTML report, not the training report).

    ``condition`` names the instrument: ``"ladder"`` for the longitudinal
    fixed-400/book-free series that Coach reads for promotion and drift, anything
    else for a one-off comparison. ``context`` records the full comparison setup
    (both sides' search settings, Pentobi's book/threads, seeds, hardware) so a
    payload is self-describing instead of relying on the reader's memory.

    ``duration_s`` is the ladder's wall-clock cost. The ladder is the backbone
    measurement of the current plan — it is the only instrument that has ever
    resolved a difference the arena called a tie — and until now its cost was
    recorded nowhere on disk, which made scheduling everything downstream of it
    guesswork. Optional so older readers and callers keep working.
    """
    timestamp = datetime.now(UTC)
    payload: dict[str, Any] = {
        "net": net,
        "sims": sims,
        "games_per_level": games_per_level,
        "timestamp": timestamp.isoformat(),
        "levels": [{k: v for k, v in row.items() if k != "records"} for row in per_level],
        "metrics": metrics,
    }
    if duration_s is not None:
        payload["duration_s"] = round(float(duration_s), 2)
    # Which instrument produced this. Payloads without the key are pre-2026-08-05
    # longitudinal-ladder results, which is why the default is "ladder" rather than
    # something explicit — see ``LADDER_CONDITION`` in evaluation/ladder_selection.
    payload["condition"] = condition
    if context is not None:
        payload["context"] = context
    directory.mkdir(parents=True, exist_ok=True)
    safe_net = net.replace("/", "_")
    path = directory / f"ladder_{safe_net}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    # Write to a temp name in the same directory, then rename. Ladder results are
    # produced out of process (scripts/mini_ladder.py) and consumed by a live
    # training run's cadence check, so a reader can otherwise observe a
    # half-written file. os.replace is atomic within a filesystem, so a reader
    # sees either nothing or the complete payload — never a prefix.
    tmp_path = path.with_name(path.name + ".partial")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def load_ladder_results(directory: Path) -> list[dict[str, Any]]:
    """All ladder result payloads in ``directory``, oldest first.

    Unreadable payloads are skipped rather than raised. The atomic write above
    makes a partial read almost impossible, but this is the belt to its braces:
    a malformed file left by an older writer, an interrupted copy, or a truncated
    object-store sync must not crash the training process that reads it — the
    cadence check can always postpone to the next generation.
    """
    if not directory.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(directory.glob("ladder_*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable ladder result {}: {}", path, exc)
    return sorted(results, key=lambda r: str(r.get("timestamp", "")))
