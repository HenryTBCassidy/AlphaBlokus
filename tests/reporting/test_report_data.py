"""Report payload assembly (reporting/data.py): signal statuses, the closed set
of events and its key, the status summary, and graceful degradation when tables
are missing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from alphablokus.reporting.data import (
    EVENT_DEFINITIONS,
    ReportEvent,
    SignalStatus,
    arena_payload,
    build_report_payload,
    build_signals,
    build_status_summary,
    event_key_payload,
    ladder_payload,
    symmetry_payload,
    target_entropy_payload,
    training_payload,
)
from alphablokus.reporting.pentobi_ladder import write_ladder_result

if TYPE_CHECKING:
    from alphablokus.config import RunConfig


def _event_ids(payload: dict) -> list[str]:
    return [event["id"] for event in payload["events"]]


# ---------------------------------------------------------------------------
# Arena events
# ---------------------------------------------------------------------------


def _arena_frame(scores: list[float], games: int = 100) -> pd.DataFrame:
    wins = [round(s * games) for s in scores]
    return pd.DataFrame(
        {
            "generation": range(1, len(scores) + 1),
            "wins": wins,
            "losses": [games - w for w in wins],
            "draws": [0] * len(scores),
            "accepted": [s >= 0.55 for s in scores],
        }
    )


def test_exact_half_scores_raise_the_event_with_the_count() -> None:
    payload = arena_payload(_arena_frame([0.5, 0.5, 0.62, 0.44]), 0.55, "threshold")
    assert payload is not None
    assert payload["exact_half"] == 2
    assert _event_ids(payload) == [str(ReportEvent.ARENA_SCORE_EXACTLY_HALF)]
    # The observation is the count; the mechanism lives in the key, not here.
    assert payload["events"][0]["detail"] == "2 of 4 generations scored exactly 0.500."


def test_sub_binomial_variance_is_flagged() -> None:
    # Sixteen scores pinned within ±0.01 of 0.51 at 100 games/gen: far tighter
    # than binomial σ₀ ≈ 0.05 — the search_harder v1 signature.
    pinned = [0.51, 0.52, 0.50999, 0.51001, 0.52, 0.51, 0.52, 0.51] * 2
    payload = arena_payload(_arena_frame(pinned), 0.55, "threshold")
    assert payload is not None
    assert payload["sub_binomial"] is True


def test_arena_with_spread_scores_raises_nothing() -> None:
    payload = arena_payload(_arena_frame([0.85, 0.75, 0.62, 0.44, 0.58, 0.71]), 0.55, "threshold")
    assert payload is not None
    assert payload["events"] == []


def test_colour_concentration_raises_the_event_when_split_is_logged() -> None:
    df = _arena_frame([0.51, 0.49, 0.52, 0.50501])
    df["white_wins"] = [96, 95, 97, 93]
    df["black_wins"] = [4, 5, 3, 7]
    payload = arena_payload(df, 0.55, "threshold")
    assert payload is not None
    assert payload["white_rate"] is not None and payload["white_rate"] > 0.9
    assert str(ReportEvent.ARENA_DECISIVE_GAMES_ONE_COLOUR) in _event_ids(payload)


def test_stored_accepted_column_wins_over_threshold_rule() -> None:
    df = _arena_frame([0.50, 0.52])
    df["accepted"] = [True, True]  # gate_mode=always accepts sub-threshold scores
    payload = arena_payload(df, 0.55, "always")
    assert payload is not None
    assert payload["accepted"] == [True, True]


# ---------------------------------------------------------------------------
# Training payload (aux series parity with the old chart behaviour)
# ---------------------------------------------------------------------------


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "generation": [1, 1, 2, 2],
            "epoch": [0, 0, 0, 0],
            "batch_number": [0, 1, 0, 1],
            "pi_loss": [9.0, 8.6, 8.2, 8.0],
            "v_loss": [0.9, 0.85, 0.8, 0.78],
            "total_loss": [9.9, 9.45, 9.0, 8.78],
        }
    )


def test_training_payload_without_aux_heads() -> None:
    payload = training_payload(_training_frame())
    assert payload is not None
    assert payload["aux"] == {}
    assert payload["gens"] == [1, 2]


def test_training_payload_includes_aux_series_when_present() -> None:
    df = _training_frame()
    df["score_loss"] = [0.6, 0.55, 0.48, 0.44]
    payload = training_payload(df)
    assert payload is not None
    assert list(payload["aux"]) == ["score_loss"]
    assert len(payload["aux"]["score_loss"]) == 2


def test_all_null_aux_column_is_treated_as_absent() -> None:
    df = _training_frame()
    df["score_loss"] = [None, None, None, None]
    payload = training_payload(df)
    assert payload is not None
    assert payload["aux"] == {}


# ---------------------------------------------------------------------------
# Signals + verdict
# ---------------------------------------------------------------------------


def _symmetry_frame(kls: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "generation": [g for g in range(1, len(kls) + 1) for _ in range(2)],
            "position_idx": [0, 1] * len(kls),
            "symmetry_idx": [0, 0] * len(kls),
            "kl_divergence": [k for k in kls for _ in range(2)],
            "top1_match": [True, False] * len(kls),
        }
    )


def _signal_by_id(signals: list[dict], signal_id: str) -> dict:
    return next(s for s in signals if s["id"] == signal_id)


def test_rising_symmetry_kl_alerts() -> None:
    # The rerun's trajectory: 0.64 → 1.24 (~1.9×) is well past the 1.5× alert.
    symmetry = symmetry_payload(_symmetry_frame([0.64, 0.66, 0.7, 0.8, 0.9, 1.0, 1.1, 1.24]))
    signals = build_signals(None, None, symmetry, None, None, None)
    assert _signal_by_id(signals, "symmetry_kl")["status"] == "alert"


def test_stable_symmetry_kl_is_ok() -> None:
    symmetry = symmetry_payload(_symmetry_frame([0.7, 0.75, 0.68, 0.72, 0.66, 0.71]))
    signals = build_signals(None, None, symmetry, None, None, None)
    assert _signal_by_id(signals, "symmetry_kl")["status"] == "ok"


def test_target_entropy_collapse_raises_the_event_naming_the_generation() -> None:
    # One generation far below the run median.
    frame = pd.DataFrame(
        {
            "generation": range(1, 11),
            "mean_policy_entropy": [0.85, 0.84, 0.86, 0.83, 0.85, 0.87, 0.506, 0.84, 0.85, 0.84],
        }
    )
    entropy = target_entropy_payload(frame)
    signals = build_signals(None, None, None, None, entropy, None)
    tile = _signal_by_id(signals, "target_entropy")
    assert tile["status"] == str(SignalStatus.ALERT)
    assert _event_ids(tile) == [str(ReportEvent.TARGET_ENTROPY_GENERATION_COLLAPSE)]
    assert tile["events"][0]["detail"] == "Generation 7 at 0.506 nats against a run median of 0.850."
    assert "generation 7" in tile["sub"]


def test_tournament_final_below_anchor_alerts_and_slippage_warns() -> None:
    rerun_like = {"gens": list(range(21)), "rating": [0.0] + [5.0] * 19 + [-44.0], "n_games": [100] * 21}
    v3_like = {"gens": list(range(5)), "rating": [0.0, 150.0, 286.0, 250.0, 240.0], "n_games": [100] * 5}
    assert _signal_by_id(build_signals(None, rerun_like, None, None, None, None), "tournament")["status"] == "alert"
    assert _signal_by_id(build_signals(None, v3_like, None, None, None, None), "tournament")["status"] == "warn"


def test_summary_with_nothing_recorded_says_so() -> None:
    signals = build_signals(None, None, None, None, None, None)
    summary = build_status_summary(signals)
    assert summary["status"] == str(SignalStatus.MISSING)
    assert "Not recorded:" in summary["detail"]


def test_summary_counts_events_without_drawing_a_conclusion() -> None:
    arena = arena_payload(_arena_frame([0.5, 0.5, 0.5, 0.5]), 0.55, "threshold")
    symmetry = symmetry_payload(_symmetry_frame([0.7, 0.72, 0.69, 0.71, 0.7, 0.7]))
    signals = build_signals(None, None, symmetry, None, None, arena)
    summary = build_status_summary(signals)
    # The arena is not measured against an outside reference, so it does not set
    # the summary's status — but its event is still counted and named.
    assert _signal_by_id(signals, "arena")["status"] == str(SignalStatus.ALERT)
    assert summary["status"] != str(SignalStatus.ALERT)
    # Four identical 0.500 scores raise both arena events, on that one signal.
    assert summary["headline"] == "2 events raised on 1 of 6 signals"
    assert "Arena score exactly 0.500" in summary["detail"]
    assert "Arena score spread below binomial" in summary["detail"]


# ---------------------------------------------------------------------------
# The event enum is a closed set, and the key documents all of it
# ---------------------------------------------------------------------------


def test_every_event_has_a_definition_with_a_numeric_trigger() -> None:
    for event in ReportEvent:
        definition = EVENT_DEFINITIONS[event]
        assert definition.label and definition.means
        if event is not ReportEvent.NOT_RECORDED:
            assert any(char.isdigit() for char in definition.trigger), (
                f"{event} must state its numeric threshold, got {definition.trigger!r}"
            )


def test_the_key_lists_every_enum_value_and_the_calibration_caveat() -> None:
    key = event_key_payload()
    assert [row["id"] for row in key["events"]] == [str(e) for e in ReportEvent]
    assert [row["id"] for row in key["statuses"]] == [str(s) for s in SignalStatus]
    assert "not validated out of sample" in key["calibration_note"]


def test_every_raised_event_is_a_member_of_the_enum() -> None:
    """No signal may invent a flag outside ``ReportEvent``."""
    arena = arena_payload(_arena_frame([0.5, 0.5, 0.5, 0.5]), 0.55, "threshold")
    symmetry = symmetry_payload(_symmetry_frame([0.64, 0.66, 0.7, 0.8, 0.9, 1.0, 1.1, 1.24]))
    tournament = {"gens": list(range(21)), "rating": [0.0] + [5.0] * 19 + [-44.0], "n_games": [100] * 21}
    signals = build_signals(None, tournament, symmetry, None, None, arena)
    raised = [event["id"] for signal in signals for event in signal["events"]]
    assert raised, "this fixture is meant to raise events"
    for event_id in raised:
        assert ReportEvent(event_id) in EVENT_DEFINITIONS


def test_signal_status_is_the_worst_status_of_its_events() -> None:
    """A tile's colour follows from named events, never from a free choice."""
    tournament = {"gens": [0, 1, 2], "rating": [0.0, 286.0, 240.0], "n_games": [100] * 3}
    tile = _signal_by_id(build_signals(None, tournament, None, None, None, None), "tournament")
    assert _event_ids(tile) == [str(ReportEvent.POOL_ELO_BELOW_PEAK)]
    assert tile["status"] == str(EVENT_DEFINITIONS[ReportEvent.POOL_ELO_BELOW_PEAK].status)
    # Peak and final are both stated; no advice about which net to pick.
    assert tile["sub"] == "Peak +286 at generation 1."
    assert tile["value"] == "+240 final"


# ---------------------------------------------------------------------------
# Ladder payload: keep-best + drift alarm
# ---------------------------------------------------------------------------


def _write_history(config: RunConfig, weighted: list[float]) -> None:
    history_dir = config.run_directory / "MiniLadder"
    history_dir.mkdir(parents=True, exist_ok=True)
    points = [
        {"label": f"accepted_{5 * (i + 1)}.pth.tar", "weighted_score": w, "generation": 5 * (i + 1)}
        for i, w in enumerate(weighted)
    ]
    (history_dir / "history.json").write_text(json.dumps({"points": points}), encoding="utf-8")


def test_ladder_payload_absent_without_any_results(test_config: RunConfig) -> None:
    assert ladder_payload(test_config) is None


def test_ladder_payload_merges_results_and_history(test_config: RunConfig) -> None:
    write_ladder_result(
        test_config.pentobi_ladder_directory,
        net="accepted_10.pth.tar",
        sims=400,
        games_per_level=50,
        per_level=[
            {"level": 3, "games": 50, "net_wins": 30, "pentobi_wins": 20, "draws": 0, "win_rate": 0.6, "ci": (0.4, 0.7)}
        ],
        metrics={"pentobi_level": 3, "score": 0.6, "weighted_score": 0.6},
    )
    _write_history(test_config, [0.30, 0.35, 0.33])
    payload = ladder_payload(test_config)
    assert payload is not None
    assert payload["from_mini_ladder"] is True
    assert payload["keep_best"]["label"] == "accepted_10.pth.tar"
    assert payload["drift"] is None
    assert payload["entries"][0]["levels"][0]["win_rate"] == 0.6


def test_ladder_drift_alarm_is_surfaced(test_config: RunConfig) -> None:
    # Two consecutive evaluations ≥5pp below best trip the circuit-breaker
    # (the rerun replayed through this logic trips by ~gen 8-10).
    _write_history(test_config, [0.34, 0.28, 0.27])
    payload = ladder_payload(test_config)
    assert payload is not None
    assert payload["drift"] is not None
    assert payload["drift"]["best_before"] == "accepted_5.pth.tar"
    assert payload["drift"]["consecutive_drops"] == 2


# ---------------------------------------------------------------------------
# Whole-payload degradation
# ---------------------------------------------------------------------------


def test_build_report_payload_on_an_empty_run_dir(test_config: RunConfig) -> None:
    """A run directory with no tables at all must still yield a valid payload
    whose signals are all explicitly 'missing' — absence is visible, not fatal."""
    payload = build_report_payload(test_config)
    assert payload["summary"]["status"] == "missing"
    anchored = [s for s in payload["signals"] if s["anchored"]]
    assert anchored and all(s["status"] == "missing" for s in anchored)
    assert "ArenaData" in payload["meta"]["missing_tables"]
    assert payload["replays"] is None
    assert payload["event_key"]["events"]  # the key ships even for an empty run
    json.dumps(payload)  # payload must be JSON-serialisable end-to-end
