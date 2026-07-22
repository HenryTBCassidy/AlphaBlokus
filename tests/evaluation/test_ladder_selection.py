"""Tests for keep-best-by-ladder selection and the drift circuit-breaker."""

from __future__ import annotations

import pytest

from alphablokus.evaluation.ladder_selection import (
    DriftAlarm,
    LadderPoint,
    checkpoint_generation,
    detect_drift,
    ladder_point_from_payload,
    select_best,
)


def _point(label: str, weighted: float) -> LadderPoint:
    return LadderPoint(label=label, weighted_score=weighted, generation=checkpoint_generation(label))


def test_checkpoint_generation_parses_coach_names() -> None:
    assert checkpoint_generation("accepted_12.pth.tar") == 12
    assert checkpoint_generation("rejected_3.pth.tar") == 3
    assert checkpoint_generation("/runs/blokus/x/Nets/accepted_40.pth.tar") == 40
    assert checkpoint_generation("best.pth.tar") is None
    assert checkpoint_generation("v3_gen40_donor.pth.tar") is None


def test_ladder_point_from_payload_reads_benchmark_schema() -> None:
    """Parses the JSON written by reporting/pentobi_ladder.write_ladder_result."""
    payload = {
        "net": "accepted_20.pth.tar",
        "sims": 400,
        "games_per_level": 50,
        "levels": [{"level": 3, "games": 50, "net_wins": 29, "win_rate": 0.58}],
        "metrics": {"pentobi_level": 3, "score": 0.412, "weighted_score": 0.298},
    }
    point = ladder_point_from_payload(payload)
    assert point == LadderPoint(
        label="accepted_20.pth.tar",
        weighted_score=0.298,
        generation=20,
        pentobi_level=3,
        score=0.412,
    )


def test_select_best_picks_highest_weighted_score() -> None:
    points = [
        _point("accepted_5.pth.tar", 0.31),
        _point("accepted_10.pth.tar", 0.344),
        _point("accepted_15.pth.tar", 0.30),
    ]
    assert select_best(points).label == "accepted_10.pth.tar"


def test_select_best_tie_breaks_to_lowest_generation() -> None:
    """Equal ladder scores → keep the earlier checkpoint (least drift exposure)."""
    points = [_point("accepted_18.pth.tar", 0.34), _point("accepted_6.pth.tar", 0.34)]
    assert select_best(points).label == "accepted_6.pth.tar"


def test_select_best_generationless_points_lose_ties() -> None:
    points = [LadderPoint(label="donor.pth.tar", weighted_score=0.34), _point("accepted_9.pth.tar", 0.34)]
    assert select_best(points).label == "accepted_9.pth.tar"


def test_select_best_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_best([])


def test_detect_drift_trips_on_two_consecutive_drops() -> None:
    points = [
        _point("accepted_5.pth.tar", 0.344),
        _point("accepted_10.pth.tar", 0.33),  # within noise: not a drop
        _point("accepted_15.pth.tar", 0.29),  # drop 1 (≥0.05 below best 0.344)
        _point("accepted_20.pth.tar", 0.28),  # drop 2 → trip
    ]
    alarm = detect_drift(points)
    assert isinstance(alarm, DriftAlarm)
    assert alarm.tripped_at.label == "accepted_20.pth.tar"
    assert alarm.best_before.label == "accepted_5.pth.tar"
    assert alarm.consecutive_drops == 2


def test_detect_drift_single_drop_recovery_resets_streak() -> None:
    """One bad evaluation bracketed by healthy ones never trips."""
    points = [
        _point("accepted_5.pth.tar", 0.344),
        _point("accepted_10.pth.tar", 0.28),  # drop 1
        _point("accepted_15.pth.tar", 0.34),  # recovery resets
        _point("accepted_20.pth.tar", 0.27),  # drop 1 again — still no trip
    ]
    assert detect_drift(points) is None


def test_detect_drift_best_advances_with_improvements() -> None:
    """The drop threshold tracks the best-so-far, not the first point."""
    points = [
        _point("accepted_5.pth.tar", 0.30),
        _point("accepted_10.pth.tar", 0.40),  # new best
        _point("accepted_15.pth.tar", 0.34),  # ≥0.05 below 0.40 → drop 1
        _point("accepted_20.pth.tar", 0.33),  # drop 2 → trip
    ]
    alarm = detect_drift(points)
    assert alarm is not None
    assert alarm.best_before.label == "accepted_10.pth.tar"


def test_detect_drift_rerun_trajectory_trips_early() -> None:
    """P3's success criterion: the paired_gate rerun's pooled-Elo trajectory,
    mapped to weighted-ladder space, trips before gen 10.

    The rerun's pooled tournament sat every gen at or below the gen-40 anchor
    (max +5.5 Elo) and drifted down; the corresponding mini-ladder history is a
    plateau at ~0.34 decaying toward 0.298 (research §1.1). Modelled at
    5-generation evaluation cadence with the measured endpoints.
    """
    points = [
        LadderPoint(label="anchor_gen40.pth.tar", weighted_score=0.344),
        _point("accepted_1.pth.tar", 0.335),
        _point("accepted_5.pth.tar", 0.29),  # drop 1
        _point("accepted_9.pth.tar", 0.285),  # drop 2 → trip at gen 9
        _point("accepted_13.pth.tar", 0.27),
        _point("accepted_20.pth.tar", 0.298),
    ]
    alarm = detect_drift(points)
    assert alarm is not None
    assert alarm.tripped_at.generation == 9
    assert alarm.best_before.label == "anchor_gen40.pth.tar"


def test_detect_drift_validates_consecutive() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        detect_drift([_point("accepted_1.pth.tar", 0.3)], consecutive=0)
