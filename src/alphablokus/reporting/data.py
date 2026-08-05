"""Report payload assembly — every metric table reduced to plain JSON-ready dicts.

The end-of-run report is a single self-contained HTML page whose charts are
rendered client-side from one embedded JSON payload. This module builds that
payload. Design rules:

1. **Every table is optional.** Runs sync partially, schemas evolve, and older
   runs predate newer diagnostics. A missing table renders as an explicit
   "not recorded" state — absence of evidence must be visible, never silent.
2. **The report presents, it does not interpret.** Values, counts and
   thresholds are stated; what they imply about the run is left to the reader.
   Prose exists only to say what a metric is.
3. **Statuses and events are a closed, documented set.** Every automatic flag
   is a member of :class:`ReportEvent` with a deterministic trigger recorded in
   :data:`EVENT_DEFINITIONS`, which the report renders as a key so the reader
   sees the rule and the numbers, not just the label.
4. **Signals record what they are measured against.** Some are measured against
   an outside opponent or a game invariance, some against the run's own data;
   ``anchored`` and ``measured_against`` carry that as plain fact.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from alphablokus.evaluation.ladder_selection import (
    DEFAULT_CONSECUTIVE_DROPS,
    DEFAULT_DROP,
    LadderPoint,
    checkpoint_generation,
    detect_drift,
    ladder_point_from_payload,
    select_best,
)
from alphablokus.reporting.pentobi_ladder import load_ladder_results

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.config import RunConfig

# ---------------------------------------------------------------------------
# Event thresholds.
#
# Every number here is quoted verbatim in the report's event key (built from
# these constants in ``EVENT_DEFINITIONS``, so the key cannot drift from the
# code). All of them were fitted to two runs from this project's history —
# ``blokus_paired_gate_rerun`` and ``blokus_cloud_v3`` — the same runs used to
# demonstrate them, so they are not validated out of sample. The report says so
# in the key (``CALIBRATION_NOTE``).
# ---------------------------------------------------------------------------

# Symmetry KL / value-symmetry MAE trend = mean(last k) / mean(first k).
_TREND_WARN_RATIO = 1.20
_TREND_ALERT_RATIO = 1.50

# Self-play target entropy, as a fraction of the run's own median.
_ENTROPY_COLLAPSE_RATIO = 0.70
_ENTROPY_WARN_RATIO = 0.85

# Arena score checks: exact-0.500 scores, per-generation score spread against
# the binomial expectation, and the share of decisive games taken by White.
_SUB_BINOMIAL_FACTOR = 0.5
_SUB_BINOMIAL_MIN_GENS = 4
_COLOUR_PINNED_WHITE_RATE = 0.85

# Pooled BayesElo: distance from the run's own peak, and the gen-0 anchor at 0.
_POOL_ELO_WARN_DROP = 15.0
_POOL_ELO_ALERT_FINAL_BELOW_ANCHOR = -10.0
_POOL_ELO_ALERT_DROP = 60.0

# Bound embedded chart payloads: per-batch loss traces are EWM-smoothed then
# downsampled to at most this many points per series.
_TIMELINE_MAX_POINTS = 1200


# ---------------------------------------------------------------------------
# Statuses and events — the closed set the report is allowed to raise
# ---------------------------------------------------------------------------


class SignalStatus(StrEnum):
    """Severity carried by a signal tile. Four values, no others.

    The values double as CSS class names in ``assets/report.css``.
    """

    OK = "ok"
    WARN = "warn"
    ALERT = "alert"
    MISSING = "missing"


_STATUS_ORDER: dict[SignalStatus, int] = {
    SignalStatus.MISSING: 0,
    SignalStatus.OK: 1,
    SignalStatus.WARN: 2,
    SignalStatus.ALERT: 3,
}


class ReportEvent(StrEnum):
    """Every named event the report can raise, as a closed set.

    Each member has an entry in :data:`EVENT_DEFINITIONS` giving the exact
    trigger condition (with its numeric threshold) and a plain statement of
    what the measurement indicates. Nothing outside this enum may appear as a
    flag in the payload.
    """

    NOT_RECORDED = "not_recorded"
    ARENA_SCORE_EXACTLY_HALF = "arena_score_exactly_half"
    ARENA_SCORE_SPREAD_BELOW_BINOMIAL = "arena_score_spread_below_binomial"
    ARENA_DECISIVE_GAMES_ONE_COLOUR = "arena_decisive_games_one_colour"
    LADDER_LATEST_BELOW_BEST = "ladder_latest_below_best"
    LADDER_DRIFT_BREAKER_TRIPPED = "ladder_drift_breaker_tripped"
    POOL_ELO_BELOW_PEAK = "pool_elo_below_peak"
    POOL_ELO_FAR_BELOW_PEAK = "pool_elo_far_below_peak"
    POOL_ELO_BELOW_ANCHOR = "pool_elo_below_anchor"
    SYMMETRY_TREND_RISING = "symmetry_trend_rising"
    SYMMETRY_TREND_RISING_STEEPLY = "symmetry_trend_rising_steeply"
    TARGET_ENTROPY_GENERATION_COLLAPSE = "target_entropy_generation_collapse"
    TARGET_ENTROPY_LATEST_BELOW_MEDIAN = "target_entropy_latest_below_median"


@dataclass(frozen=True)
class EventDefinition:
    """One row of the report's event key.

    Attributes:
        label: The event, named as an event rather than a judgement.
        status: Severity the event gives the signal that raised it.
        trigger: The exact condition that fires it, numeric threshold included.
        means: What the measurement indicates, stated as mechanism — no advice
            and no claim about whether the run is good.
    """

    label: str
    status: SignalStatus
    trigger: str
    means: str


EVENT_DEFINITIONS: dict[ReportEvent, EventDefinition] = {
    ReportEvent.NOT_RECORDED: EventDefinition(
        label="Not recorded",
        status=SignalStatus.MISSING,
        trigger="The metric table this signal reads is absent from the run directory.",
        means="The quantity was never measured for this run. The signal is neither pass nor fail.",
    ),
    ReportEvent.ARENA_SCORE_EXACTLY_HALF: EventDefinition(
        label="Arena score exactly 0.500",
        status=SignalStatus.ALERT,
        trigger="A generation's arena score equals 0.500 to within 1e-9.",
        means=(
            "Arena games are played deterministically, so two nets of equal strength replay identical "
            "games: a colour-swapped pair splits 1–1 and the score lands on exactly 0.500. A score of "
            "exactly 0.500 records that the arena separated the two nets by nothing — it is not a "
            "measured tie."
        ),
    ),
    ReportEvent.ARENA_SCORE_SPREAD_BELOW_BINOMIAL: EventDefinition(
        label="Arena score spread below binomial",
        status=SignalStatus.ALERT,
        trigger=(
            f"Standard deviation of the per-generation scores is below {_SUB_BINOMIAL_FACTOR:g}× the "
            "binomial σ₀ = √(p̄(1−p̄)/n̄) expected at the run's mean score p̄ and mean games per "
            f"generation n̄, over at least {_SUB_BINOMIAL_MIN_GENS} generations."
        ),
        means=(
            "Scores repeat more closely than independent games of that size would, so something "
            "identical in every generation is contributing to the result alongside net strength."
        ),
    ),
    ReportEvent.ARENA_DECISIVE_GAMES_ONE_COLOUR: EventDefinition(
        label="Decisive arena games concentrated in one colour",
        status=SignalStatus.ALERT,
        trigger=f"White's share of decisive arena games is at least {_COLOUR_PINNED_WHITE_RATE:.0%}.",
        means=(
            "In Blokus Duo the first mover wins ~93–97% of decisive deterministic games, so an unpaired "
            "arena score varies with which net played White as well as with net strength."
        ),
    ),
    ReportEvent.LADDER_LATEST_BELOW_BEST: EventDefinition(
        label="Latest ladder evaluation below the run's best",
        status=SignalStatus.WARN,
        trigger=(
            "The most recent mini-ladder weighted score is below the highest one recorded for this run, by "
            "more than 1e-9. There is no tuned margin: any drop fires it."
        ),
        means="The most recently evaluated checkpoint is not the run's highest-scoring checkpoint on the ladder.",
    ),
    ReportEvent.LADDER_DRIFT_BREAKER_TRIPPED: EventDefinition(
        label="Ladder drift breaker tripped",
        status=SignalStatus.ALERT,
        trigger=(
            f"{DEFAULT_CONSECUTIVE_DROPS} consecutive mini-ladder evaluations sit at least "
            f"{DEFAULT_DROP:.2f} weighted score below the best seen so far "
            "(evaluation/ladder_selection.py), or MiniLadder/DRIFT_ALARM exists in the run directory."
        ),
        means="Ladder score fell and stayed below the run's best across consecutive evaluations.",
    ),
    ReportEvent.POOL_ELO_BELOW_PEAK: EventDefinition(
        label="Pool Elo below the run's peak",
        status=SignalStatus.WARN,
        trigger=f"Final pooled rating is at least {_POOL_ELO_WARN_DROP:.0f} Elo below the run's peak rating.",
        means="The final checkpoint is not the highest-rated checkpoint in the pooled tournament.",
    ),
    ReportEvent.POOL_ELO_FAR_BELOW_PEAK: EventDefinition(
        label="Pool Elo far below the run's peak",
        status=SignalStatus.ALERT,
        trigger=f"Final pooled rating is at least {_POOL_ELO_ALERT_DROP:.0f} Elo below the run's peak rating.",
        means="Same measurement as the event above, past a wider margin.",
    ),
    ReportEvent.POOL_ELO_BELOW_ANCHOR: EventDefinition(
        label="Pool Elo below the gen-0 anchor",
        status=SignalStatus.ALERT,
        trigger=(
            f"Final pooled rating is {_POOL_ELO_ALERT_FINAL_BELOW_ANCHOR:.0f} Elo or lower, the gen-0 "
            "anchor being 0 by construction."
        ),
        means="The final checkpoint rates below the checkpoint the run started from.",
    ),
    ReportEvent.SYMMETRY_TREND_RISING: EventDefinition(
        label="Symmetry error trend rising",
        status=SignalStatus.WARN,
        trigger=(
            f"mean of the last k generations ÷ mean of the first k is at least {_TREND_WARN_RATIO:.2f}, "
            "k = min(3, generations ÷ 3), needing at least 4 generations. Applies to policy symmetry KL "
            "and to value symmetry MAE."
        ),
        means=(
            "The board's symmetries are exact game invariances, so the true policy and value are identical "
            "across them. A rising trend means the net's outputs agree less across those transforms at the "
            "end of the run than at the start."
        ),
    ),
    ReportEvent.SYMMETRY_TREND_RISING_STEEPLY: EventDefinition(
        label="Symmetry error trend rising steeply",
        status=SignalStatus.ALERT,
        trigger=f"The same ratio as the event above is at least {_TREND_ALERT_RATIO:.2f}.",
        means="Same measurement as the event above, past a wider margin.",
    ),
    ReportEvent.TARGET_ENTROPY_GENERATION_COLLAPSE: EventDefinition(
        label="Target entropy collapse in one generation",
        status=SignalStatus.ALERT,
        trigger=(
            f"Some generation's mean self-play target entropy is below {_ENTROPY_COLLAPSE_RATIO:.2f}× the run's median."
        ),
        means=(
            "In that generation search spread its visits over far fewer moves than the run's typical "
            "spread, so the policy targets stored for training were correspondingly sharper."
        ),
    ),
    ReportEvent.TARGET_ENTROPY_LATEST_BELOW_MEDIAN: EventDefinition(
        label="Latest target entropy below the run median",
        status=SignalStatus.WARN,
        trigger=(
            f"The last generation's mean self-play target entropy is below {_ENTROPY_WARN_RATIO:.2f}× the run's median."
        ),
        means="The run finished producing sharper policy targets than it produced on average.",
    ),
}

CALIBRATION_NOTE = (
    "Every threshold above was fitted to two runs from this project's history — blokus_paired_gate_rerun, "
    "which regressed, and blokus_cloud_v3 — which are also the runs used to demonstrate the rules. They "
    "have not been checked against any run outside that pair, so they are not validated out of sample. "
    "The trigger column is the whole rule: read it and the measured value rather than the label alone."
)


def _event(event: ReportEvent, detail: str) -> dict[str, Any]:
    """One raised event: its enum id, its key label, and the observed numbers."""
    return {"id": str(event), "label": EVENT_DEFINITIONS[event].label, "detail": detail}


def _status_of(events: list[dict[str, Any]]) -> SignalStatus:
    """The most severe status among the raised events (``OK`` when none did)."""
    if not events:
        return SignalStatus.OK
    statuses = [EVENT_DEFINITIONS[ReportEvent(e["id"])].status for e in events]
    return max(statuses, key=lambda s: _STATUS_ORDER[s])


def event_key_payload() -> dict[str, Any]:
    """The report's key: every status, every event, its trigger, and the caveat."""
    return {
        "statuses": [
            {"id": str(SignalStatus.OK), "label": "No event raised"},
            {"id": str(SignalStatus.WARN), "label": "An event with a warn-level trigger was raised"},
            {"id": str(SignalStatus.ALERT), "label": "An event with an alert-level trigger was raised"},
            {"id": str(SignalStatus.MISSING), "label": "Not recorded — the underlying table is absent"},
        ],
        "events": [
            {
                "id": str(event),
                "label": definition.label,
                "status": str(definition.status),
                "trigger": definition.trigger,
                "means": definition.means,
            }
            for event, definition in EVENT_DEFINITIONS.items()
        ],
        "calibration_note": CALIBRATION_NOTE,
    }


def load_metrics(directory: Path) -> pd.DataFrame | None:
    """Read a hive-partitioned metrics directory, or ``None`` when absent.

    Pandas/PyArrow infer hive-partition keys as ``category`` dtype, which sorts
    lexically (1, 10, 11, ..., 2); the ``generation`` column is cast to ``int``
    here so every downstream sort and groupby is numeric.
    """
    if not directory.exists():
        return None
    try:
        df = pd.read_parquet(directory)
    except (OSError, ValueError) as err:
        logger.warning("Could not read metrics from {} ({}); section omitted.", directory, err)
        return None
    if df.empty:
        return None
    if "generation" in df.columns:
        df["generation"] = df["generation"].astype(int)
    return df


def _round_list(values: Any, digits: int = 4) -> list[float]:
    """Round a numeric series for compact JSON embedding."""
    return [round(float(v), digits) for v in values]


def _trend_ratio(values: list[float]) -> float | None:
    """Mean of the last k points over mean of the first k (k = n//3, capped at 3).

    ``None`` when there are fewer than 4 points or the early mean is ~0.
    """
    if len(values) < 4:
        return None
    k = max(1, min(3, len(values) // 3))
    early = sum(values[:k]) / k
    late = sum(values[-k:]) / k
    if abs(early) < 1e-12:
        return None
    return late / early


# ---------------------------------------------------------------------------
# Per-table payload builders — one per parquet table, each ``None``-tolerant.
# ---------------------------------------------------------------------------


def training_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Per-generation loss summary + a smoothed per-batch timeline.

    Per-gen values are the mean over each generation's *last* epoch (the most
    trained state; avoids single-batch noise). Auxiliary head losses
    (``score_loss`` / ``ownership_loss`` / ``reply_loss``) appear only when the
    column exists and holds any non-null value, so runs without those heads
    render unchanged.
    """
    if df is None or df.empty:
        return None
    sorted_df = df.sort_values(["generation", "epoch", "batch_number"]).reset_index(drop=True)
    last_epoch = sorted_df.groupby("generation")["epoch"].transform("max")
    last_epoch_df = sorted_df[sorted_df["epoch"] == last_epoch]
    per_gen = last_epoch_df.groupby("generation")[["pi_loss", "v_loss", "total_loss"]].mean().reset_index()
    per_gen = per_gen.sort_values("generation")

    aux: dict[str, list[float]] = {}
    for column in ("score_loss", "ownership_loss", "reply_loss"):
        if column in sorted_df.columns and sorted_df[column].notna().any():
            aux[column] = _round_list(last_epoch_df.groupby("generation")[column].mean().sort_index())

    # Timeline: EWM-smoothed raw batches over the whole run, downsampled.
    sorted_df["step"] = range(len(sorted_df))
    stride = max(1, len(sorted_df) // _TIMELINE_MAX_POINTS)
    span = max(5, len(sorted_df) // 100)
    smooth = sorted_df[["pi_loss", "v_loss", "total_loss"]].ewm(span=span).mean()
    sampled = smooth.iloc[::stride]
    steps = sorted_df["step"].iloc[::stride]
    gen_starts = sorted_df.groupby("generation")["step"].min()

    return {
        "gens": [int(g) for g in per_gen["generation"]],
        "pi": _round_list(per_gen["pi_loss"]),
        "v": _round_list(per_gen["v_loss"]),
        "total": _round_list(per_gen["total_loss"]),
        "aux": aux,
        "timeline": {
            "x": [int(s) for s in steps],
            "pi": _round_list(sampled["pi_loss"]),
            "v": _round_list(sampled["v_loss"]),
            "total": _round_list(sampled["total_loss"]),
            "gen_starts": [[int(g), int(s)] for g, s in gen_starts.items()],
        },
    }


def arena_payload(df: pd.DataFrame | None, update_threshold: float, gate_mode: str) -> dict[str, Any] | None:
    """Per-generation arena tallies plus any :class:`ReportEvent` they raise.

    Three checks run over the score series: scores of exactly 0.500, the spread
    of the per-generation scores against the binomial expectation, and the share
    of decisive games taken by White (when the per-colour split was logged).
    Each ``events`` entry states the count or rate observed; the trigger rule
    and what the measurement indicates live in :data:`EVENT_DEFINITIONS`.
    """
    if df is None or df.empty:
        return None
    df = df.sort_values("generation")
    total = df["wins"] + df["losses"] + df["draws"]
    played = total > 0
    if not played.any():
        return None
    df = df[played]
    total = total[played]
    scores = (df["wins"] + 0.5 * df["draws"]) / total

    if "accepted" in df.columns:
        accepted = [bool(a) for a in df["accepted"]]
    else:  # very old runs: reconstruct from the threshold rule
        accepted = [bool(s >= update_threshold) for s in scores]

    events: list[dict[str, Any]] = []
    exact_half = int((scores.sub(0.5).abs() < 1e-9).sum())
    if exact_half > 0:
        events.append(
            _event(
                ReportEvent.ARENA_SCORE_EXACTLY_HALF,
                f"{exact_half} of {len(scores)} generations scored exactly 0.500.",
            )
        )
    sub_binomial = False
    if len(scores) >= _SUB_BINOMIAL_MIN_GENS:
        mean_p = float(scores.mean())
        mean_n = float(total.mean())
        sigma0 = (mean_p * (1.0 - mean_p) / mean_n) ** 0.5 if 0.0 < mean_p < 1.0 and mean_n > 0 else 0.0
        observed_std = float(scores.std(ddof=1))
        if sigma0 > 0 and observed_std < _SUB_BINOMIAL_FACTOR * sigma0:
            sub_binomial = True
            events.append(
                _event(
                    ReportEvent.ARENA_SCORE_SPREAD_BELOW_BINOMIAL,
                    f"Observed σ = {observed_std:.3f} against a binomial σ₀ of {sigma0:.3f} "
                    f"at p̄ = {mean_p:.3f} over {len(scores)} generations.",
                )
            )

    white_rate: float | None = None
    if "white_wins" in df.columns and "black_wins" in df.columns:
        decisive = float((df["white_wins"] + df["black_wins"]).sum())
        if decisive > 0:
            white_rate = float(df["white_wins"].sum()) / decisive
            if white_rate >= _COLOUR_PINNED_WHITE_RATE:
                events.append(
                    _event(
                        ReportEvent.ARENA_DECISIVE_GAMES_ONE_COLOUR,
                        f"White won {white_rate:.0%} of the {int(decisive):,} decisive arena games.",
                    )
                )

    payload: dict[str, Any] = {
        "gens": [int(g) for g in df["generation"]],
        "wins": [int(w) for w in df["wins"]],
        "losses": [int(w) for w in df["losses"]],
        "draws": [int(w) for w in df["draws"]],
        "score": _round_list(scores),
        "accepted": accepted,
        "threshold": update_threshold,
        "gate_mode": gate_mode,
        "events": events,
        "exact_half": exact_half,
        "sub_binomial": sub_binomial,
    }
    if "white_wins" in df.columns and "black_wins" in df.columns:
        payload["white_wins"] = [int(w) for w in df["white_wins"]]
        payload["black_wins"] = [int(w) for w in df["black_wins"]]
        payload["white_rate"] = round(white_rate, 4) if white_rate is not None else None
    return payload


def symmetry_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Per-generation policy symmetry KL — mean and max over eval positions.

    KL is averaged over symmetries within each position first, then over
    positions, matching the W&B scalar ``learning_quality/symmetry_kl_mean``.
    """
    if df is None or df.empty:
        return None
    per_position = df.groupby(["generation", "position_idx"])["kl_divergence"].mean().reset_index()
    agg = (
        per_position.groupby("generation")["kl_divergence"]
        .agg(kl_mean="mean", kl_max="max")
        .reset_index()
        .sort_values("generation")
    )
    top1 = df.groupby("generation")["top1_match"].mean().sort_index() if "top1_match" in df.columns else None
    payload = {
        "gens": [int(g) for g in agg["generation"]],
        "kl_mean": _round_list(agg["kl_mean"]),
        "kl_max": _round_list(agg["kl_max"]),
    }
    if top1 is not None:
        payload["top1_match"] = _round_list(top1)
    return payload


def pvc_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Policy–value consistency + value symmetry MAE per generation."""
    if df is None or df.empty:
        return None
    agg_spec: dict[str, Any] = {"spearman": ("pvc_spearman", "mean"), "argmax": ("pvc_argmax_match", "mean")}
    has_mae = "value_symmetry_mae" in df.columns and df["value_symmetry_mae"].notna().any()
    if has_mae:
        agg_spec["mae"] = ("value_symmetry_mae", "mean")
    agg = df.groupby("generation").agg(**agg_spec).reset_index().sort_values("generation")
    payload = {
        "gens": [int(g) for g in agg["generation"]],
        "spearman": _round_list(agg["spearman"]),
        "argmax_match": _round_list(agg["argmax"]),
    }
    if has_mae:
        payload["value_symmetry_mae"] = _round_list(agg["mae"])
    return payload


def target_entropy_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Self-play policy-target entropy per generation (mean + p10/p90 band).

    This is the entropy of the stored search-policy targets at harvest — how
    widely search spread its visits over the moves it considered, and therefore
    how sharp the targets handed to the trainer are.
    """
    if df is None or df.empty or "mean_policy_entropy" not in df.columns:
        return None
    grouped = df.groupby("generation")["mean_policy_entropy"]
    agg = grouped.agg(
        mean="mean",
        p10=lambda s: s.quantile(0.10),
        p90=lambda s: s.quantile(0.90),
    ).reset_index()
    agg = agg.sort_values("generation")
    return {
        "gens": [int(g) for g in agg["generation"]],
        "mean": _round_list(agg["mean"]),
        "p10": _round_list(agg["p10"]),
        "p90": _round_list(agg["p90"]),
    }


def rolling_elo_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Rolling arena-derived Elo: a chained estimate from the run's own games."""
    if df is None or df.empty:
        return None
    df = df.sort_values("generation")
    return {
        "gens": [int(g) for g in df["generation"]],
        "elo": _round_list(df["rolling_elo"], 1),
        "accepted": [bool(a) for a in df["accepted"]] if "accepted" in df.columns else None,
        "score_rate": _round_list(df["score_rate"]) if "score_rate" in df.columns else None,
    }


def legacy_elo_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """The retired Elo-vs-frozen-gen-0 series (older runs only) — saturating."""
    if df is None or df.empty or "elo_rating" not in df.columns:
        return None
    df = df.sort_values("generation")
    return {
        "gens": [int(g) for g in df["generation"]],
        "elo": _round_list(df["elo_rating"], 1),
    }


def tournament_payload(path: Path) -> dict[str, Any] | None:
    """The pooled BayesElo tournament curve (one shared scale, non-saturating)."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except (OSError, ValueError) as err:
        logger.warning("Could not read tournament ratings from {} ({}); section omitted.", path, err)
        return None
    if df.empty:
        return None
    df = df.sort_values("generation")
    return {
        "gens": [int(g) for g in df["generation"]],
        "rating": _round_list(df["rating"], 1),
        "n_games": [int(n) for n in df["n_games"]],
    }


def accuracy_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Eval-set policy agreement per generation, against the run's own targets."""
    if df is None or df.empty:
        return None
    agg_spec: dict[str, Any] = {"top1": ("top1_accuracy", "mean"), "top5": ("top5_accuracy", "mean")}
    for column, name in (("mcts_top1_accuracy", "mcts_top1"), ("mcts_top5_accuracy", "mcts_top5")):
        if column in df.columns and df[column].notna().any():
            agg_spec[name] = (column, "mean")
    agg = df.groupby("generation").agg(**agg_spec).reset_index().sort_values("generation")
    payload: dict[str, Any] = {"gens": [int(g) for g in agg["generation"]]}
    for name in ("top1", "top5", "mcts_top1", "mcts_top5"):
        if name in agg.columns:
            payload[name] = _round_list(agg[name])
    return payload


def net_entropy_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Network policy entropy on the frozen eval set, per generation."""
    if df is None or df.empty:
        return None
    agg = df.groupby("generation")["mean_entropy"].mean().reset_index().sort_values("generation")
    return {
        "gens": [int(g) for g in agg["generation"]],
        "mean": _round_list(agg["mean_entropy"]),
    }


def calibration_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Value-head calibration: per-gen |predicted − actual| error + the final
    generation's reliability curve."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df["abs_err"] = (df["bucket_center"] - df["bucket_mean_actual"]).abs()
    weighted = df[df["bucket_count"] > 0]
    per_gen = (
        weighted.groupby("generation")
        .apply(lambda g: float((g["abs_err"] * g["bucket_count"]).sum() / g["bucket_count"].sum()))
        .reset_index(name="error")
        .sort_values("generation")
    )
    last_gen = int(df["generation"].max())
    last = df[(df["generation"] == last_gen) & (df["bucket_count"] > 0)].sort_values("bucket_center")
    return {
        "gens": [int(g) for g in per_gen["generation"]],
        "error": _round_list(per_gen["error"]),
        "reliability": {
            "generation": last_gen,
            "centers": _round_list(last["bucket_center"]),
            "actual": _round_list(last["bucket_mean_actual"]),
            "counts": [int(c) for c in last["bucket_count"]],
        },
    }


def learning_rate_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """The learning rate the optimiser actually trained at, per generation."""
    if df is None or df.empty:
        return None
    agg = df.groupby("generation")["learning_rate"].mean().reset_index().sort_values("generation")
    return {
        "gens": [int(g) for g in agg["generation"]],
        "lr": [float(f"{v:.6g}") for v in agg["learning_rate"]],
    }


def selfplay_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Self-play shape per generation: game length band + search throughput."""
    if df is None or df.empty:
        return None
    grouped = df.groupby("generation")
    agg = grouped.agg(
        moves_mean=("num_moves", "mean"),
        moves_p10=("num_moves", lambda s: s.quantile(0.10)),
        moves_p90=("num_moves", lambda s: s.quantile(0.90)),
        sims_median=("sims_per_second", "median"),
    ).reset_index()
    agg = agg.sort_values("generation")
    return {
        "gens": [int(g) for g in agg["generation"]],
        "moves_mean": _round_list(agg["moves_mean"], 2),
        "moves_p10": _round_list(agg["moves_p10"], 2),
        "moves_p90": _round_list(agg["moves_p90"], 2),
        "sims_median": _round_list(agg["sims_median"], 1),
    }


def perf_payload(
    timings: pd.DataFrame | None,
    throughput: pd.DataFrame | None,
    resources: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """Operational telemetry: phase wall-clock, training throughput, memory."""
    payload: dict[str, Any] = {}
    if timings is not None and not timings.empty:
        stage_df = timings[timings["cycle_stage"] != "WholeCycle"]
        pivot = stage_df.pivot_table(
            index="generation", columns="cycle_stage", values="time_elapsed", aggfunc="sum"
        ).sort_index()
        payload["timing"] = {
            "gens": [int(g) for g in pivot.index],
            "stages": {str(col): _round_list(pivot[col].fillna(0.0), 1) for col in pivot.columns},
        }
        whole = timings[timings["cycle_stage"] == "WholeCycle"]["time_elapsed"].sum()
        payload["total_time_s"] = round(float(whole), 1)
    if throughput is not None and not throughput.empty:
        agg = throughput.groupby("generation")["samples_per_second"].mean().sort_index()
        payload["throughput"] = {
            "gens": [int(g) for g in agg.index],
            "sps": _round_list(agg, 1),
        }
    if resources is not None and not resources.empty:
        pivot = resources.pivot_table(
            index="generation", columns="cycle_stage", values="process_rss_bytes", aggfunc="max"
        ).sort_index()
        payload["memory"] = {
            "gens": [int(g) for g in pivot.index],
            "stages": {str(col): _round_list(pivot[col].fillna(0.0) / (1024**3), 2) for col in pivot.columns},
        }
    return payload or None


# ---------------------------------------------------------------------------
# Pentobi ladder + mini-ladder keep-best / drift state
# ---------------------------------------------------------------------------


def ladder_payload(config: RunConfig) -> dict[str, Any] | None:
    """External Pentobi ladder results + keep-best selection + drift-alarm state.

    Merges the run's ``PentobiLadder/`` result JSONs (full or mini ladders —
    same schema) with the ``MiniLadder/history.json`` selection history when
    present. The mini-ladder is the run's *selection* instrument: keep-best and
    the drift circuit-breaker are recomputed from the history via
    ``evaluation.ladder_selection`` so the report always reflects the same
    logic the box runbook uses, plus ``MiniLadder/DRIFT_ALARM`` is surfaced
    directly if the runner left one.
    """
    results = load_ladder_results(config.pentobi_ladder_directory)
    history_path = config.run_directory / "MiniLadder" / "history.json"
    alarm_path = config.run_directory / "MiniLadder" / "DRIFT_ALARM"

    entries: list[dict[str, Any]] = []
    for result in results:
        levels = sorted(result["levels"], key=lambda row: int(row["level"]))
        entries.append(
            {
                "net": result["net"],
                "generation": checkpoint_generation(str(result["net"])),
                "timestamp": str(result.get("timestamp", ""))[:16].replace("T", " "),
                "sims": result.get("sims"),
                "games_per_level": result.get("games_per_level"),
                "levels": [
                    {
                        "level": int(row["level"]),
                        "win_rate": round(float(row["win_rate"]), 4),
                        "wins": int(row["net_wins"]),
                        "losses": int(row["pentobi_wins"]),
                        "draws": int(row["draws"]),
                        "ci": [round(float(c), 4) for c in row["ci"]] if "ci" in row else None,
                    }
                    for row in levels
                ],
                "pentobi_level": result["metrics"].get("pentobi_level"),
                "weighted_score": round(float(result["metrics"]["weighted_score"]), 4)
                if "weighted_score" in result["metrics"]
                else None,
                # Wall-clock cost of this ladder. Absent for ladders run before it
                # was recorded, so the report treats it as optional.
                "duration_s": round(float(result["duration_s"]), 1) if "duration_s" in result else None,
            }
        )

    history_points: list[LadderPoint] = []
    if history_path.exists():
        try:
            rows = json.loads(history_path.read_text(encoding="utf-8"))["points"]
            history_points = [
                LadderPoint(
                    label=row["label"],
                    weighted_score=float(row["weighted_score"]),
                    generation=row.get("generation"),
                    pentobi_level=row.get("pentobi_level"),
                    score=row.get("score"),
                )
                for row in rows
            ]
        except (ValueError, KeyError, OSError) as err:
            logger.warning("Could not parse mini-ladder history at {} ({}).", history_path, err)

    # Selection points: prefer the mini-ladder history (evaluation-ordered);
    # fall back to the ladder result JSONs themselves.
    points = history_points or [ladder_point_from_payload(r) for r in results if "weighted_score" in r["metrics"]]
    if not entries and not points:
        return None

    keep_best: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None
    if points:
        best = select_best(points)
        keep_best = {
            "label": best.label,
            "generation": best.generation,
            "weighted_score": round(best.weighted_score, 4),
            "pentobi_level": best.pentobi_level,
        }
        alarm = detect_drift(points)
        if alarm is not None:
            drift = {
                "tripped_at": alarm.tripped_at.label,
                "tripped_score": round(alarm.tripped_at.weighted_score, 4),
                "best_before": alarm.best_before.label,
                "best_score": round(alarm.best_before.weighted_score, 4),
                "consecutive_drops": alarm.consecutive_drops,
            }

    return {
        "entries": entries,
        "history": [
            {
                "label": p.label,
                "generation": p.generation,
                "weighted_score": round(p.weighted_score, 4),
                "pentobi_level": p.pentobi_level,
            }
            for p in points
        ],
        "keep_best": keep_best,
        "drift": drift,
        "alarm_file": alarm_path.exists(),
        "from_mini_ladder": bool(history_points),
    }


# ---------------------------------------------------------------------------
# Front-page signals + status summary
# ---------------------------------------------------------------------------

# What each signal is measured against — stated as fact, for the report's key.
_MEASURED_AGAINST: dict[str, str] = {
    "ladder": "Games against Pentobi, an outside engine.",
    "tournament": "Pooled games between this run's own checkpoints, rated on one scale against the gen-0 anchor.",
    "symmetry_kl": "The board's symmetries, which are exact invariances of the game's rules.",
    "value_mae": "The board's symmetries, which are exact invariances of the game's rules.",
    "target_entropy": "This run's own stored self-play search targets.",
    "arena": "This run's own candidate-versus-incumbent games.",
}


def _signal(
    signal_id: str,
    label: str,
    value: str,
    sub: str,
    *,
    anchored: bool,
    events: list[dict[str, Any]] | None = None,
    status: SignalStatus | None = None,
    spark: list[float] | None = None,
    href: str | None = None,
) -> dict[str, Any]:
    """One front-page tile.

    ``status`` defaults to the most severe status among ``events``, so a tile's
    colour is always a consequence of a named event rather than a free choice.
    """
    raised = events or []
    return {
        "id": signal_id,
        "label": label,
        "status": str(status if status is not None else _status_of(raised)),
        "value": value,
        "sub": sub,
        "anchored": anchored,
        "measured_against": _MEASURED_AGAINST.get(signal_id, ""),
        "events": raised,
        "spark": spark,
        "href": href,
    }


def _not_recorded(
    signal_id: str, label: str, value: str, sub: str, *, anchored: bool, href: str | None = None
) -> dict[str, Any]:
    """A tile for a signal whose table is absent — the one MISSING path."""
    return _signal(
        signal_id,
        label,
        value,
        sub,
        anchored=anchored,
        events=[_event(ReportEvent.NOT_RECORDED, sub)],
        status=SignalStatus.MISSING,
        href=href,
    )


def _trend_events(values: list[float] | None, metric: str) -> tuple[list[dict[str, Any]], float | None]:
    """Symmetry-trend events for a rising-error metric, plus the ratio itself."""
    if not values:
        return [], None
    ratio = _trend_ratio(values)
    if ratio is None:
        return [], None
    detail = f"{metric} ended at {ratio:.2f}× its first-generations mean."
    if ratio >= _TREND_ALERT_RATIO:
        return [_event(ReportEvent.SYMMETRY_TREND_RISING_STEEPLY, detail)], ratio
    if ratio >= _TREND_WARN_RATIO:
        return [_event(ReportEvent.SYMMETRY_TREND_RISING, detail)], ratio
    return [], ratio


def build_signals(
    ladder: dict[str, Any] | None,
    tournament: dict[str, Any] | None,
    symmetry: dict[str, Any] | None,
    pvc: dict[str, Any] | None,
    target_entropy: dict[str, Any] | None,
    arena: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The front-page tiles: one per measured signal, with its raised events.

    Each tile states the latest value and the numbers behind any event it
    raised. Statuses come from :data:`EVENT_DEFINITIONS`, so a tile is only
    coloured by a named, documented rule.
    """
    signals: list[dict[str, Any]] = []

    # 1. Pentobi ladder — win rate against an outside engine.
    if ladder is None:
        signals.append(
            _not_recorded(
                "ladder",
                "Pentobi ladder",
                "Not run",
                "No PentobiLadder results and no MiniLadder history in this run directory.",
                anchored=True,
                href="#ladder",
            )
        )
    else:
        history = ladder["history"]
        spark = [p["weighted_score"] for p in history] if len(history) > 1 else None
        best = ladder["keep_best"]
        events: list[dict[str, Any]] = []
        if ladder["drift"] is not None:
            drift = ladder["drift"]
            events.append(
                _event(
                    ReportEvent.LADDER_DRIFT_BREAKER_TRIPPED,
                    f"{drift['consecutive_drops']} consecutive evaluations below {drift['best_before']} "
                    f"at {drift['best_score']:.3f}; tripped at {drift['tripped_at']} "
                    f"({drift['tripped_score']:.3f}).",
                )
            )
        elif ladder["alarm_file"]:
            events.append(
                _event(ReportEvent.LADDER_DRIFT_BREAKER_TRIPPED, "MiniLadder/DRIFT_ALARM present in the run directory.")
            )
        if best and len(history) >= 2 and history[-1]["weighted_score"] < best["weighted_score"] - 1e-9:
            events.append(
                _event(
                    ReportEvent.LADDER_LATEST_BELOW_BEST,
                    f"Latest {history[-1]['label']} at {history[-1]['weighted_score']:.3f}; "
                    f"best {best['label']} at {best['weighted_score']:.3f}.",
                )
            )
        level = f"L{best['pentobi_level']} · " if best and best.get("pentobi_level") is not None else ""
        value = f"{level}{best['weighted_score']:.3f} weighted" if best else "—"
        if best:
            sub = f"Best of {len(history)} evaluation{'s' if len(history) != 1 else ''}: {best['label']}."
        else:
            sub = "No weighted score recorded."
        signals.append(
            _signal("ladder", "Pentobi ladder", value, sub, anchored=True, events=events, spark=spark, href="#ladder")
        )

    # 2. Pooled BayesElo tournament — one shared scale over the run's checkpoints.
    if tournament is None:
        signals.append(
            _not_recorded(
                "tournament",
                "Pool Elo",
                "Not run",
                "No Tournament/tournament_ratings.parquet in this run directory.",
                anchored=True,
                href="#pool-elo",
            )
        )
    else:
        ratings = tournament["rating"]
        peak = max(ratings)
        peak_gen = tournament["gens"][ratings.index(peak)]
        final = ratings[-1]
        drop = peak - final
        events = []
        if final <= _POOL_ELO_ALERT_FINAL_BELOW_ANCHOR:
            events.append(
                _event(ReportEvent.POOL_ELO_BELOW_ANCHOR, f"Final rating {final:+.0f} against the gen-0 anchor at 0.")
            )
        if drop >= _POOL_ELO_ALERT_DROP:
            events.append(
                _event(
                    ReportEvent.POOL_ELO_FAR_BELOW_PEAK,
                    f"Peak {peak:+.0f} at generation {peak_gen}; final {final:+.0f}; difference {drop:.0f} Elo.",
                )
            )
        elif drop >= _POOL_ELO_WARN_DROP:
            events.append(
                _event(
                    ReportEvent.POOL_ELO_BELOW_PEAK,
                    f"Peak {peak:+.0f} at generation {peak_gen}; final {final:+.0f}; difference {drop:.0f} Elo.",
                )
            )
        signals.append(
            _signal(
                "tournament",
                "Pool Elo",
                f"{final:+.0f} final",
                f"Peak {peak:+.0f} at generation {peak_gen}.",
                anchored=True,
                events=events,
                spark=ratings,
                href="#pool-elo",
            )
        )

    # 3. Policy symmetry KL — the policy's disagreement across game symmetries.
    kl_values = symmetry["kl_mean"] if symmetry else None
    if not kl_values:
        signals.append(
            _not_recorded(
                "symmetry_kl",
                "Policy symmetry KL",
                "Not recorded",
                "SymmetryDiagnostic table absent from this run directory.",
                anchored=True,
                href="#diagnostics",
            )
        )
    else:
        events, ratio = _trend_events(kl_values, "Policy symmetry KL")
        trend = f" {ratio:.2f}× its first-generations mean." if ratio is not None else ""
        signals.append(
            _signal(
                "symmetry_kl",
                "Policy symmetry KL",
                f"{kl_values[-1]:.3f} nats",
                f"Final generation.{trend}",
                anchored=True,
                events=events,
                spark=kl_values,
                href="#diagnostics",
            )
        )

    # 4. Value symmetry MAE — the same measurement for the value head.
    mae_values = pvc.get("value_symmetry_mae") if pvc else None
    if not mae_values:
        signals.append(
            _not_recorded(
                "value_mae",
                "Value symmetry MAE",
                "Not recorded",
                "PolicyValueConsistency table absent, or predates the value-symmetry column.",
                anchored=True,
                href="#diagnostics",
            )
        )
    else:
        events, ratio = _trend_events(mae_values, "Value symmetry MAE")
        trend = f" {ratio:.2f}× its first-generations mean." if ratio is not None else ""
        signals.append(
            _signal(
                "value_mae",
                "Value symmetry MAE",
                f"{mae_values[-1]:.3f}",
                f"Final generation.{trend}",
                anchored=True,
                events=events,
                spark=mae_values,
                href="#diagnostics",
            )
        )

    # 5. Self-play target entropy — spread of the stored search targets.
    if target_entropy is None:
        signals.append(
            _not_recorded(
                "target_entropy",
                "Target entropy",
                "Not recorded",
                "SelfPlayProfiling table absent from this run directory.",
                anchored=True,
                href="#diagnostics",
            )
        )
    else:
        means = target_entropy["mean"]
        median = sorted(means)[len(means) // 2]
        min_value = min(means)
        min_gen = target_entropy["gens"][means.index(min_value)]
        latest = means[-1]
        events = []
        if median > 0 and min_value < _ENTROPY_COLLAPSE_RATIO * median:
            events.append(
                _event(
                    ReportEvent.TARGET_ENTROPY_GENERATION_COLLAPSE,
                    f"Generation {min_gen} at {min_value:.3f} nats against a run median of {median:.3f}.",
                )
            )
        if median > 0 and latest < _ENTROPY_WARN_RATIO * median:
            events.append(
                _event(
                    ReportEvent.TARGET_ENTROPY_LATEST_BELOW_MEDIAN,
                    f"Final generation at {latest:.3f} nats against a run median of {median:.3f}.",
                )
            )
        signals.append(
            _signal(
                "target_entropy",
                "Target entropy",
                f"{latest:.3f} nats",
                f"Final generation. Run median {median:.3f} nats, minimum {min_value:.3f} at generation {min_gen}.",
                anchored=True,
                events=events,
                spark=means,
                href="#diagnostics",
            )
        )

    # 6. Arena — the run's own candidate-versus-incumbent games.
    if arena is None:
        signals.append(
            _not_recorded(
                "arena",
                "Arena score",
                "Not recorded",
                "ArenaData table absent from this run directory.",
                anchored=False,
                href="#arena",
            )
        )
    else:
        events = list(arena["events"])
        white = arena.get("white_rate")
        if white is not None:
            sub = f"White won {white:.0%} of decisive games."
        else:
            sub = "Per-colour split not logged for this run."
        signals.append(
            _signal(
                "arena",
                "Arena score",
                f"{arena['score'][-1]:.3f} final",
                sub,
                anchored=False,
                events=events,
                spark=arena["score"],
                href="#arena",
            )
        )

    return signals


def build_status_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """A count of what the signals raised, at the top of the report.

    Purely a tally: how many events fired, on which signals, and which signals
    were not recorded. It draws no conclusion about the run — the events and
    their triggers are listed in the report's key, and the charts below carry
    the numbers.
    """
    raised = [(signal, event) for signal in signals for event in signal["events"]]
    fired = [(s, e) for s, e in raised if e["id"] != str(ReportEvent.NOT_RECORDED)]
    not_recorded = [s["label"] for s in signals if s["status"] == str(SignalStatus.MISSING)]
    statuses = [SignalStatus(s["status"]) for s in signals if s["anchored"]]
    present = [s for s in statuses if s is not SignalStatus.MISSING]

    status = max(present, key=lambda s: _STATUS_ORDER[s]) if present else SignalStatus.MISSING

    if fired:
        signal_count = len({s["id"] for s, _ in fired})
        headline = (
            f"{len(fired)} event{'s' if len(fired) != 1 else ''} raised on {signal_count} of {len(signals)} signals"
        )
    elif not present:
        headline = "No signal measured against an outside reference was recorded"
    else:
        headline = f"No events raised on {len(signals) - len(not_recorded)} recorded signals"

    parts: list[str] = []
    if fired:
        parts.append("Raised: " + "; ".join(f"{e['label']} ({s['label']})" for s, e in fired) + ".")
    if not_recorded:
        parts.append("Not recorded: " + ", ".join(not_recorded) + ".")
    parts.append("Triggers and thresholds are listed in the key.")
    return {"status": str(status), "headline": headline, "detail": " ".join(parts)}


# ---------------------------------------------------------------------------
# Meta / config
# ---------------------------------------------------------------------------


def meta_payload(config: RunConfig, generations_seen: int, missing_tables: list[str]) -> dict[str, Any]:
    """Run identity, headline config chips, and the full config table."""
    net = config.net_config
    mcts = config.mcts_config
    chips = [
        f"{config.game}",
        f"{generations_seen or config.num_generations} generations",
        f"{config.num_eps:,} games/gen",
        f"net {net.num_filters}f × {net.num_residual_blocks}b",
        f"{mcts.num_mcts_sims} sims ({mcts.search_policy})",
        f"backend {config.selfplay_backend}",
        f"gate {config.gate_mode}" + (" · paired" if config.paired_arena else ""),
    ]
    rows = [
        ("Game", config.game),
        ("Generations", config.num_generations),
        ("Episodes / generation", config.num_eps),
        ("Self-play backend", config.selfplay_backend),
        ("MCTS simulations", mcts.num_mcts_sims),
        ("Search policy", mcts.search_policy),
        ("CPUCT", mcts.cpuct),
        ("Arena matches", config.num_arena_matches),
        ("Paired arena", config.paired_arena),
        ("Gate mode", config.gate_mode),
        ("Update threshold", config.update_threshold),
        ("Guard floor", config.guard_floor),
        ("Replay buffer (games)", config.replay_buffer_games),
        ("Buffer staleness (gens)", round(config.replay_buffer_games / max(config.num_eps, 1), 1)),
        ("Emergent reuse (E×B/F)", round(net.epochs * config.replay_buffer_games / max(config.num_eps, 1), 1)),
        ("Learning rate", net.learning_rate),
        ("LR scheduler", net.lr_scheduler or "constant"),
        ("Weight decay", net.weight_decay),
        ("Batch size", net.batch_size),
        ("Epochs", net.epochs),
        ("Residual blocks", net.num_residual_blocks),
        ("Filters", net.num_filters),
        ("Dropout", net.dropout),
        ("CUDA", net.cuda),
        ("Seed", config.seed),
    ]
    return {
        "run_name": config.run_name,
        "game": config.game,
        "generations": generations_seen or config.num_generations,
        "date": datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d"),
        "chips": chips,
        "config_rows": [[str(k), str(v)] for k, v in rows],
        "missing_tables": missing_tables,
    }


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def build_report_payload(config: RunConfig) -> dict[str, Any]:
    """Assemble the full JSON payload the report page renders from.

    Every section is independently optional: the page shows an explicit
    "not recorded" placeholder for anything absent rather than failing or
    silently omitting it.
    """
    from alphablokus.reporting.arena_replays import build_replay_payload, load_sampled_replays

    tables: dict[str, pd.DataFrame | None] = {
        "TrainingData": load_metrics(config.training_data_directory),
        "ArenaData": load_metrics(config.arena_data_directory),
        "Timings": load_metrics(config.timings_directory),
        "ResourceUsage": load_metrics(config.resource_usage_directory),
        "SelfPlayProfiling": load_metrics(config.self_play_profiling_directory),
        "TrainingThroughput": load_metrics(config.training_throughput_directory),
        "TrainingEntropy": load_metrics(config.training_entropy_directory),
        "PolicyAccuracy": load_metrics(config.policy_accuracy_directory),
        "ValueCalibration": load_metrics(config.value_calibration_directory),
        "PolicyValueConsistency": load_metrics(config.policy_value_consistency_directory),
        "LearningRate": load_metrics(config.learning_rate_directory),
        "RollingElo": load_metrics(config.rolling_elo_directory),
        "EloRatings": load_metrics(config.run_directory / "EloRatings"),
        "SymmetryDiagnostic": load_metrics(config.symmetry_diagnostic_directory),
    }

    ladder = ladder_payload(config)
    tournament = tournament_payload(config.tournament_directory / "tournament_ratings.parquet")
    symmetry = symmetry_payload(tables["SymmetryDiagnostic"])
    pvc = pvc_payload(tables["PolicyValueConsistency"])
    target_entropy = target_entropy_payload(tables["SelfPlayProfiling"])
    arena = arena_payload(tables["ArenaData"], config.update_threshold, config.gate_mode)

    replays_df = (
        load_sampled_replays(config.arena_replays_directory) if config.arena_replays_directory.exists() else None
    )
    replays = build_replay_payload(replays_df, config) if replays_df is not None and not replays_df.empty else None

    signals = build_signals(ladder, tournament, symmetry, pvc, target_entropy, arena)
    summary = build_status_summary(signals)

    generations_seen = 0
    for candidates in (arena, tables["SelfPlayProfiling"]):
        if isinstance(candidates, dict) and candidates.get("gens"):
            generations_seen = max(generations_seen, int(max(candidates["gens"])))
        elif isinstance(candidates, pd.DataFrame):
            generations_seen = max(generations_seen, int(candidates["generation"].max()))

    missing = [name for name, df in tables.items() if df is None]

    payload: dict[str, Any] = {
        "meta": meta_payload(config, generations_seen, missing),
        "summary": summary,
        "signals": signals,
        "event_key": event_key_payload(),
        "ladder": ladder,
        "tournament": tournament,
        "symmetry": symmetry,
        "pvc": pvc,
        "target_entropy": target_entropy,
        "arena": arena,
        "rolling_elo": rolling_elo_payload(tables["RollingElo"]),
        "legacy_elo": legacy_elo_payload(tables["EloRatings"]),
        "training": training_payload(tables["TrainingData"]),
        "accuracy": accuracy_payload(tables["PolicyAccuracy"]),
        "net_entropy": net_entropy_payload(tables["TrainingEntropy"]),
        "calibration": calibration_payload(tables["ValueCalibration"]),
        "lr": learning_rate_payload(tables["LearningRate"]),
        "selfplay": selfplay_payload(tables["SelfPlayProfiling"]),
        "perf": perf_payload(tables["Timings"], tables["TrainingThroughput"], tables["ResourceUsage"]),
        "replays": replays,
    }
    return payload
