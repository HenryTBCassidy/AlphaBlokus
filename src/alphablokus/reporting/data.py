"""Report payload assembly — every metric table reduced to plain JSON-ready dicts.

The end-of-run report is a single self-contained HTML page whose charts are
rendered client-side from one embedded JSON payload. This module builds that
payload. Design rules, learned the hard way (docs/research/plateau-investigation.md
R8, docs/research/regression-and-next-steps.md §1.5):

1. **Every table is optional.** Runs sync partially, schemas evolve, and older
   runs predate newer diagnostics. A missing table renders as an explicit
   "not recorded" state — absence of evidence must be visible, never silent.
2. **Externally-anchored signals are separated from self-referential ones.**
   Loss, acceptance and eval-set agreement are measured against the loop's own
   outputs and can look healthy while the run regresses. Symmetry KL, value
   symmetry MAE (ground-truth game invariances), the Pentobi ladder and the
   pooled tournament are the signals that cannot be gamed by the loop, and the
   payload tags every section accordingly (``anchored``).
3. **Known failure signatures are computed, not eyeballed.** Exact-0.500 arena
   scores, sub-binomial score variance, colour pinning, target-entropy collapse
   and rising symmetry error each get an automatic status.
"""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from alphablokus.evaluation.ladder_selection import (
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
# Signal thresholds — each cites the run that motivated it.
# ---------------------------------------------------------------------------

# Symmetry KL / value-symmetry MAE trend: the paired-gate rerun's KL rose
# 0.64 → 1.24 (~1.9×) and value MAE 0.10 → 0.25 (~2.5×) while every
# self-referential signal looked healthy (regression-and-next-steps §1.3).
# Trend = mean(last k) / mean(first k); a genuinely healthy run (v3) ended
# roughly where it started.
_TREND_WARN_RATIO = 1.20
_TREND_ALERT_RATIO = 1.50

# Self-play target-entropy collapse: gen 17 of the rerun dropped to 0.506 nats
# against a run median of ~0.85 (ratio 0.60) and preceded the terminal slide
# (regression-and-next-steps §1.4).
_ENTROPY_COLLAPSE_RATIO = 0.70
_ENTROPY_WARN_RATIO = 0.85

# Arena instrument red flags (plateau-investigation §2 B8 / R8c): exact-0.500
# scores are colour-split clones; score variance far below binomial means
# something systematic (colour), not net strength, decides games; a white share
# of decisive games ≥ 85% means the gate is measuring first-mover advantage.
_SUB_BINOMIAL_FACTOR = 0.5
_SUB_BINOMIAL_MIN_GENS = 4
_COLOUR_PINNED_WHITE_RATE = 0.85

# Pool-Elo slippage: how far below its own peak a run may finish before the
# verdict flags it (the rerun finished 49 Elo below its +5.5 peak).
_POOL_ELO_WARN_DROP = 15.0
_POOL_ELO_ALERT_DROP = 30.0

# Bound embedded chart payloads: per-batch loss traces are EWM-smoothed then
# downsampled to at most this many points per series.
_TIMELINE_MAX_POINTS = 1200


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
    """Per-generation arena tallies + the automatic instrument red flags.

    The red flags are the R8c checks: exact-0.500 scores, sub-binomial score
    variance, and white-win skew (when the per-colour split was logged).
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

    flags: list[str] = []
    exact_half = int((scores.sub(0.5).abs() < 1e-9).sum())
    if exact_half > 0:
        flags.append(
            f"{exact_half} generation(s) scored exactly 0.500 — the signature of colour-split clones, not a true tie."
        )
    sub_binomial = False
    if len(scores) >= _SUB_BINOMIAL_MIN_GENS:
        mean_p = float(scores.mean())
        mean_n = float(total.mean())
        sigma0 = (mean_p * (1.0 - mean_p) / mean_n) ** 0.5 if 0.0 < mean_p < 1.0 and mean_n > 0 else 0.0
        observed_std = float(scores.std(ddof=1))
        if sigma0 > 0 and observed_std < _SUB_BINOMIAL_FACTOR * sigma0:
            sub_binomial = True
            flags.append(
                f"Per-generation score variance is sub-binomial (observed σ={observed_std:.3f} vs binomial "
                f"σ₀≈{sigma0:.3f}) — outcomes are not independent draws in net strength; something systematic "
                "is deciding games."
            )

    white_rate: float | None = None
    if "white_wins" in df.columns and "black_wins" in df.columns:
        decisive = float((df["white_wins"] + df["black_wins"]).sum())
        if decisive > 0:
            white_rate = float(df["white_wins"].sum()) / decisive
            if white_rate >= _COLOUR_PINNED_WHITE_RATE:
                flags.append(
                    f"White won {white_rate:.0%} of decisive arena games — the gate is colour-pinned "
                    "(first-mover advantage swamps net strength)."
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
        "red_flags": flags,
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

    This is the entropy of the stored search-policy targets at harvest — the
    structure of what the trainer is being asked to fit. A sharp collapse
    (rerun gen 17: 0.79 → 0.506 nats) preceded the worst regression on record.
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
    """Rolling arena-derived Elo (chained, self-referential telemetry)."""
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
    """Eval-set policy agreement per generation (self-referential for Blokus)."""
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
# Front-page signals + verdict
# ---------------------------------------------------------------------------


def _signal(
    signal_id: str,
    label: str,
    status: str,
    value: str,
    sub: str,
    *,
    anchored: bool,
    spark: list[float] | None = None,
    href: str | None = None,
) -> dict[str, Any]:
    return {
        "id": signal_id,
        "label": label,
        "status": status,
        "value": value,
        "sub": sub,
        "anchored": anchored,
        "spark": spark,
        "href": href,
    }


def _trend_status(values: list[float] | None, *, rising_is_bad: bool) -> tuple[str, float | None]:
    """Classify a metric trend as ok / warn / alert using the shared ratios."""
    if not values:
        return "missing", None
    ratio = _trend_ratio(values)
    if ratio is None:
        return "ok", None
    effective = ratio if rising_is_bad else (1.0 / ratio if ratio > 0 else float("inf"))
    if effective >= _TREND_ALERT_RATIO:
        return "alert", ratio
    if effective >= _TREND_WARN_RATIO:
        return "warn", ratio
    return "ok", ratio


def build_signals(
    ladder: dict[str, Any] | None,
    tournament: dict[str, Any] | None,
    symmetry: dict[str, Any] | None,
    pvc: dict[str, Any] | None,
    target_entropy: dict[str, Any] | None,
    arena: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The front-page status tiles: is this run improving, or fooling itself?

    Every tile is an externally-anchored (or instrument-health) signal; the
    self-referential metrics deliberately get no tile — loss going down is not
    evidence of progress in this system (regression-and-next-steps §1.5).
    """
    signals: list[dict[str, Any]] = []

    # 1. Pentobi ladder — the run's real verdict.
    if ladder is None:
        signals.append(
            _signal(
                "ladder",
                "Pentobi ladder",
                "missing",
                "Not run",
                "The only instrument that resolves what the arena calls a tie — run scripts/mini_ladder.py.",
                anchored=True,
                href="#external",
            )
        )
    else:
        history = ladder["history"]
        spark = [p["weighted_score"] for p in history] if len(history) > 1 else None
        best = ladder["keep_best"]
        if ladder["drift"] is not None or ladder["alarm_file"]:
            status = "alert"
            sub = "Drift circuit-breaker tripped — resume from the keep-best checkpoint."
        elif len(history) >= 2 and history[-1]["weighted_score"] < best["weighted_score"] - 1e-9:
            status = "warn"
            sub = f"Latest checkpoint below keep-best {best['label']}."
        else:
            status = "ok"
            sub = f"Keep-best: {best['label']}" if best else "Single evaluation."
        level = f"L{best['pentobi_level']} · " if best and best.get("pentobi_level") is not None else ""
        value = f"{level}{best['weighted_score']:.3f} weighted" if best else "—"
        signals.append(_signal("ladder", "Pentobi ladder", status, value, sub, anchored=True, spark=spark, href="#external"))

    # 2. Pooled BayesElo tournament — rigorous, non-saturating, independent code path.
    if tournament is None:
        signals.append(
            _signal(
                "tournament",
                "Pool Elo (BayesElo)",
                "missing",
                "Not run",
                "End-of-run pooled tournament absent — enable tournament.run_at_end or run scripts/tournament_elo.py.",
                anchored=True,
                href="#external",
            )
        )
    else:
        ratings = tournament["rating"]
        peak = max(ratings)
        peak_gen = tournament["gens"][ratings.index(peak)]
        final = ratings[-1]
        drop = peak - final
        status = "alert" if drop >= _POOL_ELO_ALERT_DROP else "warn" if drop >= _POOL_ELO_WARN_DROP else "ok"
        signals.append(
            _signal(
                "tournament",
                "Pool Elo (BayesElo)",
                status,
                f"{final:+.0f} final",
                f"Peak {peak:+.0f} at gen {peak_gen}." + (" Run ended below its best." if status != "ok" else ""),
                anchored=True,
                spark=ratings,
                href="#external",
            )
        )

    # 3. Policy symmetry KL — ground-truth invariance; rising = drifting net.
    kl_values = symmetry["kl_mean"] if symmetry else None
    status, ratio = _trend_status(kl_values, rising_is_bad=True)
    if status == "missing":
        signals.append(
            _signal(
                "symmetry_kl",
                "Policy symmetry KL",
                "missing",
                "Not recorded",
                "SymmetryDiagnostic table absent — the earliest honest drift warning is dark.",
                anchored=True,
                href="#external",
            )
        )
    else:
        assert kl_values is not None
        trend = f"{ratio:.2f}× vs run start" if ratio is not None else "trend n/a"
        sub = {"ok": f"Stable ({trend}).", "warn": f"Rising ({trend}).", "alert": f"Rising steeply ({trend}) — the rerun's regression signature."}[status]
        signals.append(
            _signal(
                "symmetry_kl",
                "Policy symmetry KL",
                status,
                f"{kl_values[-1]:.3f} nats",
                sub,
                anchored=True,
                spark=kl_values,
                href="#external",
            )
        )

    # 4. Value symmetry MAE — same instrument for the value head.
    mae_values = pvc.get("value_symmetry_mae") if pvc else None
    status, ratio = _trend_status(mae_values, rising_is_bad=True)
    if status == "missing":
        signals.append(
            _signal(
                "value_mae",
                "Value symmetry MAE",
                "missing",
                "Not recorded",
                "PolicyValueConsistency table absent (or predates the value-symmetry column).",
                anchored=True,
                href="#external",
            )
        )
    else:
        assert mae_values is not None
        trend = f"{ratio:.2f}× vs run start" if ratio is not None else "trend n/a"
        sub = {"ok": f"Stable ({trend}).", "warn": f"Rising ({trend}).", "alert": f"Rising steeply ({trend}) — value head drifting off the game's invariances."}[status]
        signals.append(
            _signal(
                "value_mae",
                "Value symmetry MAE",
                status,
                f"{mae_values[-1]:.3f}",
                sub,
                anchored=True,
                spark=mae_values,
                href="#external",
            )
        )

    # 5. Self-play target entropy — collapse detector.
    if target_entropy is None:
        signals.append(
            _signal(
                "target_entropy",
                "Target entropy",
                "missing",
                "Not recorded",
                "SelfPlayProfiling table absent — target-collapse events are invisible.",
                anchored=True,
                href="#external",
            )
        )
    else:
        means = target_entropy["mean"]
        median = sorted(means)[len(means) // 2]
        min_value = min(means)
        min_gen = target_entropy["gens"][means.index(min_value)]
        latest = means[-1]
        if median > 0 and min_value < _ENTROPY_COLLAPSE_RATIO * median:
            status = "alert"
            sub = f"Collapse at gen {min_gen} ({min_value:.3f} vs median {median:.3f} nats) — the gen-17 signature."
        elif median > 0 and latest < _ENTROPY_WARN_RATIO * median:
            status = "warn"
            sub = f"Latest ({latest:.3f}) sits below the run median ({median:.3f} nats)."
        else:
            status = "ok"
            sub = f"Stable around {median:.3f} nats."
        signals.append(
            _signal(
                "target_entropy",
                "Target entropy",
                status,
                f"{latest:.3f} nats",
                sub,
                anchored=True,
                spark=means,
                href="#external",
            )
        )

    # 6. Arena instrument health — is the gate measuring anything?
    if arena is None:
        signals.append(
            _signal(
                "instrument",
                "Arena instrument",
                "missing",
                "Not recorded",
                "ArenaData table absent.",
                anchored=False,
                href="#instrument",
            )
        )
    else:
        flags = arena["red_flags"]
        if flags:
            status = "alert"
            value = f"{len(flags)} red flag" + ("s" if len(flags) > 1 else "")
            sub = flags[0]
        else:
            status = "ok"
            value = "No pinning signature"
            white = arena.get("white_rate")
            sub = f"White won {white:.0%} of decisive games." if white is not None else "Colour split not logged (older run)."
            if white is None:
                status = "warn"
                value = "Colour split unknown"
        signals.append(_signal("instrument", "Arena instrument", status, value, sub, anchored=False, href="#instrument"))

    return signals


def build_verdict(signals: list[dict[str, Any]], ladder: dict[str, Any] | None) -> dict[str, Any]:
    """The one-line answer at the top of the report.

    Derived only from externally-anchored signals. When no external instrument
    ran at all, the verdict says so explicitly — that state is precisely how a
    20-generation regression once hid behind healthy-looking internal curves.
    """
    anchored = [s for s in signals if s["anchored"]]
    present = [s for s in anchored if s["status"] != "missing"]
    alerts = [s for s in present if s["status"] == "alert"]
    warns = [s for s in present if s["status"] == "warn"]

    if not present:
        return {
            "status": "missing",
            "headline": "No external evidence recorded",
            "detail": "Every externally-anchored instrument is dark. Internal metrics below cannot certify progress "
            "— run the Pentobi mini-ladder and the pooled tournament before trusting this run.",
        }
    if alerts:
        named = ", ".join(s["label"] for s in alerts)
        return {
            "status": "alert",
            "headline": "Regression signals present",
            "detail": f"Externally-anchored alarms: {named}. Treat internal improvements as unverified.",
        }
    if warns:
        named = ", ".join(s["label"] for s in warns)
        return {
            "status": "warn",
            "headline": "External signals mixed",
            "detail": f"Watch: {named}.",
        }
    detail = "All recorded externally-anchored signals are stable."
    if ladder and ladder["keep_best"]:
        best = ladder["keep_best"]
        level = f" (beats Pentobi L{best['pentobi_level']})" if best.get("pentobi_level") is not None else ""
        detail = f"Keep-best checkpoint: {best['label']}{level}, weighted ladder score {best['weighted_score']:.3f}. " + detail
    missing = [s["label"] for s in anchored if s["status"] == "missing"]
    if missing:
        detail += f" Not recorded: {', '.join(missing)}."
    return {"status": "ok", "headline": "Externally consistent with improvement", "detail": detail}


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
    verdict = build_verdict(signals, ladder)

    generations_seen = 0
    for candidates in (arena, tables["SelfPlayProfiling"]):
        if isinstance(candidates, dict) and candidates.get("gens"):
            generations_seen = max(generations_seen, int(max(candidates["gens"])))
        elif isinstance(candidates, pd.DataFrame):
            generations_seen = max(generations_seen, int(candidates["generation"].max()))

    missing = [name for name, df in tables.items() if df is None]

    payload: dict[str, Any] = {
        "meta": meta_payload(config, generations_seen, missing),
        "verdict": verdict,
        "signals": signals,
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
