"""One command that runs several supervised arms and diffs them (plan N1).

Every row of ``docs/plans/supervised-network-improvements.md`` is a two-arm comparison,
so the comparison is built once here and reused. This is **not a framework**: it drives
``scripts/distill_sl.py`` once per arm and reads the run JSONs back.

Its real job is to make an *unfair* comparison hard to construct by accident, because
that has already happened once on this project — the score-head A/B was confounded by a
side effect of adding the head (it shifted the data shuffle) at four times the magnitude
of the effect being measured. Three mechanisms, all structural rather than advisory:

1. **Data and protocol are harness-level, not arm-level.** Corpus, seed, corpus
   fraction, holdout fraction, schedule and learning rate are given *once* and forwarded
   verbatim to every arm. An arm cannot set them.
2. **Arm flags are allow-listed.** An arm may only vary the auxiliary-head switches and
   their weights (:data:`ARM_FLAG_ALLOWLIST`). Anything else is refused unless it is
   named explicitly in ``--allow-varying``, which then appears in the output so a reader
   knows the comparison was deliberately loosened.
3. **The result is re-checked after the fact.** Each arm's run JSON records the settings
   it *resolved* — including the measured holdout leakage — and those are diffed across
   arms. A disagreement marks the comparison ``comparable: false``, is logged as an
   error, and exits non-zero. It still writes the table: the numbers are useful for
   diagnosis even when the comparison is void.

**The metric set** (plan N1) comes straight out of each arm's best epoch: value skill,
top-1 and top-3 agreement with Pentobi, per-colour value calibration, each auxiliary
head's own loss *against its own baseline*, and the holdout leakage figure that qualifies
all of them.

**The noise floor.** Where a technique has a mathematically inert arm — a head at weight
0 — run it and name it with ``--noise-floor-arm``. Its distance from the control is what
"no effect" measures on this data, and any treatment delta smaller than that is not an
effect. Without such an arm, use a second seed and read the table with that in mind.

Example — the N4 ownership A/B with a noise-floor arm::

    uv run python scripts/ab_harness.py \\
        --config run_configurations/blokus_cloud_v2.json \\
        --corpus ~/corpora/pentobi_l9_v2 --out-dir temp/ab/ownership \\
        --arm control \\
        --arm zero_weight="--ownership-head --ownership-loss-weight 0" \\
        --arm ownership="--ownership-head" \\
        --noise-floor-arm zero_weight
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The only flags an arm may carry. Everything that decides *what data the arm sees* is
#: harness-level and forwarded identically, so it cannot appear here — which is what
#: makes "same seed, same data, same corpus fraction" structural rather than a
#: convention someone has to remember.
ARM_FLAG_ALLOWLIST: frozenset[str] = frozenset(
    {
        "--score-head",
        "--no-score-head",
        "--score-loss-weight",
        "--score-scale",
        "--ownership-head",
        "--no-ownership-head",
        "--ownership-loss-weight",
        "--reply-head",
        "--no-reply-head",
        "--reply-loss-weight",
    }
)

#: Per-head resolved settings, read from the run JSON. Diffing the on/off switches alone
#: is not enough: two arms can both build the score head at *different* loss weights and
#: still look like they differ in one thing, so a weight change rides along unattributed
#: with whatever head is nominally under test.
HEAD_SETTINGS: dict[str, tuple[str, ...]] = {
    "score": ("score_head", "score_loss_weight", "score_scale"),
    "ownership": ("ownership_head", "ownership_loss_weight"),
    "reply": ("reply_head", "reply_loss_weight"),
}

#: Fields of the run JSON that every arm must agree on for the comparison to mean
#: anything. Read from what each arm *resolved*, not from what the harness intended, so
#: a config file quietly pinning one of them is caught too. ``holdout_leakage`` is the
#: strongest of them: it is a measurement of the actual split, so two arms agreeing on it
#: agree on the exam as well as the syllabus.
PROTOCOL_KEYS: tuple[str, ...] = (
    "corpus",
    "corpus_version",
    "num_games",
    "max_games",
    "holdout_fraction",
    "seed",
    "epsilon",
    "target_temperature",
    "opening_value",
    "opening_mix",
    "v1_mix",
    "augment",
    "lr",
    "net_size",
    "holdout_leakage",
)

#: ``(key, label, direction)`` per reported metric. ``direction`` says how to read a
#: delta: ``up`` better higher, ``down`` better lower, ``zero`` better nearer zero (a
#: calibration bias), ``flat`` no preferred direction (context, not a verdict).
METRICS: tuple[tuple[str, str, str], ...] = (
    ("value_skill", "value skill", "up"),
    ("top1", "top-1 vs Pentobi", "up"),
    ("top3", "top-3 vs Pentobi", "up"),
    ("policy_ce", "policy CE (nats)", "down"),
    ("policy_kl", "policy KL (nats)", "down"),
    ("value_mse", "value MSE", "down"),
    ("colour_only_value_mse", "value MSE, colour-only baseline", "flat"),
    ("bias_white", "value bias, White to move", "zero"),
    ("bias_black", "value bias, Black to move", "zero"),
    ("value_mse_white", "value MSE, White to move", "down"),
    ("value_mse_black", "value MSE, Black to move", "down"),
    ("score_skill", "score-head skill", "up"),
    ("score_mse", "score-head MSE", "down"),
    ("ownership_skill", "ownership-head skill", "up"),
    ("ownership_accuracy", "ownership-head accuracy", "up"),
    ("ownership_ce", "ownership-head CE (nats)", "down"),
    ("reply_top1", "reply-head top-1", "up"),
    ("reply_ce", "reply-head CE (nats)", "down"),
    ("best_epoch", "best epoch", "flat"),
    ("num_params", "parameters", "flat"),
    ("leaked_fraction_mirror", "holdout leakage (mirror)", "down"),
)


@dataclass(frozen=True)
class Arm:
    """One arm: a name and the extra ``distill_sl.py`` flags that define it."""

    name: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ArmSummary:
    """One arm's best-epoch metrics, flattened to the names :data:`METRICS` uses."""

    name: str
    heads: dict[str, bool]
    metrics: dict[str, float | None]
    protocol: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def head_profiles(self) -> dict[str, tuple[Any, ...] | None]:
        """Each head as ``None`` when off, else its resolved settings.

        A head that is off in both arms cannot differ, whatever its weight says — so a
        disabled head collapses to ``None`` rather than to its inherited numbers.
        """
        profiles: dict[str, tuple[Any, ...] | None] = {}
        for head, keys in HEAD_SETTINGS.items():
            enabled, *rest = keys
            profiles[head] = tuple(self.settings.get(key) for key in rest) if self.settings.get(enabled) else None
        return profiles


def parse_arm(spec: str) -> Arm:
    """``"name=--flag --other 0.3"`` → an :class:`Arm`; ``"name"`` means no extra flags.

    Raises:
        SystemExit: If the name is empty or the flags do not tokenise as a shell word
            list, both of which are typos worth catching before a multi-hour GPU run.
    """
    name, separator, raw_flags = spec.partition("=")
    name = name.strip()
    if not name:
        raise SystemExit(f"Arm spec {spec!r} has no name; expected NAME or NAME=<flags>.")
    if not separator:
        return Arm(name=name, flags=())
    try:
        flags = tuple(shlex.split(raw_flags))
    except ValueError as error:
        raise SystemExit(f"Arm {name!r}: could not parse flags {raw_flags!r} ({error}).") from error
    return Arm(name=name, flags=flags)


def validate_arm_flags(arm: Arm, allow_varying: Sequence[str]) -> None:
    """Refuse an arm that varies anything outside the allowlist.

    This is the guard that stops an unfair comparison being *constructed*, as opposed to
    detected afterwards: an arm that quietly carried ``--seed`` or ``--max-games`` would
    produce a table whose two columns were never comparable in the first place.

    Raises:
        SystemExit: If the arm carries a flag that is neither allow-listed nor named in
            ``allow_varying``, or a positional argument (always a mistake here).
    """
    permitted = ARM_FLAG_ALLOWLIST | set(allow_varying)
    for token in arm.flags:
        if not token.startswith("--"):
            continue  # a value for the flag before it
        if token in permitted:
            continue
        raise SystemExit(
            f"Arm {arm.name!r} varies {token!r}, which is not an arm-level setting. "
            f"Arms may only vary {sorted(ARM_FLAG_ALLOWLIST)}; everything else "
            "(corpus, seed, corpus fraction, schedule) is set once for all arms so the "
            f"comparison stays controlled. Pass --allow-varying {token} to override "
            "deliberately — it will be recorded in the comparison."
        )
    if arm.flags and not arm.flags[0].startswith("--"):
        raise SystemExit(f"Arm {arm.name!r} starts with a positional argument {arm.flags[0]!r}; expected a flag.")


def freeze_corpus(corpus: Path, destination: Path) -> int:
    """Symlink the corpus shards that exist *now* into ``destination``; return the count.

    A corpus being generated grows every few minutes. Arms run one after another, so a
    later arm globs more shards than an earlier one, ``--max-games`` then samples a
    *different* set of games, and the arms end up sitting different exams — which is
    exactly how the first score-head A/B was wasted (one arm's holdout held 11,804 scored
    rows, another's 13,165).

    Symlinks, not copies: a snapshot of a 30 GB corpus should not cost 30 GB, and the
    shards are only ever read. Files that appear afterwards are simply not in the
    snapshot, which is the point.
    """
    shards = 0
    for source in (corpus, corpus / "games", corpus / "opening"):
        if not source.is_dir():
            continue
        found = sorted(source.glob("*.parquet"))
        if not found:
            continue
        target = destination if source == corpus else destination / source.name
        target.mkdir(parents=True, exist_ok=True)
        for path in found:
            (target / path.name).symlink_to(path.resolve())
        shards += len(found)
    if not shards:
        raise SystemExit(f"No .parquet shards under {corpus} — nothing to compare on.")
    return shards


def shared_flags(args: argparse.Namespace, arm: Arm | None = None) -> list[str]:
    """The ``distill_sl.py`` flags every arm gets verbatim — the controlled half.

    Verbatim with one deliberate exception: the noise-floor arm runs at
    ``--noise-floor-seed``. It is the control repeated under a different roll of the
    dice, so the spread between the two is what "no effect" looks like on this data.
    """
    seed = args.noise_floor_seed if arm is not None and arm.name == args.noise_floor_arm else args.seed
    flags: list[str] = [
        "--config",
        str(args.config),
        "--corpus",
        str(args.frozen_corpus or args.corpus),
        "--arms",
        args.distill_arm,
        "--holdout-frac",
        str(args.holdout_frac),
        "--seed",
        str(seed),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--tau",
        str(args.tau),
        "--opening-value",
        args.opening_value,
        "--opening-mix",
        str(args.opening_mix),
        "--v1-mix",
        str(args.v1_mix),
        "--augment" if args.augment else "--no-augment",
    ]
    if args.warm_start:
        flags += ["--warm-start", str(args.warm_start)]
    if args.net_size:
        flags += ["--net-size", args.net_size]
    if args.epsilon is not None:
        flags += ["--epsilon", str(args.epsilon)]
    if args.max_games is not None:
        flags += ["--max-games", str(args.max_games)]
    if args.v1_corpus is not None:
        flags += ["--v1-corpus", str(args.v1_corpus)]
    return flags


def build_command(arm: Arm, args: argparse.Namespace) -> list[str]:
    """The full ``distill_sl.py`` command line for one arm.

    Arm flags come **last**, so an allow-listed override genuinely overrides — argparse
    keeps the final occurrence.
    """
    out_dir = Path(args.out_dir)
    return [
        sys.executable,
        str(Path(__file__).with_name("distill_sl.py")),
        *shared_flags(args, arm),
        "--ckpt-dir",
        str(out_dir / arm.name),
        "--out",
        str(out_dir / f"{arm.name}.json"),
        *arm.flags,
    ]


def _calibration(diagnostics: dict[str, Any], player: int) -> dict[str, Any] | None:
    """One side-to-move's calibration row, or ``None`` if that colour is absent."""
    for row in diagnostics.get("calibration", ()):
        if int(row["player"]) == player:
            return row
    return None


def summarise_arm(name: str, payload: dict[str, Any], distill_arm: str) -> ArmSummary:
    """Flatten one arm's run JSON into the metric names :data:`METRICS` reports.

    Reads the **best** epoch, not the last: every arm early-stops on held-out policy CE,
    so comparing final epochs would compare arms that stopped at different points.

    Raises:
        SystemExit: If the run JSON does not contain the requested sub-arm — usually a
            harness invoked with ``--distill-arm warm`` against scratch-only runs.
    """
    arms = payload.get("arms", {})
    if distill_arm not in arms:
        raise SystemExit(f"Arm {name!r}: its run JSON has no {distill_arm!r} arm (found {sorted(arms)}).")
    run = arms[distill_arm]
    diagnostics = run["best_diagnostics"]
    aux = run.get("best_aux", {})
    white = _calibration(diagnostics, 1)
    black = _calibration(diagnostics, -1)
    score = aux.get("score")
    ownership = aux.get("ownership")
    reply = aux.get("reply")
    leakage = payload.get("holdout_leakage", {})

    metrics: dict[str, float | None] = {
        "value_skill": run.get("best_value_skill"),
        "top1": diagnostics.get("top1_accuracy"),
        "top3": diagnostics.get("top3_accuracy"),
        "policy_ce": run["best"]["policy_ce"],
        "policy_kl": run["best"]["policy_kl"],
        "value_mse": diagnostics.get("value_mse"),
        "colour_only_value_mse": diagnostics.get("colour_only_value_mse"),
        "bias_white": None if white is None else white["mean_predicted"] - white["mean_outcome"],
        "bias_black": None if black is None else black["mean_predicted"] - black["mean_outcome"],
        "value_mse_white": None if white is None else white["value_mse"],
        "value_mse_black": None if black is None else black["value_mse"],
        "score_skill": None if score is None else score["score_skill"],
        "score_mse": None if score is None else score["score_mse"],
        "ownership_skill": None if ownership is None else ownership["skill"],
        "ownership_accuracy": None if ownership is None else ownership["accuracy"],
        "ownership_ce": None if ownership is None else ownership["cross_entropy"],
        "reply_top1": None if reply is None else reply["top1_accuracy"],
        "reply_ce": None if reply is None else reply["policy_ce"],
        "best_epoch": run.get("best_epoch"),
        "num_params": run.get("num_params"),
        "leaked_fraction_mirror": leakage.get("leaked_fraction_mirror"),
    }
    return ArmSummary(
        name=name,
        heads=run.get("heads", {}),
        metrics=metrics,
        protocol={key: payload.get(key) for key in PROTOCOL_KEYS},
        settings={key: payload.get(key) for keys in HEAD_SETTINGS.values() for key in keys},
    )


def check_comparable(
    summaries: Sequence[ArmSummary],
    allow_varying: Sequence[str],
    noise_floor: str | None = None,
) -> list[str]:
    """Every way the arms failed to be a controlled comparison, in plain words.

    An empty list means the only differences between the arms are the ones their flags
    asked for. Anything else is reported rather than silently tolerated: a table of two
    arms trained on different data is worse than no table, because it looks like a
    result.
    """
    if len(summaries) < 2:
        return []
    complaints: list[str] = []
    control = summaries[0]
    base_profiles = control.head_profiles()
    for other in summaries[1:]:
        replicate = other.name == noise_floor
        for key in PROTOCOL_KEYS:
            # The noise-floor arm is the control re-run at a different seed: that one
            # difference is the whole point of it, and is checked separately below.
            if replicate and key == "seed":
                continue
            if control.protocol.get(key) != other.protocol.get(key):
                complaints.append(
                    f"{key}: {control.name}={control.protocol.get(key)!r} but {other.name}={other.protocol.get(key)!r}"
                )
        profiles = other.head_profiles()
        differing = sorted(head for head, value in profiles.items() if value != base_profiles.get(head))
        if replicate:
            # A replicate must be identical in every head *and* actually differently
            # seeded — otherwise it is bit-identical to the control, its delta is 0 on
            # every metric, and "below noise" can never fire for anybody.
            if differing:
                complaints.append(
                    f"noise-floor arm {other.name} differs from {control.name} in {', '.join(differing)}; "
                    "it must be a pure replicate — same settings, different seed"
                )
            if control.protocol.get("seed") == other.protocol.get("seed"):
                complaints.append(
                    f"noise-floor arm {other.name} ran at the same seed as {control.name} "
                    f"({other.protocol.get('seed')!r}), so it is bit-identical and measures no noise"
                )
        elif not differing:
            complaints.append(f"{other.name} has the same head settings as {control.name} — the arms differ in nothing")
        elif len(differing) > 1:
            complaints.append(
                f"{other.name} differs from {control.name} in more than one head "
                f"({', '.join(differing)}) — the result cannot be attributed"
            )
    if allow_varying:
        complaints.append(f"comparison deliberately loosened: --allow-varying {' '.join(allow_varying)}")
    return complaints


def _format(key: str, value: float | None) -> str:
    """One cell: integers as integers, everything else to four decimals."""
    if value is None:
        return "—"
    if key in {"best_epoch", "num_params"}:
        return f"{int(value):,}"
    return f"{value:.4f}"


def _delta(key: str, value: float | None, control: float | None, direction: str) -> str:
    """A delta cell against the control, annotated with whether it is an improvement."""
    if value is None or control is None or key in {"best_epoch", "num_params"}:
        return "—"
    change = value - control
    if abs(change) < 5e-5:
        return "0"
    if direction == "up":
        marker = "+" if change > 0 else "−"
    elif direction == "down":
        marker = "+" if change < 0 else "−"
    elif direction == "zero":
        marker = "+" if abs(value) < abs(control) else "−"
    else:
        return f"{change:+.4f}"
    return f"{change:+.4f} ({marker})"


def _is_below_noise(
    key: str,
    value: float | None,
    control: float | None,
    floor: ArmSummary | None,
    summary: ArmSummary,
) -> bool:
    """Whether this arm's movement on ``key`` is no larger than the replicate's.

    The replicate is the control repeated at a different seed, so whatever it moves is
    what run-to-run variation looks like on this metric with this data. A treatment that
    does not clear it has not been shown to do anything.
    """
    if floor is None or summary is floor or value is None or control is None:
        return False
    floor_value = floor.metrics.get(key)
    if floor_value is None:
        return False
    return abs(value - control) <= abs(floor_value - control)


def render_table(summaries: Sequence[ArmSummary], noise_floor: str | None) -> str:
    """The comparison table: one column per arm, plus a delta column per non-control arm.

    ``(+)`` marks a delta in the improving direction for that metric and ``(−)`` the
    other way, so a reader does not have to remember which metrics are better low. When a
    noise-floor arm is named, its own delta is the yardstick and every other arm's delta
    is annotated ``below noise`` when it does not clear it — the difference between a
    result and a coincidence.
    """
    control = summaries[0]
    others = list(summaries[1:])
    floor = next((summary for summary in others if summary.name == noise_floor), None)

    header = ["metric", control.name]
    for summary in others:
        header += [summary.name, f"Δ {summary.name}"]
    rows = [header, ["---"] * len(header)]

    for key, label, direction in METRICS:
        if all(summary.metrics.get(key) is None for summary in summaries):
            continue  # a head no arm built has nothing to say
        base = control.metrics.get(key)
        row = [label, _format(key, base)]
        for summary in others:
            value = summary.metrics.get(key)
            cell = _delta(key, value, base, direction)
            # Only a real, non-zero movement can be "below the noise floor"; annotating a
            # dash or an exact zero would be noise about noise.
            if cell not in {"—", "0"} and _is_below_noise(key, value, base, floor, summary):
                cell += " below noise"
            row += [_format(key, value), cell]
        rows.append(row)

    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = []
    for index, row in enumerate(rows):
        if index == 1:
            lines.append("| " + " | ".join("-" * width for width in widths) + " |")
            continue
        lines.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) + " |")
    return "\n".join(lines)


def render_report(summaries: Sequence[ArmSummary], complaints: Sequence[str], noise_floor: str | None) -> str:
    """The whole markdown report: verdict banner, arm definitions, then the table."""
    lines = ["# Supervised A/B comparison", ""]
    if complaints:
        lines += ["**NOT COMPARABLE — do not read the numbers as a result.**", ""]
        lines += [f"- {complaint}" for complaint in complaints]
        lines.append("")
    else:
        lines += ["Arms differ only in the head under test; every protocol field matches.", ""]
    lines += ["| arm | heads on |", "| --- | --- |"]
    for summary in summaries:
        on = ", ".join(sorted(head for head, value in summary.heads.items() if value)) or "none"
        suffix = "  *(noise floor)*" if summary.name == noise_floor else ""
        lines.append(f"| {summary.name} | {on}{suffix} |")
    lines += ["", f"Control arm: **{summaries[0].name}**. `(+)` = better, `(−)` = worse.", ""]
    lines.append(render_table(summaries, noise_floor))
    lines.append("")
    return "\n".join(lines)


def run_arm(arm: Arm, args: argparse.Namespace) -> Path:
    """Run one arm (or reuse its existing JSON) and return the path it wrote to."""
    out_path = Path(args.out_dir) / f"{arm.name}.json"
    if args.reuse_existing and out_path.exists():
        logger.info("Arm {}: reusing {}", arm.name, out_path)
        return out_path
    command = build_command(arm, args)
    logger.info("Arm {}: {}", arm.name, shlex.join(command))
    subprocess.run(command, check=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run several supervised distillation arms under one controlled protocol and diff them.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME[=FLAGS]",
        help="An arm, e.g. --arm control or --arm ownership='--ownership-head'. The FIRST arm is "
        "the control every delta is measured against. Repeat for each arm.",
    )
    parser.add_argument("--config", required=True, help="Base run config JSON, shared by every arm")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus directory, shared by every arm")
    parser.add_argument(
        "--freeze-corpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink the shards present at start into <out-dir>/_snapshot and point every arm at "
        "that, so a corpus still being generated cannot hand later arms more games than earlier "
        "ones. --no-freeze-corpus reads the live directory (only sensible for a finished corpus).",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Where per-arm JSONs and the comparison land")
    parser.add_argument(
        "--noise-floor-arm",
        default=None,
        help="Name of the replicate arm: the control's settings re-run at --noise-floor-seed. Its "
        "distance from the control is what 'no effect' measures here, and smaller deltas are flagged. "
        "A zero-weight head is NOT a valid floor — it trains bit-identically, so its delta is always 0.",
    )
    parser.add_argument(
        "--noise-floor-seed",
        type=int,
        default=None,
        help="Seed for the --noise-floor-arm replicate. Must differ from --seed; it is the only "
        "thing about that arm that differs from the control.",
    )
    parser.add_argument(
        "--allow-varying",
        action="append",
        default=[],
        metavar="FLAG",
        help="Deliberately let arms vary this normally-shared flag, e.g. --allow-varying max-games "
        "for the N2 data-fraction curve. Leading dashes are optional (and `--allow-varying "
        "--max-games` does not parse — argparse reads the value as the next flag — so write it "
        "bare, or as --allow-varying=--max-games). Recorded in the comparison so a reader knows.",
    )
    parser.add_argument("--distill-arm", choices=("scratch", "warm"), default="scratch", help="distill_sl arm to run")
    parser.add_argument("--warm-start", default=None, help="Checkpoint for --distill-arm warm")
    parser.add_argument("--net-size", default=None, help="Override net size as <F>x<B> for every arm")
    parser.add_argument("--max-games", type=int, default=None, help="Subsample the corpus (shared by every arm)")
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7, help="Split + init + subsample seed, shared by every arm")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--opening-value", choices=("blend", "outcome", "search"), default="blend")
    parser.add_argument("--opening-mix", type=float, default=0.05)
    parser.add_argument("--v1-corpus", type=Path, default=None)
    parser.add_argument("--v1-mix", type=float, default=0.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip an arm whose JSON is already in --out-dir (re-render the comparison after a crash)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print each arm's command line and stop")
    args = parser.parse_args()

    # `--allow-varying max-games` is the form that parses; normalise it to the flag
    # spelling the allowlist and the arm specs actually use.
    args.allow_varying = [f"--{token.lstrip('-')}" for token in args.allow_varying]

    arms = [parse_arm(spec) for spec in args.arm]
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"Duplicate arm names in {names}.")
    for arm in arms:
        validate_arm_flags(arm, args.allow_varying)
    if args.noise_floor_arm and args.noise_floor_arm not in names:
        raise SystemExit(f"--noise-floor-arm {args.noise_floor_arm!r} is not one of {names}.")
    if args.noise_floor_arm:
        floor_arm = next(arm for arm in arms if arm.name == args.noise_floor_arm)
        if floor_arm.flags:
            raise SystemExit(
                f"--noise-floor-arm {args.noise_floor_arm!r} carries flags {list(floor_arm.flags)}. The floor "
                "must be a pure replicate of the control — declare it as a bare --arm NAME and let "
                "--noise-floor-seed be the only difference."
            )
        if args.noise_floor_seed is None:
            raise SystemExit("--noise-floor-arm needs --noise-floor-seed: the replicate's one difference.")
        if args.noise_floor_seed == args.seed:
            raise SystemExit(
                f"--noise-floor-seed {args.noise_floor_seed} equals --seed. The replicate would train "
                "bit-identically to the control, its delta would be 0 on every metric, and nothing "
                "could ever be flagged 'below noise'."
            )
    elif args.noise_floor_seed is not None:
        raise SystemExit("--noise-floor-seed has no effect without --noise-floor-arm.")

    args.frozen_corpus = None
    if args.dry_run:
        for arm in arms:
            print(shlex.join(build_command(arm, args)))  # noqa: T201 — the point of --dry-run
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.frozen_corpus = None
    if args.freeze_corpus:
        snapshot = args.out_dir / "_snapshot"
        if not snapshot.exists():
            shards = freeze_corpus(args.corpus, snapshot)
            logger.info("Froze {} shards of {} into {}", shards, args.corpus, snapshot)
        else:
            logger.info("Reusing the existing snapshot at {}", snapshot)
        args.frozen_corpus = snapshot

    summaries = [
        summarise_arm(arm.name, json.loads(run_arm(arm, args).read_text(encoding="utf-8")), args.distill_arm)
        for arm in arms
    ]

    complaints = check_comparable(summaries, args.allow_varying, args.noise_floor_arm)
    report = render_report(summaries, complaints, args.noise_floor_arm)
    (args.out_dir / "comparison.md").write_text(report, encoding="utf-8")
    (args.out_dir / "comparison.json").write_text(
        json.dumps(
            {
                "control": summaries[0].name,
                "noise_floor_arm": args.noise_floor_arm,
                "allow_varying": args.allow_varying,
                "comparable": not complaints,
                "complaints": complaints,
                "arms": {
                    summary.name: {"heads": summary.heads, "metrics": summary.metrics, "protocol": summary.protocol}
                    for summary in summaries
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)  # noqa: T201 — the deliverable is a table on stdout

    if complaints:
        for complaint in complaints:
            logger.error("Comparison is not controlled — {}", complaint)
        raise SystemExit(1)
    logger.info("Comparison → {}", args.out_dir / "comparison.md")


if __name__ == "__main__":
    main()
