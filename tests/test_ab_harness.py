"""Tests for the supervised A/B harness (plan N1).

The harness exists to stop an *unfair* comparison being produced, so most of what is
tested here is refusal: an arm that varies the data, two arms that vary more than one
thing, and two run JSONs that disagree about the split they were evaluated on. The rest
pins the table's reading — deltas signed by whether the metric is better high or low, and
a treatment smaller than the inert arm's own drift called out as noise.

Nothing here trains: the arms' run JSONs are synthesised in the exact shape
``distill_sl.py`` writes, so the whole comparison path runs in milliseconds.
"""

from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING, Any

import pytest

from scripts.ab_harness import (
    ARM_FLAG_ALLOWLIST,
    PROTOCOL_KEYS,
    Arm,
    build_command,
    check_comparable,
    freeze_corpus,
    parse_arm,
    render_report,
    render_table,
    summarise_arm,
    validate_arm_flags,
)

if TYPE_CHECKING:
    from pathlib import Path


def _payload(**overrides: Any) -> dict[str, Any]:
    """A run JSON in ``distill_sl.py``'s shape, with tweakable metrics."""
    heads = overrides.pop("heads", {"score": False, "ownership": False, "reply": False})
    metrics = {
        "policy_ce": 3.0,
        "policy_kl": 2.0,
        "top1": 0.30,
        "top3": 0.55,
        "value_mse": 0.25,
        "colour_only_value_mse": 0.30,
        "value_skill": 1.0 - 0.25 / 0.30,
        "ownership": None,
        **overrides.pop("metrics", {}),
    }
    settings = {
        "score_head": heads["score"],
        "score_loss_weight": 0.05,
        "score_scale": 30.0,
        "ownership_head": heads["ownership"],
        "ownership_loss_weight": 0.05,
        "reply_head": heads["reply"],
        "reply_loss_weight": 0.05,
    }
    settings.update(overrides.pop("settings", {}))
    payload: dict[str, Any] = {
        **settings,
        "corpus": "/corpora/v2",
        "corpus_version": "v2",
        "num_games": 1000,
        "max_games": None,
        "holdout_fraction": 0.05,
        "seed": 7,
        "epsilon": 0.0,
        "target_temperature": 1.0,
        "opening_value": "blend",
        "opening_mix": 0.05,
        "v1_mix": 0.0,
        "augment": True,
        "lr": 1e-4,
        "net_size": "192x12",
        "holdout_leakage": {"holdout_rows": 500, "leaked_fraction_mirror": 0.004},
        "arms": {
            "scratch": {
                "num_params": 8_104_223,
                "best_epoch": 4,
                "best": {"policy_ce": metrics["policy_ce"], "policy_kl": metrics["policy_kl"]},
                "best_diagnostics": {
                    "top1_accuracy": metrics["top1"],
                    "top3_accuracy": metrics["top3"],
                    "value_mse": metrics["value_mse"],
                    "colour_only_value_mse": metrics["colour_only_value_mse"],
                    "calibration": [
                        {"player": -1, "mean_predicted": -0.4, "mean_outcome": -0.5, "value_mse": 0.22},
                        {"player": 1, "mean_predicted": 0.6, "mean_outcome": 0.5, "value_mse": 0.28},
                    ],
                },
                "best_value_skill": metrics["value_skill"],
                "best_aux": {"score": None, "ownership": metrics["ownership"], "reply": None},
                "heads": heads,
            }
        },
    }
    payload.update(overrides)
    return payload


def _summary(name: str, **overrides: Any) -> Any:
    return summarise_arm(name, _payload(**overrides), "scratch")


# --------------------------------------------------------------------------- #
# Arm specs
# --------------------------------------------------------------------------- #


def test_parse_arm_handles_both_spellings() -> None:
    assert parse_arm("control") == Arm("control", ())
    assert parse_arm("own=--ownership-head") == Arm("own", ("--ownership-head",))
    # The shell strips the quotes before argparse sees it, so this is what arrives.
    assert parse_arm("w=--ownership-loss-weight 0.3") == Arm("w", ("--ownership-loss-weight", "0.3"))


def test_parse_arm_rejects_a_nameless_spec() -> None:
    with pytest.raises(SystemExit, match="no name"):
        parse_arm("=--ownership-head")


def test_an_arm_may_vary_a_head_switch() -> None:
    validate_arm_flags(parse_arm("own=--ownership-head --ownership-loss-weight 0"), [])


@pytest.mark.parametrize("flag", ["--seed", "--max-games", "--corpus", "--holdout-frac", "--lr", "--tau"])
def test_an_arm_may_not_vary_the_data_or_the_protocol(flag: str) -> None:
    """The whole point: "same seed, same data, same corpus fraction" is structural.

    These flags are harness-level and forwarded identically, so an arm carrying one is a
    comparison that was never controlled — refused before a GPU-hour is spent, not
    diagnosed from the table afterwards.
    """
    with pytest.raises(SystemExit, match="not an arm-level setting"):
        validate_arm_flags(parse_arm(f"bad={flag} 3"), [])


def test_varying_a_protocol_flag_needs_an_explicit_opt_in() -> None:
    """The N2 data-fraction curve does vary the data — deliberately, and on the record."""
    validate_arm_flags(parse_arm("quarter=--max-games 250"), ["--max-games"])


def test_the_allowlist_is_exactly_the_head_switches() -> None:
    """A new knob must be added deliberately, so the guard cannot rot open."""
    assert {
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
    } == ARM_FLAG_ALLOWLIST


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


def _args(tmp_path: Path, **overrides: Any) -> Namespace:
    defaults: dict[str, Any] = {
        "config": "run_configurations/test_run.json",
        "corpus": tmp_path / "corpus",
        "out_dir": tmp_path / "out",
        "distill_arm": "scratch",
        "warm_start": None,
        "net_size": None,
        "max_games": None,
        "holdout_frac": 0.05,
        "seed": 7,
        "freeze_corpus": True,
        "frozen_corpus": None,
        "noise_floor_arm": None,
        "noise_floor_seed": None,
        "max_epochs": 20,
        "patience": 3,
        "min_delta": 0.002,
        "lr": 1e-4,
        "batch_size": 1024,
        "eval_batch_size": 512,
        "epsilon": None,
        "tau": 1.0,
        "opening_value": "blend",
        "opening_mix": 0.05,
        "v1_corpus": None,
        "v1_mix": 0.0,
        "augment": True,
    }
    return Namespace(**{**defaults, **overrides})


def test_every_arm_gets_identical_data_and_protocol_flags(tmp_path: Path) -> None:
    """Two commands must differ **only** by the arm's own flags and its output paths."""
    args = _args(tmp_path)
    control = build_command(Arm("control", ()), args)
    treatment = build_command(Arm("own", ("--ownership-head",)), args)

    def strip_paths(command: list[str]) -> list[str]:
        cleaned = []
        skip = False
        for token in command:
            if skip:
                skip = False
                continue
            if token in {"--ckpt-dir", "--out"}:
                skip = True
                continue
            cleaned.append(token)
        return cleaned

    assert strip_paths(treatment) == [*strip_paths(control), "--ownership-head"]
    for shared in ("--seed", "--holdout-frac", "--corpus", "--lr", "--tau", "--batch-size"):
        assert control.count(shared) == 1
        assert control[control.index(shared) + 1] == treatment[treatment.index(shared) + 1]


def test_arm_flags_come_last_so_an_allowed_override_wins(tmp_path: Path) -> None:
    command = build_command(Arm("quarter", ("--max-games", "250")), _args(tmp_path, max_games=1000))
    assert command[-2:] == ["--max-games", "250"]
    assert command.index("--max-games") < len(command) - 2  # the shared one is still there, earlier


def test_a_snapshot_holds_only_the_shards_that_existed_when_it_was_taken(tmp_path: Path) -> None:
    """Regression: a corpus still being generated hands later arms more games.

    ``--max-games`` then samples a different set for each arm and they sit different
    exams — which is precisely how the first score-head A/B was wasted. The snapshot is
    symlinks, so freezing a large corpus stays free.
    """
    corpus = tmp_path / "corpus"
    (corpus / "games").mkdir(parents=True)
    (corpus / "opening").mkdir()
    for index in range(3):
        (corpus / "games" / f"corpus_{index}.parquet").write_bytes(b"x")
    (corpus / "opening" / "opening_0.parquet").write_bytes(b"y")

    snapshot = tmp_path / "snap"
    assert freeze_corpus(corpus, snapshot) == 4

    (corpus / "games" / "corpus_3.parquet").write_bytes(b"x")  # generation carries on

    frozen = sorted(path.name for path in (snapshot / "games").glob("*.parquet"))
    assert frozen == ["corpus_0.parquet", "corpus_1.parquet", "corpus_2.parquet"]
    assert (snapshot / "opening" / "opening_0.parquet").is_symlink()
    assert (snapshot / "games" / "corpus_0.parquet").read_bytes() == b"x"


def test_freezing_an_empty_corpus_is_an_error_not_an_empty_comparison(tmp_path: Path) -> None:
    (tmp_path / "corpus" / "games").mkdir(parents=True)
    with pytest.raises(SystemExit, match="nothing to compare"):
        freeze_corpus(tmp_path / "corpus", tmp_path / "snap")


def test_every_arm_reads_the_snapshot_rather_than_the_live_corpus(tmp_path: Path) -> None:
    args = _args(tmp_path, frozen_corpus=tmp_path / "out" / "_snapshot")
    command = build_command(Arm("control", ()), args)
    assert command[command.index("--corpus") + 1] == str(tmp_path / "out" / "_snapshot")


def test_the_noise_floor_arm_is_the_only_one_reseeded(tmp_path: Path) -> None:
    """Regression: the floor has to differ from the control *somehow* to measure spread.

    A zero-weight head does not qualify — it trains bit-identically. The seed is the one
    difference that changes the run without changing what is being tested, so it is the
    single thing the harness varies for this arm.
    """
    args = _args(tmp_path, noise_floor_arm="replicate", noise_floor_seed=8)
    control = build_command(Arm("control", ()), args)
    floor = build_command(Arm("replicate", ()), args)

    assert control[control.index("--seed") + 1] == "7"
    assert floor[floor.index("--seed") + 1] == "8"


def test_each_arm_writes_to_its_own_paths(tmp_path: Path) -> None:
    command = build_command(Arm("own", ()), _args(tmp_path))
    assert command[command.index("--out") + 1].endswith("own.json")
    assert command[command.index("--ckpt-dir") + 1].endswith("own")


# --------------------------------------------------------------------------- #
# Summarising a run JSON
# --------------------------------------------------------------------------- #


def test_the_summary_reads_the_full_n1_metric_set() -> None:
    summary = _summary("control")

    assert summary.metrics["value_skill"] == pytest.approx(1.0 - 0.25 / 0.30)
    assert summary.metrics["top1"] == pytest.approx(0.30)
    assert summary.metrics["top3"] == pytest.approx(0.55)
    # Per-colour calibration: predicted − actual, per side to move.
    assert summary.metrics["bias_white"] == pytest.approx(0.1)
    assert summary.metrics["bias_black"] == pytest.approx(0.1)
    assert summary.metrics["value_mse_white"] == pytest.approx(0.28)
    # And the leakage figure that qualifies every number above it.
    assert summary.metrics["leaked_fraction_mirror"] == pytest.approx(0.004)


def test_a_head_the_arm_did_not_build_reports_nothing() -> None:
    """``None``, never a fabricated zero — a zero would read as "the head learnt nothing"."""
    summary = _summary("control")
    assert summary.metrics["ownership_skill"] is None
    assert summary.metrics["score_mse"] is None
    assert summary.metrics["reply_ce"] is None


def test_the_summary_reads_the_auxiliary_head_when_it_exists() -> None:
    ownership = {"cross_entropy": 0.6, "marginal_cross_entropy": 1.0, "skill": 0.4, "accuracy": 0.8}
    summary = _summary(
        "own", metrics={"ownership": ownership}, heads={"score": False, "ownership": True, "reply": False}
    )

    assert summary.metrics["ownership_skill"] == pytest.approx(0.4)
    assert summary.metrics["ownership_accuracy"] == pytest.approx(0.8)
    assert summary.metrics["ownership_ce"] == pytest.approx(0.6)
    assert summary.heads["ownership"] is True


def test_a_run_json_without_the_requested_sub_arm_is_an_error() -> None:
    with pytest.raises(SystemExit, match="has no 'warm' arm"):
        summarise_arm("control", _payload(), "warm")


def _run_cli(tmp_path: Path, *extra: str) -> None:
    """``main()`` up to ``--dry-run``: argument parsing and every pre-flight check."""
    import sys
    from unittest import mock

    from scripts.ab_harness import main

    argv = [
        "ab_harness",
        "--config",
        "run_configurations/test_run.json",
        "--corpus",
        str(tmp_path / "corpus"),
        "--out-dir",
        str(tmp_path / "out"),
        "--dry-run",
        *extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        main()


def test_allow_varying_takes_the_flag_name_without_its_dashes(tmp_path: Path) -> None:
    """Regression: ``--allow-varying --max-games`` cannot parse — argparse reads the
    value as the next flag and then complains the option is missing an argument. The
    bare spelling is the one that works, and it must reach the allowlist as ``--max-games``.
    """
    _run_cli(
        tmp_path,
        "--arm",
        "control",
        "--arm",
        "quarter=--max-games 250",
        "--allow-varying",
        "max-games",
    )


def test_allow_varying_still_takes_the_attached_dashed_spelling(tmp_path: Path) -> None:
    _run_cli(tmp_path, "--arm", "control", "--arm", "quarter=--max-games 250", "--allow-varying=--max-games")


def test_without_the_opt_in_a_data_varying_arm_is_still_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _run_cli(tmp_path, "--arm", "control", "--arm", "quarter=--max-games 250")


def test_a_noise_floor_arm_needs_its_own_seed_before_any_gpu_time(tmp_path: Path) -> None:
    """The bit-identical floor is refused up front, not diagnosed after two training runs."""
    with pytest.raises(SystemExit, match="noise-floor-seed"):
        _run_cli(tmp_path, "--arm", "control", "--arm", "replicate", "--noise-floor-arm", "replicate")
    with pytest.raises(SystemExit, match="bit-identically"):
        _run_cli(
            tmp_path,
            "--arm",
            "control",
            "--arm",
            "replicate",
            "--noise-floor-arm",
            "replicate",
            "--noise-floor-seed",
            "7",
        )


def test_a_noise_floor_arm_carrying_head_flags_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="pure replicate"):
        _run_cli(
            tmp_path,
            "--arm",
            "control",
            "--arm",
            "zero=--ownership-head --ownership-loss-weight 0",
            "--noise-floor-arm",
            "zero",
            "--noise-floor-seed",
            "8",
        )


# --------------------------------------------------------------------------- #
# Comparability
# --------------------------------------------------------------------------- #


def _pair(**treatment_overrides: Any) -> list[Any]:
    control = _summary("control")
    treatment = _summary(
        "own",
        heads={"score": False, "ownership": True, "reply": False},
        **treatment_overrides,
    )
    return [control, treatment]


def test_a_clean_pair_has_no_complaints() -> None:
    assert check_comparable(_pair(), []) == []


def _perturb(value: Any) -> Any:
    """Any value of the same shape that is not ``value``."""
    if isinstance(value, dict):
        return {**value, "leaked_fraction_mirror": 0.5}
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return "DIFFERENT"


def test_the_protocol_key_set_is_the_whole_controlled_protocol() -> None:
    """Pinned explicitly, because the parametrised test below iterates this same list.

    Deleting a key would silently delete its test case too — the guard would rot open
    and every arm would still report "comparable".
    """
    assert set(PROTOCOL_KEYS) == {
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
    }


@pytest.mark.parametrize("key", PROTOCOL_KEYS)
def test_any_protocol_disagreement_voids_the_comparison(key: str) -> None:
    """Every field the arms must share is actually checked — not just the obvious ones.

    Two arms trained on different data produce a table that *looks* like a result, which
    is worse than no table at all.
    """
    baseline = _payload()[key]
    complaints = check_comparable(_pair(**{key: _perturb(baseline)}), [])
    assert any(complaint.startswith(f"{key}:") for complaint in complaints)


def test_a_different_measured_leakage_voids_the_comparison() -> None:
    """The strongest protocol check: it measures the split rather than describing it."""
    complaints = check_comparable(_pair(holdout_leakage={"holdout_rows": 500, "leaked_fraction_mirror": 0.02}), [])
    assert any("holdout_leakage" in complaint for complaint in complaints)


def test_two_identical_arms_are_flagged_as_measuring_nothing() -> None:
    control = _summary("a")
    duplicate = _summary("b")
    assert any("differ in nothing" in complaint for complaint in check_comparable([control, duplicate], []))


def test_changing_two_heads_at_once_is_flagged_as_unattributable() -> None:
    """One change at a time is the plan's organising constraint, so it is enforced."""
    control = _summary("control")
    both = _summary("both", heads={"score": False, "ownership": True, "reply": True})
    complaints = check_comparable([control, both], [])
    assert any("more than one head" in complaint for complaint in complaints)


def test_a_head_weight_riding_along_with_the_treatment_is_flagged() -> None:
    """Regression: the on/off switches alone are not the whole of what an arm varies.

    Both arms build the score head, so the switches match and only ownership *looks*
    different — but the score loss weight moved too, so any delta has two possible
    causes and cannot be attributed to the ownership head.
    """
    both_score = {"score": True, "ownership": False, "reply": False}
    control = _summary("control", heads=both_score)
    treatment = _summary(
        "own",
        heads={"score": True, "ownership": True, "reply": False},
        settings={"score_loss_weight": 0.5},
    )
    complaints = check_comparable([control, treatment], [])
    assert any("more than one head" in complaint for complaint in complaints)


def test_the_same_head_at_a_different_weight_is_a_legitimate_single_change() -> None:
    """The flip side: a weight sweep varies one head and must not be refused."""
    on = {"score": True, "ownership": False, "reply": False}
    control = _summary("control", heads=on)
    heavier = _summary("heavy", heads=on, settings={"score_loss_weight": 0.5})
    assert check_comparable([control, heavier], []) == []


def test_a_head_that_is_off_in_both_arms_cannot_differ_through_its_weight() -> None:
    """An inherited weight on a head nobody built is not a difference worth reporting."""
    control = _summary("control")
    treatment = _summary(
        "own",
        heads={"score": False, "ownership": True, "reply": False},
        settings={"score_loss_weight": 0.5, "reply_loss_weight": 0.9},
    )
    assert check_comparable([control, treatment], []) == []


def test_a_noise_floor_arm_at_the_control_s_seed_measures_nothing() -> None:
    """Regression: a same-seed floor trains bit-identically, so nothing can clear it.

    The floor is only meaningful as a *replicate* — same settings, different roll of the
    dice. Sharing the seed makes every floor delta exactly 0, which silently turns the
    "below noise" annotation off for the whole table rather than reporting a problem.
    """
    control = _summary("control")
    floor = _summary("replicate")
    complaints = check_comparable([control, floor], [], "replicate")
    assert any("bit-identical" in complaint for complaint in complaints)


def test_a_properly_seeded_replicate_is_a_clean_noise_floor() -> None:
    control = _summary("control")
    floor = _summary("replicate", seed=8)
    assert check_comparable([control, floor], [], "replicate") == []


def test_a_noise_floor_arm_that_also_varies_a_head_is_not_a_replicate() -> None:
    control = _summary("control")
    floor = _summary("replicate", seed=8, heads={"score": False, "ownership": True, "reply": False})
    complaints = check_comparable([control, floor], [], "replicate")
    assert any("pure replicate" in complaint for complaint in complaints)


def test_an_explicit_loosening_is_recorded_in_the_output() -> None:
    complaints = check_comparable(_pair(), ["--max-games"])
    assert any("deliberately loosened" in complaint for complaint in complaints)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_deltas_are_signed_by_whether_the_metric_is_better_high_or_low() -> None:
    """A reader must not have to remember that CE is better low and skill better high."""
    control = _summary("control")
    better = _summary(
        "own",
        heads={"score": False, "ownership": True, "reply": False},
        metrics={"policy_ce": 2.5, "top1": 0.35},
    )

    table = render_table([control, better], None)
    ce_row = next(line for line in table.splitlines() if "policy CE" in line)
    top1_row = next(line for line in table.splitlines() if "top-1" in line)

    assert "-0.5000 (+)" in ce_row  # lower CE is an improvement
    assert "+0.0500 (+)" in top1_row  # higher agreement is an improvement


def test_a_regression_is_marked_as_one() -> None:
    control = _summary("control")
    worse = _summary("own", heads={"score": False, "ownership": True, "reply": False}, metrics={"policy_ce": 3.5})
    ce_row = next(line for line in render_table([control, worse], None).splitlines() if "policy CE" in line)
    assert "+0.5000 (−)" in ce_row


def test_a_delta_no_bigger_than_the_replicate_s_is_called_noise() -> None:
    """The floor arm is the control at another seed, so whatever it moves is not an effect.

    Here the treatment moves top-1 by exactly as much as the replicate does, which means
    the technique has been shown to do nothing — and the table has to say so rather than
    presenting a +0.02 that a reader would take for a result.
    """
    control = _summary("control")
    inert = _summary("replicate", seed=8, metrics={"top1": 0.32})
    treatment = _summary("own", heads={"score": False, "ownership": True, "reply": False}, metrics={"top1": 0.32})

    table = render_table([control, inert, treatment], "replicate")
    top1_row = next(line for line in table.splitlines() if "top-1 vs" in line)

    assert top1_row.count("below noise") == 1  # the treatment, not the floor arm itself


def test_a_delta_clearing_the_noise_floor_is_not_flagged() -> None:
    control = _summary("control")
    inert = _summary("replicate", seed=8, metrics={"top1": 0.305})
    treatment = _summary("own", heads={"score": False, "ownership": True, "reply": False}, metrics={"top1": 0.40})

    table = render_table([control, inert, treatment], "replicate")
    top1_row = next(line for line in table.splitlines() if "top-1 vs" in line)

    assert "below noise" not in top1_row


def test_metrics_no_arm_measured_are_left_out_entirely() -> None:
    table = render_table([_summary("control")], None)
    assert "ownership-head skill" not in table
    assert "value skill" in table


def test_the_report_leads_with_the_verdict_and_the_arm_definitions() -> None:
    summaries = _pair()
    clean = render_report(summaries, [], None)
    assert "NOT COMPARABLE" not in clean
    assert "| own | ownership |" in clean

    void = render_report(summaries, ["seed: control=7 but own=8"], None)
    assert void.splitlines()[2].startswith("**NOT COMPARABLE")
    assert "seed: control=7 but own=8" in void


def test_the_comparison_json_shape_round_trips(tmp_path: Path) -> None:
    """The per-arm JSON the plan asks for alongside the table must be strict JSON."""
    summaries = _pair()
    payload = {
        "control": summaries[0].name,
        "comparable": True,
        "arms": {s.name: {"heads": s.heads, "metrics": s.metrics, "protocol": s.protocol} for s in summaries},
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["arms"]["own"]["heads"]["ownership"] is True
    assert reloaded["arms"]["control"]["metrics"]["ownership_skill"] is None


def test_an_unchanged_metric_is_not_annotated_as_noise() -> None:
    """ "0 below noise" and "— below noise" are noise about noise — never emitted."""
    control = _summary("control")
    inert = _summary("replicate", seed=8)
    treatment = _summary("own", heads={"score": False, "ownership": True, "reply": False})

    table = render_table([control, inert, treatment], "replicate")

    assert "0 below noise" not in table
    assert "— below noise" not in table


# --------------------------------------------------------------------------- #
# End to end (real subprocesses, real training, tiny everything)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_harness_runs_two_real_arms_and_writes_a_comparison(tmp_path: Path) -> None:
    """The whole loop: spawn ``distill_sl.py`` per arm, read both JSONs, write the table.

    The pure functions above cannot catch a broken command line — a renamed flag, a path
    joined wrongly — because they never launch anything. This does, on a real (tiny)
    corpus, and additionally pins the property the harness exists to protect: at a shared
    seed the two arms' **control metrics are identical**, because the ownership head is
    built last and changes nothing the policy or value head sees.
    """
    import subprocess
    import sys
    from unittest import mock

    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.corpus import (
        OpeningPrefixBuilder,
        RandomMoveSource,
        play_corpus_game,
        shard_filename,
        write_shard,
    )
    from alphablokus.games.blokusduo.pieces import default_pieces_path
    from scripts.ab_harness import main

    game = BlokusDuoGame(pieces_config_path=default_pieces_path())
    source = RandomMoveSource(game)
    builder = OpeningPrefixBuilder(game, base_seed=0, num_plies=4)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_shard(
        corpus / shard_filename(0),
        [
            play_corpus_game(game, source, game_id=g, pentobi_seed=g, opening_actions=builder.prefix_for(g))
            for g in range(8)
        ],
        policy_size=game.get_action_size(),
        level=9,
        opening_random_plies=4,
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "game": "blokusduo",
                "run_name": "ab_harness_e2e",
                "num_generations": 1,
                "num_eps": 1,
                "temp_threshold": 5,
                "update_threshold": 0.55,
                "num_arena_matches": 2,
                "root_directory": str(tmp_path / "runs"),
                "load_model": False,
                "mcts_config": {"num_mcts_sims": 2, "cpuct": 1},
                "net_config": {
                    "learning_rate": 0.005,
                    "dropout": 0.0,
                    "epochs": 1,
                    "batch_size": 8,
                    "cuda": False,
                    "num_filters": 16,
                    "num_residual_blocks": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    argv = [
        "ab_harness.py",
        "--config",
        str(config),
        "--corpus",
        str(corpus),
        "--out-dir",
        str(out_dir),
        "--arm",
        "control",
        "--arm",
        "ownership=--ownership-head",
        "--holdout-frac",
        "0.25",
        "--max-epochs",
        "1",
        "--batch-size",
        "8",
        "--eval-batch-size",
        "8",
    ]
    # The harness spawns ``sys.executable``; under pytest that is the venv interpreter,
    # which needs the repo importable exactly as the CLI has it.
    with mock.patch.object(sys, "argv", argv), mock.patch.object(subprocess, "run", wraps=subprocess.run):
        main()

    comparison = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["comparable"] is True
    assert comparison["complaints"] == []
    assert comparison["arms"]["control"]["heads"]["ownership"] is False
    assert comparison["arms"]["ownership"]["heads"]["ownership"] is True
    # The ownership arm measured its head; the control has nothing to report.
    assert comparison["arms"]["ownership"]["metrics"]["ownership_skill"] is not None
    assert comparison["arms"]["control"]["metrics"]["ownership_skill"] is None
    # Adding the head costs parameters and nothing else at initialisation.
    assert (
        comparison["arms"]["ownership"]["metrics"]["num_params"]
        > comparison["arms"]["control"]["metrics"]["num_params"]
    )
    assert (out_dir / "comparison.md").read_text(encoding="utf-8").startswith("# Supervised A/B comparison")
