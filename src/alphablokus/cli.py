from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from alphablokus.config import load_args
from alphablokus.provenance import check_config_is_committed, write_provenance
from alphablokus.registry import instantiate_game_and_network
from alphablokus.reporting import create_html_report
from alphablokus.storage.object_store import create_object_store, sync_up_guarded
from alphablokus.training.coach import PROGRESS_MARKER_FILENAME, Coach, read_progress_marker

if TYPE_CHECKING:
    from alphablokus.config import RunConfig

RESOLVED_CONFIG_FILENAME = "config.resolved.json"


def persist_resolved_config(config: RunConfig) -> None:
    """Write the fully-resolved ``RunConfig`` to ``<run>/config.resolved.json`` (S4).

    Two of three recent runs ran with a config that differed from the committed
    JSON (net-preset resolution, a swapped ``*_volume.json``, LR schedule); the
    post-mortem had to reconstruct what actually ran from the parquets
    (plateau-investigation §1). Dumping the resolved dataclass — presets expanded,
    every default filled in — at launch makes the ground truth unambiguous. Paths
    are stringified; tuples become JSON arrays. Best-effort: a serialisation
    failure must never sink a training launch.
    """
    path = config.run_directory / RESOLVED_CONFIG_FILENAME
    try:
        payload = dataclasses.asdict(config)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote resolved run config to {}", path)
    except Exception as err:  # pragma: no cover - defensive; never blocks a run
        logger.warning("Could not persist resolved config to {} ({}); continuing.", path, err)


def restore_run_from_object_store(config: RunConfig, client: Any | None = None) -> None:
    """Rebuild the local run directory from the bucket when it's behind.

    Called on ``--resume`` when an object store is configured. Compares the
    remote progress marker with the local one: local missing or older →
    force-download the whole run prefix (force, because a stale checkpoint has
    the same byte size as the current one); local up to date → no-op. Failures
    propagate — resuming from silently-stale state would corrupt the run.
    """
    store = create_object_store(config, client)
    if store is None:
        return
    marker_relative = (config.log_directory / PROGRESS_MARKER_FILENAME).relative_to(config.run_directory).as_posix()
    remote_marker_path = config.log_directory / "progress.remote.json"
    if not store.download_file(marker_relative, remote_marker_path):
        logger.info("Object store: no remote progress marker — resuming from local state only.")
        return
    remote_generation = int(json.loads(remote_marker_path.read_text(encoding="utf-8"))["last_completed_generation"])
    remote_marker_path.unlink()
    local_marker = read_progress_marker(config)
    local_generation = int(local_marker["last_completed_generation"]) if local_marker else -1
    if local_generation >= remote_generation:
        logger.info(
            "Object store: local run is up to date (gen {} >= remote gen {}) — no restore needed.",
            local_generation,
            remote_generation,
        )
        return
    logger.info(
        "Object store: local gen {} behind remote gen {} — restoring run directory from bucket...",
        local_generation,
        remote_generation,
    )
    downloaded = store.sync_down(config.run_directory, force=True)
    logger.info("Object store: restored {} file(s).", downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaBlokus training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="run_configurations/test_run.json",
        help="Path to the JSON run configuration file (default: run_configurations/test_run.json)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the HTML report from existing data without training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a crashed/stopped run from its last completed generation "
        "(reuses the frozen Elo baseline; continues generation numbering).",
    )
    parser.add_argument(
        "--allow-uncommitted-config",
        action="store_true",
        help="Start even though the config file differs from (or is untracked by) git. "
        "Recorded in the run's provenance. Use while iterating on a config; do not use "
        "for a run whose result you intend to quote.",
    )
    cli_args = parser.parse_args()
    args = load_args(cli_args.config)

    if cli_args.report_only:
        create_html_report(args)
        return

    # Refuse to start when the committed config does not describe this run (A5).
    # Before anything is written, so a rejected launch leaves no trace.
    config_path = Path(cli_args.config)
    config_state = check_config_is_committed(
        config_path,
        allow_uncommitted=cli_args.allow_uncommitted_config,
    )

    args.run_directory.mkdir(parents=True, exist_ok=True)

    # Persist the fully-resolved config at launch so what actually ran is never
    # ambiguous again (S4b). Written every launch (fresh + resume) so a resumed
    # run records the config it resumed under too.
    persist_resolved_config(args)

    # Stamp code version + config-commit state + input-data manifest alongside it.
    write_provenance(
        args,
        config_path=config_path,
        config_state=config_state,
        override_used=cli_args.allow_uncommitted_config,
    )

    # Add rotating file sink alongside default stderr
    log_dir = args.log_directory
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "alpha.log", rotation="10 MB", retention=3)

    start = time.perf_counter()

    logger.info(f"Loading game: {args.game}")
    game, nnet = instantiate_game_and_network(args)

    if cli_args.resume:
        # With an object store configured, an interrupted cloud run resumes on
        # a fresh machine: pull the run directory down before reading markers.
        restore_run_from_object_store(args)
        marker = read_progress_marker(args)
        if marker is None:
            raise SystemExit(
                f"--resume: no progress marker at {log_dir / 'progress.json'} — nothing to resume.",
            )
        last_gen = int(marker["last_completed_generation"])
        logger.info("Resuming run after generation {} (loading latest.pth.tar)", last_gen)
        nnet.load_checkpoint("latest.pth.tar")
        c = Coach(game, nnet, args, resume=True, resume_wandb_run_id=marker.get("wandb_run_id"))
        c.restore_cumulative_totals(marker)
        c.load_self_play_history_for_resume(last_gen)
        start_generation = last_gen + 1
    else:
        if args.load_model:
            # Warm start: weights only, so the optimizer + LR schedule start
            # fresh at this run's configured learning rate rather than inheriting
            # the donor checkpoint's annealed LR / scheduler position (L4).
            logger.info("Warm-starting weights from best.pth.tar (fresh optimizer + LR schedule)...")
            nnet.load_weights("best.pth.tar")
        else:
            logger.warning("Not loading a checkpoint!")

        logger.info("Loading the Coach...")
        c = Coach(game, nnet, args)

        if args.load_model:
            logger.info("Loading self-play history...")
            c.load_self_play_history(up_to_generation=0)
        start_generation = 1

    logger.info("Starting the learning process")
    # Render the report on the way out even if learn() raises. The crash that
    # ended blokus_cloud_60 at gen 59 left no report at all, because the render
    # sat *after* learn() returned; per-generation parquets are already on disk,
    # so a finally-render recovers a report from whatever generations completed
    # (the same data --report-only reads). See docs/plans/archive/harden-long-runs.md H2.
    try:
        c.learn(start_generation=start_generation)
        # Normal completion only (a crash skips this): optionally play the
        # post-hoc pool BayesElo tournament so the report includes the rigorous,
        # non-saturating strength curve without a manual step. Crash-safe — a
        # tournament failure must never lose the run's training artifacts (all
        # already on disk), so log and fall through to the report render.
        if args.tournament.run_at_end:
            try:
                from alphablokus.evaluation.tournament_run import run_tournament

                logger.info("Running end-of-run pool BayesElo tournament...")
                run_tournament(args)
            except Exception:
                logger.exception(
                    "End-of-run pool tournament failed; training artifacts intact. "
                    "Run it manually with: python -m scripts.tournament_elo --config <cfg>.",
                )
    finally:
        # A finished (or crashed) run must not be sunk by report rendering (R7):
        # all data is already on disk, so log and continue — regenerate later
        # with --report-only.
        try:
            create_html_report(args)
        except Exception:
            logger.exception(
                "Report generation failed, but training data is intact. Regenerate with: --report-only.",
            )
        # Final mirror so the rendered report (and anything else since the last
        # per-generation sync) reaches the bucket even on crash. Reuses the
        # Coach's store so the sync stays incremental. Best-effort, like the
        # report itself.
        sync_up_guarded(c.object_store, args.run_directory, "final")

    end = time.perf_counter()
    logger.info(f"Total time elapsed: {end - start}")


if __name__ == "__main__":
    main()
