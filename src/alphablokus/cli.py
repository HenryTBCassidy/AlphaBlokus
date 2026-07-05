from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from alphablokus.config import load_args
from alphablokus.registry import instantiate_game_and_network
from alphablokus.reporting import create_html_report
from alphablokus.storage.object_store import create_object_store, sync_up_guarded
from alphablokus.training.coach import PROGRESS_MARKER_FILENAME, Coach, read_progress_marker

if TYPE_CHECKING:
    from alphablokus.config import RunConfig


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
    cli_args = parser.parse_args()
    args = load_args(cli_args.config)

    if cli_args.report_only:
        create_html_report(args)
        return

    args.run_directory.mkdir(parents=True, exist_ok=True)

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
        c.load_self_play_history_for_resume(last_gen)
        start_generation = last_gen + 1
    else:
        if args.load_model:
            logger.info("Loading checkpoint from best.pth.tar...")
            nnet.load_checkpoint("best.pth.tar")
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
