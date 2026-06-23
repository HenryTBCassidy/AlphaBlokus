import argparse
import time

from loguru import logger

from core.coach import Coach, read_progress_marker
from core.config import load_args
from core.game_factory import instantiate_game_and_network
from reporting import create_html_report


def main():
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
        marker = read_progress_marker(args)
        if marker is None:
            raise SystemExit(
                f"--resume: no progress marker at {log_dir / 'progress.json'} — "
                "nothing to resume.",
            )
        last_gen = int(marker["last_completed_generation"])
        logger.info("Resuming run after generation {} (loading latest.pth.tar)", last_gen)
        nnet.load_checkpoint('latest.pth.tar')
        c = Coach(game, nnet, args, resume=True, resume_wandb_run_id=marker.get("wandb_run_id"))
        c.load_self_play_history_for_resume(last_gen)
        start_generation = last_gen + 1
    else:
        if args.load_model:
            logger.info("Loading checkpoint from best.pth.tar...")
            nnet.load_checkpoint('best.pth.tar')
        else:
            logger.warning('Not loading a checkpoint!')

        logger.info('Loading the Coach...')
        c = Coach(game, nnet, args)

        if args.load_model:
            logger.info("Loading self-play history...")
            c.load_self_play_history(up_to_generation=0)
        start_generation = 1

    logger.info('Starting the learning process')
    c.learn(start_generation=start_generation)

    # A finished training run must not be sunk by report rendering (R7): all data
    # is already on disk, so log and continue — regenerate later with --report-only.
    try:
        create_html_report(args)
    except Exception:
        logger.exception(
            "Report generation failed, but training data is intact. "
            "Regenerate with: --report-only.",
        )

    end = time.perf_counter()
    logger.info(f"Total time elapsed: {end - start}")


if __name__ == "__main__":
    main()
