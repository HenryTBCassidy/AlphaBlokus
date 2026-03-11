import argparse
import time

from loguru import logger

from core.coach import Coach
from games.blokusduo.neuralnets.wrapper import NNetWrapper
from games.tictactoe.game import TicTacToeGame as Game
from core.config import load_args
from reporting import create_html_report


def main():
    parser = argparse.ArgumentParser(description="AlphaBlokus training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="run_configurations/test_run.json",
        help="Path to the JSON run configuration file (default: run_configurations/test_run.json)",
    )
    cli_args = parser.parse_args()
    args = load_args(cli_args.config)

    args.run_directory.mkdir(parents=True, exist_ok=True)

    # Add rotating file sink alongside default stderr
    log_dir = args.log_directory
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "alpha.log", rotation="10 MB", retention=3)

    start = time.perf_counter()

    logger.info(f'Loading {Game.__name__}...')
    g = Game(3)

    logger.info(f'Loading {NNetWrapper.__name__}...')
    nnet = NNetWrapper(args)

    # TODO: Fix this
    if args.load_model:
        raise NotImplementedError("You do not have a way of loading models currently!")
    else:
        logger.warning('Not loading a checkpoint!')

    logger.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    # TODO: Fix this
    if args.load_model:
        raise NotImplementedError("You do not have a way of loading models currently!")
        logger.info("Loading 'trainExamples' from file...")
        c.load_train_examples()

    logger.info('Starting the learning process')

    c.learn()
    create_html_report(args)
    end = time.perf_counter()
    logger.info(f"Total time elapsed: {end - start}")


if __name__ == "__main__":
    main()
