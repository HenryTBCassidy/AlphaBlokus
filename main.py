import argparse
import time
from pathlib import Path

from loguru import logger

from core.coach import Coach
from core.config import RunConfig, load_args
from core.interfaces import IGame, INeuralNetWrapper
from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper as TicTacToeNNetWrapper
from reporting import create_html_report

GAMES_DIR = Path(__file__).parent / "games"


def initialise_game_and_network(config: RunConfig) -> tuple[IGame, INeuralNetWrapper]:
    """Instantiate the game and neural net wrapper from the run config."""
    match config.game:
        case "tictactoe":
            game = TicTacToeGame()
            nnet = TicTacToeNNetWrapper(game, config)
        case "blokusduo":
            game = BlokusDuoGame(pieces_config_path=GAMES_DIR / "blokusduo" / "pieces.json")
            nnet = BlokusDuoNNetWrapper(game, config)
        case unknown:
            raise ValueError(f"Unknown game: {unknown!r}. Expected 'tictactoe' or 'blokusduo'.")
    return game, nnet


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

    logger.info(f"Loading game: {args.game}")
    game, nnet = initialise_game_and_network(args)

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

    logger.info('Starting the learning process')

    c.learn()
    create_html_report(args)
    end = time.perf_counter()
    logger.info(f"Total time elapsed: {end - start}")


if __name__ == "__main__":
    main()
