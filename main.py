import logging
import time

import coloredlogs

from core.coach import Coach
from core.config import LOGGER_NAME
from tictactoe.neuralnets.wrapper import NNetWrapper
from tictactoe.tictactoegame import TicTacToeGame as Game
from utils import setup_logging, load_args

log = logging.getLogger(LOGGER_NAME)
coloredlogs.install(level='INFO')

args = load_args("full_run.json")


def main():
    args.run_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(args.log_directory)
    start = time.perf_counter()

    log.info('Loading %s...', Game.__name__)
    g = Game(3)

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g, args)

    # TODO: Fix this
    if args.load_model:
        # log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        raise NotImplementedError("You do not have a way of loading models currently!")
        # nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    # TODO: Fix this
    if args.load_model:
        raise NotImplementedError("You do not have a way of loading models currently!")
        log.info("Loading 'trainExamples' from file...")
        c.load_train_examples()

    log.info('Starting the learning process ')

    c.learn()
    nnet.collect_training_data()
    end = time.perf_counter()
    log.info(f"Total time elapsed: {end - start}")


if __name__ == "__main__":
    main()
