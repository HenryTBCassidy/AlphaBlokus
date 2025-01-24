import logging
import time
from pathlib import Path

import coloredlogs

from coach import Coach
from config import RunConfig, NetConfig, MCTSConfig, LOGGER_NAME
from tictactoe.tictactoegame import TicTacToeGame as Game
from tictactoe.neuralnets.wrapper import NNetWrapper
from utils import setup_logging

log = logging.getLogger(LOGGER_NAME)

# Change this to DEBUG to see more info.
coloredlogs.install(level='INFO')

# TODO: Add args for whether or not to log out data
args = RunConfig(
    run_name="tiny_check_run_1",
    num_generations=1,  # 30
    num_eps=10,  # 100 TODO: These can happen in parallel
    temp_threshold=15,
    update_threshold=0.55,
    max_queue_length=10,  # 10000 TODO: Why is this so high?
    num_arena_matches=2,  # 50 TODO: These can happen in parallel
    root_directory=Path('./temp/'),
    load_model=False,
    load_folder_file=[Path(p) for p in ('/dev/models/8x100x50', 'best.pth.tar')],
    max_generations_lookback=1,  # 10 This is how many iterations of training history the algorith will look back
    mcts_config=MCTSConfig(
        num_mcts_sims=2,  # 25
        cpuct=1
    ),
    net_config=NetConfig(
        learning_rate=0.001,  # TODO: Make a schedule for this
        dropout=0.3,
        epochs=1,  # 10
        batch_size=10,
        cuda=True,
        num_channels=512,
    )
)

args.run_directory.mkdir(parents=True, exist_ok=True)


def main():
    setup_logging(args.log_directory)
    start = time.perf_counter()
    log.info('Loading %s...', Game.__name__)

    g = Game(3)

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g, args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.load_train_examples()

    log.info('Starting the learning process ')

    c.learn()
    end = time.perf_counter()
    log.info(f"Total time elapsed: {end - start}")


if __name__ == "__main__":
    main()
