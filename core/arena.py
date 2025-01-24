import logging
import time
from pathlib import Path

import pandas as pd
from dataclasses import dataclass

from tqdm import tqdm

from core.config import LOGGER_NAME
from core.interfaces import IGame

log = logging.getLogger(LOGGER_NAME)


@dataclass
class ArenaDataLoggable:
    generation: int
    wins: int
    losses: int
    draws: int


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game: IGame, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.get_init_board()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.get_game_ended(board, cur_player) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(cur_player))
                # self.display(board)
            action = players[cur_player + 1](self.game.get_canonical_form(board, cur_player))

            valids = self.game.get_valid_moves(self.game.get_canonical_form(board, cur_player), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-cur_player + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, cur_player = self.game.get_next_state(board, cur_player, action)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.get_game_ended(board, 1)))
            self.display(board)
        return cur_player * self.game.get_game_ended(board, cur_player)

    def play_games(self, num, verbose=False, generation: int = None, directory: Path = None):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

        if generation and directory:
            # Player one is the previous net, player two is the next net
            data = ArenaDataLoggable(generation=generation, wins=two_won, losses=one_won, draws=draws)
            self._write_arena_data(data, generation, directory)

        return one_won, two_won, draws

    @staticmethod
    def _write_arena_data(arena_data: ArenaDataLoggable, generation: int, directory: Path):
        start = time.perf_counter()

        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([arena_data.__dict__]).to_parquet(directory / f"arena_data_{generation}.parquet")

        end = time.perf_counter()
        logging.info(f"Took {end - start} seconds to write arena data for generation # {generation}!")
