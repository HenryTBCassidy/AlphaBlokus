import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any, Tuple, List, Union, Protocol, TypeAlias
from numpy.typing import NDArray

import pandas as pd
from tqdm import tqdm

from core.config import LOGGER_NAME
from core.interfaces import IGame

log = logging.getLogger(LOGGER_NAME)


# Type aliases for improved readability
Player: TypeAlias = Callable[[NDArray], int]  # Function that takes a board state and returns an action
DisplayFn: TypeAlias = Callable[[NDArray], None]  # Function to display the game board
GameResult: TypeAlias = Union[int, float]  # Game outcome (-1, 0, 1, or small float for draws)


@dataclass(frozen=True)
class ArenaDataLoggable:
    """
    Data class for storing the results of arena evaluation matches.
    
    This class captures the outcomes of evaluation games between two neural networks,
    typically comparing a newly trained network against the previous best network.
    
    Attributes:
        generation: Training iteration number
        wins: Number of games won by the new network
        losses: Number of games lost by the new network
        draws: Number of games that ended in a draw
    """
    generation: int
    wins: int
    losses: int
    draws: int


class Arena:
    """
    Arena for evaluating and comparing game-playing agents.
    
    This class provides functionality to:
    1. Pit any two game-playing agents against each other
    2. Play multiple games with alternating starting positions
    3. Track and record game outcomes for training purposes
    
    The agents can be any callable that takes a board state and returns an action,
    such as:
    - Neural network-based players
    - MCTS-based players
    - Rule-based players
    - Human players
    """

    def __init__(
        self, 
        player1: Player,
        player2: Player,
        game: IGame,
        display: Optional[DisplayFn] = None
    ) -> None:
        """
        Initialize the arena with two players and game rules.

        Args:
            player1: First player (function that takes board state and returns action)
            player2: Second player (function that takes board state and returns action)
            game: Game implementation providing rules and mechanics
            display: Optional function to display the game board (used in verbose mode)
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose: bool = False) -> GameResult:
        """
        Execute one complete game between the two players.

        The game continues until a terminal state is reached. Each turn:
        1. The current player observes the board state
        2. The player chooses an action
        3. The action is validated and applied
        4. The game checks if a terminal state is reached

        Args:
            verbose: Whether to display the game state after each move

        Returns:
            GameResult: The game outcome where:
                       1 = player1 won
                       -1 = player2 won
                       0 = game continues
                       small non-zero value = draw

        Raises:
            AssertionError: If a player attempts an invalid move
            AssertionError: If verbose=True but no display function was provided
        """
        players: List[Optional[Player]] = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.initialise_board()
        move_count = 0

        # Initialize players if they have a start-game hook
        for player in [players[0], players[2]]:
            if player and hasattr(player, "startGame"):
                player.startGame()

        # Main game loop
        while self.game.get_game_ended(board, cur_player) == 0:
            move_count += 1
            if verbose:
                assert self.display, "Display function must be provided for verbose mode"
                print(f"Turn {move_count}, Player {cur_player}")
                self.display(board)

            # Get and validate the player's action
            current_player = players[cur_player + 1]
            assert current_player is not None, "Invalid player index"
            
            canonical_board = self.game.get_canonical_form(board, cur_player)
            action = current_player(canonical_board)
            valids = self.game.valid_move_masking(canonical_board, 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0, f"Player {cur_player} attempted invalid move {action}"

            # Notify opponent of the move if they implement the notification hook
            opponent = players[-cur_player + 1]
            if opponent and hasattr(opponent, "notify"):
                opponent.notify(board, action)

            # Apply the move and switch players
            board, cur_player = self.game.get_next_state(board, cur_player, action)

        # Cleanup - call end-game hooks if implemented
        for player in [players[0], players[2]]:
            if player and hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display, "Display function must be provided for verbose mode"
            print(f"Game over: Turn {move_count}, Result {self.game.get_game_ended(board, 1)}")
            self.display(board)

        return cur_player * self.game.get_game_ended(board, cur_player)

    def play_games(
        self,
        num: int,
        verbose: bool = False,
        generation: Optional[int] = None,
        directory: Optional[Path] = None
    ) -> Tuple[int, int, int]:
        """
        Play multiple games between the two players with alternating start positions.

        To ensure fairness, each player starts an equal number of games. The total
        number of games played will be the nearest even number <= num.

        Args:
            num: Number of games to play (will be rounded down to nearest even number)
            verbose: Whether to display each game state
            generation: Optional training iteration number for logging
            directory: Optional directory to save evaluation results

        Returns:
            Tuple containing:
                - Number of games won by player1
                - Number of games won by player2
                - Number of games that ended in a draw

        Notes:
            If generation and directory are provided, the results will be saved
            to a parquet file in the specified directory.
        """
        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0

        # First half: player1 starts
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

        # Swap players for second half
        self.player1, self.player2 = self.player2, self.player1

        # Second half: original player2 starts
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

        # Save results if logging is enabled
        if generation is not None and directory is not None:
            # Player one is the previous net, player two is the next net
            data = ArenaDataLoggable(generation=generation, wins=two_won, losses=one_won, draws=draws)
            self._write_arena_data(data, generation, directory)

        return one_won, two_won, draws

    @staticmethod
    def _write_arena_data(arena_data: ArenaDataLoggable, generation: int, directory: Path) -> None:
        """
        Save arena evaluation results to a parquet file.

        Args:
            arena_data: Game results to save
            generation: Training iteration number
            directory: Directory to save the results in

        Notes:
            Results are saved in parquet format for efficient storage and reading.
            The filename includes the generation number for easy identification.
        """
        start = time.perf_counter()

        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([arena_data.__dict__]).to_parquet(directory / f"arena_data_{generation}.parquet")

        end = time.perf_counter()
        logging.info(f"Took {end - start} seconds to write arena data for generation # {generation}!")
