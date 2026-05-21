from dataclasses import dataclass, field
from typing import Callable, TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from tqdm import tqdm

from core.interfaces import IBoard, IGame


# Type aliases for improved readability
Player: TypeAlias = Callable[[IBoard], int]  # Function that takes a board state and returns an action
DisplayFn: TypeAlias = Callable[[IBoard], None]  # Function to display the game board
GameResult: TypeAlias = int | float  # Game outcome (-1, 0, 1, or small float for draws)


@dataclass(frozen=True)
class MoveRecord:
    """One move within a recorded arena game.

    ``top_k_actions`` and ``top_k_probs`` are populated when the player
    exposes a ``get_last_policy()`` method (i.e. it's a NetworkPlayer); for
    Random / Minimax / etc they are empty lists.
    """
    player: int          # +1 or -1 — who moved
    action: int          # action index chosen
    top_k_actions: tuple[int, ...]  # in descending probability order
    top_k_probs: tuple[float, ...]  # aligned with top_k_actions


@dataclass(frozen=True)
class GameRecord:
    """A recorded arena game — moves + final outcome.

    ``outcome`` is from player 1's perspective: +1 if player1 won, -1 if
    player2 won, ~0 for a draw.
    """
    moves: tuple[MoveRecord, ...]
    outcome: GameResult
    player1_was_white: bool  # which side player1 played (alternates in play_games)


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
        display: DisplayFn | None = None
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

    def play_game(
        self,
        verbose: bool = False,
        record: bool = False,
        top_k: int = 5,
    ) -> tuple[GameResult, GameRecord | None]:
        """
        Execute one complete game between the two players.

        The game continues until a terminal state is reached. Each turn:
        1. The current player observes the board state
        2. The player chooses an action
        3. The action is validated and applied
        4. The game checks if a terminal state is reached

        Args:
            verbose: Whether to display the game state after each move

        Args:
            verbose: Whether to display the game state after each move.
            record: If True, also return a ``GameRecord`` capturing every
                move and the players' top-K policy info (when available).
            top_k: How many candidate actions to retain per move when
                ``record=True``. Only NetworkPlayer-style players expose a
                policy; for others the lists will be empty.

        Returns:
            ``(outcome, record)``. ``record`` is None if ``record=False``.

        Raises:
            AssertionError: If a player attempts an invalid move
            AssertionError: If verbose=True but no display function was provided
        """
        players: dict[int, Player] = {1: self.player1, -1: self.player2}
        cur_player = 1
        board = self.game.initialise_board()
        move_count = 0
        recorded_moves: list[MoveRecord] = [] if record else []

        # Initialize players if they have a start-game hook
        for player in players.values():
            if hasattr(player, "startGame"):
                player.startGame()

        # Main game loop
        while self.game.get_game_ended(board, cur_player) == 0:
            move_count += 1
            if verbose:
                assert self.display, "Display function must be provided for verbose mode"
                print(f"Turn {move_count}, Player {cur_player}")
                self.display(board)

            # Get and validate the player's action
            current_player = players[cur_player]

            canonical_board = self.game.get_canonical_form(board, cur_player)
            action = current_player(canonical_board)
            valids = self.game.valid_move_masking(canonical_board, 1)

            if record:
                top_actions, top_probs = _extract_top_k(current_player, top_k)
                recorded_moves.append(MoveRecord(
                    player=cur_player,
                    action=int(action),
                    top_k_actions=tuple(top_actions),
                    top_k_probs=tuple(top_probs),
                ))

            if valids[action] == 0:
                logger.error(f'Action {action} is not valid!')
                logger.debug(f'valids = {valids}')
                assert valids[action] > 0, f"Player {cur_player} attempted invalid move {action}"

            # Notify opponent of the move if they implement the notification hook
            opponent = players[-cur_player]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            # Apply the move and switch players
            board, cur_player = self.game.get_next_state(board, cur_player, action)

        # Cleanup - call end-game hooks if implemented
        for player in players.values():
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display, "Display function must be provided for verbose mode"
            print(f"Game over: Turn {move_count}, Result {self.game.get_game_ended(board, 1)}")
            self.display(board)

        outcome = cur_player * self.game.get_game_ended(board, cur_player)
        recorded = None
        if record:
            recorded = GameRecord(
                moves=tuple(recorded_moves),
                outcome=outcome,
                player1_was_white=True,  # set per-game by play_games when it alternates
            )
        return outcome, recorded


    def play_games(
        self,
        num: int,
        verbose: bool = False,
        record: bool = False,
        top_k: int = 5,
    ) -> tuple[int, int, int, list[GameRecord]]:
        """
        Play multiple games between the two players with alternating start positions.

        To ensure fairness, each player starts an equal number of games. The total
        number of games played will be the nearest even number <= num.

        Args:
            num: Number of games to play (will be rounded down to nearest even number)
            verbose: Whether to display each game state
            record: If True, also return a list of GameRecord objects (one per game).
            top_k: How many top moves to record per move when ``record=True``.

        Returns:
            ``(player1_wins, player2_wins, draws, records)``. ``records`` is an
            empty list when ``record=False``.
        """
        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        records: list[GameRecord] = []

        # First half: player1 starts
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            game_result, rec = self.play_game(verbose=verbose, record=record, top_k=top_k)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
            if rec is not None:
                # First half: player1 played as White (cur_player=1 starts).
                records.append(GameRecord(
                    moves=rec.moves, outcome=rec.outcome, player1_was_white=True,
                ))

        # Swap players for second half
        self.player1, self.player2 = self.player2, self.player1

        # Second half: original player2 starts
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            game_result, rec = self.play_game(verbose=verbose, record=record, top_k=top_k)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1
            if rec is not None:
                # Second half: the (swapped) self.player1 is actually original
                # player2. From original-player1's perspective, they played as
                # Black this game.
                records.append(GameRecord(
                    moves=rec.moves, outcome=-rec.outcome, player1_was_white=False,
                ))

        # Swap back so the Arena ends in its original state.
        self.player1, self.player2 = self.player2, self.player1
        return one_won, two_won, draws, records


def _extract_top_k(player: Player, k: int) -> tuple[list[int], list[float]]:
    """Pull top-K actions + probs from a player's last policy, if exposed.

    Players that have ``get_last_policy()`` (i.e. NetworkPlayer) return their
    full policy vector; we sort and take the K entries with highest probability.
    Players that don't expose policy return ``([], [])`` — fine for random /
    minimax opponents, where top-K isn't meaningful.
    """
    if not hasattr(player, "get_last_policy"):
        return [], []
    pi: NDArray | None = player.get_last_policy()
    if pi is None:
        return [], []
    if k >= len(pi):
        order = np.argsort(-pi)
    else:
        # argpartition + sort within the top slice for efficiency at huge action sizes.
        partition = np.argpartition(-pi, k - 1)[:k]
        order = partition[np.argsort(-pi[partition])]
    top_actions = order.tolist()
    top_probs = pi[order].tolist()
    return top_actions, top_probs
