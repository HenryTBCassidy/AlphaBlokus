from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from loguru import logger
from tqdm import tqdm

from alphablokus.evaluation.players import sample_opening_prefix
from alphablokus.interfaces import RESIGN_ACTION, IBoard, IGame

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphablokus.evaluation.players import Player

# Type aliases for improved readability
DisplayFn: TypeAlias = Callable[[IBoard], None]  # Function to display the game board
GameResult: TypeAlias = int | float  # Game outcome (-1, 0, 1, or small float for draws)


@dataclass(frozen=True)
class MoveRecord:
    """One move within a recorded arena game.

    ``top_k_actions`` and ``top_k_probs`` are populated when the player
    exposes a ``get_last_policy()`` method (i.e. it's a NetworkPlayer);
    for Random / Minimax / etc they are empty lists.

    ``played_prob`` is the action's *raw MCTS visit fraction* — what share
    of total MCTS visits the played action received. Stored explicitly
    because with sparse policies (e.g. Blokus with 50 sims over 17k
    actions) the played action may be tied with many others and fall
    outside the top-K storage window even though it was the one MCTS
    selected. Defaults to 0.0 for non-NetworkPlayer moves where no policy
    was exposed.
    """

    player: int  # +1 or -1 — who moved
    action: int  # action index chosen
    top_k_actions: tuple[int, ...]  # in descending probability order
    top_k_probs: tuple[float, ...]  # aligned with top_k_actions
    played_prob: float = 0.0  # raw visit fraction for the played action


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

    def __init__(self, player1: Player, player2: Player, game: IGame, display: DisplayFn | None = None) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(
        self,
        verbose: bool = False,
        record: bool = False,
        top_k: int = 5,
        forced_opening: tuple[int, ...] = (),
    ) -> tuple[GameResult, GameRecord | None]:
        """
        Execute one complete game between the two players.

        The game continues until a terminal state is reached. Each turn:
        1. The current player observes the board state
        2. The player chooses an action
        3. The action is validated and applied
        4. The game checks if a terminal state is reached

        Args:
            verbose: Whether to display the game state after each move.
            record: If True, also return a ``GameRecord`` capturing every
                move and the players' top-K policy info (when available).
            top_k: How many candidate actions to retain per move when
                ``record=True``. Only NetworkPlayer-style players expose a
                policy; for others the lists will be empty.
            forced_opening: An optional scripted opening prefix. For the first
                ``len(forced_opening)`` plies the game replays ``forced_opening``
                verbatim — applied to *whichever* side is to move — instead of
                consulting the player to move. Used by :meth:`play_games_paired`
                so the two colour-swapped games of a pair share an identical
                opening (the first-mover advantage then cancels). Forced plies
                record an empty top-K policy (the player was not searched).

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
        recorded_moves: list[MoveRecord] = []

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

            # Forced opening: replay the shared scripted prefix for its length,
            # bypassing the player to move (so a colour-swapped pair plays the
            # identical opening). These plies are legal by construction (sampled
            # from real play) and never a resignation, so no policy is recorded.
            forced = move_count <= len(forced_opening)
            action = forced_opening[move_count - 1] if forced else current_player(canonical_board)

            # A player may resign instead of moving (e.g. Pentobi's GTP genmove returns
            # "resign"). Score it immediately as a loss for the resigner — before the
            # legality assert, since RESIGN_ACTION is not a valid board move. Outcome is
            # from player1's (cur_player==1 slot) perspective, so the opponent's win is
            # simply -cur_player.
            if action == RESIGN_ACTION:
                resign_outcome: GameResult = float(-cur_player)
                logger.info("Player {} resigned on move {} → outcome {}", cur_player, move_count, resign_outcome)
                resign_record: GameRecord | None = None
                if record:
                    resign_record = GameRecord(
                        moves=tuple(recorded_moves),
                        outcome=resign_outcome,
                        player1_was_white=True,  # set per-game by play_games when it alternates
                    )
                # Give players a chance to tear down as they would at a normal game end.
                for player in players.values():
                    if hasattr(player, "endGame"):
                        player.endGame()
                return resign_outcome, resign_record

            valids = self.game.valid_move_masking(canonical_board, 1)

            if record:
                if forced:
                    top_actions: list[int] = []
                    top_probs: list[float] = []
                    played_prob = 0.0
                else:
                    top_actions, top_probs, played_prob = _extract_top_k(
                        current_player,
                        top_k,
                        played_action=int(action),
                    )
                recorded_moves.append(
                    MoveRecord(
                        player=cur_player,
                        action=int(action),
                        top_k_actions=tuple(top_actions),
                        top_k_probs=tuple(top_probs),
                        played_prob=played_prob,
                    )
                )

            if valids[action] == 0:
                logger.error(f"Action {action} is not valid!")
                logger.debug(f"valids = {valids}")
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
                records.append(
                    GameRecord(
                        moves=rec.moves,
                        outcome=rec.outcome,
                        player1_was_white=True,
                    )
                )

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
                records.append(
                    GameRecord(
                        moves=rec.moves,
                        outcome=-rec.outcome,
                        player1_was_white=False,
                    )
                )

        # Swap back so the Arena ends in its original state.
        self.player1, self.player2 = self.player2, self.player1
        return one_won, two_won, draws, records

    def play_games_paired(
        self,
        num: int,
        prefix_sampler: Player,
        opening_moves: int,
        verbose: bool = False,
        record: bool = False,
        top_k: int = 5,
    ) -> tuple[int, int, int, list[GameRecord]]:
        """Play colour-swapped *paired* games to cancel the first-mover advantage.

        ``num`` is split into ``num // 2`` pairs. For each pair we sample one
        shared opening prefix (``opening_moves`` plies from ``prefix_sampler``,
        see :func:`alphablokus.evaluation.players.sample_opening_prefix`) and
        play it out to completion **twice** — once with ``player1`` as White and
        once with ``player2`` as White — both replaying that identical prefix.
        Because both sides play the same opening from both colours, the ~96%
        first-mover advantage cancels within each pair, so the returned score
        reflects true net-strength differential rather than a colour coin-flip
        (plateau-investigation §2 B8).

        **Scoring — paired win-differential (rule (a)).** Each pair contributes
        ``candidate_wins − incumbent_wins ∈ {−2,−1,0,+1,+2}``. Aggregated
        linearly across pairs and mapped to ``[0, 1]``, this is algebraically
        identical to the ordinary score ``(p2_wins + 0.5·draws) / total`` tallied
        over all ``2·num_pairs`` games — summation is linear, so the higher
        resolution comes from the *variance reduction* of shared-opening pairing,
        not from a different arithmetic. We therefore return plain integer game
        tallies and let the existing acceptance/Elo score rule compute rule (a)
        unchanged. (Rule (b), pair-outcome win/split/loss, would *not* reduce to
        the game tally; we deliberately chose (a) — see
        docs/plans/fix-arena-colour-pinning.md S1.)

        Args:
            num: Total games (rounded down to an even ``num // 2`` pairs × 2).
            prefix_sampler: Player used to sample each pair's shared opening
                prefix (typically the incumbent at a >0 opening temperature).
            opening_moves: Plies in each sampled prefix (0 disables the prefix —
                pairs then play deterministic clones, still colour-cancelled).
            verbose: Whether to display each game state.
            record: If True, also return a list of ``GameRecord`` (one per game),
                tagged with ``player1_was_white`` exactly as :meth:`play_games`.
            top_k: How many top moves to record per move when ``record=True``.

        Returns:
            ``(player1_wins, player2_wins, draws, records)`` tallied over the
            paired games, from player1's perspective — same shape as
            :meth:`play_games`.
        """
        num_pairs = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        records: list[GameRecord] = []

        for _ in tqdm(range(num_pairs), desc="Arena.playGamesPaired"):
            prefix = sample_opening_prefix(self.game, prefix_sampler, opening_moves)

            # Game A: player1 plays White, replaying the shared prefix.
            result_a, rec_a = self.play_game(verbose=verbose, record=record, top_k=top_k, forced_opening=prefix)
            if result_a == 1:
                one_won += 1
            elif result_a == -1:
                two_won += 1
            else:
                draws += 1
            if rec_a is not None:
                records.append(GameRecord(moves=rec_a.moves, outcome=rec_a.outcome, player1_was_white=True))

            # Game B: swap colours, replay the SAME prefix. play_game now reports
            # from the swapped player1's (original player2's) perspective, so flip
            # back to original player1's frame — mirroring play_games' second half.
            self.player1, self.player2 = self.player2, self.player1
            result_b, rec_b = self.play_game(verbose=verbose, record=record, top_k=top_k, forced_opening=prefix)
            self.player1, self.player2 = self.player2, self.player1  # restore original orientation
            if result_b == -1:
                one_won += 1
            elif result_b == 1:
                two_won += 1
            else:
                draws += 1
            if rec_b is not None:
                records.append(GameRecord(moves=rec_b.moves, outcome=-rec_b.outcome, player1_was_white=False))

        return one_won, two_won, draws, records


def _extract_top_k(
    player: Player,
    k: int,
    played_action: int | None = None,
) -> tuple[list[int], list[float], float]:
    """Pull top-K **visited** actions + probs + the played action's prob.

    Players with ``get_last_policy()`` (i.e. ``NetworkPlayer``) return
    their full MCTS visit-count distribution; we sort and take the K
    entries with highest probability — but only entries with ``prob > 0``.

    The zero-probability filter matters for sparse policies: with (say)
    50 MCTS sims over Blokus's 17,837-action space, only ~15-20 actions
    get any visits at all. Without the filter, ``argpartition`` would
    deterministically pad the top-K with arbitrary unvisited actions —
    and those unvisited actions might not even be legal.

    The played action's probability is returned separately so it can be
    surfaced in the replay viewer even when the played action ties with
    many others on visit count and falls outside the top-K window —
    which happens often with low sim counts on Blokus.

    Returns ``(top_actions, top_probs, played_prob)``. For
    non-NetworkPlayer moves where no policy is exposed, returns
    ``([], [], 0.0)``.
    """
    if not hasattr(player, "get_last_policy"):
        return [], [], 0.0
    pi: NDArray | None = player.get_last_policy()
    if pi is None:
        return [], [], 0.0
    nonzero_idx = np.flatnonzero(pi > 0)
    if len(nonzero_idx) == 0:
        played_prob = float(pi[played_action]) if played_action is not None else 0.0
        return [], [], played_prob
    nonzero_probs = pi[nonzero_idx]
    order_within = np.argsort(-nonzero_probs)[:k]
    top_actions = nonzero_idx[order_within].tolist()
    top_probs = pi[top_actions].tolist()
    played_prob = float(pi[played_action]) if played_action is not None else 0.0
    return top_actions, top_probs, played_prob
