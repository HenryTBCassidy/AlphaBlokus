"""Engine-interface backend over the real game + net + MCTS stack.

Game-agnostic by construction: the wire format is flat action ids plus a
move history, so everything routes through the ``IGame``/``INeuralNetWrapper``
protocols — no ``games.*`` imports (composition happens in ``registry.py``
via ``server.py``). Stateless: each request replays its (short) move history
from the initial board, so the server holds no per-session state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from alphablokus.config import MCTSConfig
from alphablokus.search.mcts import MCTS

if TYPE_CHECKING:
    from alphablokus.interfaces import IBoard, IGame, INeuralNetWrapper


@dataclass(frozen=True)
class ServerDifficulty:
    """One selectable strength level of the local server."""

    id: str
    label: str
    search_policy: Literal["policy", "puct"]
    sims: int
    description: str


#: Levels 1-4 mirror the browser tier's budgets (same net, same search family)
#: so strength labels carry over; level 5 is the uncompromised full-strength
#: setting — the training-arena PUCT budget with batched leaf evaluation.
SERVER_DIFFICULTIES: tuple[ServerDifficulty, ...] = (
    ServerDifficulty("level-1", "Level 1 — Instinct", "policy", 0, "Raw policy network, no search."),
    ServerDifficulty("level-2", "Level 2 — Quick", "puct", 32, "PUCT search, 32 simulations per move."),
    ServerDifficulty("level-3", "Level 3 — Club", "puct", 128, "PUCT search, 128 simulations per move."),
    ServerDifficulty("level-4", "Level 4 — Strong", "puct", 400, "PUCT search, 400 simulations per move."),
    ServerDifficulty(
        "level-5", "Level 5 — Max", "puct", 800, "Full-strength PUCT at the training-arena budget (800 simulations)."
    ),
)


@dataclass(frozen=True)
class BestMoveResult:
    """One engine reply, ready to serialise."""

    action: int
    value: float
    legal: list[int]
    sims: int
    elapsed_ms: float


class PlayService:
    """Answers the frontend's engine interface with the real torch/MCTS stack."""

    def __init__(self, game: IGame, nnet: INeuralNetWrapper, *, cpuct: float = 2.5, mcts_batch_size: int = 8) -> None:
        self._game = game
        self._nnet = nnet
        self._cpuct = cpuct
        self._mcts_batch_size = mcts_batch_size
        self._difficulties = {level.id: level for level in SERVER_DIFFICULTIES}

    @property
    def difficulties(self) -> tuple[ServerDifficulty, ...]:
        return SERVER_DIFFICULTIES

    @property
    def action_size(self) -> int:
        return self._game.get_action_size()

    def replay(self, history: list[int]) -> tuple[IBoard, int]:
        """Rebuild (board, player-to-move) from a move history off the initial board."""
        board = self._game.initialise_board()
        player = 1
        for action in history:
            board, player = self._game.get_next_state(board, player, action)
        return board, player

    def legal_actions(self, history: list[int]) -> list[int]:
        """Legal action ids for the player to move after ``history``."""
        board, player = self.replay(history)
        mask = self._game.valid_move_masking(board, player)
        return [int(action) for action in np.flatnonzero(mask)]

    def best_move(self, history: list[int], difficulty_id: str) -> BestMoveResult:
        """Choose a move for the player to move after ``history``.

        Raw-policy levels take the network argmax over legal moves; search
        levels run a fresh PUCT MCTS (no Dirichlet noise — evaluation-style
        play) and pick the most-visited action.
        """
        difficulty = self._difficulties.get(difficulty_id)
        if difficulty is None:
            raise KeyError(f"Unknown difficulty {difficulty_id!r}. Expected one of {sorted(self._difficulties)}.")

        start = time.perf_counter()
        board, player = self.replay(history)
        canonical = self._game.get_canonical_form(board, player)
        mask = self._game.valid_move_masking(board, player)
        legal = [int(action) for action in np.flatnonzero(mask)]

        priors, value = self._nnet.predict(canonical)
        if difficulty.search_policy == "policy" or len(legal) == 1:
            legal_priors = [priors[action] for action in legal]
            action = legal[int(np.argmax(legal_priors))]
        else:
            mcts_config = MCTSConfig(
                num_mcts_sims=difficulty.sims,
                cpuct=self._cpuct,
                profiling_level="none",
                mcts_batch_size=self._mcts_batch_size,
            )
            mcts = MCTS(self._game, self._nnet, mcts_config)
            probs = mcts.get_action_prob(canonical, temp=0)
            action = int(np.argmax(probs))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return BestMoveResult(
            action=action,
            value=float(value),
            legal=legal,
            sims=difficulty.sims,
            elapsed_ms=elapsed_ms,
        )
