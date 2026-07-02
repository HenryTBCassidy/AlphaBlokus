"""Host-side assembly of actor traces into training games (plan step G5).

Consumes :class:`core.jaxplay.actors.WaveTrace` rows (as numpy) and produces
``list[ProcessedExample]`` per completed game in the **exact** representation
``core/self_play.py::play_self_play_episode`` stores:

- board: canonical compact int8 ``(14, 14)`` placement board — current
  player's pieces positive (``get_canonical_form(board, p).to_compact()``
  equals ``raw_ppb * p`` reshaped);
- policy: ``sparsify``-ed dense visit distribution; the transpose-augmented
  twin uses the game's own ``transpose_policy`` permutation, so augmentation
  is bit-compatible with ``get_symmetries`` (identity + main-diagonal
  transpose, 2× examples);
- value: game outcome from that position's player perspective, with the python
  path's draw-sign convention (+1e-4 to the terminal state's player-to-move,
  −1e-4 to the opponent — see ``self_play.py:109-116``).

Games truncated by the end of the final wave are dropped (and counted) — the
same trailing-truncation policy as pgx's value mask; nothing is silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from core.sparse_policy import sparsify

if TYPE_CHECKING:
    from core.jaxplay.actors import WaveTrace
    from core.self_play import ProcessedExample
    from games.blokusduo.game import BlokusDuoGame

#: Board side (Blokus Duo); kept in sync with BlokusDuoBoard.N via the tests.
_BOARD_SIZE = 14

#: get_game_ended's draw sentinel.
_DRAW_VALUE = 1e-4


@dataclass
class _OpenGame:
    """Positions of a not-yet-finished game in one actor slot."""

    boards: list[np.ndarray] = field(default_factory=list)  # canonical (14,14) int8
    players: list[int] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)  # dense float32 (A,)
    entropies: list[float] = field(default_factory=list)


@dataclass
class GameRecord:
    """One completed game plus the diagnostics the stats layer needs."""

    examples: list[ProcessedExample]
    num_moves: int
    mean_policy_entropy: float


class TraceHarvester:
    """Stateful across waves: carries each slot's open game between harvests."""

    def __init__(self, game: BlokusDuoGame, batch_size: int) -> None:
        self._game = game
        self._action_size = game.get_action_size()
        self._transpose_perm = self._build_transpose_perm()
        self._slots = [_OpenGame() for _ in range(batch_size)]
        self.truncated_games = 0  # open games discarded by finalize()

    def _build_transpose_perm(self) -> np.ndarray:
        # transpose_policy(pi)[a] = pi[perm[a]]; build once via a probe of the
        # game's own permutation so augmentation can't drift from get_symmetries.
        probe = np.arange(self._action_size, dtype=np.int64)
        return self._game.transpose_policy(probe).astype(np.int64)

    def harvest(self, trace: WaveTrace) -> list[GameRecord]:
        """Fold one wave's trace into slot buffers; return games completed."""
        ppb = np.asarray(trace.ppb)  # (T, B, 196) int8
        player = np.asarray(trace.player)
        move_count = np.asarray(trace.move_count)
        weights = np.asarray(trace.action_weights)  # (T, B, K) float32
        topk_ids = np.asarray(trace.topk_ids)
        terminated = np.asarray(trace.terminated)
        result_white = np.asarray(trace.result_white)
        end_player = np.asarray(trace.end_player)

        completed: list[GameRecord] = []
        plies, batch = terminated.shape
        for t in range(plies):
            for b in range(batch):
                slot = self._slots[b]
                mover = int(player[t, b])
                assert move_count[t, b] == len(slot.boards), (
                    f"slot {b} trace desync: move_count {move_count[t, b]} vs buffered {len(slot.boards)}"
                )
                dense = np.zeros(self._action_size, dtype=np.float32)
                np.add.at(dense, topk_ids[t, b], weights[t, b])
                canonical = (ppb[t, b].astype(np.int8) * mover).reshape(_BOARD_SIZE, _BOARD_SIZE)
                slot.boards.append(canonical)
                slot.players.append(mover)
                slot.policies.append(dense)
                nonzero = dense[dense > 0]
                slot.entropies.append(float(-(nonzero * np.log(nonzero)).sum()))
                if terminated[t, b]:
                    completed.append(self._finish_game(slot, float(result_white[t, b]), int(end_player[t, b])))
                    self._slots[b] = _OpenGame()
        return completed

    def _finish_game(self, slot: _OpenGame, result_white: float, end_player: int) -> GameRecord:
        examples: list[ProcessedExample] = []
        is_draw = abs(result_white) < 0.5  # results are ±1.0 or the ~1e-4 draw sentinel (float32)
        for board, mover, dense in zip(slot.boards, slot.players, slot.policies, strict=True):
            if is_draw:
                value = _DRAW_VALUE if mover == end_player else -_DRAW_VALUE
            else:
                value = float(np.sign(result_white)) if mover == 1 else -float(np.sign(result_white))
            transposed_board = board.T.copy()
            transposed_pi = dense[self._transpose_perm]
            examples.append((board, sparsify(dense), float(value)))
            examples.append((transposed_board, sparsify(transposed_pi), float(value)))
        return GameRecord(
            examples=examples,
            num_moves=len(slot.boards),
            mean_policy_entropy=float(np.mean(slot.entropies)) if slot.entropies else 0.0,
        )

    def finalize(self) -> None:
        """Drop (and count) games left open when generation ends — truncated tails."""
        self.truncated_games = sum(1 for slot in self._slots if slot.boards)
