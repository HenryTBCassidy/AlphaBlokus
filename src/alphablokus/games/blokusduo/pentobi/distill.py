"""Pentobi distillation dataloader: corpus shards → supervised training examples.

The training half's *data* side (plan ``docs/plans/pentobi-distillation.md`` D6): stream
the expert corpus written by :mod:`alphablokus.games.blokusduo.pentobi.corpus` into the
exact ``ProcessedExample`` shape ``BaseNNetWrapper.train`` consumes, so the SL trainer
(``scripts/distill_sl.py``) reuses the whole existing training path (lazy board
re-encode via ``_LazyPolicyDataset``, DataLoader, AdamW) with zero pipeline surgery.

Three training-time transforms happen here, none of which is stored in the corpus:

- **Label smoothing over legal moves.** The stored policy is the one-hot of Pentobi's
  move (behavioural cloning). :func:`smooth_policy` re-targets it as
  ``(1 − ε)·one_hot + ε·uniform(legal)`` — mass is spread over exactly the position's
  legal moves (the valid mask from the rebuilt board), never over illegal actions.
  The smoothed target stays **sparse** (support = the legal set), so it flows through
  the existing sparse-policy machinery (``as_dense`` densifies one batch at a time).
- **Order-2 symmetry augmentation.** Each example gains its main-diagonal twin: the
  transposed compact board (grids transpose directly — ``BlokusDuoBoard.transposed``
  transposes the placement grid) with every policy index mapped through
  ``BlokusDuoGame.transpose_action``. 2× data for one move-generation call, since the
  twin's legal set is exactly the transpose of the original's.
- **Game-granular held-out split.** :func:`load_corpus_games` groups rows by game so
  ``alphablokus.training.holdout.split_games_holdout`` can partition at *game*
  granularity — no position of a held-out game (nor its symmetry twin) ever reaches
  training.

Shards contribute in proportion to the games they hold: :func:`sample_games` draws
uniformly over the pooled game list (never "first N shards"), so a subsampled corpus
keeps the deterministic opening keys' even coverage.

**Memory.** Boards stay compact (196 B each); the smoothed sparse targets carry one
``(int32, float32)`` pair per legal move (~8 B × avg. legal moves ≈ a few KB/position).
A full 50k-game corpus (~3 M positions after augmentation) fits box RAM; ``--max-games``
in the trainer is the relief valve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from alphablokus.games.blokusduo.pentobi.corpus import read_shard_meta

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.corpus import CorpusExample

# Games between progress log lines while building training examples (the legal-mask
# pass is the only non-trivial cost of loading a large corpus).
_LOG_EVERY_GAMES = 1_000


@dataclass(frozen=True)
class CorpusGameRows:
    """One corpus game's stored rows, grouped for game-granular splitting.

    Positions are in play order and the four tuples are index-aligned: position ``i``
    was played by ``players[i]``, Pentobi chose ``actions[i]``, and the game outcome
    from that side to move is ``values[i]``. ``players`` feeds the colour-conditional
    value-calibration diagnostic (D7); ``actions`` is the top-1 accuracy target.
    """

    game_id: int
    boards: tuple[NDArray[np.int8], ...]  # canonical compact boards (side-to-move)
    actions: tuple[int, ...]  # the action index Pentobi played per position
    players: tuple[int, ...]  # side to move per position: +1 White, -1 Black
    values: tuple[float, ...]  # outcome from the side to move: +1 / -1 / 0

    def __len__(self) -> int:
        return len(self.boards)


def load_corpus_games(paths: Sequence[Path]) -> list[CorpusGameRows]:
    """Read corpus shards into per-game row groups.

    Rows within a shard are stored game-by-game in play order (``write_shard``'s
    layout, sizes in the footer's ``game_sizes``), so grouping is a cursor walk —
    no sort, no join. The per-slice ``game_id`` column is cross-checked against the
    footer metadata as a cheap corruption guard.

    Args:
        paths: Shard files (e.g. from ``corpus_shards(directory)``).

    Returns:
        One :class:`CorpusGameRows` per stored game, in shard-then-play order.
    """
    games: list[CorpusGameRows] = []
    for path in paths:
        meta = read_shard_meta(path)
        table = pq.read_table(path, columns=["board", "action", "player", "value", "game_id"])
        boards = table.column("board").to_pylist()
        actions = table.column("action").to_pylist()
        players = table.column("player").to_pylist()
        values = table.column("value").to_pylist()
        game_ids = table.column("game_id").to_pylist()
        if len(boards) != sum(meta.game_sizes):
            raise ValueError(f"{path.name}: {len(boards)} rows != game_sizes total {sum(meta.game_sizes)}")
        cursor = 0
        for game_meta, size in zip(meta.games, meta.game_sizes, strict=True):
            rows = slice(cursor, cursor + size)
            if any(int(g) != game_meta.game_id for g in game_ids[rows]):
                raise ValueError(f"{path.name}: rows {rows} do not all belong to game {game_meta.game_id}")
            games.append(
                CorpusGameRows(
                    game_id=game_meta.game_id,
                    boards=tuple(
                        np.frombuffer(b, dtype=np.dtype(meta.board_dtype)).reshape(meta.board_shape).copy()
                        for b in boards[rows]
                    ),
                    actions=tuple(int(a) for a in actions[rows]),
                    players=tuple(int(p) for p in players[rows]),
                    values=tuple(float(v) for v in values[rows]),
                ),
            )
            cursor += size
    return games


def sample_games(games: Sequence[CorpusGameRows], max_games: int, seed: int) -> list[CorpusGameRows]:
    """Subsample ``max_games`` games uniformly over the pooled corpus.

    Drawing over the *pooled* game list weights each shard by the games it holds
    (vs. "take the first N shards", which would truncate the deterministic opening
    sweep and bias opening coverage). Deterministic in ``seed``; a ``max_games`` at
    or above the corpus size returns every game unchanged.
    """
    if max_games >= len(games):
        return list(games)
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(len(games), size=max_games, replace=False).tolist())
    return [game for i, game in enumerate(games) if i in keep]


def smooth_policy(
    action: int,
    legal_actions: NDArray[np.int32],
    epsilon: float,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Label-smooth a one-hot expert move over the position's legal moves.

    The target is ``(1 − ε)·one_hot(action) + ε·uniform(legal_actions)``, returned
    sparse as ``(indices, values)`` with support exactly the legal set — the played
    action carries ``1 − ε + ε/L`` and every other legal move ``ε/L``, so the vector
    sums to 1 and puts zero mass on illegal actions. ``ε = 0`` reproduces the stored
    one-hot restricted to a wider (legal) support of explicit zeros — callers wanting
    the pure one-hot should just use the stored policy.

    Args:
        action: The expert (Pentobi) action index.
        legal_actions: Legal action indices for the position (must contain ``action``).
        epsilon: Smoothing mass ε in ``[0, 1)``.

    Raises:
        ValueError: If ``epsilon`` is out of range or ``action`` is not legal — the
            latter means the corpus and the rules engine disagree (desync).
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    matches = np.flatnonzero(legal_actions == action)
    if matches.size != 1:
        raise ValueError(f"expert action {action} is not in the position's legal set")
    values = np.full(len(legal_actions), epsilon / len(legal_actions), dtype=np.float32)
    values[int(matches[0])] += np.float32(1.0 - epsilon)
    return legal_actions.astype(np.int32, copy=True), values


def build_training_examples(
    game: BlokusDuoGame,
    games: Sequence[CorpusGameRows],
    *,
    epsilon: float,
    augment: bool,
) -> list[CorpusExample]:
    """Turn corpus games into net-ready ``(board, sparse_policy, value)`` examples.

    Each position yields its smoothed-target example, and — with ``augment`` — its
    main-diagonal symmetry twin directly after it: the transposed compact board with
    the smoothed policy's support mapped through ``transpose_action``. The twin costs
    no second move generation (transposition is a bijection on legal moves), so the
    one legal-mask pass here is the whole load-time cost of a corpus.

    Position order is preserved (twins interleaved when augmenting), so with
    ``augment=False`` the output aligns index-for-index with the flattened
    ``actions``/``players``/``values`` of ``games`` — the alignment the held-out
    diagnostics rely on.

    Args:
        game: The rules engine (rebuilds boards and legal masks; supplies the
            action-transpose bijection).
        games: Corpus games (typically one side of ``split_games_holdout``).
        epsilon: Label-smoothing mass spread over legal moves (plan D6: ~0.1).
        augment: Apply the order-2 symmetry as 2× augmentation (train side yes,
            held-out side no).
    """
    examples: list[CorpusExample] = []
    for count, rows in enumerate(games, start=1):
        for compact, action, value in zip(rows.boards, rows.actions, rows.values, strict=True):
            board = game.board_from_compact(compact)
            legal = np.flatnonzero(game.valid_move_masking(board, 1)).astype(np.int32)
            indices, values = smooth_policy(action, legal, epsilon)
            examples.append((compact, (indices, values), value))
            if augment:
                transposed = np.ascontiguousarray(compact.T)
                transposed_indices = np.array([game.transpose_action(int(a)) for a in indices], dtype=np.int32)
                examples.append((transposed, (transposed_indices, values), value))
        if count % _LOG_EVERY_GAMES == 0:
            logger.info("Built examples for {}/{} games ({} positions so far)", count, len(games), len(examples))
    return examples
