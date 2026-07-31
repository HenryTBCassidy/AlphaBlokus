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

**The v2 path (plan ``docs/plans/pentobi-corpus-v2.md`` V9) changes three of those.** The
loss needs no change at all — ``BaseNNetWrapper.loss_pi`` is already a KL against a full
distribution, and v1's one-hot was the degenerate case — so the work is all here:

- :func:`load_corpus_games_v2` carries the **stored soft target** per position, and
  :func:`build_training_examples` uses it directly instead of synthesising one with
  :func:`smooth_policy`. A load-time **target temperature τ** softens it; the legal-set
  floor ε stays available but defaults to 0.
- The holdout splits **by opening subtree, not by game** — :func:`split_opening_units`.
  v2 deliberately gives many games a shared opening, so a game-level split would put
  literally identical opening rows on both sides of the boundary and report a falsely
  good held-out score. v1's split was safe only because every v1 game's opening was
  unique.
- :func:`load_opening_examples` adds the opening dataset (whose value label is chosen by
  :func:`~alphablokus.games.blokusduo.pentobi.corpus_v2.opening_value`), and
  :func:`mix_examples` combines the three sources — opening rows, v2 game rows, v1's
  corpus as a mid-game supplement — at requested proportions rather than natural sizes.

**The auxiliary heads' targets are derived here too, from data already on disk.**
:func:`build_training_examples` returns :class:`TrainingRow` items that carry, alongside
the ``(board, policy, value)`` example, the score head's ``margin`` (a stored column),
the ownership head's final-board map (:func:`final_ownership`, a replay of the stored
actions — no regeneration) and the reply head's target (the *next* position's policy
target, by reference). They ride on the row rather than in parallel lists because
:func:`mix_examples` resamples and shuffles, which no side list could survive in
alignment.

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
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from alphablokus.games.blokusduo.pentobi.corpus import read_shard_meta
from alphablokus.games.blokusduo.pentobi.corpus_v2 import (
    apply_target_temperature,
    opening_value,
    read_game_shard_meta,
    read_opening_meta,
)
from alphablokus.games.blokusduo.pentobi.store import canonical_key

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.corpus import CorpusExample

# Games between progress log lines while building training examples (the legal-mask
# pass is the only non-trivial cost of loading a large corpus).
_LOG_EVERY_GAMES = 1_000

# Strata the v2 holdout samples opening units from, ordered by game mass, so the held-out
# subtrees are not accidentally all tiny ones.
_HOLDOUT_STRATA = 5

# What :func:`mix_examples` resamples — in practice a :class:`TrainingRow`, kept
# generic so the mixer stays a pure resampler with no knowledge of the payload.
TPoolItem = TypeVar("TPoolItem")


@dataclass(frozen=True)
class CorpusGameRows:
    """One corpus game's stored rows, grouped for game-granular splitting.

    Positions are in play order and the five required tuples are index-aligned: position
    ``i`` was played by ``players[i]``, Pentobi chose ``actions[i]``, the game outcome from
    that side to move is ``values[i]``, and the signed final score margin from that same
    side is ``margins[i]``. ``players`` feeds the colour-conditional value-calibration
    diagnostic (D7); ``actions`` is the top-1 accuracy target; ``margins`` is the auxiliary
    score head's target (docs/plans/score-auxiliary-target.md S5), stored by both the v1 and
    v2 game schemas.

    The last two fields are **v2 only** and default to ``None``, so v1 corpora load and
    train exactly as before:

    - ``policies`` — the stored soft target per position. When present it is used
      directly and :func:`smooth_policy` is skipped: v1 had to synthesise a target from
      the played move because that was all it stored, while v2 stores Pentobi's whole
      preference distribution.
    - ``opening_unit`` — the game's **canonical ply-1 position**, the unit of the v2
      holdout split (see :func:`split_opening_units`).
    """

    game_id: int
    boards: tuple[NDArray[np.int8], ...]  # canonical compact boards (side-to-move)
    actions: tuple[int, ...]  # the action index Pentobi played per position
    players: tuple[int, ...]  # side to move per position: +1 White, -1 Black
    values: tuple[float, ...]  # outcome from the side to move: +1 / -1 / 0
    margins: tuple[float, ...]  # final score margin from the side to move (e.g. +3, -21)
    policies: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...] | None = None
    opening_unit: bytes | None = None

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
        table = pq.read_table(path, columns=["board", "action", "player", "value", "margin", "game_id"])
        boards = table.column("board").to_pylist()
        actions = table.column("action").to_pylist()
        players = table.column("player").to_pylist()
        values = table.column("value").to_pylist()
        margins = table.column("margin").to_pylist()
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
                    margins=tuple(float(m) for m in margins[rows]),
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


def load_corpus_games_v2(paths: Sequence[Path], game: BlokusDuoGame) -> list[CorpusGameRows]:
    """Read **v2** games shards into per-game row groups carrying their soft targets.

    Same cursor walk as :func:`load_corpus_games`, plus the two v2 fields: the stored
    ``(indices, values)`` policy per position, and the game's canonical ply-1 position —
    computed by replaying the first action of the shard footer's witness path, which
    every game has exactly one of.
    """
    games: list[CorpusGameRows] = []
    for path in paths:
        meta = read_game_shard_meta(path)
        table = pq.read_table(
            path,
            columns=["board", "action", "player", "value", "margin", "game_id", "policy_indices", "policy_values"],
        )
        boards = table.column("board").to_pylist()
        actions = table.column("action").to_pylist()
        players = table.column("player").to_pylist()
        values = table.column("value").to_pylist()
        margins = table.column("margin").to_pylist()
        game_ids = table.column("game_id").to_pylist()
        policy_indices = table.column("policy_indices").to_pylist()
        policy_values = table.column("policy_values").to_pylist()
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
                    margins=tuple(float(m) for m in margins[rows]),
                    policies=tuple(
                        (
                            np.frombuffer(indices, dtype=np.int32).copy(),
                            np.frombuffer(target, dtype=np.float32).copy(),
                        )
                        for indices, target in zip(policy_indices[rows], policy_values[rows], strict=True)
                    ),
                    opening_unit=opening_unit_for(game, game_meta.witness_actions),
                ),
            )
            cursor += size
    return games


def opening_unit_for(game: BlokusDuoGame, witness_actions: Sequence[int]) -> bytes | None:
    """The canonical ply-1 position a witness path passes through — the split unit.

    Every witness path goes through exactly one root child, and mirror twins collapse to
    one unit, so this is a well-defined label for any game or opening node regardless of
    the depth its start sits at. ``None`` only for the root itself.
    """
    if not witness_actions:
        return None
    board, player = game.get_next_state(game.initialise_board(), 1, int(witness_actions[0]))
    compact = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
    return canonical_key(compact)[0]


def split_opening_units(
    units: Sequence[bytes | None],
    weights: Sequence[float],
    fraction: float,
    seed: int,
) -> set[bytes]:
    """Choose the opening subtrees to hold out — the fix for a latent leak.

    v1's game-level split was valid because every v1 game had a unique random opening.
    v2 deliberately gives many games a **shared** opening, so splitting by game would put
    near-identical early positions (and literally identical opening rows) on both sides of
    the boundary and report a falsely good held-out score. The split unit is therefore the
    canonical ply-1 position, and *everything* whose witness path starts there — games and
    opening rows alike — goes to one side.

    Units are sampled **stratified by game mass** (heavy and light openings alike are
    represented) rather than uniformly, so the holdout is not accidentally all tiny
    subtrees. Deterministic in ``seed``.

    Args:
        units: One unit per item (games or rows); ``None`` items are never held out.
        weights: The mass of each item (game counts, or 1.0 per row).
        fraction: Share of the total mass to hold out.
        seed: Sampling seed.

    Returns:
        The set of held-out units.
    """
    mass: dict[bytes, float] = {}
    for unit, weight in zip(units, weights, strict=True):
        if unit is not None:
            mass[unit] = mass.get(unit, 0.0) + weight
    if not mass or fraction <= 0.0:
        return set()
    if len(mass) < 2:
        logger.warning("only {} opening unit(s): cannot hold one out without emptying training", len(mass))
        return set()

    ordered = sorted(mass, key=lambda unit: (-mass[unit], unit))
    rng = np.random.default_rng(seed)
    # Interleave a shuffle of each mass stratum, so the walk below alternates heavy and
    # light subtrees instead of consuming the biggest (or the smallest) ones first.
    stratum_size = max(1, len(ordered) // _HOLDOUT_STRATA)
    strata: list[list[bytes]] = []
    for start in range(0, len(ordered), stratum_size):
        stratum = list(ordered[start : start + stratum_size])
        rng.shuffle(stratum)
        strata.append(stratum)
    walk = [unit for row in zip_longest(*strata) for unit in row if unit is not None]

    # Take whole units until the target *mass* is reached. Selecting a fraction of the
    # units in every stratum (the previous rule) collapses at small unit counts — with
    # five strata of one unit each, ``ceil`` takes all five and the training set is
    # empty. Accumulating mass instead is exact at scale and degrades to "one unit" at
    # the granularity limit, never to "everything".
    target = fraction * sum(mass.values())
    holdout: set[bytes] = set()
    accumulated = 0.0
    for unit in walk[:-1]:  # never the whole corpus: at least one unit always trains
        if accumulated >= target:
            break
        holdout.add(unit)
        accumulated += mass[unit]
    realised = accumulated / sum(mass.values())
    if realised > 2.0 * fraction:
        logger.warning(
            "holdout is {:.1%} of mass against a {:.1%} target — only {} opening units to choose from",
            realised,
            fraction,
            len(mass),
        )
    return holdout


def partition_by_unit(
    games: Sequence[CorpusGameRows],
    holdout_units: set[bytes],
) -> tuple[list[CorpusGameRows], list[CorpusGameRows]]:
    """Split games into ``(train, holdout)`` by their opening unit."""
    train = [rows for rows in games if rows.opening_unit not in holdout_units]
    holdout = [rows for rows in games if rows.opening_unit in holdout_units]
    return train, holdout


@dataclass(frozen=True)
class LeakageReport:
    """How much of the held-out set the training set has already seen (V9's metric).

    Splitting by opening subtree keeps whole *lines* apart, but it cannot keep whole
    *positions* apart: two different openings can transpose into the same board later in
    the game, and that board then sits on both sides of the split legitimately. Every such
    position is a question the model saw the answer to before the exam, so a high rate
    means the held-out score is flattered — and that score is an input to the gate.

    ``mirror`` counts a position and its main-diagonal twin as the same position. Training
    augments every example with its twin, so a holdout position whose *mirror* is in the
    training set has leaked just as surely as one whose exact board is.
    """

    train_rows: int
    holdout_rows: int
    train_positions: int
    holdout_positions: int
    shared_positions: int
    leaked_rows: int
    shared_positions_mirror: int
    leaked_rows_mirror: int

    @property
    def leaked_fraction(self) -> float:
        """Share of held-out rows whose exact board also appears in training."""
        return self.leaked_rows / self.holdout_rows if self.holdout_rows else 0.0

    @property
    def leaked_fraction_mirror(self) -> float:
        """Share of held-out rows whose board *or its mirror* appears in training."""
        return self.leaked_rows_mirror / self.holdout_rows if self.holdout_rows else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form, for the report and the diagnostics CLI."""
        return {
            "train_rows": self.train_rows,
            "holdout_rows": self.holdout_rows,
            "train_positions": self.train_positions,
            "holdout_positions": self.holdout_positions,
            "shared_positions": self.shared_positions,
            "leaked_rows": self.leaked_rows,
            "leaked_fraction": self.leaked_fraction,
            "shared_positions_mirror": self.shared_positions_mirror,
            "leaked_rows_mirror": self.leaked_rows_mirror,
            "leaked_fraction_mirror": self.leaked_fraction_mirror,
        }


def measure_holdout_leakage(
    train_boards: Iterable[NDArray[np.int8]],
    holdout_boards: Iterable[NDArray[np.int8]],
) -> LeakageReport:
    """Count the positions the two sides of a split have in common.

    Runs on a finished corpus and a *chosen* split — the split is what it measures, so it
    cannot be computed at generation time, and it is not part of training because it
    changes nothing the model learns. It needs no engine and no GPU: two passes over the
    stored boards.

    Args:
        train_boards: Canonical compact boards on the training side.
        holdout_boards: Canonical compact boards on the held-out side.
    """
    train_exact: set[bytes] = set()
    train_mirror: set[bytes] = set()
    train_rows = 0
    for board in train_boards:
        grid = np.ascontiguousarray(board, dtype=np.int8)
        train_rows += 1
        train_exact.add(grid.tobytes())
        train_mirror.add(canonical_key(grid)[0])

    holdout_exact: set[bytes] = set()
    holdout_mirror: set[bytes] = set()
    holdout_rows = leaked_rows = leaked_rows_mirror = 0
    for board in holdout_boards:
        grid = np.ascontiguousarray(board, dtype=np.int8)
        exact = grid.tobytes()
        mirror = canonical_key(grid)[0]
        holdout_rows += 1
        holdout_exact.add(exact)
        holdout_mirror.add(mirror)
        leaked_rows += exact in train_exact
        leaked_rows_mirror += mirror in train_mirror

    return LeakageReport(
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        train_positions=len(train_exact),
        holdout_positions=len(holdout_exact),
        shared_positions=len(train_exact & holdout_exact),
        leaked_rows=leaked_rows,
        shared_positions_mirror=len(train_mirror & holdout_mirror),
        leaked_rows_mirror=leaked_rows_mirror,
    )


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """One net-ready training position plus every auxiliary head's target for it.

    The auxiliary targets travel **with** the example rather than in parallel lists,
    because :func:`mix_examples` resamples and shuffles the pool: a side list could not
    survive ``rng.choice`` in alignment, and a misalignment would train each head on
    other positions' targets while every other metric looked fine.

    ``ProcessedExample`` itself is untouched — ``example`` is exactly the
    ``(compact_board, sparse_policy, value)`` tuple the trainer has always consumed, and
    the auxiliary targets reach ``BaseNNetWrapper.train`` as separate arguments, so the
    self-play pipeline never carries one.

    Attributes:
        example: ``(compact_board, sparse_policy, value)``.
        margin: Final score margin from the side to move; ``None`` = no single margin
            (v2 opening rows). Score head, plan S5.
        ownership: ``(rows, cols)`` ``{-1, 0, +1}`` map of who holds each cell when the
            game ends, **in this position's own canonical frame** (``+1`` = the side to
            move); ``None`` = no final board. Ownership head, plan N4.
        reply: The opponent's next-ply distribution — literally the *next* position's
            policy target, so it costs one shared reference and no extra memory;
            ``None`` on each game's final position. Reply head, plan N5.
    """

    example: CorpusExample
    margin: float | None
    ownership: NDArray[np.int8] | None
    reply: tuple[NDArray[np.int32], NDArray[np.float32]] | None


def final_ownership(game: BlokusDuoGame, rows: CorpusGameRows) -> NDArray[np.int8] | None:
    """The finished board's ownership map for one corpus game, White-positive.

    Derived by **replaying the stored actions** from the game's first stored position —
    no regeneration, no engine, no extra corpus column. Corpus games start after a random
    (v1) or DAG-chosen (v2) opening prefix, so the replay begins at ``rows.boards[0]``
    rather than at the empty board, and every subsequent ply through the game's end is
    stored.

    Returns:
        A ``(rows, cols)`` int8 array: ``+1`` where **White** holds the cell at the end
        of the game, ``-1`` Black, ``0`` neither. ``None`` when the replay does not reach
        a terminal position, which would mean the stored rows are not a whole game — the
        honest answer being "no final board here", never a half-played one.

    Note:
        The stored boards are canonical (side-to-move positive) and the recorded action
        indices are **colour-free** — an action is a ``(square, piece-orientation)`` pair
        — so replaying from a canonical board with an alternating player yields the final
        grid in ``rows.players[0]``'s frame; multiplying by ``rows.players[0]`` puts it
        in the absolute White-positive frame. Callers convert to a *position's* frame by
        multiplying by that position's ``player`` (see :func:`build_training_examples`).
    """
    board = game.board_from_compact(rows.boards[0])
    player = 1  # ``boards[0]`` is canonical, so its mover is the positive side here.
    for action in rows.actions:
        # Re-derive the forced passes the v2 schema deliberately does not store. A pass
        # carries no target and no choice was made, so ``write_game_shard`` omits it and
        # ``validate_game_shard`` re-derives it the same way — a side with no placement
        # *must* pass, so it is a function of the position. Replaying the stored actions
        # with strict alternation therefore places every ply after the first skipped pass
        # for the **wrong colour**: measured on the real corpus, 16.8% of games contain
        # such a gap. Silent, because the head is then scored against its own bad labels.
        while not game.valid_move_masking(board, player)[: game.action_codec.pass_action_index].any():
            board, player = game.get_next_state(board, player, game.action_codec.pass_action_index)
        board, player = game.get_next_state(board, player, action)
    if game.get_game_ended(board, player) == 0.0:
        logger.warning(
            "game {}: replaying its {} stored actions does not reach a terminal position, so it has no "
            "final board — the ownership target is masked for every one of its rows",
            rows.game_id,
            len(rows.actions),
        )
        return None
    absolute = np.asarray(board.to_compact(), dtype=np.int8) * rows.players[0]
    return np.sign(absolute).astype(np.int8)


def build_training_examples(
    game: BlokusDuoGame,
    games: Sequence[CorpusGameRows],
    *,
    epsilon: float,
    augment: bool,
    temperature: float = 1.0,
    with_ownership: bool = False,
) -> list[TrainingRow]:
    """Turn corpus games into net-ready :class:`TrainingRow` items.

    Each position yields its smoothed-target example, and — with ``augment`` — its
    main-diagonal symmetry twin directly after it: the transposed compact board with
    the smoothed policy's support mapped through ``transpose_action``. The twin costs
    no second move generation (transposition is a bijection on legal moves), so the
    one legal-mask pass here is the whole load-time cost of a corpus.

    Position order is preserved (twins interleaved when augmenting), so with
    ``augment=False`` the output aligns index-for-index with the flattened
    ``actions``/``players``/``values`` of ``games`` — the alignment the held-out
    diagnostics rely on.

    **v2 corpora take the stored soft target instead.** When ``rows.policies`` is present
    the target is Pentobi's own distribution, softened by ``temperature`` and (only if
    ``epsilon > 0``) floored over the legal set; ``smooth_policy`` is not involved. The
    support is asserted to lie inside the position's legal moves — a violation means the
    corpus and the rules engine have desynced.

    **The auxiliary targets each follow the example's own transform.** The symmetry twin
    shares its original's *margin* (transposing a board does not change the score), takes
    the *transposed* ownership map, and takes the *twin* of the next position's policy as
    its reply target — never the original's, which would teach the head a reflected move.

    Args:
        game: The rules engine (rebuilds boards and legal masks; supplies the
            action-transpose bijection).
        games: Corpus games (typically one side of a holdout split).
        epsilon: Label-smoothing mass spread over legal moves (v1: ~0.1; v2 default 0,
            because the stored target already has support beyond the played move).
        augment: Apply the order-2 symmetry as 2× augmentation (train side yes,
            held-out side no).
        temperature: v2 target temperature τ — ``p^(1/τ)`` renormalised over the stored
            support. Confidence softening only: it is order-preserving, so it cannot fix
            a misordered target (see the v2 plan's imitation-error block).
        with_ownership: Derive each game's final board (:func:`final_ownership`) and
            attach the per-position ownership map. Off by default so the ownership head's
            one extra game replay per game is paid only by the arms that use it.
    """
    built: list[TrainingRow] = []
    for count, rows in enumerate(games, start=1):
        built.extend(_rows_for_game(game, rows, epsilon, augment, temperature, with_ownership))
        if count % _LOG_EVERY_GAMES == 0:
            logger.info("Built examples for {}/{} games ({} positions so far)", count, len(games), len(built))
    return built


def _rows_for_game(
    game: BlokusDuoGame,
    rows: CorpusGameRows,
    epsilon: float,
    augment: bool,
    temperature: float,
    with_ownership: bool,
) -> list[TrainingRow]:
    """One corpus game's :class:`TrainingRow` items, twins interleaved when augmenting.

    Built in two passes because the reply target is a *lookahead*: position ``i``'s
    target is position ``i + 1``'s policy, so every policy has to exist before any row
    can be assembled. Building per game is also what keeps the last ply of each game
    masked instead of borrowing the next game's first row.
    """
    stored = rows.policies if rows.policies is not None else (None,) * len(rows.boards)
    policies: list[tuple[NDArray[np.int32], NDArray[np.float32]]] = []
    twin_policies: list[tuple[NDArray[np.int32], NDArray[np.float32]]] = []
    for compact, action, policy in zip(rows.boards, rows.actions, stored, strict=True):
        board = game.board_from_compact(compact)
        legal = np.flatnonzero(game.valid_move_masking(board, 1)).astype(np.int32)
        if policy is None:
            indices, values = smooth_policy(action, legal, epsilon)
        else:
            indices, values = soft_target_over_legal(policy, legal, epsilon=epsilon, temperature=temperature)
        policies.append((indices, values))
        if augment:
            transposed_indices = np.array([game.transpose_action(int(a)) for a in indices], dtype=np.int32)
            twin_policies.append((transposed_indices, values))

    # At most four distinct ownership arrays per game — the final board seen from
    # either side to move, and the transpose of each for the symmetry twins — held by
    # reference from every row, so a corpus costs a pointer per row and not 196 bytes.
    ownership = final_ownership(game, rows) if with_ownership else None
    by_player: dict[int, NDArray[np.int8] | None] = {1: None, -1: None}
    by_player_twin: dict[int, NDArray[np.int8] | None] = {1: None, -1: None}
    if ownership is not None:
        for side in (1, -1):
            # ``ownership`` is White-positive; a position's frame makes its own mover
            # positive, which is exactly a multiply by that position's ``player``.
            in_frame = (ownership * side).astype(np.int8)
            by_player[side] = in_frame
            by_player_twin[side] = np.ascontiguousarray(in_frame.T)

    built: list[TrainingRow] = []
    last = len(policies) - 1

    def reply_index(index: int) -> int | None:
        """The row holding the *opponent's* answer to row ``index``, if it is stored.

        Not simply ``index + 1``: a forced pass is never stored (the v2 schema derives
        it), so when one falls between two rows the next row is the **same** side moving
        again. Measured on the real corpus, 0.86% of pairs — teaching the head that a
        player's own follow-up is their opponent's reply. Mask those rather than lie.
        """
        if index == last:
            return None
        return index + 1 if rows.players[index + 1] != rows.players[index] else None

    for index, (compact, value, margin, player) in enumerate(
        zip(rows.boards, rows.values, rows.margins, rows.players, strict=True)
    ):
        nxt = reply_index(index)
        built.append(
            TrainingRow(
                example=(compact, policies[index], value),
                margin=margin,
                ownership=by_player[player],
                reply=None if nxt is None else policies[nxt],
            )
        )
        if augment:
            built.append(
                TrainingRow(
                    example=(np.ascontiguousarray(compact.T), twin_policies[index], value),
                    margin=margin,
                    ownership=by_player_twin[player],
                    reply=None if nxt is None else twin_policies[nxt],
                )
            )
    return built


def soft_target_over_legal(
    policy: tuple[NDArray[np.int32], NDArray[np.float32]],
    legal_actions: NDArray[np.int32],
    *,
    epsilon: float = 0.0,
    temperature: float = 1.0,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Prepare a stored v2 soft target for training.

    Applies the load-time temperature, optionally floors ``epsilon`` of the mass over the
    whole legal set (default 0 — the stored target is not a one-hot, so it needs no
    smoothing to be informative), and returns the sparse pair the trainer consumes.

    Raises:
        ValueError: If the stored support contains an action the rules engine says is
            illegal — corpus/rules-engine desync, never something to paper over.
    """
    indices, values = policy
    softened = apply_target_temperature(values, temperature)
    legal_set = set(legal_actions.tolist())
    illegal = [int(a) for a in indices if int(a) not in legal_set]
    if illegal:
        raise ValueError(f"stored target support contains illegal actions {illegal[:4]} — corpus/rules desync")
    if epsilon <= 0.0:
        return indices.astype(np.int32, copy=True), softened.astype(np.float32, copy=True)
    position = {int(action): slot for slot, action in enumerate(legal_actions.tolist())}
    floored = np.full(len(legal_actions), epsilon / len(legal_actions), dtype=np.float32)
    for action, value in zip(indices.tolist(), softened.tolist(), strict=True):
        floored[position[int(action)]] += np.float32((1.0 - epsilon) * value)
    return legal_actions.astype(np.int32, copy=True), floored


def mix_examples(
    pools: Mapping[str, Sequence[TPoolItem]],
    weights: Mapping[str, float],
    *,
    seed: int,
) -> list[TPoolItem]:
    """Combine training pools into one list with the requested sampling proportions.

    Generic in the pool item so a caller that must keep per-example side values attached
    — the auxiliary heads' targets, carried on :class:`TrainingRow` — mixes them through
    this **same** resampling and shuffle rather than a parallel copy of it, which could
    not stay aligned through the ``rng.choice`` draws anyway.

    The v2 corpus has three sources with wildly different natural sizes — a game harvests
    ~26 rows while a whole opening node is one row, so openings are ~0.6% of the corpus by
    count despite being the strategic edge. ``weights`` are the **target fractions of the
    mixed output** (normalised here), so "openings at 5%" means what it says: an opening
    row must not be a 1-in-160,000 sampling event.

    **Under-represented pools are repeated; nothing is discarded.** The output is sized by
    the pool that is *largest* relative to its share (``max(len / share)``): that pool is
    used in full, and every other pool is resampled **up** to its share with replacement.
    Sizing by the smallest such ratio instead would make the tiny opening pool the binding
    constraint and trim the game rows down to match it — at the shipped 5% default that
    discards ~88% of a corpus which costs days of engine time to produce. "Openings at 5%"
    must mean "openings repeated until they are 5%", never "everything else thrown away".

    Deterministic in ``seed``.
    """
    active = {name: pool for name, pool in pools.items() if pool and weights.get(name, 0.0) > 0.0}
    if not active:
        return []
    total_weight = sum(weights[name] for name in active)
    shares = {name: weights[name] / total_weight for name in active}
    scale = max(len(pool) / shares[name] for name, pool in active.items())
    rng = np.random.default_rng(seed)
    mixed: list[TPoolItem] = []
    for name, pool in active.items():
        target = int(round(scale * shares[name]))
        replace = target > len(pool)
        picks = rng.choice(len(pool), size=target, replace=replace)
        mixed.extend(pool[int(index)] for index in picks)
    return [mixed[int(index)] for index in rng.permutation(len(mixed))]


def load_opening_examples(
    paths: Sequence[Path],
    game: BlokusDuoGame,
    *,
    value_target: str = "blend",
    blend_k: int = 5,
    temperature: float = 1.0,
    epsilon: float = 0.0,
    augment: bool = False,
) -> tuple[list[TrainingRow], list[bytes | None]]:
    """Load the v2 opening dataset as training rows plus their holdout units.

    Opening rows are the positions the whole v2 thesis rests on — depths 1–3 exist in no
    other dataset — and they are stored in their node's key frame, board and policy
    together, so they need no special handling beyond the same legality assertion the
    game rows get.

    Each row's holdout unit is found by walking ``parent_id`` up to the depth-1 ancestor,
    whose stored board *is* the canonical ply-1 key; the root row (depth 0) has no unit
    and always trains.

    Returns:
        ``(rows, units)`` — index-aligned, with the symmetry twin (when ``augment``)
        directly after its original and sharing its unit.

        **Every auxiliary target on these rows is ``None``**, and every one for a
        structural reason rather than an omission:

        - *margin*: an opening node has many games through it, so it has no single score
          margin, and the opening schema stores none — ``link`` aggregates the playouts'
          **outcomes** (``outcome_mean``, the sign of each margin) and never their
          magnitudes (docs/plans/score-auxiliary-target.md S5).
        - *ownership*: for the same reason there is no single final board.
        - *reply*: an opening row is a DAG node, not a ply of a specific game, so it has
          no "next position" whose policy could be the opponent's reply.

        Each head's loss masks them out rather than being taught an invented number.
    """
    # Resolve ancestry across *all* shards first. A node's depth-1 ancestor is very often
    # in a different shard from the node itself, and a per-shard walk simply fails to find
    # it — every such row then gets ``unit = None`` and trains unconditionally, which is
    # precisely the leak the subtree split exists to prevent. Cheap: four columns, and
    # parquet is columnar.
    ancestry_rows: dict[str, list[Any]] = {"node_id": [], "parent_id": [], "depth": [], "board": []}
    for path in paths:
        table = pq.read_table(path, columns=list(ancestry_rows))
        for name in ancestry_rows:
            ancestry_rows[name].extend(table.column(name).to_pylist())
    ancestry = _opening_units(ancestry_rows)

    built: list[TrainingRow] = []
    units: list[bytes | None] = []
    for path in paths:
        meta = read_opening_meta(path)
        table = pq.read_table(path)
        rows = {name: table.column(name).to_pylist() for name in table.column_names}
        for index in range(table.num_rows):
            compact = (
                np.frombuffer(rows["board"][index], dtype=np.dtype(meta.board_dtype)).reshape(meta.board_shape).copy()
            )
            legal = np.flatnonzero(game.valid_move_masking(game.board_from_compact(compact), 1)).astype(np.int32)
            policy = (
                np.frombuffer(rows["policy_indices"][index], dtype=np.int32).copy(),
                np.frombuffer(rows["policy_values"][index], dtype=np.float32).copy(),
            )
            indices, values = soft_target_over_legal(policy, legal, epsilon=epsilon, temperature=temperature)
            value = opening_value(
                float(rows["search_value"][index]),
                float(rows["outcome_mean"][index]),
                int(rows["outcome_count"][index]),
                target=value_target,
                blend_k=blend_k,
            )
            unit = ancestry[int(rows["node_id"][index])]
            built.append(
                TrainingRow(example=(compact, (indices, values), value), margin=None, ownership=None, reply=None)
            )
            units.append(unit)
            if augment:
                transposed = np.ascontiguousarray(compact.T)
                transposed_indices = np.array([game.transpose_action(int(a)) for a in indices], dtype=np.int32)
                built.append(
                    TrainingRow(
                        example=(transposed, (transposed_indices, values), value),
                        margin=None,
                        ownership=None,
                        reply=None,
                    )
                )
                units.append(unit)
    return built, units


def _opening_units(rows: dict[str, list[Any]]) -> dict[int, bytes | None]:
    """Map each opening node to its depth-1 ancestor's board (its holdout unit)."""
    parents = {
        int(node): (None if parent is None else int(parent))
        for node, parent in zip(rows["node_id"], rows["parent_id"], strict=True)
    }
    depths = {int(node): int(depth) for node, depth in zip(rows["node_id"], rows["depth"], strict=True)}
    boards = {int(node): bytes(board) for node, board in zip(rows["node_id"], rows["board"], strict=True)}
    units: dict[int, bytes | None] = {}
    for node_id in parents:
        cursor: int | None = node_id
        while cursor is not None and depths.get(cursor, 0) > 1:
            cursor = parents.get(cursor)
        units[node_id] = boards.get(cursor) if cursor is not None and depths.get(cursor) == 1 else None
    return units
