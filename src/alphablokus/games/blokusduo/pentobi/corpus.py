"""Pentobi distillation corpus: expert-game generation, parquet shards, diversity metrics.

The training curriculum fix for the L4 plateau is to distil Pentobi's play into the net
(AlphaGo-2016 style: supervised move prediction from an expert corpus, then RL on top).
This module is the *data* half: it plays Pentobi-vs-Pentobi games at one level (L9 for
the production corpus), harvests one training example per Pentobi ply, and persists them
as parquet shards a supervised trainer can consume directly.

**What one example is.** For the position *before* each Pentobi ply we store the
canonical compact board (side-to-move perspective, the same int8 14x14 placement grid the
self-play buffer holds), the played action as a **one-hot sparse policy** (behavioural
cloning target; label smoothing is a training-time concern), the **game outcome from the
side to move** (+1 win / -1 loss / 0 draw), the signed **final score margin** from the
side to move, and the side to move itself. Random-opening-prefix plies are *never*
harvested — they diversify the start position but are not expert moves.

**Diversity.** Pentobi at a fixed level is near-deterministic, so a naive corpus is a
pile of near-identical games. Two mechanisms, both first-class here: per-game engine
seeds (``set_random_seed``) and a uniformly random opening prefix of ``k`` plies played
by both sides before Pentobi takes over. :func:`compute_diversity` quantifies the result
(unique games / opening prefixes / positions) so a corpus can be *proven* diverse rather
than assumed to be.

**Schema compatibility.** Shards carry the exact ``board``/``policy_indices``/
``policy_values``/``value`` columns and ``board_kind``/``policy_kind`` markers of
:class:`alphablokus.storage.selfplay_store.SelfPlayStore` (asserted equal in tests), plus
corpus-only columns (``margin``, ``player``, ``game_id``, ``ply``, ``action``) and
footer metadata. See ``docs/plans/pentobi-distillation.md`` for the schema table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alphablokus.games.blokusduo.pentobi.gtp import PentobiGtp
from alphablokus.games.blokusduo.pentobi.translation import PASS, PentobiMoveTranslator
from alphablokus.interfaces import RESIGN_ACTION

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame

# One training example: (compact canonical board, sparse one-hot policy, value).
# Structurally identical to ``alphablokus.selfplay.episode.ProcessedExample`` — kept as a
# local alias so game-layer code does not import the framework's self-play module.
CorpusExample: TypeAlias = "tuple[NDArray[np.int8], tuple[NDArray[np.int32], NDArray[np.float32]], float]"

# Storage format markers. Deliberately the same values as ``SelfPlayStore.BOARD_KIND`` /
# ``POLICY_KIND`` (tests assert equality) so trainers can share row-decoding code, plus a
# corpus-specific marker identifying the extended schema.
BOARD_KIND = "compact_v1"
POLICY_KIND = "sparse_v1"
DATASET_KIND = "pentobi_distill_v1"

# Our White (+1) maps to Pentobi Color(0) = GTP "b"; Black (-1) to "w" (pinned in the
# pentobi-harness plan H2 and used throughout the benchmark player).
_GTP_COLOR = {1: "b", -1: "w"}

# Hard ply cap: 42 placements + every legal interleaving of forced passes fits well
# under this; exceeding it means the loop lost sync with the rules engine.
_MAX_PLIES = 200


class CorpusGenerationError(RuntimeError):
    """Raised when a generated game desyncs from our rules engine (illegal move,
    resign despite ``--noresign``, score mismatch, or a runaway game loop)."""


# --------------------------------------------------------------------------- #
# Move sources
# --------------------------------------------------------------------------- #


class IMoveSource(Protocol):
    """Where expert moves come from during corpus generation.

    The production implementation is :class:`PentobiMoveSource`; tests drive the same
    game loop with a real uniform-random source (no engine binary needed on CI).
    """

    def begin_game(self, seed: int) -> None:
        """Reset for a new game and reseed the source's RNG."""
        ...

    def observe(self, board: BlokusDuoBoard, player: int, action: int) -> None:
        """Inform the source of a move it did not choose (a random opening ply)."""
        ...

    def choose(self, board: BlokusDuoBoard, player: int) -> int:
        """Return the source's move for ``player`` as an action index."""
        ...

    def final_white_margin(self) -> int | None:
        """Source's own view of the final margin for White, or ``None`` if it has none.

        Used as a desync guard: when a value is returned it must match the margin our
        rules engine computes from the final board.
        """
        ...


class PentobiMoveSource(IMoveSource):
    """A single ``pentobi-gtp`` engine playing *both* colours of a corpus game.

    One engine process generates both sides' moves (``genmove b`` / ``genmove w``
    alternately), halving process count vs one engine per colour. The engine is reused
    across games — :meth:`begin_game` clears the board and reseeds, exactly the
    independence scheme validated for :class:`~alphablokus.games.blokusduo.pentobi.player.PentobiPlayer`.
    Constructed with ``--noresign`` so every game reaches its true final position (a
    resignation would forfeit the score margin the value labels need).
    """

    def __init__(
        self,
        game: BlokusDuoGame,
        level: int,
        *,
        binary: str | None = None,
        threads: int = 1,
    ) -> None:
        self._translator = PentobiMoveTranslator(game)
        self._engine = PentobiGtp(level, binary=binary, threads=threads, noresign=True)

    def begin_game(self, seed: int) -> None:
        self._engine.clear_board()
        self._engine.set_random_seed(seed)

    def observe(self, board: BlokusDuoBoard, player: int, action: int) -> None:  # noqa: ARG002 — engine tracks state
        move = self._translator.action_index_to_pentobi(action)
        if move == PASS:  # pragma: no cover — opening prefixes never pass
            return  # Pentobi has no ``play <c> pass``; a pass places nothing, boards stay in sync
        self._engine.play(_GTP_COLOR[player], move)

    def choose(self, board: BlokusDuoBoard, player: int) -> int:  # noqa: ARG002 — engine tracks state
        action = self._translator.pentobi_to_action_index(self._engine.genmove(_GTP_COLOR[player]))
        if action == RESIGN_ACTION:
            raise CorpusGenerationError("engine resigned despite --noresign")
        return action

    def final_white_margin(self) -> int | None:
        return parse_gtp_score(self._engine.final_score())

    def close(self) -> None:
        self._engine.close()

    def __enter__(self) -> PentobiMoveSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RandomMoveSource(IMoveSource):
    """Uniform-random legal mover — the engine-free source used by the test suite.

    A real object playing real legal moves through the real rules engine (per the
    no-mocks-for-game-logic convention); it simply isn't strong. Lets every part of the
    loop/harvest/persist/validate pipeline run without a ``pentobi-gtp`` binary.
    """

    def __init__(self, game: BlokusDuoGame) -> None:
        self._game = game
        self._rng = np.random.default_rng(0)

    def begin_game(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def observe(self, board: BlokusDuoBoard, player: int, action: int) -> None:
        """Stateless — nothing to track (the loop hands us the board on ``choose``)."""

    def choose(self, board: BlokusDuoBoard, player: int) -> int:
        return sample_random_action(self._game, board, player, self._rng)

    def final_white_margin(self) -> int | None:
        return None


def sample_random_action(
    game: BlokusDuoGame,
    board: BlokusDuoBoard,
    player: int,
    rng: np.random.Generator,
) -> int:
    """Sample uniformly among legal *placement* actions (pass only when forced)."""
    mask = game.valid_move_masking(board, player)
    legal = np.flatnonzero(mask)
    placements = legal[legal != game.action_codec.pass_action_index]
    if placements.size == 0:
        return game.action_codec.pass_action_index
    return int(rng.choice(placements))


def parse_gtp_score(score: str) -> int:
    """Parse a GTP ``final_score`` string into a margin for our White.

    ``B+N`` is a win for GTP black — which is *our White* (harness mapping) — so it
    parses to ``+N``; ``W+N`` to ``-N``; ``0`` (a draw) to ``0``.
    """
    s = score.strip()
    if s == "0":
        return 0
    if len(s) > 2 and s[1] == "+" and s[0] in "BW":
        margin = int(float(s[2:]))
        return margin if s[0] == "B" else -margin
    raise CorpusGenerationError(f"unparseable GTP final_score {score!r}")


# --------------------------------------------------------------------------- #
# Game loop + harvesting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HarvestedPly:
    """One expert ply: the position it was played from and the move itself."""

    ply: int  # 0-based ply index within the full game (opening plies count)
    player: int  # side to move: +1 White, -1 Black
    action: int  # action index Pentobi played (may be the pass action)
    compact_board: NDArray[np.int8]  # canonical (side-to-move) 14x14 placement grid


@dataclass(frozen=True)
class CorpusGame:
    """One finished corpus game: harvested expert plies + labels + provenance."""

    game_id: int
    pentobi_seed: int
    opening_actions: tuple[int, ...]  # the k random prefix plies (not harvested)
    plies: tuple[HarvestedPly, ...]  # expert plies only, in play order
    white_score: int
    black_score: int

    @property
    def white_margin(self) -> int:
        """Final score margin for White (positive = White won)."""
        return self.white_score - self.black_score

    @property
    def actions(self) -> tuple[int, ...]:
        """The complete game as action indices: opening prefix + expert plies."""
        return self.opening_actions + tuple(p.action for p in self.plies)


def play_corpus_game(
    game: BlokusDuoGame,
    source: IMoveSource,
    *,
    game_id: int,
    pentobi_seed: int,
    opening_random_plies: int,
    opening_rng: np.random.Generator,
) -> CorpusGame:
    """Play one corpus game and harvest an example per expert ply.

    The first ``opening_random_plies`` plies are sampled uniformly among legal
    placements (diversifying the start position) and relayed to the source via
    ``observe``; every subsequent ply is the source's own move, validated against our
    rules engine before being recorded. The final margin is cross-checked against the
    source's view when it has one.

    Raises:
        CorpusGenerationError: On any desync — an illegal source move, a score
            disagreement, or a game exceeding the hard ply cap.
    """
    source.begin_game(pentobi_seed)
    board = game.initialise_board()
    player = 1

    opening_actions: list[int] = []
    for _ in range(opening_random_plies):
        action = sample_random_action(game, board, player, opening_rng)
        source.observe(board, player, action)
        opening_actions.append(action)
        board, player = game.get_next_state(board, player, action)

    plies: list[HarvestedPly] = []
    ply = len(opening_actions)
    while game.get_game_ended(board, player) == 0:
        if ply >= _MAX_PLIES:
            raise CorpusGenerationError(f"game {game_id} exceeded {_MAX_PLIES} plies — loop desync")
        action = source.choose(board, player)
        mask = game.valid_move_masking(board, player)
        if not mask[action]:
            raise CorpusGenerationError(f"game {game_id} ply {ply}: source move {action} is illegal here")
        compact = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
        plies.append(HarvestedPly(ply=ply, player=player, action=action, compact_board=compact))
        board, player = game.get_next_state(board, player, action)
        ply += 1

    white_score, black_score = game.final_scores(board)
    source_margin = source.final_white_margin()
    if source_margin is not None and source_margin != white_score - black_score:
        raise CorpusGenerationError(
            f"game {game_id}: engine margin {source_margin} != rules-engine margin {white_score - black_score}",
        )
    return CorpusGame(
        game_id=game_id,
        pentobi_seed=pentobi_seed,
        opening_actions=tuple(opening_actions),
        plies=tuple(plies),
        white_score=white_score,
        black_score=black_score,
    )


# --------------------------------------------------------------------------- #
# Parquet shards
# --------------------------------------------------------------------------- #


def shard_filename(index: int) -> str:
    """Canonical shard filename for shard ``index``."""
    return f"corpus_{index:05d}.parquet"


def write_shard(
    path: Path,
    games: Sequence[CorpusGame],
    *,
    policy_size: int,
    level: int,
    opening_random_plies: int,
) -> int:
    """Write one shard of finished games to ``path`` atomically; returns rows written.

    Written to ``<path>.tmp`` then renamed, so a killed run never leaves a torn shard —
    resume logic can trust any file matching the final name.
    """
    sample = games[0].plies[0].compact_board
    games_meta = [
        {
            "game_id": g.game_id,
            "pentobi_seed": g.pentobi_seed,
            "opening_actions": list(g.opening_actions),
            "white_score": g.white_score,
            "black_score": g.black_score,
        }
        for g in games
    ]
    metadata = {
        "dataset_kind": DATASET_KIND,
        "board_kind": BOARD_KIND,
        "board_shape": ",".join(str(d) for d in sample.shape),
        "board_dtype": str(sample.dtype),
        "policy_kind": POLICY_KIND,
        "policy_size": str(policy_size),
        "level": str(level),
        "opening_random_plies": str(opening_random_plies),
        "game_sizes": ",".join(str(len(g.plies)) for g in games),
        "games_meta": json.dumps(games_meta),
    }
    schema = pa.schema(
        [
            pa.field("board", pa.binary()),
            pa.field("policy_indices", pa.binary()),
            pa.field("policy_values", pa.binary()),
            pa.field("value", pa.float64()),
            pa.field("margin", pa.int32()),
            pa.field("player", pa.int8()),
            pa.field("game_id", pa.int64()),
            pa.field("ply", pa.int32()),
            pa.field("action", pa.int32()),
        ],
        metadata={k.encode(): v.encode() for k, v in metadata.items()},
    )

    columns: dict[str, list[object]] = {name: [] for name in schema.names}
    for g in games:
        winner = int(np.sign(g.white_margin))  # +1 White won, -1 Black won, 0 draw
        for p in g.plies:
            columns["board"].append(p.compact_board.tobytes())
            columns["policy_indices"].append(np.array([p.action], dtype=np.int32).tobytes())
            columns["policy_values"].append(np.array([1.0], dtype=np.float32).tobytes())
            columns["value"].append(float(winner * p.player))
            columns["margin"].append(g.white_margin * p.player)
            columns["player"].append(p.player)
            columns["game_id"].append(g.game_id)
            columns["ply"].append(p.ply)
            columns["action"].append(p.action)

    table = pa.Table.from_pydict(columns, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)
    return table.num_rows


@dataclass(frozen=True)
class ShardGameMeta:
    """Per-game provenance stored in a shard's footer metadata."""

    game_id: int
    pentobi_seed: int
    opening_actions: tuple[int, ...]
    white_score: int
    black_score: int


@dataclass(frozen=True)
class ShardMeta:
    """A shard's footer metadata, decoded."""

    level: int
    opening_random_plies: int
    policy_size: int
    board_shape: tuple[int, ...]
    board_dtype: str
    game_sizes: tuple[int, ...]
    games: tuple[ShardGameMeta, ...]


def corpus_shards(directory: Path) -> list[Path]:
    """All final (non-``.tmp``) shard files in ``directory``, sorted by index."""
    return sorted(directory.glob("corpus_*.parquet"))


def read_shard_meta(path: Path) -> ShardMeta:
    """Decode a shard's footer metadata (reads only the parquet footer)."""
    raw = pq.read_schema(path).metadata or {}
    meta = {k.decode(): v.decode() for k, v in raw.items()}
    if meta.get("dataset_kind") != DATASET_KIND:
        raise ValueError(f"{path.name}: dataset_kind={meta.get('dataset_kind')!r}, expected {DATASET_KIND!r}")
    games = tuple(
        ShardGameMeta(
            game_id=int(g["game_id"]),
            pentobi_seed=int(g["pentobi_seed"]),
            opening_actions=tuple(int(a) for a in g["opening_actions"]),
            white_score=int(g["white_score"]),
            black_score=int(g["black_score"]),
        )
        for g in json.loads(meta["games_meta"])
    )
    return ShardMeta(
        level=int(meta["level"]),
        opening_random_plies=int(meta["opening_random_plies"]),
        policy_size=int(meta["policy_size"]),
        board_shape=tuple(int(d) for d in meta["board_shape"].split(",")),
        board_dtype=meta["board_dtype"],
        game_sizes=tuple(int(s) for s in meta["game_sizes"].split(",")),
        games=games,
    )


def iter_corpus_examples(paths: Sequence[Path]) -> Iterator[CorpusExample]:
    """Stream ``(board, (indices, values), value)`` training tuples from shards.

    The tuple is structurally identical to the self-play pipeline's
    ``ProcessedExample`` — a trainer can consume either source through the same code.
    """
    for path in paths:
        meta = read_shard_meta(path)
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["board", "policy_indices", "policy_values", "value"]):
            for board_bytes, indices_bytes, values_bytes, value in zip(
                batch.column("board").to_pylist(),
                batch.column("policy_indices").to_pylist(),
                batch.column("policy_values").to_pylist(),
                batch.column("value").to_pylist(),
                strict=True,
            ):
                board = np.frombuffer(board_bytes, dtype=np.dtype(meta.board_dtype)).reshape(meta.board_shape).copy()
                indices = np.frombuffer(indices_bytes, dtype=np.int32).copy()
                values = np.frombuffer(values_bytes, dtype=np.float32).copy()
                yield board, (indices, values), float(value)


# --------------------------------------------------------------------------- #
# Diversity measurement
# --------------------------------------------------------------------------- #

#: Opening-prefix depths at which distinct-prefix counts are reported.
PREFIX_LENGTHS = (1, 2, 4, 6, 8)


@dataclass(frozen=True)
class DiversityReport:
    """Corpus diversity metrics — the proof the corpus is not a pile of clones."""

    num_games: int
    num_positions: int
    unique_games: int  # distinct full action sequences
    unique_positions: int  # distinct stored (canonical board, side-to-move) grids
    unique_openings_by_prefix: dict[int, int]  # prefix length -> distinct prefixes

    @property
    def unique_game_fraction(self) -> float:
        """Fraction of games whose full move sequence is unique in the corpus."""
        return self.unique_games / self.num_games if self.num_games else 0.0

    @property
    def unique_position_fraction(self) -> float:
        """Fraction of stored positions that are distinct across the corpus."""
        return self.unique_positions / self.num_positions if self.num_positions else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form (for logs and the analysis CLI)."""
        return {
            "num_games": self.num_games,
            "num_positions": self.num_positions,
            "unique_games": self.unique_games,
            "unique_game_fraction": self.unique_game_fraction,
            "unique_positions": self.unique_positions,
            "unique_position_fraction": self.unique_position_fraction,
            "unique_openings_by_prefix": {str(k): v for k, v in self.unique_openings_by_prefix.items()},
        }


def compute_diversity(
    sequences: Sequence[tuple[int, ...]],
    position_keys: Sequence[bytes],
    prefix_lengths: Sequence[int] = PREFIX_LENGTHS,
) -> DiversityReport:
    """Compute diversity metrics from full game sequences + stored position keys.

    Args:
        sequences: One full action-index sequence per game (opening + expert plies).
        position_keys: One hashable key per *stored* position (canonical board bytes).
        prefix_lengths: Opening depths at which to count distinct prefixes.
    """
    return DiversityReport(
        num_games=len(sequences),
        num_positions=len(position_keys),
        unique_games=len(set(sequences)),
        unique_positions=len(set(position_keys)),
        unique_openings_by_prefix={k: len({s[:k] for s in sequences}) for k in prefix_lengths},
    )


def analyze_corpus(directory: Path) -> DiversityReport:
    """Read every shard in ``directory`` and compute the corpus diversity report."""
    sequences: list[tuple[int, ...]] = []
    position_keys: list[bytes] = []
    for path in corpus_shards(directory):
        meta = read_shard_meta(path)
        table = pq.read_table(path, columns=["board", "action"])
        boards = table.column("board").to_pylist()
        actions = table.column("action").to_pylist()
        position_keys.extend(boards)
        cursor = 0
        for g, size in zip(meta.games, meta.game_sizes, strict=True):
            expert_actions = tuple(int(a) for a in actions[cursor : cursor + size])
            sequences.append(g.opening_actions + expert_actions)
            cursor += size
    return compute_diversity(sequences, position_keys)


# --------------------------------------------------------------------------- #
# Validation (pilot correctness checks)
# --------------------------------------------------------------------------- #


def validate_shard(path: Path, game: BlokusDuoGame) -> int:
    """Replay every game in a shard through the rules engine and check every row.

    For each game: replays opening + expert actions move by move, asserting that each
    stored row's board equals the canonical compact board at that ply, its player is
    the true side to move, and its action is legal there; then checks the terminal
    board's scores against the stored labels (value sign, margin, side-to-move
    perspective). Returns the number of positions checked; raises on any mismatch.
    """
    meta = read_shard_meta(path)
    table = pq.read_table(path)
    rows = {name: table.column(name).to_pylist() for name in table.column_names}
    cursor = 0
    checked = 0
    for g, size in zip(meta.games, meta.game_sizes, strict=True):
        board = game.initialise_board()
        player = 1
        for action in g.opening_actions:
            _require(bool(game.valid_move_masking(board, player)[action]), path, g.game_id, "illegal opening action")
            board, player = game.get_next_state(board, player, action)
        white_margin = g.white_score - g.black_score
        for i in range(cursor, cursor + size):
            action = int(rows["action"][i])
            _require(int(rows["player"][i]) == player, path, g.game_id, f"row {i}: wrong side-to-move")
            _require(bool(game.valid_move_masking(board, player)[action]), path, g.game_id, f"row {i}: illegal action")
            expected = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
            _require(rows["board"][i] == expected.tobytes(), path, g.game_id, f"row {i}: board mismatch")
            indices = np.frombuffer(rows["policy_indices"][i], dtype=np.int32)
            values = np.frombuffer(rows["policy_values"][i], dtype=np.float32)
            one_hot = indices.tolist() == [action] and values.tolist() == [1.0]
            _require(one_hot, path, g.game_id, f"row {i}: policy is not one-hot of the played action")
            _require(int(rows["margin"][i]) == white_margin * player, path, g.game_id, f"row {i}: margin mismatch")
            _require(
                float(rows["value"][i]) == float(np.sign(white_margin) * player),
                path,
                g.game_id,
                f"row {i}: value mismatch",
            )
            board, player = game.get_next_state(board, player, action)
            checked += 1
        cursor += size
        _require(game.get_game_ended(board, player) != 0, path, g.game_id, "replayed game is not terminal")
        _require(
            game.final_scores(board) == (g.white_score, g.black_score),
            path,
            g.game_id,
            "final scores do not match stored labels",
        )
    return checked


def _require(condition: bool, path: Path, game_id: int, message: str) -> None:
    """Raise a uniform validation error when ``condition`` fails."""
    if not condition:
        raise CorpusGenerationError(f"{path.name} game {game_id}: {message}")
