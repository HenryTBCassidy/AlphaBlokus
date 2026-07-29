"""Driving Pentobi for the v2 corpus: phase A mapping and phase B generation.

The whole v2 generator is one drive pattern, applied twice. At a position we
``reg_genmove`` (search *without* playing), read ``move_values`` (free: Pentobi built the
tree anyway), and then decide ourselves what to play. That is what lets us harvest the
expert's full preference distribution at a position and then play a *different* move —
the thing v1 could not do.

- **Phase A (`map_plan`)** searches the nodes an allocation plan wants mapped, recording
  each search into the store. It is search-on-demand: compute the plan, search whatever
  landed in the mapping queue, recompute, repeat until the queue is empty. The DAG grows
  exactly as far as the plan needs and no further.
- **Phase B (`play_corpus_game`)** starts from a planned node, replays its witness prefix
  into the engine with ``play`` (never re-searching plies the DAG already labels), then
  lets Pentobi play at **full strength** to the end — harvesting every ply on the way.

The engine sits behind :class:`ISearchSource` so the whole loop is testable without a
``pentobi-gtp`` binary: :class:`RandomSearchSource` plays real legal moves through the
real rules engine (it is simply not strong), matching v1's ``RandomMoveSource``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from loguru import logger

from alphablokus.games.blokusduo.pentobi.corpus import CorpusGenerationError, parse_gtp_score
from alphablokus.games.blokusduo.pentobi.corpus_v2 import SoftTarget, build_soft_target
from alphablokus.games.blokusduo.pentobi.gtp import PentobiGtp
from alphablokus.games.blokusduo.pentobi.store import (
    STORE_K,
    SearchChild,
    canonical_key,
    children_from_move_values,
    node_seed,
)
from alphablokus.games.blokusduo.pentobi.translation import PASS, PentobiMoveTranslator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.store import (
        PlanDraft,
        PlanParameters,
        PlayoutJob,
        SearchSpaceStore,
    )

# Our White (+1) is Pentobi Color(0) = GTP "b"; Black (-1) is "w" (harness mapping H2).
_GTP_COLOR = {1: "b", -1: "w"}

#: Safety bound on the plan/map loop. Each round maps one more level of the plan, and a
#: 10k-game plan is ~7 levels deep, so this only ever fires on a bug.
_MAX_MAPPING_ROUNDS = 64

#: Hard ply cap: 42 placements plus every legal interleaving of forced passes fits well
#: under this, so exceeding it means the loop lost sync with the rules engine.
_MAX_PLIES = 200


@dataclass(frozen=True)
class SearchResult:
    """One Pentobi search at one position, as the store wants it."""

    children: tuple[SearchChild, ...]  # all move_values children, as-played frame
    search_value: float | None  # the top child's value (never GTP get_value)
    seconds: float
    engine_move: int | None = None  # what reg_genmove itself chose, when the source knows

    @property
    def top_action(self) -> int | None:
        """``argmax(visits)`` — the move a full-strength continuation plays."""
        return max(self.children, key=lambda child: child.visits).action if self.children else None


class ISearchSource(Protocol):
    """Where searches come from: the real engine in production, a random mover on CI."""

    def begin_position(self, seed: int, prefix: Sequence[tuple[int, int]]) -> None:
        """Reset, reseed, and replay ``(player, action)`` pairs into the source."""
        ...

    def search(self, board: BlokusDuoBoard, player: int) -> SearchResult:
        """Search ``board`` for ``player`` **without** playing the move."""
        ...

    def advance(self, board: BlokusDuoBoard, player: int, action: int) -> None:
        """Tell the source a move was played (it may be the search's move or not)."""
        ...

    def final_white_margin(self) -> int | None:
        """The source's own final margin for White, or ``None`` if it has none."""
        ...


class PentobiSearchSource(ISearchSource):
    """A ``pentobi-gtp`` process driven ``reg_genmove`` → ``move_values`` → ``play``.

    Always ``--nobook`` (a book hit returns a move with no search tree, so
    ``move_values`` would come back empty and the ply would be unharvestable) and always
    ``--noresign`` (a resignation forfeits the score margin the value labels need).
    """

    def __init__(self, game: BlokusDuoGame, level: int, *, binary: str | None = None, threads: int = 1) -> None:
        self._translator = PentobiMoveTranslator(game)
        self._engine = PentobiGtp(level, binary=binary, threads=threads, noresign=True, nobook=True)

    def begin_position(self, seed: int, prefix: Sequence[tuple[int, int]]) -> None:
        self._engine.clear_board()
        self._engine.set_random_seed(seed)
        for player, action in prefix:
            self._play(player, action)

    def search(self, board: BlokusDuoBoard, player: int) -> SearchResult:  # noqa: ARG002 — engine tracks state
        started = time.perf_counter()
        move = self._engine.reg_genmove(_GTP_COLOR[player])
        values = self._engine.move_values()
        seconds = time.perf_counter() - started
        engine_move = self._translator.pentobi_to_action_index(move) if move else None
        return SearchResult(
            children=children_from_move_values(values, self._translator),
            search_value=values.search_value,
            seconds=seconds,
            engine_move=engine_move,
        )

    def advance(self, board: BlokusDuoBoard, player: int, action: int) -> None:  # noqa: ARG002 — engine tracks state
        self._play(player, action)

    def final_white_margin(self) -> int | None:
        return parse_gtp_score(self._engine.final_score())

    def _play(self, player: int, action: int) -> None:
        move = self._translator.action_index_to_pentobi(action)
        if move == PASS:
            return  # Pentobi has no ``play <c> pass``; a pass places nothing
        self._engine.play(_GTP_COLOR[player], move)

    def close(self) -> None:
        self._engine.close()

    def __enter__(self) -> PentobiSearchSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RandomSearchSource(ISearchSource):
    """The engine-free stand-in: real legal moves, geometric visit counts, no strength.

    Produces the *shape* Pentobi produces — a hyper-concentrated visit distribution over
    a handful of the position's legal moves — so the mapping and generation loops, the
    store, the schema and the validator all run end to end on CI.
    """

    def __init__(self, game: BlokusDuoGame, *, breadth: int = 6, seed: int = 0) -> None:
        self._game = game
        self._breadth = breadth
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def begin_position(self, seed: int, prefix: Sequence[tuple[int, int]]) -> None:
        self._rng = np.random.default_rng((self._seed, seed))

    def search(self, board: BlokusDuoBoard, player: int) -> SearchResult:
        legal = np.flatnonzero(self._game.valid_move_masking(board, player))
        chosen = legal if len(legal) <= self._breadth else self._rng.choice(legal, self._breadth, replace=False)
        children = tuple(
            SearchChild(action=int(action), visits=1000 >> index, value=0.6 - 0.01 * index)
            for index, action in enumerate(sorted(int(a) for a in chosen))
        )
        top = max(children, key=lambda child: child.visits) if children else None
        return SearchResult(
            children=children,
            search_value=top.value if top else None,
            seconds=0.0,
            engine_move=top.action if top else None,
        )

    def advance(self, board: BlokusDuoBoard, player: int, action: int) -> None:
        """Stateless — the loop hands us the board on every call."""

    def final_white_margin(self) -> int | None:
        return None


# --------------------------------------------------------------------------- #
# Phase A: mapping the search space the plan needs
# --------------------------------------------------------------------------- #


def replay_witness(
    game: BlokusDuoGame,
    actions: Sequence[int],
) -> tuple[BlokusDuoBoard, int, list[tuple[int, int]]]:
    """Replay a witness path: returns the position, the side to move, and the prefix pairs.

    The prefix pairs are ``(player, action)`` — what a source needs to put its own engine
    into the same position with ``play``.
    """
    board = game.initialise_board()
    player = 1
    prefix: list[tuple[int, int]] = []
    for action in actions:
        prefix.append((player, int(action)))
        board, player = game.get_next_state(board, player, int(action))
    return board, player, prefix


def witness_prefix(game: BlokusDuoGame, actions: Sequence[int]) -> list[tuple[int, int]]:
    """The ``(player, action)`` pairs of a witness path."""
    return replay_witness(game, actions)[2]


def search_node(store: SearchSpaceStore, source: ISearchSource, node_id: int) -> int:
    """Search one node and record it. Returns the number of edges written.

    The engine is positioned by replaying the node's **witness path** — the one as-played
    move order the store keeps for it — so the search happens at exactly the position the
    node's key frame is defined against. The seed is content-derived, so re-searching a
    position after a crash reproduces the same tree.
    """
    record = store.node(node_id)
    board, player = store.board_at(node_id)
    source.begin_position(node_seed(record.board_key), witness_prefix(store.game, record.witness_actions))
    result = source.search(board, player)
    if not result.children:
        # No search tree: a forced pass or a terminal position. Recorded as searched with
        # no children so the allocator treats it as a playout start and never retries it.
        logger.debug("node {} (depth {}) returned an empty move_values — recording as a leaf", node_id, record.depth)
    return store.record_search(node_id, result.children, seconds=result.seconds, search_value=result.search_value)


def map_plan(
    store: SearchSpaceStore,
    source: ISearchSource,
    params: PlanParameters,
    *,
    max_rounds: int = _MAX_MAPPING_ROUNDS,
) -> PlanDraft:
    """Compute the plan, search what it needs, repeat until nothing is missing.

    This is phase A. Each round deepens the plan by one level: the allocator hands back
    the nodes it wants to split but cannot (they are unsearched), we search exactly
    those, and recompute. The result is a plan with **zero mapping debt** — every node it
    allocates games to is either searched or a deliberate playout start.

    Mapping is cheap relative to the games it directs (the measured stage-1 shape is
    ~1,600 searches ≈ 1 box-hour against ~58 box-hours of games), so there is no
    tree-vs-games budget tension to manage here.
    """
    searched = 0
    for round_index in range(max_rounds):
        draft = store.compute_plan(params)
        if not draft.mapping_queue:
            searched += _search_book_positions(store, source)
            logger.info(
                "Plan mapped in {} rounds: {} nodes searched, {} openings, {} games planned",
                round_index,
                searched,
                len(draft.starts),
                draft.planned_games,
            )
            return draft
        logger.info(
            "Mapping round {}: searching {} nodes ({} allocated so far)",
            round_index,
            len(draft.mapping_queue),
            len(draft.allocations),
        )
        for node_id in draft.mapping_queue:
            search_node(store, source, node_id)
            searched += 1
    raise CorpusGenerationError(f"plan mapping did not converge in {max_rounds} rounds")


def _search_book_positions(store: SearchSpaceStore, source: ISearchSource) -> int:
    """Search every position along a book line that the allocation did not reach.

    The allocator searches a node only when it needs that node's move list in order to
    divide games between its children. Book lines are inserted whole and handed a fixed
    number of games at their end, so nothing along them ever needs dividing — and a
    position nobody searched carries no target, so it produces no training data at all.

    That silently discards the most valuable positions available to us: the book is the
    Pentobi author's hand-curated set of strongest openings, and learning openings is the
    entire point of v2. So once the plan has converged, search whatever the book
    contributed and the allocation missed — a few dozen extra searches against a run
    measured in days.

    Returns:
        The number of positions searched.
    """
    pending = [record for record in store.nodes(status="pending") if record.source == "book"]
    if not pending:
        return 0
    logger.info("Searching {} book-line positions the allocation did not need", len(pending))
    for record in pending:
        search_node(store, source, record.node_id)
    return len(pending)


# --------------------------------------------------------------------------- #
# Phase B: playing planned games out and harvesting every ply
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HarvestedPly:
    """One harvested ply: the position, the expert's whole preference, and its choice."""

    ply: int  # 0-based ply index in the full game (the witness prefix counts)
    player: int  # side to move: +1 White, -1 Black
    action: int  # the move actually played
    top_action: int  # argmax(visits) — equal to ``action`` on a full-strength continuation
    compact_board: NDArray[np.int8]  # canonical (side-to-move) 14x14 placement grid
    target: SoftTarget  # the soft policy target, as-played frame
    search_value: float  # the top child's value for the side to move


@dataclass(frozen=True)
class HarvestedGame:
    """One finished v2 game: its identity in the DAG, its plies, and its result."""

    game_id: int
    node_id: int
    board_key: bytes
    replica: int
    engine_seed: int
    witness_actions: tuple[int, ...]
    plies: tuple[HarvestedPly, ...]
    white_score: int
    black_score: int
    #: Every action played after the prefix, **including forced passes**. Passes are not
    #: harvested (no move to rate, no choice made) so they have no entry in ``plies``,
    #: which would make a replay of ``plies`` alone desync from the real game.
    played_actions: tuple[int, ...] = ()

    @property
    def white_margin(self) -> int:
        """Final score margin for White (positive = White won)."""
        return self.white_score - self.black_score

    @property
    def actions(self) -> tuple[int, ...]:
        """The complete game: the witness prefix plus every action played after it.

        Includes the forced passes, so replaying this sequence through the rules engine
        reproduces the game exactly. ``plies`` covers only the *harvested* positions and
        skips passes, so the two sequences are deliberately different lengths.
        """
        return self.witness_actions + self.played_actions


def play_planned_game(
    game: BlokusDuoGame,
    source: ISearchSource,
    job: PlayoutJob,
    *,
    top_k: int = STORE_K,
) -> HarvestedGame:
    """Play one planned game from its start node and harvest **every** ply.

    The witness prefix is *replayed*, not searched: those plies' targets already live in
    the DAG as opening rows, so re-searching them would be pure waste. From the start
    position on, every ply is ``reg_genmove`` → ``move_values`` → harvest → play the
    engine's own ``argmax(visits)``.

    **Continuations are full strength — no temperature, no move sampling.** Temperature
    would weaken the very play being distilled, and deliberate deviation belongs in the
    allocation (where it gets a plan entry and a harvested label), not in an unlabelled
    in-game coin flip. Per-game seed variation supplies the continuation diversity.

    Desync guards, all non-negotiable: the replayed prefix must be legal and must land on
    the start node's own position; every engine move is legality-checked against our
    rules engine; the engine's ``final_score`` is cross-checked against
    ``BlokusDuoGame.final_scores``; and a hard ply cap catches a runaway loop.

    Raises:
        CorpusGenerationError: On any of those desyncs.
    """
    board = game.initialise_board()
    player = 1
    for index, action in enumerate(job.witness_actions):
        if not game.valid_move_masking(board, player)[action]:
            raise CorpusGenerationError(f"game {job.game_id} witness ply {index}: action {action} is illegal here")
        board, player = game.get_next_state(board, player, action)
    compact = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
    if canonical_key(compact)[0] != job.board_key:
        raise CorpusGenerationError(f"game {job.game_id}: the replayed witness path does not reach its start node")

    source.begin_position(job.engine_seed, witness_prefix(game, job.witness_actions))
    plies: list[HarvestedPly] = []
    played: list[int] = []
    ply = len(job.witness_actions)
    while game.get_game_ended(board, player) == 0:
        if ply >= _MAX_PLIES:
            raise CorpusGenerationError(f"game {job.game_id} exceeded {_MAX_PLIES} plies — loop desync")
        mask = game.valid_move_masking(board, player)
        if not mask[: game.action_codec.pass_action_index].any():
            # Forced pass. Decide this from *our* rules engine and skip the search
            # entirely: Pentobi's ``reg_genmove`` raises "player failed to generate a
            # move" when the side to move has none (``GtpEngine::cmd_reg_genmove``
            # rejects a null move), so asking first aborts the whole run. (v1 never hit
            # this because ``genmove`` returns the string "pass" instead of raising —
            # the two commands differ, and only the real binary shows it.) There is
            # nothing to harvest at a position with one legal move anyway, and Pentobi
            # has no ``play <c> pass``, so we just advance our own board.
            played.append(game.action_codec.pass_action_index)
            board, player = game.get_next_state(board, player, game.action_codec.pass_action_index)
            ply += 1
            continue
        result = source.search(board, player)
        if not result.children:
            raise CorpusGenerationError(f"game {job.game_id} ply {ply}: empty move_values but moves are available")
        target = build_soft_target(result.children, top_k)
        action = target.top_action
        if not mask[action]:
            raise CorpusGenerationError(f"game {job.game_id} ply {ply}: engine move {action} is illegal here")
        plies.append(
            HarvestedPly(
                ply=ply,
                player=player,
                action=action,
                top_action=action,
                compact_board=np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8),
                target=target,
                search_value=float(result.search_value if result.search_value is not None else 0.0),
            ),
        )
        # Relay *before* advancing: ``get_next_state`` flips ``player`` to the opponent,
        # and the source needs the colour that actually played the move (v1's
        # ``play_corpus_game`` orders these the same way). Relaying the flipped colour
        # desyncs the engine's board from ours on the very first continuation ply.
        played.append(action)
        source.advance(board, player, action)
        board, player = game.get_next_state(board, player, action)
        ply += 1

    white_score, black_score = game.final_scores(board)
    source_margin = source.final_white_margin()
    if source_margin is not None and source_margin != white_score - black_score:
        raise CorpusGenerationError(
            f"game {job.game_id}: engine margin {source_margin} != rules-engine margin {white_score - black_score}",
        )
    return HarvestedGame(
        game_id=job.game_id,
        node_id=job.node_id,
        board_key=job.board_key,
        replica=job.replica,
        engine_seed=job.engine_seed,
        witness_actions=job.witness_actions,
        plies=tuple(plies),
        white_score=white_score,
        black_score=black_score,
        played_actions=tuple(played),
    )


# --------------------------------------------------------------------------- #
# The opening book
# --------------------------------------------------------------------------- #

#: SGF tokens we care about in a ``.blksgf`` book: branch delimiters and moves.
_SGF_TOKEN = re.compile(r"\(|\)|;([BW])\[([^\]]*)\]")


def read_book_lines(path: Path, game: BlokusDuoGame) -> list[tuple[int, ...]]:
    """Parse Pentobi's Duo opening book into root-to-leaf lines of action indices.

    The book is an SGF game tree: ``(`` opens a variation, ``;B[cells]`` / ``;W[cells]``
    are moves, and every leaf is one opening line. **Book ``B`` is our White** — the SGF
    follows Go's convention where the colour listed first is the first player, and our
    White is the first mover.

    Lines are legality-checked against the rules engine as they are read; an illegal move
    means the book and our rules disagree, which must not pass silently.
    """
    translator = PentobiMoveTranslator(game)
    lines: list[tuple[int, ...]] = []
    path_actions: list[int] = []
    stack: list[tuple[int, bool]] = []
    for match in _SGF_TOKEN.finditer(path.read_text()):
        token = match.group(0)
        if token == "(":
            if stack:
                stack[-1] = (stack[-1][0], True)
            stack.append((len(path_actions), False))
        elif token == ")":
            entry_length, has_child = stack.pop()
            if not has_child:
                lines.append(tuple(path_actions))
            del path_actions[entry_length:]
        else:
            path_actions.append(translator.pentobi_to_action_index(match.group(2)))
    return [line for line in lines if _is_playable(game, line)]


def _is_playable(game: BlokusDuoGame, actions: Sequence[int]) -> bool:
    """Replay a book line through the rules engine; raise if it is not legal."""
    board = game.initialise_board()
    player = 1
    for index, action in enumerate(actions):
        if not game.valid_move_masking(board, player)[action]:
            raise CorpusGenerationError(f"book line ply {index}: action {action} is illegal in our rules engine")
        board, player = game.get_next_state(board, player, action)
    return bool(actions)
