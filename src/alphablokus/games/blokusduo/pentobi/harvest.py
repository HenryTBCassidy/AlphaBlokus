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
from alphablokus.games.blokusduo.pentobi.gtp import PentobiGtp
from alphablokus.games.blokusduo.pentobi.store import SearchChild, children_from_move_values, node_seed
from alphablokus.games.blokusduo.pentobi.translation import PASS, PentobiMoveTranslator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from alphablokus.games.blokusduo.board import BlokusDuoBoard
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.store import PlanDraft, PlanParameters, SearchSpaceStore

# Our White (+1) is Pentobi Color(0) = GTP "b"; Black (-1) is "w" (harness mapping H2).
_GTP_COLOR = {1: "b", -1: "w"}

#: Safety bound on the plan/map loop. Each round maps one more level of the plan, and a
#: 10k-game plan is ~7 levels deep, so this only ever fires on a bug.
_MAX_MAPPING_ROUNDS = 64


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


def witness_prefix(game: BlokusDuoGame, actions: Sequence[int]) -> list[tuple[int, int]]:
    """Turn a witness path into ``(player, action)`` pairs by replaying it."""
    board = game.initialise_board()
    player = 1
    prefix: list[tuple[int, int]] = []
    for action in actions:
        prefix.append((player, int(action)))
        board, player = game.get_next_state(board, player, int(action))
    return prefix


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
