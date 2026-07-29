"""Tests for the v2 engine drive patterns: phase A mapping and the book parser (V4).

Driven by :class:`RandomSearchSource` — real legal moves through the real rules engine,
no ``pentobi-gtp`` binary — so the whole plan/map loop runs on CI exactly as it will on
the box, minus the engine's strength.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.corpus import CorpusGenerationError
from alphablokus.games.blokusduo.pentobi.gtp import GtpError
from alphablokus.games.blokusduo.pentobi.harvest import (
    RandomSearchSource,
    SearchResult,
    map_plan,
    play_planned_game,
    read_book_lines,
    search_node,
    witness_prefix,
)
from alphablokus.games.blokusduo.pentobi.store import (
    PlanParameters,
    PlayoutJob,
    SearchSpaceStore,
    canonical_key,
    node_seed,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from alphablokus.games.blokusduo.board import BlokusDuoBoard

_BOOK = """(
;GM[Blokus Duo]
(
 ;B[f9,e10,f10,g10,f11]TE[1]
 (
  ;W[i4,h5,i5,j5,i6]TE[1]
  (
   ;B[h7,g8,h8,h9,i9]TE[1]
  )
  (
   ;B[g7,g8,h8,i8,h9]TE[1]
  )
 )
 (
  ;W[j5,i6,j6,k6,j7]TE[1]
 )
)
)
"""


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


@pytest.fixture
def store(game: BlokusDuoGame, tmp_path: Path) -> Iterator[SearchSpaceStore]:
    with SearchSpaceStore(tmp_path / "store.sqlite", game, level=9) as opened:
        yield opened


# --------------------------------------------------------------------------- #
# Phase A
# --------------------------------------------------------------------------- #


def test_witness_prefix_alternates_colours(game: BlokusDuoGame) -> None:
    """White moves first; the prefix carries the side that played each action."""
    board = game.initialise_board()
    first = int(np.flatnonzero(game.valid_move_masking(board, 1))[0])
    board, player = game.get_next_state(board, 1, first)
    second = int(np.flatnonzero(game.valid_move_masking(board, player))[0])
    assert witness_prefix(game, (first, second)) == [(1, first), (-1, second)]


def test_search_node_records_the_position_it_replayed(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """A node is searched at its own witness position, with its content-derived seed."""
    root = store.root_node()
    source = RandomSearchSource(game, breadth=4)
    assert search_node(store, source, root) > 0
    record = store.node(root)
    assert record.is_searched
    assert record.engine_seed == node_seed(record.board_key)
    assert record.root_visits is not None and record.root_visits > 0
    mask = game.valid_move_masking(game.initialise_board(), 1)
    assert all(mask[edge.action] == 1 for edge in store.edges(root))


def test_map_plan_leaves_no_mapping_debt(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Phase A's contract: every node the plan allocates games to is searched or a start."""
    params = PlanParameters(budget=200, temperature=2.0, min_replicas=2)
    draft = map_plan(store, RandomSearchSource(game, breadth=4), params)
    assert draft.mapping_queue == ()
    assert draft.planned_games == 200
    store.save_plan(draft)
    assert store.coverage().mapping_debt == 0
    assert store.coverage().planned_games == 200
    # Depth emerged from the budget rather than being configured.
    assert max(store.node(a.node_id).depth for a in draft.starts) > 1


def test_map_plan_is_incremental_across_budgets(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A top-up re-plans and maps only the newly needed nodes — nothing is re-searched."""
    source = RandomSearchSource(game, breadth=4)
    map_plan(store, source, PlanParameters(budget=200, temperature=2.0, min_replicas=2))
    searched_before = {record.node_id for record in store.nodes(status="searched")}
    stamps_before = {record.node_id: record.searched_at for record in store.nodes(status="searched")}

    draft = map_plan(store, source, PlanParameters(budget=800, temperature=2.0, min_replicas=2))
    assert draft.mapping_queue == ()
    searched_after = {record.node_id for record in store.nodes(status="searched")}
    assert searched_before <= searched_after
    assert len(searched_after) > len(searched_before)  # the bigger budget mapped further
    for node_id, searched_at in stamps_before.items():
        assert store.node(node_id).searched_at == searched_at  # untouched, not re-searched


def test_map_plan_raises_rather_than_looping_forever(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A mapping loop that will not converge is a bug, and must say so."""
    with pytest.raises(CorpusGenerationError, match="did not converge"):
        map_plan(store, RandomSearchSource(game, breadth=4), PlanParameters(10_000, 2.0, 2), max_rounds=1)


# --------------------------------------------------------------------------- #
# The opening book
# --------------------------------------------------------------------------- #


def test_read_book_lines_extracts_root_to_leaf_variations(game: BlokusDuoGame, tmp_path: Path) -> None:
    """Every SGF leaf is one opening line, in our action indices, White first.

    Book ``B`` is our White (the SGF lists the first player first), so a line's first
    action must be legal for White from the empty board — which is what the legality
    replay inside the parser checks.
    """
    path = tmp_path / "book.blksgf"
    path.write_text(_BOOK)
    lines = read_book_lines(path, game)
    assert len(lines) == 3  # two B-leaves under the first W move, one lone W leaf
    assert [len(line) for line in lines] == [3, 3, 2]
    assert len({line[0] for line in lines}) == 1  # all three share the book's first move
    mask = game.valid_move_masking(game.initialise_board(), 1)
    assert mask[lines[0][0]] == 1


def test_book_lines_enter_the_dag_with_a_games_floor(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    tmp_path: Path,
) -> None:
    """Book lines are force-mapped and floored at R games whatever the allocation says."""
    path = tmp_path / "book.blksgf"
    path.write_text(_BOOK)
    lines = read_book_lines(path, game)
    terminals = store.insert_book_paths(lines)
    assert len(terminals) == len(lines)
    assert all(store.node(node_id).book_terminal for node_id in terminals)
    assert all(store.node(node_id).source == "book" for node_id in terminals)

    params = PlanParameters(budget=300, temperature=2.0, min_replicas=2)
    draft = map_plan(store, RandomSearchSource(game, breadth=4), params)
    planned = {a.node_id: a.planned_games for a in draft.allocations}
    assert all(planned[node_id] >= params.min_replicas for node_id in terminals)
    assert draft.planned_games == 300


def test_an_illegal_book_line_is_rejected(game: BlokusDuoGame, tmp_path: Path) -> None:
    """A book that disagrees with our rules engine must fail loudly, not be skipped."""
    path = tmp_path / "book.blksgf"
    path.write_text("(\n;GM[Blokus Duo]\n(\n ;B[a1,a2,a3,a4,a5]TE[1]\n)\n)\n")  # nowhere near (4, 4)
    with pytest.raises(CorpusGenerationError, match="illegal"):
        read_book_lines(path, game)


# --------------------------------------------------------------------------- #
# Phase B
# --------------------------------------------------------------------------- #


class BoardTrackingSearchSource(RandomSearchSource):
    """A source that keeps its **own** board, the way ``pentobi-gtp`` does.

    :class:`RandomSearchSource` ignores everything handed to ``advance``, so a caller
    could relay moves under the wrong colour — or not at all — and no test would notice.
    The real engine tracks the position itself and rejects a move that is illegal for the
    colour it is told, which is exactly the failure this double reproduces on CI.
    """

    def __init__(self, game: BlokusDuoGame, *, breadth: int = 4, seed: int = 0) -> None:
        super().__init__(game, breadth=breadth, seed=seed)
        self._own_game = game
        self._board = game.initialise_board()
        self._to_move = 1
        self.relayed: list[tuple[int, int]] = []

    def begin_position(self, seed: int, prefix: Sequence[tuple[int, int]]) -> None:
        super().begin_position(seed, prefix)
        self._board = self._own_game.initialise_board()
        self._to_move = 1
        for player, action in prefix:
            self.advance(self._board, player, action)

    def advance(self, board: BlokusDuoBoard, player: int, action: int) -> None:
        """Apply the move as the *stated* colour, rejecting it if that is illegal.

        Deliberately does **not** require strict alternation: every GTP command names its
        colour explicitly, so the real engine never assumes whose turn it is — and a
        forced pass is never announced to it at all (there is no ``play <c> pass``). The
        legality check is what catches a move sent under the wrong colour, and it is
        exactly the check the real engine applies.
        """
        self.relayed.append((player, action))
        if not self._own_game.valid_move_masking(self._board, player)[action]:
            raise AssertionError(f"move {action} is illegal for player {player:+d}")
        self._board, self._to_move = self._own_game.get_next_state(self._board, player, action)

    def final_white_margin(self) -> int | None:
        white, black = self._own_game.final_scores(self._board)
        return white - black


def _root_job(game: BlokusDuoGame) -> PlayoutJob:
    compact = np.asarray(game.get_canonical_form(game.initialise_board(), 1).to_compact(), dtype=np.int8)
    return PlayoutJob(
        node_id=1,
        replica=0,
        game_id=0,
        engine_seed=0,
        board_key=canonical_key(compact)[0],
        witness_actions=(),
    )


def test_continuation_moves_are_relayed_under_the_colour_that_played_them(game: BlokusDuoGame) -> None:
    """The engine must be told *who* moved, not who moves next.

    Relaying a move under the opponent's colour desyncs the engine's board from ours
    immediately: real ``pentobi-gtp`` either rejects the move outright or silently
    diverges, and every game in the run fails. The bug is invisible to a source that
    ignores its arguments, so this asserts the relayed stream directly as well.
    """
    source = BoardTrackingSearchSource(game)
    harvested = play_planned_game(game, source, _root_job(game))

    assert harvested.plies, "the game produced no harvested plies"
    expected = [(ply.player, ply.action) for ply in harvested.plies]
    assert source.relayed == expected


def test_a_planned_game_replays_its_witness_prefix_under_the_right_colours(
    game: BlokusDuoGame,
    store: SearchSpaceStore,
) -> None:
    """The prefix is replayed into the engine before play resumes — also colour-checked."""
    root = store.root_node()
    search_node(store, RandomSearchSource(game, breadth=4), root)
    edge = store.edges(root)[0]
    child = store.expand_child(root, edge.action)
    record = store.node(child)
    job = PlayoutJob(
        node_id=child,
        replica=0,
        game_id=0,
        engine_seed=0,
        board_key=record.board_key,
        witness_actions=record.witness_actions,
    )
    source = BoardTrackingSearchSource(game)
    harvested = play_planned_game(game, source, job)

    prefix = witness_prefix(game, record.witness_actions)
    assert source.relayed[: len(prefix)] == prefix
    assert harvested.plies[0].ply == len(record.witness_actions)


class ForcedPassRaisingSource(BoardTrackingSearchSource):
    """Refuses to search when the side to move has no placement, as the engine does.

    ``pentobi-gtp``'s ``reg_genmove`` throws ``player failed to generate a move`` rather
    than returning a pass (``GtpEngine::cmd_reg_genmove`` rejects a null move) — unlike
    ``genmove``, which v1 used and which returns the string "pass". A double that returns
    an empty result instead lets the caller ask a question the real engine refuses, so
    the whole run aborts the first time anyone runs out of moves.
    """

    def search(self, board: BlokusDuoBoard, player: int) -> SearchResult:
        mask = self._own_game.valid_move_masking(board, player)
        if not mask[: self._own_game.action_codec.pass_action_index].any():
            raise GtpError("player failed to generate a move")
        return super().search(board, player)


def test_a_forced_pass_never_asks_the_engine_to_move(game: BlokusDuoGame) -> None:
    """Games must run to the end even when a player is passed out.

    Every Blokus game ends with one side unable to move, so this is not an edge case —
    it happens in the closing plies of every single game.
    """
    source = ForcedPassRaisingSource(game)
    harvested = play_planned_game(game, source, _root_job(game))

    assert harvested.plies, "no plies harvested"
    assert harvested.white_margin == source.final_white_margin()
    # The pass plies are skipped, not harvested: there is nothing to learn from a
    # position whose only legal move is to do nothing.
    assert all(ply.action != game.action_codec.pass_action_index for ply in harvested.plies)
