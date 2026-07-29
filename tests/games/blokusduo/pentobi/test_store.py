"""Tests for the v2 corpus search-space store (plan V3 / store design S1–S6).

Everything here runs against the **real** rules engine and a real on-disk SQLite file —
no ``pentobi-gtp`` binary (CI has none) and no mocks of game logic. Searches are handed
to the store as :class:`SearchChild` lists, either synthetic or parsed from the captured
L9 ``move_values`` fixture the V1 parser tests use.

The symmetry-key behaviour gets the most attention: node keys are
``min(compact, transposed compact)`` and every action stored against a node must be in
that node's key frame, which is the single most likely source of a subtle bug. Measured
ground truth asserted below: of the 414 legal first moves exactly **10** are
self-symmetric and there are **212** distinct positions after canonicalisation, and
Pentobi's 315 searched root children collapse to **160** canonical children.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from loguru import logger

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.move_values import parse_move_values
from alphablokus.games.blokusduo.pentobi.store import (
    SearchChild,
    SearchSpaceStore,
    StoreError,
    canonical_key,
    children_from_move_values,
    is_symmetric_key,
    node_seed,
    playout_seed,
)
from alphablokus.games.blokusduo.pentobi.translation import PentobiMoveTranslator
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alphablokus.games.blokusduo.board import BlokusDuoBoard

_FIXTURE = Path(__file__).parent / "data" / "move_values_l9.txt"

# Measured constants (v2 plan facts 4 / store design D-b).
_N_FIRST_MOVES = 414
_N_SELF_SYMMETRIC_FIRST_MOVES = 10
_N_CANONICAL_FIRST_POSITIONS = 212
_N_SEARCHED_ROOT_CHILDREN = 315
_N_CANONICAL_SEARCHED_CHILDREN = 160


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


@pytest.fixture(scope="module")
def first_moves(game: BlokusDuoGame) -> tuple[int, ...]:
    mask = game.valid_move_masking(game.initialise_board(), 1)
    return tuple(int(a) for a in np.flatnonzero(mask) if a != game.action_codec.pass_action_index)


@pytest.fixture
def store(game: BlokusDuoGame, tmp_path: Path) -> Iterator[SearchSpaceStore]:
    with SearchSpaceStore(tmp_path / "store.sqlite", game, level=9) as opened:
        yield opened


def _compact_after(game: BlokusDuoGame, actions: tuple[int, ...]) -> np.ndarray:
    """The side-to-move canonical compact grid after playing ``actions`` from the start."""
    board = game.initialise_board()
    player = 1
    for action in actions:
        board, player = game.get_next_state(board, player, action)
    return np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)


def _board_after(game: BlokusDuoGame, actions: tuple[int, ...]) -> tuple[BlokusDuoBoard, int]:
    board = game.initialise_board()
    player = 1
    for action in actions:
        board, player = game.get_next_state(board, player, action)
    return board, player


def _asymmetric_first_move(game: BlokusDuoGame, first_moves: tuple[int, ...]) -> int:
    """A legal first move whose mirror is a different move (i.e. not self-symmetric)."""
    return next(action for action in first_moves if game.transpose_action(action) != action)


def _transposed_frame_first_move(game: BlokusDuoGame) -> int:
    """A legal first move whose position canonicalises to ``key_frame = 1``."""
    mask = game.valid_move_masking(game.initialise_board(), 1)
    return next(int(a) for a in np.flatnonzero(mask) if canonical_key(_compact_after(game, (int(a),)))[1] == 1)


# --------------------------------------------------------------------------- #
# S1: symmetry-canonical keys
# --------------------------------------------------------------------------- #


def test_empty_board_is_self_symmetric(game: BlokusDuoGame) -> None:
    """Both starting squares sit on the main diagonal, so the root is its own mirror."""
    key, frame = canonical_key(_compact_after(game, ()))
    assert frame == 0
    assert is_symmetric_key(key, game.board_size)


def test_canonical_key_collapses_every_mirror_pair(game: BlokusDuoGame, first_moves: tuple[int, ...]) -> None:
    """A position and its main-diagonal mirror produce the same key, in opposite frames.

    The property test the store design asks for, run over the *entire* ply-1 fan: for all
    414 legal first moves, canonicalising the position after ``a`` and after
    ``transpose_action(a)`` must agree byte-for-byte, and the two frames must differ
    exactly when the position is not self-symmetric.
    """
    for action in first_moves:
        mirror = game.transpose_action(action)
        key, frame = canonical_key(_compact_after(game, (action,)))
        mirror_key, mirror_frame = canonical_key(_compact_after(game, (mirror,)))
        assert key == mirror_key
        if is_symmetric_key(key, game.board_size):
            assert mirror == action
            assert frame == mirror_frame == 0
        else:
            assert mirror != action
            assert {frame, mirror_frame} == {0, 1}


def test_first_move_canonicalisation_matches_the_measured_counts(
    game: BlokusDuoGame,
    first_moves: tuple[int, ...],
) -> None:
    """414 legal first moves → 212 distinct positions, 10 of them self-symmetric."""
    keys = [canonical_key(_compact_after(game, (action,)))[0] for action in first_moves]
    assert len(first_moves) == _N_FIRST_MOVES
    assert len(set(keys)) == _N_CANONICAL_FIRST_POSITIONS
    assert sum(1 for key in set(keys) if is_symmetric_key(key, game.board_size)) == _N_SELF_SYMMETRIC_FIRST_MOVES


def test_every_first_position_round_trips_through_the_store(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    first_moves: tuple[int, ...],
) -> None:
    """Insert all 414 ply-1 positions and check every stored node, exhaustively.

    Per node: the key is reproduced by canonicalising the position its own witness path
    replays to; ``key_frame`` correctly says which of the two orientations the key is;
    and the key-frame action mapping is an involution on the position's whole legal set.
    """
    node_ids = [store.upsert_node(*_board_after(game, (action,)), (action,)) for action in first_moves]
    assert len(set(node_ids)) == _N_CANONICAL_FIRST_POSITIONS  # mirror twins share a node

    for node_id in set(node_ids):
        record = store.node(node_id)
        board, player = store.board_at(node_id)
        compact = np.asarray(game.get_canonical_form(board, player).to_compact(), dtype=np.int8)
        key, frame = canonical_key(compact)
        assert (key, frame) == (record.board_key, record.key_frame)
        grid = np.frombuffer(record.board_key, dtype=np.int8).reshape(game.board_size, game.board_size)
        as_played = grid.T if record.key_frame else grid
        assert np.array_equal(np.ascontiguousarray(as_played), compact)
        legal = np.flatnonzero(game.valid_move_masking(board, player))
        for action in (int(legal[0]), int(legal[len(legal) // 2]), int(legal[-1])):
            assert store.from_key_frame(record, store.to_key_frame(record, action)) == action


def test_upsert_is_idempotent_and_keeps_the_first_witness(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    first_moves: tuple[int, ...],
) -> None:
    """Re-inserting a position (by any move order) is a no-op returning the same node."""
    action = _asymmetric_first_move(game, first_moves)
    mirror = game.transpose_action(action)
    first = store.upsert_node(*_board_after(game, (action,)), (action,))
    again = store.upsert_node(*_board_after(game, (action,)), (action,))
    twin = store.upsert_node(*_board_after(game, (mirror,)), (mirror,))
    assert first == again == twin
    assert store.node(first).witness_actions == (action,)
    assert len(store.nodes()) == 1


def test_root_node_is_stable(store: SearchSpaceStore) -> None:
    """The root is the empty board with White to move — and is inserted only once."""
    root = store.root_node()
    assert store.root_node() == root
    record = store.node(root)
    assert (record.depth, record.player, record.witness_actions, record.source) == (0, 1, (), "root")


# --------------------------------------------------------------------------- #
# S1: content-derived seeds and pinned metadata
# --------------------------------------------------------------------------- #


def test_seeds_are_content_derived_and_stable() -> None:
    """Seeds are a pure function of content, in range, and distinct per replica."""
    key = bytes(range(196))
    other = bytes(range(1, 197))
    assert node_seed(key) == node_seed(key)
    assert node_seed(key) != node_seed(other)
    assert 0 <= node_seed(key) <= 0x7FFFFFFF
    replicas = {playout_seed(key, replica) for replica in range(64)}
    assert len(replicas) == 64
    assert all(0 <= seed <= 0x7FFFFFFF for seed in replicas)
    assert playout_seed(key, 3) == playout_seed(key, 3)
    assert playout_seed(key, 3) != playout_seed(other, 3)


def test_node_records_its_search_seed(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    root = store.root_node()
    assert store.node(root).engine_seed == node_seed(store.node(root).board_key)
    assert store.meta["level"] == "9"
    assert store.meta["policy_size"] == str(game.get_action_size())


def test_reopening_at_a_different_level_raises(game: BlokusDuoGame, tmp_path: Path) -> None:
    """Level is pinned in ``meta``: mixing teachers in one DAG must fail loudly."""
    path = tmp_path / "store.sqlite"
    SearchSpaceStore(path, game, level=9).close()
    SearchSpaceStore(path, game, level=9).close()  # same level reopens fine
    with pytest.raises(StoreError, match="level"):
        SearchSpaceStore(path, game, level=5)


# --------------------------------------------------------------------------- #
# S2: search recording
# --------------------------------------------------------------------------- #


def test_records_the_captured_l9_root_search(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The real 315-child L9 root search collapses to 160 canonical children.

    The root is self-symmetric, so each mirror pair is one position and must carry the
    pair's **combined** visits — otherwise every canonical opening is charged half its
    true share and the allocator systematically under-funds the engine's own favourites.
    """
    children = children_from_move_values(parse_move_values(_FIXTURE.read_text()), PentobiMoveTranslator(game))
    assert len(children) == _N_SEARCHED_ROOT_CHILDREN
    root = store.root_node()
    written = store.record_search(root, children, seconds=29.6)
    assert written == _N_CANONICAL_SEARCHED_CHILDREN

    edges = store.edges(root)
    record = store.node(root)
    assert record.is_searched
    assert record.root_visits == sum(child.visits for child in children)  # merging conserves visits
    assert record.search_value == pytest.approx(0.747)  # the top child's value, not GTP get_value
    assert record.search_seconds == pytest.approx(29.6)
    assert [edge.rank for edge in edges] == list(range(len(edges)))
    assert [edge.visits for edge in edges] == sorted((edge.visits for edge in edges), reverse=True)
    assert sum(edge.visit_share for edge in edges) == pytest.approx(1.0)
    assert len({edge.action for edge in edges}) == len(edges)
    # Every merged action is the canonical member of its mirror pair, and legal at the root.
    mask = game.valid_move_masking(game.initialise_board(), 1)
    for edge in edges:
        assert edge.action <= game.transpose_action(edge.action)
        assert mask[edge.action] == 1


def test_mirror_pair_visits_are_summed_at_a_symmetric_node(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    first_moves: tuple[int, ...],
) -> None:
    """A synthetic two-child search makes the merge arithmetic explicit."""
    action = _asymmetric_first_move(game, first_moves)
    mirror = game.transpose_action(action)
    root = store.root_node()
    store.record_search(
        root,
        [SearchChild(action=action, visits=30, value=0.7), SearchChild(action=mirror, visits=10, value=0.4)],
    )
    (edge,) = store.edges(root)
    assert edge.action == min(action, mirror)
    assert edge.visits == 40
    assert edge.visit_share == pytest.approx(1.0)
    assert edge.child_value == pytest.approx(0.7)  # the better-supported member's value


def test_edges_are_stored_in_the_key_frame(game: BlokusDuoGame, tmp_path: Path) -> None:
    """Two stores reach the same position from mirrored move orders and store identical rows.

    This is the frame invariant with teeth: node A is inserted via ``a`` (say frame 0) and
    node B via ``transpose_action(a)`` (frame 1), so B's engine-frame actions are all
    mirrored relative to A's. After ``record_search`` maps both into the key frame the
    stored edges must be byte-identical — that is what lets the allocator, the exporter
    and the trainer read edges without ever asking which order produced them.
    """
    action = _transposed_frame_first_move(game)
    mirror = game.transpose_action(action)
    board, player = _board_after(game, (action,))
    children = [int(a) for a in np.flatnonzero(game.valid_move_masking(board, player))][:6]

    def record(path: Path, first: int, moves: list[int]) -> tuple[bytes, int, list[tuple[int, int, int, float]]]:
        with SearchSpaceStore(path, game, level=9) as store:
            node_id = store.upsert_node(*_board_after(game, (first,)), (first,))
            store.record_search(
                node_id,
                [SearchChild(action=a, visits=100 - 10 * i, value=0.5 - 0.01 * i) for i, a in enumerate(moves)],
            )
            record = store.node(node_id)
            edges = [(e.action, e.rank, e.visits, e.child_value) for e in store.edges(node_id)]
            return record.board_key, record.key_frame, edges

    key_a, frame_a, edges_a = record(tmp_path / "a.sqlite", action, children)
    # The mirrored engine reports the mirrored moves of the mirrored position.
    key_b, frame_b, edges_b = record(tmp_path / "b.sqlite", mirror, [game.transpose_action(a) for a in children])
    assert key_a == key_b
    assert {frame_a, frame_b} == {0, 1}
    assert edges_a == edges_b


def test_record_search_rejects_children_illegal_at_the_node(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
    first_moves: tuple[int, ...],
) -> None:
    """Children must be legal where they were recorded — the cheap desync guard.

    It catches a search recorded against the wrong node (or in the wrong frame) whenever
    that produces an illegal move; it is not a *proof* of frame correctness, because a
    position and its mirror share much of their legal set. The frame invariant itself is
    what :func:`test_edges_are_stored_in_the_key_frame` pins.
    """
    action = _asymmetric_first_move(game, first_moves)
    node_id = store.upsert_node(*_board_after(game, (action,)), (action,))
    with pytest.raises(StoreError, match="illegal"):  # replaying White's own move: cells occupied
        store.record_search(node_id, [SearchChild(action=action, visits=10, value=0.5)])


def test_re_recording_a_search_preserves_child_links(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """Repair after a crash is idempotent and does not orphan expanded children."""
    root = store.root_node()
    legal = [int(a) for a in np.flatnonzero(game.valid_move_masking(game.initialise_board(), 1))[:4]]
    children = [SearchChild(action=a, visits=40 - i, value=0.6) for i, a in enumerate(legal)]
    store.record_search(root, children)
    first_edge = store.edges(root)[0]
    child_id = store.expand_child(root, first_edge.action)
    store.record_search(root, children)
    assert store.edges(root)[0].child_id == child_id


def test_dag_hash_tracks_content(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """The hash pins which DAG a plan or an export was computed from."""
    root = store.root_node()
    empty = store.dag_hash()
    legal = [int(a) for a in np.flatnonzero(game.valid_move_masking(game.initialise_board(), 1))[:3]]
    store.record_search(root, [SearchChild(action=a, visits=10 * (3 - i), value=0.5) for i, a in enumerate(legal)])
    searched = store.dag_hash()
    assert searched != empty
    assert store.dag_hash() == searched  # stable for unchanged content
    store.record_search(root, [SearchChild(action=a, visits=1, value=0.5) for a in legal])
    assert store.dag_hash() != searched


def test_expand_child_extends_the_witness_path(store: SearchSpaceStore, game: BlokusDuoGame) -> None:
    """A child node is the parent's position plus the edge's move, in the as-played frame."""
    action = _transposed_frame_first_move(game)
    node_id = store.upsert_node(*_board_after(game, (action,)), (action,))
    board, player = _board_after(game, (action,))
    reply = int(np.flatnonzero(game.valid_move_masking(board, player))[0])
    record = store.node(node_id)
    store.record_search(node_id, [SearchChild(action=reply, visits=5, value=0.3)])
    child_id = store.expand_child(node_id, store.to_key_frame(record, reply))
    child = store.node(child_id)
    assert child.witness_actions == (action, reply)
    assert child.depth == 2
    assert child.player == 1
    assert store.edges(node_id)[0].child_id == child_id
    child_board, child_player = store.board_at(child_id)
    key, frame = canonical_key(np.asarray(game.get_canonical_form(child_board, child_player).to_compact(), np.int8))
    assert (key, frame) == (child.board_key, child.key_frame)


def test_searching_a_node_preserves_book_edges_the_engine_did_not_report(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """A book line must survive its parent being searched.

    ``record_search`` replaces a node's edge list wholesale. The 44 curated lines are
    force-inserted *because* Pentobi may not favour them, so any book move outside the
    engine's reported children would be deleted the moment the root is searched — which
    always happens. The child node keeps its games floor via ``book_terminal``, so the
    loss is silent: it simply drops out of the graph, contributing nothing to reach
    weights, to ``link``'s outcome aggregation, or to the export's ancestry.
    """
    root = store.root_node()
    legal = [
        int(action)
        for action in np.flatnonzero(game.valid_move_masking(game.initialise_board(), 1))
        if action != game.action_codec.pass_action_index
    ]
    book_action, searched = legal[0], legal[1:5]
    (terminal,) = store.insert_book_paths([[book_action]])

    store.record_search(root, [SearchChild(action=action, visits=100, value=0.6) for action in searched])

    edges = {edge.action: edge for edge in store.edges(root)}
    key_action = store.to_key_frame(store.node(root), book_action)
    assert key_action in edges, "the book edge was deleted by the search that followed it"
    assert edges[key_action].source == "book"
    assert edges[key_action].child_id == terminal
    assert store.node(terminal).book_terminal
    # ...and the searched children are all present too, ranked ahead of it.
    for action in searched:
        assert store.to_key_frame(store.node(root), action) in edges
    assert edges[key_action].rank >= len(searched)


def test_reconcile_flags_a_shard_from_a_different_corpus(
    store: SearchSpaceStore,
    game: BlokusDuoGame,
) -> None:
    """Adopting a shard written by another run should not happen quietly.

    Shards are self-describing, so ``reconcile`` rebuilds the playout registry from them
    — which means any shard sitting in the games directory gets adopted. Two corpora
    built from the same book share many positions, so a stray shard's games look
    perfectly legitimate. (Observed for real: a killed run's worker outlived the kill and
    wrote into a freshly recreated corpus; its 8 games were absorbed silently.) The DAG
    hash in the footer is the discriminator, and it must at least be shouted about.
    """
    root = store.root_node()
    legal = [
        int(a)
        for a in np.flatnonzero(game.valid_move_masking(game.initialise_board(), 1))
        if a != game.action_codec.pass_action_index
    ]
    store.record_search(root, [SearchChild(action=a, visits=100, value=0.6) for a in legal[:3]])
    known = store.dag_hash()

    from alphablokus.games.blokusduo.pentobi.store import ReconcileEntry

    entry = ReconcileEntry(
        board_key=store.node(root).board_key,
        replica=0,
        game_id=0,
        shard="corpus_00000.parquet",
        white_margin=3,
        plies=20,
    )
    assert store.knows_dag_hash(known)
    assert not store.knows_dag_hash("f" * 64)

    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING")
    try:
        store.reconcile([dataclasses.replace(entry, dag_hash="f" * 64)])
        assert any("never produced" in message for message in warnings)
        warnings.clear()
        store.reconcile([dataclasses.replace(entry, dag_hash=known)])
        assert not [message for message in warnings if "never produced" in message]
    finally:
        logger.remove(sink)
