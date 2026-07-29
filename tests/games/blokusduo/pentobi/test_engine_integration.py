"""End-to-end tests against the **real** ``pentobi-gtp`` binary.

Every other test in this package drives the pipeline through
:class:`RandomSearchSource`, a stand-in for the engine. That stand-in was written from
the same assumptions as the code it exercises, so it can only confirm that the code
agrees with its author's beliefs about Pentobi — never that those beliefs are right.
Two production-stopping bugs got through a full green suite that way:

- continuation moves were relayed under the opponent's colour (the stand-in ignored the
  colour argument entirely);
- a forced pass was expected to yield an empty ``move_values``, when the real engine
  raises ``player failed to generate a move`` from ``reg_genmove`` (the stand-in returned
  the empty list the code expected).

Both are caught in seconds by playing one real game. These tests are marked ``slow`` and
skip themselves when no binary is present, so CI and the laptop stay fast — but they must
pass on the box before a corpus run is trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.gtp import find_pentobi_gtp
from alphablokus.games.blokusduo.pentobi.harvest import (
    PentobiSearchSource,
    map_plan_serially,
    play_planned_game,
)
from alphablokus.games.blokusduo.pentobi.store import (
    PlanParameters,
    PlayoutJob,
    SearchSpaceStore,
    canonical_key,
)
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(find_pentobi_gtp() is None, reason="pentobi-gtp not built here"),
]

#: Level 1 keeps a full game to a few seconds. The engine *protocol* is identical at
#: every level, which is what these tests are about — strength is not.
_LEVEL = 1


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


def test_a_whole_game_plays_against_the_real_engine(game: BlokusDuoGame) -> None:
    """The headline check: can this pipeline produce a single game at all?

    Plays from the empty board to a terminal position through the real binary. Catches
    the colour-relay bug (the engine rejects a move sent under the wrong colour) and the
    forced-pass bug (every game ends with one side passed out), which between them made
    generation impossible while every stand-in test passed.
    """
    compact = np.asarray(game.get_canonical_form(game.initialise_board(), 1).to_compact(), dtype=np.int8)
    job = PlayoutJob(
        node_id=1,
        replica=0,
        game_id=0,
        engine_seed=7,
        board_key=canonical_key(compact)[0],
        witness_actions=(),
    )
    with PentobiSearchSource(game, _LEVEL) as source:
        harvested = play_planned_game(game, source, job)

    assert len(harvested.plies) > 10, "a real game should harvest more than a handful of plies"
    # The engine's own final_score agreed with our rules engine, or play_planned_game
    # would already have raised. (Duo scores are differentials and go negative, so there
    # is nothing to assert about their sign.)
    assert harvested.white_margin == harvested.white_score - harvested.black_score
    assert harvested.actions[: len(job.witness_actions)] == job.witness_actions
    # Forced passes are in the action stream but never harvested as training rows.
    assert game.action_codec.pass_action_index in harvested.actions
    assert all(ply.action != game.action_codec.pass_action_index for ply in harvested.plies)

    # Replaying the full action sequence must reproduce the game exactly.
    board, player = game.initialise_board(), 1
    for action in harvested.actions:
        assert game.valid_move_masking(board, player)[action] == 1
        board, player = game.get_next_state(board, player, action)
    assert game.get_game_ended(board, player) != 0
    assert game.final_scores(board) == (harvested.white_score, harvested.black_score)


def test_every_harvested_ply_carries_a_real_distribution(game: BlokusDuoGame) -> None:
    """``move_values`` parses and the targets are usable at every ply of a real game."""
    compact = np.asarray(game.get_canonical_form(game.initialise_board(), 1).to_compact(), dtype=np.int8)
    job = PlayoutJob(1, 0, 0, 11, canonical_key(compact)[0], ())
    with PentobiSearchSource(game, _LEVEL) as source:
        harvested = play_planned_game(game, source, job)

    board, player = game.initialise_board(), 1
    by_ply = {ply.ply: ply for ply in harvested.plies}
    for index, action in enumerate(harvested.actions):
        if index in by_ply:
            ply = by_ply[index]
            legal = set(np.flatnonzero(game.valid_move_masking(board, player)).tolist())
            assert float(ply.target.values.sum()) == pytest.approx(1.0, abs=1e-5)
            assert set(ply.target.indices.tolist()) <= legal  # support never leaves the legal set
            assert ply.action in legal
        board, player = game.get_next_state(board, player, action)


def test_a_planned_run_maps_and_generates_end_to_end(game: BlokusDuoGame, tmp_path: Path) -> None:
    """Phase A then phase B, both against the real engine, writing a real store.

    The smallest possible version of a production run: the check that would have failed
    in seconds rather than after a multi-day generation.
    """
    with SearchSpaceStore(tmp_path / "store.sqlite", game, level=_LEVEL) as store:
        with PentobiSearchSource(game, _LEVEL) as source:
            draft = map_plan_serially(store, source, PlanParameters(budget=8, temperature=2.0, min_replicas=2))
            store.save_plan(draft)
            assert draft.mapping_queue == ()
            assert draft.planned_games == 8

            jobs = store.schedule(2)
            assert len(jobs) == 2
            for job in jobs:
                harvested = play_planned_game(game, source, job)
                assert harvested.plies
                store.mark_done(
                    harvested.node_id,
                    harvested.replica,
                    shard="s0",
                    white_margin=harvested.white_margin,
                    plies=len(harvested.plies),
                )

        assert store.playout_counts()["done"] == 2
        store.link()
        assert store.node(store.root_node()).outcome_count == 2
