"""Tests for the Pentobi ``move_values`` parser (v2 corpus V1).

Fixture-driven on **real captured L9 output** (``data/move_values_l9.txt``, the first
block of ``local/probes/mv_deep.txt``: the 315 root children of a fresh search from the
empty board), so the parser is pinned against the engine's actual formatting rather than
an assumed one. No ``pentobi-gtp`` binary is involved — CI has none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.pentobi.move_values import (
    MoveValuesParseError,
    parse_move_values,
    strip_piece_prefix,
)
from alphablokus.games.blokusduo.pentobi.translation import PentobiMoveTranslator
from alphablokus.games.blokusduo.pieces import default_pieces_path

_FIXTURE = Path(__file__).parent / "data" / "move_values_l9.txt"

# Measured facts about the fixture position (v2 plan fact 4): Pentobi searches 315 of the
# 414 legal first moves, and its top child takes just over half the visits.
_FIXTURE_CHILDREN = 315


@pytest.fixture(scope="module")
def response() -> str:
    return _FIXTURE.read_text()


@pytest.fixture(scope="module")
def game() -> BlokusDuoGame:
    return BlokusDuoGame(pieces_config_path=default_pieces_path())


def test_parses_every_captured_child(response: str) -> None:
    """All 315 emitted root children survive the parse, in the engine's own order."""
    values = parse_move_values(response)
    assert len(values) == _FIXTURE_CHILDREN
    assert values.total_visits == sum(entry.visits for entry in values.entries)
    assert [entry.visits for entry in values.entries] == sorted(
        (entry.visits for entry in values.entries),
        reverse=True,
    ), "the engine emits children strongest-first"


def test_top_child_fields_match_the_captured_line(response: str) -> None:
    """The first line's three numeric fields and cells land in the right slots."""
    top = parse_move_values(response).top
    assert top is not None
    assert top.visits == 846665
    assert top.value_count == pytest.approx(846668.0)
    assert top.value == pytest.approx(0.747)
    assert top.cells == "f8,f9,g9,e10,f10"


def test_search_value_is_the_top_childs_value(response: str) -> None:
    """A position's backed-up value is read here, never from GTP ``get_value`` (fact 5)."""
    values = parse_move_values(response)
    assert values.search_value == pytest.approx(0.747)
    assert values.entries[0].value == values.search_value


def test_piece_prefix_is_stripped_from_every_move(response: str) -> None:
    """``[F]`` / ``[T5]`` / ``[I5]`` annotations never reach the translator."""
    values = parse_move_values(response)
    assert all("[" not in entry.cells and "]" not in entry.cells for entry in values.entries)
    # Multi-character piece names are in the fixture and must be stripped whole.
    assert any(len(name) > 1 for name in _piece_names(response))


def test_parsed_cells_translate_to_legal_first_moves(response: str, game: BlokusDuoGame) -> None:
    """The stripped cells are exactly what ``cells_to_action`` consumes, and every one of
    them is a legal first placement — the end-to-end check that prefix stripping is
    correct rather than merely cosmetic."""
    translator = PentobiMoveTranslator(game)
    mask = game.valid_move_masking(game.initialise_board(), 1)
    actions = {translator.pentobi_to_action_index(entry.cells) for entry in parse_move_values(response).entries}
    assert len(actions) == _FIXTURE_CHILDREN  # no two children collapse to one action
    assert all(mask[action] == 1 for action in actions)


def test_visit_shares_sum_to_one(response: str) -> None:
    """Shares normalise over the emitted children (the allocator's weights)."""
    shares = parse_move_values(response).visit_shares
    assert sum(shares) == pytest.approx(1.0)
    assert shares[0] == pytest.approx(846665 / parse_move_values(response).total_visits)


def test_unvisited_children_are_flagged(response: str) -> None:
    """Zero-visit children report the prior's value at ``value_count`` ~3 (fact 5): their
    value is not a search result and consumers must be able to tell."""
    values = parse_move_values(response)
    unvisited = [entry for entry in values.entries if not entry.is_visited]
    assert unvisited, "the captured search leaves most children unvisited"
    assert all(entry.value_count < 100.0 for entry in unvisited)
    assert all(entry.is_visited for entry in values.entries[:8])


def test_negative_values_are_accepted(response: str) -> None:
    """Values are win-probability-*like*, not bounded to [0, 1] (fact 5)."""
    values = parse_move_values(response)
    assert min(entry.value for entry in values.entries) < 0.0


def test_empty_response_is_legal() -> None:
    """No search tree (forced pass / terminal / book hit) → an empty result, not a crash.

    The captured evidence is the final ``=`` of ``local/probes/mv_deep.txt``.
    """
    for empty in ("", "   ", "\n\n"):
        values = parse_move_values(empty)
        assert len(values) == 0
        assert not values
        assert values.total_visits == 0
        assert values.top is None
        assert values.search_value is None
        assert values.visit_shares == ()


def test_line_without_a_piece_prefix_is_accepted() -> None:
    """A move with no ``[`` annotation parses as the whole remainder."""
    (entry,) = parse_move_values("123 126.0 0.5 e10,f10").entries
    assert entry.cells == "e10,f10"
    assert strip_piece_prefix("e10,f10") == "e10,f10"


def test_pass_move_is_accepted() -> None:
    """A pass has no cells to strip; the token survives for the caller to recognise."""
    (entry,) = parse_move_values("7 10.0 -0.25 pass").entries
    assert entry.cells == "pass"
    assert entry.value == pytest.approx(-0.25)


@pytest.mark.parametrize("bad", ["846665 846668.0 0.747", "not a line at all", "1 2.0 three [F]e10"])
def test_malformed_lines_raise(bad: str) -> None:
    """A shape change in the engine must fail loudly, not silently drop children."""
    with pytest.raises(MoveValuesParseError):
        parse_move_values(bad)


def test_blank_lines_are_skipped(response: str) -> None:
    """Stray blank lines in a response body are not children."""
    padded = "\n" + response.replace("\n", "\n\n") + "\n"
    assert len(parse_move_values(padded)) == _FIXTURE_CHILDREN


def _piece_names(response: str) -> list[str]:
    """The ``[PIECE]`` names in a raw response (fixture introspection only)."""
    return [line.split("[", 1)[1].split("]", 1)[0] for line in response.splitlines() if "[" in line]
