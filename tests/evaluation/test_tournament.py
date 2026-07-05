"""Tests for the sparse pairing-schedule generator (``evaluation/tournament.py``).

The key property is **connectivity**: the BayesElo fit is only well-posed when
every checkpoint is reachable from every other through the games played. These
tests assert that directly via union-find over the produced pairs.
"""

from __future__ import annotations

import pytest

from alphablokus.evaluation.tournament import build_pairings


def _is_connected(ids: list[str], pairs: list[tuple[str, str]]) -> bool:
    """Union-find: is the comparison graph a single connected component?"""
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = {find(i) for i in ids}
    return len(roots) == 1


def _ids(n: int) -> list[str]:
    return [f"gen{i}" for i in range(n)]


def test_no_self_pairs_or_duplicates() -> None:
    ids = _ids(20)
    pairs = build_pairings(ids, back_ref_offsets=(1, 2, 4, 8, 16), include_first_last=True)
    assert all(a != b for a, b in pairs)
    unordered = {frozenset((a, b)) for a, b in pairs}
    assert len(unordered) == len(pairs)


def test_every_checkpoint_appears() -> None:
    ids = _ids(30)
    pairs = build_pairings(ids, back_ref_offsets=(1, 2, 4, 8, 16, 32), include_first_last=True)
    seen = {p for pair in pairs for p in pair}
    assert seen == set(ids)


def test_graph_is_connected_with_back_refs_only() -> None:
    """Consecutive offset 1 alone already chains every node together."""
    ids = _ids(25)
    pairs = build_pairings(ids, back_ref_offsets=(1,), include_first_last=False)
    assert _is_connected(ids, pairs)


def test_graph_is_connected_default_schedule() -> None:
    ids = _ids(60)
    pairs = build_pairings(ids, back_ref_offsets=(1, 2, 4, 8, 16, 32), include_first_last=True)
    assert _is_connected(ids, pairs)


def test_include_first_last_alone_connects() -> None:
    """Even with no back-refs, first+last hubs make the graph connected."""
    ids = _ids(15)
    pairs = build_pairings(ids, back_ref_offsets=(), include_first_last=True)
    assert _is_connected(ids, pairs)


def test_sparse_count_is_far_below_full_round_robin() -> None:
    """The whole point: O(K·log K), not O(K²)."""
    ids = _ids(60)
    pairs = build_pairings(ids, back_ref_offsets=(1, 2, 4, 8, 16, 32), include_first_last=True)
    full_round_robin = 60 * 59 // 2  # 1770
    assert len(pairs) < full_round_robin // 3


def test_a_is_earlier_than_b() -> None:
    ids = _ids(10)
    pairs = build_pairings(ids, back_ref_offsets=(1, 2), include_first_last=True)
    order = {p: i for i, p in enumerate(ids)}
    assert all(order[a] < order[b] for a, b in pairs)


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_pairings(["gen0", "gen0"], back_ref_offsets=(1,), include_first_last=False)


def test_rejects_non_positive_offset() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_pairings(_ids(5), back_ref_offsets=(0, 1), include_first_last=False)


def test_deterministic_order() -> None:
    ids = _ids(12)
    a = build_pairings(ids, back_ref_offsets=(1, 2, 4), include_first_last=True)
    b = build_pairings(ids, back_ref_offsets=(1, 2, 4), include_first_last=True)
    assert a == b
