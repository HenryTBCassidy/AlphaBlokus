"""Sparse-but-connected pairing schedule for the pool BayesElo tournament.

A full round-robin over K checkpoints is O(K²) pairings (60 gens → 1,770), and
each pairing plays real MCTS arena games — far more compute than a rising Elo
curve needs. Instead we pair each checkpoint with a handful of earlier ones at
exponentially spaced offsets. That keeps the comparison graph **connected** (the
precondition for a well-conditioned BayesElo fit) at O(K·log K) pairings.

The single public function :func:`build_pairings` is pure — it takes ordered
checkpoint ids and returns unordered pairs, ready to feed to the arena and the
:func:`alphablokus.evaluation.rating.fit_bayeselo` fit.
"""

from __future__ import annotations


def build_pairings(
    checkpoint_ids: list[str],
    back_ref_offsets: tuple[int, ...],
    include_first_last: bool,
) -> list[tuple[str, str]]:
    """Build a sparse, connected list of unordered checkpoint pairings.

    Args:
        checkpoint_ids: Checkpoints ordered by generation, e.g.
            ``["gen0", "gen1", ..., "gen59"]``. Must be unique.
        back_ref_offsets: For each checkpoint *i*, pair it with checkpoint
            *i − offset* for every offset in range. Exponentially spaced offsets
            (e.g. ``(1, 2, 4, 8, 16, 32)``) keep the graph connected cheaply.
        include_first_last: If True, also pair every checkpoint with the first
            (gen-0 anchor) and the last checkpoint. Guarantees connectivity even
            when ``back_ref_offsets`` is empty or sparse.

    Returns:
        Unordered pairs ``(a, b)`` with ``a`` earlier than ``b`` in
        ``checkpoint_ids``. No self-pairs, no duplicates, order deterministic.

    Raises:
        ValueError: If ``checkpoint_ids`` contains duplicates or a non-positive
            offset is given.
    """
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint_ids must be unique")
    if any(off <= 0 for off in back_ref_offsets):
        raise ValueError(f"back_ref_offsets must all be positive, got {back_ref_offsets}")

    n = len(checkpoint_ids)
    # Collect index pairs as an ordered set: (lo, hi) with lo < hi.
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    def _add(a: int, b: int) -> None:
        if a == b:
            return
        lo, hi = (a, b) if a < b else (b, a)
        if (lo, hi) not in seen:
            seen.add((lo, hi))
            ordered.append((lo, hi))

    for i in range(n):
        for off in back_ref_offsets:
            j = i - off
            if j >= 0:
                _add(j, i)

    if include_first_last and n > 0:
        first, last = 0, n - 1
        for i in range(n):
            _add(first, i)
            _add(last, i)

    return [(checkpoint_ids[lo], checkpoint_ids[hi]) for lo, hi in ordered]
