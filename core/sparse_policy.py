"""Sparse storage for MCTS policy targets (self-play memory fix).

A self-play training example's policy target is the MCTS visit distribution over
the action space (17,837 actions for Blokus Duo). It is **sparse** — only the
legal, visited moves are nonzero (tens to a few hundred) — yet it was stored
dense (a 17,837-float vector ≈ 71 KB each), which dominated replay-buffer RAM
and made the scaled run swap-thrash.

This module stores the policy as ``(indices, values)`` (the nonzero entries),
~30×+ smaller, and densifies it only where the network needs a dense target
(at the train step). It is a *lossless* re-encoding: ``densify(sparsify(p)) == p``
exactly. See ``docs/plans/self-play-memory-fix.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# (indices, values): the nonzero entries of a policy vector. ``indices`` is the
# action ids (int32); ``values`` the corresponding probabilities (float32).
SparsePolicy: TypeAlias = "tuple[NDArray[np.int32], NDArray[np.float32]]"


def sparsify(dense: NDArray) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Encode a dense policy vector as its nonzero ``(indices, values)``.

    Args:
        dense: Dense policy of shape ``(action_size,)``.

    Returns:
        ``(indices, values)`` — ``indices`` as ``int32``, ``values`` as
        ``float32``, holding exactly the nonzero entries of ``dense``.
    """
    indices = np.nonzero(dense)[0].astype(np.int32)
    values = np.asarray(dense, dtype=np.float32)[indices]
    return indices, values


def densify(indices: NDArray, values: NDArray, action_size: int) -> NDArray[np.float32]:
    """Reconstruct a dense policy vector from ``(indices, values)``.

    Inverse of :func:`sparsify`: ``densify(*sparsify(p), len(p))`` equals ``p``
    exactly (for a float32 ``p``).

    Args:
        indices: Nonzero action ids.
        values: Probabilities at those ids.
        action_size: Length of the dense vector to produce.
    """
    dense = np.zeros(action_size, dtype=np.float32)
    dense[indices] = values
    return dense


def as_dense(policy: object, action_size: int) -> NDArray[np.float32]:
    """Normalise a stored policy to dense, accepting either encoding.

    Self-play emits sparse ``(indices, values)`` policies (the RAM win), but an
    already-dense vector may reach a consumer too — e.g. examples loaded back
    from the on-disk store (which keeps dense), or hand-built test fixtures. This
    accepts both and always returns the dense vector the network needs, so every
    consumer (training, eval-set build, save) handles either uniformly.
    """
    if isinstance(policy, tuple):
        indices, values = policy
        return densify(indices, values, action_size)
    return np.asarray(policy, dtype=np.float32)
