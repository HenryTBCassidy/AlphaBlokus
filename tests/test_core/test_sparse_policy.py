"""Tests for sparse policy storage (self-play memory fix, M2/M5)."""

from __future__ import annotations

import numpy as np

from core.sparse_policy import densify, sparsify


def test_densify_inverts_sparsify_exactly() -> None:
    """densify(sparsify(p)) == p exactly, over many random sparse policies."""
    rng = np.random.default_rng(0)
    size = 17837
    for _ in range(50):
        dense = np.zeros(size, dtype=np.float32)
        k = int(rng.integers(1, 300))
        idx = rng.choice(size, size=k, replace=False)
        vals = rng.random(k).astype(np.float32)
        vals /= vals.sum()
        dense[idx] = vals
        indices, values = sparsify(dense)
        np.testing.assert_array_equal(densify(indices, values, size), dense)


def test_sparsify_dtypes_and_content() -> None:
    dense = np.array([0, 0.5, 0, 0.5, 0], dtype=np.float32)
    indices, values = sparsify(dense)
    assert indices.dtype == np.int32
    assert values.dtype == np.float32
    np.testing.assert_array_equal(indices, [1, 3])
    np.testing.assert_array_equal(values, [0.5, 0.5])


def test_all_zero_policy_round_trips() -> None:
    dense = np.zeros(10, dtype=np.float32)
    indices, values = sparsify(dense)
    assert len(indices) == 0
    assert len(values) == 0
    np.testing.assert_array_equal(densify(indices, values, 10), dense)


def test_memory_footprint_is_much_smaller() -> None:
    """A realistic sparse policy is dramatically smaller than dense."""
    size = 17837
    dense = np.zeros(size, dtype=np.float32)
    idx = np.arange(150)
    dense[idx] = 1.0 / 150
    indices, values = sparsify(dense)
    sparse_bytes = indices.nbytes + values.nbytes
    assert sparse_bytes < dense.nbytes / 20  # >20x smaller


def test_as_dense_accepts_sparse_and_dense() -> None:
    from core.sparse_policy import as_dense

    dense = np.array([0, 0.5, 0, 0.5, 0], dtype=np.float32)
    sparse = sparsify(dense)
    np.testing.assert_array_equal(as_dense(sparse, 5), dense)  # sparse tuple -> dense
    np.testing.assert_array_equal(as_dense(dense, 5), dense)   # already-dense -> passthrough
