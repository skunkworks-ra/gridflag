"""Tests for gridflag.gridder."""

from __future__ import annotations

import numpy as np

from gridflag.gridder import compute_cell_stats


class TestComputeCellStats:
    def test_single_cell(self):
        """All data in one cell."""
        cell_u = np.array([1, 1, 1, 1, 1], dtype=np.intp)
        cell_v = np.array([0, 0, 0, 0, 0], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        shape = (3, 2)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        assert cnt[1, 0] == 5
        np.testing.assert_allclose(med[1, 0], 3.0)
        # MAD of [1,2,3,4,5] around 3 = median([2,1,0,1,2]) = 1
        np.testing.assert_allclose(std[1, 0], 1.4826, rtol=1e-3)

    def test_empty_cells(self):
        """Cells with no data should have zero count."""
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([42.0], dtype=np.float32)
        shape = (3, 3)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        assert cnt[0, 0] == 1
        assert cnt[1, 1] == 0
        assert cnt[2, 2] == 0

    def test_multiple_cells(self):
        """Two cells with different statistics."""
        cell_u = np.array([0, 0, 0, 1, 1], dtype=np.intp)
        cell_v = np.array([0, 0, 0, 1, 1], dtype=np.intp)
        values = np.array([10.0, 10.0, 10.0, 100.0, 200.0], dtype=np.float32)
        shape = (2, 2)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        assert cnt[0, 0] == 3
        assert cnt[1, 1] == 2
        np.testing.assert_allclose(med[0, 0], 10.0)
        np.testing.assert_allclose(med[1, 1], 150.0)

    def test_output_shapes(self):
        cell_u = np.array([0, 1, 2], dtype=np.intp)
        cell_v = np.array([0, 0, 0], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        shape = (5, 3)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        assert med.shape == shape
        assert std.shape == shape
        assert cnt.shape == shape

    def test_single_visibility_per_cell(self):
        """One visibility → median=value, MAD=0, std=0."""
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([7.5], dtype=np.float32)
        shape = (1, 1)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        np.testing.assert_allclose(med[0, 0], 7.5)
        np.testing.assert_allclose(std[0, 0], 0.0)
        assert cnt[0, 0] == 1

    def test_two_visibilities(self):
        """Two values → median is average, MAD is half-difference."""
        cell_u = np.array([0, 0], dtype=np.intp)
        cell_v = np.array([0, 0], dtype=np.intp)
        values = np.array([10.0, 20.0], dtype=np.float32)
        shape = (1, 1)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        np.testing.assert_allclose(med[0, 0], 15.0)
        # MAD = median(|10-15|, |20-15|) = median(5, 5) = 5
        np.testing.assert_allclose(std[0, 0], 1.4826 * 5.0, rtol=1e-3)

    def test_outlier_robustness(self):
        """MAD should be robust to a single large outlier."""
        cell_u = np.zeros(11, dtype=np.intp)
        cell_v = np.zeros(11, dtype=np.intp)
        values = np.array([1.0] * 10 + [1000.0], dtype=np.float32)
        shape = (1, 1)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        np.testing.assert_allclose(med[0, 0], 1.0)
        # MAD = median of [0]*10 + [999] = 0
        np.testing.assert_allclose(std[0, 0], 0.0)

    def test_negative_values(self):
        """Should handle negative values (e.g. from phase or real part)."""
        cell_u = np.array([0, 0, 0], dtype=np.intp)
        cell_v = np.array([0, 0, 0], dtype=np.intp)
        values = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        shape = (1, 1)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        np.testing.assert_allclose(med[0, 0], 0.0)
        # MAD = median(5, 0, 5) = 5
        np.testing.assert_allclose(std[0, 0], 1.4826 * 5.0, rtol=1e-3)

    def test_output_dtypes(self):
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([1.0], dtype=np.float32)
        shape = (1, 1)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        assert med.dtype == np.float32
        assert std.dtype == np.float32
        assert cnt.dtype == np.int32

    def test_constant_values(self):
        """All identical values → median=value, std=0."""
        cell_u = np.zeros(20, dtype=np.intp)
        cell_v = np.zeros(20, dtype=np.intp)
        values = np.full(20, 42.0, dtype=np.float32)
        shape = (1, 1)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        np.testing.assert_allclose(med[0, 0], 42.0)
        np.testing.assert_allclose(std[0, 0], 0.0)
        assert cnt[0, 0] == 20

    def test_large_random_dataset(self, rng):
        """Stress test with many visibilities across many cells."""
        shape = (51, 26)
        n = 10_000
        cell_u = rng.integers(0, shape[0], size=n).astype(np.intp)
        cell_v = rng.integers(0, shape[1], size=n).astype(np.intp)
        values = rng.exponential(1.0, size=n).astype(np.float32)

        med, std, cnt = compute_cell_stats(cell_u, cell_v, values, shape)
        assert med.shape == shape
        assert std.shape == shape
        assert cnt.shape == shape
        # Total count across all cells should equal n.
        assert cnt.sum() == n
        # No negative std.
        assert np.all(std >= 0)

    def test_even_count_median(self):
        """Even number of values: median should be average of middle two."""
        cell_u = np.array([0, 0, 0, 0], dtype=np.intp)
        cell_v = np.array([0, 0, 0, 0], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        shape = (1, 1)

        med, _, _ = compute_cell_stats(cell_u, cell_v, values, shape)
        np.testing.assert_allclose(med[0, 0], 2.5)
