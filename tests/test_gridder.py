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
