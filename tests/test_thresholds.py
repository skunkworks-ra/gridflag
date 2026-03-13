"""Tests for gridflag.thresholds."""

from __future__ import annotations

import numpy as np

from gridflag.thresholds import (
    annular_threshold,
    combine_thresholds,
    local_neighborhood_threshold,
    neighbor_count,
)


class TestLocalNeighborhoodThreshold:
    def test_uniform_grid(self):
        """Uniform grid should produce threshold = median + nsigma * std."""
        shape = (11, 6)
        median_grid = np.full(shape, 5.0, dtype=np.float32)
        std_grid = np.full(shape, 1.0, dtype=np.float32)
        count_grid = np.full(shape, 100, dtype=np.int32)

        thr = local_neighborhood_threshold(median_grid, std_grid, count_grid,
                                           nsigma=3.0, kernel_size=3)
        # Interior cells should get threshold ≈ 5 + 3*1 = 8.
        np.testing.assert_allclose(thr[5, 3], 8.0, atol=0.01)

    def test_empty_cells_nan(self):
        """Empty cells should have NaN threshold."""
        shape = (5, 3)
        median_grid = np.zeros(shape, dtype=np.float32)
        std_grid = np.zeros(shape, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)

        thr = local_neighborhood_threshold(median_grid, std_grid, count_grid,
                                           nsigma=3.0, kernel_size=3)
        assert np.all(np.isnan(thr))


class TestNeighborCount:
    def test_fully_occupied(self):
        """All cells occupied → each interior cell has K²-1 neighbors."""
        count_grid = np.ones((7, 4), dtype=np.int32)
        n = neighbor_count(count_grid, kernel_size=3)
        # Interior cell should have 8 neighbors.
        assert n[3, 2] == 8

    def test_sparse(self):
        count_grid = np.zeros((5, 5), dtype=np.int32)
        count_grid[2, 2] = 1
        n = neighbor_count(count_grid, kernel_size=3)
        # The lone cell has 0 occupied neighbors.
        assert n[2, 2] == 0


class TestAnnularThreshold:
    def test_single_annulus(self):
        """Single wide annulus covering everything."""
        N = 5
        shape = (2 * N + 1, N + 1)
        median_grid = np.full(shape, 10.0, dtype=np.float32)
        std_grid = np.full(shape, 2.0, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32)

        thr = annular_threshold(
            median_grid, std_grid, count_grid,
            cell_size=1.0, annulus_widths=(1000.0,), nsigma=3.0, N=N,
        )
        # All cells should get 10 + 3*2 = 16.
        occupied = count_grid > 0
        np.testing.assert_allclose(thr[occupied], 16.0, atol=0.01)


class TestCombineThresholds:
    def test_takes_minimum(self):
        """Should take the lower of local and annular."""
        shape = (5, 3)
        local = np.full(shape, 10.0, dtype=np.float32)
        annular = np.full(shape, 8.0, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32) * 10

        combined = combine_thresholds(local, annular, count_grid,
                                      kernel_size=3, min_neighbors=1)
        np.testing.assert_allclose(combined, 8.0)

    def test_sparse_uses_annular(self):
        """Sparse cells should use annular only."""
        shape = (5, 3)
        local = np.full(shape, 10.0, dtype=np.float32)
        annular = np.full(shape, 12.0, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)
        count_grid[2, 1] = 1  # single occupied cell

        combined = combine_thresholds(local, annular, count_grid,
                                      kernel_size=3, min_neighbors=1)
        # The lone cell has 0 neighbors < min_neighbors=1 → annular.
        np.testing.assert_allclose(combined[2, 1], 12.0)
