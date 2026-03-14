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

        thr = local_neighborhood_threshold(
            median_grid, std_grid, count_grid, nsigma=3.0, kernel_size=3
        )
        # Interior cells should get threshold ≈ 5 + 3*1 = 8.
        np.testing.assert_allclose(thr[5, 3], 8.0, atol=0.01)

    def test_empty_cells_nan(self):
        """Empty cells should have NaN threshold."""
        shape = (5, 3)
        median_grid = np.zeros(shape, dtype=np.float32)
        std_grid = np.zeros(shape, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)

        thr = local_neighborhood_threshold(
            median_grid, std_grid, count_grid, nsigma=3.0, kernel_size=3
        )
        assert np.all(np.isnan(thr))

    def test_nsigma_zero(self):
        """nsigma=0 → threshold = smoothed median."""
        shape = (7, 4)
        median_grid = np.full(shape, 10.0, dtype=np.float32)
        std_grid = np.full(shape, 5.0, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32)

        thr = local_neighborhood_threshold(
            median_grid, std_grid, count_grid, nsigma=0.0, kernel_size=3
        )
        # Interior should be ~10.0 (median, no sigma contribution).
        np.testing.assert_allclose(thr[3, 2], 10.0, atol=0.5)

    def test_kernel_size_1(self):
        """K=1 → no smoothing, per-cell threshold."""
        shape = (3, 3)
        median_grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        std_grid = np.ones(shape, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32)

        thr = local_neighborhood_threshold(
            median_grid, std_grid, count_grid, nsigma=1.0, kernel_size=1
        )
        # Each cell: median + 1*std = median + 1.
        expected = median_grid + 1.0
        np.testing.assert_allclose(thr, expected, atol=1e-5)

    def test_partial_occupancy(self):
        """Only some cells occupied — empty cells should be NaN."""
        shape = (5, 3)
        median_grid = np.zeros(shape, dtype=np.float32)
        std_grid = np.zeros(shape, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)
        # Populate only centre.
        median_grid[2, 1] = 10.0
        std_grid[2, 1] = 2.0
        count_grid[2, 1] = 50

        thr = local_neighborhood_threshold(
            median_grid, std_grid, count_grid, nsigma=3.0, kernel_size=3
        )
        assert not np.isnan(thr[2, 1])  # occupied cell has a threshold
        assert np.isnan(thr[0, 0])  # far empty cell


class TestNeighborCount:
    def test_fully_occupied(self):
        """All cells occupied → each interior cell has K²-1 neighbors."""
        count_grid = np.ones((7, 4), dtype=np.int32)
        n = neighbor_count(count_grid, kernel_size=3)
        assert n[3, 2] == 8

    def test_sparse(self):
        count_grid = np.zeros((5, 5), dtype=np.int32)
        count_grid[2, 2] = 1
        n = neighbor_count(count_grid, kernel_size=3)
        assert n[2, 2] == 0

    def test_all_empty(self):
        count_grid = np.zeros((5, 5), dtype=np.int32)
        n = neighbor_count(count_grid, kernel_size=3)
        assert np.all(n == 0)

    def test_corner_cell(self):
        """Corner cells have fewer potential neighbors."""
        count_grid = np.ones((5, 5), dtype=np.int32)
        n = neighbor_count(count_grid, kernel_size=3)
        # Corner (0,0): uniform_filter with mode='constant' pads zeros.
        # Of 9 positions, only 4 are inside → 4 occupied - 1 self = 3.
        assert n[0, 0] == 3

    def test_kernel_size_5(self):
        """K=5 → 24 potential neighbors for interior cells."""
        count_grid = np.ones((9, 9), dtype=np.int32)
        n = neighbor_count(count_grid, kernel_size=5)
        assert n[4, 4] == 24  # 25 - 1


class TestAnnularThreshold:
    def test_single_annulus(self):
        """Single wide annulus covering everything."""
        N = 5
        shape = (2 * N + 1, N + 1)
        median_grid = np.full(shape, 10.0, dtype=np.float32)
        std_grid = np.full(shape, 2.0, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32)

        thr = annular_threshold(
            median_grid,
            std_grid,
            count_grid,
            cell_size=1.0,
            annulus_widths=(1000.0,),
            nsigma=3.0,
            N=N,
        )
        occupied = count_grid > 0
        np.testing.assert_allclose(thr[occupied], 16.0, atol=0.01)

    def test_multiple_annuli(self):
        """Different annuli can produce different thresholds."""
        N = 10
        shape = (2 * N + 1, N + 1)
        median_grid = np.zeros(shape, dtype=np.float32)
        std_grid = np.zeros(shape, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)

        # Inner annulus: high values.
        median_grid[N, 0] = 100.0
        std_grid[N, 0] = 10.0
        count_grid[N, 0] = 1
        # Outer cell.
        median_grid[N, N] = 1.0
        std_grid[N, N] = 0.1
        count_grid[N, N] = 1

        thr = annular_threshold(
            median_grid,
            std_grid,
            count_grid,
            cell_size=1.0,
            annulus_widths=(5.0, 100.0),
            nsigma=3.0,
            N=N,
        )
        # The two cells are in different annuli → different thresholds.
        assert thr[N, 0] != thr[N, N]

    def test_empty_grid(self):
        """All-empty grid → all NaN thresholds."""
        N = 3
        shape = (2 * N + 1, N + 1)
        thr = annular_threshold(
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.int32),
            cell_size=1.0,
            annulus_widths=(10.0,),
            nsigma=3.0,
            N=N,
        )
        assert np.all(np.isnan(thr))


class TestCombineThresholds:
    def test_takes_minimum(self):
        """Should take the lower of local and annular."""
        shape = (5, 3)
        local = np.full(shape, 10.0, dtype=np.float32)
        annular = np.full(shape, 8.0, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32) * 10

        combined = combine_thresholds(local, annular, count_grid, kernel_size=3, min_neighbors=1)
        np.testing.assert_allclose(combined, 8.0)

    def test_sparse_uses_annular(self):
        """Sparse cells should use annular only."""
        shape = (5, 3)
        local = np.full(shape, 10.0, dtype=np.float32)
        annular = np.full(shape, 12.0, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)
        count_grid[2, 1] = 1  # single occupied cell

        combined = combine_thresholds(local, annular, count_grid, kernel_size=3, min_neighbors=1)
        np.testing.assert_allclose(combined[2, 1], 12.0)

    def test_min_neighbors_zero(self):
        """min_neighbors=0 → always use min(local, annular)."""
        shape = (5, 3)
        local = np.full(shape, 5.0, dtype=np.float32)
        annular = np.full(shape, 8.0, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)
        count_grid[2, 1] = 1

        combined = combine_thresholds(local, annular, count_grid, kernel_size=3, min_neighbors=0)
        # 0 neighbors >= min_neighbors=0, so use min(5, 8) = 5.
        np.testing.assert_allclose(combined[2, 1], 5.0)

    def test_empty_cells_nan(self):
        """Empty cells should remain NaN regardless."""
        shape = (3, 3)
        local = np.full(shape, 5.0, dtype=np.float32)
        annular = np.full(shape, 8.0, dtype=np.float32)
        count_grid = np.zeros(shape, dtype=np.int32)

        combined = combine_thresholds(local, annular, count_grid, kernel_size=3, min_neighbors=1)
        assert np.all(np.isnan(combined))

    def test_nan_local_uses_annular(self):
        """When local_thr is NaN, fmin should fall through to annular."""
        shape = (5, 3)
        local = np.full(shape, np.nan, dtype=np.float32)
        annular = np.full(shape, 7.0, dtype=np.float32)
        count_grid = np.ones(shape, dtype=np.int32) * 10

        combined = combine_thresholds(local, annular, count_grid, kernel_size=3, min_neighbors=1)
        # Interior cells should use annular (fmin ignores NaN).
        np.testing.assert_allclose(combined[2, 1], 7.0)
