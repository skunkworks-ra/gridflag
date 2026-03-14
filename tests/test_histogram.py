"""Tests for gridflag.histogram — streaming two-pass statistics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

from gridflag.gridder import compute_cell_stats
from gridflag.histogram import (
    _EXACT_THRESHOLD,
    compute_cell_stats_streaming,
    pass1_ranges,
)


def _write_shard(
    shard_dir: Path,
    name: str,
    spw_id: int,
    corr_id: int,
    cell_u: np.ndarray,
    cell_v: np.ndarray,
    values: np.ndarray,
    row_indices: np.ndarray | None = None,
    chan_indices: np.ndarray | None = None,
) -> str:
    """Write a minimal zarr shard for testing."""
    shard_path = str(shard_dir / f"{name}.zarr")
    root = zarr.open(shard_path, mode="w")
    grp = root.require_group(f"spw_{spw_id}/corr_{corr_id}")
    grp.array("cell_u", cell_u.astype(np.int32), overwrite=True)
    grp.array("cell_v", cell_v.astype(np.int32), overwrite=True)
    grp.array("values", values.astype(np.float32), overwrite=True)
    if row_indices is None:
        row_indices = np.arange(len(values), dtype=np.int64)
    if chan_indices is None:
        chan_indices = np.zeros(len(values), dtype=np.int32)
    grp.array("row_indices", row_indices, overwrite=True)
    grp.array("chan_indices", chan_indices, overwrite=True)
    return shard_path


class TestPass1Ranges:
    def test_single_shard(self, rng):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cell_u = np.array([0, 0, 1], dtype=np.int32)
            cell_v = np.array([0, 0, 1], dtype=np.int32)
            values = np.array([1.0, 5.0, 3.0], dtype=np.float32)
            sp = _write_shard(td_path, "s0", 0, 0, cell_u, cell_v, values)

            grid_shape = (2, 2)
            cmin, cmax, ccount = pass1_ranges(
                [sp], "spw_0", "corr_0", grid_shape, n_threads=1
            )
            assert ccount[0 * 2 + 0] == 2  # cell (0,0)
            assert ccount[1 * 2 + 1] == 1  # cell (1,1)
            assert cmin[0] == 1.0
            assert cmax[0] == 5.0

    def test_two_shards_reduce(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sp1 = _write_shard(
                td_path, "s0", 0, 0,
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([10.0], dtype=np.float32),
            )
            sp2 = _write_shard(
                td_path, "s1", 0, 0,
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([2.0], dtype=np.float32),
            )

            cmin, cmax, ccount = pass1_ranges(
                [sp1, sp2], "spw_0", "corr_0", (1, 1), n_threads=2
            )
            assert ccount[0] == 2
            assert cmin[0] == 2.0
            assert cmax[0] == 10.0

    def test_missing_group_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sp = _write_shard(
                td_path, "s0", 0, 0,
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([1.0], dtype=np.float32),
            )
            # Ask for a different (spw, corr) → should return zeros.
            _, _, ccount = pass1_ranges(
                [sp], "spw_1", "corr_0", (1, 1), n_threads=1
            )
            assert ccount[0] == 0


class TestSingleCellAccuracy:
    """Validate histogram stats against exact compute_cell_stats for single cell."""

    def test_known_values(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
            cell_u = np.zeros(5, dtype=np.int32)
            cell_v = np.zeros(5, dtype=np.int32)
            sp = _write_shard(td_path, "s0", 0, 0, cell_u, cell_v, values)

            grid_shape = (1, 1)
            # These are low-count (<=32), so exact fallback kicks in.
            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", grid_shape, n_bins=256, n_threads=1
            )
            assert cnt[0, 0] == 5
            np.testing.assert_allclose(med[0, 0], 3.0)
            np.testing.assert_allclose(std[0, 0], 1.4826, rtol=1e-3)

    def test_constant_values(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            values = np.full(20, 42.0, dtype=np.float32)
            cell_u = np.zeros(20, dtype=np.int32)
            cell_v = np.zeros(20, dtype=np.int32)
            sp = _write_shard(td_path, "s0", 0, 0, cell_u, cell_v, values)

            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )
            np.testing.assert_allclose(med[0, 0], 42.0)
            np.testing.assert_allclose(std[0, 0], 0.0)

    def test_two_values(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            values = np.array([10.0, 20.0], dtype=np.float32)
            cell_u = np.zeros(2, dtype=np.int32)
            cell_v = np.zeros(2, dtype=np.int32)
            sp = _write_shard(td_path, "s0", 0, 0, cell_u, cell_v, values)

            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )
            np.testing.assert_allclose(med[0, 0], 15.0)
            np.testing.assert_allclose(std[0, 0], 1.4826 * 5.0, rtol=1e-3)


class TestMultiShardAccuracy:
    """Split data across shards, verify results match single-shard."""

    def test_split_across_two_shards(self, rng):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            n = 30  # low count → exact fallback
            cell_u = np.zeros(n, dtype=np.int32)
            cell_v = np.zeros(n, dtype=np.int32)
            values = rng.exponential(1.0, size=n).astype(np.float32)

            # Single shard.
            sp_all = _write_shard(td_path, "all", 0, 0, cell_u, cell_v, values)
            med_all, std_all, cnt_all = compute_cell_stats_streaming(
                [sp_all], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )

            # Split into two shards.
            mid = n // 2
            sp1 = _write_shard(td_path, "s0", 0, 0, cell_u[:mid], cell_v[:mid], values[:mid])
            sp2 = _write_shard(td_path, "s1", 0, 0, cell_u[mid:], cell_v[mid:], values[mid:])
            med_split, std_split, cnt_split = compute_cell_stats_streaming(
                [sp1, sp2], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=2
            )

            assert cnt_all[0, 0] == cnt_split[0, 0]
            np.testing.assert_allclose(med_all[0, 0], med_split[0, 0], rtol=1e-5)
            np.testing.assert_allclose(std_all[0, 0], std_split[0, 0], rtol=1e-5)


class TestHistogramVsExact:
    """Validate histogram-based stats agree with exact compute_cell_stats."""

    def test_large_random_single_cell(self, rng):
        """Many values in one cell — uses histogram path, not exact fallback."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            n = 500  # well above _EXACT_THRESHOLD
            values = rng.normal(10.0, 2.0, size=n).astype(np.float32)
            cell_u = np.zeros(n, dtype=np.intp)
            cell_v = np.zeros(n, dtype=np.intp)

            # Exact reference.
            med_exact, std_exact, cnt_exact = compute_cell_stats(
                cell_u, cell_v, values, (1, 1)
            )

            # Histogram-based.
            sp = _write_shard(td_path, "s0", 0, 0, cell_u.astype(np.int32), cell_v.astype(np.int32), values)
            med_hist, std_hist, cnt_hist = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )

            assert cnt_exact[0, 0] == cnt_hist[0, 0]
            np.testing.assert_allclose(med_hist[0, 0], med_exact[0, 0], rtol=0.05)
            np.testing.assert_allclose(std_hist[0, 0], std_exact[0, 0], rtol=0.05)

    def test_multi_cell_random(self, rng):
        """Multiple cells with Gaussian data (IQR/1.349 ≈ 1.4826*MAD for Gaussian)."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            grid_shape = (5, 5)
            n = 5000
            cell_u = rng.integers(0, grid_shape[0], size=n).astype(np.intp)
            cell_v = rng.integers(0, grid_shape[1], size=n).astype(np.intp)
            values = rng.normal(10.0, 2.0, size=n).astype(np.float32)

            # Exact reference.
            med_exact, std_exact, cnt_exact = compute_cell_stats(
                cell_u, cell_v, values, grid_shape
            )

            # Histogram-based (split across 3 shards).
            chunk = n // 3
            shards = []
            for i in range(3):
                s = i * chunk
                e = n if i == 2 else (i + 1) * chunk
                sp = _write_shard(
                    td_path, f"s{i}", 0, 0,
                    cell_u[s:e].astype(np.int32),
                    cell_v[s:e].astype(np.int32),
                    values[s:e],
                )
                shards.append(sp)

            med_hist, std_hist, cnt_hist = compute_cell_stats_streaming(
                shards, "spw_0", "corr_0", grid_shape, n_bins=256, n_threads=2
            )

            np.testing.assert_array_equal(cnt_hist, cnt_exact)

            # Check occupied cells only.
            occupied = cnt_exact > 0
            np.testing.assert_allclose(
                med_hist[occupied], med_exact[occupied], rtol=0.05,
                err_msg="Median mismatch",
            )
            # Std can be 0 for cells with constant values, skip those.
            nonzero_std = occupied & (std_exact > 0)
            if np.any(nonzero_std):
                # rtol=0.08: histogram binning + IQR vs MAD estimator difference.
                np.testing.assert_allclose(
                    std_hist[nonzero_std], std_exact[nonzero_std], rtol=0.08,
                    err_msg="Robust std mismatch",
                )

    def test_uniform_distribution(self, rng):
        """Uniform distribution — histogram should do well."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            n = 1000
            values = rng.uniform(0.0, 100.0, size=n).astype(np.float32)
            cell_u = np.zeros(n, dtype=np.intp)
            cell_v = np.zeros(n, dtype=np.intp)

            med_exact, std_exact, _ = compute_cell_stats(
                cell_u, cell_v, values, (1, 1)
            )

            sp = _write_shard(td_path, "s0", 0, 0, cell_u.astype(np.int32), cell_v.astype(np.int32), values)
            med_hist, std_hist, _ = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )

            np.testing.assert_allclose(med_hist[0, 0], med_exact[0, 0], rtol=0.05)
            np.testing.assert_allclose(std_hist[0, 0], std_exact[0, 0], rtol=0.05)


class TestEdgeCases:
    def test_empty_shards(self):
        """No data at all → zero grids."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Create shard with a different (spw, corr) than what we query.
            sp = _write_shard(
                td_path, "s0", 0, 0,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
            )
            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (3, 3), n_bins=256, n_threads=1
            )
            assert cnt.sum() == 0
            assert med.shape == (3, 3)

    def test_single_value_per_cell(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sp = _write_shard(
                td_path, "s0", 0, 0,
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([7.5], dtype=np.float32),
            )
            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )
            np.testing.assert_allclose(med[0, 0], 7.5)
            np.testing.assert_allclose(std[0, 0], 0.0)
            assert cnt[0, 0] == 1

    def test_all_identical_large_count(self, rng):
        """All identical values with count > EXACT_THRESHOLD → histogram path."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            n = _EXACT_THRESHOLD + 10
            values = np.full(n, 42.0, dtype=np.float32)
            cell_u = np.zeros(n, dtype=np.int32)
            cell_v = np.zeros(n, dtype=np.int32)
            sp = _write_shard(td_path, "s0", 0, 0, cell_u, cell_v, values)

            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", (1, 1), n_bins=256, n_threads=1
            )
            np.testing.assert_allclose(med[0, 0], 42.0, rtol=0.01)
            # Std should be ~0 (all values in one bin).
            assert std[0, 0] < 0.5

    def test_output_shapes_and_dtypes(self, rng):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            grid_shape = (5, 3)
            sp = _write_shard(
                td_path, "s0", 0, 0,
                np.array([0, 1, 2], dtype=np.int32),
                np.array([0, 0, 0], dtype=np.int32),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
            med, std, cnt = compute_cell_stats_streaming(
                [sp], "spw_0", "corr_0", grid_shape, n_bins=256, n_threads=1
            )
            assert med.shape == grid_shape
            assert std.shape == grid_shape
            assert cnt.shape == grid_shape
            assert med.dtype == np.float32
            assert std.dtype == np.float32
            assert cnt.dtype == np.int32
