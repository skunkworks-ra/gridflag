"""Tests for gridflag.plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridflag.config import GridFlagConfig
from gridflag.gridder import compute_cell_stats
from gridflag.plotting import plot_before_after
from gridflag.zarr_store import ZarrStore


@pytest.fixture
def populated_store(tmp_path):
    """Create a ZarrStore with synthetic data and pre-computed grids."""
    config = GridFlagConfig(cell_size=10.0)
    store = ZarrStore(tmp_path / "test.zarr", config, "/fake/test.ms")

    N = 5
    gshape = (2 * N + 1, N + 1)
    store.set_grid_shape(gshape)
    store.init_spw(0, n_chan=1, n_corr=1, ref_freq=1e9,
                   chan_freqs=np.array([1e9]))

    rng = np.random.default_rng(42)

    # Generate synthetic flat arrays: ~200 visibilities scattered across cells.
    n_vis = 200
    cell_u = rng.integers(0, gshape[0], size=n_vis).astype(np.int32)
    cell_v = rng.integers(0, gshape[1], size=n_vis).astype(np.int32)
    values = rng.exponential(1.0, size=n_vis).astype(np.float32)
    # Inject a few outliers.
    values[:10] = 50.0

    store.append(
        0, 0,
        np.arange(n_vis, dtype=np.int64),
        np.zeros(n_vis, dtype=np.int32),
        cell_u, cell_v, values,
    )

    # Compute and store "before" grids.
    median_grid, std_grid, count_grid = compute_cell_stats(
        cell_u.astype(np.intp), cell_v.astype(np.intp), values, gshape
    )
    store.store_grid(0, 0, "median_grid", median_grid)
    store.store_grid(0, 0, "std_grid", std_grid)
    store.store_grid(0, 0, "count_grid", count_grid)

    # Flag the first 10 (the injected outliers).
    flag_mask = np.zeros(n_vis, dtype=np.uint8)
    flag_mask[:10] = 1
    store.store_grid(0, 0, "flag_mask", flag_mask)

    return store, N


class TestPlotBeforeAfter:
    def test_produces_two_pngs(self, populated_store, tmp_path):
        store, N = populated_store
        output_dir = tmp_path / "plots"

        paths = plot_before_after(store, 0, 0, 10.0, N, output_dir)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0

    def test_filenames(self, populated_store, tmp_path):
        store, N = populated_store
        output_dir = tmp_path / "plots"

        paths = plot_before_after(store, 0, 0, 10.0, N, output_dir)
        names = {p.name for p in paths}
        assert "spw0_corr0_median.png" in names
        assert "spw0_corr0_std.png" in names
