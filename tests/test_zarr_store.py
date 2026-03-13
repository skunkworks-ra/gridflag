"""Tests for gridflag.zarr_store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gridflag.config import GridFlagConfig
from gridflag.zarr_store import ZarrStore


@pytest.fixture
def store(tmp_path):
    config = GridFlagConfig()
    return ZarrStore(tmp_path / "test.zarr", config, "/fake/test.ms")


class TestZarrStore:
    def test_init_and_append(self, store):
        store.init_spw(0, n_chan=4, n_corr=2, ref_freq=1e9,
                       chan_freqs=np.linspace(1e9, 1.1e9, 4))

        row = np.array([0, 0, 1, 1], dtype=np.int64)
        chan = np.array([0, 1, 0, 1], dtype=np.int32)
        cu = np.array([5, 5, 6, 6], dtype=np.int32)
        cv = np.array([0, 0, 1, 1], dtype=np.int32)
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        store.append(0, 0, row, chan, cu, cv, vals)
        store.append(0, 0, row, chan, cu, cv, vals)  # second batch

        flat = store.load_flat(0, 0)
        assert len(flat["values"]) == 8
        np.testing.assert_array_equal(flat["values"][:4], vals)

    def test_store_and_load_grid(self, store):
        store.init_spw(0, n_chan=1, n_corr=1, ref_freq=1e9,
                       chan_freqs=np.array([1e9]))

        grid = np.random.rand(5, 3).astype(np.float32)
        store.store_grid(0, 0, "median_grid", grid)
        loaded = store.load_grid(0, 0, "median_grid")
        np.testing.assert_array_equal(loaded, grid)

    def test_grid_shape(self, store):
        store.set_grid_shape((11, 6))
        assert store.get_grid_shape() == (11, 6)

    def test_spw_attrs(self, store):
        store.init_spw(2, n_chan=8, n_corr=4, ref_freq=3e9,
                       chan_freqs=np.ones(8) * 3e9)
        attrs = store.spw_attrs(2)
        assert attrs["n_chan"] == 8
        assert attrs["n_corr"] == 4
