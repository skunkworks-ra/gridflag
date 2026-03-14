"""Tests for gridflag.zarr_store."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from gridflag.config import GridFlagConfig
from gridflag.zarr_store import AccumulatorGroup, ZarrStore, open_readonly


@pytest.fixture
def store(tmp_path):
    config = GridFlagConfig()
    return ZarrStore(tmp_path / "test.zarr", config, "/fake/test.ms")


# ── AccumulatorGroup ──────────────────────────────────────────────


class TestAccumulatorGroup:
    def test_empty_concatenate(self):
        """Empty accumulator returns empty arrays with correct dtypes."""
        acc = AccumulatorGroup()
        result = acc.concatenate()
        assert result["row_indices"].dtype == np.dtype("i8")
        assert result["values"].dtype == np.dtype("f4")
        assert len(result["cell_u"]) == 0

    def test_single_append(self):
        acc = AccumulatorGroup()
        acc.append(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            np.array([1.0, 2.0], dtype=np.float32),
        )
        result = acc.concatenate()
        assert len(result["values"]) == 2
        np.testing.assert_array_equal(result["values"], [1.0, 2.0])

    def test_multiple_appends_concatenated(self):
        acc = AccumulatorGroup()
        for i in range(3):
            acc.append(
                np.array([i], dtype=np.int64),
                np.array([0], dtype=np.int32),
                np.array([5], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([float(i)], dtype=np.float32),
            )
        result = acc.concatenate()
        assert len(result["values"]) == 3
        np.testing.assert_array_equal(result["values"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result["row_indices"], [0, 1, 2])


# ── ZarrStore ─────────────────────────────────────────────────────


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

    def test_root_attrs(self, store):
        """Root attrs should record ms_path and config."""
        assert store.root.attrs["ms_path"] == "/fake/test.ms"
        assert store.root.attrs["cell_size"] == 10.0
        config_rt = GridFlagConfig.from_json(store.root.attrs["config_json"])
        assert config_rt == GridFlagConfig()

    def test_flush_all(self, store):
        """flush_all should flush every (spw, corr) accumulator."""
        store.init_spw(0, n_chan=1, n_corr=2, ref_freq=1e9,
                       chan_freqs=np.array([1e9]))

        for corr in range(2):
            store.append(
                0, corr,
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int32),
                np.array([5], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([float(corr)], dtype=np.float32),
            )

        store.flush_all()
        # Both corrs should be readable now.
        for corr in range(2):
            flat = store.load_flat(0, corr)
            assert len(flat["values"]) == 1
            np.testing.assert_allclose(flat["values"][0], float(corr))

    def test_load_flat_triggers_flush(self, store):
        """load_flat on an unflushed accumulator should auto-flush."""
        store.init_spw(0, n_chan=1, n_corr=1, ref_freq=1e9,
                       chan_freqs=np.array([1e9]))
        store.append(
            0, 0,
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([5], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([7.0], dtype=np.float32),
        )
        # load_flat should flush implicitly.
        flat = store.load_flat(0, 0)
        np.testing.assert_allclose(flat["values"], [7.0])

    def test_multiple_spws(self, store):
        """Data for different SPWs should be independent."""
        for spw in [0, 1]:
            store.init_spw(spw, n_chan=1, n_corr=1, ref_freq=1e9,
                           chan_freqs=np.array([1e9]))
            store.append(
                spw, 0,
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int32),
                np.array([5], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([float(spw * 10)], dtype=np.float32),
            )

        store.flush_all()
        flat0 = store.load_flat(0, 0)
        flat1 = store.load_flat(1, 0)
        np.testing.assert_allclose(flat0["values"], [0.0])
        np.testing.assert_allclose(flat1["values"], [10.0])

    def test_empty_accumulator_flush(self, store):
        """Flushing an accumulator with no appends produces empty arrays."""
        store.init_spw(0, n_chan=1, n_corr=1, ref_freq=1e9,
                       chan_freqs=np.array([1e9]))
        store.flush_all()
        flat = store.load_flat(0, 0)
        assert len(flat["values"]) == 0

    def test_store_grid_overwrite(self, store):
        """Storing a grid twice should overwrite the first."""
        store.init_spw(0, n_chan=1, n_corr=1, ref_freq=1e9,
                       chan_freqs=np.array([1e9]))
        grid1 = np.ones((3, 2), dtype=np.float32)
        grid2 = np.full((3, 2), 99.0, dtype=np.float32)
        store.store_grid(0, 0, "test", grid1)
        store.store_grid(0, 0, "test", grid2)
        loaded = store.load_grid(0, 0, "test")
        np.testing.assert_array_equal(loaded, grid2)


# ── open_readonly ─────────────────────────────────────────────────


class TestOpenReadonly:
    def test_open_existing_store(self, store, tmp_path):
        """open_readonly should return a read-only group."""
        store.set_grid_shape((5, 3))
        store.init_spw(0, n_chan=1, n_corr=1, ref_freq=1e9,
                       chan_freqs=np.array([1e9]))
        store.store_grid(0, 0, "median_grid",
                         np.ones((5, 3), dtype=np.float32))

        root = open_readonly(tmp_path / "test.zarr")
        assert root.attrs["grid_shape"] == [5, 3]
        loaded = root["spw_0/corr_0/median_grid"][:]
        np.testing.assert_array_equal(loaded, np.ones((5, 3), dtype=np.float32))
