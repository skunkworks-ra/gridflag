"""Tests for gridflag.config."""

from __future__ import annotations

import pytest

from gridflag.config import GridFlagConfig


class TestGridFlagConfig:
    def test_defaults(self):
        c = GridFlagConfig()
        assert c.cell_size == 10.0
        assert c.nsigma == 3.0
        assert c.smoothing_window == 5
        assert c.data_column == "auto"
        assert c.quantity == "amplitude"
        assert c.zarr_path is None
        assert c.spw_ids is None
        assert c.field_ids is None
        assert c.uvrange is None

    def test_frozen(self):
        c = GridFlagConfig()
        with pytest.raises(AttributeError):
            c.nsigma = 5.0  # type: ignore[misc]

    def test_json_roundtrip_defaults(self):
        c = GridFlagConfig()
        c2 = GridFlagConfig.from_json(c.to_json())
        assert c == c2

    def test_json_roundtrip_custom(self):
        c = GridFlagConfig(
            cell_size=20.0,
            nsigma=5.0,
            annulus_widths=(1000.0, 2000.0),
            spw_ids=(0, 2),
            field_ids=(1,),
            uvrange=(100.0, 50000.0),
            zarr_path="/tmp/test.zarr",
        )
        c2 = GridFlagConfig.from_json(c.to_json())
        assert c == c2

    def test_json_roundtrip_none_fields(self):
        c = GridFlagConfig(spw_ids=None, field_ids=None, uvrange=None)
        c2 = GridFlagConfig.from_json(c.to_json())
        assert c2.spw_ids is None
        assert c2.field_ids is None
        assert c2.uvrange is None

    def test_json_tuples_preserved(self):
        """JSON serialises tuples as lists; from_json must restore tuples."""
        c = GridFlagConfig(annulus_widths=(100.0,))
        c2 = GridFlagConfig.from_json(c.to_json())
        assert isinstance(c2.annulus_widths, tuple)
        assert c2.annulus_widths == (100.0,)
