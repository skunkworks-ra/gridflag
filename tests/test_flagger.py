"""Tests for gridflag.flagger."""

from __future__ import annotations

import numpy as np

from gridflag.flagger import flag_visibilities


class TestFlagVisibilities:
    def test_above_threshold(self):
        threshold = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
        cell_u = np.array([0, 0, 1], dtype=np.intp)
        cell_v = np.array([0, 0, 1], dtype=np.intp)
        values = np.array([5.0, 15.0, 8.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert not flags[0]
        assert flags[1]
        assert not flags[2]

    def test_nan_threshold_flags(self):
        """NaN thresholds should flag everything in that cell."""
        threshold = np.array([[np.nan]], dtype=np.float32)
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([1.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert flags[0]

    def test_equal_to_threshold(self):
        """Values equal to threshold should NOT be flagged."""
        threshold = np.array([[10.0]], dtype=np.float32)
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([10.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert not flags[0]

    def test_all_below(self):
        """All values below threshold → no flags."""
        threshold = np.full((3, 3), 100.0, dtype=np.float32)
        cell_u = np.array([0, 1, 2], dtype=np.intp)
        cell_v = np.array([0, 1, 2], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert not np.any(flags)

    def test_all_above(self):
        """All values above threshold → all flagged."""
        threshold = np.full((2, 2), 1.0, dtype=np.float32)
        cell_u = np.array([0, 1, 0, 1], dtype=np.intp)
        cell_v = np.array([0, 0, 1, 1], dtype=np.intp)
        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert np.all(flags)

    def test_empty_input(self):
        threshold = np.full((3, 3), 10.0, dtype=np.float32)
        cell_u = np.array([], dtype=np.intp)
        cell_v = np.array([], dtype=np.intp)
        values = np.array([], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert len(flags) == 0

    def test_per_cell_thresholds(self):
        """Different cells with different thresholds."""
        threshold = np.array([[5.0, 100.0]], dtype=np.float32)
        cell_u = np.array([0, 0], dtype=np.intp)
        cell_v = np.array([0, 1], dtype=np.intp)
        values = np.array([10.0, 10.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert flags[0]  # 10 > 5
        assert not flags[1]  # 10 < 100

    def test_negative_values(self):
        """Negative values (e.g. from real/imag) should compare correctly."""
        threshold = np.array([[0.0]], dtype=np.float32)
        cell_u = np.array([0, 0], dtype=np.intp)
        cell_v = np.array([0, 0], dtype=np.intp)
        values = np.array([-5.0, 5.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert not flags[0]  # -5 < 0
        assert flags[1]  # 5 > 0

    def test_output_dtype(self):
        threshold = np.array([[10.0]], dtype=np.float32)
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([5.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert flags.dtype == np.bool_
