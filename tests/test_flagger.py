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
        assert flags[0] == False
        assert flags[1] == True
        assert flags[2] == False

    def test_nan_threshold_flags(self):
        """NaN thresholds should flag everything in that cell."""
        threshold = np.array([[np.nan]], dtype=np.float32)
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([1.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert flags[0] == True

    def test_equal_to_threshold(self):
        """Values equal to threshold should NOT be flagged."""
        threshold = np.array([[10.0]], dtype=np.float32)
        cell_u = np.array([0], dtype=np.intp)
        cell_v = np.array([0], dtype=np.intp)
        values = np.array([10.0], dtype=np.float32)

        flags = flag_visibilities(cell_u, cell_v, values, threshold)
        assert flags[0] == False
