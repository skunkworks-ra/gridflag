"""Tests for gridflag.coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from gridflag.coordinates import grid_shape, hermitian_fold, scale_uv, uv_to_cell


class TestScaleUV:
    def test_single_channel_at_ref_freq(self):
        """At the reference frequency, uv_lambda = uvw_metres * freq / c."""
        c = 299_792_458.0
        uvw = np.array([[100.0, 200.0, 50.0]])
        freq = np.array([1.0e9])
        ref_freq = 1.0e9
        u, v, w = scale_uv(uvw, freq, ref_freq)
        np.testing.assert_allclose(u[0, 0], 100.0 * 1e9 / c)
        np.testing.assert_allclose(v[0, 0], 200.0 * 1e9 / c)

    def test_frequency_scaling(self):
        """Higher frequency → larger uv in wavelengths."""
        c = 299_792_458.0
        uvw = np.array([[100.0, 0.0, 0.0]])
        freq = np.array([1.0e9, 2.0e9])
        ref_freq = 1.0e9
        u, v, w = scale_uv(uvw, freq, ref_freq)
        # u should double when freq doubles.
        np.testing.assert_allclose(u[0, 1] / u[0, 0], 2.0)

    def test_output_shapes(self):
        uvw = np.ones((10, 3))
        freq = np.linspace(1e9, 2e9, 4)
        u, v, w = scale_uv(uvw, freq, 1.5e9)
        assert u.shape == (10, 4)
        assert v.shape == (10, 4)


class TestHermitianFold:
    def test_positive_v_unchanged(self):
        u = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        vis = np.array([1 + 2j, 3 + 4j])
        u2, v2, vis2 = hermitian_fold(u, v, vis)
        np.testing.assert_array_equal(u2, u)
        np.testing.assert_array_equal(v2, v)
        np.testing.assert_array_equal(vis2, vis)

    def test_negative_v_folded(self):
        u = np.array([1.0])
        v = np.array([-3.0])
        vis = np.array([1 + 2j])
        u2, v2, vis2 = hermitian_fold(u, v, vis)
        assert u2[0] == -1.0
        assert v2[0] == 3.0
        np.testing.assert_allclose(vis2[0], 1 - 2j)

    def test_mixed(self):
        u = np.array([1.0, -2.0, 3.0])
        v = np.array([1.0, -1.0, 0.0])
        vis = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        u2, v2, vis2 = hermitian_fold(u, v, vis)
        assert v2[0] == 1.0  # unchanged
        assert v2[1] == 1.0  # folded
        assert u2[1] == 2.0  # flipped
        np.testing.assert_allclose(vis2[1], 2 - 2j)
        assert v2[2] == 0.0  # v=0, unchanged

    def test_does_not_mutate_input(self):
        u = np.array([1.0])
        v = np.array([-1.0])
        vis = np.array([1 + 1j])
        hermitian_fold(u, v, vis)
        assert v[0] == -1.0  # original unchanged


class TestUVToCell:
    def test_origin(self):
        u = np.array([0.0])
        v = np.array([0.0])
        cu, cv, N = uv_to_cell(u, v, cell_size=10.0)
        assert cu[0] == N
        assert cv[0] == 0

    def test_positive_uv(self):
        u = np.array([25.0])
        v = np.array([15.0])
        cu, cv, N = uv_to_cell(u, v, cell_size=10.0)
        # round(25/10) = 2, + N offset
        assert cu[0] == N + 2  # was 3, N = ceil(25/10) = 3
        assert cv[0] == 2  # round(15/10) = 2

    def test_negative_u(self):
        u = np.array([-15.0])
        v = np.array([5.0])
        cu, cv, N = uv_to_cell(u, v, cell_size=10.0)
        assert cu[0] == N - 2  # round(-15/10) = -2, + N


class TestGridShape:
    def test_basic(self):
        assert grid_shape(3) == (7, 4)
        assert grid_shape(0) == (1, 1)
