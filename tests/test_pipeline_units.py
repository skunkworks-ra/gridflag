"""Unit tests for pipeline internals that don't require an MS."""

from __future__ import annotations

import numpy as np
import pytest

from gridflag.pipeline import _extract_quantity


class TestExtractQuantity:
    def test_amplitude(self):
        data = np.array([3 + 4j, -1 + 0j], dtype=np.complex64)
        result = _extract_quantity(data, "amplitude")
        np.testing.assert_allclose(result, [5.0, 1.0])
        assert result.dtype == np.float32

    def test_phase(self):
        data = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex64)
        result = _extract_quantity(data, "phase")
        np.testing.assert_allclose(result, [0.0, np.pi / 2, np.pi], atol=1e-6)
        assert result.dtype == np.float32

    def test_real(self):
        data = np.array([3 + 4j, -2 + 7j], dtype=np.complex64)
        result = _extract_quantity(data, "real")
        np.testing.assert_allclose(result, [3.0, -2.0])

    def test_imag(self):
        data = np.array([3 + 4j, -2 + 7j], dtype=np.complex64)
        result = _extract_quantity(data, "imag")
        np.testing.assert_allclose(result, [4.0, 7.0])

    def test_invalid_raises(self):
        data = np.array([1 + 1j])
        with pytest.raises(ValueError, match="Unknown quantity"):
            _extract_quantity(data, "power")

    def test_zero_values(self):
        data = np.array([0 + 0j], dtype=np.complex64)
        assert _extract_quantity(data, "amplitude")[0] == 0.0
        assert _extract_quantity(data, "phase")[0] == 0.0
