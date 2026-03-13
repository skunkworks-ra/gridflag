"""Shared fixtures for gridflag tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)
