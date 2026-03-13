"""Flag assignment: compare per-visibility values against cell thresholds."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def flag_visibilities(
    cell_u: NDArray[np.intp],
    cell_v: NDArray[np.intp],
    values: NDArray[np.floating],
    threshold_grid: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Flag visibilities whose value exceeds the threshold for their cell.

    Parameters
    ----------
    cell_u, cell_v : 1-D int arrays, cell indices per visibility.
    values : 1-D float array, quantity per visibility.
    threshold_grid : 2-D float array (N_u, N_v).

    Returns
    -------
    flags : 1-D bool array, True = flagged.
    """
    thr = threshold_grid[cell_u, cell_v]
    # Flag if value exceeds threshold or threshold is NaN (empty cell — shouldn't
    # happen, but be safe).
    flags = values > thr
    flags |= np.isnan(thr)
    return flags
