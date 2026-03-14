"""Per-cell robust statistics (median, MAD) via segmented reduction."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def _segmented_median_mad(
    vals_sorted: np.ndarray,
    seg_starts: np.ndarray,
    seg_counts: np.ndarray,
    unique_cells: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median and 1.4826*MAD per segment, scatter into flat grids.

    All arrays are 1-D.  Runs under numba for JIT speed.
    """
    median_flat = np.zeros(n_cells, dtype=np.float32)
    std_flat = np.zeros(n_cells, dtype=np.float32)
    count_flat = np.zeros(n_cells, dtype=np.int32)

    for i in range(len(seg_starts)):
        s = seg_starts[i]
        n = seg_counts[i]
        cell = unique_cells[i]
        seg = vals_sorted[s : s + n].copy()
        seg.sort()

        # Median.
        if n % 2 == 1:
            med = seg[n // 2]
        else:
            med = (seg[n // 2 - 1] + seg[n // 2]) * 0.5

        # MAD = median(|x - median|).
        absdev = np.empty(n, dtype=np.float32)
        for j in range(n):
            absdev[j] = abs(seg[j] - med)
        absdev.sort()
        if n % 2 == 1:
            mad = absdev[n // 2]
        else:
            mad = (absdev[n // 2 - 1] + absdev[n // 2]) * 0.5

        median_flat[cell] = med
        std_flat[cell] = np.float32(1.4826) * mad
        count_flat[cell] = n

    return median_flat, std_flat, count_flat


def compute_cell_stats(
    cell_u: NDArray[np.intp],
    cell_v: NDArray[np.intp],
    values: NDArray[np.floating],
    grid_shape: tuple[int, int],
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute per-cell median and robust std from flat value arrays.

    Parameters
    ----------
    cell_u, cell_v : 1-D int arrays, cell indices for each visibility.
    values : 1-D float array, the quantity (amplitude, etc.) per visibility.
    grid_shape : (N_u, N_v) — the 2-D grid dimensions.

    Returns
    -------
    median_grid : (N_u, N_v) float32, per-cell median.
    std_grid : (N_u, N_v) float32, per-cell robust std (1.4826 * MAD).
    count_grid : (N_u, N_v) int32, per-cell visibility count.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    # Flatten 2D cell indices to 1D.
    flat_idx = cell_u * N_v + cell_v

    # Sort by cell index.
    order = flat_idx.argsort()
    flat_sorted = flat_idx[order]
    vals_sorted = np.ascontiguousarray(values[order], dtype=np.float32)

    # Segment boundaries.
    unique_cells, seg_starts, seg_counts = np.unique(
        flat_sorted, return_index=True, return_counts=True
    )

    # JIT-compiled per-segment statistics.
    median_flat, std_flat, count_flat = _segmented_median_mad(
        vals_sorted,
        seg_starts.astype(np.int64),
        seg_counts.astype(np.int64),
        unique_cells.astype(np.int64),
        n_cells,
    )

    return (
        median_flat.reshape(grid_shape),
        std_flat.reshape(grid_shape),
        count_flat.reshape(grid_shape),
    )
