"""Per-cell robust statistics (median, MAD) via segmented reduction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    # Flatten 2D cell indices to 1D for sorting.
    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)

    # Sort by cell index.
    order = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    vals_sorted = values[order]

    # Find segment boundaries.
    unique_cells, seg_starts, seg_counts = np.unique(
        flat_sorted, return_index=True, return_counts=True
    )

    # Allocate output grids.
    median_grid = np.zeros(N_u * N_v, dtype=np.float32)
    std_grid = np.zeros(N_u * N_v, dtype=np.float32)
    count_grid = np.zeros(N_u * N_v, dtype=np.int32)

    for i, (cell, start, cnt) in enumerate(
        zip(unique_cells, seg_starts, seg_counts)
    ):
        seg = vals_sorted[start : start + cnt]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        median_grid[cell] = med
        std_grid[cell] = 1.4826 * mad
        count_grid[cell] = cnt

    return (
        median_grid.reshape(grid_shape),
        std_grid.reshape(grid_shape),
        count_grid.reshape(grid_shape),
    )
