"""Local-neighborhood and annular threshold computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter


def local_neighborhood_threshold(
    median_grid: NDArray,
    std_grid: NDArray,
    count_grid: NDArray,
    nsigma: float,
    kernel_size: int,
) -> NDArray:
    """Compute thresholds from a smoothed local neighborhood.

    Uses a masked uniform filter: averages only over cells with count > 0.

    Parameters
    ----------
    median_grid, std_grid, count_grid : 2-D arrays, same shape.
    nsigma : number of sigma for threshold.
    kernel_size : side length of the uniform smoothing kernel.

    Returns
    -------
    threshold : 2-D float32, same shape.  NaN where count == 0.
    """
    mask = count_grid > 0
    count_f = count_grid.astype(np.float64)

    # Weighted sums via uniform filter.
    sum_median = uniform_filter(
        (median_grid * count_f).astype(np.float64), size=kernel_size, mode="constant"
    )
    sum_std = uniform_filter(
        (std_grid * count_f).astype(np.float64), size=kernel_size, mode="constant"
    )
    sum_count = uniform_filter(count_f, size=kernel_size, mode="constant")

    # Avoid division by zero.
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_median = np.where(sum_count > 0, sum_median / sum_count, np.nan)
        avg_std = np.where(sum_count > 0, sum_std / sum_count, np.nan)

    threshold = (avg_median + nsigma * avg_std).astype(np.float32)
    threshold[~mask] = np.nan
    return threshold


def neighbor_count(count_grid: NDArray, kernel_size: int) -> NDArray:
    """Count occupied neighbors within the smoothing kernel for each cell.

    Returns an int array of how many cells with count > 0 fall within the
    kernel centred on each cell (excluding the cell itself).
    """
    mask = (count_grid > 0).astype(np.float64)
    total = uniform_filter(mask, size=kernel_size, mode="constant") * (
        kernel_size ** 2
    )
    # Subtract self.
    n_neighbors = np.rint(total).astype(np.int32) - (count_grid > 0).astype(np.int32)
    return n_neighbors


def annular_threshold(
    median_grid: NDArray,
    std_grid: NDArray,
    count_grid: NDArray,
    cell_size: float,
    annulus_widths: tuple[float, ...],
    nsigma: float,
    N: int,
) -> NDArray:
    """Compute thresholds via radial annuli in the UV plane.

    Parameters
    ----------
    median_grid, std_grid, count_grid : 2-D grids, shape (2*N+1, N+1).
    cell_size : grid cell size in λ.
    annulus_widths : widths of successive annuli in λ, from the origin outward.
    nsigma : sigma multiplier.
    N : grid half-size parameter.

    Returns
    -------
    threshold : 2-D float32, same shape. NaN where count == 0.
    """
    N_u, N_v = median_grid.shape
    # Cell centre coordinates in λ.
    u_idx = np.arange(N_u) - N  # centred
    v_idx = np.arange(N_v)
    uu, vv = np.meshgrid(u_idx, v_idx, indexing="ij")
    r = np.sqrt((uu * cell_size) ** 2 + (vv * cell_size) ** 2)

    # Build annulus boundaries.
    boundaries = [0.0]
    for w in annulus_widths:
        boundaries.append(boundaries[-1] + w)
    # Extend last annulus to cover everything beyond.
    boundaries.append(float(np.max(r)) + 1.0)

    threshold = np.full_like(median_grid, np.nan, dtype=np.float32)
    mask = count_grid > 0
    count_f = count_grid.astype(np.float64)

    for i in range(len(boundaries) - 1):
        r_lo, r_hi = boundaries[i], boundaries[i + 1]
        ring = (r >= r_lo) & (r < r_hi) & mask
        if not np.any(ring):
            continue
        total_count = np.sum(count_f[ring])
        if total_count == 0:
            continue
        avg_med = np.sum(median_grid[ring] * count_f[ring]) / total_count
        avg_std = np.sum(std_grid[ring] * count_f[ring]) / total_count
        thr = avg_med + nsigma * avg_std
        # Assign to all occupied cells in this annulus.
        threshold[ring] = thr

    return threshold


def combine_thresholds(
    local_thr: NDArray,
    annular_thr: NDArray,
    count_grid: NDArray,
    kernel_size: int,
    min_neighbors: int,
) -> NDArray:
    """Combine local and annular thresholds.

    Use ``min(local, annular)`` per cell.  If a cell has fewer than
    *min_neighbors* occupied neighbors, use annular only.

    Returns
    -------
    threshold : 2-D float32.
    """
    n_nbrs = neighbor_count(count_grid, kernel_size)
    sparse = n_nbrs < min_neighbors

    combined = np.where(
        sparse,
        annular_thr,
        np.fmin(local_thr, annular_thr),  # fmin ignores NaN
    )
    # Keep NaN where cell is empty.
    combined[count_grid == 0] = np.nan
    return combined.astype(np.float32)
