"""Streaming two-pass histogram-based robust statistics.

Computes per-cell median and robust std from zarr shards without ever
loading all visibilities into memory at once.  Only one shard's flat
data is resident at a time.

Pass 1 — Range discovery: track per-cell min, max, count.
Pass 2 — Histogram fill: bin values into per-cell histograms.
Extraction: median and IQR-based robust std from cumulative histograms.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import zarr
from numba import njit
from numpy.typing import NDArray

log = logging.getLogger("gridflag.histogram")

# Cells with count <= this threshold use exact values instead of histogram.
_EXACT_THRESHOLD = 32


# ── Pass 1: range discovery ─────────────────────────────────────


def _pass1_one_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    n_cells: int,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Read one shard, return (cell_min, cell_max, cell_count) arrays.

    Returns None if the shard has no data for this (spw, corr).
    """
    root = zarr.open(shard_path, mode="r")
    try:
        grp = root[spw_key][corr_key]
    except KeyError:
        return None

    cell_u = grp["cell_u"][:]
    cell_v = grp["cell_v"][:]
    values = grp["values"][:]
    if len(values) == 0:
        return None

    flat_idx = cell_u.astype(np.int64) * _pass1_one_shard._N_v + cell_v.astype(np.int64)

    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    cell_max = np.full(n_cells, -np.inf, dtype=np.float64)
    cell_count = np.zeros(n_cells, dtype=np.int64)

    # Use numpy advanced indexing for speed.
    np.minimum.at(cell_min, flat_idx, values.astype(np.float64))
    np.maximum.at(cell_max, flat_idx, values.astype(np.float64))
    np.add.at(cell_count, flat_idx, 1)

    return cell_min, cell_max, cell_count


def pass1_ranges(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    grid_shape: tuple[int, int],
    n_threads: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Discover per-cell min, max, count across all shards.

    Returns (cell_min, cell_max, cell_count) as flat 1-D arrays of length
    N_u * N_v.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    # Stash N_v on the function for the worker to use (avoids closure overhead
    # with ThreadPoolExecutor).
    _pass1_one_shard._N_v = N_v

    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    cell_max = np.full(n_cells, -np.inf, dtype=np.float64)
    cell_count = np.zeros(n_cells, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(_pass1_one_shard, sp, spw_key, corr_key, n_cells): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            s_min, s_max, s_count = result
            np.minimum(cell_min, s_min, out=cell_min)
            np.maximum(cell_max, s_max, out=cell_max)
            cell_count += s_count

    return cell_min, cell_max, cell_count


# ── Pass 2: histogram fill ──────────────────────────────────────


def _pass2_one_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    occupied_cells: NDArray[np.int64],
    cell_to_idx: NDArray[np.int64],
    bin_edges: NDArray[np.float64],
    N_v: int,
) -> tuple[NDArray, list[list[float]]] | None:
    """Read one shard, accumulate into thread-local histogram counts.

    Returns (hist_counts, exact_values) where exact_values is a list per
    occupied cell of values for low-count cells.
    """
    root = zarr.open(shard_path, mode="r")
    try:
        grp = root[spw_key][corr_key]
    except KeyError:
        return None

    cell_u = grp["cell_u"][:]
    cell_v = grp["cell_v"][:]
    values = grp["values"][:]
    if len(values) == 0:
        return None

    n_occupied = len(occupied_cells)
    n_bins = len(bin_edges) - 1

    hist_counts = np.zeros((n_occupied, n_bins), dtype=np.int64)
    exact_values: list[list[float]] = [[] for _ in range(n_occupied)]

    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)
    occ_idx = cell_to_idx[flat_idx]  # map flat cell → occupied index

    # Bin indices for all values (clipped to [0, n_bins-1]).
    bin_idx = np.searchsorted(bin_edges[1:], values.astype(np.float64), side="right")
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

    # Vectorised histogram fill.
    combo = occ_idx * n_bins + bin_idx
    np.add.at(hist_counts.ravel(), combo, 1)

    # Collect exact values for low-count cells.
    # We collect all values here; the caller decides which cells are low-count.
    _collect_exact(occ_idx, values, exact_values)

    return hist_counts, exact_values


def _collect_exact(
    occ_idx: NDArray[np.int64],
    values: NDArray,
    exact_values: list[list[float]],
) -> None:
    """Append values into per-cell exact lists (Python loop, only for small cells)."""
    for i in range(len(values)):
        exact_values[occ_idx[i]].append(float(values[i]))


def pass2_histograms(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    grid_shape: tuple[int, int],
    cell_min: NDArray,
    cell_max: NDArray,
    cell_count: NDArray,
    n_bins: int,
    n_threads: int,
) -> tuple[NDArray, NDArray, NDArray, list[NDArray | None]]:
    """Fill per-cell histograms across all shards.

    Returns (occupied_cells, bin_edges_per_cell, hist_counts, exact_values)
    where:
    - occupied_cells: 1-D int64, flat cell indices with count > 0
    - bin_edges_per_cell: (n_occupied, n_bins+1) float64
    - hist_counts: (n_occupied, n_bins) int64
    - exact_values: list of (ndarray or None) per occupied cell
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    occupied_mask = cell_count > 0
    occupied_cells = np.where(occupied_mask)[0].astype(np.int64)
    n_occupied = len(occupied_cells)

    if n_occupied == 0:
        return (
            occupied_cells,
            np.empty((0, n_bins + 1), dtype=np.float64),
            np.empty((0, n_bins), dtype=np.int64),
            [],
        )

    # Map flat cell index → occupied index (-1 for unoccupied).
    cell_to_idx = np.full(n_cells, -1, dtype=np.int64)
    cell_to_idx[occupied_cells] = np.arange(n_occupied, dtype=np.int64)

    # Per-cell bin edges.  Uniform bins between [min, max] per cell.
    occ_min = cell_min[occupied_cells]
    occ_max = cell_max[occupied_cells]

    # Handle cells where min == max (constant value): widen range slightly.
    equal_mask = occ_min == occ_max
    occ_max[equal_mask] = occ_min[equal_mask] + 1.0

    # bin_edges: (n_occupied, n_bins+1)
    t = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)  # (n_bins+1,)
    bin_edges_per_cell = occ_min[:, None] + (occ_max - occ_min)[:, None] * t[None, :]

    # For the parallel workers, we use a single global bin edge array
    # (the per-cell edges are used in extraction, but during fill we need
    # per-cell binning).  Since searchsorted works on sorted 1-D arrays,
    # we'll do per-cell binning in the shard reader.

    # Actually, for efficiency, each shard worker bins values using the
    # per-cell edges.  We pass the full bin_edges_per_cell to each worker.
    # But that's a lot of data to pass.  Instead, workers compute bin index
    # from min/max directly.

    hist_counts = np.zeros((n_occupied, n_bins), dtype=np.int64)
    exact_collected: list[list[float]] = [[] for _ in range(n_occupied)]

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(
                _pass2_one_shard_binned,
                sp,
                spw_key,
                corr_key,
                occupied_cells,
                cell_to_idx,
                occ_min,
                occ_max,
                n_bins,
                N_v,
            ): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            s_hist, s_exact = result
            hist_counts += s_hist
            for i, vals in enumerate(s_exact):
                exact_collected[i].extend(vals)

    # Convert exact lists to arrays for low-count cells.
    occ_count = cell_count[occupied_cells]
    exact_arrays: list[NDArray | None] = []
    for i in range(n_occupied):
        if occ_count[i] <= _EXACT_THRESHOLD:
            exact_arrays.append(np.array(exact_collected[i], dtype=np.float64))
        else:
            exact_arrays.append(None)

    return occupied_cells, bin_edges_per_cell, hist_counts, exact_arrays


def _pass2_one_shard_binned(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    occupied_cells: NDArray[np.int64],
    cell_to_idx: NDArray[np.int64],
    occ_min: NDArray[np.float64],
    occ_max: NDArray[np.float64],
    n_bins: int,
    N_v: int,
) -> tuple[NDArray, list[list[float]]] | None:
    """Read one shard, bin values using per-cell min/max."""
    root = zarr.open(shard_path, mode="r")
    try:
        grp = root[spw_key][corr_key]
    except KeyError:
        return None

    cell_u = grp["cell_u"][:]
    cell_v = grp["cell_v"][:]
    values = grp["values"][:].astype(np.float64)
    if len(values) == 0:
        return None

    n_occupied = len(occupied_cells)
    hist_counts = np.zeros((n_occupied, n_bins), dtype=np.int64)
    exact_values: list[list[float]] = [[] for _ in range(n_occupied)]

    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)
    occ_idx = cell_to_idx[flat_idx]

    # Per-value bin index: bin_i = floor((val - min) / (max - min) * n_bins)
    cell_min_v = occ_min[occ_idx]
    cell_range_v = occ_max[occ_idx] - cell_min_v
    normalised = (values - cell_min_v) / cell_range_v  # [0, 1]
    bin_idx = np.floor(normalised * n_bins).astype(np.int64)
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

    # Vectorised fill.
    combo = occ_idx * n_bins + bin_idx
    np.add.at(hist_counts.ravel(), combo, 1)

    # Collect exact values for all cells (caller filters by count).
    _collect_exact(occ_idx, values, exact_values)

    return hist_counts, exact_values


# ── Extraction ──────────────────────────────────────────────────


@njit(cache=True)
def _extract_stats_jit(
    hist_counts: np.ndarray,
    occ_min: np.ndarray,
    occ_max: np.ndarray,
    occ_count: np.ndarray,
    n_bins: int,
    n_occupied: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract median and robust std from histograms (numba JIT).

    Median: linear interpolation within the bin containing the n/2-th value.
    Robust std: IQR / 1.349 (Gaussian equivalent).
    """
    medians = np.zeros(n_occupied, dtype=np.float32)
    stds = np.zeros(n_occupied, dtype=np.float32)

    for i in range(n_occupied):
        n = occ_count[i]
        if n == 0:
            continue

        lo = occ_min[i]
        hi = occ_max[i]
        bin_width = (hi - lo) / n_bins

        # Cumulative sum.
        cumsum = np.zeros(n_bins, dtype=np.int64)
        cumsum[0] = hist_counts[i, 0]
        for b in range(1, n_bins):
            cumsum[b] = cumsum[b - 1] + hist_counts[i, b]

        # Find median (n/2-th value).
        median_rank = n / 2.0
        medians[i] = np.float32(_interpolate_quantile(cumsum, median_rank, lo, bin_width, n_bins))

        # Q1 and Q3.
        q1_rank = n / 4.0
        q3_rank = 3.0 * n / 4.0
        q1 = _interpolate_quantile(cumsum, q1_rank, lo, bin_width, n_bins)
        q3 = _interpolate_quantile(cumsum, q3_rank, lo, bin_width, n_bins)

        iqr = q3 - q1
        stds[i] = np.float32(iqr / 1.3490)

    return medians, stds


@njit(cache=True)
def _interpolate_quantile(
    cumsum: np.ndarray,
    rank: float,
    lo: float,
    bin_width: float,
    n_bins: int,
) -> float:
    """Find the value at the given rank via linear interpolation within a bin."""
    # Find the first bin where cumsum >= rank.
    for b in range(n_bins):
        if cumsum[b] >= rank:
            # Interpolate within this bin.
            prev_cum = 0 if b == 0 else cumsum[b - 1]
            count_in_bin = cumsum[b] - prev_cum
            if count_in_bin == 0:
                return lo + (b + 0.5) * bin_width
            fraction = (rank - prev_cum) / count_in_bin
            return lo + (b + fraction) * bin_width
    # Shouldn't reach here, but return upper bound.
    return lo + n_bins * bin_width


def extract_stats(
    occupied_cells: NDArray[np.int64],
    bin_edges_per_cell: NDArray[np.float64],
    hist_counts: NDArray[np.int64],
    cell_count: NDArray[np.int64],
    exact_arrays: list[NDArray | None],
    grid_shape: tuple[int, int],
    n_bins: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Extract median_grid, std_grid, count_grid from histogram data.

    Low-count cells (count <= _EXACT_THRESHOLD) use exact computation.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v
    n_occupied = len(occupied_cells)

    occ_count = cell_count[occupied_cells]
    occ_min = bin_edges_per_cell[:, 0]
    occ_max = bin_edges_per_cell[:, -1]

    # JIT extraction for histogram cells.
    medians, stds = _extract_stats_jit(
        hist_counts, occ_min, occ_max, occ_count, n_bins, n_occupied
    )

    # Override with exact computation for low-count cells.
    for i in range(n_occupied):
        if occ_count[i] <= _EXACT_THRESHOLD and exact_arrays[i] is not None:
            arr = np.sort(exact_arrays[i])
            n = len(arr)
            if n == 0:
                continue
            # Exact median.
            if n % 2 == 1:
                med = arr[n // 2]
            else:
                med = (arr[n // 2 - 1] + arr[n // 2]) * 0.5
            medians[i] = np.float32(med)
            # Exact MAD → robust std.
            absdev = np.sort(np.abs(arr - med))
            if n % 2 == 1:
                mad = absdev[n // 2]
            else:
                mad = (absdev[n // 2 - 1] + absdev[n // 2]) * 0.5
            stds[i] = np.float32(1.4826 * mad)

    # Scatter into full grids.
    median_grid = np.zeros(n_cells, dtype=np.float32)
    std_grid = np.zeros(n_cells, dtype=np.float32)
    count_grid = np.zeros(n_cells, dtype=np.int32)

    median_grid[occupied_cells] = medians
    std_grid[occupied_cells] = stds
    count_grid[occupied_cells] = occ_count.astype(np.int32)

    return (
        median_grid.reshape(grid_shape),
        std_grid.reshape(grid_shape),
        count_grid.reshape(grid_shape),
    )


# ── Convenience: full two-pass pipeline ─────────────────────────


def compute_cell_stats_streaming(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    grid_shape: tuple[int, int],
    n_bins: int = 256,
    n_threads: int = 4,
) -> tuple[NDArray, NDArray, NDArray]:
    """Two-pass streaming statistics over zarr shards.

    Returns (median_grid, std_grid, count_grid) with same shapes and
    semantics as ``gridder.compute_cell_stats``.
    """
    cell_min, cell_max, cell_count = pass1_ranges(
        shard_paths, spw_key, corr_key, grid_shape, n_threads
    )

    if cell_count.sum() == 0:
        return (
            np.zeros(grid_shape, dtype=np.float32),
            np.zeros(grid_shape, dtype=np.float32),
            np.zeros(grid_shape, dtype=np.int32),
        )

    occupied_cells, bin_edges, hist_counts, exact_arrays = pass2_histograms(
        shard_paths,
        spw_key,
        corr_key,
        grid_shape,
        cell_min,
        cell_max,
        cell_count,
        n_bins,
        n_threads,
    )

    return extract_stats(
        occupied_cells, bin_edges, hist_counts, cell_count, exact_arrays,
        grid_shape, n_bins,
    )
