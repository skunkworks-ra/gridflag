"""Streaming two-pass histogram-based robust statistics.

Computes per-cell median and robust std from zarr shards without ever
loading all visibilities into memory at once.  Only one shard's flat
data is resident at a time.

Pass 1 — Range discovery: track per-cell min, max, count (sparse per shard).
Pass 2 — Histogram fill: bin values into per-cell histograms, chunked by
         occupied cells to bound memory.
Extraction: median and IQR-based robust std from cumulative histograms.

Threading model: shard reads + numpy compute release the GIL and run in
parallel via ThreadPoolExecutor.  Accumulation into shared arrays
(np.add.at) is done on the main thread only.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import zarr
from numba import njit
from numpy.typing import NDArray

log = logging.getLogger("gridflag.histogram")

# Cells with count <= this threshold use exact values instead of histogram.
_EXACT_THRESHOLD = 32

# Maximum memory (bytes) for the histogram array in pass 2.
# Occupied cells are processed in chunks that fit within this budget.
_HIST_MEM_BUDGET = 2 * 1024**3  # 2 GB


# ── Pass 1: range discovery ─────────────────────────────────────


def _pass1_one_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    N_v: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray] | None:
    """Read one shard, return sparse (unique_cells, min, max, count).

    Returns None if the shard has no data for this (spw, corr).
    All numpy ops release the GIL.
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

    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)

    # Sparse: only allocate for unique cells in this shard.
    unique_cells, inverse = np.unique(flat_idx, return_inverse=True)
    n = len(unique_cells)
    shard_min = np.full(n, np.inf, dtype=np.float64)
    shard_max = np.full(n, -np.inf, dtype=np.float64)
    shard_count = np.zeros(n, dtype=np.int64)

    vals64 = values.astype(np.float64)
    np.minimum.at(shard_min, inverse, vals64)
    np.maximum.at(shard_max, inverse, vals64)
    np.add.at(shard_count, inverse, 1)

    return unique_cells, shard_min, shard_max, shard_count


def pass1_ranges(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    grid_shape: tuple[int, int],
    n_threads: int = 4,
) -> tuple[NDArray, NDArray, NDArray]:
    """Discover per-cell min, max, count across all shards.

    Returns (cell_min, cell_max, cell_count) as flat 1-D arrays of length
    N_u * N_v.  Shard reads run in threads (GIL-releasing I/O + numpy);
    accumulation into the dense result arrays is main-thread only.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    cell_max = np.full(n_cells, -np.inf, dtype=np.float64)
    cell_count = np.zeros(n_cells, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(_pass1_one_shard, sp, spw_key, corr_key, N_v): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            cells, s_min, s_max, s_count = result
            # Accumulate on main thread (np.minimum.at is not thread-safe).
            np.minimum.at(cell_min, cells, s_min)
            np.maximum.at(cell_max, cells, s_max)
            np.add.at(cell_count, cells, s_count)

    return cell_min, cell_max, cell_count


# ── Pass 2: chunked histogram fill + extraction ─────────────────


def _read_and_bin_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    cell_to_chunk_idx: NDArray[np.int64],
    occ_min: NDArray[np.float64],
    occ_max: NDArray[np.float64],
    n_bins: int,
    N_v: int,
    low_count_mask: NDArray[np.bool_],
) -> tuple[NDArray, NDArray | None, NDArray | None] | None:
    """Thread worker: read shard, compute bin assignments.

    Returns (combo, exact_chunk_idx, exact_vals) — lightweight arrays
    for main-thread accumulation.  All heavy work (zarr I/O, numpy)
    releases the GIL.
    """
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

    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)

    # Filter to values belonging to this chunk's cells.
    chunk_idx = cell_to_chunk_idx[flat_idx]
    in_chunk = chunk_idx >= 0
    if not np.any(in_chunk):
        return None

    chunk_idx_f = chunk_idx[in_chunk]
    values_f = values[in_chunk]

    # Bin values.
    cell_min_v = occ_min[chunk_idx_f]
    cell_range_v = occ_max[chunk_idx_f] - cell_min_v
    normalised = (values_f - cell_min_v) / cell_range_v
    bin_idx = np.floor(normalised * n_bins).astype(np.int64)
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

    combo = chunk_idx_f.astype(np.int64) * n_bins + bin_idx

    # Extract exact values for low-count cells only.
    exact_mask = low_count_mask[chunk_idx_f]
    if np.any(exact_mask):
        exact_chunk_idx = chunk_idx_f[exact_mask]
        exact_vals = values_f[exact_mask]
    else:
        exact_chunk_idx = None
        exact_vals = None

    return combo, exact_chunk_idx, exact_vals


def _pass2_chunk(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    chunk_cells: NDArray[np.int64],
    cell_to_chunk_idx: NDArray[np.int64],
    occ_min: NDArray[np.float64],
    occ_max: NDArray[np.float64],
    occ_count: NDArray[np.int64],
    n_bins: int,
    N_v: int,
    n_threads: int,
) -> tuple[NDArray, NDArray, list[NDArray | None]]:
    """Fill histograms and collect exact values for a chunk of occupied cells.

    Shard reads run in threads; accumulation is main-thread only.
    Returns (hist_counts, occ_count, exact_arrays) for this chunk.

    hist_counts : (n_chunk, n_bins) int32
    """
    n_chunk = len(chunk_cells)
    hist_counts = np.zeros((n_chunk, n_bins), dtype=np.int32)

    # Identify which cells need exact collection.
    low_count_mask = occ_count <= _EXACT_THRESHOLD
    has_low_count = np.any(low_count_mask)
    exact_lists: list[list[float] | None] | None = None
    if has_low_count:
        exact_lists = [[] if low_count_mask[i] else None for i in range(n_chunk)]

    hist_ravel = hist_counts.ravel()

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(
                _read_and_bin_shard, sp, spw_key, corr_key,
                cell_to_chunk_idx, occ_min, occ_max,
                n_bins, N_v, low_count_mask,
            ): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            combo, exact_ci, exact_v = result

            # Accumulate histogram (main thread only).
            np.add.at(hist_ravel, combo, 1)

            # Collect exact values for low-count cells.
            if exact_ci is not None and exact_lists is not None:
                for i in range(len(exact_v)):
                    exact_lists[exact_ci[i]].append(float(exact_v[i]))

    # Convert exact lists to arrays.
    exact_arrays: list[NDArray | None] = []
    for i in range(n_chunk):
        if exact_lists is not None and exact_lists[i] is not None:
            exact_arrays.append(np.array(exact_lists[i], dtype=np.float64))
        else:
            exact_arrays.append(None)

    return hist_counts, occ_count, exact_arrays


# ── Extraction ──────────────────────────────────────────────────


@njit(cache=True)
def _extract_stats_jit(
    hist_counts: np.ndarray,
    occ_min: np.ndarray,
    occ_max: np.ndarray,
    occ_count: np.ndarray,
    n_bins: int,
    n_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract median and robust std from histograms (numba JIT).

    Median: linear interpolation within the bin containing the n/2-th value.
    Robust std: IQR / 1.349 (Gaussian equivalent).
    """
    medians = np.zeros(n_chunk, dtype=np.float32)
    stds = np.zeros(n_chunk, dtype=np.float32)

    for i in range(n_chunk):
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
    for b in range(n_bins):
        if cumsum[b] >= rank:
            prev_cum = 0 if b == 0 else cumsum[b - 1]
            count_in_bin = cumsum[b] - prev_cum
            if count_in_bin == 0:
                return lo + (b + 0.5) * bin_width
            fraction = (rank - prev_cum) / count_in_bin
            return lo + (b + fraction) * bin_width
    return lo + n_bins * bin_width


def _extract_chunk(
    hist_counts: NDArray[np.int32],
    occ_min: NDArray[np.float64],
    occ_max: NDArray[np.float64],
    occ_count: NDArray[np.int64],
    exact_arrays: list[NDArray | None],
    n_bins: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extract median and std for one chunk of occupied cells."""
    n_chunk = len(occ_count)

    # JIT extraction for histogram cells.
    medians, stds = _extract_stats_jit(
        hist_counts.astype(np.int64), occ_min, occ_max, occ_count,
        n_bins, n_chunk,
    )

    # Override with exact computation for low-count cells.
    for i in range(n_chunk):
        if occ_count[i] <= _EXACT_THRESHOLD and exact_arrays[i] is not None:
            arr = np.sort(exact_arrays[i])
            n = len(arr)
            if n == 0:
                continue
            if n % 2 == 1:
                med = arr[n // 2]
            else:
                med = (arr[n // 2 - 1] + arr[n // 2]) * 0.5
            medians[i] = np.float32(med)
            absdev = np.sort(np.abs(arr - med))
            if n % 2 == 1:
                mad = absdev[n // 2]
            else:
                mad = (absdev[n // 2 - 1] + absdev[n // 2]) * 0.5
            stds[i] = np.float32(1.4826 * mad)

    return medians, stds


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

    Memory is bounded: occupied cells are processed in chunks sized to
    keep the histogram array under ~2 GB.  Pass 1 uses sparse per-shard
    results.  Shard reads run in threads (zarr I/O + numpy release the
    GIL); accumulation into shared arrays is main-thread only.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    # Pass 1: range discovery.
    cell_min, cell_max, cell_count = pass1_ranges(
        shard_paths, spw_key, corr_key, grid_shape, n_threads
    )

    if cell_count.sum() == 0:
        return (
            np.zeros(grid_shape, dtype=np.float32),
            np.zeros(grid_shape, dtype=np.float32),
            np.zeros(grid_shape, dtype=np.int32),
        )

    # Identify occupied cells.
    occupied_cells = np.where(cell_count > 0)[0].astype(np.int64)
    n_occupied = len(occupied_cells)

    occ_min_all = cell_min[occupied_cells].copy()
    occ_max_all = cell_max[occupied_cells].copy()
    occ_count_all = cell_count[occupied_cells]

    # Free dense pass 1 arrays (no longer needed).
    del cell_min, cell_max, cell_count

    # Widen range for constant-value cells.
    equal_mask = occ_min_all == occ_max_all
    occ_max_all[equal_mask] = occ_min_all[equal_mask] + 1.0

    log.info(
        "Histogram: %d occupied cells (%.1f%% of %d grid cells)",
        n_occupied, 100.0 * n_occupied / max(n_cells, 1), n_cells,
    )

    # Chunk sizing: keep histogram array under budget.
    bytes_per_cell = n_bins * 4  # int32
    max_cells_per_chunk = max(1, int(_HIST_MEM_BUDGET / bytes_per_cell))
    n_chunks = math.ceil(n_occupied / max_cells_per_chunk)
    if n_chunks > 1:
        log.info(
            "Processing in %d chunks (%d cells/chunk, %.1f GB budget)",
            n_chunks, max_cells_per_chunk, _HIST_MEM_BUDGET / 1024**3,
        )

    # Output grids.
    median_grid = np.zeros(n_cells, dtype=np.float32)
    std_grid = np.zeros(n_cells, dtype=np.float32)
    count_grid = np.zeros(n_cells, dtype=np.int32)

    # Process occupied cells in chunks.
    for chunk_i in range(n_chunks):
        c_start = chunk_i * max_cells_per_chunk
        c_end = min(c_start + max_cells_per_chunk, n_occupied)
        chunk_cells = occupied_cells[c_start:c_end]
        chunk_min = occ_min_all[c_start:c_end]
        chunk_max = occ_max_all[c_start:c_end]
        chunk_count = occ_count_all[c_start:c_end]
        n_chunk = len(chunk_cells)

        # Build cell → chunk-local index mapping.
        cell_to_chunk = np.full(n_cells, -1, dtype=np.int64)
        cell_to_chunk[chunk_cells] = np.arange(n_chunk, dtype=np.int64)

        # Pass 2: fill histograms for this chunk.
        hist_counts, _, exact_arrays = _pass2_chunk(
            shard_paths, spw_key, corr_key,
            chunk_cells, cell_to_chunk,
            chunk_min, chunk_max, chunk_count,
            n_bins, N_v, n_threads,
        )

        del cell_to_chunk

        # Extract stats for this chunk.
        medians, stds = _extract_chunk(
            hist_counts, chunk_min, chunk_max, chunk_count,
            exact_arrays, n_bins,
        )

        # Scatter into output grids.
        median_grid[chunk_cells] = medians
        std_grid[chunk_cells] = stds
        count_grid[chunk_cells] = chunk_count.astype(np.int32)

        del hist_counts, exact_arrays

    return (
        median_grid.reshape(grid_shape),
        std_grid.reshape(grid_shape),
        count_grid.reshape(grid_shape),
    )
