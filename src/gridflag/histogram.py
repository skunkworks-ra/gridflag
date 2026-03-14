"""Streaming adaptive histogram-based robust statistics.

Computes per-cell median and robust std from zarr shards without ever
loading all visibilities into memory at once.

Pass 0 — Count discovery: per-cell visibility counts (reads only cell
         indices, skips values for speed when no threshold is applied).
Pass 1 — Adaptive histogram fill: bin values into per-cell histograms,
         initialising cell ranges on first data and rebinning on range
         expansion (rare after the first shard).  Chunked by occupied
         cells to bound memory.
Extraction: median and IQR-based robust std from cumulative histograms.
            Low-count cells (≤ _EXACT_THRESHOLD) use exact median/MAD.

Threading model: shard reads + numpy compute release the GIL and run in
parallel via ThreadPoolExecutor.  Accumulation into shared arrays is
done on the main thread only.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import zarr
from numba import njit
from numpy.typing import NDArray

from gridflag.gridder import _segmented_median_mad

log = logging.getLogger("gridflag.histogram")

# Cells with count <= this threshold use exact values instead of histogram.
_EXACT_THRESHOLD = 32

# Maximum memory (bytes) for the histogram array.
# Occupied cells are processed in chunks that fit within this budget.
_HIST_MEM_BUDGET = 2 * 1024**3  # 2 GB


# ── Pass 0: count discovery ──────────────────────────────────────


def _pass0_one_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    N_v: int,
    n_cells: int,
    threshold_grid: NDArray | None,
) -> NDArray[np.int64]:
    """Worker: read cell indices, return flat count-per-cell array.

    When threshold_grid is None, values are not read (faster I/O).
    When threshold_grid is provided, values are read to apply the filter.
    """
    root = zarr.open(shard_path, mode="r")
    try:
        grp = root[spw_key][corr_key]
    except KeyError:
        return np.zeros(n_cells, dtype=np.int64)

    cell_u = grp["cell_u"][:]
    cell_v = grp["cell_v"][:]
    if len(cell_u) == 0:
        return np.zeros(n_cells, dtype=np.int64)

    if threshold_grid is not None:
        values = grp["values"][:]
        thr = threshold_grid[cell_u.astype(np.intp), cell_v.astype(np.intp)]
        keep = (values <= thr) & ~np.isnan(thr)
        cell_u = cell_u[keep]
        cell_v = cell_v[keep]
        if len(cell_u) == 0:
            return np.zeros(n_cells, dtype=np.int64)

    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)
    return np.bincount(flat_idx, minlength=n_cells).astype(np.int64)


def pass0_counts(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    grid_shape: tuple[int, int],
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
) -> NDArray[np.int64]:
    """Count visibilities per cell across all shards.

    Returns cell_count: flat 1-D array of length N_u * N_v.
    When threshold_grid is None, value arrays are skipped for faster I/O.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v
    cell_count = np.zeros(n_cells, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(
                _pass0_one_shard, sp, spw_key, corr_key, N_v, n_cells, threshold_grid,
            ): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            cell_count += fut.result()

    return cell_count


# ── Adaptive histogram fill ──────────────────────────────────────


@njit(cache=True)
def _rebin_histogram(
    hist_row: np.ndarray,
    old_lo: float,
    old_hi: float,
    new_lo: float,
    new_hi: float,
    n_bins: int,
) -> np.ndarray:
    """Redistribute one cell's histogram counts into a wider range. O(n_bins)."""
    new_hist = np.zeros(n_bins, dtype=hist_row.dtype)
    old_width = (old_hi - old_lo) / n_bins
    new_width = (new_hi - new_lo) / n_bins
    for b in range(n_bins):
        if hist_row[b] == 0:
            continue
        center = old_lo + (b + 0.5) * old_width
        new_b = int((center - new_lo) / new_width)
        if new_b < 0:
            new_b = 0
        elif new_b >= n_bins:
            new_b = n_bins - 1
        new_hist[new_b] += hist_row[b]
    return new_hist


def _read_raw_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    cell_to_chunk_idx: NDArray[np.int64],
    N_v: int,
    low_count_mask: NDArray[np.bool_],
    threshold_grid: NDArray | None,
) -> tuple[NDArray, NDArray, NDArray | None, NDArray | None] | None:
    """Worker: read shard, filter to chunk cells, return raw values.

    Returns (chunk_idx_f, values_f, exact_ci, exact_v) or None.
    All heavy work (zarr I/O, numpy) releases the GIL.
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

    if threshold_grid is not None:
        thr = threshold_grid[cell_u.astype(np.intp), cell_v.astype(np.intp)]
        keep = (values <= thr) & ~np.isnan(thr)
        cell_u = cell_u[keep]
        cell_v = cell_v[keep]
        values = values[keep]
        if len(values) == 0:
            return None

    flat_idx = cell_u.astype(np.int64) * N_v + cell_v.astype(np.int64)
    chunk_idx = cell_to_chunk_idx[flat_idx]
    in_chunk = chunk_idx >= 0
    if not np.any(in_chunk):
        return None

    chunk_idx_f = chunk_idx[in_chunk]
    values_f = values[in_chunk]

    # Collect exact values for low-count cells.
    exact_mask = low_count_mask[chunk_idx_f]
    if np.any(exact_mask):
        exact_ci = chunk_idx_f[exact_mask]
        exact_v = values_f[exact_mask]
    else:
        exact_ci = None
        exact_v = None

    return chunk_idx_f, values_f, exact_ci, exact_v


def _adaptive_fill_chunk(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    chunk_cells: NDArray[np.int64],
    cell_to_chunk_idx: NDArray[np.int64],
    occ_count: NDArray[np.int64],
    n_bins: int,
    N_v: int,
    n_threads: int,
    threshold_grid: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None, NDArray | None]:
    """Adaptive histogram fill for a chunk of occupied cells.

    Initialises per-cell ranges from first data seen; rebins on range
    expansion (rare after first shard).  Shard reads run in threads;
    histogram accumulation is main-thread only.

    Returns
    -------
    hist_counts    : (n_chunk, n_bins) int32
    occ_lo         : (n_chunk,) float64 — actual lower bound used for binning
    occ_hi         : (n_chunk,) float64 — actual upper bound used for binning
    all_exact_cidx : concatenated chunk indices for exact-path cells, or None
    all_exact_vals : concatenated values for exact-path cells, or None
    """
    n_chunk = len(chunk_cells)
    hist_counts = np.zeros((n_chunk, n_bins), dtype=np.int32)
    hist_ravel = hist_counts.ravel()
    occ_lo = np.full(n_chunk, np.inf, dtype=np.float64)
    occ_hi = np.full(n_chunk, -np.inf, dtype=np.float64)
    initialized = np.zeros(n_chunk, dtype=np.bool_)

    low_count_mask = occ_count <= _EXACT_THRESHOLD
    exact_cidx_parts: list[NDArray] = []
    exact_vals_parts: list[NDArray] = []

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(
                _read_raw_shard, sp, spw_key, corr_key,
                cell_to_chunk_idx, N_v, low_count_mask, threshold_grid,
            ): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            chunk_idx_f, values_f, exact_ci, exact_v = result

            # Collect exact values.
            if exact_ci is not None:
                exact_cidx_parts.append(exact_ci)
                exact_vals_parts.append(exact_v)

            # Per-cell min/max for this batch.
            unique_ci, inv = np.unique(chunk_idx_f, return_inverse=True)
            batch_lo = np.full(len(unique_ci), np.inf)
            batch_hi = np.full(len(unique_ci), -np.inf)
            np.minimum.at(batch_lo, inv, values_f)
            np.maximum.at(batch_hi, inv, values_f)

            # Update per-cell ranges; rebin if an existing range expands.
            # Rebin events are rare (only when a later shard extends the range).
            for k in range(len(unique_ci)):
                ci = int(unique_ci[k])
                b_lo = float(batch_lo[k])
                b_hi = float(batch_hi[k])
                if not initialized[ci]:
                    occ_lo[ci] = b_lo
                    # Ensure non-zero width for constant-value cells.
                    occ_hi[ci] = b_hi if b_hi > b_lo else b_lo + 1.0
                    initialized[ci] = True
                else:
                    new_lo = min(occ_lo[ci], b_lo)
                    new_hi = max(occ_hi[ci], b_hi)
                    if new_lo < occ_lo[ci] or new_hi > occ_hi[ci]:
                        hi_safe = new_hi if new_hi > new_lo else new_lo + 1.0
                        hist_counts[ci] = _rebin_histogram(
                            hist_counts[ci],
                            occ_lo[ci], occ_hi[ci],
                            new_lo, hi_safe,
                            n_bins,
                        )
                        occ_lo[ci] = new_lo
                        occ_hi[ci] = hi_safe

            # Vectorized histogram binning for this shard's batch.
            lo_f = occ_lo[chunk_idx_f]
            range_f = occ_hi[chunk_idx_f] - lo_f
            normalised = (values_f - lo_f) / range_f
            bin_idx = np.floor(normalised * n_bins).astype(np.int64)
            np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
            combo = chunk_idx_f.astype(np.int64) * n_bins + bin_idx
            hist_ravel += np.bincount(combo, minlength=n_chunk * n_bins).astype(np.int32)

    all_exact_cidx = np.concatenate(exact_cidx_parts) if exact_cidx_parts else None
    all_exact_vals = np.concatenate(exact_vals_parts) if exact_vals_parts else None

    return hist_counts, occ_lo, occ_hi, all_exact_cidx, all_exact_vals


# ── Extraction ───────────────────────────────────────────────────


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
    all_exact_cidx: NDArray[np.int64] | None,
    all_exact_vals: NDArray[np.float64] | None,
    n_bins: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extract median and std for one chunk of occupied cells.

    Histogram-based stats are computed for all cells via JIT; low-count
    cells (≤ _EXACT_THRESHOLD) are overridden with exact median/MAD using
    a vectorized segmented reduction.
    """
    n_chunk = len(occ_count)

    # JIT extraction for histogram cells.
    medians, stds = _extract_stats_jit(
        hist_counts.astype(np.int64), occ_min, occ_max, occ_count,
        n_bins, n_chunk,
    )

    # Vectorized exact path: override low-count cells via _segmented_median_mad.
    if all_exact_cidx is not None and len(all_exact_cidx) > 0:
        sort_order = np.argsort(all_exact_cidx, kind="stable")
        sorted_cidx = all_exact_cidx[sort_order]
        sorted_vals = all_exact_vals[sort_order].astype(np.float32)

        unique_cidx, seg_starts, seg_counts = np.unique(
            sorted_cidx, return_index=True, return_counts=True,
        )

        ex_med, ex_std, _ = _segmented_median_mad(
            sorted_vals,
            seg_starts.astype(np.int64),
            seg_counts.astype(np.int64),
            unique_cidx.astype(np.int64),
            n_chunk,
        )

        medians[unique_cidx] = ex_med[unique_cidx]
        stds[unique_cidx] = ex_std[unique_cidx]

    return medians, stds


# ── Convenience: full pipeline ───────────────────────────────────


def compute_cell_stats_streaming(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    grid_shape: tuple[int, int],
    n_bins: int = 256,
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Adaptive single-pass streaming statistics over zarr shards.

    Pass 0 discovers per-cell counts (skipping value arrays when no
    threshold is applied, for faster I/O).  Pass 1 fills histograms
    adaptively, initialising cell ranges on first data and rebinning on
    range expansion.  Low-count cells (≤ _EXACT_THRESHOLD) are computed
    exactly via vectorized segmented median/MAD.

    Returns (median_grid, std_grid, count_grid) with the same shapes
    and semantics as ``gridder.compute_cell_stats``.

    If threshold_grid is provided, only values <= their cell's threshold
    are included.  Used for computing post-flag "after" grids for plotting.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    # Pass 0: count discovery.
    cell_count = pass0_counts(
        shard_paths, spw_key, corr_key, grid_shape, n_threads, threshold_grid,
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
    occ_count_all = cell_count[occupied_cells]

    del cell_count

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
        chunk_count = occ_count_all[c_start:c_end]
        n_chunk = len(chunk_cells)

        # Build cell → chunk-local index mapping.
        cell_to_chunk = np.full(n_cells, -1, dtype=np.int64)
        cell_to_chunk[chunk_cells] = np.arange(n_chunk, dtype=np.int64)

        # Adaptive histogram fill for this chunk.
        hist_counts, chunk_lo, chunk_hi, all_exact_cidx, all_exact_vals = (
            _adaptive_fill_chunk(
                shard_paths, spw_key, corr_key,
                chunk_cells, cell_to_chunk,
                chunk_count, n_bins, N_v, n_threads, threshold_grid,
            )
        )

        del cell_to_chunk

        # Extract stats for this chunk.
        medians, stds = _extract_chunk(
            hist_counts, chunk_lo, chunk_hi, chunk_count,
            all_exact_cidx, all_exact_vals, n_bins,
        )

        # Scatter into output grids.
        median_grid[chunk_cells] = medians
        std_grid[chunk_cells] = stds
        count_grid[chunk_cells] = chunk_count.astype(np.int32)

        del hist_counts, all_exact_cidx, all_exact_vals

    return (
        median_grid.reshape(grid_shape),
        std_grid.reshape(grid_shape),
        count_grid.reshape(grid_shape),
    )
