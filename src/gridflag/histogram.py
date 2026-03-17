"""Fixed-range parallel histogram-based robust statistics.

Computes per-cell median and robust std from a consolidated zarr store
without loading all visibilities into memory at once.

Pass 0 — Count + range discovery: per-cell visibility counts plus
         per-cell min/max.  Zarr chunk reads run in parallel via
         ThreadPoolExecutor; reduction (bincount / min / max) is serial
         on the main thread to avoid N×n_cells memory blowup.
Pass 1 — Fixed-range histogram fill: bin values using pre-computed
         per-cell ranges (no rebinning).  Zarr chunk reads are parallel;
         JIT histogram accumulation is serial into a single shared array
         (same two-phase pattern — avoids N copies of the histogram).
Extraction: median and IQR-based robust std from cumulative histograms.
            Low-count cells (≤ _EXACT_THRESHOLD) use exact median/MAD.

Memory model: at most ONE histogram array of size
(max_cells_per_chunk × n_bins × 4 bytes) exists.  Workers return only
compact (flat_idx, values) tuples (~16 MB each for a 1M-element zarr
chunk), so peak worker memory is n_threads × ~16 MB.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import zarr
from numba import njit, prange
from numpy.typing import NDArray

from gridflag.gridder import _segmented_median_mad

log = logging.getLogger("gridflag.histogram")

# Cells with count <= this threshold use exact values instead of histogram.
_EXACT_THRESHOLD = 32

# Maximum memory (bytes) for the histogram array.
# Occupied cells are processed in chunks that fit within this budget.
_HIST_MEM_BUDGET = 2 * 1024**3  # 2 GB


# ── Pass 0: count + range discovery ─────────────────────────────


@njit(cache=True)
def _reduce_pass0_jit(flat_idx, values, cell_count, cell_min, cell_max, skip_counts):
    """Fused count + min/max reduction — no transient allocations."""
    for i in range(len(flat_idx)):
        idx = flat_idx[i]
        if not skip_counts:
            cell_count[idx] += 1
        v = values[i]
        if v < cell_min[idx]:
            cell_min[idx] = v
        if v > cell_max[idx]:
            cell_max[idx] = v


def _pass0_read_chunk(
    zarr_group: zarr.hierarchy.Group,
    start: int,
    end: int,
    N_v: int,
    threshold_grid: NDArray | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]] | None:
    """Worker: read one zarr chunk, return compact (flat_idx, values).

    Returns None when the chunk contributes no data.
    All heavy work (zarr I/O, numpy) releases the GIL.
    """
    cell_u = zarr_group["cell_u"][start:end]
    cell_v = zarr_group["cell_v"][start:end]
    values = zarr_group["values"][start:end].astype(np.float64)

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
    return flat_idx, values


def pass0_counts_and_ranges(
    zarr_group: zarr.hierarchy.Group,
    grid_shape: tuple[int, int],
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
    pre_counts: NDArray[np.int64] | None = None,
    chunk_cache: list | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Count + min/max per cell over consolidated zarr chunks.

    Two-phase: workers read + filter zarr chunks in parallel (returning
    compact flat_idx + values); main thread reduces serially (bincount,
    element-wise min/max).  Peak worker memory is n_threads × ~16 MB.

    Returns (cell_count, cell_min, cell_max) — all flat 1-D (n_cells,).
    When *pre_counts* is provided and no threshold filter is active,
    bincount is skipped.

    When *chunk_cache* is a list, (flat_idx, values) tuples are appended
    so the fill phase can reuse them without re-reading zarr.
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    if "values" not in zarr_group or zarr_group["values"].shape[0] == 0:
        return (
            pre_counts if pre_counts is not None
            else np.zeros(n_cells, dtype=np.int64),
            np.full(n_cells, np.inf, dtype=np.float64),
            np.full(n_cells, -np.inf, dtype=np.float64),
        )

    total_len = zarr_group["values"].shape[0]
    chunk_size = zarr_group["values"].chunks[0]
    chunk_ranges = [
        (i, min(i + chunk_size, total_len))
        for i in range(0, total_len, chunk_size)
    ]

    skip_counts = pre_counts is not None and threshold_grid is None
    cell_count = (
        pre_counts.copy() if skip_counts
        else np.zeros(n_cells, dtype=np.int64)
    )
    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    cell_max = np.full(n_cells, -np.inf, dtype=np.float64)

    # Phase 1: parallel I/O.  Phase 2: serial reduction.
    t_io_total = 0.0
    t_reduce_total = 0.0
    n_chunks_processed = 0

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [
            pool.submit(
                _pass0_read_chunk, zarr_group, s, e, N_v, threshold_grid,
            )
            for s, e in chunk_ranges
        ]
        for fut in as_completed(futures):
            _t_r = time.perf_counter()
            result = fut.result()
            if result is None:
                continue
            flat_idx, values = result
            t_io_total += time.perf_counter() - _t_r

            if chunk_cache is not None:
                chunk_cache.append((flat_idx, values))

            _t_red = time.perf_counter()
            _reduce_pass0_jit(flat_idx, values, cell_count, cell_min, cell_max, skip_counts)
            t_reduce_total += time.perf_counter() - _t_red
            n_chunks_processed += 1

    log.debug(
        "  pass0: %d zarr chunks, I/O wait %.3fs, reduce %.3fs",
        n_chunks_processed, t_io_total, t_reduce_total,
    )
    return cell_count, cell_min, cell_max


# ── Pass 1: fixed-range parallel histogram fill ─────────────────


@njit(cache=True)
def _fill_histogram_jit(
    chunk_idx: np.ndarray,
    values: np.ndarray,
    occ_lo: np.ndarray,
    occ_hi: np.ndarray,
    hist_counts: np.ndarray,
    n_bins: int,
) -> None:
    """Fill fixed-range histogram bins (no rebinning)."""
    for i in range(len(values)):
        ci = chunk_idx[i]
        lo = occ_lo[ci]
        rng = occ_hi[ci] - lo
        if rng <= 0.0:
            b = 0
        else:
            b = int((values[i] - lo) / rng * n_bins)
            if b < 0:
                b = 0
            elif b >= n_bins:
                b = n_bins - 1
        hist_counts[ci, b] += 1


def _read_chunk_for_fill(
    zarr_group: zarr.hierarchy.Group,
    start: int,
    end: int,
    cell_to_chunk_idx: NDArray[np.int64],
    N_v: int,
    threshold_grid: NDArray | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]] | None:
    """Worker: read one zarr chunk, filter to current cell chunk.

    Returns compact (chunk_idx_f, values_f) or None.
    Does NOT allocate any histogram arrays — that happens on the main
    thread during serial accumulation.
    """
    cell_u = zarr_group["cell_u"][start:end]
    cell_v = zarr_group["cell_v"][start:end]
    values = zarr_group["values"][start:end].astype(np.float64)

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

    return chunk_idx[in_chunk], values[in_chunk]


def parallel_histogram_fill(
    zarr_group: zarr.hierarchy.Group,
    chunk_cells: NDArray[np.int64],
    cell_to_chunk_idx: NDArray[np.int64],
    occ_count: NDArray[np.int64],
    cell_min: NDArray[np.float64],
    cell_max: NDArray[np.float64],
    n_bins: int,
    N_v: int,
    n_threads: int,
    threshold_grid: NDArray | None = None,
    cached_chunks: list | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None, NDArray | None]:
    """Fixed-range parallel histogram fill over zarr chunks.

    Two-phase: workers read + filter zarr chunks in parallel (returning
    compact chunk_idx + values); main thread accumulates into a single
    shared histogram array via JIT kernel + collects exact values.
    Peak memory: ONE histogram array + n_threads × ~16 MB for I/O.

    When *cached_chunks* is provided (list of (flat_idx, values) from
    pass0), zarr re-reads are skipped entirely.

    Returns
    -------
    hist_counts    : (n_chunk, n_bins) int32
    occ_lo         : (n_chunk,) float64 — per-cell lower bound
    occ_hi         : (n_chunk,) float64 — per-cell upper bound
    all_exact_cidx : concatenated chunk indices for exact-path cells, or None
    all_exact_vals : concatenated values for exact-path cells, or None
    """
    n_chunk = len(chunk_cells)

    # Per-cell ranges from pass0 (already computed globally).
    occ_lo = cell_min[chunk_cells].copy()
    occ_hi = cell_max[chunk_cells].copy()
    # Ensure hi > lo for cells with constant values.
    equal_mask = occ_lo >= occ_hi
    occ_hi[equal_mask] = occ_lo[equal_mask] + 1.0

    low_count_mask = occ_count <= _EXACT_THRESHOLD

    # Single shared histogram — never duplicated across threads.
    hist_counts = np.zeros((n_chunk, n_bins), dtype=np.int32)
    exact_cidx_parts: list[NDArray] = []
    exact_vals_parts: list[NDArray] = []

    _t_io = time.perf_counter()

    if cached_chunks is not None:
        # Fast path: reuse (flat_idx, values) from pass0 — no zarr I/O.
        for flat_idx, values in cached_chunks:
            chunk_idx = cell_to_chunk_idx[flat_idx]
            in_chunk = chunk_idx >= 0
            if not np.any(in_chunk):
                continue
            chunk_idx_f = chunk_idx[in_chunk]
            values_f = values[in_chunk]

            exact_mask = low_count_mask[chunk_idx_f]
            if np.any(exact_mask):
                exact_cidx_parts.append(chunk_idx_f[exact_mask])
                exact_vals_parts.append(values_f[exact_mask])

            _fill_histogram_jit(
                chunk_idx_f, values_f, occ_lo, occ_hi, hist_counts, n_bins,
            )

        log.debug("  fill phase (cached): %.3fs (%d cached chunks, %d cells)",
                  time.perf_counter() - _t_io, len(cached_chunks), n_chunk)
    else:
        # Standard path: parallel zarr reads + serial JIT accumulation.
        total_len = zarr_group["values"].shape[0]
        chunk_size = zarr_group["values"].chunks[0]
        chunk_ranges = [
            (i, min(i + chunk_size, total_len))
            for i in range(0, total_len, chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [
                pool.submit(
                    _read_chunk_for_fill, zarr_group, s, e,
                    cell_to_chunk_idx, N_v, threshold_grid,
                )
                for s, e in chunk_ranges
            ]
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                chunk_idx_f, values_f = result

                exact_mask = low_count_mask[chunk_idx_f]
                if np.any(exact_mask):
                    exact_cidx_parts.append(chunk_idx_f[exact_mask])
                    exact_vals_parts.append(values_f[exact_mask])

                _fill_histogram_jit(
                    chunk_idx_f, values_f, occ_lo, occ_hi, hist_counts, n_bins,
                )

        log.debug("  fill phase: %.3fs (%d zarr reads, %d cells, %d threads)",
                  time.perf_counter() - _t_io, len(chunk_ranges), n_chunk, n_threads)

    all_exact_cidx = np.concatenate(exact_cidx_parts) if exact_cidx_parts else None
    all_exact_vals = np.concatenate(exact_vals_parts) if exact_vals_parts else None

    return hist_counts, occ_lo, occ_hi, all_exact_cidx, all_exact_vals


# ── Extraction ───────────────────────────────────────────────────


@njit(parallel=True, cache=True)
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

    for i in prange(n_chunk):
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
    zarr_group: zarr.hierarchy.Group,
    grid_shape: tuple[int, int],
    n_bins: int = 256,
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
    pre_counts: NDArray[np.int64] | None = None,
    mem_budget_bytes: int | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Fixed-range parallel statistics over a consolidated zarr group.

    Pass 0 discovers per-cell counts and value ranges.  Pass 1 fills
    fixed-range histograms (no rebinning).  Low-count cells
    (≤ _EXACT_THRESHOLD) are computed exactly via segmented median/MAD.

    Returns (median_grid, std_grid, count_grid) with the same shapes
    and semantics as ``gridder.compute_cell_stats``.

    Parameters
    ----------
    zarr_group : zarr group containing cell_u, cell_v, values arrays.
    grid_shape : (N_u, N_v) grid dimensions.
    n_bins : histogram bin count.
    n_threads : thread pool size for parallel chunk processing.
    threshold_grid : if provided, only values ≤ their cell's threshold
        are included (for post-flag "after" grids).
    pre_counts : if provided, cell counts from the merge step (skips
        bincount in pass 0).
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v
    budget = mem_budget_bytes if mem_budget_bytes is not None else _HIST_MEM_BUDGET

    # Decide whether to cache pass0 chunks for fill reuse.
    # Cache cost: ~16 bytes per visibility (int64 flat_idx + float64 value).
    total_vis = zarr_group["values"].shape[0] if "values" in zarr_group else 0
    cache_bytes = total_vis * 16
    use_cache = cache_bytes < budget * 0.6  # leave room for histogram array
    chunk_cache: list | None = [] if use_cache else None
    if use_cache:
        log.debug("Caching pass0 chunks (est. %.1f GB for %d vis)",
                  cache_bytes / 1024**3, total_vis)

    # Pass 0: count + range discovery.
    _t0 = time.perf_counter()
    cell_count, cell_min, cell_max = pass0_counts_and_ranges(
        zarr_group, grid_shape, n_threads, threshold_grid, pre_counts,
        chunk_cache=chunk_cache,
    )
    log.debug("Pass0 (counts + ranges): %.3fs", time.perf_counter() - _t0)

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

    log.info(
        "Histogram: %d occupied cells (%.1f%% of %d grid cells)",
        n_occupied, 100.0 * n_occupied / max(n_cells, 1), n_cells,
    )

    # Chunk sizing: keep histogram array under budget.
    bytes_per_cell = n_bins * 4  # int32
    max_cells_per_chunk = max(1, int(budget / bytes_per_cell))
    n_chunks = math.ceil(n_occupied / max_cells_per_chunk)
    log.info(
        "Processing in %d chunk(s) (%d cells/chunk, %.1f GB budget)",
        n_chunks, max_cells_per_chunk, budget / 1024**3,
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

        # Fixed-range histogram fill for this chunk.
        _t_fill = time.perf_counter()
        hist_counts, chunk_lo, chunk_hi, all_exact_cidx, all_exact_vals = (
            parallel_histogram_fill(
                zarr_group, chunk_cells, cell_to_chunk,
                chunk_count, cell_min, cell_max,
                n_bins, N_v, n_threads, threshold_grid,
                cached_chunks=chunk_cache,
            )
        )
        log.debug("Chunk %d/%d fill:    %.3fs", chunk_i + 1, n_chunks, time.perf_counter() - _t_fill)

        del cell_to_chunk

        # Extract stats for this chunk.
        _t_ext = time.perf_counter()
        medians, stds = _extract_chunk(
            hist_counts, chunk_lo, chunk_hi, chunk_count,
            all_exact_cidx, all_exact_vals, n_bins,
        )
        log.debug("Chunk %d/%d extract: %.3fs", chunk_i + 1, n_chunks, time.perf_counter() - _t_ext)

        # Scatter into output grids.
        median_grid[chunk_cells] = medians
        std_grid[chunk_cells] = stds
        count_grid[chunk_cells] = chunk_count.astype(np.int32)

        del hist_counts, all_exact_cidx, all_exact_vals

    del chunk_cache  # free cached pass0 data

    return (
        median_grid.reshape(grid_shape),
        std_grid.reshape(grid_shape),
        count_grid.reshape(grid_shape),
    )
