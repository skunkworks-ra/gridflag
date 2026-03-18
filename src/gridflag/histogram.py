"""Fixed-range parallel histogram-based robust statistics.

Computes per-cell median and robust std from a consolidated zarr store
without loading all visibilities into memory at once.

Fused scatter + range: scatter values into cell-sorted order AND discover
    per-cell min/max in a single zarr pass.  Requires pre-computed cell
    counts (from merge phase).  Zarr chunk reads run in parallel via
    ThreadPoolExecutor; scatter/min/max reduction is serial on the main
    thread to avoid race conditions on the shared sorted_values array.
Prange fill: bin cell-sorted values into fixed-range histograms using
    numba prange — each cell writes only to its own histogram row, so
    there are no conflicts.  Purely CPU-bound, no I/O.
Extraction: median and IQR-based robust std from cumulative histograms.
            Low-count cells (≤ _EXACT_THRESHOLD) use exact median/MAD.

Memory model: sorted_values (8 bytes × total_vis) + offsets/pos
(~16 bytes × n_cells) + one histogram chunk (max_cells × n_bins × 4).
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
from gridflag import gpu

log = logging.getLogger("gridflag.histogram")

# Cells with count <= this threshold use exact values instead of histogram.
_EXACT_THRESHOLD = 32

# Maximum memory (bytes) for the histogram array.
# Occupied cells are processed in chunks that fit within this budget.
_HIST_MEM_BUDGET = 2 * 1024**3  # 2 GB


# ── Shared zarr chunk reader ────────────────────────────────────


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


# ── Pass 0: count + range discovery (threshold_grid "after" path) ──


@njit(cache=True)
def _reduce_pass0_jit(flat_idx, values, cell_count, cell_min, cell_max):
    """Count + min/max reduction — no transient allocations."""
    for i in range(len(flat_idx)):
        idx = flat_idx[i]
        cell_count[idx] += 1
        v = values[i]
        if v < cell_min[idx]:
            cell_min[idx] = v
        if v > cell_max[idx]:
            cell_max[idx] = v


def pass0_counts_and_ranges(
    zarr_group: zarr.hierarchy.Group,
    grid_shape: tuple[int, int],
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Count + min/max per cell over consolidated zarr chunks.

    Used only for the threshold_grid "after" path where pre_counts are
    unavailable (threshold filtering changes the counts).

    Returns (cell_count, cell_min, cell_max) — all flat 1-D (n_cells,).
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v

    if "values" not in zarr_group or zarr_group["values"].shape[0] == 0:
        return (
            np.zeros(n_cells, dtype=np.int64),
            np.full(n_cells, np.inf, dtype=np.float64),
            np.full(n_cells, -np.inf, dtype=np.float64),
        )

    total_len = zarr_group["values"].shape[0]
    chunk_size = zarr_group["values"].chunks[0]
    chunk_ranges = [
        (i, min(i + chunk_size, total_len))
        for i in range(0, total_len, chunk_size)
    ]

    cell_count = np.zeros(n_cells, dtype=np.int64)
    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    cell_max = np.full(n_cells, -np.inf, dtype=np.float64)

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

            _t_red = time.perf_counter()
            _reduce_pass0_jit(flat_idx, values, cell_count, cell_min, cell_max)
            t_reduce_total += time.perf_counter() - _t_red
            n_chunks_processed += 1

    log.debug(
        "  pass0: %d zarr chunks, I/O wait %.3fs, reduce %.3fs",
        n_chunks_processed, t_io_total, t_reduce_total,
    )
    return cell_count, cell_min, cell_max


# ── Fused scatter + range discovery ──────────────────────────────


@njit(cache=True)
def _scatter_and_range_jit(flat_idx, values, sorted_values, pos,
                           cell_min, cell_max):
    """Fused scatter + min/max discovery. One pass over each chunk."""
    for i in range(len(flat_idx)):
        cell = flat_idx[i]
        v = values[i]
        # Scatter into cell-sorted position.
        sorted_values[pos[cell]] = v
        pos[cell] += 1
        # Min/max.
        if v < cell_min[cell]:
            cell_min[cell] = v
        if v > cell_max[cell]:
            cell_max[cell] = v


def fused_scatter_and_ranges(
    zarr_group: zarr.hierarchy.Group,
    grid_shape: tuple[int, int],
    cell_count: NDArray[np.int64],
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Single zarr pass: scatter values into cell-sorted order + discover min/max.

    Requires pre-computed cell_count (from merge phase or pass0).

    Returns (sorted_values, offsets, cell_min, cell_max).
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v
    total_vis = int(cell_count.sum())

    # Compute offsets via cumsum.
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(cell_count, out=offsets[1:])

    # Pre-allocate sorted output.
    sorted_values = np.empty(total_vis, dtype=np.float64)
    pos = offsets[:-1].copy()

    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    cell_max = np.full(n_cells, -np.inf, dtype=np.float64)

    if "values" not in zarr_group or zarr_group["values"].shape[0] == 0:
        return sorted_values, offsets, cell_min, cell_max

    total_len = zarr_group["values"].shape[0]
    chunk_size = zarr_group["values"].chunks[0]
    chunk_ranges = [
        (i, min(i + chunk_size, total_len))
        for i in range(0, total_len, chunk_size)
    ]

    t_io_total = 0.0
    t_scatter_total = 0.0
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

            _t_s = time.perf_counter()
            _scatter_and_range_jit(
                flat_idx, values, sorted_values, pos, cell_min, cell_max,
            )
            t_scatter_total += time.perf_counter() - _t_s
            n_chunks_processed += 1

    log.debug(
        "  fused scatter+range: %d zarr chunks, I/O wait %.3fs, scatter %.3fs",
        n_chunks_processed, t_io_total, t_scatter_total,
    )
    return sorted_values, offsets, cell_min, cell_max


# ── Prange histogram fill ────────────────────────────────────────


@njit(parallel=True, cache=True)
def _fill_histogram_sorted_jit(chunk_cells, sorted_values, offsets,
                                occ_lo, occ_hi, hist_counts, n_bins):
    """Fill histograms from cell-sorted values via prange.

    Each prange iteration writes only to hist_counts[ci, :] — no conflicts.
    Reads contiguous slices of sorted_values.
    """
    n_chunk = len(chunk_cells)
    for ci in prange(n_chunk):
        cell = chunk_cells[ci]
        lo = occ_lo[ci]
        rng = occ_hi[ci] - lo
        for j in range(offsets[cell], offsets[cell + 1]):
            v = sorted_values[j]
            if rng <= 0.0:
                b = 0
            else:
                b = int((v - lo) / rng * n_bins)
                if b < 0:
                    b = 0
                elif b >= n_bins:
                    b = n_bins - 1
            hist_counts[ci, b] += 1


def parallel_histogram_fill(
    sorted_values: NDArray[np.float64],
    offsets: NDArray[np.int64],
    chunk_cells: NDArray[np.int64],
    occ_count: NDArray[np.int64],
    cell_min: NDArray[np.float64],
    cell_max: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None, NDArray | None]:
    """CPU-only histogram fill from pre-scattered sorted_values.

    Returns
    -------
    hist_counts    : (n_chunk, n_bins) int32
    occ_lo         : (n_chunk,) float64 — per-cell lower bound
    occ_hi         : (n_chunk,) float64 — per-cell upper bound
    all_exact_cidx : concatenated chunk indices for exact-path cells, or None
    all_exact_vals : concatenated values for exact-path cells, or None
    """
    n_chunk = len(chunk_cells)

    # Per-cell ranges from fused scatter pass.
    occ_lo = cell_min[chunk_cells].copy()
    occ_hi = cell_max[chunk_cells].copy()
    # Ensure hi > lo for cells with constant values.
    equal_mask = occ_lo >= occ_hi
    occ_hi[equal_mask] = occ_lo[equal_mask] + 1.0

    low_count_mask = occ_count <= _EXACT_THRESHOLD

    # Single shared histogram.
    hist_counts = np.zeros((n_chunk, n_bins), dtype=np.int32)

    _t_fill = time.perf_counter()
    _fill_histogram_sorted_jit(
        chunk_cells, sorted_values, offsets,
        occ_lo, occ_hi, hist_counts, n_bins,
    )
    log.debug(
        "  prange fill: %.3fs (%d cells)",
        time.perf_counter() - _t_fill, n_chunk,
    )

    # Collect exact values for low-count cells by slicing sorted_values.
    low_indices = np.where(low_count_mask)[0]
    exact_cidx_parts: list[NDArray] = []
    exact_vals_parts: list[NDArray] = []
    for ci in low_indices:
        cell = chunk_cells[ci]
        vals = sorted_values[offsets[cell]:offsets[cell + 1]]
        if len(vals) > 0:
            exact_cidx_parts.append(np.full(len(vals), ci, dtype=np.int64))
            exact_vals_parts.append(vals)

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
    """CPU-only: extract median and std for one chunk of occupied cells.

    Histogram-based stats are computed for all cells via JIT; low-count
    cells (≤ _EXACT_THRESHOLD) are overridden with exact median/MAD using
    a vectorized segmented reduction.
    """
    n_chunk = len(occ_count)

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


def _apply_exact_overrides(
    medians: NDArray[np.float32],
    stds: NDArray[np.float32],
    chunk_cells: NDArray[np.int64],
    chunk_count: NDArray[np.int64],
    sorted_values: NDArray[np.float64],
    offsets: NDArray[np.int64],
    use_gpu: bool,
) -> None:
    """Override histogram stats with exact median/MAD for low-count cells.

    Collects values for cells with count ≤ _EXACT_THRESHOLD, then
    dispatches to GPU or CPU segmented median/MAD.  Modifies medians
    and stds in-place.
    """
    low_count_mask = chunk_count <= _EXACT_THRESHOLD
    low_indices = np.where(low_count_mask)[0]
    if len(low_indices) == 0:
        return

    exact_cidx_parts: list[NDArray] = []
    exact_vals_parts: list[NDArray] = []
    for ci in low_indices:
        cell = chunk_cells[ci]
        vals = sorted_values[offsets[cell]:offsets[cell + 1]]
        if len(vals) > 0:
            exact_cidx_parts.append(np.full(len(vals), ci, dtype=np.int64))
            exact_vals_parts.append(vals)

    if not exact_cidx_parts:
        return

    all_exact_cidx = np.concatenate(exact_cidx_parts)
    all_exact_vals = np.concatenate(exact_vals_parts)

    sort_order = np.argsort(all_exact_cidx, kind="stable")
    sorted_cidx = all_exact_cidx[sort_order]
    sorted_vals = all_exact_vals[sort_order].astype(np.float32)

    unique_cidx, seg_starts, seg_counts = np.unique(
        sorted_cidx, return_index=True, return_counts=True,
    )

    n_chunk = len(chunk_count)
    if use_gpu:
        ex_med, ex_std, _ = gpu.segmented_median_mad_cuda(
            sorted_vals,
            seg_starts.astype(np.int64),
            seg_counts.astype(np.int64),
            unique_cidx.astype(np.int64),
            n_chunk,
        )
    else:
        ex_med, ex_std, _ = _segmented_median_mad(
            sorted_vals,
            seg_starts.astype(np.int64),
            seg_counts.astype(np.int64),
            unique_cidx.astype(np.int64),
            n_chunk,
        )

    medians[unique_cidx] = ex_med[unique_cidx]
    stds[unique_cidx] = ex_std[unique_cidx]


# ── Convenience: full pipeline ───────────────────────────────────


def compute_cell_stats_streaming(
    zarr_group: zarr.hierarchy.Group,
    grid_shape: tuple[int, int],
    n_bins: int = 256,
    n_threads: int = 4,
    threshold_grid: NDArray | None = None,
    pre_counts: NDArray[np.int64] | None = None,
    mem_budget_bytes: int | None = None,
    device: str = "auto",
) -> tuple[NDArray, NDArray, NDArray]:
    """Fixed-range parallel statistics over a consolidated zarr group.

    When *pre_counts* is provided (main path): single fused zarr pass
    scatters values into cell-sorted order and discovers min/max.
    When *threshold_grid* is set without *pre_counts* ("after" path):
    pass0 discovers filtered counts, then fused scatter uses them.

    Histogram fill uses numba prange over cell-sorted values — purely
    CPU-bound, no I/O.  When device="gpu" or "auto" with a CUDA GPU
    available, the fill, extraction, and exact-path kernels run on GPU.

    Returns (median_grid, std_grid, count_grid).
    """
    N_u, N_v = grid_shape
    n_cells = N_u * N_v
    budget = mem_budget_bytes if mem_budget_bytes is not None else _HIST_MEM_BUDGET

    # Resolve device.
    if device == "auto":
        use_gpu = gpu.is_available()
    elif device == "gpu":
        use_gpu = True
        if not gpu.is_available():
            log.warning("GPU requested but not available; falling back to CPU")
            use_gpu = False
    else:
        use_gpu = False

    if use_gpu:
        log.info("Using CUDA GPU for histogram kernels")

    _t0 = time.perf_counter()

    # Determine cell counts.
    if pre_counts is not None:
        cell_count = pre_counts
    else:
        # "After" path or fallback: need pass0 for counts.
        cell_count, _, _ = pass0_counts_and_ranges(
            zarr_group, grid_shape, n_threads, threshold_grid,
        )
        log.debug("Pass0 (counts): %.3fs", time.perf_counter() - _t0)

    if cell_count.sum() == 0:
        return (
            np.zeros(grid_shape, dtype=np.float32),
            np.zeros(grid_shape, dtype=np.float32),
            np.zeros(grid_shape, dtype=np.int32),
        )

    # Fused scatter + range discovery (single zarr pass).
    _t_fused = time.perf_counter()
    sorted_values, offsets, cell_min, cell_max = fused_scatter_and_ranges(
        zarr_group, grid_shape, cell_count, n_threads, threshold_grid,
    )
    log.debug("Fused scatter+range: %.3fs", time.perf_counter() - _t_fused)

    # Identify occupied cells.
    occupied_cells = np.where(cell_count > 0)[0].astype(np.int64)
    n_occupied = len(occupied_cells)
    occ_count_all = cell_count[occupied_cells]

    log.info(
        "Histogram: %d occupied cells (%.1f%% of %d grid cells)",
        n_occupied, 100.0 * n_occupied / max(n_cells, 1), n_cells,
    )

    # Chunk sizing: keep histogram array under budget.
    total_vis = len(sorted_values)
    if use_gpu:
        max_cells_per_chunk = gpu.max_cells_for_vram(
            total_vis, n_occupied, n_bins,
        )
        label = "GPU VRAM"
    else:
        bytes_per_cell = n_bins * 4  # int32
        max_cells_per_chunk = max(1, int(budget / bytes_per_cell))
        label = "CPU RAM"
    n_chunks = math.ceil(n_occupied / max_cells_per_chunk)
    log.info(
        "Processing in %d chunk(s) (%d cells/chunk, %s budget)",
        n_chunks, max_cells_per_chunk, label,
    )

    # Output grids.
    median_grid = np.zeros(n_cells, dtype=np.float32)
    std_grid = np.zeros(n_cells, dtype=np.float32)
    count_grid = np.zeros(n_cells, dtype=np.int32)

    # Process occupied cells in chunks (no I/O).
    for chunk_i in range(n_chunks):
        c_start = chunk_i * max_cells_per_chunk
        c_end = min(c_start + max_cells_per_chunk, n_occupied)
        chunk_cells = occupied_cells[c_start:c_end]
        chunk_count = occ_count_all[c_start:c_end]

        if use_gpu:
            # Fused GPU path: fill + extract in single dispatch,
            # hist_counts never leaves GPU, offsets never transferred.
            occ_lo = cell_min[chunk_cells].copy()
            occ_hi = cell_max[chunk_cells].copy()
            equal_mask = occ_lo >= occ_hi
            occ_hi[equal_mask] = occ_lo[equal_mask] + 1.0

            _t_gpu = time.perf_counter()
            medians, stds = gpu.fill_and_extract_cuda(
                chunk_cells, sorted_values, offsets,
                occ_lo, occ_hi, chunk_count, n_bins,
            )
            log.debug(
                "Chunk %d/%d GPU fused: %.3fs",
                chunk_i + 1, n_chunks, time.perf_counter() - _t_gpu,
            )

            # Exact-path override for low-count cells.
            _apply_exact_overrides(
                medians, stds, chunk_cells, chunk_count,
                sorted_values, offsets, use_gpu,
            )
        else:
            # CPU path: separate fill + extract.
            _t_fill = time.perf_counter()
            hist_counts, chunk_lo, chunk_hi, all_exact_cidx, all_exact_vals = (
                parallel_histogram_fill(
                    sorted_values, offsets, chunk_cells,
                    chunk_count, cell_min, cell_max, n_bins,
                )
            )
            log.debug(
                "Chunk %d/%d fill:    %.3fs",
                chunk_i + 1, n_chunks, time.perf_counter() - _t_fill,
            )

            _t_ext = time.perf_counter()
            medians, stds = _extract_chunk(
                hist_counts, chunk_lo, chunk_hi, chunk_count,
                all_exact_cidx, all_exact_vals, n_bins,
            )
            log.debug(
                "Chunk %d/%d extract: %.3fs",
                chunk_i + 1, n_chunks, time.perf_counter() - _t_ext,
            )
            del hist_counts, all_exact_cidx, all_exact_vals

        # Scatter into output grids.
        median_grid[chunk_cells] = medians
        std_grid[chunk_cells] = stds
        count_grid[chunk_cells] = chunk_count.astype(np.int32)

    del sorted_values  # free the large scatter buffer

    return (
        median_grid.reshape(grid_shape),
        std_grid.reshape(grid_shape),
        count_grid.reshape(grid_shape),
    )
