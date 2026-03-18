"""GPU runtime detection, VRAM management, and CUDA kernels.

Provides CUDA-accelerated kernels:
  1. fill_and_extract_cuda — fused histogram fill + stats extraction
     (hist_counts never leaves GPU; offsets never transferred)
  2. segmented_median_mad_cuda — exact median/MAD for low-count cells

All kernels have CPU fallbacks; calling code selects based on is_available().
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("gridflag.gpu")

_cuda = None  # lazy import
_HAS_CUDA: bool | None = None  # tri-state: None = not checked yet


def is_available() -> bool:
    """Check if a CUDA GPU is available (cached after first call)."""
    global _HAS_CUDA, _cuda
    if _HAS_CUDA is not None:
        return _HAS_CUDA
    try:
        from numba import cuda
        _cuda = cuda
        _HAS_CUDA = cuda.is_available()
    except Exception:
        _HAS_CUDA = False
    if _HAS_CUDA:
        dev = _cuda.get_current_device()
        log.info("CUDA device: %s", dev.name.decode() if isinstance(dev.name, bytes) else dev.name)
    else:
        log.debug("No CUDA device available; using CPU kernels")
    return _HAS_CUDA


def free_vram_bytes() -> int:
    """Return free VRAM in bytes on the current device."""
    ctx = _cuda.current_context()
    free, _total = ctx.get_memory_info()
    return int(free)


def max_cells_for_vram(
    total_vis: int,
    n_occupied: int,
    n_bins: int,
    headroom: float = 0.85,
) -> int:
    """Estimate how many cells we can process in one GPU batch.

    Peak VRAM during fused fill+extract (all coexist on device):
      sorted_values slice:  8 * C * (total_vis / n_occupied)  (float64, proportional)
      chunk_starts/ends:    16 * C                             (int64 × 2)
      occ_lo/hi:            16 * C                             (float64 × 2)
      occ_count:            8 * C                              (int64)
      hist_counts:          4 * C * n_bins                     (int32)
      medians/stds:         8 * C                              (float32 × 2)

    No fixed offsets array — compact starts/ends replace it.
    """
    free_raw = free_vram_bytes()
    free = int(free_raw * headroom)
    log.info(
        "VRAM: %.2f GB free, %.0f%% headroom → %.2f GB usable",
        free_raw / 1024**3, headroom * 100, free / 1024**3,
    )

    # Per-cell cost: proportional sorted_values slice + hist + metadata.
    avg_vis_per_cell = total_vis / max(n_occupied, 1)
    per_cell = (
        int(8 * avg_vis_per_cell)  # sorted_values slice (float64)
        + 4 * n_bins               # hist_counts (int32)
        + 16                       # chunk_starts + chunk_ends (int64 × 2)
        + 16                       # occ_lo + occ_hi (float64 × 2)
        + 8                        # occ_count (int64)
        + 8                        # medians + stds (float32 × 2)
    )

    max_c = max(1, int(free / per_cell))
    log.info(
        "VRAM budget: %.0f B/cell (%.0f vis/cell avg), max %d cells/chunk",
        per_cell, avg_vis_per_cell, max_c,
    )
    return max_c


# ── CUDA Kernels ─────────────────────────────────────────────────
# These are compiled lazily on first call to avoid import-time CUDA init.

_compiled_kernels: dict = {}


def _get_kernels():
    """Lazily compile and cache CUDA kernels."""
    if _compiled_kernels:
        return _compiled_kernels

    from numba import cuda

    # ── Kernel 1: histogram fill (v2, compact starts/ends) ────────

    @cuda.jit
    def _fill_histogram_kernel_v2(sorted_values, chunk_starts, chunk_ends,
                                  occ_lo, occ_hi, hist_counts, n_bins):
        ci = cuda.grid(1)
        if ci >= chunk_starts.shape[0]:
            return
        start = chunk_starts[ci]
        end = chunk_ends[ci]
        lo = occ_lo[ci]
        rng = occ_hi[ci] - lo
        for j in range(start, end):
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

    # ── Kernel 2: stats extraction ────────────────────────────────

    @cuda.jit(device=True)
    def _interpolate_quantile_device(cumsum, rank, lo, bin_width, n_bins):
        for b in range(n_bins):
            if cumsum[b] >= rank:
                prev_cum = 0 if b == 0 else cumsum[b - 1]
                count_in_bin = cumsum[b] - prev_cum
                if count_in_bin == 0:
                    return lo + (b + 0.5) * bin_width
                fraction = (rank - prev_cum) / count_in_bin
                return lo + (b + fraction) * bin_width
        return lo + n_bins * bin_width

    @cuda.jit
    def _extract_stats_kernel(hist_counts, occ_min, occ_max, occ_count,
                              n_bins, medians, stds):
        i = cuda.grid(1)
        if i >= occ_count.shape[0]:
            return
        n = occ_count[i]
        if n == 0:
            return

        lo = occ_min[i]
        hi = occ_max[i]
        bin_width = (hi - lo) / n_bins

        # Cumulative sum in local memory.
        cumsum = cuda.local.array(256, dtype=np.int64)
        cumsum[0] = hist_counts[i, 0]
        for b in range(1, n_bins):
            cumsum[b] = cumsum[b - 1] + hist_counts[i, b]

        # Median.
        median_rank = n / 2.0
        med = _interpolate_quantile_device(cumsum, median_rank, lo, bin_width, n_bins)
        medians[i] = np.float32(med)

        # IQR.
        q1 = _interpolate_quantile_device(cumsum, n / 4.0, lo, bin_width, n_bins)
        q3 = _interpolate_quantile_device(cumsum, 3.0 * n / 4.0, lo, bin_width, n_bins)
        stds[i] = np.float32((q3 - q1) / 1.3490)

    # ── Kernel 3: segmented median/MAD ────────────────────────────

    @cuda.jit
    def _segmented_median_mad_kernel(vals, seg_starts, seg_counts,
                                     unique_cells, median_flat, std_flat,
                                     count_flat):
        i = cuda.grid(1)
        if i >= seg_starts.shape[0]:
            return
        s = seg_starts[i]
        n = seg_counts[i]
        cell = unique_cells[i]
        if n == 0:
            return

        # Copy into local array and insertion-sort (n <= 32).
        seg = cuda.local.array(32, dtype=np.float32)
        for j in range(n):
            seg[j] = vals[s + j]
        # Insertion sort.
        for j in range(1, n):
            key = seg[j]
            k = j - 1
            while k >= 0 and seg[k] > key:
                seg[k + 1] = seg[k]
                k -= 1
            seg[k + 1] = key

        # Median.
        if n % 2 == 1:
            med = seg[n // 2]
        else:
            med = (seg[n // 2 - 1] + seg[n // 2]) * 0.5

        # MAD.
        absdev = cuda.local.array(32, dtype=np.float32)
        for j in range(n):
            d = seg[j] - med
            absdev[j] = d if d >= 0.0 else -d
        # Insertion sort absdev.
        for j in range(1, n):
            key = absdev[j]
            k = j - 1
            while k >= 0 and absdev[k] > key:
                absdev[k + 1] = absdev[k]
                k -= 1
            absdev[k + 1] = key

        if n % 2 == 1:
            mad = absdev[n // 2]
        else:
            mad = (absdev[n // 2 - 1] + absdev[n // 2]) * 0.5

        median_flat[cell] = med
        std_flat[cell] = np.float32(1.4826) * mad
        count_flat[cell] = n

    _compiled_kernels["fill_histogram_v2"] = _fill_histogram_kernel_v2
    _compiled_kernels["extract_stats"] = _extract_stats_kernel
    _compiled_kernels["segmented_median_mad"] = _segmented_median_mad_kernel

    return _compiled_kernels


# ── Public dispatch functions ────────────────────────────────────

_BLOCK_SIZE = 256


def fill_and_extract_cuda(
    chunk_cells: NDArray[np.int64],
    sorted_values: NDArray[np.float64],
    offsets: NDArray[np.int64],
    occ_lo: NDArray[np.float64],
    occ_hi: NDArray[np.float64],
    occ_count: NDArray[np.int64],
    n_bins: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Fused GPU histogram fill + stats extraction.

    hist_counts never leaves the GPU. The full offsets array is never
    transferred — compact chunk_starts/chunk_ends are computed on CPU
    and sent instead.

    Returns (medians, stds).
    """
    kernels = _get_kernels()
    n_chunk = len(chunk_cells)
    grid_dim = math.ceil(n_chunk / _BLOCK_SIZE)

    # Compute the contiguous sorted_values slice for this chunk.
    sv_start = int(offsets[chunk_cells[0]])
    sv_end = int(offsets[chunk_cells[-1] + 1])
    sv_slice = sorted_values[sv_start:sv_end]

    # Compact starts/ends relative to the slice (n_chunk-sized, not n_cells).
    chunk_starts = offsets[chunk_cells] - sv_start
    chunk_ends = offsets[chunk_cells + 1] - sv_start

    log.debug(
        "  GPU H2D: sorted_values slice %.2f GB (of %.2f GB total), "
        "starts/ends %.2f MB",
        sv_slice.nbytes / 1024**3, sorted_values.nbytes / 1024**3,
        (chunk_starts.nbytes + chunk_ends.nbytes) / 1024**2,
    )

    # Allocate all device arrays once.
    d_sorted = _cuda.to_device(sv_slice)
    d_starts = _cuda.to_device(chunk_starts)
    d_ends = _cuda.to_device(chunk_ends)
    d_lo = _cuda.to_device(occ_lo)
    d_hi = _cuda.to_device(occ_hi)
    d_count = _cuda.to_device(occ_count)
    medians = np.zeros(n_chunk, dtype=np.float32)
    stds = np.zeros(n_chunk, dtype=np.float32)
    d_hist = _cuda.to_device(np.zeros((n_chunk, n_bins), dtype=np.int32))
    d_med = _cuda.to_device(medians)
    d_std = _cuda.to_device(stds)

    # Launch fill kernel.
    kernels["fill_histogram_v2"][grid_dim, _BLOCK_SIZE](
        d_sorted, d_starts, d_ends, d_lo, d_hi, d_hist, n_bins,
    )
    _cuda.synchronize()

    # Launch extract kernel (d_hist stays on GPU).
    kernels["extract_stats"][grid_dim, _BLOCK_SIZE](
        d_hist, d_lo, d_hi, d_count, n_bins, d_med, d_std,
    )
    _cuda.synchronize()

    # D2H: only medians and stds.
    d_med.copy_to_host(medians)
    d_std.copy_to_host(stds)

    del d_sorted, d_starts, d_ends, d_lo, d_hi, d_count, d_hist, d_med, d_std
    _cuda.current_context().deallocations.clear()
    return medians, stds


def segmented_median_mad_cuda(
    vals_sorted: NDArray[np.float32],
    seg_starts: NDArray[np.int64],
    seg_counts: NDArray[np.int64],
    unique_cells: NDArray[np.int64],
    n_cells: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
    """GPU segmented median/MAD — returns (median_flat, std_flat, count_flat)."""
    kernels = _get_kernels()
    n_segs = len(seg_starts)
    grid_dim = math.ceil(n_segs / _BLOCK_SIZE)

    median_flat = np.zeros(n_cells, dtype=np.float32)
    std_flat = np.zeros(n_cells, dtype=np.float32)
    count_flat = np.zeros(n_cells, dtype=np.int32)

    d_vals = _cuda.to_device(vals_sorted)
    d_starts = _cuda.to_device(seg_starts)
    d_counts = _cuda.to_device(seg_counts)
    d_cells = _cuda.to_device(unique_cells)
    d_med = _cuda.to_device(median_flat)
    d_std = _cuda.to_device(std_flat)
    d_cnt = _cuda.to_device(count_flat)

    kernels["segmented_median_mad"][grid_dim, _BLOCK_SIZE](
        d_vals, d_starts, d_counts, d_cells, d_med, d_std, d_cnt,
    )

    d_med.copy_to_host(median_flat)
    d_std.copy_to_host(std_flat)
    d_cnt.copy_to_host(count_flat)
    del d_vals, d_starts, d_counts, d_cells, d_med, d_std, d_cnt
    _cuda.current_context().deallocations.clear()
    return median_flat, std_flat, count_flat
