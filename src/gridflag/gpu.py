"""GPU runtime detection, VRAM management, and CUDA kernels.

Provides CUDA-accelerated versions of the three compute-hot kernels:
  1. _fill_histogram_cuda — histogram binning from cell-sorted values
  2. _extract_stats_cuda — median + IQR extraction from histograms
  3. _segmented_median_mad_cuda — exact median/MAD for low-count cells

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
    n_bins: int,
    headroom: float = 0.9,
) -> int:
    """Estimate how many cells we can process in one GPU batch.

    Memory model per batch of C cells covering V values:
      sorted_values slice:  8 * V  bytes  (float64)
      offsets:              8 * (C+1)     (int64)
      chunk_cells:          8 * C         (int64)
      occ_lo/hi:            16 * C        (float64 × 2)
      occ_count:            8 * C         (int64)
      hist_counts:          4 * C * n_bins (int32)
      medians/stds:         8 * C         (float32 × 2)
    Dominant term is the sorted_values slice.

    Returns the max number of cells that fit, or total_vis if everything fits.
    """
    free_raw = free_vram_bytes()
    free = int(free_raw * headroom)
    log.debug(
        "VRAM budget: %.2f GB free (%.0f%% headroom → %.2f GB usable)",
        free_raw / 1024**3, headroom * 100, free / 1024**3,
    )
    # Fixed cost: sorted_values for the whole dataset
    sv_bytes = 8 * total_vis
    if sv_bytes >= free:
        # Must sub-chunk; estimate cells per sub-chunk
        # Each sub-chunk transfers a proportional slice of sorted_values
        # plus per-cell overhead
        per_cell_overhead = 8 + 16 + 8 + 4 * n_bins + 8  # offsets, lo/hi, count, hist, out
        # Assume uniform distribution: avg_vis_per_cell
        # We'll compute exact fits in the caller; return a conservative estimate
        usable = free - 1024 * 1024  # 1 MB headroom for misc
        if usable <= 0:
            return 1
        return max(1, int(usable / (per_cell_overhead + 8 * 64)))  # rough avg 64 vis/cell
    else:
        # Everything fits; return large number
        remaining = free - sv_bytes
        per_cell = 8 + 16 + 8 + 4 * n_bins + 8
        return max(1, int(remaining / per_cell))


# ── CUDA Kernels ─────────────────────────────────────────────────
# These are compiled lazily on first call to avoid import-time CUDA init.

_compiled_kernels: dict = {}


def _get_kernels():
    """Lazily compile and cache CUDA kernels."""
    if _compiled_kernels:
        return _compiled_kernels

    from numba import cuda

    # ── Kernel 1: histogram fill ──────────────────────────────────

    @cuda.jit
    def _fill_histogram_kernel(chunk_cells, sorted_values, offsets,
                               occ_lo, occ_hi, hist_counts, n_bins):
        ci = cuda.grid(1)
        if ci >= chunk_cells.shape[0]:
            return
        cell = chunk_cells[ci]
        lo = occ_lo[ci]
        rng = occ_hi[ci] - lo
        start = offsets[cell]
        end = offsets[cell + 1]
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

    _compiled_kernels["fill_histogram"] = _fill_histogram_kernel
    _compiled_kernels["extract_stats"] = _extract_stats_kernel
    _compiled_kernels["segmented_median_mad"] = _segmented_median_mad_kernel

    return _compiled_kernels


# ── Public dispatch functions ────────────────────────────────────

_BLOCK_SIZE = 256


def fill_histogram_cuda(
    chunk_cells: NDArray[np.int64],
    sorted_values: NDArray[np.float64],
    offsets: NDArray[np.int64],
    occ_lo: NDArray[np.float64],
    occ_hi: NDArray[np.float64],
    hist_counts: NDArray[np.int32],
    n_bins: int,
) -> None:
    """GPU histogram fill — in-place into hist_counts."""
    kernels = _get_kernels()
    n_chunk = len(chunk_cells)
    grid_dim = math.ceil(n_chunk / _BLOCK_SIZE)

    d_chunk_cells = _cuda.to_device(chunk_cells)
    d_sorted = _cuda.to_device(sorted_values)
    d_offsets = _cuda.to_device(offsets)
    d_lo = _cuda.to_device(occ_lo)
    d_hi = _cuda.to_device(occ_hi)
    d_hist = _cuda.to_device(hist_counts)

    kernels["fill_histogram"][grid_dim, _BLOCK_SIZE](
        d_chunk_cells, d_sorted, d_offsets, d_lo, d_hi, d_hist, n_bins,
    )

    d_hist.copy_to_host(hist_counts)
    del d_chunk_cells, d_sorted, d_offsets, d_lo, d_hi, d_hist
    _cuda.current_context().deallocations.clear()


def extract_stats_cuda(
    hist_counts: NDArray,
    occ_min: NDArray[np.float64],
    occ_max: NDArray[np.float64],
    occ_count: NDArray[np.int64],
    n_bins: int,
    n_chunk: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """GPU stats extraction — returns (medians, stds)."""
    kernels = _get_kernels()
    grid_dim = math.ceil(n_chunk / _BLOCK_SIZE)

    medians = np.zeros(n_chunk, dtype=np.float32)
    stds = np.zeros(n_chunk, dtype=np.float32)

    d_hist = _cuda.to_device(hist_counts)
    d_min = _cuda.to_device(occ_min)
    d_max = _cuda.to_device(occ_max)
    d_count = _cuda.to_device(occ_count)
    d_med = _cuda.to_device(medians)
    d_std = _cuda.to_device(stds)

    kernels["extract_stats"][grid_dim, _BLOCK_SIZE](
        d_hist, d_min, d_max, d_count, n_bins, d_med, d_std,
    )

    d_med.copy_to_host(medians)
    d_std.copy_to_host(stds)
    del d_hist, d_min, d_max, d_count, d_med, d_std
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
