"""Before/after visualization of median and std grids."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gridflag.gridder import compute_cell_stats
from gridflag.zarr_store import ZarrStore

log = logging.getLogger("gridflag.plotting")


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_comparison(
    before: NDArray,
    after: NDArray,
    title_before: str,
    title_after: str,
    suptitle: str,
    cell_size: float,
    N: int,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    """Plot two grids stacked vertically with shared log/symlog colour scale."""
    plt = _import_matplotlib()
    from matplotlib.colors import LogNorm, SymLogNorm

    # Mask zeros/NaN for display.
    before_masked = np.where(before == 0, np.nan, before.astype(np.float64))
    after_masked = np.where(after == 0, np.nan, after.astype(np.float64))

    all_finite = np.concatenate(
        [
            before_masked[np.isfinite(before_masked)],
            after_masked[np.isfinite(after_masked)],
        ]
    )
    if len(all_finite) == 0:
        log.warning("No finite data for %s, skipping", output_path)
        return

    data_min = float(np.min(all_finite))
    data_max = float(np.max(all_finite))

    # Choose norm: LogNorm if all positive, SymLogNorm if negatives present.
    if data_min > 0:
        norm = LogNorm(vmin=data_min, vmax=data_max)
    else:
        # linthresh: linear region around zero; use a small fraction of the range.
        linthresh = max(abs(data_min), abs(data_max)) * 1e-3
        if linthresh == 0:
            linthresh = 1.0
        norm = SymLogNorm(linthresh=linthresh, vmin=data_min, vmax=data_max)

    # Axis extents in kλ.
    N_u, N_v = before.shape
    u_extent = [-N * cell_size / 1e3, N * cell_size / 1e3]
    v_extent = [0, (N_v - 1) * cell_size / 1e3]
    extent = [u_extent[0], u_extent[1], v_extent[0], v_extent[1]]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        constrained_layout=True,
        sharex=True,
    )

    ax1.imshow(
        before_masked.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=norm,
        cmap=cmap,
    )
    ax1.set_title(title_before)
    ax1.set_ylabel("v (kλ)")

    im2 = ax2.imshow(
        after_masked.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=norm,
        cmap=cmap,
    )
    ax2.set_title(title_after)
    ax2.set_xlabel("u (kλ)")
    ax2.set_ylabel("v (kλ)")

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.9)
    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", output_path)


def plot_before_after(
    store: ZarrStore,
    spw_id: int,
    corr_id: int,
    cell_size: float,
    N: int,
    output_dir: str | Path,
) -> list[Path]:
    """Generate before/after comparison plots from ZarrStore data.

    Requires that ``flag_mask``, ``median_grid``, ``std_grid`` have been
    stored in the Zarr for this (spw, corr) pair.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gshape = store.get_grid_shape()

    median_before = store.load_grid(spw_id, corr_id, "median_grid")
    std_before = store.load_grid(spw_id, corr_id, "std_grid")

    # Recompute post-flag grids.
    flat = store.load_flat(spw_id, corr_id)
    flag_arr = store.load_grid(spw_id, corr_id, "flag_mask")
    keep = ~flag_arr.astype(bool)

    cu = flat["cell_u"][keep].astype(np.intp)
    cv = flat["cell_v"][keep].astype(np.intp)
    vals = flat["values"][keep]

    if len(vals) > 0:
        median_after, std_after, _ = compute_cell_stats(cu, cv, vals, gshape)
    else:
        median_after = np.full(gshape, np.nan, dtype=np.float32)
        std_after = np.full(gshape, np.nan, dtype=np.float32)

    prefix = f"spw{spw_id}_corr{corr_id}"
    saved = []

    median_path = output_dir / f"{prefix}_median.png"
    _plot_comparison(
        median_before,
        median_after,
        "Median (before)",
        "Median (after)",
        f"Median grid — SPW {spw_id}, Corr {corr_id}",
        cell_size,
        N,
        median_path,
    )
    saved.append(median_path)

    std_path = output_dir / f"{prefix}_std.png"
    _plot_comparison(
        std_before,
        std_after,
        "Robust σ (before)",
        "Robust σ (after)",
        f"Robust σ grid — SPW {spw_id}, Corr {corr_id}",
        cell_size,
        N,
        std_path,
    )
    saved.append(std_path)

    return saved


def plot_grids_from_arrays(
    median_before: NDArray,
    std_before: NDArray,
    median_after: NDArray,
    std_after: NDArray,
    cell_size: float,
    N: int,
    spw_id: int,
    corr_id: int,
    output_dir: str | Path,
) -> list[Path]:
    """Generate before/after plots from pre-computed grid arrays (no per-vis data)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"spw{spw_id}_corr{corr_id}"
    saved = []

    median_path = output_dir / f"{prefix}_median.png"
    _plot_comparison(
        median_before,
        median_after,
        "Median (before)",
        "Median (after)",
        f"Median grid — SPW {spw_id}, Corr {corr_id}",
        cell_size,
        N,
        median_path,
    )
    saved.append(median_path)

    std_path = output_dir / f"{prefix}_std.png"
    _plot_comparison(
        std_before,
        std_after,
        "Robust σ (before)",
        "Robust σ (after)",
        f"Robust σ grid — SPW {spw_id}, Corr {corr_id}",
        cell_size,
        N,
        std_path,
    )
    saved.append(std_path)

    return saved


def plot_grids_before_after(
    median_before: NDArray,
    std_before: NDArray,
    cu: NDArray,
    cv: NDArray,
    vals: NDArray,
    flags: NDArray,
    gshape: tuple[int, int],
    cell_size: float,
    N: int,
    spw_id: int,
    corr_id: int,
    output_dir: str | Path,
) -> list[Path]:
    """Generate before/after plots from pre-computed arrays (no Zarr I/O)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep = ~flags
    if np.any(keep):
        median_after, std_after, _ = compute_cell_stats(
            cu[keep],
            cv[keep],
            vals[keep],
            gshape,
        )
    else:
        median_after = np.full(gshape, np.nan, dtype=np.float32)
        std_after = np.full(gshape, np.nan, dtype=np.float32)

    prefix = f"spw{spw_id}_corr{corr_id}"
    saved = []

    median_path = output_dir / f"{prefix}_median.png"
    _plot_comparison(
        median_before,
        median_after,
        "Median (before)",
        "Median (after)",
        f"Median grid — SPW {spw_id}, Corr {corr_id}",
        cell_size,
        N,
        median_path,
    )
    saved.append(median_path)

    std_path = output_dir / f"{prefix}_std.png"
    _plot_comparison(
        std_before,
        std_after,
        "Robust σ (before)",
        "Robust σ (after)",
        f"Robust σ grid — SPW {spw_id}, Corr {corr_id}",
        cell_size,
        N,
        std_path,
    )
    saved.append(std_path)

    return saved
