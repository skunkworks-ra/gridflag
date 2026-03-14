"""Orchestrator: read → grid → threshold → flag → write."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from gridflag.config import GridFlagConfig
from gridflag.coordinates import (
    C_M_S,
    grid_shape,
)
from gridflag.flagger import flag_visibilities
from gridflag.gridder import compute_cell_stats
from gridflag.msio import (
    get_max_baseline_m,
    get_spw_info,
    read_chunks,
    resolve_data_column,
    write_flags_batched,
)
from gridflag.thresholds import (
    annular_threshold,
    combine_thresholds,
    local_neighborhood_threshold,
)
from gridflag.zarr_store import ZarrStore

log = logging.getLogger("gridflag.pipeline")


def _extract_quantity(data: np.ndarray, quantity: str) -> np.ndarray:
    """Convert complex visibilities to the requested real quantity."""
    if quantity == "amplitude":
        return np.abs(data).astype(np.float32)
    elif quantity == "phase":
        return np.angle(data).astype(np.float32)
    elif quantity == "real":
        return data.real.astype(np.float32)
    elif quantity == "imag":
        return data.imag.astype(np.float32)
    else:
        raise ValueError(f"Unknown quantity: {quantity!r}")


def _compute_global_N(
    ms_path: str,
    all_spws: list[dict],
    cell_size: float,
) -> int:
    """Determine grid half-size N from antenna positions and channel frequencies."""
    max_bl = get_max_baseline_m(ms_path)
    max_freq = max(float(np.max(s["chan_freqs"])) for s in all_spws)
    uv_max = max_bl * max_freq / C_M_S
    return int(np.ceil(uv_max / cell_size))


def run(
    ms_path: str,
    config: GridFlagConfig | None = None,
    plot_dir: str | Path | None = None,
    persist_cache: bool = False,
) -> dict:
    """Run the full GRIDflag pipeline on a measurement set.

    Parameters
    ----------
    ms_path : path to CASA Measurement Set.
    config : algorithm configuration (defaults applied if None).
    plot_dir : if set, write before/after diagnostic PNGs here.
    persist_cache : if True, keep the Zarr store after the run.
        Default is False (clean up).

    Data flow:
      1. Read MS in chunks → compute UV coords, cell indices → accumulate
         flat arrays in memory, flush once to Zarr per (spw, corr)
      2. Per (SPW, corr): load from Zarr → cell stats → thresholds → flags
      3. Batch-write flags back to MS
    """
    if config is None:
        config = GridFlagConfig()

    t0 = time.monotonic()
    log.info("GRIDflag starting on %s", ms_path)

    # Resolve data column.
    data_column = resolve_data_column(ms_path, config.data_column)
    log.info("Using data column: %s", data_column)

    # Get SPW metadata.
    all_spws = get_spw_info(ms_path)
    if config.spw_ids is not None:
        all_spws = [s for s in all_spws if s["spw_id"] in config.spw_ids]
    log.info("Processing %d SPW(s)", len(all_spws))

    # Determine global grid size.
    if config.uvrange is not None:
        uv_min, uv_max = config.uvrange
        global_N = int(np.ceil(uv_max / config.cell_size))
        log.info("UV range: %.1f – %.1f λ (user-specified)", uv_min, uv_max)
    else:
        uv_min = 0.0
        global_N = _compute_global_N(ms_path, all_spws, config.cell_size)
        uv_max = global_N * config.cell_size
    gshape = grid_shape(global_N)
    log.info("Grid shape: %s  (N=%d)", gshape, global_N)

    # Set up Zarr store.  Explicit zarr_path implies persistence.
    if config.zarr_path is not None:
        zarr_path = Path(config.zarr_path)
        persist_cache = True
    else:
        uid = uuid.uuid4().hex[:8]
        zarr_path = Path.cwd() / f"tmp_gridflag_uv_{uid}.zarr"
    store = ZarrStore(zarr_path, config, ms_path)
    store.set_grid_shape(gshape)
    log.info("Zarr store: %s", zarr_path)

    # ── Pass 1: Read MS → accumulate → flush to Zarr ────────────────
    t_read_start = time.monotonic()

    for spw_info in all_spws:
        spw_id = spw_info["spw_id"]
        n_chan = spw_info["n_chan"]
        n_corr = spw_info["n_corr"]
        ref_freq = spw_info["ref_freq"]
        chan_freqs = np.array(spw_info["chan_freqs"])
        freq_over_c = chan_freqs / C_M_S  # precompute (n_chan,)

        store.init_spw(spw_id, n_chan, n_corr, ref_freq, chan_freqs)
        log.info(
            "SPW %d: %d chan, %d corr, ref_freq=%.3f MHz",
            spw_id, n_chan, n_corr, ref_freq / 1e6,
        )

        for chunk in read_chunks(ms_path, data_column, config.chunk_size, spw_id):
            if config.field_ids is not None and chunk.field_id not in config.field_ids:
                continue

            n_row = chunk.data.shape[0]
            uvw = chunk.uvw  # (n_row, 3)

            # Per-channel UV in wavelengths: (n_row, n_chan).
            u_ch = uvw[:, 0:1] * freq_over_c[np.newaxis, :]
            v_ch = uvw[:, 1:2] * freq_over_c[np.newaxis, :]

            # Hermitian fold (in-place on our local arrays).
            neg = v_ch < 0
            u_ch[neg] = -u_ch[neg]
            v_ch[neg] = -v_ch[neg]
            # vis conjugation handled per-corr below.

            # Cell assignment with global N.
            cell_u = (np.rint(u_ch / config.cell_size).astype(np.int32) + global_N)
            cell_v = np.rint(v_ch / config.cell_size).astype(np.int32)

            # UV distance filter.
            if uv_min > 0 or config.uvrange is not None:
                uv_dist_sq = u_ch * u_ch + v_ch * v_ch
                uv_keep = (uv_dist_sq >= uv_min * uv_min) & (uv_dist_sq <= uv_max * uv_max)
            else:
                uv_keep = None

            # Pre-flatten shared arrays (computed once, used per corr).
            cell_u_flat = cell_u.ravel()
            cell_v_flat = cell_v.ravel()
            row_idx = np.repeat(chunk.row_indices, n_chan)
            chan_idx = np.tile(np.arange(n_chan, dtype=np.int32), n_row)

            for corr in range(n_corr):
                vis_corr = chunk.data[:, :, corr]  # (n_row, n_chan) complex
                # Conjugate where v was negative.
                has_neg = np.any(neg)
                if has_neg:
                    vis_corr = vis_corr.copy()
                    vis_corr[neg] = np.conj(vis_corr[neg])

                vals = _extract_quantity(vis_corr, config.quantity).ravel()
                flag_corr = chunk.flags[:, :, corr].ravel()

                # Combined keep mask.
                keep = ~flag_corr
                if uv_keep is not None:
                    keep = keep & uv_keep.ravel()

                if not np.any(keep):
                    continue

                store.append(
                    spw_id, corr,
                    row_idx[keep], chan_idx[keep],
                    cell_u_flat[keep], cell_v_flat[keep],
                    vals[keep],
                )

    # Flush all accumulated data to Zarr.
    store.flush_all()

    t_read = time.monotonic() - t_read_start
    log.info("Read + accumulate: %.1fs", t_read)

    # ── Pass 1.5: Compute statistics, thresholds, flags ─────────────
    t_compute_start = time.monotonic()

    all_flag_rows: list[np.ndarray] = []
    all_flag_chans: list[np.ndarray] = []
    all_flag_corrs: list[np.ndarray] = []

    # For optional plotting.
    grid_cache: dict[tuple[int, int], dict] = {}

    for spw_info in all_spws:
        spw_id = spw_info["spw_id"]
        n_corr = spw_info["n_corr"]

        for corr in range(n_corr):
            flat = store.load_flat(spw_id, corr)
            vals = flat["values"]
            if len(vals) == 0:
                log.info("SPW %d corr %d: no data, skipping", spw_id, corr)
                continue

            cu = flat["cell_u"].astype(np.intp)
            cv = flat["cell_v"].astype(np.intp)
            n_total = len(vals)
            log.info("SPW %d corr %d: %d unflagged visibilities", spw_id, corr, n_total)

            # Per-cell statistics (numba JIT).
            median_grid, std_grid, count_grid = compute_cell_stats(
                cu, cv, vals, gshape
            )
            store.store_grid(spw_id, corr, "median_grid", median_grid)
            store.store_grid(spw_id, corr, "std_grid", std_grid)
            store.store_grid(spw_id, corr, "count_grid", count_grid)

            # Thresholds.
            local_thr = local_neighborhood_threshold(
                median_grid, std_grid, count_grid,
                config.nsigma, config.smoothing_window,
            )
            annular_thr = annular_threshold(
                median_grid, std_grid, count_grid,
                config.cell_size, config.annulus_widths,
                config.nsigma, global_N,
            )
            threshold_grid = combine_thresholds(
                local_thr, annular_thr, count_grid,
                config.smoothing_window, config.min_neighbors,
            )
            store.store_grid(spw_id, corr, "threshold_grid", threshold_grid)

            # Flag.
            flags = flag_visibilities(cu, cv, vals, threshold_grid)
            n_flagged = int(np.sum(flags))
            log.info(
                "SPW %d corr %d: %d / %d flagged (%.2f%%)",
                spw_id, corr, n_flagged, n_total,
                100.0 * n_flagged / max(n_total, 1),
            )

            # Store flag mask for plotting.
            store.store_grid(spw_id, corr, "flag_mask", flags.astype(np.uint8))

            if plot_dir is not None:
                grid_cache[(spw_id, corr)] = {
                    "median_before": median_grid,
                    "std_before": std_grid,
                    "cu": cu, "cv": cv,
                    "vals": vals, "flags": flags,
                }

            if n_flagged > 0:
                idx = np.where(flags)[0]
                all_flag_rows.append(flat["row_indices"][idx])
                all_flag_chans.append(flat["chan_indices"][idx])
                all_flag_corrs.append(np.full(n_flagged, corr, dtype=np.int32))

    t_compute = time.monotonic() - t_compute_start
    log.info("Compute: %.1fs", t_compute)

    # ── Pass 2: Batch-write flags back to MS ────────────────────────
    t_write_start = time.monotonic()
    total_newly_flagged = 0

    if all_flag_rows:
        all_rows = np.concatenate(all_flag_rows)
        all_chans = np.concatenate(all_flag_chans)
        all_corrs = np.concatenate(all_flag_corrs)

        total_newly_flagged = write_flags_batched(
            ms_path, all_rows, all_chans, all_corrs,
        )

    t_write = time.monotonic() - t_write_start
    log.info("Write: %.1fs", t_write)

    t_total = time.monotonic() - t0
    log.info("Total newly flagged: %d  (%.1fs total)", total_newly_flagged, t_total)

    # ── Diagnostic plots ────────────────────────────────────────────
    plot_paths: list[str] = []
    if plot_dir is not None and grid_cache:
        from gridflag.plotting import plot_grids_before_after

        for (spw_id, corr), cached in grid_cache.items():
            try:
                paths = plot_grids_before_after(
                    cached["median_before"],
                    cached["std_before"],
                    cached["cu"],
                    cached["cv"],
                    cached["vals"],
                    cached["flags"],
                    gshape,
                    config.cell_size,
                    global_N,
                    spw_id,
                    corr,
                    plot_dir,
                )
                plot_paths.extend(str(p) for p in paths)
            except Exception:
                log.warning(
                    "Failed to plot SPW %d corr %d", spw_id, corr,
                    exc_info=True,
                )

    # ── Clean up Zarr store unless persisting ──────────────────────
    if not persist_cache and zarr_path.exists():
        shutil.rmtree(zarr_path)
        log.debug("Removed Zarr cache: %s", zarr_path)

    return {
        "ms_path": ms_path,
        "zarr_path": str(zarr_path) if persist_cache else None,
        "grid_shape": gshape,
        "total_newly_flagged": total_newly_flagged,
        "elapsed_s": t_total,
        "plots": plot_paths,
    }
