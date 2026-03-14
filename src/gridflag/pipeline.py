"""Orchestrator: read → grid → threshold → flag → write."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from gridflag.config import GridFlagConfig
from gridflag.coordinates import (
    C_M_S,
    grid_shape,
    hermitian_fold,
    scale_uv,
    uv_to_cell,
)
from gridflag.flagger import flag_visibilities
from gridflag.gridder import compute_cell_stats
from gridflag.msio import (
    get_max_baseline_m,
    get_spw_info,
    read_chunks,
    resolve_data_column,
    write_flags,
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
    """Determine grid half-size N from antenna positions and channel frequencies.

    ``uv_max = max_baseline_m × max_freq / c``, derived entirely from
    the ANTENNA and SPECTRAL_WINDOW metadata tables — no visibility read.
    """
    max_bl = get_max_baseline_m(ms_path)
    max_freq = max(float(np.max(s["chan_freqs"])) for s in all_spws)
    uv_max = max_bl * max_freq / C_M_S
    N = int(np.ceil(uv_max / cell_size))
    return N


def run(
    ms_path: str,
    config: GridFlagConfig | None = None,
    plot_dir: str | Path | None = None,
) -> dict:
    """Run the full GRIDflag pipeline on a measurement set.

    Parameters
    ----------
    ms_path : path to CASA Measurement Set.
    config : algorithm configuration (defaults applied if None).
    plot_dir : if set, write before/after diagnostic PNGs here.

    Returns a summary dict with flag statistics.
    """
    if config is None:
        config = GridFlagConfig()

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
        uv_max = global_N * config.cell_size  # effective max
    gshape = grid_shape(global_N)
    log.info("Grid shape: %s  (N=%d)", gshape, global_N)

    # Set up Zarr store.
    if config.zarr_path is not None:
        zarr_path = Path(config.zarr_path)
    else:
        zarr_path = Path(tempfile.mkdtemp(prefix="gridflag_")) / "store.zarr"
    store = ZarrStore(zarr_path, config, ms_path)
    store.set_grid_shape(gshape)
    log.info("Zarr store: %s", zarr_path)

    # ── Pass 1: Read MS → accumulate into Zarr ──────────────────────
    for spw_info in all_spws:
        spw_id = spw_info["spw_id"]
        n_chan = spw_info["n_chan"]
        n_corr = spw_info["n_corr"]
        ref_freq = spw_info["ref_freq"]
        chan_freqs = np.array(spw_info["chan_freqs"])

        store.init_spw(spw_id, n_chan, n_corr, ref_freq, chan_freqs)
        log.info(
            "SPW %d: %d chan, %d corr, ref_freq=%.3f MHz",
            spw_id, n_chan, n_corr, ref_freq / 1e6,
        )

        for chunk in read_chunks(ms_path, data_column, config.chunk_size, spw_id):
            if config.field_ids is not None and chunk.field_id not in config.field_ids:
                continue

            u_ch, v_ch, _ = scale_uv(chunk.uvw, chan_freqs, ref_freq)
            n_row = chunk.data.shape[0]

            for corr in range(n_corr):
                vis_corr = chunk.data[:, :, corr]
                flag_corr = chunk.flags[:, :, corr]

                u_f, v_f, vis_f = hermitian_fold(u_ch, v_ch, vis_corr)
                cell_u, cell_v, _ = uv_to_cell(u_f, v_f, config.cell_size, N=global_N)

                vals = _extract_quantity(vis_f, config.quantity)

                row_idx = np.repeat(chunk.row_indices, n_chan)
                chan_idx = np.tile(np.arange(n_chan, dtype=np.int32), n_row)

                cell_u_flat = cell_u.ravel().astype(np.int32)
                cell_v_flat = cell_v.ravel().astype(np.int32)
                vals_flat = vals.ravel()

                keep = ~flag_corr.ravel()

                # UV distance filter.
                uv_dist = np.sqrt(u_f**2 + v_f**2).ravel()
                keep &= (uv_dist >= uv_min) & (uv_dist <= uv_max)
                store.append(
                    spw_id, corr,
                    row_idx[keep], chan_idx[keep],
                    cell_u_flat[keep], cell_v_flat[keep],
                    vals_flat[keep],
                )

    # ── Pass 1.5: Compute statistics and thresholds ─────────────────
    all_flags: list[dict] = []

    for spw_info in all_spws:
        spw_id = spw_info["spw_id"]
        n_corr = spw_info["n_corr"]

        for corr in range(n_corr):
            flat = store.load_flat(spw_id, corr)
            if len(flat["values"]) == 0:
                log.info("SPW %d corr %d: no data, skipping", spw_id, corr)
                continue

            cu = flat["cell_u"].astype(np.intp)
            cv = flat["cell_v"].astype(np.intp)
            vals = flat["values"]

            median_grid, std_grid, count_grid = compute_cell_stats(
                cu, cv, vals, gshape
            )
            store.store_grid(spw_id, corr, "median_grid", median_grid)
            store.store_grid(spw_id, corr, "std_grid", std_grid)
            store.store_grid(spw_id, corr, "count_grid", count_grid)

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

            flags = flag_visibilities(cu, cv, vals, threshold_grid)
            n_flagged = int(np.sum(flags))
            n_total = len(flags)
            log.info(
                "SPW %d corr %d: %d / %d flagged (%.2f%%)",
                spw_id, corr, n_flagged, n_total,
                100.0 * n_flagged / max(n_total, 1),
            )

            # Store flag mask in Zarr for post-flag grid recomputation.
            store.store_grid(spw_id, corr, "flag_mask", flags.astype(np.uint8))

            if n_flagged > 0:
                idx = np.where(flags)[0]
                all_flags.append({
                    "row_indices": flat["row_indices"][idx],
                    "chan_indices": flat["chan_indices"][idx],
                    "corr_id": corr,
                    "spw_id": spw_id,
                    "n_flagged": n_flagged,
                })

    # ── Pass 2: Write flags back to MS ──────────────────────────────
    total_newly_flagged = 0
    for rec in all_flags:
        corr_arr = np.full(len(rec["row_indices"]), rec["corr_id"], dtype=np.int32)
        flag_vals = np.ones(len(rec["row_indices"]), dtype=bool)
        n = write_flags(
            ms_path,
            rec["row_indices"],
            rec["chan_indices"],
            corr_arr,
            flag_vals,
        )
        total_newly_flagged += n

    log.info("Total newly flagged: %d", total_newly_flagged)

    # ── Diagnostic plots ────────────────────────────────────────────
    plot_paths: list[str] = []
    if plot_dir is not None:
        from gridflag.plotting import plot_before_after

        for spw_info in all_spws:
            spw_id = spw_info["spw_id"]
            for corr in range(spw_info["n_corr"]):
                try:
                    paths = plot_before_after(
                        store, spw_id, corr,
                        config.cell_size, global_N,
                        plot_dir,
                    )
                    plot_paths.extend(str(p) for p in paths)
                except Exception:
                    log.warning(
                        "Failed to plot SPW %d corr %d", spw_id, corr,
                        exc_info=True,
                    )

    return {
        "ms_path": ms_path,
        "zarr_path": str(zarr_path),
        "grid_shape": gshape,
        "total_newly_flagged": total_newly_flagged,
        "plots": plot_paths,
    }
