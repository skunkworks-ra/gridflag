"""Orchestrator: read → grid → threshold → flag → write."""

from __future__ import annotations

import logging
import multiprocessing
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr

from gridflag.config import GridFlagConfig
from gridflag.coordinates import (
    C_M_S,
    grid_shape,
)
from gridflag.flagger import flag_visibilities
from gridflag.histogram import compute_cell_stats_streaming
from gridflag.msio import (
    available_cpus,
    available_memory_gb,
    compute_row_chunks,
    get_max_baseline_m,
    get_ms_row_count,
    get_spw_info,
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


# ── Parallel worker ─────────────────────────────────────────────


def _process_chunk_worker(args: tuple) -> str:
    """Worker: read a row range from the MS, compute UV/cell/quantity, write zarr shard.

    Each worker opens its own casatools.table instance, reads its
    assigned non-overlapping row range, processes all SPWs/corrs found
    in that range, and writes flat arrays to its own zarr shard.

    Parameters
    ----------
    args : tuple
        (ms_path, data_column, startrow, nrow, chunk_index, shard_dir,
         spw_lookup, global_N, cell_size, quantity, field_ids,
         uv_min, uv_max)

    Returns
    -------
    str
        Path to the written zarr shard.
    """
    (
        ms_path,
        data_column,
        startrow,
        nrow,
        chunk_index,
        shard_dir,
        spw_lookup,
        global_N,
        cell_size,
        quantity,
        field_ids,
        uv_min,
        uv_max,
    ) = args

    import casatools  # type: ignore[import-untyped]

    from gridflag.msio import _read_column

    tb = casatools.table()
    tb.open(ms_path, nomodify=True)

    # Read raw arrays for this row range.
    raw_data = _read_column(tb, data_column, startrow, nrow)
    # casatools returns (n_corr, n_chan, n_row)
    data = raw_data.transpose(2, 1, 0)  # → (n_row, n_chan, n_corr)

    flags_raw = tb.getcol("FLAG", startrow=startrow, nrow=nrow)
    flags = flags_raw.transpose(2, 1, 0).astype(bool)

    uvw = tb.getcol("UVW", startrow=startrow, nrow=nrow).T  # → (n_row, 3)
    ddid = tb.getcol("DATA_DESC_ID", startrow=startrow, nrow=nrow)
    field_col = tb.getcol("FIELD_ID", startrow=startrow, nrow=nrow)

    tb.close()

    # Absolute row indices for flag write-back.
    abs_rows = np.arange(startrow, startrow + nrow, dtype=np.int64)

    # Open a zarr shard for this chunk.
    shard_path = str(Path(shard_dir) / f"chunk_{chunk_index}.zarr")
    root = zarr.open(shard_path, mode="w")

    has_uv_filter = uv_min > 0 or uv_max < np.inf

    # Process each SPW present in this row range.
    unique_ddids = np.unique(ddid)
    for spw_id_val in unique_ddids:
        spw_id = int(spw_id_val)
        if spw_id not in spw_lookup:
            continue

        spw_info = spw_lookup[spw_id]
        n_corr = spw_info["n_corr"]
        chan_freqs = np.array(spw_info["chan_freqs"])
        freq_over_c = chan_freqs / C_M_S
        n_chan = len(chan_freqs)

        # Select rows for this SPW.
        spw_mask = ddid == spw_id
        spw_data = data[spw_mask]
        spw_flags = flags[spw_mask]
        spw_uvw = uvw[spw_mask]
        spw_abs_rows = abs_rows[spw_mask]
        spw_fields = field_col[spw_mask]

        # Field filter.
        if field_ids is not None:
            field_keep = np.isin(spw_fields, field_ids)
            if not np.any(field_keep):
                continue
            spw_data = spw_data[field_keep]
            spw_flags = spw_flags[field_keep]
            spw_uvw = spw_uvw[field_keep]
            spw_abs_rows = spw_abs_rows[field_keep]

        n_row_spw = spw_data.shape[0]

        # Per-channel UV in wavelengths: (n_row, n_chan).
        u_ch = spw_uvw[:, 0:1] * freq_over_c[np.newaxis, :]
        v_ch = spw_uvw[:, 1:2] * freq_over_c[np.newaxis, :]

        # Hermitian fold.
        neg = v_ch < 0
        u_ch[neg] = -u_ch[neg]
        v_ch[neg] = -v_ch[neg]

        # Cell assignment.
        cell_u = np.rint(u_ch / cell_size).astype(np.int32) + global_N
        cell_v = np.rint(v_ch / cell_size).astype(np.int32)

        # UV distance filter.
        if has_uv_filter:
            uv_dist_sq = u_ch * u_ch + v_ch * v_ch
            uv_keep = (uv_dist_sq >= uv_min * uv_min) & (uv_dist_sq <= uv_max * uv_max)
        else:
            uv_keep = None

        # Pre-flatten shared arrays.
        cell_u_flat = cell_u.ravel()
        cell_v_flat = cell_v.ravel()
        row_idx = np.repeat(spw_abs_rows, n_chan)
        chan_idx = np.tile(np.arange(n_chan, dtype=np.int32), n_row_spw)

        for corr in range(n_corr):
            vis_corr = spw_data[:, :, corr]
            has_neg = np.any(neg)
            if has_neg:
                vis_corr = vis_corr.copy()
                vis_corr[neg] = np.conj(vis_corr[neg])

            vals = _extract_quantity(vis_corr, quantity).ravel()
            flag_corr = spw_flags[:, :, corr].ravel()

            keep = ~flag_corr
            if uv_keep is not None:
                keep = keep & uv_keep.ravel()

            if not np.any(keep):
                continue

            # Write flat arrays to zarr shard.
            grp = root.require_group(f"spw_{spw_id}/corr_{corr}")
            grp.array("row_indices", row_idx[keep], overwrite=True)
            grp.array("chan_indices", chan_idx[keep], overwrite=True)
            grp.array("cell_u", cell_u_flat[keep], overwrite=True)
            grp.array("cell_v", cell_v_flat[keep], overwrite=True)
            grp.array("values", vals[keep], overwrite=True)

    return shard_path


# ── Shard indexing and streaming flag pass ──────────────────────


def _index_shards(shard_paths: list[str]) -> dict[tuple[int, int], list[str]]:
    """Pre-scan shard zarr stores, return {(spw_id, corr_id): [shard_path, ...]}."""
    index: dict[tuple[int, int], list[str]] = {}
    for sp in shard_paths:
        root = zarr.open(sp, mode="r")
        for spw_key in root:
            spw_id = int(spw_key.split("_")[1])
            spw_grp = root[spw_key]
            for corr_key in spw_grp:
                corr_id = int(corr_key.split("_")[1])
                index.setdefault((spw_id, corr_id), []).append(sp)
    return index


def _flag_one_shard(
    shard_path: str,
    spw_key: str,
    corr_key: str,
    threshold_grid: np.ndarray,
) -> dict:
    """Read one shard, flag visibilities, return flag indices."""
    root = zarr.open(shard_path, mode="r")
    try:
        grp = root[spw_key][corr_key]
    except KeyError:
        return {"n_flagged": 0, "flag_rows": None, "flag_chans": None}

    cu = grp["cell_u"][:].astype(np.intp)
    cv = grp["cell_v"][:].astype(np.intp)
    vals = grp["values"][:]
    row_indices = grp["row_indices"][:]
    chan_indices = grp["chan_indices"][:]

    if len(vals) == 0:
        return {"n_flagged": 0, "flag_rows": None, "flag_chans": None}

    flags = flag_visibilities(cu, cv, vals, threshold_grid)
    n_flagged = int(np.sum(flags))

    flag_rows = None
    flag_chans = None
    if n_flagged > 0:
        idx = np.where(flags)[0]
        flag_rows = row_indices[idx]
        flag_chans = chan_indices[idx]

    return {
        "n_flagged": n_flagged,
        "flag_rows": flag_rows,
        "flag_chans": flag_chans,
    }


def _flag_shards(
    shard_paths: list[str],
    spw_key: str,
    corr_key: str,
    threshold_grid: np.ndarray,
    n_threads: int,
) -> dict:
    """Flag visibilities across all shards for one (spw, corr).

    Returns dict with keys: n_flagged, flag_rows, flag_chans.
    """
    all_flag_rows: list[np.ndarray] = []
    all_flag_chans: list[np.ndarray] = []
    total_flagged = 0

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(
                _flag_one_shard, sp, spw_key, corr_key, threshold_grid,
            ): sp
            for sp in shard_paths
        }
        for fut in as_completed(futures):
            result = fut.result()
            total_flagged += result["n_flagged"]
            if result["flag_rows"] is not None:
                all_flag_rows.append(result["flag_rows"])
                all_flag_chans.append(result["flag_chans"])

    return {
        "n_flagged": total_flagged,
        "flag_rows": np.concatenate(all_flag_rows) if all_flag_rows else None,
        "flag_chans": np.concatenate(all_flag_chans) if all_flag_chans else None,
    }


# ── Main pipeline ───────────────────────────────────────────────


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
      1. Parallel read MS → compute UV coords, cell indices → each worker
         writes its own zarr shard
      2. Per (SPW, corr): streaming histogram stats over shards →
         thresholds → streaming flag pass over shards
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

    # Build SPW lookup for workers (must be picklable).
    spw_lookup = {
        s["spw_id"]: {
            "n_chan": s["n_chan"],
            "n_corr": s["n_corr"],
            "ref_freq": s["ref_freq"],
            "chan_freqs": s["chan_freqs"].tolist(),
        }
        for s in all_spws
    }

    # Determine global grid size.
    if config.uvrange is not None:
        uv_min, uv_max = config.uvrange
        global_N = int(np.ceil(uv_max / config.cell_size))
        log.info("UV range: %.1f – %.1f λ (user-specified)", uv_min, uv_max)
    else:
        uv_min = 0.0
        global_N = _compute_global_N(ms_path, all_spws, config.cell_size)
        uv_max = float(global_N * config.cell_size)
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

    # Initialise SPW groups in the store.
    for spw_info in all_spws:
        store.init_spw(
            spw_info["spw_id"],
            spw_info["n_chan"],
            spw_info["n_corr"],
            spw_info["ref_freq"],
            np.array(spw_info["chan_freqs"]),
        )

    # ── Pass 1: Parallel read MS → zarr shards ─────────────────────
    t_read_start = time.monotonic()

    n_workers = config.n_workers if config.n_workers > 0 else available_cpus()
    n_workers = min(n_workers, 8)  # cap at 8 to avoid CASA lock contention

    # Memory-budgeted chunk sizing: ensure each worker's peak allocation
    # fits within available RAM.  Per row, a worker holds:
    #   DATA (complex128 internally in CASA, copied to complex64: ~16B × n_corr × n_chan)
    #   FLAG (1B × n_corr × n_chan) + UVW (24B)
    #   + derived arrays (u_ch, v_ch, cell_u, cell_v, vals ~5 × 4-8B × n_chan)
    #   + CASA internal getcol buffers (another ~1× DATA copy)
    # Safe multiplier: n_corr × n_chan × 120 bytes/row.
    max_nchan = max(s["n_chan"] for s in all_spws)
    max_ncorr = max(s["n_corr"] for s in all_spws)
    bytes_per_row = max_ncorr * max_nchan * 120
    # Reserve 20% of RAM for OS + main process.
    usable_mem_bytes = available_memory_gb() * 0.8 * 1024**3
    mem_per_worker = usable_mem_bytes / n_workers
    rows_per_chunk = max(1, int(mem_per_worker / bytes_per_row))

    total_rows = get_ms_row_count(ms_path)
    npartitions = max(n_workers, int(np.ceil(total_rows / rows_per_chunk)))

    # Compute non-overlapping row chunks (may be more than n_workers).
    chunks = compute_row_chunks(ms_path, npartitions)
    log.info(
        "Reading %d rows in %d chunks with %d workers (%.1f GB usable, %d rows/chunk)",
        total_rows,
        len(chunks),
        n_workers,
        usable_mem_bytes / 1024**3,
        rows_per_chunk,
    )

    # Temp directory for zarr shards (alongside main store).
    shard_dir = str(zarr_path.parent / f"tmp_gridflag_shards_{uuid.uuid4().hex[:8]}")
    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    # Build worker args.
    field_ids_list = list(config.field_ids) if config.field_ids is not None else None
    worker_args = [
        (
            ms_path,
            data_column,
            startrow,
            nrow,
            i,
            shard_dir,
            spw_lookup,
            global_N,
            config.cell_size,
            config.quantity,
            field_ids_list,
            uv_min,
            uv_max if config.uvrange is not None else float("inf"),
        )
        for i, (startrow, nrow) in enumerate(chunks)
    ]

    # Dispatch workers.
    if n_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            shard_paths = pool.map(_process_chunk_worker, worker_args)
    else:
        shard_paths = [_process_chunk_worker(a) for a in worker_args]

    t_read = time.monotonic() - t_read_start
    log.info("Parallel read + compute: %.1fs (%d workers)", t_read, n_workers)

    # ── Index shards by (spw, corr) ─────────────────────────────────
    shard_index = _index_shards(shard_paths)
    log.info("Shard index: %d (spw, corr) groups across %d shards", len(shard_index), len(shard_paths))

    # ── Streaming statistics, thresholds, flags ──────────────────────
    t_compute_start = time.monotonic()

    # Thread count for streaming passes (zarr reads + numpy release GIL).
    n_stat_threads = min(n_workers, 8)

    all_flag_rows: list[np.ndarray] = []
    all_flag_chans: list[np.ndarray] = []
    all_flag_corrs: list[np.ndarray] = []

    # For optional plotting.
    grid_cache: dict[tuple[int, int], dict] = {}

    for spw_info in all_spws:
        spw_id = spw_info["spw_id"]
        n_corr = spw_info["n_corr"]
        spw_key = f"spw_{spw_id}"

        for corr in range(n_corr):
            corr_key = f"corr_{corr}"
            group_shards = shard_index.get((spw_id, corr), [])
            if not group_shards:
                log.info("SPW %d corr %d: no data, skipping", spw_id, corr)
                continue

            # Two-pass streaming statistics over shards.
            median_grid, std_grid, count_grid = compute_cell_stats_streaming(
                group_shards, spw_key, corr_key, gshape,
                n_bins=config.n_bins, n_threads=n_stat_threads,
            )

            n_total = int(count_grid.sum())
            log.info("SPW %d corr %d: %d unflagged visibilities", spw_id, corr, n_total)

            store.store_grid(spw_id, corr, "median_grid", median_grid)
            store.store_grid(spw_id, corr, "std_grid", std_grid)
            store.store_grid(spw_id, corr, "count_grid", count_grid)

            # Thresholds.
            local_thr = local_neighborhood_threshold(
                median_grid,
                std_grid,
                count_grid,
                config.nsigma,
                config.smoothing_window,
            )
            annular_thr = annular_threshold(
                median_grid,
                std_grid,
                count_grid,
                config.cell_size,
                config.annulus_widths,
                config.nsigma,
                global_N,
            )
            threshold_grid = combine_thresholds(
                local_thr,
                annular_thr,
                count_grid,
                config.smoothing_window,
                config.min_neighbors,
            )
            store.store_grid(spw_id, corr, "threshold_grid", threshold_grid)

            # Streaming flag pass: read each shard, flag, collect results.
            flag_results = _flag_shards(
                group_shards, spw_key, corr_key, threshold_grid,
                n_stat_threads,
            )

            n_flagged = flag_results["n_flagged"]
            log.info(
                "SPW %d corr %d: %d / %d flagged (%.2f%%)",
                spw_id,
                corr,
                n_flagged,
                n_total,
                100.0 * n_flagged / max(n_total, 1),
            )

            store.store_grid(spw_id, corr, "flag_mask",
                             np.zeros(gshape, dtype=np.uint8))

            if plot_dir is not None and n_flagged > 0:
                # Compute "after" grids by streaming with threshold filter.
                median_after, std_after, _ = compute_cell_stats_streaming(
                    group_shards, spw_key, corr_key, gshape,
                    n_bins=config.n_bins, n_threads=n_stat_threads,
                    threshold_grid=threshold_grid,
                )
                grid_cache[(spw_id, corr)] = {
                    "median_before": median_grid,
                    "std_before": std_grid,
                    "median_after": median_after,
                    "std_after": std_after,
                }

            if flag_results["flag_rows"] is not None:
                all_flag_rows.append(flag_results["flag_rows"])
                all_flag_chans.append(flag_results["flag_chans"])
                all_flag_corrs.append(
                    np.full(n_flagged, corr, dtype=np.int32)
                )

    t_compute = time.monotonic() - t_compute_start
    log.info("Streaming compute + flag: %.1fs", t_compute)

    # Clean up shards.
    shutil.rmtree(shard_dir, ignore_errors=True)

    # ── Pass 2: Batch-write flags back to MS ────────────────────────
    t_write_start = time.monotonic()
    total_newly_flagged = 0

    if all_flag_rows:
        all_rows = np.concatenate(all_flag_rows)
        all_chans = np.concatenate(all_flag_chans)
        all_corrs = np.concatenate(all_flag_corrs)

        total_newly_flagged = write_flags_batched(
            ms_path,
            all_rows,
            all_chans,
            all_corrs,
        )

    t_write = time.monotonic() - t_write_start
    log.info("Write: %.1fs", t_write)

    t_total = time.monotonic() - t0
    log.info("Total newly flagged: %d  (%.1fs total)", total_newly_flagged, t_total)

    # ── Diagnostic plots ────────────────────────────────────────────
    plot_paths: list[str] = []
    if plot_dir is not None and grid_cache:
        from gridflag.plotting import plot_grids_from_arrays

        for (spw_id, corr), cached in grid_cache.items():
            try:
                paths = plot_grids_from_arrays(
                    cached["median_before"],
                    cached["std_before"],
                    cached["median_after"],
                    cached["std_after"],
                    config.cell_size,
                    global_N,
                    spw_id,
                    corr,
                    plot_dir,
                )
                plot_paths.extend(str(p) for p in paths)
            except Exception:
                log.warning(
                    "Failed to plot SPW %d corr %d",
                    spw_id,
                    corr,
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
