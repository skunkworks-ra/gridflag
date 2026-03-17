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
from gridflag.zarr_store import ZarrStore, merge_shard_into_consolidated

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


# ── Persistent pool worker state ────────────────────────────────

_worker_tb = None  # module-level; set once per worker process by _init_worker


def _init_worker(ms_path: str) -> None:
    """Pool initializer: open casatools.table once per worker process."""
    global _worker_tb
    import casatools  # type: ignore[import-untyped]

    _worker_tb = casatools.table()
    _worker_tb.open(ms_path, nomodify=True)


def _process_chunk_worker(args: tuple) -> str:
    """Worker: read a row range from the MS, compute UV/cell/quantity, write zarr shard.

    Each worker uses a persistent casatools.table opened by _init_worker.
    Reads its assigned non-overlapping row range, processes all SPWs/corrs
    found in that range, and writes flat arrays to its own zarr shard.

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

    from gridflag.msio import _read_column

    # Use persistent worker table if available (pool mode), else open locally.
    if _worker_tb is not None:
        tb = _worker_tb
    else:
        import casatools  # type: ignore[import-untyped]
        tb = casatools.table()
        tb.open(ms_path, nomodify=True)
    _opened_locally = _worker_tb is None

    # Read raw arrays for this row range.
    raw_data = _read_column(tb, data_column, startrow, nrow)
    # casatools returns (n_corr, n_chan, n_row)
    data = raw_data.transpose(2, 1, 0)  # → (n_row, n_chan, n_corr)

    flags_raw = tb.getcol("FLAG", startrow=startrow, nrow=nrow)
    flags = flags_raw.transpose(2, 1, 0).astype(bool)

    uvw = tb.getcol("UVW", startrow=startrow, nrow=nrow).T  # → (n_row, 3)
    ddid = tb.getcol("DATA_DESC_ID", startrow=startrow, nrow=nrow)
    field_col = tb.getcol("FIELD_ID", startrow=startrow, nrow=nrow)

    if _opened_locally:
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


# ── Chunk-aware flag pass on consolidated zarr ──────────────────


def _flag_one_chunk(
    zarr_group: zarr.hierarchy.Group,
    start: int,
    end: int,
    threshold_grid: np.ndarray,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    """Flag visibilities in one zarr chunk of the consolidated store."""
    cell_u = zarr_group["cell_u"][start:end].astype(np.intp)
    cell_v = zarr_group["cell_v"][start:end].astype(np.intp)
    values = zarr_group["values"][start:end]
    row_indices = zarr_group["row_indices"][start:end]
    chan_indices = zarr_group["chan_indices"][start:end]

    if len(values) == 0:
        return 0, None, None

    flags = flag_visibilities(cell_u, cell_v, values, threshold_grid)
    n_flagged = int(np.sum(flags))

    if n_flagged > 0:
        idx = np.where(flags)[0]
        return n_flagged, row_indices[idx], chan_indices[idx]
    return n_flagged, None, None


def _flag_consolidated(
    zarr_group: zarr.hierarchy.Group,
    threshold_grid: np.ndarray,
    n_threads: int,
) -> dict:
    """Flag visibilities across consolidated zarr for one (spw, corr).

    Returns dict with keys: n_flagged, flag_rows, flag_chans.
    """
    if "values" not in zarr_group or zarr_group["values"].shape[0] == 0:
        return {"n_flagged": 0, "flag_rows": None, "flag_chans": None}

    total_len = zarr_group["values"].shape[0]
    chunk_size = zarr_group["values"].chunks[0]
    chunk_ranges = [
        (i, min(i + chunk_size, total_len))
        for i in range(0, total_len, chunk_size)
    ]

    all_flag_rows: list[np.ndarray] = []
    all_flag_chans: list[np.ndarray] = []
    total_flagged = 0

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(
                _flag_one_chunk, zarr_group, s, e, threshold_grid,
            ): (s, e)
            for s, e in chunk_ranges
        }
        for fut in as_completed(futures):
            n_f, rows, chans = fut.result()
            total_flagged += n_f
            if rows is not None:
                all_flag_rows.append(rows)
                all_flag_chans.append(chans)

    return {
        "n_flagged": total_flagged,
        "flag_rows": np.concatenate(all_flag_rows) if all_flag_rows else None,
        "flag_chans": np.concatenate(all_flag_chans) if all_flag_chans else None,
    }


# ── Per-(SPW, corr) processing ──────────────────────────────────


def _process_spw_corr(
    zarr_group: zarr.hierarchy.Group,
    spw_id: int,
    corr: int,
    gshape: tuple[int, int],
    config: "GridFlagConfig",
    n_stat_threads: int,
    global_N: int,
    plot_dir: "str | Path | None",
    persist_cache: bool,
    store: "ZarrStore",
    pre_counts: "np.ndarray | None" = None,
) -> dict:
    """Run stats → thresholds → flags for one (SPW, corr) pair.

    Operates on the consolidated zarr group for this (spw, corr).

    Returns a dict with keys: spw_id, corr, n_total, n_flagged,
    flag_rows, flag_chans, grid_cache_entry.
    """
    # Streaming statistics.
    median_grid, std_grid, count_grid = compute_cell_stats_streaming(
        zarr_group, gshape,
        n_bins=config.n_bins, n_threads=n_stat_threads,
        pre_counts=pre_counts,
    )

    n_total = int(count_grid.sum())
    log.info("SPW %d corr %d: %d unflagged visibilities", spw_id, corr, n_total)

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
        config.cell_size, config.annulus_widths, config.nsigma, global_N,
    )
    threshold_grid = combine_thresholds(
        local_thr, annular_thr, count_grid,
        config.smoothing_window, config.min_neighbors,
    )
    store.store_grid(spw_id, corr, "threshold_grid", threshold_grid)

    # Chunk-aware flag pass on consolidated zarr.
    flag_results = _flag_consolidated(
        zarr_group, threshold_grid, n_stat_threads,
    )

    n_flagged = flag_results["n_flagged"]
    log.info(
        "SPW %d corr %d: %d / %d flagged (%.2f%%)",
        spw_id, corr, n_flagged, n_total,
        100.0 * n_flagged / max(n_total, 1),
    )

    store.store_grid(spw_id, corr, "flag_mask", np.zeros(gshape, dtype=np.uint8))

    # Optional "after" grids for plotting / caching.
    grid_cache_entry = None
    if (plot_dir is not None or persist_cache) and n_flagged > 0:
        median_after, std_after, _ = compute_cell_stats_streaming(
            zarr_group, gshape,
            n_bins=config.n_bins, n_threads=n_stat_threads,
            threshold_grid=threshold_grid,
        )
        store.store_grid(spw_id, corr, "median_after", median_after)
        store.store_grid(spw_id, corr, "std_after", std_after)

        if plot_dir is not None:
            grid_cache_entry = {
                "median_before": median_grid,
                "std_before": std_grid,
                "median_after": median_after,
                "std_after": std_after,
            }

    return {
        "spw_id": spw_id,
        "corr": corr,
        "n_total": n_total,
        "n_flagged": n_flagged,
        "flag_rows": flag_results["flag_rows"],
        "flag_chans": flag_results["flag_chans"],
        "grid_cache_entry": grid_cache_entry,
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
      1. Parallel read MS → each worker writes its own zarr shard
      2. imap_unordered yields shards → main thread merges into
         consolidated zarr, deletes shard
      3. Per (SPW, corr): parallel histogram stats over consolidated
         zarr chunks → thresholds → parallel flag pass
      4. Batch-write flags back to MS
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

    N_u, N_v = gshape
    n_cells = N_u * N_v

    # ── Pass 1: Parallel read MS → zarr shards ─────────────────────
    t_read_start = time.monotonic()

    n_workers = config.n_workers if config.n_workers > 0 else available_cpus()

    # Memory-budgeted chunk sizing.
    max_nchan = max(s["n_chan"] for s in all_spws)
    max_ncorr = max(s["n_corr"] for s in all_spws)
    bytes_per_row = max_ncorr * max_nchan * 120
    usable_mem_bytes = available_memory_gb() * 0.8 * 1024**3
    mem_per_worker = usable_mem_bytes / n_workers
    rows_per_chunk = max(1, int(mem_per_worker / bytes_per_row))

    total_rows = get_ms_row_count(ms_path)
    npartitions = max(n_workers, int(np.ceil(total_rows / rows_per_chunk)))

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

    # ── Read + merge loop ────────────────────────────────────────────
    # Workers write temporary shards; main thread merges each shard
    # into the consolidated store as it arrives, then deletes it.
    all_pre_counts: dict[tuple[int, int], np.ndarray] = {}

    if n_workers > 1:
        with multiprocessing.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(ms_path,),
        ) as pool:
            for shard_path in pool.imap_unordered(_process_chunk_worker, worker_args):
                shard_counts = merge_shard_into_consolidated(
                    shard_path, store, N_v, n_cells,
                )
                for key, bc in shard_counts.items():
                    if key in all_pre_counts:
                        all_pre_counts[key] += bc
                    else:
                        all_pre_counts[key] = bc
                shutil.rmtree(shard_path, ignore_errors=True)
    else:
        for a in worker_args:
            shard_path = _process_chunk_worker(a)
            shard_counts = merge_shard_into_consolidated(
                shard_path, store, N_v, n_cells,
            )
            for key, bc in shard_counts.items():
                if key in all_pre_counts:
                    all_pre_counts[key] += bc
                else:
                    all_pre_counts[key] = bc
            shutil.rmtree(shard_path, ignore_errors=True)

    # Shard directory should be empty now; clean up.
    shutil.rmtree(shard_dir, ignore_errors=True)

    # Clear accumulators created by init_spw (unused — we used append_direct).
    store._accumulators.clear()

    t_read = time.monotonic() - t_read_start
    log.info("Parallel read + merge: %.1fs (%d workers)", t_read, n_workers)
    log.info("Consolidated store: %d (spw, corr) groups with data", len(all_pre_counts))

    # ── Streaming statistics, thresholds, flags ──────────────────────
    t_compute_start = time.monotonic()

    # Thread count for streaming passes (zarr reads + numpy release GIL).
    n_stat_threads = n_workers

    import numba
    try:
        layer = numba.threading_layer()
    except ValueError:
        layer = "uninitialized"
    log.debug(
        "Numba threading layer: %s  threads: %d",
        layer, numba.get_num_threads(),
    )

    all_flag_rows: list[np.ndarray] = []
    all_flag_chans: list[np.ndarray] = []
    all_flag_corrs: list[np.ndarray] = []

    # For optional plotting.
    grid_cache: dict[tuple[int, int], dict] = {}

    # Build list of (SPW, corr) pairs that have data.
    all_pairs: list[tuple[int, int]] = []
    for spw_info in all_spws:
        spw_id_v = spw_info["spw_id"]
        for corr in range(spw_info["n_corr"]):
            if (spw_id_v, corr) not in all_pre_counts:
                log.info("SPW %d corr %d: no data, skipping", spw_id_v, corr)
            else:
                all_pairs.append((spw_id_v, corr))

    # Process pairs serially on the main thread so Numba prange has uncontested
    # access to the full thread pool.
    for spw_id_v, corr in all_pairs:
        zarr_group = store.root[f"spw_{spw_id_v}/corr_{corr}"]
        pre_counts = all_pre_counts.get((spw_id_v, corr))

        result = _process_spw_corr(
            zarr_group, spw_id_v, corr, gshape, config,
            n_stat_threads, global_N, plot_dir, persist_cache, store,
            pre_counts=pre_counts,
        )
        corr = result["corr"]
        n_flagged = result["n_flagged"]
        if result["flag_rows"] is not None:
            all_flag_rows.append(result["flag_rows"])
            all_flag_chans.append(result["flag_chans"])
            all_flag_corrs.append(
                np.full(n_flagged, corr, dtype=np.int32)
            )
        if result["grid_cache_entry"] is not None:
            grid_cache[(result["spw_id"], corr)] = result["grid_cache_entry"]

    t_compute = time.monotonic() - t_compute_start
    log.info("Streaming compute + flag: %.1fs", t_compute)

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


def plot_from_cache(
    zarr_path: str | Path,
    plot_dir: str | Path,
) -> list[str]:
    """Generate diagnostic plots from a persisted Zarr cache.

    Reads median_grid, std_grid, median_after, std_after from the cache
    and produces before/after comparison PNGs.  No MS access required.
    """
    from gridflag.plotting import plot_grids_from_arrays
    from gridflag.zarr_store import open_readonly

    zarr_path = Path(zarr_path)
    root = open_readonly(zarr_path)

    config = GridFlagConfig.from_json(root.attrs["config_json"])
    gshape = tuple(root.attrs["grid_shape"])
    N_u = gshape[0]
    global_N = (N_u - 1) // 2

    plot_paths: list[str] = []

    for spw_key in root:
        if not spw_key.startswith("spw_"):
            continue
        spw_id = int(spw_key.split("_")[1])
        spw_grp = root[spw_key]

        for corr_key in spw_grp:
            if not corr_key.startswith("corr_"):
                continue
            corr_id = int(corr_key.split("_")[1])
            cgrp = spw_grp[corr_key]

            if "median_grid" not in cgrp or "median_after" not in cgrp:
                log.info(
                    "SPW %d corr %d: missing grids, skipping",
                    spw_id, corr_id,
                )
                continue

            median_before = cgrp["median_grid"][:]
            std_before = cgrp["std_grid"][:]
            median_after = cgrp["median_after"][:]
            std_after = cgrp["std_after"][:]

            try:
                paths = plot_grids_from_arrays(
                    median_before,
                    std_before,
                    median_after,
                    std_after,
                    config.cell_size,
                    global_N,
                    spw_id,
                    corr_id,
                    plot_dir,
                )
                plot_paths.extend(str(p) for p in paths)
            except Exception:
                log.warning(
                    "Failed to plot SPW %d corr %d",
                    spw_id,
                    corr_id,
                    exc_info=True,
                )

    log.info("Generated %d plots from cache %s", len(plot_paths), zarr_path)
    return plot_paths
