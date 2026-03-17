"""Measurement Set I/O via arcae."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("gridflag.msio")


def available_cpus() -> int:
    """Return the number of CPUs available to this process (cgroups-aware)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def available_memory_gb() -> float:
    """Return total system RAM in GB."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError):
        return 8.0  # conservative fallback


def get_ms_row_count(ms_path: str) -> int:
    """Return the total number of rows in the MS main table."""
    arcae = _import_arcae()
    with arcae.table(ms_path, readonly=True) as tb:
        return tb.nrow()


@dataclass
class MSChunk:
    """A chunk of rows read from an MS."""

    data: NDArray[np.complexfloating]  # (n_row, n_chan, n_corr)
    uvw: NDArray[np.float64]  # (n_row, 3)
    flags: NDArray[np.bool_]  # (n_row, n_chan, n_corr)
    row_indices: NDArray[np.int64]  # (n_row,) — absolute row indices in MS
    spw_id: int
    field_id: int


def _import_arcae():
    """Lazy import of arcae."""
    import arcae  # type: ignore[import-untyped]

    return arcae


def resolve_data_column(ms_path: str, data_column: str) -> str:
    """Determine which data column to use.

    When *data_column* is ``"auto"``, checks in priority order:
    RESIDUAL, DATA−MODEL, CORRECTED−MODEL, DATA.

    Returns the resolved column name (or a special ``"DATA-MODEL"`` /
    ``"CORRECTED_DATA-MODEL"`` sentinel).
    """
    if data_column != "auto":
        return data_column

    arcae = _import_arcae()
    with arcae.table(ms_path, readonly=True) as tb:
        cols = tb.columns()

        if "RESIDUAL" in cols:
            log.info("Auto-resolved data column: RESIDUAL")
            return "RESIDUAL"

        has_model = "MODEL_DATA" in cols
        if has_model:
            # Check if MODEL_DATA is non-zero (sample first chunk).
            nrows = tb.nrow()
            n_sample = min(1000, nrows)
            model = tb.getcol("MODEL_DATA", index=(slice(0, n_sample),))
            model_nonzero = np.max(np.abs(model)) > 0.0
        else:
            model_nonzero = False

        if model_nonzero and "DATA" in cols:
            log.info("Auto-resolved data column: DATA - MODEL_DATA")
            return "DATA-MODEL"

        if model_nonzero and "CORRECTED_DATA" in cols:
            log.info("Auto-resolved data column: CORRECTED_DATA - MODEL_DATA")
            return "CORRECTED_DATA-MODEL"

        log.info("Auto-resolved data column: DATA")
        return "DATA"


def _read_column(tb, col_spec: str, startrow: int, nrow: int) -> NDArray:
    """Read a data column, handling subtraction sentinels.

    Returns data in (n_row, n_chan, n_corr) order (arcae returns row-first).
    """
    idx = (slice(startrow, startrow + nrow),)
    if col_spec == "DATA-MODEL":
        d = tb.getcol("DATA", index=idx)
        m = tb.getcol("MODEL_DATA", index=idx)
        return d - m
    elif col_spec == "CORRECTED_DATA-MODEL":
        d = tb.getcol("CORRECTED_DATA", index=idx)
        m = tb.getcol("MODEL_DATA", index=idx)
        return d - m
    else:
        return tb.getcol(col_spec, index=idx)


# ── Row-range chunking (parallel reads) ─────────────────────────


def compute_row_chunks(ms_path: str, nchunks: int) -> list[tuple[int, int]]:
    """Compute non-overlapping (startrow, nrow) ranges respecting integration boundaries.

    Reads the TIME column once, finds unique time-step boundaries, and
    distributes them evenly across *nchunks* chunks so that no
    integration is split across workers.
    """
    arcae = _import_arcae()
    with arcae.table(ms_path, readonly=True) as tb:
        time_col = tb.getcol("TIME")
        total_rows = tb.nrow()

    unique_times, first_indices = np.unique(time_col, return_index=True)
    n_times = len(unique_times)

    sorted_indices = np.argsort(first_indices)

    if n_times <= nchunks:
        # More chunks than time steps — one chunk per time step.
        chunks = []
        for i, idx in enumerate(sorted_indices):
            start = first_indices[idx]
            if i + 1 < len(sorted_indices):
                end = first_indices[sorted_indices[i + 1]]
            else:
                end = total_rows
            chunks.append((int(start), int(end - start)))
        return chunks

    # Distribute time steps evenly across chunks.
    times_per_chunk = n_times // nchunks
    chunks = []
    for w in range(nchunks):
        start_time_idx = w * times_per_chunk
        if w == nchunks - 1:
            end_time_idx = n_times
        else:
            end_time_idx = (w + 1) * times_per_chunk

        startrow = first_indices[sorted_indices[start_time_idx]]
        if end_time_idx >= n_times:
            endrow = total_rows
        else:
            endrow = first_indices[sorted_indices[end_time_idx]]

        chunks.append((int(startrow), int(endrow - startrow)))

    return chunks


# ── Legacy sequential reader ────────────────────────────────────


def read_chunks(
    ms_path: str,
    data_column: str,
    chunk_size: int,
    spw_id: int,
) -> Iterator[MSChunk]:
    """Yield MSChunks for a given SPW by reading the MS in row chunks.

    Reads the full table and filters rows by DATA_DESC_ID == spw_id
    using numpy masking (arcae does not support TAQL subtable queries
    with rownumbers).
    """
    arcae = _import_arcae()
    with arcae.table(ms_path, readonly=True) as tb:
        ddid_col = tb.getcol("DATA_DESC_ID")
        spw_mask = ddid_col == spw_id
        abs_rows = np.where(spw_mask)[0].astype(np.int64)

        if len(abs_rows) == 0:
            return

        for offset in range(0, len(abs_rows), chunk_size):
            batch_rows = abs_rows[offset : offset + chunk_size]
            start = int(batch_rows[0])
            end = int(batch_rows[-1]) + 1

            # Read the contiguous range then filter to matching rows.
            idx = (slice(start, end),)
            raw = _read_column(tb, data_column, start, end - start)
            # arcae returns (n_row, n_corr, n_chan) → (n_row, n_chan, n_corr)
            local_mask = spw_mask[start:end]
            data = raw[local_mask].transpose(0, 2, 1)

            flags_raw = tb.getcol("FLAG", index=idx)
            flags = flags_raw[local_mask].transpose(0, 2, 1).astype(bool)

            uvw = tb.getcol("UVW", index=idx)  # (n_row, 3) — no transpose
            uvw = uvw[local_mask]

            field_ids = tb.getcol("FIELD_ID", index=idx)
            field_ids = field_ids[local_mask]

            yield MSChunk(
                data=data,
                uvw=uvw,
                flags=flags,
                row_indices=batch_rows,
                spw_id=spw_id,
                field_id=int(field_ids[0]),
            )


# ── Metadata helpers ────────────────────────────────────────────


def get_max_baseline_m(ms_path: str) -> float:
    """Return the maximum baseline length in metres from the ANTENNA table."""
    arcae = _import_arcae()
    with arcae.table(f"{ms_path}::ANTENNA", readonly=True) as tb:
        pos = tb.getcol("POSITION")  # (n_ant, 3) — row-first
    # Max pairwise distance.
    from scipy.spatial.distance import pdist

    return float(np.max(pdist(pos)))


def get_spw_info(ms_path: str) -> list[dict]:
    """Return per-SPW metadata: n_chan, n_corr, ref_freq, chan_freqs."""
    arcae = _import_arcae()

    # Spectral window table.
    with arcae.table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True) as tb:
        n_spw = tb.nrow()
        spws = []
        for i in range(n_spw):
            chan_freqs = tb.getcol("CHAN_FREQ", index=(slice(i, i + 1),)).flatten()
            ref_freq = tb.getcol("REF_FREQUENCY", index=(slice(i, i + 1),)).item()
            spws.append(
                {
                    "spw_id": i,
                    "n_chan": len(chan_freqs),
                    "ref_freq": ref_freq,
                    "chan_freqs": chan_freqs,
                }
            )

    # Get n_corr from polarization table.
    with arcae.table(f"{ms_path}::POLARIZATION", readonly=True) as tb:
        n_corr = tb.getcol("NUM_CORR", index=(slice(0, 1),)).item()
    for s in spws:
        s["n_corr"] = n_corr

    return spws


# ── Flag writing ────────────────────────────────────────────────


def write_flags(
    ms_path: str,
    row_indices: NDArray[np.int64],
    chan_indices: NDArray[np.int32],
    corr_indices: NDArray[np.int32],
    flag_values: NDArray[np.bool_],
) -> int:
    """Write new flags into the MS FLAG column.

    Only sets flags to True (logical OR with existing); never unflags.
    Batches writes by unique row for efficiency.

    Returns the number of visibilities newly flagged.
    """
    return write_flags_batched(ms_path, row_indices, chan_indices, corr_indices)


def write_flags_batched(
    ms_path: str,
    row_indices: NDArray[np.int64],
    chan_indices: NDArray[np.int32],
    corr_indices: NDArray[np.int32],
    batch_size: int = 10_000,
) -> int:
    """Write new flags into the MS FLAG column using batched I/O.

    Reads/writes contiguous row ranges in batches rather than one row
    at a time.  All entries are flagged (set to True); never unflags.

    Returns the number of visibilities newly flagged.
    """
    if len(row_indices) == 0:
        return 0

    arcae = _import_arcae()
    with arcae.table(ms_path, readonly=False) as tb:
        total_flagged = 0

        # Sort by row for contiguous access.
        order = np.argsort(row_indices)
        rows = row_indices[order]
        chans = chan_indices[order]
        corrs = corr_indices[order]

        # Process in contiguous row batches.
        min_row = int(rows[0])
        max_row = int(rows[-1])

        for batch_start in range(min_row, max_row + 1, batch_size):
            batch_end = min(batch_start + batch_size, max_row + 1)
            nrow = batch_end - batch_start

            # Find which flag entries fall in this batch.
            lo = np.searchsorted(rows, batch_start, side="left")
            hi = np.searchsorted(rows, batch_end, side="left")
            if lo == hi:
                continue

            # Read existing flags: arcae returns (nrow, n_corr, n_chan).
            idx = (slice(batch_start, batch_start + nrow),)
            flag_block = tb.getcol("FLAG", index=idx)
            count_before = int(np.sum(flag_block))

            # Vectorized update: set flags using relative row indices.
            # arcae shape is (nrow, n_corr, n_chan).
            rel_rows = rows[lo:hi] - batch_start
            flag_block[rel_rows, corrs[lo:hi], chans[lo:hi]] = True

            tb.putcol("FLAG", flag_block, index=idx)
            total_flagged += int(np.sum(flag_block)) - count_before

        return total_flagged
