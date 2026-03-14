"""Measurement Set I/O via casatools."""

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
    casatools = _import_casatools()
    tb = casatools.table()
    tb.open(ms_path, nomodify=True)
    nrow = tb.nrows()
    tb.close()
    return nrow


@dataclass
class MSChunk:
    """A chunk of rows read from an MS."""

    data: NDArray[np.complexfloating]  # (n_row, n_chan, n_corr)
    uvw: NDArray[np.float64]  # (n_row, 3)
    flags: NDArray[np.bool_]  # (n_row, n_chan, n_corr)
    row_indices: NDArray[np.int64]  # (n_row,) — absolute row indices in MS
    spw_id: int
    field_id: int


def _import_casatools():
    """Lazy import of casatools."""
    import casatools  # type: ignore[import-untyped]

    return casatools


def resolve_data_column(ms_path: str, data_column: str) -> str:
    """Determine which data column to use.

    When *data_column* is ``"auto"``, checks in priority order:
    RESIDUAL, DATA−MODEL, CORRECTED−MODEL, DATA.

    Returns the resolved column name (or a special ``"DATA-MODEL"`` /
    ``"CORRECTED_DATA-MODEL"`` sentinel).
    """
    if data_column != "auto":
        return data_column

    casatools = _import_casatools()
    tb = casatools.table()
    tb.open(ms_path, nomodify=True)
    try:
        cols = tb.colnames()

        if "RESIDUAL" in cols:
            log.info("Auto-resolved data column: RESIDUAL")
            return "RESIDUAL"

        has_model = "MODEL_DATA" in cols
        if has_model:
            # Check if MODEL_DATA is non-zero (sample first chunk).
            model = tb.getcol("MODEL_DATA", startrow=0, nrow=min(1000, tb.nrows()))
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
    finally:
        tb.close()


def _read_column(tb, col_spec: str, startrow: int, nrow: int) -> NDArray:
    """Read a data column, handling subtraction sentinels."""
    if col_spec == "DATA-MODEL":
        d = tb.getcol("DATA", startrow=startrow, nrow=nrow)
        m = tb.getcol("MODEL_DATA", startrow=startrow, nrow=nrow)
        return d - m
    elif col_spec == "CORRECTED_DATA-MODEL":
        d = tb.getcol("CORRECTED_DATA", startrow=startrow, nrow=nrow)
        m = tb.getcol("MODEL_DATA", startrow=startrow, nrow=nrow)
        return d - m
    else:
        return tb.getcol(col_spec, startrow=startrow, nrow=nrow)


# ── Row-range chunking (parallel reads) ─────────────────────────


def compute_row_chunks(ms_path: str, nchunks: int) -> list[tuple[int, int]]:
    """Compute non-overlapping (startrow, nrow) ranges respecting integration boundaries.

    Reads the TIME column once, finds unique time-step boundaries, and
    distributes them evenly across *nchunks* chunks so that no
    integration is split across workers.
    """
    casatools = _import_casatools()
    tb = casatools.table()
    tb.open(ms_path, nomodify=True)
    time_col = tb.getcol("TIME")
    total_rows = tb.nrows()
    tb.close()

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

    Uses TAQL to select rows for the desired DATA_DESC_ID (== spw_id for
    standard MSes).
    """
    casatools = _import_casatools()
    tb = casatools.table()
    tb.open(ms_path, nomodify=True)
    try:
        # Select rows for this SPW via DATA_DESC_ID.
        subtb = tb.query(f"DATA_DESC_ID == {spw_id}")
        n_rows = subtb.nrows()
        if n_rows == 0:
            subtb.close()
            return

        for start in range(0, n_rows, chunk_size):
            nrow = min(chunk_size, n_rows - start)
            # casatools getcol returns (n_corr, n_chan, n_row) — transpose.
            raw = _read_column(subtb, data_column, start, nrow)
            data = raw.transpose(2, 1, 0)  # → (n_row, n_chan, n_corr)

            flags_raw = subtb.getcol("FLAG", startrow=start, nrow=nrow)
            flags = flags_raw.transpose(2, 1, 0).astype(bool)

            uvw = subtb.getcol("UVW", startrow=start, nrow=nrow).T  # (3, n_row) → (n_row, 3)

            field_ids = subtb.getcol("FIELD_ID", startrow=start, nrow=nrow)

            # Absolute row numbers for write-back.
            rownrs = subtb.rownumbers()[start : start + nrow]

            yield MSChunk(
                data=data,
                uvw=uvw,
                flags=flags,
                row_indices=np.array(rownrs, dtype=np.int64),
                spw_id=spw_id,
                field_id=int(field_ids[0]),
            )

        subtb.close()
    finally:
        tb.close()


# ── Metadata helpers ────────────────────────────────────────────


def get_max_baseline_m(ms_path: str) -> float:
    """Return the maximum baseline length in metres from the ANTENNA table."""
    casatools = _import_casatools()
    tb = casatools.table()
    tb.open(f"{ms_path}/ANTENNA", nomodify=True)
    pos = tb.getcol("POSITION")  # (3, n_ant)
    tb.close()
    pos = pos.T  # (n_ant, 3)
    # Max pairwise distance.
    from scipy.spatial.distance import pdist

    return float(np.max(pdist(pos)))


def get_spw_info(ms_path: str) -> list[dict]:
    """Return per-SPW metadata: n_chan, n_corr, ref_freq, chan_freqs."""
    casatools = _import_casatools()
    tb = casatools.table()

    # Spectral window table.
    tb.open(f"{ms_path}/SPECTRAL_WINDOW", nomodify=True)
    n_spw = tb.nrows()
    spws = []
    for i in range(n_spw):
        chan_freqs = tb.getcol("CHAN_FREQ", startrow=i, nrow=1).flatten()
        ref_freq = tb.getcol("REF_FREQUENCY", startrow=i, nrow=1).item()
        spws.append(
            {
                "spw_id": i,
                "n_chan": len(chan_freqs),
                "ref_freq": ref_freq,
                "chan_freqs": chan_freqs,
            }
        )
    tb.close()

    # Get n_corr from polarization table.
    tb.open(f"{ms_path}/POLARIZATION", nomodify=True)
    n_corr = tb.getcol("NUM_CORR", startrow=0, nrow=1).item()
    tb.close()
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

    casatools = _import_casatools()
    tb = casatools.table()
    tb.open(ms_path, nomodify=False)
    try:
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

            # Read existing flags for this row range: (n_corr, n_chan, nrow).
            flag_block = tb.getcol("FLAG", startrow=batch_start, nrow=nrow)
            count_before = int(np.sum(flag_block))

            # Vectorized update: set flags using relative row indices.
            rel_rows = rows[lo:hi] - batch_start
            flag_block[corrs[lo:hi], chans[lo:hi], rel_rows] = True

            tb.putcol("FLAG", flag_block, startrow=batch_start, nrow=nrow)
            total_flagged += int(np.sum(flag_block)) - count_before

        return total_flagged
    finally:
        tb.close()
