"""Zarr intermediate storage for accumulation and grid results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

from gridflag.config import GridFlagConfig

log = logging.getLogger("gridflag.zarr_store")

# Default chunk size for flat 1-D arrays.
_FLAT_CHUNK = 1_000_000


class AccumulatorGroup:
    """In-memory accumulator for a single (spw, corr) pair.

    Collects numpy array chunks in lists, then flushes once to Zarr.
    Avoids repeated Zarr resize/append overhead.
    """

    __slots__ = ("row_indices", "chan_indices", "cell_u", "cell_v", "values")

    def __init__(self) -> None:
        self.row_indices: list[np.ndarray] = []
        self.chan_indices: list[np.ndarray] = []
        self.cell_u: list[np.ndarray] = []
        self.cell_v: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    def append(
        self,
        row_indices: NDArray,
        chan_indices: NDArray,
        cell_u: NDArray,
        cell_v: NDArray,
        values: NDArray,
    ) -> None:
        self.row_indices.append(row_indices)
        self.chan_indices.append(chan_indices)
        self.cell_u.append(cell_u)
        self.cell_v.append(cell_v)
        self.values.append(values)

    def concatenate(self) -> dict[str, NDArray]:
        """Concatenate all chunks into single arrays."""
        if not self.values:
            return {
                k: np.array([], dtype=d)
                for k, d in [
                    ("row_indices", "i8"),
                    ("chan_indices", "i4"),
                    ("cell_u", "i4"),
                    ("cell_v", "i4"),
                    ("values", "f4"),
                ]
            }
        return {
            "row_indices": np.concatenate(self.row_indices),
            "chan_indices": np.concatenate(self.chan_indices),
            "cell_u": np.concatenate(self.cell_u),
            "cell_v": np.concatenate(self.cell_v),
            "values": np.concatenate(self.values),
        }


class ZarrStore:
    """Manages the Zarr hierarchy for a single GRIDflag run.

    Flat arrays are accumulated in memory via AccumulatorGroup, then
    flushed to Zarr in a single write per (spw, corr).
    """

    def __init__(self, path: str | Path, config: GridFlagConfig, ms_path: str):
        self.path = Path(path)
        self.root = zarr.open(str(self.path), mode="w")
        self.root.attrs["ms_path"] = ms_path
        self.root.attrs["cell_size"] = config.cell_size
        self.root.attrs["config_json"] = config.to_json()
        self._config = config
        self._accumulators: dict[tuple[int, int], AccumulatorGroup] = {}

    # ------------------------------------------------------------------
    # SPW / correlation group setup
    # ------------------------------------------------------------------

    def init_spw(
        self,
        spw_id: int,
        n_chan: int,
        n_corr: int,
        ref_freq: float,
        chan_freqs: NDArray[np.floating],
    ) -> None:
        """Create the group for a spectral window and its correlations."""
        grp = self.root.require_group(f"spw_{spw_id}")
        grp.attrs["n_chan"] = n_chan
        grp.attrs["n_corr"] = n_corr
        grp.attrs["ref_freq"] = float(ref_freq)
        grp.attrs["chan_freqs"] = chan_freqs.tolist()

        for corr in range(n_corr):
            grp.require_group(f"corr_{corr}")
            self._accumulators[(spw_id, corr)] = AccumulatorGroup()

    # ------------------------------------------------------------------
    # Accumulate flat arrays (pass 1) — in memory
    # ------------------------------------------------------------------

    def append(
        self,
        spw_id: int,
        corr_id: int,
        row_indices: NDArray,
        chan_indices: NDArray,
        cell_u: NDArray,
        cell_v: NDArray,
        values: NDArray,
    ) -> None:
        """Append a batch of flat visibility records (in-memory accumulation)."""
        self._accumulators[(spw_id, corr_id)].append(
            row_indices,
            chan_indices,
            cell_u,
            cell_v,
            values,
        )

    def flush(self, spw_id: int, corr_id: int) -> None:
        """Flush accumulated arrays to Zarr in a single write."""
        acc = self._accumulators.pop((spw_id, corr_id))
        data = acc.concatenate()
        grp = self.root[f"spw_{spw_id}/corr_{corr_id}"]
        for name, arr in data.items():
            grp.array(name, arr, chunks=(_FLAT_CHUNK,), overwrite=True)

    def flush_all(self) -> None:
        """Flush all remaining accumulators to Zarr."""
        for spw_id, corr_id in list(self._accumulators.keys()):
            self.flush(spw_id, corr_id)

    # ------------------------------------------------------------------
    # Read flat arrays back (pass 1.5)
    # ------------------------------------------------------------------

    def load_flat(self, spw_id: int, corr_id: int) -> dict[str, NDArray]:
        """Load the flat arrays for one (spw, corr) pair."""
        # If not yet flushed, flush first.
        if (spw_id, corr_id) in self._accumulators:
            self.flush(spw_id, corr_id)
        grp = self.root[f"spw_{spw_id}/corr_{corr_id}"]
        return {
            "row_indices": grp["row_indices"][:],
            "chan_indices": grp["chan_indices"][:],
            "cell_u": grp["cell_u"][:],
            "cell_v": grp["cell_v"][:],
            "values": grp["values"][:],
        }

    # ------------------------------------------------------------------
    # Store / load computed grids (pass 1.5)
    # ------------------------------------------------------------------

    def store_grid(
        self,
        spw_id: int,
        corr_id: int,
        name: str,
        data: NDArray,
    ) -> None:
        """Store a 2-D grid (median, std, count, threshold)."""
        grp = self.root[f"spw_{spw_id}/corr_{corr_id}"]
        grp.array(name, data, overwrite=True)

    def load_grid(self, spw_id: int, corr_id: int, name: str) -> NDArray:
        return self.root[f"spw_{spw_id}/corr_{corr_id}/{name}"][:]

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def spw_attrs(self, spw_id: int) -> dict:
        return dict(self.root[f"spw_{spw_id}"].attrs)

    def set_grid_shape(self, shape: tuple[int, int]) -> None:
        self.root.attrs["grid_shape"] = list(shape)

    def get_grid_shape(self) -> tuple[int, int]:
        s = self.root.attrs["grid_shape"]
        return (int(s[0]), int(s[1]))


def open_readonly(path: str | Path) -> zarr.hierarchy.Group:
    """Open an existing Zarr store read-only."""
    return zarr.open(str(path), mode="r")
