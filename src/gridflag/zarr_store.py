"""Zarr intermediate storage for accumulation and grid results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray

from gridflag.config import GridFlagConfig

log = logging.getLogger("gridflag.zarr_store")

# Default chunk size for flat 1-D arrays.
_FLAT_CHUNK = 1_000_000


class ZarrStore:
    """Manages the Zarr hierarchy for a single GRIDflag run."""

    def __init__(self, path: str | Path, config: GridFlagConfig, ms_path: str):
        self.path = Path(path)
        self.root = zarr.open(str(self.path), mode="w")
        self.root.attrs["ms_path"] = ms_path
        self.root.attrs["cell_size"] = config.cell_size
        self.root.attrs["config_json"] = config.to_json()
        self._config = config

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
            cg = grp.require_group(f"corr_{corr}")
            for name, dtype in [
                ("row_indices", "i8"),
                ("chan_indices", "i4"),
                ("cell_u", "i4"),
                ("cell_v", "i4"),
                ("values", "f4"),
            ]:
                cg.zeros(
                    name, shape=(0,), chunks=(_FLAT_CHUNK,), dtype=dtype, overwrite=True
                )

    # ------------------------------------------------------------------
    # Append flat arrays (pass 1)
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
        """Append a batch of flat visibility records."""
        grp = self.root[f"spw_{spw_id}/corr_{corr_id}"]
        n = len(row_indices)
        for name, arr in [
            ("row_indices", row_indices),
            ("chan_indices", chan_indices),
            ("cell_u", cell_u),
            ("cell_v", cell_v),
            ("values", values),
        ]:
            ds = grp[name]
            old = ds.shape[0]
            ds.resize(old + n)
            ds[old:] = arr

    # ------------------------------------------------------------------
    # Read flat arrays back (pass 1.5)
    # ------------------------------------------------------------------

    def load_flat(
        self, spw_id: int, corr_id: int
    ) -> dict[str, NDArray]:
        """Load the flat arrays for one (spw, corr) pair."""
        grp = self.root[f"spw_{spw_id}/corr_{corr_id}"]
        return {
            "row_indices": grp["row_indices"][:],
            "chan_indices": grp["chan_indices"][:],
            "cell_u": grp["cell_u"][:],
            "cell_v": grp["cell_v"][:],
            "values": grp["values"][:],
        }

    # ------------------------------------------------------------------
    # Store computed grids (pass 1.5)
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
