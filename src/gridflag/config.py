"""Configuration dataclass for GRIDflag."""

from __future__ import annotations

import dataclasses
import json


@dataclasses.dataclass(frozen=True)
class GridFlagConfig:
    """All tuneable parameters for a GRIDflag run."""

    cell_size: float = 10.0  # lambda
    nsigma: float = 3.0
    smoothing_window: int = 5  # neighborhood kernel K×K
    annulus_widths: tuple[float, ...] = (3000.0, 3000.0, 6500.0)  # lambda
    data_column: str = "auto"  # auto | DATA | CORRECTED_DATA | RESIDUAL
    quantity: str = "amplitude"  # amplitude | phase | real | imag
    zarr_path: str | None = None  # None → tempdir
    chunk_size: int = 50_000  # rows per read chunk
    n_readers: int = 4
    min_neighbors: int = 3  # fallback to annular if fewer
    spw_ids: tuple[int, ...] | None = None
    field_ids: tuple[int, ...] | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> GridFlagConfig:
        d = json.loads(s)
        # Convert lists back to tuples for frozen fields.
        for key in ("annulus_widths", "spw_ids", "field_ids"):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        return cls(**d)
