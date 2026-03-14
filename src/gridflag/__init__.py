"""GRIDflag: UV-plane RFI flagging for radio interferometric data.

Python API
----------
>>> from gridflag import gridflag
>>> result = gridflag("/path/to/data.ms", nsigma=3.0, cell_size=10.0)

>>> from gridflag import GridFlagConfig, run
>>> config = GridFlagConfig(nsigma=5.0, quantity="phase")
>>> result = run("/path/to/data.ms", config)
"""

__version__ = "0.1.0"

from gridflag.config import GridFlagConfig
from gridflag.pipeline import run


def gridflag(
    ms_path: str,
    *,
    cell_size: float = 10.0,
    nsigma: float = 3.0,
    smoothing_window: int = 5,
    annulus_widths: tuple[float, ...] = (3000.0, 3000.0, 6500.0),
    data_column: str = "auto",
    quantity: str = "amplitude",
    zarr_path: str | None = None,
    chunk_size: int = 50_000,
    n_readers: int = 4,
    min_neighbors: int = 3,
    spw_ids: tuple[int, ...] | None = None,
    field_ids: tuple[int, ...] | None = None,
    uvrange: tuple[float, float] | None = None,
    plot_dir: str | None = None,
    log_level: str = "INFO",
) -> dict:
    """Run GRIDflag on a CASA Measurement Set.

    Convenience wrapper that constructs a GridFlagConfig and calls
    ``pipeline.run()``.  All parameters are keyword-only with sensible
    defaults.

    Parameters
    ----------
    ms_path : str
        Path to the CASA Measurement Set.
    cell_size : float
        Grid cell size in lambda.
    nsigma : float
        Sigma threshold multiplier.
    smoothing_window : int
        Local neighborhood kernel size (K x K).
    annulus_widths : tuple of float
        Widths of successive radial annuli in lambda.
    data_column : str
        Data column selection (auto | DATA | CORRECTED_DATA | RESIDUAL).
    quantity : str
        Visibility quantity to threshold (amplitude | phase | real | imag).
    zarr_path : str or None
        Path for Zarr intermediate store.  None uses a tempdir.
    chunk_size : int
        Rows per MS read chunk.
    n_readers : int
        Number of parallel reader processes.
    min_neighbors : int
        Minimum occupied neighbors for local threshold; fewer falls back
        to annular.
    spw_ids : tuple of int or None
        Restrict to these spectral window IDs.
    field_ids : tuple of int or None
        Restrict to these field IDs.
    uvrange : (float, float) or None
        UV distance range in lambda as (uv_min, uv_max).
    plot_dir : str or None
        If set, write before/after diagnostic PNGs to this directory.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns
    -------
    dict
        Keys: ms_path, zarr_path, grid_shape, total_newly_flagged,
        elapsed_s, plots.
    """
    from gridflag.utils import setup_logging

    setup_logging(log_level)

    config = GridFlagConfig(
        cell_size=cell_size,
        nsigma=nsigma,
        smoothing_window=smoothing_window,
        annulus_widths=annulus_widths,
        data_column=data_column,
        quantity=quantity,
        zarr_path=zarr_path,
        chunk_size=chunk_size,
        n_readers=n_readers,
        min_neighbors=min_neighbors,
        spw_ids=spw_ids,
        field_ids=field_ids,
        uvrange=uvrange,
    )

    return run(ms_path, config, plot_dir=plot_dir)
