"""Click CLI for GRIDflag."""

from __future__ import annotations

import click

from gridflag.config import GridFlagConfig
from gridflag.utils import setup_logging


@click.command()
@click.argument("ms_path", type=click.Path(exists=True))
@click.option("--cell-size", default=10.0, help="Grid cell size in lambda.")
@click.option("--nsigma", default=3.0, help="Sigma threshold multiplier.")
@click.option("--smoothing-window", default=5, help="Neighborhood kernel size.")
@click.option("--data-column", default="auto",
              help="Data column (auto | DATA | CORRECTED_DATA | RESIDUAL).")
@click.option("--quantity", default="amplitude",
              type=click.Choice(["amplitude", "phase", "real", "imag"]),
              help="Quantity to threshold on.")
@click.option("--zarr-path", default=None, help="Path for Zarr store (default: tempdir).")
@click.option("--chunk-size", default=50_000, help="Rows per MS read chunk.")
@click.option("--n-readers", default=4, help="Number of parallel reader processes.")
@click.option("--min-neighbors", default=3, help="Min occupied neighbors for local threshold.")
@click.option("--spw", "spw_ids", multiple=True, type=int, help="SPW IDs to process.")
@click.option("--field", "field_ids", multiple=True, type=int, help="Field IDs to process.")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(
    ms_path: str,
    cell_size: float,
    nsigma: float,
    smoothing_window: int,
    data_column: str,
    quantity: str,
    zarr_path: str | None,
    chunk_size: int,
    n_readers: int,
    min_neighbors: int,
    spw_ids: tuple[int, ...],
    field_ids: tuple[int, ...],
    log_level: str,
) -> None:
    """Run GRIDflag on a CASA Measurement Set."""
    setup_logging(log_level)

    config = GridFlagConfig(
        cell_size=cell_size,
        nsigma=nsigma,
        smoothing_window=smoothing_window,
        data_column=data_column,
        quantity=quantity,
        zarr_path=zarr_path,
        chunk_size=chunk_size,
        n_readers=n_readers,
        min_neighbors=min_neighbors,
        spw_ids=spw_ids or None,
        field_ids=field_ids or None,
    )

    from gridflag.pipeline import run

    result = run(ms_path, config)
    click.echo(f"Flagged {result['total_newly_flagged']} visibilities.")
    click.echo(f"Zarr store: {result['zarr_path']}")
