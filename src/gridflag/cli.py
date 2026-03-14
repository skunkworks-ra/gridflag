"""Click CLI for GRIDflag."""

from __future__ import annotations

import click

from gridflag.config import GridFlagConfig
from gridflag.utils import setup_logging

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("ms_path", type=click.Path(exists=True), required=False, default=None)
@click.option("--cell-size", default=10.0, show_default=True, help="Grid cell size in lambda.")
@click.option("--nsigma", default=3.0, show_default=True, help="Sigma threshold multiplier.")
@click.option("--smoothing-window", default=5, show_default=True, help="Neighborhood kernel size.")
@click.option(
    "--data-column",
    default="auto",
    show_default=True,
    help="Data column (auto | DATA | CORRECTED_DATA | RESIDUAL).",
)
@click.option(
    "--quantity",
    default="amplitude",
    show_default=True,
    type=click.Choice(["amplitude", "phase", "real", "imag"]),
    help="Quantity to threshold on.",
)
@click.option(
    "--zarr-path", default=None, show_default=True, help="Path for Zarr store (default: tempdir)."
)
@click.option(
    "--n-workers",
    default=0,
    show_default=True,
    help="Number of parallel reader processes (0 = auto).",
)
@click.option(
    "--min-neighbors",
    default=3,
    show_default=True,
    help="Min occupied neighbors for local threshold.",
)
@click.option("--uvrange", default=None, help="UV range in lambda as UVMIN,UVMAX (e.g. 100,50000).")
@click.option("--spw", "spw_ids", multiple=True, type=int, help="SPW IDs to process (repeatable).")
@click.option(
    "--field", "field_ids", multiple=True, type=int, help="Field IDs to process (repeatable)."
)
@click.option(
    "--plot-dir",
    default=None,
    type=click.Path(),
    help="Directory for before/after diagnostic plots.",
)
@click.option(
    "--persist-cache",
    is_flag=True,
    default=False,
    help="Keep the Zarr intermediate store after the run.",
)
@click.option(
    "--plot-cached",
    default=None,
    type=click.Path(exists=True),
    help="Generate plots from an existing Zarr cache (no MS needed).",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
def main(
    ms_path: str | None,
    cell_size: float,
    nsigma: float,
    smoothing_window: int,
    data_column: str,
    quantity: str,
    zarr_path: str | None,
    n_workers: int,
    uvrange: str | None,
    min_neighbors: int,
    spw_ids: tuple[int, ...],
    field_ids: tuple[int, ...],
    plot_dir: str | None,
    persist_cache: bool,
    plot_cached: str | None,
    log_level: str,
) -> None:
    """Run GRIDflag UV-plane RFI flagging on a CASA Measurement Set."""
    setup_logging(log_level)

    # ── Plot-from-cache mode ──────────────────────────────────────
    if plot_cached is not None:
        if plot_dir is None:
            raise click.UsageError("--plot-cached requires --plot-dir")
        from gridflag.pipeline import plot_from_cache

        paths = plot_from_cache(plot_cached, plot_dir)
        click.echo(f"Generated {len(paths)} plots from {plot_cached}")
        if paths:
            click.echo(f"Plots: {', '.join(paths)}")
        return

    # ── Normal pipeline mode ──────────────────────────────────────
    if ms_path is None:
        raise click.UsageError("MS_PATH is required (unless using --plot-cached)")

    # Parse uvrange string.
    parsed_uvrange = None
    if uvrange is not None:
        parts = uvrange.split(",")
        if len(parts) != 2:
            raise click.BadParameter(
                "Must be two comma-separated values: UVMIN,UVMAX", param_hint="--uvrange"
            )
        parsed_uvrange = (float(parts[0]), float(parts[1]))

    config = GridFlagConfig(
        cell_size=cell_size,
        nsigma=nsigma,
        smoothing_window=smoothing_window,
        data_column=data_column,
        quantity=quantity,
        zarr_path=zarr_path,
        n_workers=n_workers,
        min_neighbors=min_neighbors,
        spw_ids=spw_ids or None,
        field_ids=field_ids or None,
        uvrange=parsed_uvrange,
    )

    from gridflag.pipeline import run

    result = run(ms_path, config, plot_dir=plot_dir, persist_cache=persist_cache)
    click.echo(f"Flagged {result['total_newly_flagged']} visibilities.")
    if result.get("zarr_path"):
        click.echo(f"Zarr store: {result['zarr_path']}")
    if result.get("plots"):
        click.echo(f"Plots: {', '.join(result['plots'])}")
