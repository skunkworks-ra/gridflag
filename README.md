# GRIDflag

UV-plane RFI flagging for radio interferometric data, implementing the algorithm described in [Sekhar et al. (2018)](https://doi.org/10.3847/1538-3881/aab167).

GRIDflag grids visibilities onto a 2D UV plane, computes robust per-cell statistics (median, MAD), derives thresholds via local neighborhood and annular methods, and flags outliers. It operates per-correlation (RR/RL/LR/LL or XX/XY/YX/YY) and supports any CASA Measurement Set.

## Installation

Requires Python 3.10+ and [casatools](https://casa.nrao.edu/).

```bash
pip install -e ".[dev]"
```

If casatools is not already in your environment:

```bash
pip install casatools
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array computation |
| scipy | Uniform filter for local thresholds |
| numba | JIT-compiled per-cell median/MAD |
| zarr (v2) | Intermediate storage |
| click | CLI |
| casatools | Measurement Set I/O |

## Quick start

### Command line

```bash
# Basic usage with defaults
gridflag /path/to/data.ms

# Custom threshold and cell size
gridflag /path/to/data.ms --nsigma 5.0 --cell-size 20.0

# Restrict UV range and generate diagnostic plots
gridflag /path/to/data.ms --uvrange 100,50000 --plot-dir ./plots

# Process specific SPWs and fields
gridflag /path/to/data.ms --spw 0 --spw 2 --field 1

# Use a specific data column
gridflag /path/to/data.ms --data-column CORRECTED_DATA
```

Also available as `python -m gridflag`.

### Python API

```python
from gridflag import gridflag

# Simple — all keyword args with defaults
result = gridflag("/path/to/data.ms", nsigma=3.0, cell_size=10.0)

print(f"Flagged {result['total_newly_flagged']} visibilities in {result['elapsed_s']:.1f}s")
```

```python
from gridflag import GridFlagConfig, run

# Advanced — explicit config object
config = GridFlagConfig(
    nsigma=5.0,
    cell_size=20.0,
    quantity="phase",
    uvrange=(100.0, 50000.0),
    data_column="auto",
)

result = run("/path/to/data.ms", config, plot_dir="./plots")
```

The return dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `ms_path` | str | Input MS path |
| `zarr_path` | str | Zarr intermediate store path |
| `grid_shape` | (int, int) | UV grid dimensions |
| `total_newly_flagged` | int | Number of visibilities newly flagged |
| `elapsed_s` | float | Wall-clock time in seconds |
| `plots` | list[str] | Paths to diagnostic PNGs (if `plot_dir` was set) |

## CLI reference

```
Usage: gridflag [OPTIONS] MS_PATH

Options:
  --cell-size FLOAT               Grid cell size in lambda.  [default: 10.0]
  --nsigma FLOAT                  Sigma threshold multiplier.  [default: 3.0]
  --smoothing-window INTEGER      Neighborhood kernel size.  [default: 5]
  --data-column TEXT              Data column (auto | DATA | CORRECTED_DATA
                                  | RESIDUAL).  [default: auto]
  --quantity [amplitude|phase|real|imag]
                                  Quantity to threshold on.  [default: amplitude]
  --zarr-path TEXT                Path for Zarr store (default: CWD).
  --chunk-size INTEGER            Rows per MS read chunk.  [default: 50000]
  --n-readers INTEGER             Number of parallel reader processes.  [default: 4]
  --min-neighbors INTEGER         Min occupied neighbors for local threshold.  [default: 3]
  --uvrange TEXT                  UV range in lambda as UVMIN,UVMAX (e.g. 100,50000).
  --spw INTEGER                   SPW IDs to process (repeatable).
  --field INTEGER                 Field IDs to process (repeatable).
  --plot-dir PATH                 Directory for before/after diagnostic plots.
  --log-level [DEBUG|INFO|WARNING|ERROR]  [default: INFO]
  -h, --help                      Show this message and exit.
```

## Algorithm overview

### Data flow

```
Pass 1: READ
  MS (chunked by row) → per-channel UV coordinates (λ)
    → Hermitian fold (v<0 → v≥0, conjugate vis)
    → cell assignment (nearest-neighbor gridding)
    → extract quantity (amplitude/phase/real/imag)
    → accumulate flat arrays → flush to Zarr

Pass 1.5: COMPUTE (Zarr → NumPy, no MS access)
  Per (SPW, correlation):
    → per-cell median, MAD → robust σ (1.4826 × MAD)
    → local neighborhood threshold (masked uniform filter)
    → annular threshold (radial bin averages)
    → combined threshold: min(local, annular); annular-only if sparse
    → flag visibilities exceeding threshold

Pass 2: WRITE
  Batch flag write-back to MS FLAG column (logical OR, never unflags)
```

### Data column resolution

When `data_column="auto"` (default), GRIDflag selects data in priority order:

1. `RESIDUAL` column (use directly if present)
2. `DATA - MODEL_DATA` (if MODEL_DATA exists and is non-zero)
3. `CORRECTED_DATA - MODEL_DATA` (if CORRECTED_DATA exists and MODEL_DATA is non-zero)
4. `DATA` (fallback)

The resolved column is logged at startup. Override with `--data-column`.

### Threshold computation

**Local neighborhood**: A K×K uniform filter (default 5×5) averages the median and σ grids over occupied cells, producing a smoothed local threshold per cell.

**Annular**: Cells are binned by UV distance from the origin into configurable annuli. Per-annulus weighted averages of median and σ yield a radially-symmetric threshold.

**Combined**: `threshold = min(local, annular)` per cell. Cells with fewer than `min_neighbors` occupied neighbors use the annular threshold only.

A visibility is flagged if its value exceeds `threshold = avg_median + nsigma × avg_σ` for its cell.

### Hermitian symmetry

Radio interferometric visibilities satisfy V(u,v) = V*(-u,-v). GRIDflag folds the v<0 half-plane onto v≥0 (conjugating visibilities), halving the grid size. Grid shape: `(2N+1, N+1)` where `N = ceil(uv_max / cell_size)`.

### Intermediate storage

Zarr is used as intermediate storage between the read and compute passes. By default the store is created in the current working directory as `tmp_gridflag_uv_<id>.zarr`. This avoids dependence on `/tmp` which may not be writable in cluster environments. Use `--zarr-path` to specify an explicit location.

## Diagnostic plots

Pass `--plot-dir ./plots` to generate before/after comparison plots for each (SPW, correlation). Two PNGs are produced per pair:

- `spw{N}_corr{M}_median.png` — median grid before and after flagging
- `spw{N}_corr{M}_std.png` — robust σ grid before and after flagging

Both panels share the same colour scale with a colorbar, and axes are labelled in kλ.

## Performance

Benchmarked on a 1.5 GB GMRT 150 MHz dataset (629k rows, 42 channels, 2 correlations):

| Phase | Time |
|-------|------|
| Read + accumulate | 2.1s |
| Compute (numba gridder + thresholds) | 4.1s |
| Write flags | 0.5s |
| **Total** | **~8s** |

Key optimisations:
- In-memory accumulation with single Zarr flush (no per-chunk resize)
- Numba JIT for per-cell median/MAD computation
- Batched flag writes with vectorised indexing (10k-row blocks)
- Precomputed frequency/c ratios, UV distance² comparisons (no sqrt)

## References

Sekhar, S. & Athreya, R. (2018). "GRIDflag: A GMRT RFI Flagging Pipeline." *The Astronomical Journal*, 156(1), 9. [doi:10.3847/1538-3881/aab167](https://doi.org/10.3847/1538-3881/aab167)

## License

MIT
