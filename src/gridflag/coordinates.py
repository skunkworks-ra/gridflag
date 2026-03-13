"""UV coordinate scaling, Hermitian folding, and cell assignment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

C_M_S = 299_792_458.0  # speed of light, m/s


def scale_uv(
    uvw_ref: NDArray[np.floating],
    freq_chan: NDArray[np.floating],
    freq_ref: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Scale UVW (metres) to per-channel UVW in wavelengths.

    The MS UVW column stores geometric coordinates in metres, independent
    of frequency.  Converting to wavelengths: ``uvw_λ = uvw_m × ν / c``.

    Parameters
    ----------
    uvw_ref : (N_row, 3) float64 — UVW in metres.
    freq_chan : (N_chan,) float64 — channel frequencies in Hz.
    freq_ref : float — reference frequency (unused, kept for API compat).

    Returns
    -------
    u, v, w : each (N_row, N_chan) float64, in wavelengths.
    """
    freq = freq_chan[np.newaxis, :]  # (1, N_chan)
    u = uvw_ref[:, 0:1] * freq / C_M_S
    v = uvw_ref[:, 1:2] * freq / C_M_S
    w = uvw_ref[:, 2:3] * freq / C_M_S
    return u, v, w


def hermitian_fold(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    vis: NDArray[np.complexfloating],
) -> tuple[NDArray, NDArray, NDArray]:
    """Fold v<0 half-plane onto v≥0 by conjugating visibilities.

    Returns copies; inputs are not mutated.
    """
    u = u.copy()
    v = v.copy()
    vis = vis.copy()
    neg = v < 0
    u[neg] = -u[neg]
    v[neg] = -v[neg]
    vis[neg] = np.conj(vis[neg])
    return u, v, vis


def compute_N(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    cell_size: float,
) -> int:
    """Compute the grid half-size N from UV extents."""
    uv_max = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))))
    return int(np.ceil(uv_max / cell_size))


def uv_to_cell(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    cell_size: float,
    N: int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp], int]:
    """Convert UV coordinates (λ) to integer cell indices.

    After Hermitian folding, v ≥ 0.  u may be negative.

    Parameters
    ----------
    u, v : arrays of UV coordinates in lambda.
    cell_size : grid cell size in lambda.
    N : grid half-size.  If None, computed from the data.

    Returns
    -------
    cell_u : int array, offset so ≥ 0.
    cell_v : int array, ≥ 0.
    N : the grid half-size used.
    """
    if N is None:
        N = compute_N(u, v, cell_size)
    cell_u = np.rint(u / cell_size).astype(np.intp) + N
    cell_v = np.rint(v / cell_size).astype(np.intp)
    return cell_u, cell_v, N


def grid_shape(N: int) -> tuple[int, int]:
    """Grid shape ``(2*N + 1, N + 1)`` — full u range, half-plane v."""
    return (2 * N + 1, N + 1)
