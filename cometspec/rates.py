"""Radiative rate matrix, steady-state solver, g-factors, spectrum synthesis,
and line-spread-function helpers.

Routines
------------
- :func:`build_rate_matrix_nbar` -- Build the radiative rate matrix from normalized transitions.
- :func:`solve_with_normalization` -- Solve the steady-state system ``M @ n = 0`` with ``sum(n) = 1``.
- :func:`solve_with_normalization_fast` -- Fast buffer-reusing version of :func:`solve_with_normalization`.
- :func:`g_factors` -- Compute per-line and total photon/energy g-factors.
- :func:`g_factors_fast_from_cache` -- Fast g-factor evaluation using precomputed arrays.
- :func:`synth_spectrum_from_lines` -- Build a synthetic emission spectrum from line g-factors.
- :func:`make_lsf` -- Create a line-spread function callable from parameter values.
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional, Callable, Any

import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.table import Table
import warnings
from scipy.linalg import LinAlgWarning, solve
from scipy.signal import fftconvolve



__all__ = [
    "build_rate_matrix_nbar",
    "solve_with_normalization",
    "solve_with_normalization_fast",
    "g_factors",
    "g_factors_fast_from_cache",
    "synth_spectrum_from_lines",
    "make_lsf",
]


def _as_array(obj: Any, name: str) -> np.ndarray:
    """Return a named column as a NumPy array.

    Parameters
    ----------
    obj : Any
        Table-like container with named columns.
    name : str
        Column name.

    Returns
    -------
    numpy.ndarray
        Column values as a NumPy array.
    """
    if hasattr(obj, "colnames"):        # astropy Table
        return np.asarray(obj[name])
    if hasattr(obj, "columns"):         # pandas DataFrame
        return np.asarray(obj[name].values)
    return np.asarray(obj[name])


def build_rate_matrix_nbar(
    lines: Table,
    *,
    include_stim_emission: bool = False,
    verbose: bool = True,
    A_col: str = "A_ul",
    upper_id_col: str = "upper_id",
    lower_id_col: str = "lower_id",
    g_upper_col: str = "g_upper",
    g_lower_col: str = "g_lower",
):
    """Build the radiative rate matrix from normalized transitions.

    Parameters
    ----------
    lines : astropy.table.Table
        Transition table with solar incident irradiance quantities attached.
    include_stim_emission : bool, optional, default False
        Include stimulated emission in downward rates.
    verbose : bool, optional, default True
        Print diagnostic information.
    A_col : str, optional, default "A_ul"
        Einstein A column name.
    upper_id_col : str, optional, default "upper_id"
        Upper-level ID column name.
    lower_id_col : str, optional, default "lower_id"
        Lower-level ID column name.
    g_upper_col : str, optional, default "g_upper"
        Upper-level degeneracy column name.
    g_lower_col : str, optional, default "g_lower"
        Lower-level degeneracy column name.

    Returns
    -------
    tuple[numpy.ndarray, dict[int, str], astropy.table.Table]
        Rate matrix, a dictionary that maps each row/column index of ``M`` back to its physical level identifier (the string from ``upper_id_col`` / ``lower_id_col``), and the input line lists with new columns (`__nu_Hz`, `__R_lu`, `__R_ul`, `__upper_idx`, `__lower_idx`).
    """
    lines_out = lines.copy()

    nu = np.asarray(lines_out["Frequency_Hz"], float) * u.Hz
    A_ul = np.asarray(lines_out[A_col], float) / u.s

    Jnu_cgs = np.asarray(lines_out["J_nu_erg_cm2_s_Hz_sr"], float) * (u.erg / (u.cm**2 * u.s * u.Hz * u.sr))
    Jnu_SI = Jnu_cgs.to(u.W / (u.m**2 * u.Hz * u.sr))

    gu = np.asarray(lines_out[g_upper_col], float)
    gl = np.asarray(lines_out[g_lower_col], float)

    B_lu = (A_ul * const.c**2 / (2.0 * const.h * nu**3) * (gu / gl)).decompose().value
    B_ul = (A_ul * const.c**2 / (2.0 * const.h * nu**3)).decompose().value

    R_lu = B_lu * Jnu_SI.value
    R_ul = A_ul.value
    if include_stim_emission:
        R_ul = R_ul + B_ul * Jnu_SI.value

    upper_ids = np.asarray(lines_out[upper_id_col], str)
    lower_ids = np.asarray(lines_out[lower_id_col], str)

    level_to_idx: Dict[str, int] = {}
    upper_idx: list[int] = []
    lower_idx: list[int] = []

    for u_id, l_id in zip(upper_ids, lower_ids):
        if u_id not in level_to_idx:
            level_to_idx[u_id] = len(level_to_idx)
        if l_id not in level_to_idx:
            level_to_idx[l_id] = len(level_to_idx)
        upper_idx.append(level_to_idx[u_id])
        lower_idx.append(level_to_idx[l_id])

    idx_to_level = {v: k for k, v in level_to_idx.items()}
    n_levels = len(idx_to_level)

    M = np.zeros((n_levels, n_levels), float)

    def add_rate(dest: int, src: int, rate: float):
        if not np.isfinite(rate) or rate <= 0.0:
            return
        M[src, src] -= rate
        M[dest, src] += rate

    for iu, il, rlu, rul in zip(upper_idx, lower_idx, R_lu, R_ul):
        add_rate(iu, il, float(rlu))
        add_rate(il, iu, float(rul))

    lines_out["__nu_Hz"] = nu.to_value(u.Hz)
    lines_out["__R_lu"] = np.asarray(R_lu, float)
    lines_out["__R_ul"] = np.asarray(R_ul, float)
    lines_out["__upper_idx"] = np.asarray(upper_idx, int)
    lines_out["__lower_idx"] = np.asarray(lower_idx, int)

    if verbose:
        print(f"[diag] N levels: {n_levels} | all finite: {np.isfinite(M).all()}")
    return M, idx_to_level, lines_out


def solve_with_normalization(M: np.ndarray, *, verbose: bool = True) -> np.ndarray:
    """Solve the steady-state system ``Mn = 0`` with ``sum(n) = 1``.

    Parameters
    ----------
    M : numpy.ndarray
        Rate matrix.
    verbose : bool, optional, default True
        Print solver diagnostics.

    Returns
    -------
    numpy.ndarray
        Normalized non-negative level populations.

    Raises
    ------
    RuntimeError
        If no positive normalized solution can be formed.

    Notes
    -----
    If ``np.linalg.lstsq`` fails to converge (e.g. because M contains NaN or
    Inf from extreme parameter values), a ``LinAlgError`` is caught and an
    array of NaN is returned. In the MCMC context this propagates to a
    non-finite model flux so the walker step is rejected (log-likelihood → -inf)
    rather than crashing. In the direct-modelling context ``best_model`` will
    contain NaN, signalling that the chosen parameters are unphysical.
    """
    n_levels = M.shape[0]
    A = M.astype(float)
    b = np.zeros(n_levels)
    A[0, :] = 1.0
    b[0] = 1.0

    try:
        n, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(n_levels, np.nan)
    if np.any(n < 0):
        n = np.clip(n, 0.0, None)
    s = n.sum()
    if s <= 0:
        raise RuntimeError("Degenerate M: no positive steady-state solution.")
    n /= s

    if verbose:
        print("[solver] sum(n) =", n.sum())
    return n

def solve_with_normalization_fast(M, A_work, b_work):
    """Fast buffer-reusing version of :func:`solve_with_normalization`.

    Parameters
    ----------
    M : numpy.ndarray
        Rate matrix.
    A_work : numpy.ndarray
        Reusable matrix buffer.
    b_work : numpy.ndarray
        Reusable right-hand-side buffer.

    Returns
    -------
    numpy.ndarray
        Normalized non-negative level populations.

    Raises
    ------
    RuntimeError
        If no positive normalized solution can be formed.

    Notes
    -----
    The first row of ``M`` is overwritten with ones (and ``b[0] = 1``) to
    impose ``sum(n) = 1`` directly, so the resulting square system is
    non-singular for any well-posed rate matrix and is solved with
    :func:`scipy.linalg.solve` (LU, with ``overwrite_a``/``overwrite_b`` and
    ``check_finite=False`` to skip the input scan and reuse the work buffers
    in-place — several times faster than the SVD-based :func:`numpy.linalg.lstsq`
    used by :func:`solve_with_normalization`). If the matrix is singular
    (e.g. NaN/Inf collision rates at extreme parameter values), the raised
    ``LinAlgError`` is caught and an array of NaN is returned. In the MCMC
    context this propagates to a non-finite model flux so the walker step
    is rejected (log-likelihood -> -inf) rather than crashing. In the
    direct-modelling context ``best_model`` will contain NaN, signalling
    that the chosen parameters are unphysical.
    """

    np.copyto(A_work, M)
    b_work.fill(0.0)
    A_work[0, :] = 1.0
    b_work[0] = 1.0

    # Rank-deficiency / extreme-ill-conditioning screen. LU and lstsq pick
    # different valid solutions on (near-)singular systems, so route those
    # to lstsq (minimum-norm) to stay consistent with solve_with_normalization.
    # Cheap O(N) test: zero diagonals (dead-end levels) or wide dynamic range
    # in loss rates (effective rank deficiency from huge magnitude spread).
    diag = np.abs(np.diag(M))
    diag_max = diag.max() if diag.size else 0.0
    diag_min = diag.min() if diag.size else 0.0
    rank_deficient = diag_max > 0.0 and diag_min < 1e-12 * diag_max

    if rank_deficient:
        lu_ok = False
        n = None
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                n = solve(A_work, b_work,
                          overwrite_a=True, overwrite_b=True,
                          check_finite=False, assume_a='gen')
            lu_ok = np.all(np.isfinite(n))
            if lu_ok:
                resid = np.linalg.norm(M @ n, ord=np.inf)
                scale = np.linalg.norm(M, ord=np.inf) * np.linalg.norm(n, ord=np.inf)
                if scale > 0.0 and resid / scale > 1e-6:
                    lu_ok = False
        except np.linalg.LinAlgError:
            lu_ok = False
            n = None

    if not lu_ok:
        A_fb = M.astype(float, copy=True)
        A_fb[0, :] = 1.0
        b_fb = np.zeros(M.shape[0], dtype=float)
        b_fb[0] = 1.0
        try:
            n, *_ = np.linalg.lstsq(A_fb, b_fb, rcond=None)
        except np.linalg.LinAlgError:
            return np.full(M.shape[0], np.nan)
        if not np.all(np.isfinite(n)):
            return np.full(M.shape[0], np.nan)

    n = np.clip(n, 0.0, None)
    s = n.sum()
    if s <= 0.0:
        return np.full(M.shape[0], np.nan)
    n = n / s
    return n


def g_factors(
    lines_with_rates: Table,
    n: np.ndarray,
    *,
    A_col: str = "A_ul",
):
    """Compute per-line and total photon/energy g-factors.

    The g-factors are in units of photons/s/molecule for ``g_phot`` and erg/s/molecule for ``g_energy``.

    Parameters
    ----------
    lines_with_rates : astropy.table.Table
        Transition table with cached rate-matrix indices.
    n : numpy.ndarray
        Level populations.
    A_col : str, optional, default "A_ul"
        Einstein A column name.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        ``(g_phot, g_energy, sum_g_phot, sum_g_energy)``.
    """
    nu = np.asarray(lines_with_rates["__nu_Hz"], float)
    A_ul = np.asarray(lines_with_rates[A_col], float)
    ui = np.asarray(lines_with_rates["__upper_idx"], int)

    nu = np.nan_to_num(nu, 0.0, 0.0, 0.0)
    A_ul = np.nan_to_num(A_ul, 0.0, 0.0, 0.0)

    n_u = n[ui]
    g_ph = n_u * A_ul
    g_en = const.h.cgs.value * nu * g_ph

    g_ph = np.nan_to_num(g_ph, 0.0, 0.0, 0.0)
    g_en = np.nan_to_num(g_en, 0.0, 0.0, 0.0)

    return g_ph, g_en, float(g_ph.sum()), float(g_en.sum())

def g_factors_fast_from_cache(
    *,
    ui: np.ndarray,     
    A_ul: np.ndarray,    
    hnu: np.ndarray,    
    n: np.ndarray,
    out_g_ph: Optional[np.ndarray] = None,
    out_g_en: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast g-factor evaluation using precomputed arrays.

    Parameters
    ----------
    ui : numpy.ndarray
        Upper-level indices per line.
    A_ul : numpy.ndarray
        Einstein A coefficients per line.
    hnu : numpy.ndarray
        ``h * nu`` per line in erg.
    n : numpy.ndarray
        Level populations.
    out_g_ph : numpy.ndarray, optional, default None
        Optional output buffer for photon g-factors.
    out_g_en : numpy.ndarray, optional, default None
        Optional output buffer for energy g-factors.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(g_ph, g_en)`` arrays.
    """
    n_u = n[ui]

    if out_g_ph is None:
        g_ph = n_u * A_ul
    else:
        np.multiply(n_u, A_ul, out=out_g_ph)
        g_ph = out_g_ph

    if out_g_en is None:
        g_en = hnu * g_ph
    else:
        np.multiply(hnu, g_ph, out=out_g_en)
        g_en = out_g_en

    np.nan_to_num(g_ph, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(g_en, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return g_ph, g_en


def synth_spectrum_from_lines(
    df_lines: Table,
    N_col_cm2: float,
    *,
    g_line_energy: Optional[np.ndarray] = None,
    g_line_phot: Optional[np.ndarray] = None,
    fwhm_A: float = 0.02,
    dlam_A: float = 0.05,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    lam_col: str = "Wave_vac_AA",
    Omega_sr: Optional[float] = None,
    grid: Optional[np.ndarray] = None,
    lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    v_shift_kms: float = 0.0,
    dlam_shift_A: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a synthetic emission spectrum from line g-factors.

    Parameters
    ----------
    df_lines : astropy.table.Table
        Transition table. **Required**
    N_col_cm2 : float
        Column density in molecules cm^-2. **Required**
    g_line_energy : numpy.ndarray, optional, default None
        Per-line energy g-factors. If not provided ``g_line_phot`` **must** be used to compute line intensities.
    g_line_phot : numpy.ndarray, optional, default None
        Per-line photon g-factors. If not provided, ``g_line_energy`` **must** be used to compute line intensities.
    fwhm_A : float, optional, default 0.02
        Gaussian FWHM in Angstrom if no custom LSF is provided.
    dlam_A : float, optional, default 0.05
        Wavelength step for an auto-generated grid.
    lam_min : float, optional, default None
        Optional minimum wavelength (after wavelength shifts) for an auto-generated grid. The minimum grid value will be at least 5 FWHM below the minimum line wavelength.
    lam_max : float, optional, default None
        Optional maximum wavelength (after wavelength shifts) for an auto-generated grid. The maximum grid value will be at least 5 FWHM above the maximum line wavelength.
    lam_col : str, optional, default "Wave_vac_AA". If None, it will be set to "Wave_vac_AA".
        Wavelength column in ``df_lines``.
    Omega_sr : float, optional, default None
        Optional solid angle scaling in sr.
    grid : numpy.ndarray, optional, default None
        Optional output wavelength grid.
    lsf : Callable[[numpy.ndarray], numpy.ndarray], optional, default None
        Optional custom line-spread function.
    v_shift_kms : float, optional, default 0.0
        Velocity shift in km/s for the synthetic spectrum lines.
    dlam_shift_A : float, optional, default 0.0
        Additive wavelength shift in Angstrom for the synthetic spectrum lines.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(wavelength_grid, synthetic_flux)``.

    Raises
    ------
    ValueError
        If required inputs are missing.
    """
    if lam_col not in df_lines.colnames:
        raise ValueError(f"{lam_col!r} not found in df_lines.")

    lam_rest = np.asarray(df_lines[lam_col], float)

    if v_shift_kms is not None:
        c_kms = const.c.to("km/s").value
        lam = lam_rest * (1.0 + v_shift_kms / c_kms)
        if dlam_shift_A != 0.0:
            lam = lam + dlam_shift_A
    else:
        lam = lam_rest + dlam_shift_A

    if g_line_energy is not None:
        I_line = np.asarray(g_line_energy, float)
    elif g_line_phot is not None:
        if "__nu_Hz" in df_lines.colnames:
            nu = np.asarray(df_lines["__nu_Hz"], float)
        else:
            nu = (const.c / (lam * u.AA)).to_value(u.Hz)
        I_line = const.h.cgs.value * nu * np.asarray(g_line_phot, float)
    else:
        raise ValueError("Provide g_line_energy or g_line_phot.")

    m = np.isfinite(lam) & np.isfinite(I_line) & (I_line > 0.0)
    lam = lam[m]
    I_line = I_line[m]
    if lam.size == 0:
        if grid is None:
            return np.array([]), np.array([])
        return np.asarray(grid, float), np.zeros_like(grid, float)

    if lam_min is None:
        lam_min = float(lam.min() - 5.0 * fwhm_A)
    if lam_max is None:
        lam_max = float(lam.max() + 5.0 * fwhm_A)

    if grid is None:
        if dlam_A <= 0.0:
            dlam_A = fwhm_A / 3.0
        ngrid = int(np.ceil((lam_max - lam_min) / dlam_A)) + 1
        grid = lam_min + np.arange(ngrid, dtype=float) * dlam_A
    else:
        grid = np.asarray(grid, float)

    if lsf is None:
        sigma = fwhm_A / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        norm = sigma * np.sqrt(2.0 * np.pi)
        spec_per_mol = np.zeros_like(grid, dtype=float)
        for l0, I0 in zip(lam, I_line):
            spec_per_mol += I0 * np.exp(-0.5 * ((grid - l0) / sigma) ** 2) / norm
    else:
        # Stick-and-convolve fast path for translation-invariant LSFs on a
        # uniform grid: build a delta-stick spectrum (each line linearly
        # distributed into its two nearest bins), then FFT-convolve once with
        # a single LSF kernel. This replaces the O(N_grid * N_lines) dense
        # outer product with O(N_grid log N_grid + N_lines), which is the
        # dominant per-step cost in the MCMC inner loop when the line list is
        # large. Falls back to the dense path when the grid is non-uniform.
        N_grid = grid.size
        if N_grid >= 2:
            dlam_grid = float(grid[1] - grid[0])
            uniform = (
                dlam_grid > 0.0
                and np.allclose(np.diff(grid), dlam_grid, rtol=1e-6, atol=0.0)
            )
        else:
            uniform = False

        if uniform:
            x = (lam - grid[0]) / dlam_grid
            idx_lo = np.floor(x).astype(np.intp)
            frac = x - idx_lo
            m_lo = (idx_lo >= 0) & (idx_lo < N_grid)
            m_hi = (idx_lo + 1 >= 0) & (idx_lo + 1 < N_grid)
            sticks = np.zeros(N_grid, dtype=float)
            if m_lo.any():
                sticks += np.bincount(
                    idx_lo[m_lo],
                    weights=((1.0 - frac) * I_line)[m_lo],
                    minlength=N_grid,
                )
            if m_hi.any():
                sticks += np.bincount(
                    idx_lo[m_hi] + 1,
                    weights=(frac * I_line)[m_hi],
                    minlength=N_grid,
                )
            M_k = N_grid if (N_grid % 2 == 1) else (N_grid - 1)
            K = M_k // 2
            dl_kernel = (np.arange(M_k) - K) * dlam_grid
            kernel = lsf(dl_kernel)
            spec_per_mol = fftconvolve(sticks, kernel, mode="same")
        else:
            dl = grid[:, None] - lam[None, :]
            prof = lsf(dl)
            spec_per_mol = (prof * I_line).sum(axis=1)

    I_lambda = (N_col_cm2 / (4.0 * np.pi)) * spec_per_mol

    if Omega_sr is not None:
        return grid, I_lambda * Omega_sr
    else:
        return grid, I_lambda


# =============================================================================
# LSF helper
# =============================================================================

class _Gauss2LSF:
    """Two-Gaussian line-spread function: ``ratio * G(sigma1) + (1-ratio) * G(sigma2)``."""

    def __init__(self, sigma1: float, sigma2: float, ratio: float):
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.ratio = float(ratio)

    def __call__(self, dl: np.ndarray) -> np.ndarray:
        g1 = np.exp(-0.5 * (dl / self.sigma1) ** 2) / (self.sigma1 * np.sqrt(2.0 * np.pi))
        g2 = np.exp(-0.5 * (dl / self.sigma2) ** 2) / (self.sigma2 * np.sqrt(2.0 * np.pi))
        return self.ratio * g1 + (1.0 - self.ratio) * g2


class _GaussLSF:
    """Single-Gaussian line-spread function with width ``sigma``."""

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, dl: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (dl / self.sigma) ** 2) / (self.sigma * np.sqrt(2.0 * np.pi))


class _GaussLorentzLSF:
    """Gaussian + Lorentzian mixture: ``ratio * G(sigma_G) + (1-ratio) * L(fwhm_L)``."""

    def __init__(self, sigma_G: float, fwhm_L: float, ratio: float):
        self.sigma_G = float(sigma_G)
        self.fwhm_L = float(fwhm_L)
        self.ratio = float(ratio)

    def __call__(self, dl: np.ndarray) -> np.ndarray:
        gamma = self.fwhm_L / 2.0
        A = 2.0 / (np.pi * self.fwhm_L)
        gauss = np.exp(-0.5 * (dl / self.sigma_G) ** 2) / (self.sigma_G * np.sqrt(2.0 * np.pi))
        lorentz = A * gamma**2 / (gamma**2 + dl**2)
        return self.ratio * gauss + (1.0 - self.ratio) * lorentz


class _LorentzLSF:
    """Lorentzian line-spread function with full width at half max ``fwhm_L``."""

    def __init__(self, fwhm_L: float):
        self.fwhm_L = float(fwhm_L)

    def __call__(self, dl: np.ndarray) -> np.ndarray:
        gamma = self.fwhm_L / 2.0
        A = 2.0 / (np.pi * self.fwhm_L)
        return A * gamma**2 / (gamma**2 + dl**2)


def make_lsf(params: Dict[str, float], mode: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a line-spread function callable from parameter values.

    - If LSF mode == "2Gauss", the parameters are ``sigma1``, ``sigma2``, and ``ratio``.
    - If LSF mode == "Gauss", the parameter is ``sigma``.
    - If LSF mode == "Gauss_Lorentz", the parameters are ``sigma_G``, ``fwhm_L``, and ``ratio``.
    - If LSF mode == "Lorentz", the parameter is ``fwhm_L``.

    Parameters
    ----------
    params : dict[str, float]
        LSF parameter dictionary.
    mode : str
        LSF mode: ``2Gauss``, ``Gauss``, ``Gauss_Lorentz``, or ``Lorentz``.

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        LSF callable. Returned as a small picklable class instance (not a
        closure), so it can be sent to ``spawn``-based multiprocessing workers
        when used as the ``lsf`` argument to :func:`cometspec.mcmc.mcmc_fitting`.

    Raises
    ------
    ValueError
        If ``mode`` is invalid.
    """
    if mode == "2Gauss":
        return _Gauss2LSF(
            sigma1=params.get("sigma1", 0.01),
            sigma2=params.get("sigma2", 0.005),
            ratio=params.get("ratio", 0.9),
        )

    if mode == "Gauss":
        return _GaussLSF(sigma=params.get("sigma", 0.01))

    if mode == "Gauss_Lorentz":
        return _GaussLorentzLSF(
            sigma_G=params.get("sigma_G", 0.01),
            fwhm_L=params.get("fwhm_L", 0.02),
            ratio=params.get("ratio", 0.9),
        )

    if mode == "Lorentz":
        return _LorentzLSF(fwhm_L=params.get("fwhm_L", 0.02))

    raise ValueError("Invalid LSF mode.")
