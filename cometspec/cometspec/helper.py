"""Helper utilities for :mod:`cometspec`.

This module groups small, self-contained utilities used across the package and the jupyter notebook examples:
table, spectrum and ephemeris loaders, and some numerical and analytical helpers for
seeing-dependent slit-loss estimates.

Notes
-----
- Packaged data files live under ``cometspec/data/``. The ``get_*_path``
  helpers resolve those locations relative to the installed package.
- Wavelengths returned by line-list loaders are in vacuum Angstrom unless
  documented otherwise.

Routines
--------
- :func:`make_fwhm_lambda_bounds`
    Wavelength-dependent seeing bounds from min/max zenith seeing and min/max zenith angles.
- :func:`frac_in_circular_aperture_gaussian`, :func:`frac_in_rectangular_aperture_gaussian`
    Encircled-energy fraction of a 2-D Gaussian PSF in circular / rectangular apertures.
- :func:`throughput_vs_lambda`
    Wavelength-dependent flux throughput of an aperture derived from min/max zenith seeing and min/max zenith angles.
- :func:`add_slit_loss_error_scalar`
    Include the slit-loss systematic error to a scalar.
- :func:`open_table`
    Open CSV tables into an :class:`astropy.table.Table`.
- :func:`find_spectrum_file`, :func:`load_spectrum`
    Locate and load science spectra from a directory tree.
- :func:`load_ephemeris_summary`, :func:`get_ephemeris_for_night`
    Load an ephemeris summary table and select a given UT night.
- :func:`get_kurucz_irradiance_path`, :func:`open_kurucz_irradiance`
    Locate and open the `Kurucz <http://kurucz.harvard.edu/sun/irradiance2005/>`_ solar irradiance [2]_.
- :func:`get_hall_anderson_irradiance_path`, :func:`open_hall_anderson_irradiance`
    Locate and open the `Hall & Anderson <https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/SOLAR_UV/MID_UV/>`_ UV solar irradiance [3]_.
- :func:`load_cn_linelist`
    Load the packaged CN line lists for a given isotopologue [4]_ [5]_.
"""
from __future__ import annotations

import os
import io
import math
import warnings

from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u

#: Package path
PACKAGE_DIR: Path = Path(__file__).resolve().parent
#: PACKAGE_DIR / "data"
DATA_DIR: Path = PACKAGE_DIR / "data"


def make_fwhm_lambda_bounds(
    *,
    eps_min_arcsec_500: float,
    eps_max_arcsec_500: float,
    zmin_deg: float,
    zmax_deg: float,
    lambda0_nm: float = 500.0,
    alpha: float = -1 / 5,
    k: float = 0.6,
):
    """Build wavelength-dependent seeing bounds from the minimum and maximum seeing values (at zenith) and the minimum and maximum zenith angles. See Persson (2022) [1]_.

    Parameters
    ----------
    eps_min_arcsec_500 : float
        Minimum seeing at 500nm and zenith, in arcsec. It can be at any other wavelength, but lambda0_nm and alpha must be set accordingly.
    eps_max_arcsec_500 : float
        Maximum seeing at 500nm and zenith, in arcsec.
    zmin_deg : float
        Minimum zenith angle in degrees (zenith angle = 90° - elevation angle).
    zmax_deg : float
        Maximum zenith angle in degrees (zenith angle = 90° - elevation angle).
    lambda0_nm : float, optional, default 500.0
        Reference wavelength for the seeing scaling, in nm.
    alpha : float, optional, default -1 / 5
        Wavelength scaling exponent.
    k : float, optional, default 0.6
        Airmass scaling exponent.

    Returns
    -------
    tuple[callable, callable]
        A pair ``(fwhm_min, fwhm_max)`` of callables that evaluate the minimum and maximum FWHM in arcsec as a fuction of wavelength given the max and min seeing and zenith angles. Each callable takes ``x : float or ndarray of float`` and returns ``f(x) : float or ndarray of float of the same shape``.

    Raises
    ------
    ValueError
        If a zenith angle is greater than or equal to 90 degrees.

    References
    ----------
        .. [1] Persson, S. E. 2022, PASP, 134, 075001 (`link <https://iopscience.iop.org/article/10.1088/1538-3873/ac67b0>`_).
    """

    if eps_min_arcsec_500 > eps_max_arcsec_500:
        warnings.warn(
            f"eps_min_arcsec_500 ({eps_min_arcsec_500}) > eps_max_arcsec_500 "
            f"({eps_max_arcsec_500}); swapping so that '_min' corresponds to "
            "the best (smallest) seeing.",
            stacklevel=2,
        )
        eps_min_arcsec_500, eps_max_arcsec_500 = eps_max_arcsec_500, eps_min_arcsec_500
    if zmin_deg > zmax_deg:
        warnings.warn(
            f"zmin_deg ({zmin_deg}) > zmax_deg ({zmax_deg}); swapping so that "
            "'_min' corresponds to the best (smallest) zenith angle.",
            stacklevel=2,
        )
        zmin_deg, zmax_deg = zmax_deg, zmin_deg

    def secz(z_deg):
        z = np.deg2rad(z_deg)
        c = np.cos(z)
        if np.any(c <= 0):
            raise ValueError("Zenith angles must be < 90° (cos(Z) > 0).")
        return 1.0 / c

    sec_min = secz(zmin_deg)
    sec_max = secz(zmax_deg)

    def eps_zenith_from_500(lambda_nm, eps500):
        lam = np.asarray(lambda_nm, dtype=float)
        return eps500 * (lam / lambda0_nm) ** alpha

    def fwhm_min(lambda_nm):
        eps_z = eps_zenith_from_500(lambda_nm, eps_min_arcsec_500)
        return eps_z * (sec_min**k)

    def fwhm_max(lambda_nm):
        eps_z = eps_zenith_from_500(lambda_nm, eps_max_arcsec_500)
        return eps_z * (sec_max**k)

    return fwhm_min, fwhm_max


def frac_in_circular_aperture_gaussian(fwhm_arcsec, radius_arcsec):
    """Compute the fraction of a 2D Gaussian inside a circular aperture.

    Parameters
    ----------
    fwhm_arcsec : float or numpy.ndarray of float
        Gaussian full width at half maximum in arcsec.
    radius_arcsec : float
        Aperture radius in arcsec.

    Returns
    -------
    numpy.ndarray or float
        Fraction of the Gaussian flux inside the aperture.
    """
    fwhm = np.asarray(fwhm_arcsec, dtype=float)
    R = float(radius_arcsec)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return 1.0 - np.exp(-(R * R) / (2.0 * sigma * sigma))


def frac_in_rectangular_aperture_gaussian(fwhm_arcsec, *, width_arcsec, length_arcsec):
    """Compute the fraction of a 2D Gaussian inside a rectangular aperture.

    Parameters
    ----------
    fwhm_arcsec : float or numpy.ndarray of float
        Gaussian full width at half maximum in arcsec.
    width_arcsec : float
        Rectangle width in arcsec.
    length_arcsec : float
        Rectangle length in arcsec.

    Returns
    -------
    numpy.ndarray or float
        Fraction of the Gaussian flux inside the rectangle.
    """
    fwhm = np.asarray(fwhm_arcsec, dtype=float)
    w = float(width_arcsec)
    l = float(length_arcsec)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    a = w / (2.0 * np.sqrt(2.0) * sigma)
    b = l / (2.0 * np.sqrt(2.0) * sigma)
    erf = np.vectorize(math.erf)
    return erf(a) * erf(b)


def throughput_vs_lambda(
    *,
    lambda_min_nm: float,
    lambda_max_nm: float,
    eps_min_arcsec_500: float,
    eps_max_arcsec_500: float,
    zmin_deg: float,
    zmax_deg: float,
    aperture: dict,
    n_points: int = 2000,
):
    """Estimate the aperture-enclosed flux fraction and slit loss as a function of wavelength.

    Computes, over a wavelength grid, the fraction of a Gaussian PSF that falls
    within the given aperture for both the best-case (sharpest PSF) and worst-case
    (broadest PSF) observing conditions, defined by the seeing and zenith-angle ranges.

    .. note::
        The ``_min`` / ``_max`` suffix in the output keys refers to the **input**
        seeing/zenith extremes ( ``_min`` / ``_max`` **FWHM**), not to the numerical ordering of the output values.
        Because a sharper PSF concentrates more flux, ``frac_min`` (best seeing) is
        numerically *larger* than ``frac_max`` (worst seeing), and ``loss_min`` is
        numerically *smaller* than ``loss_max``.

    Parameters
    ----------
    lambda_min_nm : float
        Minimum wavelength of the evaluation grid, in nm.
    lambda_max_nm : float
        Maximum wavelength of the evaluation grid, in nm.
    eps_min_arcsec_500 : float
        Best (minimum) seeing FWHM at 500 nm and zenith, in arcsec.
    eps_max_arcsec_500 : float
        Worst (maximum) seeing FWHM at 500 nm and zenith, in arcsec.
    zmin_deg : float
        Minimum (best) zenith angle during observations, in degrees (zenith angle = 90° - elevation angle).
    zmax_deg : float
        Maximum (worst) zenith angle during observations, in degrees (zenith angle = 90° - elevation angle).
    aperture : dict
        Aperture definition. Keys:

        * ``type`` ({'circular', 'rectangular'}) -- Aperture shape.
        * ``radius_arcsec`` (float, optional) -- Radius in arcsec. Required if ``type='circular'``.
        * ``width_arcsec`` (float, optional) -- Width in arcsec. Required if ``type='rectangular'``.
        * ``length_arcsec`` (float, optional) -- Length in arcsec. Required if ``type='rectangular'``.

    n_points : int, optional, default 2000
        Number of wavelength points in the evaluation grid. Must be >= 2. Defaults to 2000.

    Returns
    -------
    dict
        Dictionary with the following keys. Each value is an ``ndarray`` of
        length ``n_points``:

        * ``lambda_nm`` -- Wavelength grid, in nm.
        * ``fwhm_min_arcsec`` -- PSF FWHM at best seeing (``eps_min``, ``zmin``)
          as a function of wavelength, in arcsec. Numerically the smallest FWHM values.
        * ``fwhm_max_arcsec`` -- PSF FWHM at worst seeing (``eps_max``, ``zmax``)
          as a function of wavelength, in arcsec. Numerically the largest FWHM values.
        * ``frac_min`` -- Enclosed flux fraction at best seeing conditions.
          Numerically the *largest* fraction (sharpest PSF → most flux within aperture).
        * ``frac_max`` -- Enclosed flux fraction at worst seeing conditions.
          Numerically the *smallest* fraction (broadest PSF → least flux within aperture).
        * ``loss_min`` -- Slit loss at best seeing, i.e. ``1 - frac_min``.
          Numerically the *smallest* loss.
        * ``loss_max`` -- Slit loss at worst seeing, i.e. ``1 - frac_max``.
          Numerically the *largest* loss.

    Raises
    ------
    ValueError
        If ``n_points`` is smaller than 2 or ``aperture['type']``
        is not ``'circular'`` or ``'rectangular'``.
    """

    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    
    if eps_min_arcsec_500 > eps_max_arcsec_500:
        warnings.warn(
            f"eps_min_arcsec_500 ({eps_min_arcsec_500}) > eps_max_arcsec_500 "
            f"({eps_max_arcsec_500}); swapping so that '_min' corresponds to "
            "the best (smallest) seeing.",
            stacklevel=2,
        )
        eps_min_arcsec_500, eps_max_arcsec_500 = eps_max_arcsec_500, eps_min_arcsec_500

    if zmin_deg > zmax_deg:
        warnings.warn(
            f"zmin_deg ({zmin_deg}) > zmax_deg ({zmax_deg}); swapping so that "
            "'_min' corresponds to the best (smallest) zenith angle.",
            stacklevel=2,
        )
        zmin_deg, zmax_deg = zmax_deg, zmin_deg
        
    lam = np.linspace(lambda_min_nm, lambda_max_nm, n_points)

    fmin_fun, fmax_fun = make_fwhm_lambda_bounds(
        eps_min_arcsec_500=eps_min_arcsec_500,
        eps_max_arcsec_500=eps_max_arcsec_500,
        zmin_deg=zmin_deg,
        zmax_deg=zmax_deg,
    )

    fwhm_min = fmin_fun(lam) #
    fwhm_max = fmax_fun(lam)

    ap_type = aperture.get("type", "").lower().strip()
    if ap_type == "circular":
        R = float(aperture["radius_arcsec"])
        frac_min = frac_in_circular_aperture_gaussian(fwhm_min, R)
        frac_max = frac_in_circular_aperture_gaussian(fwhm_max, R)
    elif ap_type == "rectangular":
        w = float(aperture["width_arcsec"])
        l = float(aperture["length_arcsec"])
        frac_min = frac_in_rectangular_aperture_gaussian(fwhm_min, width_arcsec=w, length_arcsec=l)
        frac_max = frac_in_rectangular_aperture_gaussian(fwhm_max, width_arcsec=w, length_arcsec=l)
    else:
        raise ValueError("aperture['type'] must be 'circular' or 'rectangular'")

    loss_min = 1.0 - frac_min
    loss_max = 1.0 - frac_max

    return {
        "lambda_nm": lam,
        "fwhm_min_arcsec": fwhm_min,
        "fwhm_max_arcsec": fwhm_max,
        "frac_min": frac_min,
        "frac_max": frac_max,
        "loss_min": loss_min,
        "loss_max": loss_max,
    }


def add_slit_loss_error_scalar(
    q_err: float,
    *,
    lambda_nm: float,
    aperture: dict,
    eps_min_arcsec_500: float = 0.7,
    eps_max_arcsec_500: float = 1.2,
    zmin_deg: float = 45.0,
    zmax_deg: float = 45.0,
    n_points: int = 2000,
):
    r"""Add slit-loss uncertainty to the log10 error of a log10 quantity.

    Evaluates the aperture-enclosed flux fraction over a narrow wavelength window
    centred on ``lambda_nm`` for both the best- and worst-case seeing/zenith
    conditions (see :func:`throughput_vs_lambda`). From those two flux fractions it derives a symmetric systematic
    uncertainty in log10 space via a geometric-mean scaling, then adds it in
    quadrature to the input statistical error.

    The systematic term is computed as follows:

    .. math::

        \begin{aligned}
            \sigma_\mathrm{sys} &= \tfrac{1}{2}\log_{10} \left( \frac{\bar{f}_\mathrm{max}}{\bar{f}_\mathrm{min}} \right) \\[4pt]
            \sigma_\mathrm{tot} &= \sqrt{\sigma_\mathrm{stat}^{2} + \sigma_\mathrm{sys}^{2}}
        \end{aligned}

    Where :math:`\sigma_\mathrm{stat}` is the input statistical error ``q_err``, and :math:`\bar{f}_\mathrm{min}` and :math:`\bar{f}_\mathrm{max}` are the mean flux fractions over the wavelength window for the best and worst seeing/zenith conditions, respectively.
    
    .. important::
        ``q_err`` **must be the error of a log10 quantity
        proportional to the aperture-collected flux** (e.g., ``log10(F)``,
        ``log10(N)`` for a column density inferred from line flux, or any
        derived quantity ``Q`` such that ``Q ∝ F``). The slit loss multiplies
        the true flux by a fraction ``f ∈ (0, 1]``, so on a log scale it acts
        as an additive shift ``Δlog10(Q) = log10(f)``. The systematic-error
        derivation in this function assumes exactly that additive structure;
        feeding it a linear-space value or a quantity not proportional to flux
        will produce an incorrect uncertainty.

    .. note::
        The wavelength window used for the flux-fraction estimate is
        ``[lambda_nm - 0.01, lambda_nm + 0.01]`` nm, sampled with ``n_points``
        points, and the result flux fraction is averaged over that window.

    Parameters
    ----------
    q_err : float
        One-sigma statistical uncertainty in log10 space from a quantity proportional to the aperture-collected flux (e.g., the error of log10(column density)).
    lambda_nm : float
        Central wavelength at which to evaluate the slit-loss
        systematic, in nm.
    aperture : dict
        Aperture definition. Must contain the key ``'type'`` with
        value ``'circular'`` or ``'rectangular'``. For circular apertures, also
        requires ``'radius_arcsec'`` (float). For rectangular apertures, requires
        ``'width_arcsec'`` and ``'length_arcsec'`` (both float).
    eps_min_arcsec_500 : float, optional, default 0.7
        Best (minimum) seeing FWHM at 500 nm and zenith,
        in arcsec. Defaults to 0.7.
    eps_max_arcsec_500 : float, optional, default 1.2
        Worst (maximum) seeing FWHM at 500 nm and zenith,
        in arcsec. Defaults to 1.2.
    zmin_deg : float, optional, default 45.0
        Minimum (best) zenith angle during observations, in degrees.
        Defaults to 45.0.
    zmax_deg : float, optional, default 45.0
        Maximum (worst) zenith angle during observations, in degrees.
        Defaults to 45.0.
    n_points : int, optional, default 2000
        Number of wavelength points used to sample the narrow window
        around ``lambda_nm``. Must be >= 2. Defaults to 2000.

    Returns
    -------
    float
        Total one-sigma uncertainty on ``q_log10`` in dex, equal to the
        quadrature sum of the input statistical error ``q_err`` and the symmetric
        slit-loss systematic ``sigma_sys``:

    Raises
    ------
    ValueError
        If ``n_points`` is smaller than 2 or ``aperture['type']``
        is not ``'circular'`` or ``'rectangular'``.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if eps_min_arcsec_500 > eps_max_arcsec_500:
        warnings.warn(
            f"eps_min_arcsec_500 ({eps_min_arcsec_500}) > eps_max_arcsec_500 "
            f"({eps_max_arcsec_500}); swapping so that '_min' corresponds to "
            "the best (smallest) seeing.",
            stacklevel=2,
        )
        eps_min_arcsec_500, eps_max_arcsec_500 = eps_max_arcsec_500, eps_min_arcsec_500
    if zmin_deg > zmax_deg:
        warnings.warn(
            f"zmin_deg ({zmin_deg}) > zmax_deg ({zmax_deg}); swapping so that "
            "'_min' corresponds to the best (smallest) zenith angle.",
            stacklevel=2,
        )
        zmin_deg, zmax_deg = zmax_deg, zmin_deg

    out = throughput_vs_lambda(
        lambda_min_nm=lambda_nm - 0.01,
        lambda_max_nm=lambda_nm + 0.01,
        eps_min_arcsec_500=eps_min_arcsec_500,
        eps_max_arcsec_500=eps_max_arcsec_500,
        zmin_deg=zmin_deg,
        zmax_deg=zmax_deg,
        n_points=n_points,
        aperture=aperture,
    )
    f_min = float(np.mean(out["frac_min"]))
    f_max = float(np.mean(out["frac_max"]))

    sys_sym = 0.5 * np.log10(f_max / f_min)

    err = float(np.sqrt(q_err**2 + sys_sym**2))
    return err


def open_table(
    file_path: os.PathLike | str,
    *,
    header_row: int = 0,
    units_row: Optional[int] = 1,
    data_start: int = 2,
    delimiter: str = ",",

) -> Table:
    """Read a CSV table with an optional units row.

    .. note::
        The ``units_row`` expect strings that can be parsed by ``astropy.units.Unit``. If a unit string cannot be parsed, a warning is issued and the column is left unitless. If the units row contains empty strings or placeholders like ``"-"`` or ``"None"``, those columns are also left unitless.

    Parameters
    ----------
    file_path : os.PathLike or str
        Path to the table file.
    header_row : int, optional, default 0
        Zero-based row index containing the column names.
    units_row : int, optional, default 1
        Zero-based row index containing the units row, or ``None`` to skip unit parsing.
    data_start : int, optional, default 2
        Zero-based row index where table data begin.
    delimiter : str, optional, default ","
        Delimiter used in the CSV file.

    Returns
    -------
    astropy.table.Table
        The loaded table.

    Raises
    ------
    ValueError
        If ``units_row`` is provided but is greater than or equal to ``data_start``.
    """
    if units_row is not None and units_row >= data_start:
        raise ValueError(
            f"units_row ({units_row}) must be < data_start ({data_start})"
        )
    file_path = Path(file_path)

    t = Table.read(
        file_path,
        format='ascii.csv',
        delimiter=delimiter,
        header_start=header_row,
        data_start=data_start,
    )

    if units_row is None:
        for col in t.colnames:
            t[col].unit = None
        return t

    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):
            if i == units_row:
                units_line = line.strip().split(",")
                break
        else:
            raise ValueError(f"units_row={units_row} exceeds number of rows")

    for col, unit_str in zip(t.colnames, units_line):
        unit_str = unit_str.strip()
        if unit_str and unit_str not in {"-", "None"}:
            try:
                t[col].unit = u.Unit(unit_str)
            except (ValueError, TypeError):
                warnings.warn(f"Could not parse unit {unit_str!r} for column {col!r}; "
                            "leaving unitless.", stacklevel=2)
                t[col].unit = None
        else:
            t[col].unit = None

    return t



def find_spectrum_file(
    dir_path: os.PathLike | str,
    *,
    night: str,
    fibre: str,
    suffix: str = ".csv",
) -> Optional[Path]:
    """Find the first spectrum file matching a night and fibre.

    Parameters
    ----------
    dir_path : os.PathLike or str
        Directory to search.
    night : str
        Night substring to match in the filename.
    fibre : str
        Fibre substring to match in the filename. Technicallly the function checks if both ``night`` and ``fibre`` are substrings of the filename, so they can be any distinctive part of the filename as long as they uniquely identify the file. If multiple files match, the first one when sorted by name is returned.
    suffix : str, optional, default ".csv"
        Filename suffix to accept. Defaults to ``".csv"``.

    Returns
    -------
    pathlib.Path or None
        The first matching path, or ``None`` if no file matches.
    """
    dir_path = Path(dir_path)
    fibre = str(fibre)
    for fname in sorted(dir_path.iterdir()):
        if not fname.name.endswith(suffix):
            continue
        if night in fname.name and fibre in fname.name:
            return fname
    return None


def load_spectrum(
    dir_path: os.PathLike | str,
    *,
    night: str,
    fibre: str,
    header_row: int = 0,
    units_row: int = 1,
    data_start: int = 2,
) -> Table:
    """Load a stacked spectrum for a given night and fibre.

    .. note::

        - The units_row expect strings that can be parsed by astropy.units.Unit. If a unit string cannot be parsed, a warning is issued and the column is left unitless. If the units row contains empty strings or placeholders like "-" or "None", those columns are also left unitless.
        - Technicallly the function checks if both ``night`` and ``fibre`` are substrings of the filename, so they can be any distinctive part of the filename as long as they uniquely identify the file. If multiple files match, the first one when sorted by name is returned.
    
    Parameters
    ----------
    dir_path : os.PathLike or str
        Directory containing the spectrum files.
    night : str
        Night substring used to locate the file.
    fibre : str
        Fibre substring used to locate the file.
    header_row : int, optional, default 0
        Zero-based row index containing the column names.
    units_row : int, optional, default 1
        Zero-based row index containing the units row.
    data_start : int, optional, default 2
        Zero-based row index where table data begin.

    Returns
    -------
    astropy.table.Table
        The loaded spectrum table.

    Raises
    ------
    FileNotFoundError
        If no matching spectrum file is found.
    ValueError
        If ``units_row`` is provided but is greater than or equal to ``data_start``.
    """
    path = find_spectrum_file(dir_path, night=night, fibre=fibre)
    if path is None:
        raise FileNotFoundError(
            f"No spectrum file found in {dir_path!s} for night={night!r}, fibre={fibre!r}"
        )
    return open_table(path, header_row=header_row, units_row=units_row, data_start=data_start)


def load_ephemeris_summary(
    path: os.PathLike | str,
    *,
    key_column: str = "date_obs",
    delimiter: str = ",",
) -> Dict[str, Dict[str, Any]]:
    """Read a csv file into a nested dictionary where the key of the outer dictionary is determined by `key_column` and the inner dictionary contains the row values with the column names as key.

    Parameters
    ----------
    path : os.PathLike or str
        Path to ephemeris csv file.
    key_column : str, optional, default "date_obs"
        Column used as the dictionary key.
    delimiter : str, optional, default ","
        Delimiter used in the CSV file. Defaults to ``","``.

    Returns
    -------
    dict[str, dict[str, Any]]
        A mapping from observation key to row values.

    Raises
    ------
    FileNotFoundError
        If the ephemeris summary file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ephemeris summary file not found at {path!s}.")

    table = Table.read(path, delimiter=delimiter)
    epi: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(table):
        if key_column in table.colnames:
            key = str(row[key_column])
        else:
            key = str(idx)

        record: Dict[str, Any] = {}
        for col in table.colnames:
            val = row[col]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except (AttributeError, ValueError):
                    pass
            record[col] = val
        epi[key] = record

    return epi


def get_ephemeris_for_night(
    ephemeris: Dict[str, Dict[str, Any]],
    night: str,
) -> Dict[str, Any]:
    """Return the ephemeris record for a single night given the night and the ephemeris dictionary (output of :func:`load_ephemeris_summary`).

    Parameters
    ----------
    ephemeris : dict[str, dict[str, Any]]
        Nested ephemeris mapping returned by :func:`load_ephemeris_summary`.
    night : str
        Night key to retrieve. It can be any distinctive substring of the key used in the ephemeris dictionary.

    Returns
    -------
    dict[str, Any]
        The matching record, or an empty dictionary if the night is missing.
    """
    return dict(ephemeris.get(str(night), {}))


def get_kurucz_irradiance_path() -> Path:
    """Return the path to the packaged Kurucz solar irradiance file.

    Returns
    -------
    pathlib.Path
        Path to ``kurucz_irradiance.txt``.

    Raises
    ------
    FileNotFoundError
        If the packaged file is missing.
    """
    candidate = DATA_DIR / "kurucz_irradiance.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            "Kurucz solar irradiance file not found in data/. "
            "Place the file there or provide your own path when loading pumping spectra."
        )
    return candidate


def open_kurucz_irradiance() -> pd.DataFrame:
    """Load the packaged Kurucz [2]_ solar irradiance file. See `Kurucz <http://kurucz.harvard.edu/sun/irradiance2005/>`_.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``WAVE`` in units of :math:`\AA` and ``FLUX`` in units of :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}`..

    Raises
    ------
    FileNotFoundError
        If the packaged Kurucz file cannot be found.

    References
    ----------
        .. [2] Kurucz, R. L. 2005, Memorie della Societa Astronomica Italiana Supplementi, 8, 189. (`link <https://ui.adsabs.harvard.edu/abs/2005MSAIS...8..189K/abstract>`_)
    """
    path = get_kurucz_irradiance_path()
    df = pd.read_csv(path, sep='\s+', names=['nm', 'flux'])
    wave = np.asarray(df['nm']*10, dtype=float)
    flux = np.asarray(df['flux'], dtype=float)
    flux = flux * u.W / (u.m**2 * u.nm)
    flux = flux.to(u.erg / (u.s * u.cm**2 * u.AA))
    flux = flux.value

    solar = pd.DataFrame({'WAVE': wave, 'FLUX': flux})

    return solar


def get_hall_anderson_irradiance_path() -> Path:
    """Return the path to the packaged Hall & Anderson UV solar irradiance file. See `Hall & Anderson <https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/SOLAR_UV/MID_UV/>`_.

    Returns
    -------
    pathlib.Path
        Path to ``Hall_Anderson.txt``.

    Raises
    ------
    FileNotFoundError
        If the packaged file is missing.
    """

    candidate = DATA_DIR / "Hall_Anderson.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            "Hall & Anderson solar irradiance file not found in data/. "
            "Place the file there or provide your own path when loading pumping spectra."
        )
    return candidate


def open_hall_anderson_irradiance(wave_max_AA: float = 2990.0) -> pd.DataFrame:
    """Load the packaged Hall & Anderson [3]_ UV solar irradiance file.

    The on-disk file has wavelength in Angstrom and irradiance in
    :math:`\mathrm{photons\,s^{-1}\,cm^{-2}\,\AA^{-1}}`. The output is converted to the same units
    as :func:`open_kurucz_irradiance` and truncated at ``wave_max_AA`` so the two
    spectra concatenate without overlap. See `Hall & Anderson <https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-indices/SOLAR_UV/MID_UV/>`_.

    Parameters
    ----------
    wave_max_AA : float, optional, default 2990.0
        Upper wavelength cutoff in Angstrom (inclusive). Default
        ``2990.0`` matches the Kurucz file's lower bound.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``WAVE`` in units of :math:`\AA` and ``FLUX``
        in units of :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}`.

    Raises
    ------
    FileNotFoundError
        If the packaged Hall & Anderson file cannot be found.
    
    References
    ----------
        .. [3] Hall, L. A. & Anderson, G. P. 1991, J. Geophys. Res., 96, 12,927. (`link <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/91JD01111>`_)
    """
    
    path = get_hall_anderson_irradiance_path()
    df = pd.read_csv(path, sep=r'\s+', names=['AA', 'photons'])
    wave = np.asarray(df['AA'], dtype=float)
    photon_flux = np.asarray(df['photons'], dtype=float)

    mask = wave <= float(wave_max_AA)
    wave = wave[mask]
    photon_flux = photon_flux[mask]

    photon_unit = u.photon / (u.s * u.cm**2 * u.AA)
    target_unit = u.erg / (u.s * u.cm**2 * u.AA)
    flux = (photon_flux * photon_unit).to(
        target_unit, equivalencies=u.spectral_density(wave * u.AA)
    ).value

    return pd.DataFrame({'WAVE': wave, 'FLUX': flux})


def load_cn_linelist(path_or_text: Union[str, os.PathLike]) -> pd.DataFrame:
    r"""Load a Brooke- or Sneden-style CN line list.

    Parses CN molecular line lists distributed in the fixed-width "machine-readable
    table" format used by ApJS, as produced by Brooke et al. (2014) [4]_ and
    Sneden et al. (2014) [5]_. Three isotopologues are supported, each available
    from the journal's online supplementary materials:

    - `12C14N <https://content.cld.iop.org/journals/0067-0049/210/2/23/revision1/apjs489210t4_mrt.txt>`_
      (Brooke et al. 2014) — :math:`^{12}\mathrm{C}^{14}\mathrm{N}`
    - `13C14N <https://content.cld.iop.org/journals/0067-0049/214/2/26/revision1/apjs500517t1_mrt.txt>`_
      (Sneden et al. 2014) — :math:`^{13}\mathrm{C}^{14}\mathrm{N}`
    - `12C15N <https://content.cld.iop.org/journals/0067-0049/214/2/26/revision1/apjs500517t2_mrt.txt>`_
      (Sneden et al. 2014) — :math:`^{12}\mathrm{C}^{15}\mathrm{N}`

    Parameters
    ----------
    path_or_text : str or os.PathLike
        Path to the line-list file, or the file contents as a string.

    Returns
    -------
    pandas.DataFrame
        Parsed line list. Each row corresponds to one rovibronic transition.
        The columns reproduce the fields of the source machine-readable tables,
        plus two derived wavelength columns appended at the end:

        Electronic state and vibrational quantum numbers

        * ``eS'`` -- Upper electronic state label (``A``, ``B``, or ``X``).
        * ``eS''`` -- Lower electronic state label (``A``, ``B``, or ``X``).
        * ``v'`` -- Upper vibrational quantum number :math:`v'`.
        * ``v''`` -- Lower vibrational quantum number :math:`v''`.

        Rotational quantum numbers and fine-structure / parity labels

        * ``J'`` -- Upper total angular momentum :math:`J'` (excluding nuclear spin).
        * ``J''`` -- Lower total angular momentum :math:`J''`.
        * ``F'`` -- Upper-state spin/parity component: in :math:`A^{2}\Pi`, ``1`` for :math:`\Omega = 1/2`, ``2`` for :math:`\Omega = 3/2`; in :math:`B^{2}\Sigma^{+}` and :math:`X^{2}\Sigma^{+}`, ``1`` for :math:`e`, ``2`` for :math:`f` parity.
        * ``p'`` -- Upper-state parity / e-f label (``e`` or ``f``).
        * ``p''`` -- Lower-state parity / e-f label (``e`` or ``f``).
        * ``N'``, ``N''`` -- Upper and lower :math:`N` quantum numbers as defined in Brooke et al. (2014) [4]_. The ``N'`` column is stored as text in the source file (allowing blank entries) and is coerced to numeric here.
        * ``Obs`` -- Observed transition wavenumber, in cm\ :sup:`-1`. May be  ``NaN`` for lines without a measured position (predicted-only).
        * ``Cal`` -- Calculated transition wavenumber from the term-value fit, in cm\ :sup:`-1`.
        * ``Res`` -- Residual ``Obs - Cal``, in cm\ :sup:`-1`.
        * ``E''`` -- Lower-state energy :math:`E''` relative to ``v"=0``, in cm\ :sup:`-1`. 
        * ``A`` -- Einstein :math:`A` coefficient (spontaneous emission rate), in s\ :sup:`-1`.
        * ``f`` -- Absorption oscillator strength (dimensionless).
        * ``Des`` -- Transition description from Brooke/Sneden line list [4]_ [5]_.
        * ``lambda_vac_A_from_Cal`` -- Vacuum wavelength in Å, computed as :math:`10^{8}/\mathrm{Cal}`.
        * ``lambda_vac_A_from_Obs`` -- Vacuum wavelength in Å, computed as :math:`10^{8}/\mathrm{Obs}`. ``NaN`` where ``Obs`` is missing.

    Raises
    ------
    ValueError
        If no Brooke/Sneden-style data lines are found in the input.

    References
    ----------
    .. [4] Brooke, J. S. A., Ram, R. S., Western, C. M., et al. 2014, ApJS, 210, 23 (`link <https://doi.org/10.1088/0067-0049/210/2/23>`_).
    .. [5] Sneden, C., Lucatello, S., Ram, R. S., Brooke, J. S. A., & Bernath, P. 2014, ApJS, 1050 214, 26 (`link <https://doi.org/10.1088/0067-0049/214/2/26>`_).
    """

    colspecs = [
        (0, 1),  (2, 3),  (4, 6),  (7, 9),
        (10, 15), (16, 21),
        (22, 23), (24, 25),
        (26, 27), (28, 29),
        (30, 33), (34, 37),
        (38, 49), (50, 60), (61, 69),
        (70, 80), (81, 93), (94, 106), (107, 118),
    ]
    names = [
        "eS'", "eS''", "v'", "v''", "J'", "J''",
        "F'", "F''", "p'", "p''",
        "N'", "N''",
        "Obs", "Cal", "Res",
        "E''", "A", "f", "Des",
    ]

    if isinstance(path_or_text, (os.PathLike,)):
        path_str = os.fspath(path_or_text)
        is_text = False
    elif isinstance(path_or_text, str):
        if "\n" in path_or_text or path_or_text.lstrip().startswith("Title:"):
            is_text = True
            text = path_or_text
        else:
            path_str = path_or_text
            is_text = False
    else:
        path_str = os.fspath(path_or_text)
        is_text = False

    if not is_text:
        with open(path_str, "r", encoding="utf-8") as f:
            text = f.read()

    data_lines = [
        ln for ln in text.splitlines()
        if len(ln) > 2 and ln[0] in "ABX" and ln[1].isspace() and ln[2] in "ABX"
    ]
    if not data_lines:
        raise ValueError("No Brooke-style data lines found in input.")

    df = pd.read_fwf(io.StringIO("\n".join(data_lines)),
                     colspecs=colspecs, names=names)

    num_cols = ["v'", "v''", "J'", "J''", "F'", "F''",
                "N'", "N''", "Obs", "Cal", "Res", "E''", "A", "f"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["eS'", "eS''", "p'", "p''", "Des"]:
        df[c] = df[c].astype(str).str.strip()

    wn_cal = df["Cal"].to_numpy(float)
    wn_obs = df["Obs"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_cal_A = 1e8 / wn_cal
        lam_obs_A = 1e8 / wn_obs

    df["lambda_vac_A_from_Cal"] = lam_cal_A
    df["lambda_vac_A_from_Obs"] = lam_obs_A

    return df

