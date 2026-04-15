from __future__ import annotations

"""Helper utilities for :mod:`cometspec`.

This module provides helpers to:
- load and open tables
- locate and load spectra
- load ephemeris summaries
- load pumping spectra
- get packaged molecular and atomic line lists
- work with slit-loss and seeing estimates
"""

import os
import re
import io
import math


from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from specutils.utils.wcs_utils import air_to_vac


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

# Slit-loss and seeing helpers for a Gaussian PSF.
# ---------------------------------------------------------------------
def make_fwhm_lambda_bounds(
    eps_min_arcsec_500: float,
    eps_max_arcsec_500: float,
    zmin_deg: float,
    zmax_deg: float,
    lambda0_nm: float = 500.0,
    alpha: float = -1 / 5,
    k: float = 0.6,
):
    """Build wavelength-dependent seeing bounds.

    :param eps_min_arcsec_500: Minimum seeing at 500nm and zenith, in arcsec.
    :type eps_min_arcsec_500: float
    :param eps_max_arcsec_500: Maximum seeing at 500nm and zenith, in arcsec.
    :type eps_max_arcsec_500: float
    :param zmin_deg: Minimum zenith angle in degrees.
    :type zmin_deg: float
    :param zmax_deg: Maximum zenith angle in degrees.
    :type zmax_deg: float
    :param lambda0_nm: Reference wavelength for the seeing scaling, in nm.
    :type lambda0_nm: float
    :param alpha: Wavelength scaling exponent. (see, e.g., Persson, S. E. 2022, PASP, 134, 075001,1082)
    :type alpha: float
    :param k: Airmass scaling exponent.
    :type k: float
    :returns: A pair ``(fwhm_min, fwhm_max)`` of callables that evaluate the minimum and maximum FWHM in arcsec given the max and min seeing and zenith angles.
    :rtype: tuple[callable, callable]
    :raises ValueError: If a zenith angle is greater than or equal to 90 degrees.
    """
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

    :param fwhm_arcsec: Gaussian full width at half maximum in arcsec.
    :type fwhm_arcsec: float or array-like
    :param radius_arcsec: Aperture radius in arcsec.
    :type radius_arcsec: float
    :returns: Fraction of the Gaussian flux inside the aperture.
    :rtype: numpy.ndarray or float
    """
    fwhm = np.asarray(fwhm_arcsec, dtype=float)
    R = float(radius_arcsec)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return 1.0 - np.exp(-(R * R) / (2.0 * sigma * sigma))


def frac_in_rectangular_aperture_gaussian(fwhm_arcsec, width_arcsec, length_arcsec):
    """Compute the fraction of a 2D Gaussian inside a rectangular aperture.

    :param fwhm_arcsec: Gaussian full width at half maximum in arcsec.
    :type fwhm_arcsec: float or array-like
    :param width_arcsec: Rectangle width in arcsec.
    :type width_arcsec: float
    :param length_arcsec: Rectangle length in arcsec.
    :type length_arcsec: float
    :returns: Fraction of the Gaussian flux inside the rectangle.
    :rtype: numpy.ndarray or float
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
    lambda_min_nm: float,
    lambda_max_nm: float,
    eps_min_arcsec_500: float,
    eps_max_arcsec_500: float,
    zmin_deg: float,
    zmax_deg: float,
    n_points: int,
    aperture: dict,
):
    """Estimate the aperture-enclosed flux fraction and slit loss as a function of wavelength.

    Computes, over a wavelength grid, the fraction of a Gaussian PSF that falls
    within the given aperture for both the best-case (sharpest PSF) and worst-case
    (broadest PSF) observing conditions, defined by the seeing and zenith-angle ranges.

    .. note::
        The ``_min`` / ``_max`` suffix in the output keys refers to the **input**
        seeing/zenith extremes, not to the numerical ordering of the output values.
        Because a sharper PSF concentrates more flux, ``frac_min`` (best seeing) is
        numerically *larger* than ``frac_max`` (worst seeing), and ``loss_min`` is
        numerically *smaller* than ``loss_max``.

    :param lambda_min_nm: Minimum wavelength of the evaluation grid, in nm.
    :type lambda_min_nm: float
    :param lambda_max_nm: Maximum wavelength of the evaluation grid, in nm.
    :type lambda_max_nm: float
    :param eps_min_arcsec_500: Best (minimum) seeing FWHM at 500 nm and zenith, in arcsec.
    :type eps_min_arcsec_500: float
    :param eps_max_arcsec_500: Worst (maximum) seeing FWHM at 500 nm and zenith, in arcsec.
    :type eps_max_arcsec_500: float
    :param zmin_deg: Minimum (best) zenith angle during observations, in degrees.
    :type zmin_deg: float
    :param zmax_deg: Maximum (worst) zenith angle during observations, in degrees.
    :type zmax_deg: float
    :param n_points: Number of wavelength points in the evaluation grid. Must be >= 2.
    :type n_points: int
    :param aperture: Aperture definition. Must contain the key ``'type'`` with value
        ``'circular'`` or ``'rectangular'``. For circular apertures, also requires
        ``'radius_arcsec'`` (float). For rectangular apertures, requires
        ``'width_arcsec'`` and ``'length_arcsec'`` (both float).
    :type aperture: dict

    :returns: Dictionary with the following keys, each an ``ndarray`` of length ``n_points``:

        - **lambda_nm** (*ndarray*): Wavelength grid, in nm.
        - **fwhm_min_arcsec** (*ndarray*): PSF FWHM at best seeing (``eps_min``, ``zmin``)
          as a function of wavelength, in arcsec. Numerically the smallest FWHM values.
        - **fwhm_max_arcsec** (*ndarray*): PSF FWHM at worst seeing (``eps_max``, ``zmax``)
          as a function of wavelength, in arcsec. Numerically the largest FWHM values.
        - **frac_min** (*ndarray*): Enclosed flux fraction at best seeing conditions.
          Numerically the *largest* fraction (sharpest PSF → most flux within aperture).
        - **frac_max** (*ndarray*): Enclosed flux fraction at worst seeing conditions.
          Numerically the *smallest* fraction (broadest PSF → least flux within aperture).
        - **loss_min** (*ndarray*): Slit loss at best seeing, i.e. ``1 - frac_min``.
          Numerically the *smallest* loss.
        - **loss_max** (*ndarray*): Slit loss at worst seeing, i.e. ``1 - frac_max``.
          Numerically the *largest* loss.

    :rtype: dict
    :raises ValueError: If ``n_points`` is smaller than 2 or ``aperture['type']``
        is not ``'circular'`` or ``'rectangular'``.
    """

    if n_points < 2:
        raise ValueError("n_points must be >= 2")

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
        frac_min = frac_in_rectangular_aperture_gaussian(fwhm_min, w, l)
        frac_max = frac_in_rectangular_aperture_gaussian(fwhm_max, w, l)
    else:
        raise ValueError("aperture['type'] must be 'circular' or 'rectangular'")

    # frac_min come from the smalles PSF, then is higher than frac_max 
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
    q_log10: float,
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
    """Propagate slit-loss uncertainty into a scalar log10 abundance error.

    Evaluates the aperture-enclosed flux fraction over a narrow wavelength window
    centred on ``lambda_nm`` for both the best- and worst-case seeing/zenith
    conditions. From those two flux fractions it derives a symmetric systematic
    uncertainty in log10 space via a geometric-mean scaling, then adds it in
    quadrature to the input statistical error.

    The systematic term is computed as follows:

    .. math::

        f_\\text{mid} = \\sqrt{f_\\text{best} \\cdot f_\\text{worst}}

        \\sigma_\\text{sys} = \\log_{10}\\!\\left(\\sqrt{\\frac{f_\\text{best}}{f_\\text{worst}}}\\right)
        = \\frac{1}{2}\\log_{10}\\!\\left(\\frac{f_\\text{best}}{f_\\text{worst}}\\right)

        \\sigma_\\text{total} = \\sqrt{\\sigma_\\text{stat}^2 + \\sigma_\\text{sys}^2}

    .. note::
        The wavelength window used for the flux-fraction estimate is
        ``[lambda_nm - 0.01, lambda_nm + 0.01]`` nm, sampled with ``n_points``
        points, and the result is averaged over that window.

    :param q_log10: Log10 of the production rate or abundance (e.g. log10 Q).
    :type q_log10: float
    :param q_err: One-sigma statistical uncertainty on ``q_log10``, in dex.
    :type q_err: float
    :param lambda_nm: Central wavelength at which to evaluate the slit-loss
        systematic, in nm.
    :type lambda_nm: float
    :param aperture: Aperture definition. Must contain the key ``'type'`` with
        value ``'circular'`` or ``'rectangular'``. For circular apertures, also
        requires ``'radius_arcsec'`` (float). For rectangular apertures, requires
        ``'width_arcsec'`` and ``'length_arcsec'`` (both float).
    :type aperture: dict
    :param eps_min_arcsec_500: Best (minimum) seeing FWHM at 500 nm and zenith,
        in arcsec. Defaults to 0.7.
    :type eps_min_arcsec_500: float
    :param eps_max_arcsec_500: Worst (maximum) seeing FWHM at 500 nm and zenith,
        in arcsec. Defaults to 1.2.
    :type eps_max_arcsec_500: float
    :param zmin_deg: Minimum (best) zenith angle during observations, in degrees.
        Defaults to 45.0.
    :type zmin_deg: float
    :param zmax_deg: Maximum (worst) zenith angle during observations, in degrees.
        Defaults to 45.0.
    :type zmax_deg: float
    :param n_points: Number of wavelength points used to sample the narrow window
        around ``lambda_nm``. Must be >= 2. Defaults to 2000.
    :type n_points: int

    :returns: Total one-sigma uncertainty on ``q_log10`` in dex, equal to the
        quadrature sum of the input statistical error ``q_err`` and the symmetric
        slit-loss systematic ``sigma_sys``:

        .. math::
            \\sigma_\\text{total} = \\sqrt{q_\\text{err}^2 + \\sigma_\\text{sys}^2}

    :rtype: float
    :raises ValueError: If ``n_points`` is smaller than 2 or ``aperture['type']``
        is not ``'circular'`` or ``'rectangular'``.
    """
    out = throughput_vs_lambda(
        lambda_nm - 0.01,
        lambda_nm + 0.01,
        eps_min_arcsec_500=eps_min_arcsec_500,
        eps_max_arcsec_500=eps_max_arcsec_500,
        zmin_deg=zmin_deg,
        zmax_deg=zmax_deg,
        n_points=n_points,
        aperture=aperture,
    )
    f_min = float(np.mean(out["frac_min"]))
    f_max = float(np.mean(out["frac_max"]))

    # Apply the same scaling logic used in the notebook workflow.
    s_min = np.sqrt(f_min * f_max) / f_max
    s_max = np.sqrt(f_min * f_max) / f_min

    q_low = q_log10 + np.log10(s_min)
    q_high = q_log10 + np.log10(s_max)
    sys_sym = float(np.mean([abs(q_log10 - q_low), abs(q_high - q_log10)]))

    err = float(np.sqrt(q_err**2 + sys_sym**2))
    return err

def open_table(
    file_path: os.PathLike | str,
    *,
    header_row: int = 0,
    units_row: Optional[int] = 1,
    data_start: int = 2,
    fmt: str = "ascii.csv",
) -> Table:
    """Read a CSV/ASCII table with an optional units row.

    :param file_path: Path to the table file.
    :type file_path: os.PathLike | str
    :param header_row: Zero-based row index containing the column names.
    :type header_row: int
    :param units_row: Zero-based row index containing the units row, or ``None`` to skip unit parsing.
    :type units_row: int or None
    :param data_start: Zero-based row index where table data begin.
    :type data_start: int
    :param fmt: Astropy table format string.
    :type fmt: str
    :returns: The loaded table.
    :rtype: astropy.table.Table
    :raises ValueError: If the units row is requested but is outside the file.
    """
    file_path = Path(file_path)

    t = Table.read(
        file_path,
        format=fmt,
        header_start=header_row,
        data_start=data_start,
    )

    # If no units row is provided, leave the columns unitless.
    if units_row is None:
        for col in t.colnames:
            t[col].unit = None
        return t

    # Read the units row using 0-based indexing.
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
            except Exception:
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

    :param dir_path: Directory to search.
    :type dir_path: os.PathLike | str
    :param night: Night substring to match in the filename.
    :type night: str
    :param fibre: Fibre substring to match in the filename.
    :type fibre: str
    :param suffix: Filename suffix to accept.
    :type suffix: str
    :returns: The first matching path, or ``None`` if no file matches.
    :rtype: pathlib.Path or None
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

    :param dir_path: Directory containing the spectrum files.
    :type dir_path: os.PathLike | str
    :param night: Night substring used to locate the file.
    :type night: str
    :param fibre: Fibre substring used to locate the file.
    :type fibre: str
    :param header_row: Zero-based row index containing the column names.
    :type header_row: int
    :param units_row: Zero-based row index containing the units row.
    :type units_row: int
    :param data_start: Zero-based row index where table data begin.
    :type data_start: int
    :returns: The loaded spectrum table.
    :rtype: astropy.table.Table
    :raises FileNotFoundError: If no matching spectrum file is found.
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
) -> Dict[str, Dict[str, Any]]:
    """Read the ephemeris summary table into a nested dictionary.

    :param path: Path to ``ephemeris_means_by_observation.csv``.
    :type path: os.PathLike | str
    :param key_column: Column used as the dictionary key.
    :type key_column: str
    :returns: A mapping from observation key to row values.
    :rtype: dict[str, dict[str, Any]]
    :raises FileNotFoundError: If the ephemeris summary file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ephemeris summary file not found at {path!s}.")

    table = Table.read(path, format="ascii.csv")
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
                except Exception:
                    pass
            record[col] = val
        epi[key] = record

    return epi


def get_ephemeris_for_night(
    ephemeris: Dict[str, Dict[str, Any]],
    night: str,
) -> Dict[str, Any]:
    """Return the ephemeris record for a single night.

    :param ephemeris: Nested ephemeris mapping returned by :func:`load_ephemeris_summary`.
    :type ephemeris: dict[str, dict[str, Any]]
    :param night: Night key to retrieve.
    :type night: str
    :returns: The matching record, or an empty dictionary if the night is missing.
    :rtype: dict[str, Any]
    """
    return dict(ephemeris.get(str(night), {}))


def load_pumping_file(
    night: str,
    *,
    directory: os.PathLike | str | None = None,
    pattern: str = "pumping_{night}.txt",
    wavelength_col: str = "WAVE",
    flux_col: str = "FLUX",
    scale_by_r_au: Optional[float] = None,
) -> pd.DataFrame:
    """Load the incident on the comet spectrum for a given night (for instance solar spectrum doppler shifted).

    :param night: Night identifier used to format the file name.
    :type night: str
    :param directory: Directory containing the pumping file, or ``None`` to use the packaged data directory.
    :type directory: os.PathLike | str | None
    :param pattern: Filename pattern with a ``{night}`` placeholder.
    :type pattern: str
    :param wavelength_col: Name of the wavelength column.
    :type wavelength_col: str
    :param flux_col: Name of the flux column.
    :type flux_col: str
    :param scale_by_r_au: If provided, scale the flux by ``1 / r_au^2``.
    :type scale_by_r_au: float or None
    :returns: The pumping spectrum as a DataFrame.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the pumping file does not exist.
    :raises ValueError: If the expected columns are missing.
    """
    if directory is None:
        directory = DATA_DIR
    directory = Path(directory)

    fname = pattern.format(night=night)
    path = directory / fname
    if not path.exists():
        raise FileNotFoundError(f"Pumping file not found: {path!s}")

    df = pd.read_csv(path, sep=None, engine="python")

    if wavelength_col not in df.columns or flux_col not in df.columns:
        raise ValueError(
            f"Pumping file must contain columns "
            f"'{wavelength_col}' and '{flux_col}'."
        )

    if scale_by_r_au is not None:
        df[flux_col] = df[flux_col] * (1.0 / float(scale_by_r_au)) ** 2

    return df

# Helper for the packaged Kurucz irradiance file.
def get_kurucz_irradiance_path() -> Path:
    """Return the path to the packaged Kurucz solar irradiance file.

    :returns: Path to ``kurucz_irradiance.txt``.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the packaged file is missing.
    """
    candidate = DATA_DIR / "kurucz_irradiance.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            "Kurucz solar irradiance file not found in data/. "
            "Place the file there or provide your own path when loading pumping spectra."
        )
    return candidate

def open_kurucz_irradiance() -> pd.DataFrame:
    """Load the packaged Kurucz solar irradiance file.

    :returns: A DataFrame with columns ``WAVE`` and ``FLUX``.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the packaged Kurucz file cannot be found.
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

Mol = Literal["CN", "C2", "C3", "NH", "CH"]
CNIso = Literal["12C14N", "13C14N", "12C15N"]


def get_default_mol_linelist_path(
    mol: Mol = "CN",
    *,
    isotope: CNIso = "12C14N",
) -> Path:
    """Return the packaged default molecular line-list path.

    :param mol: Molecule name.
    :type mol: Literal["CN", "C2", "C3", "NH", "CH"]
    :param isotope: CN isotope label used when ``mol`` is ``CN``.
    :type isotope: Literal["12C14N", "13C14N", "12C15N"]
    :returns: Path to the packaged molecular line list.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the expected packaged line list is missing.
    :raises ValueError: If ``mol`` is not supported.
    """
    mol = mol.upper()

    if mol == "CN":
        candidate = DATA_DIR / "CN" / f"{isotope}.txt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Default CN line list not found: {candidate!s}. "
                "Place the file in data/CN/ or pass an explicit path."
            )
        return candidate

    if mol in {"C2", "C3", "NH", "CH"}:
        candidate = DATA_DIR / "neowise_lines.txt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Default NEOWISE molecular line list not found: {candidate!s}. "
                "Place neowise_lines.txt in data/ or pass an explicit path."
            )
        return candidate

    raise ValueError(f"Unknown mol={mol!r}. Expected one of: CN, C2, C3, NH, CH.")


# Atomic line list helpers.

# Minimal name-to-symbol map from hydrogen through yttrium.
# Accepts either a full name ("hydrogen") or a symbol ("H").
_NAME_TO_SYMBOL = {
    "hydrogen": "H", "helium": "He",
    "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C", "nitrogen": "N", "oxygen": "O",
    "fluorine": "F", "neon": "Ne",
    "sodium": "Na", "magnesium": "Mg", "aluminium": "Al", "aluminum": "Al", "silicon": "Si",
    "phosphorus": "P", "sulfur": "S", "sulphur": "S", "chlorine": "Cl", "argon": "Ar",
    "potassium": "K", "calcium": "Ca", "scandium": "Sc", "titanium": "Ti", "vanadium": "V",
    "chromium": "Cr", "manganese": "Mn", "iron": "Fe", "cobalt": "Co", "nickel": "Ni",
    "copper": "Cu", "zinc": "Zn", "gallium": "Ga", "germanium": "Ge", "arsenic": "As",
    "selenium": "Se", "bromine": "Br", "krypton": "Kr",
    "rubidium": "Rb", "strontium": "Sr", "yttrium": "Y", "itrium": "Y",
}

_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


def _normalize_element(element: str) -> str:
    s = element.strip()
    if _SYMBOL_RE.match(s):
        return s  # Already a symbol such as "H", "Fe", or "Y".
    key = s.lower()
    if key in _NAME_TO_SYMBOL:
        return _NAME_TO_SYMBOL[key]
    raise ValueError(
        f"Unknown element={element!r}. Provide a symbol (e.g., 'Fe') or a name from hydrogen..yttrium."
    )


def get_default_atomic_linelist_path(element: str) -> Path:
    """Return the packaged default atomic line-list path.

    :param element: Element name or symbol.
    :type element: str
    :returns: Path to the packaged atomic line list.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If the expected packaged line list is missing.
    :raises ValueError: If the element cannot be normalized to a supported symbol.
    """
    symbol = _normalize_element(element)
    candidate = DATA_DIR / "Element_lines" / f"{symbol}.txt"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Default atomic line list not found: {candidate!s}. "
            "Place the file in data/Element_lines/ or pass an explicit path."
        )
    return candidate


def print_available_linelist_inventory() -> None:
    """Print the packaged line-list inventory.

    :returns: ``None``. The inventory is written to standard output.
    :rtype: None
    """
    data_dir = Path(DATA_DIR)

    # Fluorescence defaults currently include CN isotopologues only.
    cn_dir = data_dir / "CN"
    cn_files = sorted(cn_dir.glob("*.txt")) if cn_dir.exists() else []

    print("Default available for fluorescence:")
    if cn_files:
        for p in cn_files:
            print(f"  - CN {p.stem}")
    else:
        print("  (none found)")

    print("You can also provide a custom line list for fluorescence modeling.")

    # Line plotting includes all available molecular and atomic lists.
    print("\nAvailable for line plotting:")

    if cn_files:
        for p in cn_files:
            print(f"  - [mol] CN {p.stem}")

    neowise = data_dir / "neowise_lines.txt"
    if neowise.exists():
        print("  - [mol] NEOWISE (C2, C3, NH, CH)")

    elem_dir = data_dir / "Element_lines"
    elem_files = sorted(elem_dir.glob("*.txt")) if elem_dir.exists() else []
    if elem_files:
        for p in elem_files:
            print(f"  - [atom] {p.stem}")

    if not (cn_files or neowise.exists() or elem_files):
        print("  (none found)")

def load_cn_linelist(path_or_text: Union[str, os.PathLike]) -> pd.DataFrame:
    """Load a Brooke/PGOPHER-style CN line list.

    :param path_or_text: Path to the file or the file contents themselves.
    :type path_or_text: str or os.PathLike
    :returns: Parsed CN line list as a DataFrame.
    :rtype: pandas.DataFrame
    :raises ValueError: If no Brooke-style data lines are found.
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

    # Normalize the input so the code can handle both paths and raw text.
    if isinstance(path_or_text, (os.PathLike,)):
        # Path-like input.
        path_str = os.fspath(path_or_text)
        is_text = False
    elif isinstance(path_or_text, str):
        # Strings can be either file paths or literal file contents.
        # If the string looks like multiline content or a Brooke header,
        # treat it as text; otherwise assume it is a path.
        if "\n" in path_or_text or path_or_text.lstrip().startswith("Title:"):
            is_text = True
            text = path_or_text
        else:
            path_str = path_or_text
            is_text = False
    else:
        # Fallback: interpret anything else as path-like.
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

_NEOWISE_MOL_MAP = {
    # Map user-facing molecule names to the token stored in the file.
    "CN": "CN",
    "CH": "CH",
    "NH2": "NH2",
    "C3": "C_3_",
    "C2": "C_2_",
}


def _airA_to_vacA(wave_air_A: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Convert wavelength(s) from air Å to vacuum Å."""
    return air_to_vac(np.asarray(wave_air_A) * u.AA).to(u.AA).value


def load_neowise_linelist(path: Union[str, os.PathLike]) -> pd.DataFrame:
    """Load a NEOWISE molecular line list.

    :param path: Path to the NEOWISE line-list file.
    :type path: str or os.PathLike
    :returns: Parsed NEOWISE table with columns ``Wave``, ``REL_Intensity``, ``MOL``, and ``identifier``.
    :rtype: pandas.DataFrame
    """
    df = {"Wave": [], "REL_Intensity": [], "MOL": [], "identifier": []}
    path = os.fspath(path)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = [x for x in raw.strip().split(" ") if x != ""]
            if len(parts) < 3:
                continue
            wave = float(parts[0])
            rel = float(parts[1])
            mol = parts[2]
            ident = " ".join(parts[3:]) if len(parts) > 3 else ""
            df["Wave"].append(wave)
            df["REL_Intensity"].append(rel)
            df["MOL"].append(mol)
            df["identifier"].append(ident)

    return pd.DataFrame(df)


def get_linelist_wavelengths_vacuum(
    species: str,
    *,
    # CN selection options.
    isotope: CNIso = "12C14N",
    upper_state: Optional[str] = "B",
    lower_state: Optional[str] = "X",
    v_upper: Optional[int] = 0,
    v_lower: Optional[int] = 0,
    # General selection options.
    n_lines: Optional[int] = None,
    path: Optional[Union[str, os.PathLike]] = None,
) -> list[float]:
    """Return line-list wavelengths converted to vacuum Angstrom.

    :param species: Species identifier. Supported values include ``CN``, ``C2``, ``C3``, ``NH``, ``NH2``, and ``CH``.
    :type species: str
    :param isotope: CN isotope label used when ``species`` is ``CN``.
    :type isotope: Literal["12C14N", "13C14N", "12C15N"]
    :param upper_state: Upper electronic state filter for CN.
    :type upper_state: str or None
    :param lower_state: Lower electronic state filter for CN.
    :type lower_state: str or None
    :param v_upper: Upper vibrational level filter for CN.
    :type v_upper: int or None
    :param v_lower: Lower vibrational level filter for CN.
    :type v_lower: int or None
    :param n_lines: Optional number of lines to keep after sorting.
    :type n_lines: int or None
    :param path: Optional explicit path to the line-list file.
    :type path: str or os.PathLike or None
    :returns: Vacuum wavelengths in Angstrom.
    :rtype: list[float]
    :raises ValueError: If the selection is empty or the species is unsupported.
    """

    sp = species.strip()
    mol_up = sp.upper()

    # CN branch uses Brooke-style line lists.
    if mol_up == "CN":
        if path is None:
            path = get_default_mol_linelist_path("CN", isotope=isotope)

        df = load_cn_linelist(path)

        # Apply the optional CN filters.
        if upper_state is not None:
            df = df[df["eS'"] == upper_state]
        if lower_state is not None:
            df = df[df["eS''"] == lower_state]
        if v_upper is not None:
            df = df[df["v'"] == v_upper]
        if v_lower is not None:
            df = df[df["v''"] == v_lower]

        if df.empty:
            raise ValueError(
                "CN selection produced no lines. "
                "Check electronic states or vibrational numbers."
            )

        # Sort by Einstein A coefficient.
        df = df.sort_values(by="A", ascending=False)

        if n_lines is not None:
            df = df.head(int(n_lines))

        waves_vac = df["lambda_vac_A_from_Cal"].astype(float).to_list()

        print(
            "CN lines: sorted by Einstein A coefficient (descending). "
            f"Filters: eS'={upper_state}, eS''={lower_state}, "
            f"v'={v_upper}, v''={v_lower}"
        )

        return waves_vac

    # The NEOWISE file contains multiple molecules, so filter to the requested one.
    if mol_up in {"C2", "C3", "NH", "CH", "NH2"}:
        if path is None:
            # Any supported NEOWISE molecule resolves to the shared file.
            path = get_default_mol_linelist_path("C2")

        neowise = load_neowise_linelist(path)

        token = _NEOWISE_MOL_MAP.get(mol_up)
        if token is None:
            raise ValueError(f"Unsupported molecule {mol_up!r}. Try CN, CH, NH2, C2, C3.")

        sub = neowise[neowise["MOL"] == token].copy()  # Keep only the requested species.
        if sub.empty:
            raise ValueError(f"No lines found for {mol_up} (token={token!r}) in {path!s}.")

        sub = sub.sort_values(by="REL_Intensity", ascending=False)
        if n_lines is not None:
            sub = sub.head(int(n_lines))

        waves_vac = _airA_to_vacA(sub["Wave"].astype(float).to_numpy()).tolist()
        print("NEOWISE lines: sorted by REL_Intensity in the NEOWISE comet (descending).")

        return waves_vac

    # Atomic files are already per element, so no extra filtering is needed.
    if path is None:
        path = get_default_atomic_linelist_path(sp)

    atom = pd.read_csv(os.fspath(path), sep='\s+', names=["WAVE", "name", "ion"])
    waves_vac = _airA_to_vacA(atom["WAVE"].astype(float).to_numpy()).tolist()

    print("Atomic lines: lines from NIST (not completed lists, just for quick plotting).")
    return waves_vac