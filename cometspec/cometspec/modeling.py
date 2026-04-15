from __future__ import annotations

"""
Core fluorescence modeling.

This version supports:
- user-provided transition linelists (generic)
- CN Brooke linelist normalization (12C14N / 12C15N / 13C14N etc.)
- selecting isotopologues (str or list[str])
- selecting CN systems to keep (B–X(0,0), A–X(Δv=+1), both, or ALL)
- multi-isotopologue fitting (sum of spectra; separate logN per iso)
- OPTIONAL rotational collisions (Option A): use explicit lower-state columns
  rather than parsing Brooke-style IDs.

Important notes
---------------
A) IDs:
   - For user linelists: upper_id/lower_id can be ANY strings (no Brooke format).
   - For Brooke linelists: IDs follow Brooke-derived format but we do not parse them.

B) Collisions (rotations):
   - If include_rotations=False -> collisions are a no-op (empty scaffold).
   - If include_rotations=True  -> you must provide lower-state properties:
        lower_es, lower_v, lower_J, lower_sym, E_lower_cm1
     OR, for Brooke lists, set include_rotations=True and provide those columns
    by mapping from the Brooke file (not included by default).

C) ΔE meaning:
   - Photon energy: E_cm1 = 1/lambda(cm)  (transition wavenumber)
   - Rotational collision energy gaps use LEVEL energies:
        dE_cm1 = |E_lower_cm1(level_u) - E_lower_cm1(level_l)|

"""

from typing import Dict, Tuple, Sequence, Optional, Callable, Any, Union, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import warnings


from astropy import constants as const
from astropy import units as u
from astropy.table import Table

from . import helper

HC_OVER_K_B_KCM = (const.h * const.c / const.k_B).to_value(u.K * u.cm)
H_CGS = const.h.cgs.value  # erg*s

# =============================================================================
# Small utilities
# =============================================================================

def _as_list(x: str | Sequence[str] | None) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _as_array(obj: Any, name: str) -> np.ndarray:
    """Return a named column as a NumPy array.

    :param obj: Table-like container with named columns.
    :type obj: Any
    :param name: Column name.
    :type name: str
    :returns: Column values as a NumPy array.
    :rtype: numpy.ndarray
    """
    if hasattr(obj, "colnames"):        # astropy Table
        return np.asarray(obj[name])
    if hasattr(obj, "columns"):         # pandas DataFrame
        return np.asarray(obj[name].values)
    return np.asarray(obj[name])


def normalize_systems_arg(systems: str | Sequence[str] | None) -> list[str]:
    """Normalize CN system selectors to internal tokens.

    :param systems: User system selector or selectors.
    :type systems: str or Sequence[str] or None
    :returns: Normalized token list.
    :rtype: list[str]
    """
    if systems is None:
        return ["BX00", "AX_dv1"]

    if isinstance(systems, str):
        s = systems.strip().lower()
        if s in ("both", "bx+ax", "bxax"):
            return ["BX00", "AX_dv1", 'AX_dv2', 'AX_dv3']
        if s in ("all",):
            return ["ALL"]
        if s in ("bx", "b-x", "bx(0,0)", "bx00", "bx_00", "b_x_00"):
            return ["BX00"]
        if s in ("ax", "a-x"):
            return ["AX_dv1", 'AX_dv2']
        if s in ("ax(dv=1)", "ax_dv1"):
            return ["AX_dv1"]
        if s in ("ax(dv=2)", "ax_dv2"):
            return ["AX_dv2"]
        if s in ("ax(dv=3)", "ax_dv3"):
            return ["AX_dv3"]
        return [systems]
    else:
        out: list[str] = []
        for item in systems:
            out.extend(normalize_systems_arg(item))
        seen = set()
        out2 = []
        for t in out:
            if t not in seen:
                seen.add(t)
                out2.append(t)
        return out2


# =============================================================================
# Normalization: user linelist -> internal schema
# =============================================================================

def from_user_linelist(
    df: pd.DataFrame,
    *,
    lam_col: str,
    A_col: str,
    upper_id_col: str,
    lower_id_col: str,
    g_upper_col: str,
    g_lower_col: str,

    # Optional: provide these for rotational collisions (Option A)
    lower_es_col: str | None = None,     # e.g. "lower_es" values like "X"
    lower_v_col: str | None = None,      # e.g. "vpp"
    lower_J_col: str | None = None,      # e.g. "Jpp"
    lower_sym_col: str | None = None,    # e.g. parity/sym label
    E_lower_cm1_col: str | None = None,  # e.g. "Epp_cm1" (must be cm^-1)
) -> pd.DataFrame:
    """Convert a user line list into the normalized transition schema.

    :param df: Input line list table.
    :type df: pandas.DataFrame
    :param lam_col: Wavelength column in vacuum Angstrom.
    :type lam_col: str
    :param A_col: Einstein A coefficient column in s^-1.
    :type A_col: str
    :param upper_id_col: Upper-state identifier column.
    :type upper_id_col: str
    :param lower_id_col: Lower-state identifier column.
    :type lower_id_col: str
    :param g_upper_col: Upper-state degeneracy column.
    :type g_upper_col: str
    :param g_lower_col: Lower-state degeneracy column.
    :type g_lower_col: str
    :param lower_es_col: Optional lower electronic-state column.
    :type lower_es_col: str or None
    :param lower_v_col: Optional lower vibrational-level column.
    :type lower_v_col: str or None
    :param lower_J_col: Optional lower rotational-level column.
    :type lower_J_col: str or None
    :param lower_sym_col: Optional lower-state symmetry/parity column.
    :type lower_sym_col: str or None
    :param E_lower_cm1_col: Optional lower-state energy column in cm^-1.
    :type E_lower_cm1_col: str or None
    :returns: Normalized transition table.
    :rtype: pandas.DataFrame
    :raises ValueError: If required columns are missing or values are invalid.
    """
    required = [lam_col, A_col, upper_id_col, lower_id_col, g_upper_col, g_lower_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)
    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)
    out["upper_id"] = df[upper_id_col].astype(str)
    out["lower_id"] = df[lower_id_col].astype(str)
    out["g_upper"] = pd.to_numeric(df[g_upper_col], errors="coerce").astype(float)
    out["g_lower"] = pd.to_numeric(df[g_lower_col], errors="coerce").astype(float)

    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8
    if np.any(~np.isfinite(lam_cm)) or np.any(lam_cm <= 0.0):
        raise ValueError("Invalid lambda values (must be finite, >0).")
    out["E_cm1"] = 1.0 / lam_cm

    if np.any(~np.isfinite(out["A_ul"])) or np.any(out["A_ul"] < 0.0):
        raise ValueError("Invalid A_ul values.")
    if np.any(~np.isfinite(out["g_upper"])) or np.any(out["g_upper"] <= 0.0):
        raise ValueError("Invalid g_upper values.")
    if np.any(~np.isfinite(out["g_lower"])) or np.any(out["g_lower"] <= 0.0):
        raise ValueError("Invalid g_lower values.")

    # Optional columns for collisions
    if lower_es_col is not None:
        if lower_es_col not in df.columns:
            raise ValueError(f"lower_es_col={lower_es_col!r} not found.")
        out["lower_es"] = df[lower_es_col].astype(str).str.strip().str.upper()

    if lower_v_col is not None:
        if lower_v_col not in df.columns:
            raise ValueError(f"lower_v_col={lower_v_col!r} not found.")
        out["lower_v"] = pd.to_numeric(df[lower_v_col], errors="coerce").astype(float)

    if lower_J_col is not None:
        if lower_J_col not in df.columns:
            raise ValueError(f"lower_J_col={lower_J_col!r} not found.")
        out["lower_J"] = pd.to_numeric(df[lower_J_col], errors="coerce").astype(float)

    if lower_sym_col is not None:
        if lower_sym_col not in df.columns:
            raise ValueError(f"lower_sym_col={lower_sym_col!r} not found.")
        out["lower_sym"] = df[lower_sym_col].astype(str).str.strip()

    if E_lower_cm1_col is not None:
        if E_lower_cm1_col not in df.columns:
            raise ValueError(f"E_lower_cm1_col={E_lower_cm1_col!r} not found.")
        out["E_lower_cm1"] = pd.to_numeric(df[E_lower_cm1_col], errors="coerce").astype(float)

    return out


# =============================================================================
# CN Brooke -> internal schema
# =============================================================================

def make_sym(F, p, use_omega: bool = False, es: Optional[str] = None) -> str:
    """Build a compact CN-style symmetry label.

    :param F: Spin component or branch label.
    :type F: Any
    :param p: Parity label.
    :type p: Any
    :param use_omega: Whether to emit Omega-style labels for A states.
    :type use_omega: bool
    :param es: Electronic-state label.
    :type es: str or None
    :returns: Compact symmetry token.
    :rtype: str
    """
    ptag = str(p).strip().lower()[:1] if p not in (None, "") else "?"
    try:
        Fint = int(F)
    except Exception:
        Fint = F

    if use_omega and str(es).strip().upper().startswith("A"):
        comp = "Ω3/2" if Fint == 1 else "Ω1/2"
        return f"{comp}_{ptag}"

    return f"F{Fint}_{ptag}"


def from_cn_brooke(
    df: pd.DataFrame,
    *,
    lam_col: str = "lambda_vac_A_from_Cal",
    A_col: str = "A",
    use_omega_labels: bool = False,
    # Brooke lower-level energy column (cm^-1)
    E_lower_col: str = "E''",
) -> pd.DataFrame:
    """Convert a Brooke CN line list to the normalized schema.

    :param df: Brooke-format CN line list.
    :type df: pandas.DataFrame
    :param lam_col: Wavelength column in vacuum Angstrom.
    :type lam_col: str
    :param A_col: Einstein A coefficient column.
    :type A_col: str
    :param use_omega_labels: Use Omega labels for A-state symmetry tags.
    :type use_omega_labels: bool
    :param E_lower_col: Lower-state energy column in cm^-1.
    :type E_lower_col: str
    :returns: Normalized transition table.
    :rtype: pandas.DataFrame
    :raises ValueError: If required columns are missing or contain invalid values.
    """
    out = pd.DataFrame(index=df.index)

    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)

    # Build symmetry labels using the existing CN mapping logic.
    sym_u = [
        make_sym(F, p, use_omega_labels, es)
        for F, p, es in zip(df["F'"], df["p'"], df["eS'"])
    ]
    sym_l = [
        make_sym(F, p, use_omega_labels, es)
        for F, p, es in zip(df["F''"], df["p''"], df["eS''"])
    ]

    J_u = pd.to_numeric(df["J'"], errors="coerce").astype(float)
    J_l = pd.to_numeric(df["J''"], errors="coerce").astype(float)

    # IDs can remain Brooke-style because collisions use explicit lower-state columns.
    out["upper_id"] = [
        f"{str(es).strip().upper()}|v={int(round(v))}|J={J:.6g}|sym={s}"
        for es, v, J, s in zip(df["eS'"], df["v'"], J_u, sym_u)
    ]
    out["lower_id"] = [
        f"{'X' if str(es).strip().upper().startswith('X') else str(es).strip().upper()}|"
        f"v={int(round(v))}|J={J:.6g}|sym={s}"
        for es, v, J, s in zip(df["eS''"], df["v''"], J_l, sym_l)
    ]

    out["g_upper"] = 2.0 * J_u + 1.0
    out["g_lower"] = 2.0 * J_l + 1.0

    # Photon wavenumber in cm^-1.
    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8  # Å -> cm
    if np.any(~np.isfinite(lam_cm)) or np.any(lam_cm <= 0.0):
        raise ValueError("Invalid lambda_vac_A values in CN linelist.")
    out["E_cm1"] = 1.0 / lam_cm

    # Required fields when rotational collisions are enabled.
    out["lower_es"] = df["eS''"].astype(str).str.strip().str.upper()
    out["lower_v"] = pd.to_numeric(df["v''"], errors="coerce").astype(float)
    out["lower_J"] = J_l
    out["lower_sym"] = np.asarray(sym_l, dtype=str)

    if E_lower_col not in df.columns:
        raise ValueError(
            f"Brooke dataframe is missing {E_lower_col!r}. "
            "Needed to build E_lower_cm1 for rotational collisions."
        )
    out["E_lower_cm1"] = pd.to_numeric(df[E_lower_col], errors="coerce").astype(float)

    # Validate normalized line values.
    if np.any(~np.isfinite(out["A_ul"])) or np.any(out["A_ul"] < 0.0):
        raise ValueError("Invalid A_ul values in CN linelist.")
    if np.any(~np.isfinite(out["g_upper"])) or np.any(out["g_upper"] <= 0.0):
        raise ValueError("Invalid g_upper values in CN linelist.")
    if np.any(~np.isfinite(out["g_lower"])) or np.any(out["g_lower"] <= 0.0):
        raise ValueError("Invalid g_lower values in CN linelist.")

    # Collision terms require finite lower-state energies.
    bad_E = ~np.isfinite(out["E_lower_cm1"])
    if np.any(bad_E):
        # Keep this strict to fail early before building collision terms.
        raise ValueError("Invalid (non-finite) E_lower_cm1 values from Brooke E'' column.")

    return out



def filter_cn_systems(
    df_all: pd.DataFrame,
    *,
    systems: str | Sequence[str] | None = None,
    lambda_min_A: float = 2990.001,
    lambda_max_A: float = 10009.998,
    A_min: float | None = 1e4,
    lam_col: str = "lambda_vac_A_from_Cal",
) -> pd.DataFrame:
    """Filter a Brooke CN line list by system, wavelength, and A (Einstein coefficient) threshold.

    :param df_all: Full Brooke CN table.
    :type df_all: pandas.DataFrame
    :param systems: System selector(s) accepted by :func:`normalize_systems_arg`.
    :type systems: str or Sequence[str] or None
    :param lambda_min_A: Minimum wavelength in Angstrom.
    :type lambda_min_A: float
    :param lambda_max_A: Maximum wavelength in Angstrom.
    :type lambda_max_A: float
    :param A_min: Minimum Einstein A threshold, or ``None`` to disable.
    :type A_min: float or None
    :param lam_col: Wavelength column name.
    :type lam_col: str
    :returns: Filtered CN line list.
    :rtype: pandas.DataFrame
    """
    df = df_all.copy()
    tokens = normalize_systems_arg(systems)

    if "ALL" not in tokens:
        df = df[df["eS''"].astype(str).str.upper().str.startswith("X")]
        masks = []
        if "BX00" in tokens:
            masks.append((df["eS'"] == "B") & (df["v'"] == 0) & (df["v''"] == 0))
        if "AX_dv1" in tokens:
            masks.append((df["eS'"] == "A") & (np.abs(df["v'"] - df["v''"]) == 1))
        if "AX_dv2" in tokens:
            masks.append((df["eS'"] == "A") & (np.abs(df["v'"] - df["v''"]) == 2))
        if not masks:
            return df.iloc[0:0].reset_index(drop=True)

        m = masks[0]
        for mm in masks[1:]:
            m = m | mm
        df = df[m]

    df = df[(df[lam_col] >= lambda_min_A) & (df[lam_col] <= lambda_max_A)]
    if A_min is not None:
        df = df[df["A"] >= float(A_min)]
    return df.reset_index(drop=True)


def load_default_cn_transitions(
    *,
    isotopologues: str | Sequence[str] = "12C14N",
    systems: str | Sequence[str] | None = None,
    A_min: float = 1e4,
    lambda_min_A: float = 2990.001,
    lambda_max_A: float = 10009.998,
    use_omega_labels: bool = False,
    line_paths: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and normalize packaged CN transitions per isotopologue.
    Default system is BX(0,0) and AX(Δv=+1), but this can be changed with the ``systems`` argument.
    The options for ``systems`` is list containing one or more of the following str:
    - "both" or "bx+ax": BX(0,0), AX(Δv=±1), AX(Δv=±2) and AX(Δv=±3)
    - "all": all systems in the Brooke linelist (including minor ones, this will lead to extremly high computation times)
    - "bx", "b-x", "bx(0,0)", "bx00", "bx_00", "b_x_00" or "b-x": BX(0,0) only
    - "ax" or "a-x": for "AX_dv1", 'AX_dv2'
    - "ax(dv=1)", "ax_dv1": AX(Δv=±1) only
    - "ax(dv=2)", "ax_dv2": AX(Δv=±2) only
    - "ax(dv=3)", "ax_dv3": AX(Δv=±3) only
 
    :param isotopologues: One or more isotopologue labels.
    :type isotopologues: str or Sequence[str]
    :param systems: CN system selector(s).
    :type systems: str or Sequence[str] or None
    :param A_min: Minimum Einstein A threshold.
    :type A_min: float
    :param lambda_min_A: Minimum wavelength in Angstrom.
    :type lambda_min_A: float
    :param lambda_max_A: Maximum wavelength in Angstrom.
    :type lambda_max_A: float
    :param use_omega_labels: Use Omega labels for A-state symmetry tags.
    :type use_omega_labels: bool
    :param line_paths: Optional mapping of isotopologue to explicit file path.
    :type line_paths: dict[str, str] or None
    :returns: Mapping from isotopologue name to normalized transition table.
    :rtype: dict[str, pandas.DataFrame]
    """
    iso_list = _as_list(isotopologues)
    out: dict[str, pd.DataFrame] = {}
    sys_tokens = normalize_systems_arg(systems)
    for iso in iso_list:
        if line_paths is not None and iso in line_paths:
            path = line_paths[iso]
        else:
            try:
                path = str(helper.get_default_mol_linelist_path(isotope=iso))
            except TypeError:
                path = str(helper.get_default_mol_linelist_path())

        df_all = helper.load_cn_linelist(path)
        df_filt = filter_cn_systems(
            df_all,
            systems=sys_tokens,
            lambda_min_A=lambda_min_A,
            lambda_max_A=lambda_max_A,
            A_min=A_min,
            lam_col="lambda_vac_A_from_Cal",
        )
        out[iso] = from_cn_brooke(
            df_filt,
            lam_col="lambda_vac_A_from_Cal",
            A_col="A",
            use_omega_labels=use_omega_labels,
        )
    return out


# =============================================================================
# Pumping: compute J_nu for each transition wavelength
# =============================================================================

def attach_pumping_and_labels(
    df: pd.DataFrame,
    pumping: Any,
    *,
    line_v_kms: float = 0.0,
    line_dlam_A: float = 0.0,
    lsf_for_Jnu: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lam_col: str = "lambda_vac_A",
) -> Table:
    """Attach the solar flux incident in the comet for a given wavelength to a transition table.

    :param df: Normalized transition DataFrame.
    :type df: pandas.DataFrame
    :param pumping: Pumping spectrum with ``WAVE`` and ``FLUX`` columns.
    :type pumping: Any
    :param line_v_kms: Doppler velocity shift applied to line wavelengths, in km/s.
    :type line_v_kms: float
    :param line_dlam_A: Additive wavelength shift in Angstrom.
    :type line_dlam_A: float
    :param lsf_for_Jnu: Optional kernel used to average flux around each line.
    :type lsf_for_Jnu: Callable[[numpy.ndarray], numpy.ndarray] or None
    :param lam_col: Input wavelength column name in ``df``.
    :type lam_col: str
    :returns: Astropy table with wavelength, frequency, flux-at-line, and J_nu columns.
    :rtype: astropy.table.Table
    """
    lam_rest = np.asarray(df[lam_col], float)

    lam = lam_rest.copy()
    if line_v_kms != 0.0:
        c_kms = const.c.to("km/s").value
        lam *= (1.0 + line_v_kms / c_kms)
    if line_dlam_A != 0.0:
        lam += line_dlam_A

    wave_AA = _as_array(pumping, "WAVE")
    F_vals = _as_array(pumping, "FLUX")
    F_lambda = F_vals * (u.erg / (u.s * u.cm**2 * u.AA))

    lines = Table.from_pandas(df.copy())

    lam_q = lam * u.AA
    lines["Wave_vac_AA"] = lam
    lines["Frequency_Hz"] = (const.c / lam_q).to(u.Hz)

    if lsf_for_Jnu is None:
        F_interp = np.interp(lam, wave_AA, F_lambda.value) * F_lambda.unit
    else:
        F_eff = []
        for lam0 in lam:
            dl = wave_AA - lam0
            kern = np.asarray(lsf_for_Jnu(dl), float)
            kern = np.where(np.isfinite(kern), kern, 0.0)
            s = kern.sum()
            if s <= 0.0:
                f_val = np.interp(lam0, wave_AA, F_lambda.value)
            else:
                f_val = np.sum(F_lambda.value * kern) / s
            F_eff.append(f_val)
        F_interp = np.asarray(F_eff) * F_lambda.unit

    lines["F_lambda_at_comet_erg_s_cm2_AA"] = F_interp

    F_nu = F_interp.to(
        u.erg / (u.s * u.cm**2 * u.Hz),
        equivalencies=u.spectral_density(lam_q),
    )
    J_nu = (F_nu / (4.0 * np.pi)) * (1.0 / u.sr)
    lines["J_nu_erg_cm2_s_Hz_sr"] = J_nu.to(u.erg / (u.cm**2 * u.s * u.Hz * u.sr))
    return lines


# =============================================================================
# Radiative rate matrix (generic)
# =============================================================================

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

    :param lines: Transition table with solar incident irradiance quantities attached.
    :type lines: astropy.table.Table
    :param include_stim_emission: Include stimulated emission in downward rates.
    :type include_stim_emission: bool
    :param verbose: Print diagnostic information.
    :type verbose: bool
    :param A_col: Einstein A column name.
    :type A_col: str
    :param upper_id_col: Upper-level ID column name.
    :type upper_id_col: str
    :param lower_id_col: Lower-level ID column name.
    :type lower_id_col: str
    :param g_upper_col: Upper-level degeneracy column name.
    :type g_upper_col: str
    :param g_lower_col: Lower-level degeneracy column name.
    :type g_lower_col: str
    :returns: Rate matrix, index-to-level mapping, and annotated line table.
    :rtype: tuple[numpy.ndarray, dict[int, str], astropy.table.Table]
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


# =============================================================================
# Collisions (Option A: explicit lower-state columns; no ID parsing)
# =============================================================================

def _empty_scaffold() -> dict:
    return dict(iu=np.array([], int), il=np.array([], int),
                gu=np.array([]), gl=np.array([]), dE=np.array([]))


def precompute_cn_collision_scaffold(
    lines_out: Any,
    idx_to_level: dict,
    *,
    upper_id_col: str = "upper_id",
    lower_id_col: str = "lower_id",

    # level descriptors on the LOWER state (must exist if include_rotations=True)
    lower_es_col: str = "lower_es",
    lower_v_col: str = "lower_v",
    lower_J_col: str = "lower_J",
    lower_sym_col: str = "lower_sym",
    E_lower_cm1_col: str = "E_lower_cm1",

    include_deltaJ0_parity_mix: bool = True,
    require_X_only: bool = True,
) -> dict:
    """Build a rotational-collision scaffold from explicit lower-state columns.

    :param lines_out: Annotated transition table.
    :type lines_out: Any
    :param idx_to_level: Mapping from matrix index to level identifier.
    :type idx_to_level: dict
    :param upper_id_col: Upper-level ID column name.
    :type upper_id_col: str
    :param lower_id_col: Lower-level ID column name.
    :type lower_id_col: str
    :param lower_es_col: Lower electronic-state column name.
    :type lower_es_col: str
    :param lower_v_col: Lower vibrational-state column name.
    :type lower_v_col: str
    :param lower_J_col: Lower rotational-state column name.
    :type lower_J_col: str
    :param lower_sym_col: Lower-state symmetry/parity column name.
    :type lower_sym_col: str
    :param E_lower_cm1_col: Lower-state energy column name in cm^-1.
    :type E_lower_cm1_col: str
    :param include_deltaJ0_parity_mix: Allow parity-changing ``Delta J = 0`` collisions.
    :type include_deltaJ0_parity_mix: bool
    :param require_X_only: Restrict collisions to X-state levels.
    :type require_X_only: bool
    :returns: Scaffold dictionary with indices, degeneracies, and energy gaps.
    :rtype: dict
    :raises ValueError: If required columns are missing.
    """
    if lines_out is None or len(lines_out) == 0:
        return _empty_scaffold()

    def has_col(obj, col: str) -> bool:
        if hasattr(obj, "columns"):
            return col in obj.columns
        if hasattr(obj, "colnames"):
            return col in obj.colnames
        try:
            obj[col]
            return True
        except Exception:
            return False

    def col_as_array(obj, col: str) -> np.ndarray:
        return np.asarray(obj[col])

    needed = [upper_id_col, lower_id_col, lower_es_col, lower_v_col, lower_J_col, lower_sym_col, E_lower_cm1_col]
    missing = [c for c in needed if not has_col(lines_out, c)]
    if missing:
        raise ValueError(
            "include_rotations=True requires these columns in the linelist: "
            + ", ".join(missing)
        )

    # matrix level id -> matrix index
    level_id_to_idx: dict[str, int] = {}
    for i, v in idx_to_level.items():
        level_id_to_idx[str(v)] = int(i)

    # Take LOWER-state properties from transitions
    lower_ids = col_as_array(lines_out, lower_id_col).astype(str)
    les  = col_as_array(lines_out, lower_es_col).astype(str)
    lv   = col_as_array(lines_out, lower_v_col).astype(float)
    lJ   = col_as_array(lines_out, lower_J_col).astype(float)
    lsym = col_as_array(lines_out, lower_sym_col).astype(str)
    Elow = col_as_array(lines_out, E_lower_cm1_col).astype(float)

    good = np.isfinite(lv) & np.isfinite(lJ) & np.isfinite(Elow)
    lower_ids, les, lv, lJ, lsym, Elow = lower_ids[good], les[good], lv[good], lJ[good], lsym[good], Elow[good]

    if require_X_only:
        mX = np.array([str(es).strip().upper().startswith("X") for es in les], dtype=bool)
        lower_ids, les, lv, lJ, lsym, Elow = lower_ids[mX], les[mX], lv[mX], lJ[mX], lsym[mX], Elow[mX]

    if lower_ids.size == 0:
        return _empty_scaffold()

    # Each unique ground level is defined by (es, v, J, sym)
    keys = list(zip(les.astype(str), lv.astype(float), lJ.astype(float), lsym.astype(str)))

    from collections import defaultdict
    E_by_key = defaultdict(list)
    id_by_key = {}
    for k, e, lid in zip(keys, Elow, lower_ids):
        if np.isfinite(e):
            E_by_key[k].append(float(e))
            if k not in id_by_key:
                id_by_key[k] = str(lid)

    unique_keys = [k for k in E_by_key.keys() if len(E_by_key[k]) > 0 and k in id_by_key]
    if len(unique_keys) < 2:
        return _empty_scaffold()

    E_cm1_level = {k: float(np.median(E_by_key[k])) for k in unique_keys}

    # Map each unique key -> matrix index using the representative lower_id
    key_to_idx = {}
    for k in unique_keys:
        lid = id_by_key[k]
        if lid in level_id_to_idx:
            key_to_idx[k] = level_id_to_idx[lid]

    ground = [k for k in unique_keys if k in key_to_idx]
    if len(ground) < 2:
        return _empty_scaffold()

    # Sort by (J, sym) like your old code
    ground = sorted(ground, key=lambda k: (float(k[2]), str(k[3])))

    iu_list, il_list, gu_list, gl_list, dE_list = [], [], [], [], []
    seen = set()

    for a in range(len(ground)):
        esa, va, Ja, sa = ground[a]
        for b in range(a + 1, len(ground)):
            esb, vb, Jb, sb = ground[b]

            dJ = abs(float(Ja) - float(Jb))
            allow = (dJ == 1) or (include_deltaJ0_parity_mix and dJ == 0 and str(sa) != str(sb))
            if not allow:
                continue

            Ea = E_cm1_level[ground[a]]
            Eb = E_cm1_level[ground[b]]

            if Eb > Ea:
                ku, kl = ground[b], ground[a]
                dE_cm1 = Eb - Ea
            else:
                ku, kl = ground[a], ground[b]
                dE_cm1 = Ea - Eb

            mu = int(key_to_idx[ku])
            ml = int(key_to_idx[kl])
            keypair = (min(mu, ml), max(mu, ml))
            if keypair in seen:
                continue
            seen.add(keypair)

            gu = 2.0 * float(ku[2]) + 1.0
            gl = 2.0 * float(kl[2]) + 1.0

            iu_list.append(mu)
            il_list.append(ml)
            gu_list.append(gu)
            gl_list.append(gl)
            dE_list.append(dE_cm1)

    return dict(
        iu=np.asarray(iu_list, int),
        il=np.asarray(il_list, int),
        gu=np.asarray(gu_list, float),
        gl=np.asarray(gl_list, float),
        dE=np.asarray(dE_list, float),  # cm^-1
    )

def precompute_cn_collision_scaffold_fast(*args, **kwargs) -> dict:
    """Build collision scaffold and add cached arrays for fast updates.

    :param args: Positional arguments forwarded to :func:`precompute_cn_collision_scaffold`.
    :type args: tuple
    :param kwargs: Keyword arguments forwarded to :func:`precompute_cn_collision_scaffold`.
    :type kwargs: dict
    :returns: Collision scaffold with ``dE_over_k_K`` and ``gu_over_gl`` caches.
    :rtype: dict
    """
    sc = precompute_cn_collision_scaffold(*args, **kwargs)

    if sc.get("iu", np.array([])).size == 0:
        sc["dE_over_k_K"] = np.array([], float)
        sc["gu_over_gl"] = np.array([], float)
        return sc

    dE_cm1 = np.asarray(sc["dE"], float)
    gu = np.asarray(sc["gu"], float)
    gl = np.asarray(sc["gl"], float)

    sc["dE_over_k_K"] = HC_OVER_K_B_KCM * dE_cm1
    sc["gu_over_gl"] = gu / gl
    return sc

def apply_collisions_inplace(M: np.ndarray, scaffold: Dict[str, np.ndarray], Q: float, T: float) -> np.ndarray:
    """Apply rotational-collision rates to a matrix in place.

    :param M: Rate matrix to modify.
    :type M: numpy.ndarray
    :param scaffold: Collision scaffold containing ``iu``, ``il``, ``gu``, ``gl``, and ``dE``.
    :type scaffold: dict[str, numpy.ndarray]
    :param Q: Downward collision rate scale.
    :type Q: float
    :param T: Temperature in K (see our paper for the meaning of this T).
    :type T: float
    :returns: The modified matrix ``M``.
    :rtype: numpy.ndarray
    """
    if scaffold.get("iu", np.array([])).size == 0 or Q <= 0:
        return M

    iu = scaffold["iu"]
    il = scaffold["il"]
    gu = scaffold["gu"]
    gl = scaffold["gl"]
    dE_cm1 = scaffold["dE"]

    kT = (const.k_B * (T * u.K)).to(u.erg).value

    # Convert cm^-1 energy gaps to erg with h*c*wn.
    dE_erg = (const.h * const.c * (dE_cm1 / u.cm)).to(u.erg).value

    Cdown = float(Q)  # scalar
    Cup = (gu / gl) * Cdown * np.exp(-dE_erg / kT)

    np.add.at(M, (iu, iu), -Cdown)
    np.add.at(M, (il, il), -Cup)
    np.add.at(M, (il, iu),  Cdown)
    np.add.at(M, (iu, il),  Cup)
    return M

def apply_collisions_inplace_fast(
    M: np.ndarray,
    scaffold: Dict[str, np.ndarray],
    Q: float,
    T: float,
    Cup_work: np.ndarray,
) -> np.ndarray:
    """Apply collisions in place using cached arrays and reusable buffers.

    :param M: Rate matrix to modify.
    :type M: numpy.ndarray
    :param scaffold: Collision scaffold with ``dE_over_k_K`` and ``gu_over_gl``.
    :type scaffold: dict[str, numpy.ndarray]
    :param Q: Downward collision rate scale.
    :type Q: float
    :param T: Temperature in K (see our paper for the meaning of this T).
    :type T: float
    :param Cup_work: Reusable working array for upward rates.
    :type Cup_work: numpy.ndarray
    :returns: The modified matrix ``M``.
    :rtype: numpy.ndarray
    """
    iu = scaffold.get("iu", None)
    if iu is None or iu.size == 0 or Q <= 0.0 or not np.isfinite(T) or T <= 0.0:
        return M

    il = scaffold["il"]
    dE_over_k_K = scaffold["dE_over_k_K"]
    gu_over_gl = scaffold["gu_over_gl"]

    # Cup_work = exp(-dE/T) * (gu/gl) * Q
    np.divide(dE_over_k_K, T, out=Cup_work)     # Cup_work = dE/T
    np.negative(Cup_work, out=Cup_work)         # Cup_work = -dE/T
    np.exp(Cup_work, out=Cup_work)              # Cup_work = exp(-dE/T)
    Cup_work *= gu_over_gl                      # *= gu/gl
    Cup_work *= Q                               # *= Q

    # Cdown is scalar Q.
    Cdown = Q

    np.add.at(M, (iu, iu), -Cdown)
    np.add.at(M, (il, il), -Cup_work)
    np.add.at(M, (il, iu),  Cdown)
    np.add.at(M, (iu, il),  Cup_work)
    return M


# =============================================================================
# Solver & g-factors
# =============================================================================

def solve_with_normalization(M: np.ndarray, *, verbose: bool = True) -> np.ndarray:
    """Solve the steady-state system ``M @ n = 0`` with ``sum(n) = 1``.

    :param M: Rate matrix.
    :type M: numpy.ndarray
    :param verbose: Print solver diagnostics.
    :type verbose: bool
    :returns: Normalized non-negative level populations.
    :rtype: numpy.ndarray
    :raises RuntimeError: If no positive normalized solution can be formed.
    """
    n_levels = M.shape[0]
    A = M.astype(float)
    b = np.zeros(n_levels)
    A[0, :] = 1.0
    b[0] = 1.0

    n, *_ = np.linalg.lstsq(A, b, rcond=None)
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

    :param M: Rate matrix.
    :type M: numpy.ndarray
    :param A_work: Reusable matrix buffer.
    :type A_work: numpy.ndarray
    :param b_work: Reusable right-hand-side buffer.
    :type b_work: numpy.ndarray
    :returns: Normalized non-negative level populations.
    :rtype: numpy.ndarray
    :raises RuntimeError: If no positive normalized solution can be formed.
    """
    n_levels = M.shape[0]

    # Copy M → A_work (no allocation)
    np.copyto(A_work, M)

    # Reset b
    b_work.fill(0.0)

    # Normalization row
    A_work[0, :] = 1.0
    b_work[0] = 1.0

    n, *_ = np.linalg.lstsq(A_work, b_work, rcond=None)

    # Safety
    n = np.clip(n, 0.0, None)
    s = n.sum()
    if s <= 0.0:
        raise RuntimeError("Degenerate M")
    n /= s
    return n


def g_factors(
    lines_with_rates: Table,
    n: np.ndarray,
    *,
    A_col: str = "A_ul",
):
    """Compute per-line and total photon/energy g-factors.

    The g-factors are in units of photons/s/molecule for ``g_phot`` and erg/s/molecule for ``g_energy``.

    :param lines_with_rates: Transition table with cached rate-matrix indices.
    :type lines_with_rates: astropy.table.Table
    :param n: Level populations.
    :type n: numpy.ndarray
    :param A_col: Einstein A column name.
    :type A_col: str
    :returns: ``(g_phot, g_energy, sum_g_phot, sum_g_energy)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float, float]
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
    ui: np.ndarray,      # int array of upper indices per line
    A_ul: np.ndarray,     # float array of A_ul per line
    hnu: np.ndarray,     # float array of (h * nu) per line in erg
    n: np.ndarray,
    out_g_ph: Optional[np.ndarray] = None,
    out_g_en: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast g-factor evaluation using precomputed arrays.

    :param ui: Upper-level indices per line.
    :type ui: numpy.ndarray
    :param A_ul: Einstein A coefficients per line.
    :type A_ul: numpy.ndarray
    :param hnu: ``h * nu`` per line in erg.
    :type hnu: numpy.ndarray
    :param n: Level populations.
    :type n: numpy.ndarray
    :param out_g_ph: Optional output buffer for photon g-factors.
    :type out_g_ph: numpy.ndarray or None
    :param out_g_en: Optional output buffer for energy g-factors.
    :type out_g_en: numpy.ndarray or None
    :returns: ``(g_ph, g_en)`` arrays.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    n_u = n[ui]  # gather upper-level populations for each line

    # g_ph = n_u * A_ul
    if out_g_ph is None:
        g_ph = n_u * A_ul
    else:
        np.multiply(n_u, A_ul, out=out_g_ph)
        g_ph = out_g_ph

    # g_en = (h*nu) * g_ph
    if out_g_en is None:
        g_en = hnu * g_ph
    else:
        np.multiply(hnu, g_ph, out=out_g_en)
        g_en = out_g_en

    # Keep behavior aligned with the original function (avoid non-finite values).
    np.nan_to_num(g_ph, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(g_en, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return g_ph, g_en

# =============================================================================
# Spectrum synthesis
# =============================================================================

def synth_spectrum_from_lines(
    df_lines: Table,
    *,
    g_line_energy: Optional[np.ndarray] = None,
    g_line_phot: Optional[np.ndarray] = None,
    fwhm_A: float = 0.02,
    dlam_A: float = 0.05,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    lam_col: str = "Wave_vac_AA",
    N_col_cm2: Optional[float] = None,
    Omega_sr: Optional[float] = None,
    grid: Optional[np.ndarray] = None,
    lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    v_shift_kms: float = 0.0,
    dlam_shift_A: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a synthetic emission spectrum from line g-factors.

    :param df_lines: Transition table.
    :type df_lines: astropy.table.Table
    :param g_line_energy: Per-line energy g-factors.
    :type g_line_energy: numpy.ndarray or None
    :param g_line_phot: Per-line photon g-factors.
    :type g_line_phot: numpy.ndarray or None
    :param fwhm_A: Gaussian FWHM in Angstrom if no custom LSF is provided.
    :type fwhm_A: float
    :param dlam_A: Wavelength step for an auto-generated grid.
    :type dlam_A: float
    :param lam_min: Optional minimum wavelength.
    :type lam_min: float or None
    :param lam_max: Optional maximum wavelength.
    :type lam_max: float or None
    :param lam_col: Wavelength column in ``df_lines``.
    :type lam_col: str
    :param N_col_cm2: Column density in cm^-2.
    :type N_col_cm2: float or None
    :param Omega_sr: Optional solid angle scaling in sr.
    :type Omega_sr: float or None
    :param grid: Optional output wavelength grid.
    :type grid: numpy.ndarray or None
    :param lsf: Optional custom line-spread function.
    :type lsf: Callable[[numpy.ndarray], numpy.ndarray] or None
    :param v_shift_kms: Velocity shift in km/s.
    :type v_shift_kms: float
    :param dlam_shift_A: Additive wavelength shift in Angstrom.
    :type dlam_shift_A: float
    :returns: ``(wavelength_grid, synthetic_flux)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If required inputs are missing.
    """
    if N_col_cm2 is None:
        raise ValueError("N_col_cm2 (cm^-2) is required.")
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
        dl = grid[:, None] - lam[None, :]
        prof = lsf(dl)  # (Ngrid, Nlines)
        spec_per_mol = (prof * I_line).sum(axis=1)

    I_lambda = (N_col_cm2 / (4.0 * np.pi)) * spec_per_mol

    if Omega_sr is not None:
        return grid, I_lambda * Omega_sr
    else:
        return grid, I_lambda


# =============================================================================
# LSF helper
# =============================================================================

def make_lsf(params: Dict[str, float], mode: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a line-spread function callable from parameter values.
        If LSF mode == "2Gauss", the parameters are ``sigma1``, ``sigma2``, and ``ratio``.
        If LSF mode == "Gauss", the parameter is ``sigma``.
        If LSF mode == "Gauss_Lorentz", the parameters are ``sigma_G``, ``fwhm_L``, and ``ratio``.
        If LSF mode == "Lorentz", the parameter is ``fwhm_L``.
        
    :param params: LSF parameter dictionary.
    :type params: dict[str, float]
    :param mode: LSF mode: ``2Gauss``, ``Gauss``, ``Gauss_Lorentz``, or ``Lorentz``.
    :type mode: str
    :returns: LSF callable.
    :rtype: Callable[[numpy.ndarray], numpy.ndarray]
    :raises ValueError: If ``mode`` is invalid.
    """
    if mode == "2Gauss":
        sigma1 = params.get("sigma1", 0.01)
        sigma2 = params.get("sigma2", 0.005)
        ratio = params.get("ratio", 0.9)

        def lsf_fun(dl: np.ndarray) -> np.ndarray:
            g1 = np.exp(-0.5 * (dl / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
            g2 = np.exp(-0.5 * (dl / sigma2) ** 2) / (sigma2 * np.sqrt(2.0 * np.pi))
            return ratio * g1 + (1.0 - ratio) * g2

        return lsf_fun

    if mode == "Gauss":
        sigma = params.get("sigma", 0.01)

        def lsf_fun(dl: np.ndarray) -> np.ndarray:
            return np.exp(-0.5 * (dl / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

        return lsf_fun

    if mode == "Gauss_Lorentz":
        sigma_G = params.get("sigma_G", 0.01)
        fwhm_L = params.get("fwhm_L", 0.02)
        ratio = params.get("ratio", 0.9)
        gamma = fwhm_L / 2.0
        A = 2.0 / (np.pi * fwhm_L)

        def lsf_fun(dl: np.ndarray) -> np.ndarray:
            gauss = np.exp(-0.5 * (dl / sigma_G) ** 2) / (sigma_G * np.sqrt(2.0 * np.pi))
            lorentz = A * gamma**2 / (gamma**2 + dl**2)
            return ratio * gauss + (1.0 - ratio) * lorentz

        return lsf_fun

    if mode == "Lorentz":
        fwhm_L = params.get("fwhm_L", 0.02)
        gamma = fwhm_L / 2.0
        A = 2.0 / (np.pi * fwhm_L)

        def lsf_fun(dl: np.ndarray) -> np.ndarray:
            return A * gamma**2 / (gamma**2 + dl**2)

        return lsf_fun

    raise ValueError("Invalid LSF mode.")


# =============================================================================
# MCMC fitting (multi-isotopologue + systems + defaults or user line lists)
# =============================================================================

def mcmc_fitting(
    data: Any,
    window: Tuple[float, float],
    *,
    pumping: Any,

    isotopologues: str | Sequence[str] = "12C14N",
    systems: str | Sequence[str] | None = None,

    # user-provided normalized transitions (single df or dict iso->df)
    linelists: pd.DataFrame | dict[str, pd.DataFrame] | None = None,

    # collisions:
    include_rotations: bool = True,
    include_deltaJ0_parity_mix: bool = True,
    require_X_only_for_rot: bool = True,

    nwalkers: int = 50,
    nsteps: int = 1000,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,

    lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lsf_method: Optional[str] = None,

    make_plots: bool = False,
    progress: bool = True,

    A_min: float = 1e4,
    a: float = 3,
    threads: int = 1,

    # NOTE: these control *pumping* wavelength shift for J_nu (radiative rates)
    velocity_kms: float = 0.0,
    delta_lambda_A: float = 0.0,

    # NOTE: these are fallbacks for parameters not present in priors
    init_logQ: float = -3.0,
    init_T: float = 300.0,
    init_v_kms: float = 0.0,
    init_dlam: float = 0.0,

    fig_file: Optional[str] = None,
    wave_col: str = "WAVE",
    flux_col: str = "FLUX_STACK",
    error_col: str = "ERR_STACK",
    continuum_col: str = "CONTINUUM",
    omega: Optional[float] = None,
    verbose: bool = True,
    pruning: bool = True,
) -> Dict[str, Any]:
    """Run MCMC fitting for the fluorescence model in a wavelength window.

        This routine builds the model for one or more isotopologues, applies an
        optional line-spread function (LSF), and samples posterior distributions for
        the parameters provided in ``priors``.

        Defaults and selector behavior
        ------------------------------
        - ``isotopologues`` defaults to ``"12C14N"``.
        - ``systems`` defaults to ``None``, which maps to ``["BX00", "AX_dv1"]``.
        - Accepted string selectors for ``systems`` include:
            ``"both"``/``"bx+ax"``/``"bxax"`` -> ``["BX00", "AX_dv1", "AX_dv2", "AX_dv3"]``
            ``"all"`` -> ``["ALL"]``
            ``"bx"``, ``"b-x"``, ``"bx(0,0)"``, ``"bx00"``, ``"bx_00"``, ``"b_x_00"`` -> ``["BX00"]``
            ``"ax"``/``"a-x"`` -> ``["AX_dv1", "AX_dv2"]``
            ``"ax(dv=1)"``/``"ax_dv1"`` -> ``["AX_dv1"]``
            ``"ax(dv=2)"``/``"ax_dv2"`` -> ``["AX_dv2"]``
            ``"ax(dv=3)"``/``"ax_dv3"`` -> ``["AX_dv3"]``
        - ``nwalkers=50`` and ``nsteps=1000`` by default.
        - Collision controls default to ``include_rotations=True``,
            ``include_deltaJ0_parity_mix=True``, and ``require_X_only_for_rot=True``.
        - Pumping-shift controls default to ``velocity_kms=0.0`` and
            ``delta_lambda_A=0.0``.
        - Fallback parameter values (used when the parameter is not sampled) default
            to ``init_logQ=-3.0``, ``init_T=300.0``, ``init_v_kms=0.0``, and
            ``init_dlam=0.0``.

    :param data: Observed spectrum table or DataFrame.
    :type data: Any
    :param window: Wavelength fitting window ``(min_A, max_A)``.
    :type window: tuple[float, float]
    :param pumping: Pumping spectrum with ``WAVE`` and ``FLUX`` columns.
    :type pumping: Any
    :param isotopologues: One or more isotopologue labels. Default is ``"12C14N"``.
    :type isotopologues: str or Sequence[str]
    :param systems: CN system selector(s). If ``None``, uses ``["BX00", "AX_dv1"]``.
    :type systems: str or Sequence[str] or None
    :param linelists: Optional normalized line-list DataFrame or isotopologue mapping.
    :type linelists: pandas.DataFrame or dict[str, pandas.DataFrame] or None
    :param include_rotations: Enable rotational collisions.
    :type include_rotations: bool
    :param include_deltaJ0_parity_mix: Allow parity-changing ``Delta J = 0`` collisions.
    :type include_deltaJ0_parity_mix: bool
    :param require_X_only_for_rot: Restrict collisions to X-state levels.
    :type require_X_only_for_rot: bool
    :param nwalkers: Number of walkers. Default is ``50``.
    :type nwalkers: int
    :param nsteps: Number of MCMC steps. Default is ``1000``.
    :type nsteps: int
    :param priors: Parameter prior ranges.
    :type priors: dict[str, tuple[float, float]] or None
    :param lsf: Optional custom LSF callable.
    :type lsf: Callable[[numpy.ndarray], numpy.ndarray] or None
    :param lsf_method: Built-in LSF method name.
    :type lsf_method: str or None
    :param make_plots: Generate diagnostic plots.
    :type make_plots: bool
    :param progress: Show emcee progress output.
    :type progress: bool
    :param A_min: Minimum Einstein A threshold for default CN line lists.
    :type A_min: float
    :param a: Stretch-move parameter for emcee.
    :type a: float
    :param threads: Number of emcee threads.
    :type threads: int
    :param velocity_kms: Velocity shift used when evaluating pumping J_nu. Default is ``0.0``.
    :type velocity_kms: float
    :param delta_lambda_A: Additive wavelength shift used when evaluating pumping J_nu. Default is ``0.0``.
    :type delta_lambda_A: float
    :param init_logQ: Fallback ``logQ`` value when not sampled.
    :type init_logQ: float
    :param init_T: Fallback temperature value when not sampled.
    :type init_T: float
    :param init_v_kms: Fallback velocity shift when not sampled.
    :type init_v_kms: float
    :param init_dlam: Fallback wavelength shift when not sampled.
    :type init_dlam: float
    :param fig_file: Base path for output figures.
    :type fig_file: str or None
    :param wave_col: Wavelength column in ``data``.
    :type wave_col: str
    :param flux_col: Flux column in ``data``.
    :type flux_col: str
    :param error_col: Uncertainty column in ``data``.
    :type error_col: str
    :param continuum_col: Continuum column in ``data``.
    :type continuum_col: str
    :param omega: Optional aperture solid angle in sr.
    :type omega: float or None
    :param verbose: Print diagnostics.
    :type verbose: bool
    :param pruning: Apply posterior pruning.
    :type pruning: bool
    :returns: Dictionary with posterior summaries, samples, and model envelopes.
    :rtype: dict[str, Any]
    :raises ValueError: If priors or required parameters are inconsistent.
    """
    if priors is None:
        raise ValueError("Please provide a dict of priors for the parameters to fit.")

    iso_list = _as_list(isotopologues)
    sys_tokens = normalize_systems_arg(systems)

    param_keys = list(priors.keys())

    # ---------- LSF prior handling ----------
    if lsf is not None:
        drop = {"sigma_G", "fwhm_L", "sigma", "sigma1", "sigma2", "ratio"}
        param_keys = [k for k in param_keys if k not in drop]
        priors = {k: priors[k] for k in param_keys}
    else:
        if lsf_method == "2Gauss":
            required = {"sigma1", "sigma2", "ratio"}
            if not required.issubset(param_keys):
                raise ValueError("For 2Gauss: priors for sigma1, sigma2, ratio required.")
            drop = {"sigma_G", "fwhm_L", "sigma"}
        elif lsf_method == "Gauss_Lorentz":
            required = {"sigma_G", "fwhm_L", "ratio"}
            if not required.issubset(param_keys):
                raise ValueError("For Gauss_Lorentz: priors for sigma_G, fwhm_L, ratio required.")
            drop = {"sigma1", "sigma2", "sigma"}
        elif lsf_method == "Gauss":
            required = {"sigma"}
            if not required.issubset(param_keys):
                raise ValueError("For Gauss: prior for sigma required.")
            drop = {"sigma_G", "fwhm_L", "sigma1", "sigma2", "ratio"}
        elif lsf_method == "Lorentz":
            required = {"fwhm_L"}
            if not required.issubset(param_keys):
                raise ValueError("For Lorentz: prior for fwhm_L required.")
            drop = {"sigma_G", "sigma1", "sigma2", "sigma", "ratio"}
        else:
            raise ValueError("Provide `lsf` or lsf_method in {'2Gauss','Gauss_Lorentz','Gauss','Lorentz'}.")

        param_keys = [k for k in param_keys if k not in drop]
        priors = {k: priors[k] for k in param_keys}

    for name in param_keys:
        lo, hi = priors[name]
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f"Bad prior for {name!r}: {priors[name]}")

    # ---------- 1) Line lists: defaults or user-provided ----------
    if linelists is None:
        trans_by_iso = load_default_cn_transitions(
            isotopologues=iso_list,
            systems=sys_tokens,
            A_min=A_min,
            use_omega_labels=False,
            line_paths=None,
        )
    else:
        if isinstance(linelists, pd.DataFrame):
            if len(iso_list) != 1:
                raise ValueError("If linelists is a single DataFrame, isotopologues must be a single iso.")
            trans_by_iso = {iso_list[0]: linelists}
        else:
            trans_by_iso = {iso: linelists[iso] for iso in iso_list}

    # If include_rotations=True, enforce required columns in each iso linelist
    if include_rotations:
        req = {"lower_es", "lower_v", "lower_J", "lower_sym", "E_lower_cm1"}
        for iso, df_trans in trans_by_iso.items():
            missing = sorted(list(req - set(df_trans.columns)))
            if missing:
                raise ValueError(
                    f"include_rotations=True but linelist for iso={iso!r} is missing columns: {missing}. "
                    "Provide them via from_user_linelist(... lower_*_col=..., E_lower_cm1_col=...)."
                )

    # ---------- 2) Radiative caches per iso ----------
    cache: dict[str, dict[str, Any]] = {}
    for iso, df_trans in trans_by_iso.items():
        lines_theta = attach_pumping_and_labels(
            df_trans,
            pumping,
            line_v_kms=float(velocity_kms),
            line_dlam_A=float(delta_lambda_A),
            lsf_for_Jnu=None,
            lam_col="lambda_vac_A",
        )

        M_rad, idx_to_level, lines_out = build_rate_matrix_nbar(
            lines_theta,
            include_stim_emission=True,
            verbose=False,
            A_col="A_ul",
            upper_id_col="upper_id",
            lower_id_col="lower_id",
            g_upper_col="g_upper",
            g_lower_col="g_lower",
        )
        ui = np.asarray(lines_out["__upper_idx"], dtype=np.int64)
        A_ul = np.asarray(lines_out["A_ul"], dtype=np.float64)          # same column you used in g_factors
        nu = np.asarray(lines_out["__nu_Hz"], dtype=np.float64)        # already cached as numeric Hz
        hnu = H_CGS * nu 
        gph_work = np.empty_like(A_ul, dtype=float)
        gen_work = np.empty_like(A_ul, dtype=float)

        if include_rotations:
            coll_scaf = precompute_cn_collision_scaffold_fast(
                lines_out, idx_to_level,
                include_deltaJ0_parity_mix=include_deltaJ0_parity_mix,
                require_X_only=require_X_only_for_rot,
            )
        else:
            coll_scaf = _empty_scaffold()

        M_work = np.empty_like(M_rad)              # reusable matrix buffer
        A_work = np.empty_like(M_rad)              # reusable solver matrix buffer
        b_work = np.zeros(M_rad.shape[0], float)   # reusable RHS vector
        Cup_work = np.empty_like(coll_scaf.get("iu", np.array([], dtype=int)), dtype=float)


        cache[iso] = dict(
            M_rad=M_rad,
            idx_to_level=idx_to_level,
            lines_out=lines_out,
            coll_scaf=coll_scaf,

            # ✅ new buffers
            M_work=M_work,
            A_work=A_work,
            b_work=b_work,
            ui=ui,
            A_ul=A_ul,
            hnu=hnu,
            Cup_work=Cup_work,
            gph_work=gph_work,
            gen_work=gen_work,

        )

    # ---------- 3) Observed data subset ----------
    def _col(obj, name: str) -> np.ndarray:
        if hasattr(obj, "colnames"):
            return np.asarray(obj[name])
        if hasattr(obj, "columns"):
            return np.asarray(obj[name].values)
        return np.asarray(obj[name])

    x_data = _col(data, wave_col)
    y_data = _col(data, flux_col)
    y_err = _col(data, error_col)
    cont = _col(data, continuum_col)

    mwin = (x_data >= window[0]) & (x_data <= window[1])
    x_fit = x_data[mwin]
    y_fit = y_data[mwin] - cont[mwin]
    y_err_fit = y_err[mwin]

    # ---------- 4) helpers ----------
    def theta_to_params(theta: Sequence[float]) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(param_keys, theta)}

    def ln_prior(theta: Sequence[float]) -> float:
        for val, name in zip(theta, param_keys):
            lo, hi = priors[name]
            if val < lo or val > hi:
                return -np.inf
        return 0.0

    def make_lsf_local(pars: Dict[str, float]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        if lsf is not None:
            return lsf
        if lsf_method is None:
            return None
        return make_lsf(pars, lsf_method)

    def model_flux(theta: Sequence[float], wave: np.ndarray) -> np.ndarray:
        pars = theta_to_params(theta)
        wmin = float(np.min(wave))
        wmax = float(np.max(wave))

        # ✅ Correct fallback behavior:
        # If not being fit, use init_* (provided by caller), not magic hardcoded defaults.
        logQ = float(pars["logQ"]) if "logQ" in pars else float(init_logQ)
        T = float(pars["T"]) if "T" in pars else float(init_T)
        v_kms = float(pars["v_kms"]) if "v_kms" in pars else float(init_v_kms)
        dlam = float(pars["dlam"]) if "dlam" in pars else float(init_dlam)

        lsf_fun = make_lsf_local(pars)

        Q = 10.0 ** logQ if np.isfinite(logQ) else 0.0
        spec_total = np.zeros_like(wave, dtype=float)

        use_reuse = (threads == 1)   # threads is captured from outer scope

        for iso in iso_list:
            C = cache[iso]
            if use_reuse:
                M = C["M_work"]
                np.copyto(M, C["M_rad"])
            else:
                M = C["M_rad"].copy()

            if Q > 0.0 and include_rotations:
                apply_collisions_inplace_fast(M, C["coll_scaf"], Q=Q, T=T, Cup_work=C["Cup_work"])

            if use_reuse:
                n = solve_with_normalization_fast(M, C["A_work"], C["b_work"])
            else:
                n = solve_with_normalization(M, verbose=False)

            g_ph, g_en = g_factors_fast_from_cache(
                                                    ui=C["ui"],
                                                    A_ul=C["A_ul"],
                                                    hnu=C["hnu"],
                                                    n=n,
                                                    out_g_ph=C["gph_work"],
                                                    out_g_en=C["gen_work"],
                                                )

            # Column density per iso
            if len(iso_list) == 1 and "logN" in pars:
                logN_i = float(pars["logN"])
            else:
                key = f"logN_{iso}"
                if key not in pars:
                    raise ValueError(f"Missing parameter {key!r} in priors for multi-isotopologue fit.")
                logN_i = float(pars[key])

            _, spec_i = synth_spectrum_from_lines(
                C["lines_out"],
                g_line_energy=g_en,
                lam_min=wmin,
                lam_max=wmax,
                lam_col="Wave_vac_AA",
                N_col_cm2=10.0 ** logN_i,
                Omega_sr=omega,
                grid=wave,
                lsf=lsf_fun,
                v_shift_kms=v_kms,
                dlam_shift_A=dlam,
            )
            spec_total += spec_i

        return spec_total
    def model_flux_post(theta: Sequence[float], wave: np.ndarray) -> np.ndarray:
        """
        Post-processing model evaluation.
        Always runs single-threaded, so safe to reuse buffers.
        """
        wmin = float(np.min(wave))
        wmax = float(np.max(wave))

        pars = theta_to_params(theta)

        logQ = float(pars["logQ"]) if "logQ" in pars else float(init_logQ)
        T = float(pars["T"]) if "T" in pars else float(init_T)
        v_kms = float(pars["v_kms"]) if "v_kms" in pars else float(init_v_kms)
        dlam = float(pars["dlam"]) if "dlam" in pars else float(init_dlam)

        lsf_fun = make_lsf_local(pars)
        Q = 10.0 ** logQ if np.isfinite(logQ) else 0.0

        spec_total = np.zeros_like(wave, dtype=float)

        for iso in iso_list:
            C = cache[iso]
            M = C["M_work"]
            np.copyto(M, C["M_rad"])

            if Q > 0.0 and include_rotations:
                apply_collisions_inplace_fast(M, C["coll_scaf"], Q=Q, T=T, Cup_work=C["Cup_work"])

            n = solve_with_normalization_fast(M, C["A_work"], C["b_work"])

            g_ph, g_en = g_factors_fast_from_cache(
                ui=C["ui"], A_ul=C["A_ul"], hnu=C["hnu"], n=n,
                out_g_ph=C["gph_work"], out_g_en=C["gen_work"]
            )

            if len(iso_list) == 1 and "logN" in pars:
                logN_i = float(pars["logN"])
            else:
                key = f"logN_{iso}"
                if key not in pars:
                    raise ValueError(f"Missing parameter {key!r} in priors for multi-isotopologue fit.")
                logN_i = float(pars[key])

            _, spec_i = synth_spectrum_from_lines(
                C["lines_out"],
                g_line_energy=g_en,
                lam_min=wmin,
                lam_max=wmax,
                lam_col="Wave_vac_AA",
                N_col_cm2=10.0 ** logN_i,
                Omega_sr=omega,
                grid=wave,
                lsf=lsf_fun,
                v_shift_kms=v_kms,
                dlam_shift_A=dlam,
            )
            spec_total += spec_i

        return spec_total

    def lnlike(theta: Sequence[float]) -> float:
        y_model = model_flux(theta, x_fit)
        if (not np.all(np.isfinite(y_model))) or (y_model.shape != x_fit.shape):
            return -np.inf
        inv_sigma2 = 1.0 / (y_err_fit**2)
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * y_err_fit**2) +
            (y_fit - y_model) ** 2 * inv_sigma2
        )

    def lnprob(theta: Sequence[float]) -> float:
        lp = ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = lnlike(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # ---------- 5) Run emcee ----------
    ndim = len(param_keys)
    nburn = nsteps // 2
    print("Number of iterations:", ndim * nwalkers * nsteps)

    p0 = np.array([[np.random.uniform(*priors[name]) for name in param_keys] for _ in range(nwalkers)])

    move = emcee.moves.StretchMove(a=a)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, moves=move, threads=threads)
    sampler.run_mcmc(p0, nsteps, progress=progress)

    chain = sampler.get_chain()
    lnprob_full = sampler.get_log_prob()

    # ---------- 6) Best-fit ----------
    flat_chain = chain.reshape(-1, ndim)
    flat_lnprob = lnprob_full.reshape(-1)
    best_idx = int(np.argmax(flat_lnprob))
    best_theta = flat_chain[best_idx]
    best_params = theta_to_params(best_theta)
    if verbose:
        print("#" * 50)
        print("*** Best fit (no pruning) ***")
        for name in param_keys:
            print(f"{name}: {best_params[name]:.6g}")

    af = sampler.acceptance_fraction
    if verbose:
        print("#" * 50)
        print("*** Acceptance Fraction ***")
        print("Mean acceptance fraction:", np.mean(af))
    af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                            between 0.2 and 0.5
            If af < 0.2 decrease the MCMCA parameter
            If af > 0.5 increase the MCMCA parameter
            '''
    if verbose:
        print("Mean acceptance fraction:", np.mean(af))
    if np.mean(af)<0.2 or np.mean(af)>0.5:
        print(af_msg)
        warnings.warn("Acceptance fraction out of bounds.", UserWarning)

    # ---------- 7) Burn-in removal ----------
    samples = chain[nburn:, :, :].reshape(-1, ndim)
    lnprob_burn = lnprob_full[nburn:, :].reshape(-1)

    # ---------- 8) Simple pruning ----------
    def prune(samples: np.ndarray,
              lnprob_arr: np.ndarray,
              scaler: float = 5.0,
              quiet: bool = False):
        minlnprob = lnprob_arr.max()
        dln = np.abs(lnprob_arr - minlnprob)
        med = np.median(dln)
        avg = np.mean(dln)
        skew = abs(avg - med)
        rms = np.std(dln)
        mask = dln < scaler * rms
        ln2 = lnprob_arr[mask]
        s2 = samples[mask]

        prev_med = 0.0
        while skew > 0.1 * med and ln2.size > 0:
            minlnprob = ln2.max()
            dln = np.abs(ln2 - minlnprob)
            rms = np.std(dln)
            mask = dln < scaler * rms
            if mask.sum() == ln2.size:
                mask = dln < (scaler / 2.0) * rms
            ln2 = ln2[mask]
            s2 = s2[mask]
            dln = np.abs(ln2 - minlnprob)
            med = np.median(dln)
            avg = np.mean(dln)
            skew = abs(avg - med)
            if not quiet:
                print(med, avg, skew)
            if med == prev_med:
                scaler /= 1.5
            prev_med = med

        good = ln2 <= ln2.max()
        return s2[good], ln2[good]
    if pruning:
        if verbose:
            print("#" * 50)
            print("*** Pruning... ***")
        try:
            samples_pruned, lnprob_pruned = prune(samples, lnprob_burn, quiet=not progress)
        except Exception as exc:
            print("Pruning failed:", exc)
            samples_pruned, lnprob_pruned = samples, lnprob_burn
    else:
        samples_pruned, lnprob_pruned = samples, lnprob_burn

    # ---------- 9) Posterior summaries ----------
    median_params: Dict[str, float] = {}
    up_errors: Dict[str, float] = {}
    low_errors: Dict[str, float] = {}

    for i, name in enumerate(param_keys):
        p16, p50, p84 = np.percentile(samples_pruned[:, i], [16, 50, 84])
        median_params[name] = float(p50)
        up_errors[name] = float(p84 - p50)
        low_errors[name] = float(p50 - p16)
        err = 0.5 * ((p84 - p50) + (p50 - p16))
        print(f"{name}: {p50:.4f} +/- {err:.4f}  [{p16:.4f}, {p84:.4f}]")

    # ---------- 10) Model ensemble ----------
    x_model = np.linspace(window[0], window[1], 20000)
    n_draw = min(200, samples_pruned.shape[0])
    model_stack = np.empty((n_draw, x_model.size))
    for i in range(n_draw):
        model_stack[i] = model_flux_post(samples_pruned[i], x_model)

    theta_med = [median_params[k] for k in param_keys]
    best_model = model_flux_post(theta_med, x_model)

    p16_m, p50_m, p84_m = np.percentile(model_stack, [16, 50, 84], axis=0)
    median_model = p50_m
    model_p16 = p16_m
    model_p84 = p84_m

    param_labels = {
        "logN": r"$\log_{10}(N$ / [mol cm$^{-2}$])",
        "logQ": r"$\log_{10}$(Q$_{\rm{col}}$ / [s$^{-1}$])",
        "T": r"T [K]",
        "v_kms": r"$\Delta$v [km s$^{-1}$]",
        "dlam": r"$\Delta \lambda$ [Å]",
        "sigma": r"$\sigma$ [Å]",
        "sigma1": r"$\sigma_1$ [Å]",
        "sigma2": r"$\sigma_2$ [Å]",
        "sigma_G": r"$\sigma_G$ [Å]",
        "fwhm_L": r"FWHM$_L$ [Å]",
        "ratio": r"Ratio",}
    if make_plots:
        # traces
        fig, axes = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
        if ndim == 1:
            axes = [axes]

        # chain has shape (nsteps, nwalkers, ndim)
        nsteps_chain, nwalkers_chain, _ = chain.shape
        steps = np.arange(nsteps_chain)

        for j, name in enumerate(param_keys):
            for w in range(nwalkers_chain):
                # use [:, w, j] not [w, :, j]
                axes[j].plot(steps, chain[:, w, j], alpha=0.7, lw=0.8)
            axes[j].set_ylabel(param_labels[name])
        axes[-1].set_xlabel("iteration")
        fig.tight_layout()
        plt.savefig(f"{fig_file}_mcmc_traces.pdf", dpi=300, format='pdf')
        plt.show()

        # corner
        corner.corner(
            samples_pruned,
            labels=[param_labels[k] for k in param_keys],
            title_kwargs={'y':1.05},title_fmt=".3f",use_math_text=True,bins=15,quantiles=[0.16, 0.5, 0.84],show_titles=True,
                color='lightseagreen',hist_kwargs={'color':'black','linewidth':1.5},contour_kwargs={'linewidths':1,'colors':'black'}, spacing=0.001
        )
        plt.savefig(f"{fig_file}_corner.pdf", dpi=300, format='pdf')
        plt.show()

        # data vs model
        plt.figure(figsize=(10, 6))
        plt.plot(x_fit, y_fit, label="Data (cont-sub)", color="black", alpha=0.8)
        plt.fill_between(
            x_fit,
            y_fit - y_err_fit,
            y_fit + y_err_fit,
            color="k",
            alpha=0.25,
            label="1σ",
        )
        plt.plot(x_model, median_model, label="Median Model", color="crimson", alpha=0.9)
        plt.xlabel("Wavelength [Å]")
        plt.ylabel(r"F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$  Å$^{-1}$]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{fig_file}_fit.pdf", dpi=300, format='pdf')
        plt.show()

    return {
        "param_keys": param_keys,
        "median_params": median_params,
        "up_errors_params": up_errors,
        "low_errors_params": low_errors,
        "samples_pruned": samples_pruned,
        "lnprob_pruned": lnprob_pruned,
        "model_wave": x_model,
        "median_model": median_model,
        "model_p16": model_p16,
        "model_p84": model_p84,
        "best_model": best_model,
    }