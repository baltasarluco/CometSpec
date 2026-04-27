from __future__ import annotations
from pathlib import Path

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

import re

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

    lam_s = pd.to_numeric(df[lam_col], errors="coerce")
    A_s = pd.to_numeric(df[A_col], errors="coerce")
    gu_s = pd.to_numeric(df[g_upper_col], errors="coerce")
    gl_s = pd.to_numeric(df[g_lower_col], errors="coerce")
    uid_s = df[upper_id_col]
    lid_s = df[lower_id_col]

    valid = (
        np.isfinite(lam_s) & (lam_s > 0)
        & np.isfinite(A_s) & (A_s >= 0)
        & np.isfinite(gu_s) & (gu_s > 0)
        & np.isfinite(gl_s) & (gl_s > 0)
        & uid_s.notna() & lid_s.notna()
    )

    opt_specs: list[tuple[str, str, bool]] = [
        (lower_es_col, "lower_es_col", False),
        (lower_v_col, "lower_v_col", True),
        (lower_J_col, "lower_J_col", True),
        (lower_sym_col, "lower_sym_col", False),
        (E_lower_cm1_col, "E_lower_cm1_col", True),
    ]
    for col, label, numeric in opt_specs:
        if col is None:
            continue
        if col not in df.columns:
            raise ValueError(f"{label}={col!r} not found.")
        if numeric:
            valid = valid & np.isfinite(pd.to_numeric(df[col], errors="coerce"))
        else:
            valid = valid & df[col].notna()

    valid = valid.to_numpy()
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        warnings.warn(
            f"from_user_linelist: dropping {n_dropped} row(s) with missing/invalid values."
        )

    df = df.iloc[valid].reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)
    out["upper_id"] = df[upper_id_col].astype(str)
    out["lower_id"] = df[lower_id_col].astype(str)
    out["g_upper"] = pd.to_numeric(df[g_upper_col], errors="coerce").astype(float)
    out["g_lower"] = pd.to_numeric(df[g_lower_col], errors="coerce").astype(float)

    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8
    out["E_cm1"] = 1.0 / lam_cm

    if lower_es_col is not None:
        out["lower_es"] = df[lower_es_col].astype(str).str.strip().str.upper()
    if lower_v_col is not None:
        out["lower_v"] = pd.to_numeric(df[lower_v_col], errors="coerce").astype(float)
    if lower_J_col is not None:
        out["lower_J"] = pd.to_numeric(df[lower_J_col], errors="coerce").astype(float)
    if lower_sym_col is not None:
        out["lower_sym"] = df[lower_sym_col].astype(str).str.strip()
    if E_lower_cm1_col is not None:
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
    src_cols = [
        lam_col, A_col,
        "F'", "p'", "eS'", "v'", "J'",
        "F''", "p''", "eS''", "v''", "J''",
        E_lower_col,
    ]
    missing = [c for c in src_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brooke linelist missing required columns: {missing}")

    lam_s = pd.to_numeric(df[lam_col], errors="coerce")
    A_s = pd.to_numeric(df[A_col], errors="coerce")
    Vu_s = pd.to_numeric(df["v'"], errors="coerce")
    Vl_s = pd.to_numeric(df["v''"], errors="coerce")
    Ju_s = pd.to_numeric(df["J'"], errors="coerce")
    Jl_s = pd.to_numeric(df["J''"], errors="coerce")
    El_s = pd.to_numeric(df[E_lower_col], errors="coerce")

    valid = (
        np.isfinite(lam_s) & (lam_s > 0)
        & np.isfinite(A_s) & (A_s >= 0)
        & np.isfinite(Vu_s) & np.isfinite(Vl_s)
        & np.isfinite(Ju_s) & np.isfinite(Jl_s)
        & np.isfinite(El_s)
        & df["F'"].notna() & df["p'"].notna() & df["eS'"].notna()
        & df["F''"].notna() & df["p''"].notna() & df["eS''"].notna()
    ).to_numpy()

    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        warnings.warn(
            f"from_cn_brooke: dropping {n_dropped} row(s) with missing/invalid values."
        )

    if not valid.any():
        return pd.DataFrame(columns=[
            "lambda_vac_A", "A_ul", "upper_id", "lower_id",
            "g_upper", "g_lower", "E_cm1",
            "lower_es", "lower_v", "lower_J", "lower_sym", "E_lower_cm1",
        ])

    df = df.iloc[valid].reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)

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
    V_u = pd.to_numeric(df["v'"], errors="coerce").astype(float)
    V_l = pd.to_numeric(df["v''"], errors="coerce").astype(float)

    out["upper_id"] = [
        f"{str(es).strip().upper()}|v={int(round(v))}|J={J:.6g}|sym={s}"
        for es, v, J, s in zip(df["eS'"], V_u, J_u, sym_u)
    ]
    out["lower_id"] = [
        f"{'X' if str(es).strip().upper().startswith('X') else str(es).strip().upper()}|"
        f"v={int(round(v))}|J={J:.6g}|sym={s}"
        for es, v, J, s in zip(df["eS''"], V_l, J_l, sym_l)
    ]

    out["g_upper"] = 2.0 * J_u + 1.0
    out["g_lower"] = 2.0 * J_l + 1.0

    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8
    out["E_cm1"] = 1.0 / lam_cm

    out["lower_es"] = df["eS''"].astype(str).str.strip().str.upper()
    out["lower_v"] = V_l
    out["lower_J"] = J_l
    out["lower_sym"] = np.asarray(sym_l, dtype=str)
    out["E_lower_cm1"] = pd.to_numeric(df[E_lower_col], errors="coerce").astype(float)

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


def load_default_transitions(
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
        matched = False
        #check condition that iso is XCYN
        if re.match(r"^\d+C\d+N$", iso):
            matched = True
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
        if 'Fe' in iso:
            matched = True
            PACKAGE_DIR = Path(__file__).resolve().parent
            DATA_DIR = PACKAGE_DIR / "data"
            path = DATA_DIR / 'fe_normalized.csv'
            tab = pd.read_csv(path)
            tab = tab[tab['A_ul'] > A_min]
            out[iso] = _drop_invalid_normalized_rows(tab, label=iso)
        #check if is XCYC or XCX
        if re.match(r"^\d+C\d+C$", iso) or re.match(r"^\d+C\d+$", iso):
            matched = True
            KEY_LINES = "/lines"
            PACKAGE_DIR = Path(__file__).resolve().parent
            DATA_DIR = PACKAGE_DIR / "data"
            canon = canonical_diatomic_name(iso) or iso
            path = DATA_DIR / f'C2/{canon}.h5'
            if not path.exists():
                # Fall back to the raw label for backward compatibility.
                path = DATA_DIR / f'C2/{iso}.h5'
            tab = pd.read_hdf(path, key=KEY_LINES)
            tab = tab[tab['A_ul'] > A_min]
            out[iso] = _drop_invalid_normalized_rows(tab, label=iso)
        if re.match(r"^\d+C\d*H$", iso) or re.match(r"^CH$", iso):
            matched = True
            KEY_LINES = "/lines"
            PACKAGE_DIR = Path(__file__).resolve().parent
            DATA_DIR = PACKAGE_DIR / "data"
            canon = canonical_diatomic_name(iso) or iso
            path = DATA_DIR / f'CH/{canon}.h5'
            if not path.exists():
                # Fall back to the raw label for backward compatibility.
                path = DATA_DIR / f'CH/{iso}.h5'
            tab = pd.read_hdf(path, key=KEY_LINES)
            tab = tab[tab['A_ul'] > A_min]
            #pass lambda_vac_A to trully the air wavelengths in the CH linelist
            out[iso] = _drop_invalid_normalized_rows(tab, label=iso)
        if not matched:
            raise ValueError(
                f"No default linelist available for isotopologue {iso!r}. "
                "Supported defaults are CN-like labels (e.g. '12C14N'), "
                "C2-like labels (e.g. '12C2', '12C13C'), or labels containing "
                "'Fe'. Provide a custom linelist via the `linelists` argument."
            )
    return out


def resolve_linelists_with_defaults(
    linelists: pd.DataFrame | dict[str, pd.DataFrame] | Sequence[pd.DataFrame] | None,
    iso_list: Sequence[str],
    *,
    systems: str | Sequence[str] | None = None,
    A_min: float = 1e4,
    lambda_min_A: float = 2990.001,
    lambda_max_A: float = 10009.998,
    use_omega_labels: bool = False,
    line_paths: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Resolve user-supplied linelists, filling in defaults for any missing isotopologues.

    Resolution rules:

    - ``linelists is None`` -> every iso loaded from packaged defaults via
      :func:`load_default_transitions`.
    - Single :class:`pandas.DataFrame` -> assigned to ``iso_list[0]``; the
      remaining isotopologues fall back to defaults.
    - :class:`dict` mapping iso label to DataFrame -> entries used for matching
      labels in ``iso_list``; any iso label not present in the dict falls back
      to defaults. Keys not in ``iso_list`` are ignored.
    - Sequence (``list``/``tuple``) of DataFrames -> positional pairing with the
      first ``len(linelists)`` entries of ``iso_list``; the remainder fall back
      to defaults.

    Loading a default for an isotopologue without a packaged file (e.g. ``"COH"``)
    raises :class:`ValueError` from :func:`load_default_transitions`.

    :returns: ``{iso: DataFrame}`` ordered exactly as ``iso_list``.
    :rtype: dict[str, pandas.DataFrame]
    """
    iso_list = list(iso_list)
    if not iso_list:
        raise ValueError("isotopologues is empty.")

    user_by_iso: dict[str, pd.DataFrame] = {}
    if linelists is None:
        pass
    elif isinstance(linelists, pd.DataFrame):
        user_by_iso[iso_list[0]] = linelists
    elif isinstance(linelists, dict):
        for iso in iso_list:
            if iso in linelists:
                user_by_iso[iso] = linelists[iso]
    elif isinstance(linelists, (list, tuple)):
        if len(linelists) > len(iso_list):
            raise ValueError(
                f"Got {len(linelists)} linelists for {len(iso_list)} isotopologues; "
                "too many."
            )
        for iso, df in zip(iso_list, linelists):
            user_by_iso[iso] = df
    else:
        raise TypeError(
            "linelists must be None, a DataFrame, a dict keyed by isotopologue, "
            f"or a sequence of DataFrames; got {type(linelists).__name__}."
        )

    missing = [iso for iso in iso_list if iso not in user_by_iso]
    if missing:
        defaults = load_default_transitions(
            isotopologues=missing,
            systems=systems,
            A_min=A_min,
            lambda_min_A=lambda_min_A,
            lambda_max_A=lambda_max_A,
            use_omega_labels=use_omega_labels,
            line_paths=line_paths,
        )
        for iso in missing:
            user_by_iso[iso] = defaults[iso]

    return {iso: user_by_iso[iso] for iso in iso_list}


def default_linelist_source(iso: str) -> str:
    """Return the file path that would be loaded for ``iso`` from packaged defaults.

    :raises ValueError: If ``iso`` does not match any supported default pattern
        (CN-like, C2-like, or containing ``"Fe"``).
    """
    PACKAGE_DIR = Path(__file__).resolve().parent
    DATA_DIR = PACKAGE_DIR / "data"
    if re.match(r"^\d+C\d+N$", iso):
        try:
            return str(helper.get_default_mol_linelist_path(isotope=iso))
        except TypeError:
            return str(helper.get_default_mol_linelist_path())
    if 'Fe' in iso:
        return str(DATA_DIR / 'fe_normalized.csv')
    if re.match(r"^\d+C\d+C$", iso) or re.match(r"^\d+C\d+$", iso):
        canon = canonical_diatomic_name(iso) or iso
        path = DATA_DIR / f'C2/{canon}.h5'
        if not path.exists():
            path = DATA_DIR / f'C2/{iso}.h5'
        return str(path)
    raise ValueError(
        f"No default linelist available for isotopologue {iso!r}."
    )


def linelist_origins(
    linelists: pd.DataFrame | dict[str, pd.DataFrame] | Sequence[pd.DataFrame] | None,
    iso_list: Sequence[str],
    *,
    line_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a per-isotopologue origin string for the configured line lists.

    Mirrors the resolution rules of :func:`resolve_linelists_with_defaults`:

    - Entries supplied by the user (DataFrame, dict entry, or positional list
      slot) are reported as ``"custom (user-provided)"``.
    - Entries with an explicit override in ``line_paths`` are reported as that
      path.
    - Otherwise the path returned by :func:`default_linelist_source` is used.

    Does not load any data.
    """
    iso_list = list(iso_list)

    user_isos: set[str] = set()
    if linelists is None:
        pass
    elif isinstance(linelists, pd.DataFrame):
        if iso_list:
            user_isos.add(iso_list[0])
    elif isinstance(linelists, dict):
        user_isos = {iso for iso in iso_list if iso in linelists}
    elif isinstance(linelists, (list, tuple)):
        user_isos = set(iso_list[: len(linelists)])
    else:
        raise TypeError(
            "linelists must be None, a DataFrame, a dict keyed by isotopologue, "
            f"or a sequence of DataFrames; got {type(linelists).__name__}."
        )

    out: dict[str, str] = {}
    for iso in iso_list:
        if iso in user_isos:
            out[iso] = "custom (user-provided)"
        elif line_paths is not None and iso in line_paths:
            out[iso] = str(line_paths[iso])
        else:
            out[iso] = default_linelist_source(iso)
    return out


def _drop_invalid_normalized_rows(tab: pd.DataFrame, *, label: str = "") -> pd.DataFrame:
    """Drop rows of an already-normalized linelist that have missing/invalid values.

    Numeric columns must be finite; string/id columns must be non-null. Wavelength
    and degeneracy columns must additionally be > 0; A_ul must be >= 0. Lines that
    fail any check are dropped (with a warning) instead of silently propagating
    NaN into the rate matrix or the collision scaffold.
    """
    if tab is None or len(tab) == 0:
        return tab.reset_index(drop=True) if tab is not None else tab

    numeric_required = {
        "lambda_vac_A": ("positive", True),
        "A_ul": ("nonneg", True),
        "g_upper": ("positive", False),
        "g_lower": ("positive", False),
        "lower_v": ("finite", False),
        "lower_J": ("finite", False),
        "E_lower_cm1": ("finite", False),
    }
    string_required = ("upper_id", "lower_id", "lower_es", "lower_sym")

    valid = pd.Series(True, index=tab.index)
    for col, (kind, _) in numeric_required.items():
        if col not in tab.columns:
            continue
        s = pd.to_numeric(tab[col], errors="coerce")
        m = np.isfinite(s)
        if kind == "positive":
            m = m & (s > 0)
        elif kind == "nonneg":
            m = m & (s >= 0)
        valid &= m
    for col in string_required:
        if col in tab.columns:
            valid &= tab[col].notna()

    valid_arr = valid.to_numpy()
    n_dropped = int((~valid_arr).sum())
    if n_dropped > 0:
        warnings.warn(
            f"load_default_transitions[{label}]: dropping {n_dropped} row(s) with missing/invalid values."
        )
    return tab.iloc[valid_arr].reset_index(drop=True)


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

    # Drop lines whose (shifted) wavelength falls outside the pumping grid.
    in_range = (lam >= wave_AA.min()) & (lam <= wave_AA.max())
    df = df.reset_index(drop=True)[in_range]
    lam = lam[in_range]

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


# (mass_number, element) for common astrophysical isotopes with nonzero nuclear
# spin. Used to detect whether an isotopologue has hyperfine structure, which
# opens rotationally-elastic (Delta J = 0) collisional channels between
# hyperfine sublevels. Anything not in this set is assumed I=0.
_NONZERO_SPIN_ISOTOPES: set[tuple[int, str]] = {
    (1, "H"), (2, "H"),
    (13, "C"),
    (14, "N"), (15, "N"),
    (17, "O"),
    (19, "F"),
    (33, "S"),
}


def _has_nonzero_nuclear_spin(iso_name: str | None) -> bool:
    if iso_name is None:
        return False
    atoms = _parse_isotopologue_atoms(iso_name)
    return any(a in _NONZERO_SPIN_ISOTOPES for a in atoms)


def canonical_diatomic_name(iso_name: str | None) -> str | None:
    """Return a canonical label for a diatomic isotopologue.

    Atoms are sorted by (mass, element) so that labels like ``"13C12C"`` and
    ``"12C13C"`` collapse to the same canonical form (``"12C13C"``). Homonuclear
    labels with subscripts (e.g. ``"12C2"``) are preserved. If the label cannot
    be parsed as a diatomic, it is returned unchanged.
    """
    if iso_name is None:
        return None
    atoms = _parse_isotopologue_atoms(iso_name)
    if len(atoms) != 2:
        return iso_name
    (m1, e1), (m2, e2) = atoms
    if (m1, e1) == (m2, e2):
        # Homonuclear: keep compact "<mass><El>2" form.
        return f"{m1}{e1}2"
    a, b = sorted(atoms, key=lambda x: (x[0], x[1]))
    return f"{a[0]}{a[1]}{b[0]}{b[1]}"


def _parse_isotopologue_atoms(name: str) -> list[tuple[int, str]]:
    """Parse an isotopologue label like ``"12C2"``, ``"12C13C"``, or ``"12C14N"`` into atoms.

    :param name: Isotopologue label using ``<mass><Element>[count]`` tokens
        (e.g., ``"12C2"`` -> two ``(12, "C")``; ``"12C13C"`` -> ``(12, "C")`` and
        ``(13, "C")``; ``"12C14N"`` -> ``(12, "C")`` and ``(14, "N")``).
    :type name: str
    :returns: List of ``(mass_number, element)`` tuples. Empty if ``name`` is
        unparseable.
    :rtype: list[tuple[int, str]]
    """
    s = str(name)
    matches = list(re.finditer(r"(\d+)([A-Z][a-z]?)", s))
    atoms: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        mass = int(m.group(1))
        elem = m.group(2)
        # Subscript count is the digits between this match's end and the next
        # match's start (or end of string). Avoids gobbling the next atom's mass.
        end = m.end()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        count_match = re.match(r"\d+", s[end:next_start])
        count = int(count_match.group()) if count_match else 1
        atoms.extend([(mass, elem)] * count)
    return atoms


def is_homonuclear_diatomic(iso_name: str | None) -> bool:
    """Detect whether an isotopologue label denotes a homonuclear diatomic.

    Homonuclear means the same element AND the same isotope on both atoms
    (e.g., ``"12C2"``, ``"13C2"``). The mixed-isotope ``"12C13C"`` is treated as
    heteronuclear because the broken nuclear-permutation symmetry removes the
    even/odd-J restriction.

    :param iso_name: Isotopologue label, or ``None``.
    :type iso_name: str or None
    :returns: ``True`` only for diatomics with two identical ``(mass, element)``
        atoms; ``False`` otherwise (including unparseable or non-diatomic labels).
    :rtype: bool
    """
    if iso_name is None:
        return False
    atoms = _parse_isotopologue_atoms(iso_name)
    if len(atoms) != 2:
        return False
    return atoms[0] == atoms[1]


def diatomic_symmetry_class(iso_name: str | None) -> str:
    """Classify a diatomic isotopologue label for collision selection rules.

    :param iso_name: Isotopologue label such as ``"12C2"``, ``"12C13C"``,
        ``"12C14N"``, or ``None``.
    :type iso_name: str or None
    :returns: One of:

        * ``"homonuclear"`` -- two identical ``(mass, element)`` atoms (e.g.,
          ``"12C2"``, ``"13C2"``). Only ``|Delta J| = 2`` collisions are physical
          (nuclear-spin manifold preserved).
        * ``"same_element_heteronuclear"`` -- same element, different isotopes
          (e.g., ``"12C13C"``, ``"1H2H"``). Symmetry is broken so all J exist,
          but the underlying near-symmetric structure makes both ``|Delta J| = 1``
          and ``|Delta J| = 2`` channels relevant.
        * ``"heteronuclear"`` -- different elements (e.g., ``"12C14N"``,
          ``"16O1H"``). Treat as a generic heteronuclear with the caller-provided
          ``dJ_allowed`` (defaults to ``|Delta J| = 1`` only).
        * ``"unknown"`` -- label could not be parsed as a diatomic.
    :rtype: str
    """
    atoms = _parse_isotopologue_atoms(iso_name) if iso_name else []
    if len(atoms) != 2:
        return "unknown"
    (m1, e1), (m2, e2) = atoms
    if (m1, e1) == (m2, e2):
        return "homonuclear"
    if e1 == e2:
        return "same_element_heteronuclear"
    return "heteronuclear"


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

    # Generalized molecule handling
    iso_name: str | None = None,
    homonuclear: bool | None = None,
    dJ_allowed: Sequence[int] = (1,),
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
    :param include_deltaJ0_parity_mix: Allow ``Delta J = 0`` collisions between
        sublevels with different ``sym`` label at the same J. Fires when either
        the molecule is truly heteronuclear (different elements, e.g. CN, OH) OR
        the isotopologue has at least one nucleus with nonzero spin (hyperfine
        structure, e.g. 13C2, 12C13C). Ignored only for strictly zero-hyperfine
        homonuclear species like 12C2.
    :type include_deltaJ0_parity_mix: bool
    :param require_X_only: Restrict collisions to the ground electronic state. The
        ground state is auto-detected as the ``lower_es`` label whose minimum
        ``E_lower_cm1`` is smallest, so any spectroscopic notation works
        (``"X"``, ``"X1Sigmag+"``, etc.).
    :type require_X_only: bool
    :param iso_name: Optional isotopologue label (e.g., ``"12C2"``, ``"13C2"``,
        ``"12C13C"``, ``"12C14N"``) used to auto-classify the diatomic via
        :func:`diatomic_symmetry_class`. Drives the ``|Delta J|`` rule:

        * homonuclear -> ``{2}`` only
        * same-element heteronuclear (e.g., ``"12C13C"``) -> ``{1, 2}``
        * heteronuclear different elements (e.g., ``"12C14N"``) -> uses
          ``dJ_allowed``

        Ignored when ``homonuclear`` is given explicitly.
    :type iso_name: str or None
    :param homonuclear: Explicit override for nuclear symmetry. ``True`` forces
        ``|Delta J| = 2`` only and disables ``include_deltaJ0_parity_mix`` (preserves
        nuclear-spin manifold). ``False`` uses ``dJ_allowed``. ``None`` auto-detects
        from ``iso_name``.
    :type homonuclear: bool or None
    :param dJ_allowed: Allowed ``|Delta J|`` values for the heteronuclear
        different-element case (e.g., CN). Defaults to ``(1,)`` to match historical
        CN behavior. Ignored for homonuclear (forced to ``(2,)``) and for
        same-element heteronuclear (forced to ``(1, 2)``).
    :type dJ_allowed: Sequence[int]
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

    if require_X_only and lower_ids.size > 0:
        # Auto-detect the ground electronic state as the lower_es label whose
        # minimum E_lower_cm1 is smallest. Notation-agnostic ("X", "X1Sigmag+", etc.).
        es_strs = np.array([str(es).strip() for es in les])
        unique_es = np.unique(es_strs)
        es_min_E = {es: float(np.min(Elow[es_strs == es])) for es in unique_es}
        ground_es = min(es_min_E, key=es_min_E.get)
        m_ground = es_strs == ground_es
        lower_ids, les, lv, lJ, lsym, Elow = (
            lower_ids[m_ground], les[m_ground], lv[m_ground], lJ[m_ground], lsym[m_ground], Elow[m_ground]
        )

    if lower_ids.size == 0:
        return _empty_scaffold()

    # Resolve molecule class: explicit homonuclear override wins; otherwise classify
    # from iso_name. Three branches:
    #   - homonuclear (e.g., 12C2): |dJ|=2 only, no parity-mix (nuclear-spin conserved)
    #   - same-element heteronuclear (e.g., 12C13C): |dJ|=1 and 2
    #   - other heteronuclear (e.g., 12C14N): caller's dJ_allowed (default {1})
    if homonuclear is True:
        sym_class = "homonuclear"
    elif homonuclear is False:
        sym_class = "heteronuclear"
    else:
        sym_class = diatomic_symmetry_class(iso_name)

    
    if sym_class == "homonuclear":
        dJ_set = {2}
    elif sym_class == "same_element_heteronuclear":
        dJ_set = {1, 2}
    else:
        dJ_set = {int(d) for d in dJ_allowed}

    # Hyperfine structure (any nucleus with I>0) opens Delta J = 0 channels
    # between hyperfine sublevels sharing the same J but different sym label,
    # regardless of homonuclear vs heteronuclear.
    has_hyperfine = _has_nonzero_nuclear_spin(iso_name)

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
            allow = (int(dJ) in dJ_set) or (
                (sym_class == "heteronuclear" or has_hyperfine)
                and include_deltaJ0_parity_mix
                and dJ == 0
                and str(sa) != str(sb)
            )
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
    linelists: pd.DataFrame | dict[str, pd.DataFrame] | Sequence[pd.DataFrame] | None = None,

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

    A_min: Optional[float] = 1e4,
    a: float = 3,
    threads: int = 1,

    # NOTE: these control *pumping* wavelength shift for J_nu (radiative rates)
    velocity_kms: float = 0.0,
    delta_lambda_A: float = 0.0,

    # NOTE: these are fallbacks for parameters not present in priors
    init_logQ: Optional[float] = None,
    init_logQ_by_iso: Optional[Dict[str, Optional[float]]] = None,
    init_T: float = 300.0,
    init_v_kms: float = 0.0,
    init_dlam: float = 0.0,
    init_logN: Optional[float] = None,
    init_logN_by_iso: Optional[Dict[str, float]] = None,
    init_sigma: Optional[float] = None,
    init_sigma1: Optional[float] = None,
    init_sigma2: Optional[float] = None,
    init_sigma_G: Optional[float] = None,
    init_fwhm_L: Optional[float] = None,
    init_ratio: Optional[float] = None,


    fig_file: Optional[str] = None,
    wave_col: str = "WAVE",
    flux_col: str = "FLUX_STACK",
    error_col: str = "ERR_STACK",
    continuum_col: str = "CONTINUUM",
    omega: Optional[float] = None,
    verbose: bool = True,
    pruning: bool = True,
    N_Model: Optional[int] = 20000,
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
        - Collisions are gated by ``logQ``: if neither a per-iso ``logQ_{iso}`` nor a
            shared ``logQ`` prior is given, and neither ``init_logQ`` nor an entry in
            ``init_logQ_by_iso`` is provided for an isotopologue, that isotopologue
            is treated as collisionless. Other collision controls default to
            ``include_deltaJ0_parity_mix=True`` and ``require_X_only_for_rot=True``.
        - Pumping-shift controls default to ``velocity_kms=0.0`` and
            ``delta_lambda_A=0.0``.
        - Fallback parameter values (used when the parameter is not sampled) default
            to ``init_logQ=None`` (no shared collision rate), ``init_T=300.0``,
            ``init_v_kms=0.0``, and ``init_dlam=0.0``.

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
    :param include_deltaJ0_parity_mix: Allow parity-changing ``Delta J = 0`` collisions.
    :type include_deltaJ0_parity_mix: bool
    :param require_X_only_for_rot: Restrict collisions to the ground electronic
        state (auto-detected as the ``lower_es`` label with the smallest minimum
        ``E_lower_cm1``; works for any spectroscopic notation).
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
    :param init_logQ: Fallback ``logQ`` value used by every isotopologue when no
        ``logQ`` prior is sampled and no per-iso entry is given. ``None`` (default)
        disables collisions for any isotopologue not covered by ``init_logQ_by_iso``.
    :type init_logQ: float or None
    :param init_logQ_by_iso: Per-isotopologue fallback ``logQ`` map. Each value may
        be ``None`` to force that isotopologue to be collisionless. Isotopologues
        not present in the map fall back to ``init_logQ``.
    :type init_logQ_by_iso: dict[str, float or None] or None
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
    :param N_Model: Number of elements in the model grid.
    :type N_Model: int
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

    # ---------- 1) Line lists: defaults or user-provided (with default fallback) ----------
    trans_by_iso = resolve_linelists_with_defaults(
        linelists,
        iso_list,
        systems=sys_tokens,
        A_min=A_min,
        use_omega_labels=False,
    )
    if linelists is not None and A_min is not None:
        # User-provided frames may not yet be A_min-filtered; defaults already are.
        trans_by_iso = {
            iso: df[df["A_ul"] >= A_min].reset_index(drop=True)
            for iso, df in trans_by_iso.items()
        }

    # ---------- Decide per-iso whether collisions can ever fire ----------
    # An isotopologue gets collisions only if logQ is reachable for it: either
    # via a sampled prior (per-iso "logQ_{iso}" or shared "logQ"), or via a
    # finite fallback (init_logQ_by_iso[iso] or init_logQ). A None fallback or
    # an absent entry disables collisions for that iso entirely.
    def _iso_can_collide(iso: str) -> bool:
        if f"logQ_{iso}" in priors:
            return True
        if "logQ" in priors:
            return True
        if init_logQ_by_iso is not None and iso in init_logQ_by_iso:
            return init_logQ_by_iso[iso] is not None
        return init_logQ is not None

    iso_collides = {iso: _iso_can_collide(iso) for iso in trans_by_iso.keys()}

    # Enforce rotational columns only for isotopologues that will use collisions.
    req = {"lower_es", "lower_v", "lower_J", "lower_sym", "E_lower_cm1"}
    for iso, df_trans in trans_by_iso.items():
        if not iso_collides[iso]:
            continue
        missing = sorted(list(req - set(df_trans.columns)))
        if missing:
            raise ValueError(
                f"Isotopologue {iso!r} would use collisions (logQ provided) but its linelist "
                f"is missing required columns: {missing}. Provide them via "
                "from_user_linelist(... lower_*_col=..., E_lower_cm1_col=...), or set its "
                "logQ to None to disable collisions for this isotopologue."
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

        if iso_collides[iso]:
            coll_scaf = precompute_cn_collision_scaffold_fast(
                lines_out, idx_to_level,
                include_deltaJ0_parity_mix=include_deltaJ0_parity_mix,
                require_X_only=require_X_only_for_rot,
                iso_name=iso,
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

    def _logQ_for_iso(iso: str, pars: Dict[str, float]) -> float:
        # Per-iso prior wins, then shared "logQ" prior, then per-iso init, then global init.
        key = f"logQ_{iso}"
        if key in pars:
            return float(pars[key])
        if "logQ" in pars:
            return float(pars["logQ"])
        if init_logQ_by_iso is not None and iso in init_logQ_by_iso:
            try:
                return float(init_logQ_by_iso[iso])
            except TypeError:
                return None
        if init_logQ is not None:
            return float(init_logQ)
        return None

    def model_flux(theta: Sequence[float], wave: np.ndarray) -> np.ndarray:
        pars = theta_to_params(theta)
        wmin = float(np.min(wave))
        wmax = float(np.max(wave))

        # ✅ Correct fallback behavior:
        # If not being fit, use init_* (provided by caller), not magic hardcoded defaults.
        try:
            T = float(pars["T"]) if "T" in pars else float(init_T)
        except TypeError:
            T = None
        try:
            v_kms = float(pars["v_kms"]) if "v_kms" in pars else float(init_v_kms)
        except TypeError:
            v_kms = None
        try:
            dlam = float(pars["dlam"]) if "dlam" in pars else float(init_dlam)
        except TypeError:
            dlam = None
        try:
            sigma = float(pars["sigma"]) if "sigma" in pars else float(init_sigma)
        except TypeError:
            sigma = None
        try:
            sigma1 = float(pars["sigma1"]) if "sigma1" in pars else float(init_sigma1)
        except TypeError:
            sigma1 = None
        try:
            sigma2 = float(pars["sigma2"]) if "sigma2" in pars else float(init_sigma2)
        except TypeError:
            sigma2 = None
        try:
            sigma_G = float(pars["sigma_G"]) if "sigma_G" in pars else float(init_sigma_G)
        except TypeError:
            sigma_G = None
        try:
            fwhm_L = float(pars["fwhm_L"]) if "fwhm_L" in pars else float(init_fwhm_L)
        except TypeError:
            fwhm_L = None
        try:
            ratio = float(pars["ratio"]) if "ratio" in pars else float(init_ratio)
        except TypeError:
            ratio = None

        dict_for_lsf = {'sigma': sigma, 'sigma1': sigma1, 'sigma2': sigma2, 'sigma_G': sigma_G, 'fwhm_L': fwhm_L, 'ratio': ratio}
        lsf_fun = make_lsf_local(dict_for_lsf)

        spec_total = np.zeros_like(wave, dtype=float)

        use_reuse = (threads == 1)   # threads is captured from outer scope

        for iso in iso_list:
            C = cache[iso]
            if use_reuse:
                M = C["M_work"]
                np.copyto(M, C["M_rad"])
            else:
                M = C["M_rad"].copy()

            logQ_i = _logQ_for_iso(iso, pars)
            
            if logQ_i is not None:
                Q_i = 10.0 ** logQ_i if np.isfinite(logQ_i) else 0.0
                if Q_i > 0.0 and include_rotations:
                    apply_collisions_inplace_fast(M, C["coll_scaf"], Q=Q_i, T=T, Cup_work=C["Cup_work"])

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
                try:
                    logN_i = float(pars["logN"]) if "logN" in pars else float(init_logN)
                except TypeError:
                    logN_i = None
            else:
                key = f"logN_{iso}"
                if key not in pars:
                    raise ValueError(f"Missing parameter {key!r} in priors for multi-isotopologue fit.")
                try:
                    logN_i = float(pars[key]) if key in pars else float(init_logN_by_iso.get(iso, init_logN))
                except TypeError:
                    logN_i = None

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

        T = float(pars["T"]) if "T" in pars else float(init_T)
        v_kms = float(pars["v_kms"]) if "v_kms" in pars else float(init_v_kms)
        dlam = float(pars["dlam"]) if "dlam" in pars else float(init_dlam)

        lsf_fun = make_lsf_local(pars)

        spec_total = np.zeros_like(wave, dtype=float)

        for iso in iso_list:
            C = cache[iso]
            M = C["M_work"]
            np.copyto(M, C["M_rad"])

            logQ_i = _logQ_for_iso(iso, pars)
            if logQ_i is not None:
                Q_i = 10.0 ** logQ_i if np.isfinite(logQ_i) else 0.0
                if Q_i > 0.0 and include_rotations:
                    apply_collisions_inplace_fast(M, C["coll_scaf"], Q=Q_i, T=T, Cup_work=C["Cup_work"])

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
    x_model = np.linspace(window[0], window[1], N_Model)
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
        "logN": r"$\log_{10}(N$ / [molecules cm$^{-2}$])",
        "logQ": r"$\log_{10}$(Q$_{\rm{col}}$ / [s$^{-1}$])",
        "T": r"$T_{kin}$ [K]",
        "v_kms": r"$\Delta$v [km s$^{-1}$]",
        "dlam": r"$\Delta \lambda$ [Å]",
        "sigma": r"$\sigma$ [Å]",
        "sigma1": r"$\sigma_1$ [Å]",
        "sigma2": r"$\sigma_2$ [Å]",
        "sigma_G": r"$\sigma_G$ [Å]",
        "fwhm_L": r"FWHM$_L$ [Å]",
        "ratio": r"Ratio",
        }
    if iso_list and len(iso_list) > 1:

        param_lab_by_iso = {f'{k}_{iso}': f"{param_labels[k]}_{iso}" for k in param_labels for iso in iso_list}
        # join both param_labels and param_lab_by_iso in one dict
        param_labels = {**param_labels, **param_lab_by_iso}
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
        fig = corner.corner(
            samples_pruned,
            labels=[param_labels[k] for k in param_keys],
            title_kwargs={'y':1.05},title_fmt=".2f",use_math_text=True,bins=15,quantiles=[0.16, 0.5, 0.84],show_titles=True,
                color='lightseagreen',hist_kwargs={'color':'black','linewidth':1.5},contour_kwargs={'linewidths':1,'colors':'black'}, spacing=0.001, label_kwargs={'fontsize': 13},
        )
        elements = [param_labels[k] for k in param_keys]
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        ndim = samples.shape[1]
        axes = np.array(fig.axes).reshape((ndim, ndim))

        for i in range(ndim):
                ax = axes[i, i]
                q16, q50, q84 = np.quantile(samples[:, i], [0.16, 0.5, 0.84])
                q_minus, q_plus = q50 - q16, q84 - q50
                title = (
                        f"{elements[i]}\n"
                        rf"=${q50:.2f}_{{-{q_minus:.2f}}}^{{+{q_plus:.2f}}}$"
                )
                ax.set_title(title, fontsize=11, y=1.05)
        for ax in fig.get_axes():
                ax.tick_params(axis='both', labelsize=10)

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
    