from __future__ import annotations

"""
Core fluorescence modeling (normalized transition schema).

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
     by mapping from the Brooke file (not included by defA_ult).

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
    """Return column `name` from Table/DataFrame/dict-like as a NumPy array."""
    if hasattr(obj, "colnames"):        # astropy Table
        return np.asarray(obj[name])
    if hasattr(obj, "columns"):         # pandas DataFrame
        return np.asarray(obj[name].values)
    return np.asarray(obj[name])


def normalize_systems_arg(systems: str | Sequence[str] | None) -> list[str]:
    """
    Normalize system selection strings to internal tokens.

    Supported tokens:
      - "BX00"     : B–X (0,0)
      - "AX_dv1"   : A–X with Δv=+1
      - "ALL"      : no CN system filtering
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
    """
    Convert a user-provided transition line list into the normalized schema.

    Required input columns:
      lam_col [Å], A_col [s^-1], upper_id_col, lower_id_col, g_upper_col, g_lower_col

    Output columns:
      lambda_vac_A, A_ul, upper_id, lower_id, g_upper, g_lower,
      E_cm1 (=1/lambda(cm))  [cm^-1]  (photon wavenumber)

    Optional extra columns for rotational collisions:
      lower_es, lower_v, lower_J, lower_sym, E_lower_cm1
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
    """Return compact symmetry tag (CN-style)."""
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
    """
    Convert a Brooke CN line list into the normalized transition schema.

    Output columns
    --------------
      lambda_vac_A
      A_ul
      upper_id
      lower_id
      g_upper
      g_lower
      E_cm1          (transition wavenumber = 1/lambda(cm))

    Plus (for rotational collisions if include_rotations=True)
    ---------------------------------------------------------
      lower_es
      lower_v
      lower_J
      lower_sym
      E_lower_cm1
    """
    out = pd.DataFrame(index=df.index)

    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)

    # --- symmetry labels (your existing logic) ---
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

    # --- IDs (can stay Brooke-style; collisions won't parse them) ---
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

    # --- photon wavenumber (cm^-1) ---
    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8  # Å -> cm
    if np.any(~np.isfinite(lam_cm)) or np.any(lam_cm <= 0.0):
        raise ValueError("Invalid lambda_vac_A values in CN linelist.")
    out["E_cm1"] = 1.0 / lam_cm

    # --- REQUIRED for include_rotations=True ---
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

    # --- validation ---
    if np.any(~np.isfinite(out["A_ul"])) or np.any(out["A_ul"] < 0.0):
        raise ValueError("Invalid A_ul values in CN linelist.")
    if np.any(~np.isfinite(out["g_upper"])) or np.any(out["g_upper"] <= 0.0):
        raise ValueError("Invalid g_upper values in CN linelist.")
    if np.any(~np.isfinite(out["g_lower"])) or np.any(out["g_lower"] <= 0.0):
        raise ValueError("Invalid g_lower values in CN linelist.")

    # for collisions, these must be finite too
    bad_E = ~np.isfinite(out["E_lower_cm1"])
    if np.any(bad_E):
        # keep this strict so you fail early (otherwise collisions explode later)
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
    """
    CN Brooke-specific filtering by electronic system + vibrational selection.
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
    """
    Returns dict: iso -> normalized transition dataframe (schema from_cn_brooke).
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
    """
    Attach pumping field (J_nu) and wavelength/frequency columns.
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
    """
    Build radiative rate matrix M from normalized transition schema.
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
    """
    Build rotational collision scaffold from explicit LOWER-state columns.

    Output scaffold:
      iu, il : matrix indices for collision-connected levels (iu is higher-energy)
      gu, gl : degeneracies for those levels (2J+1)
      dE     : energy gap in cm^-1 (positive)
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
    """
    Wrapper around precompute_cn_collision_scaffold that adds precomputed arrays
    to avoid astropy and repeated conversions in the inner loop.

    Adds:
      - dE_over_k_K : (h c / kB) * dE_cm1, in Kelvin
      - gu_over_gl  : gu/gl
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
    """
    Modify M in place with collisions.

    Scaffold expects:
      dE in cm^-1 (positive)
    """
    if scaffold.get("iu", np.array([])).size == 0 or Q <= 0:
        return M

    iu = scaffold["iu"]
    il = scaffold["il"]
    gu = scaffold["gu"]
    gl = scaffold["gl"]
    dE_cm1 = scaffold["dE"]

    kT = (const.k_B * (T * u.K)).to(u.erg).value

    # cm^-1 -> erg : (h*c)*(wavenumber)
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
    """
    Fast collisions, no allocations:
    - uses scaffold["dE_over_k_K"] (Kelvin) and scaffold["gu_over_gl"]
    - writes Cup into Cup_work
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

    # Cdown is scalar Q (same as before)
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
    """Solve M @ n = 0 with Σ n_i = 1."""
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
    """
    Solve M @ n = 0 with sum(n) = 1
    Reuses buffers, no allocation.
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
    """Return per-line (g_phot, g_energy, Σg_phot, Σg_energy)."""
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
    """
    Fast version of g_factors using precomputed arrays.
    Returns (g_ph, g_en) arrays (same length as ui/A_ul/hnu).
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

    # keep behavior similar to old function (avoid NaNs/neg)
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
    """
    Build synthetic emission spectrum from per-line g-factors.
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
# MCMC fitting (multi-isotopologue + systems + defA_ults or user linelists)
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
    """
    MCMC fit of fluorescence in a wavelength window.

    Fixes included:
    - If logQ/T/v_kms/dlam are not in priors, they no longer defA_ult to magic
      numbers (-99/300/0/0). They fall back to init_* values supplied by caller.
    - Pumping (J_nu) is evaluated using velocity_kms/delta_lambda_A (as before).
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

    # ---------- 1) Line lists: defA_ults or user ----------
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
        "logN": r"log N [mol cm$^{-2}$]",
        "logQ": r"log Q [s$^{-1}$]",
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
        plt.ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
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