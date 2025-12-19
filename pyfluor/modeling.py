from __future__ import annotations

"""Core fluorescence modeling for :mod:`pyfluor`.

Includes:
- filters for A–X / B–X systems
- labeling of levels
- radiative rate matrix construction
- simple collisional coupling
- steady-state solution & g-factors
- spectrum synthesis
- FluorescenceModel container class
"""

from typing import Dict, Tuple, Sequence, Optional, Callable, Any, Union

import emcee
import corner
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from astropy import constants as const
from astropy import units as u
from astropy.table import Table

from . import helper


import numpy as np
import pandas as pd

# make the transitions provided into one format
def from_user_linelist(
    df: pd.DataFrame,
    *,
    lam_col: str,
    A_col: str,
    upper_id_col: str,
    lower_id_col: str,
    g_upper_col: str,
    g_lower_col: str,
) -> pd.DataFrame:
    """
    Convert a user-provided transition line list into the normalized schema.

    Assumptions
    -----------
    The input DataFrame provides:
      - lam_col        : vacuum wavelength [Å]
      - A_col          : Einstein A_ul [s^-1]
      - upper_id_col   : unique upper-level identifier (string)
      - lower_id_col   : unique lower-level identifier (string)
      - g_upper_col    : upper-level degeneracy
      - g_lower_col    : lower-level degeneracy

    Energies
    --------
    A single transition energy is computed:
      E_cm1 = h c / lambda = 1 / lambda(cm)

    Output columns
    --------------
      lambda_vac_A, A_ul,
      upper_id, lower_id,
      g_upper, g_lower,
      E_cm1
    """

    # ---- required columns ----
    required = [
        lam_col,
        A_col,
        upper_id_col,
        lower_id_col,
        g_upper_col,
        g_lower_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)

    # ---- core fields ----
    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)

    out["upper_id"] = df[upper_id_col].astype(str)
    out["lower_id"] = df[lower_id_col].astype(str)

    out["g_upper"] = pd.to_numeric(df[g_upper_col], errors="coerce").astype(float)
    out["g_lower"] = pd.to_numeric(df[g_lower_col], errors="coerce").astype(float)

    # ---- compute transition energy ----
    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8
    if np.any(~np.isfinite(lam_cm)) or np.any(lam_cm <= 0.0):
        raise ValueError("Invalid lambda_vac_A values.")

    out["E_cm1"] = 1.0 / lam_cm

    # ---- validation ----
    if np.any(~np.isfinite(out["A_ul"])) or np.any(out["A_ul"] < 0.0):
        raise ValueError("Invalid A_ul values.")

    if np.any(~np.isfinite(out["g_upper"])) or np.any(out["g_upper"] <= 0.0):
        raise ValueError("Invalid g_upper values.")

    if np.any(~np.isfinite(out["g_lower"])) or np.any(out["g_lower"] <= 0.0):
        raise ValueError("Invalid g_lower values.")

    return out

# make the transitions into one format form brooke 
def from_cn_brooke(
    df: pd.DataFrame,
    *,
    lam_col: str = "lambda_vac_A_from_Cal",
    A_col: str = "A",
    use_omega_labels: bool = False,
) -> pd.DataFrame:
    """
    Convert a Brooke CN line list into the normalized internal schema.

    Output columns
    --------------
      lambda_vac_A
      A_ul
      upper_id
      lower_id
      g_upper
      g_lower
      E_cm1        (transition energy = h c / lambda)
    """

    out = pd.DataFrame(index=df.index)

    # ---- core radiative quantities ----
    out["lambda_vac_A"] = pd.to_numeric(df[lam_col], errors="coerce").astype(float)
    out["A_ul"] = pd.to_numeric(df[A_col], errors="coerce").astype(float)

    # ---- symmetry labels (reuse existing logic) ----
    sym_u = [
        make_sym(F, p, use_omega_labels, es)
        for F, p, es in zip(df["F'"], df["p'"], df["eS'"])
    ]
    sym_l = [
        make_sym(F, p, use_omega_labels, es)
        for F, p, es in zip(df["F''"], df["p''"], df["eS''"])
    ]

    # ---- rotational quantum numbers (internal only) ----
    J_u = pd.to_numeric(df["J'"], errors="coerce").astype(float)
    J_l = pd.to_numeric(df["J''"], errors="coerce").astype(float)

    # ---- unique level IDs ----
    out["upper_id"] = [
        f"{str(es).strip().upper()}|v={int(round(v))}|J={J:.6g}|sym={s}"
        for es, v, J, s in zip(df["eS'"], df["v'"], J_u, sym_u)
    ]

    out["lower_id"] = [
        f"{'X' if str(es).strip().upper().startswith('X') else str(es).strip().upper()}|"
        f"v={int(round(v))}|J={J:.6g}|sym={s}"
        for es, v, J, s in zip(df["eS''"], df["v''"], J_l, sym_l)
    ]

    # ---- degeneracies (CN assumption: g = 2J + 1) ----
    out["g_upper"] = 2.0 * J_u + 1.0
    out["g_lower"] = 2.0 * J_l + 1.0

    # ---- transition energy (single value) ----
    lam_cm = out["lambda_vac_A"].to_numpy() * 1e-8
    if np.any(~np.isfinite(lam_cm)) or np.any(lam_cm <= 0.0):
        raise ValueError("Invalid lambda_vac_A values in CN linelist.")

    out["E_cm1"] = 1.0 / lam_cm

    # ---- final validation ----
    if np.any(~np.isfinite(out["A_ul"])) or np.any(out["A_ul"] < 0.0):
        raise ValueError("Invalid A_ul values in CN linelist.")

    if np.any(~np.isfinite(out["g_upper"])) or np.any(out["g_upper"] <= 0.0):
        raise ValueError("Invalid g_upper values in CN linelist.")

    if np.any(~np.isfinite(out["g_lower"])) or np.any(out["g_lower"] <= 0.0):
        raise ValueError("Invalid g_lower values in CN linelist.")

    return out

# ---------------------------------------------------------------------------
# Filters & labels
# ---------------------------------------------------------------------------

def filter_AX_BX(
    df_all: pd.DataFrame,
    *,
    lambda_min_A: float = 2990.001,
    lambda_max_A: float = 10009.998,
    A_min: Optional[float] = 1e4,
) -> pd.DataFrame:
    """Select CN B–X(0,0) and A–X(Δv=1) lines in given λ-range."""
    df = df_all.copy()
    lam_col = "lambda_vac_A_from_Cal"

    df = df[(df["eS'"].isin(["A", "B"])) & (df["eS''"] == "X")]

    mask_B00 = (df["eS'"] == "B") & (df["v'"] == 0) & (df["v''"] == 0)
    mask_A10 = (df["eS'"] == "A") & ((df["v'"] - df["v''"]) == 1)
    df = df[mask_B00 | mask_A10]

    df = df[(df[lam_col] >= lambda_min_A) & (df[lam_col] <= lambda_max_A)]

    if A_min is not None:
        df = df[df["A"] >= float(A_min)]

    return df.reset_index(drop=True)


def make_manifold(es: str, v: float) -> str:
    es = str(es).strip().upper()
    try:
        vint = int(round(float(v)))
    except Exception:
        vint = v
    if es.startswith("X"):
        return f"Ground X v={vint}"
    return f"{es} v={vint}"


def make_sym(F, p, use_omega: bool = False, es: Optional[str] = None) -> str:
    """Return compact symmetry tag."""
    ptag = str(p).strip().lower()[:1] if p not in (None, "") else "?"
    try:
        Fint = int(F)
    except Exception:
        Fint = F

    if use_omega and str(es).strip().upper().startswith("A"):
        comp = "Ω3/2" if Fint == 1 else "Ω1/2"
        return f"{comp}_{ptag}"

    return f"F{Fint}_{ptag}"

def _as_array(obj: Any, name: str) -> np.ndarray:
    """Return column `name` from Table/DataFrame/dict-like as a NumPy array."""
    if hasattr(obj, "colnames"):        # astropy Table
        return np.asarray(obj[name])
    if hasattr(obj, "columns"):         # pandas DataFrame
        return np.asarray(obj[name].values)
    return np.asarray(obj[name])   

def attach_pumping_and_labels(
    df: pd.DataFrame,
    pumping: Any,
    *,
    use_omega_labels: bool = False,
    line_v_kms: float = 0.0,
    line_dlam_A: float = 0.0,
    lsf_for_Jnu: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Table:
    """
    Attach pumping field & level labels to the CN lines.

    Parameters
    ----------
    df : DataFrame
        Filtered line list with 'lambda_vac_A_from_Cal'.
    pumping : Table/DataFrame/dict-like
        Must contain:
        - 'WAVE' [Å]
        - 'FLUX' [erg s^-1 cm^-2 Å^-1]
    use_omega_labels : bool
        If True, use Ω3/2 / Ω1/2 labels for A-state.
    line_v_kms : float
        Velocity shift [km/s] applied to the CN line wavelengths.
        This moves the *lines* (absorption/emission) before computing Jν.
    line_dlam_A : float
        Additive shift [Å] applied to the CN line wavelengths.
    lsf_for_Jnu : callable, optional
        Line profile φ(Δλ) used to weight the pumping when computing Jν.
        If None, Jν is computed from F_λ interpolated at line center.
        If not None, for each line at λ₀:

            F_eff(λ₀) ∝ ∑ F_λ(λ_i) φ(λ_i - λ₀)

        (normalized weighted mean over the pumping spectrum).

    Notes
    -----
    - The shifted wavelength is stored in ``Wave_vac_AA``.
    - ``J_nu_erg_cm2_s_Hz_sr`` is computed using these shifted wavelengths
      and (optionally) the given line profile.
    """
    # original rest-frame λ from Brooke calib
    lam_rest = np.asarray(df["lambda_vac_A_from_Cal"], float)

    # shift lines (NOT the solar spectrum)
    lam = lam_rest.copy()
    if line_v_kms != 0.0:
        c_kms = const.c.to("km/s").value
        lam *= (1.0 + line_v_kms / c_kms)
    if line_dlam_A != 0.0:
        lam += line_dlam_A

    # pumping spectrum
    wave_AA = _as_array(pumping, "WAVE")
    F_vals = _as_array(pumping, "FLUX")
    F_lambda = F_vals * (u.erg / (u.s * u.cm**2 * u.AA))

    # build table
    lines = Table.from_pandas(df.copy())

    lam_q = lam * u.AA
    lines["Wave_vac_AA"] = lam
    lines["Frequency_Hz"] = (const.c / lam_q).to(u.Hz)

    # ----- effective F_lambda at each line (optionally using line profile) -----
    if lsf_for_Jnu is None:
        # simple interpolation
        F_interp = np.interp(lam, wave_AA, F_lambda.value) * F_lambda.unit
    else:
        # convolve local pumping with line profile centered at λ₀
        if wave_AA.size > 1:
            dlam = np.median(np.diff(wave_AA))
        else:
            dlam = 0.0

        F_eff = []
        for lam0 in lam:
            dl = wave_AA - lam0
            kern = np.asarray(lsf_for_Jnu(dl), float)
            kern = np.where(np.isfinite(kern), kern, 0.0)
            s = kern.sum()
            if s <= 0.0:
                # fallback: point-sample
                f_val = np.interp(lam0, wave_AA, F_lambda.value)
            else:
                # normalized weighted mean (discrete φ(Δλ) F_λ)
                if dlam > 0.0:
                    f_val = np.sum(F_lambda.value * kern) / s
                else:
                    f_val = np.sum(F_lambda.value * kern) / s
            F_eff.append(f_val)
        F_interp = np.asarray(F_eff) * F_lambda.unit

    lines["F_lambda_at_comet_erg_s_cm2_AA"] = F_interp

    # Fλ -> Fν -> Jν
    F_nu = F_interp.to(
        u.erg / (u.s * u.cm**2 * u.Hz),
        equivalencies=u.spectral_density(lam_q),
    )
    J_nu = (F_nu / (4.0 * np.pi)) * (1.0 / u.sr)
    lines["J_nu_erg_cm2_s_Hz_sr"] = J_nu.to(
        u.erg / (u.cm**2 * u.s * u.Hz * u.sr)
    )

    # labels
    lines["J_prime"] = lines["J'"]
    lines["J_doubleprime"] = lines["J''"]

    lines["Upper_Manifold"] = [
        make_manifold(es, v) for es, v in zip(lines["eS'"], lines["v'"])
    ]
    lines["Lower_Manifold"] = [
        make_manifold(es, v) for es, v in zip(lines["eS''"], lines["v''"])
    ]
    lines["Sym_prime"] = [
        make_sym(F, p, use_omega_labels, es)
        for F, p, es in zip(lines["F'"], lines["p'"], lines["eS'"])
    ]
    lines["Sym_doubleprime"] = [
        make_sym(F, p, use_omega_labels, es)
        for F, p, es in zip(lines["F''"], lines["p''"], lines["eS''"])
    ]

    return lines



# ---------------------------------------------------------------------------
# Radiative + collisional rate matrix
# ---------------------------------------------------------------------------

def build_rate_matrix_nbar(
    lines: Table,
    *,
    include_stim_emission: bool = False,
    verbose: bool = True,
):
    """
    Build radiative rate matrix M, store cached columns in `lines`.
    """
    lines_out = lines.copy()

    nu = np.asarray(lines_out["Frequency_Hz"], float) * u.Hz
    Aul = np.asarray(lines_out["A"], float) / u.s

    Jnu_cgs = np.asarray(
        lines_out["J_nu_erg_cm2_s_Hz_sr"], float
    ) * (u.erg / (u.cm**2 * u.s * u.Hz * u.sr))
    Jnu_SI = Jnu_cgs.to(u.W / (u.m**2 * u.Hz * u.sr))

    Jp = np.asarray(lines_out["J_prime"], float)
    Jpp = np.asarray(lines_out["J_doubleprime"], float)
    gu, gl = 2.0 * Jp + 1.0, 2.0 * Jpp + 1.0

    # photon occupation (not used directly except for completeness)
    _ = (const.c ** 2 / (2.0 * const.h * nu ** 3) * Jnu_SI).decompose().value

    B_lu = (Aul * const.c ** 2 / (2.0 * const.h * nu ** 3) * (gu / gl)).decompose().value
    B_ul = (Aul * const.c ** 2 / (2.0 * const.h * nu ** 3)).decompose().value

    R_lu = B_lu * Jnu_SI.value
    R_ul = Aul.value
    if include_stim_emission:
        R_ul = R_ul + B_ul * Jnu_SI.value

    up_man = np.asarray(lines_out["Upper_Manifold"], str)
    up_sym = np.asarray(lines_out["Sym_prime"], str)
    lo_man = np.asarray(lines_out["Lower_Manifold"], str)
    lo_sym = np.asarray(lines_out["Sym_doubleprime"], str)

    all_levels: Dict[Tuple[str, float, str], int] = {}
    upper_idx = []
    lower_idx = []

    for i in range(len(lines_out)):
        ku = (up_man[i], float(Jp[i]), up_sym[i])
        kl = (lo_man[i], float(Jpp[i]), lo_sym[i])
        if ku not in all_levels:
            all_levels[ku] = len(all_levels)
        if kl not in all_levels:
            all_levels[kl] = len(all_levels)
        upper_idx.append(all_levels[ku])
        lower_idx.append(all_levels[kl])

    idx_to_level = {v: k for k, v in all_levels.items()}
    n_levels = len(idx_to_level)

    M = np.zeros((n_levels, n_levels), float)

    def add_rate(dest, src, rate):
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


def add_cn_collisions_to_M(
    M: np.ndarray,
    lines_with_rates: Table,
    idx_to_level: Dict[int, Tuple[str, float, str]],
    *,
    Q: float = 1e-3,
    T: float = 300.0,
    include_deltaJ0: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Add simple Ground X rotational collisions to M."""
    lower_idx = np.asarray(lines_with_rates["__lower_idx"], int)
    Jpp = np.asarray(lines_with_rates["J_doubleprime"], float)
    Elower_cm = np.asarray(lines_with_rates["E''"], float)

    E_level_cm: Dict[int, float] = {}
    J_level: Dict[int, float] = {}
    sym_level: Dict[int, str] = {}

    for i, idx in enumerate(lower_idx):
        man, _, sym = idx_to_level[idx]
        if str(man).lower().startswith("ground x"):
            E_level_cm[idx] = min(E_level_cm.get(idx, Elower_cm[i]), Elower_cm[i])
            J_level[idx] = float(Jpp[i])
            sym_level[idx] = str(sym)

    ground_ids = sorted(E_level_cm.keys())
    if not ground_ids:
        if verbose:
            print("[collisions] No Ground X levels; skipping.")
        return M

    Eerg = np.array([E_level_cm[i] for i in ground_ids], float) * \
           const.h.cgs.value * const.c.cgs.value
    Jarr = np.array([J_level[i] for i in ground_ids], float)
    sym_arr = np.array([sym_level[i] for i in ground_ids], str)
    gdeg = 2.0 * Jarr + 1.0

    kT = (const.k_B.cgs * (T * u.K)).to(u.erg).value

    pairs_done = set()

    def add_pair(iu: int, il: int, C_down: float) -> None:
        if C_down <= 0.0:
            return
        gu, gl = gdeg[iu], gdeg[il]
        dE = Eerg[iu] - Eerg[il]
        Cup = (gu / gl) * C_down * np.exp(-dE / kT)

        M[ground_ids[iu], ground_ids[iu]] -= C_down
        M[ground_ids[il], ground_ids[iu]] += C_down

        if np.isfinite(Cup) and Cup > 0.0:
            M[ground_ids[il], ground_ids[il]] -= Cup
            M[ground_ids[iu], ground_ids[il]] += Cup

    for i in range(len(ground_ids)):
        Ji = Jarr[i]
        si = sym_arr[i]
        for j in range(i + 1, len(ground_ids)):
            Jj = Jarr[j]
            sj = sym_arr[j]
            dJ = abs(Ji - Jj)
            allow = (dJ == 1) or (include_deltaJ0 and dJ == 0 and si != sj)
            if not allow:
                continue

            iu = j if Eerg[j] > Eerg[i] else i
            il = i if iu == j else j
            key = (min(i, j), max(i, j))
            if key in pairs_done:
                continue
            pairs_done.add(key)

            add_pair(iu, il, Q)

    if verbose:
        print(f"[collisions] Added {len(pairs_done)} X-state links; Q={Q:.3e}, T={T:.1f}")
    return M


# ---------------------------------------------------------------------------
# Solver & g-factors
# ---------------------------------------------------------------------------

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


def g_factors(lines_with_rates: Table, n: np.ndarray):
    """Return per-line (g_phot, g_energy, Σg_phot, Σg_energy)."""
    nu = np.asarray(lines_with_rates["__nu_Hz"], float)
    Aul = np.asarray(lines_with_rates["A"], float)
    ui = np.asarray(lines_with_rates["__upper_idx"], int)

    nu = np.nan_to_num(nu, 0.0, 0.0, 0.0)
    Aul = np.nan_to_num(Aul, 0.0, 0.0, 0.0)

    n_u = n[ui]
    g_ph = n_u * Aul
    g_en = const.h.cgs.value * nu * g_ph

    g_ph = np.nan_to_num(g_ph, 0.0, 0.0, 0.0)
    g_en = np.nan_to_num(g_en, 0.0, 0.0, 0.0)

    return g_ph, g_en, float(g_ph.sum()), float(g_en.sum())


# ---------------------------------------------------------------------------
# Spectrum synthesis
# ---------------------------------------------------------------------------
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

    Assumes df_lines[lam_col] already contains any global wavelength shifts
    (e.g. from CN–Sun velocity). No extra v_shift/dlam_shift is applied here.

    Parameters
    ----------
    df_lines : Table
        Must contain `lam_col` in Å and either:
        - g_line_energy (erg s^-1 per molecule per line), or
        - g_line_phot   (photons s^-1 per molecule per line) plus frequency info.
    fwhm_A : float
        Gaussian FWHM [Å] for default LSF if `lsf` is None.
    dlam_A : float
        Sampling step [Å] for auto grid if `grid` not provided.
    lam_min, lam_max : float, optional
        Output grid limits [Å]. If None, inferred from line wavelengths.
    lam_col : str
        Column name with (possibly shifted) line wavelengths in Å.
    N_col_cm2 : float, optional
        Column density [cm^-2]. Required.
    Omega_sr : float, optional
        Aperture solid angle [sr]. If None, output is surface brightness;
        otherwise, flux = I_λ * Ω.
    grid : array_like, optional
        Custom output wavelength grid [Å].
    lsf : callable, optional
        LSF kernel: lsf(Δλ) -> weights. If None, use Gaussian with `fwhm_A`.

    Returns
    -------
    lam_grid : ndarray
    flux     : ndarray
    """
    if N_col_cm2 is None:
        raise ValueError("N_col_cm2 (cm^-2) is required.")
    if lam_col not in df_lines.colnames:
        raise ValueError(f"{lam_col!r} not found in df_lines.")

    # line wavelengths (already shifted if needed)
    lam_rest = np.asarray(df_lines[lam_col], float)

    if v_shift_kms is not None:
        c_kms = const.c.to("km/s").value
        lam = lam_rest * (1.0 + v_shift_kms / c_kms)
        if dlam_shift_A != 0.0:
            lam = lam + dlam_shift_A
    else:
        lam = lam_rest + dlam_shift_A

    # per-line power
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

    # select finite, positive lines
    m = np.isfinite(lam) & np.isfinite(I_line) & (I_line > 0.0)
    lam = lam[m]
    I_line = I_line[m]
    if lam.size == 0:
        if grid is None:
            return np.array([]), np.array([])
        return np.asarray(grid, float), np.zeros_like(grid, float)

    # output grid
    if lam_min is None:
        lam_min = float(lam.min() - 5.0 * fwhm_A)
    if lam_max is None:
        lam_max = float(lam.max() + 5.0 * fwhm_A)

    if grid is None:
        if dlam_A <= 0.0:
            dlam_A = fwhm_A / 3.0
        n = int(np.ceil((lam_max - lam_min) / dlam_A)) + 1
        grid = lam_min + np.arange(n, dtype=float) * dlam_A
    else:
        grid = np.asarray(grid, float)

    y = np.zeros_like(grid, dtype=float)

    # apply LSF
    if lsf is None:
        sigma = fwhm_A / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        norm = sigma * np.sqrt(2.0 * np.pi)
        for l0, I0 in zip(lam, I_line):
            y += I0 * np.exp(-0.5 * ((grid - l0) / sigma) ** 2) / norm

    dl = grid[:, None] - lam[None, :]
    prof = lsf(dl)              # broadcasted: (Ngrid, Nlines)
    spec_per_mol = (prof * I_line).sum(axis=1)

    fourpi = 4.0 * np.pi
    I_lambda = (N_col_cm2 / fourpi) * spec_per_mol

    if Omega_sr is not None:
        return grid, I_lambda * Omega_sr
    else:
        return grid, I_lambda
def make_lsf(
             params: Dict[str, float],
             mode: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
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
                gauss = np.exp(-0.5 * (dl / sigma_G) ** 2) / (
                    sigma_G * np.sqrt(2.0 * np.pi)
                )
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

def precompute_cn_collision_scaffold(lines_out: Any, idx_to_level: Any) -> Dict[str, np.ndarray]:
    """Return arrays to apply X-state rotational collisions quickly."""
    lower_idx = np.asarray(lines_out['__lower_idx'], int)
    Jpp       = np.asarray(lines_out['J_doubleprime'], float)
    Elower_cm = np.asarray(lines_out["E''"], float)  # cm^-1

    # collect unique Ground levels properties
    E_level_cm = {}
    J_level, sym_level = {}, {}
    for i, idx in enumerate(lower_idx):
        man, Jlab, sym = idx_to_level[idx]
        if str(man).lower().startswith("ground"):
            E_level_cm.setdefault(idx, []).append(Elower_cm[i])
            J_level[idx]   = float(Jpp[i])
            sym_level[idx] = str(sym)
    ground_ids = sorted(E_level_cm.keys())
    if not ground_ids:
        return dict(iu=np.array([],int), il=np.array([],int),
                    gu=np.array([]), gl=np.array([]), dE=np.array([]))
    # finalize energies/degeneracies
    from astropy import constants as const, units as u
    Eerg, gdeg = {}, {}
    for idx in ground_ids:
        Ecm  = np.median(np.asarray(E_level_cm[idx], float)) * (1/u.cm)
        Eerg[idx] = (const.h * const.c * Ecm).to(u.erg).value
        gdeg[idx] = 2.0 * J_level[idx] + 1.0
    ground_sorted = sorted(ground_ids, key=lambda i: (J_level[i], sym_level[i]))
    iu_list, il_list, gu_list, gl_list, dE_list = [], [], [], [], []
    seen = set()
    for a in range(len(ground_sorted)):
        ia = ground_sorted[a]
        Ja, sa = J_level[ia], sym_level[ia]
        for b in range(a+1, len(ground_sorted)):
            ib = ground_sorted[b]
            Jb, sb = J_level[ib], sym_level[ib]
            dJ = abs(Ja - Jb)
            # allow = (dJ == 1) or ((dJ == 0) and (sa != sb))  # include ΔJ=0 parity mixing
            # if not allow: 
            #     continue
            iu, il = (ib, ia) if Eerg[ib] > Eerg[ia] else (ia, ib)
            key = (min(iu, il), max(iu, il))
            if key in seen: 
                continue
            seen.add(key)
            iu_list.append(iu); il_list.append(il)
            gu_list.append(gdeg[iu]); gl_list.append(gdeg[il])
            dE_list.append(Eerg[iu] - Eerg[il])  # >0
            #I want to print the delta energy in cm-1 for checking
            delta = (Eerg[iu] - Eerg[il]) * u.erg
            delta = delta / (const.h * const.c)
            delta = delta.to(u.cm**-1)

    return dict(iu=np.asarray(iu_list,int), il=np.asarray(il_list,int),
                gu=np.asarray(gu_list,float), gl=np.asarray(gl_list,float),
                dE=np.asarray(dE_list,float))

def apply_collisions_inplace(M: np.ndarray, scaffold: Dict[str, np.ndarray], Q: float, T: float) -> np.ndarray:
    """Modify M in place with Manfroid-style collisions for X-state."""
    if scaffold['iu'].size == 0 or Q <= 0: 
        return
    from astropy import constants as const, units as u
    kT = (const.k_B * (T * u.K)).to(u.erg).value
    iu = scaffold['iu']; il = scaffold['il']
    gu = scaffold['gu']; gl = scaffold['gl']; dE = scaffold['dE']
    Cdown = Q * np.ones_like(iu, dtype=float)
    Cup   = (gu/gl) * Cdown * np.exp(-dE / kT)
    # diagonal losses
    np.add.at(M, (iu, iu), -Cdown)
    np.add.at(M, (il, il), -Cup)
    # off-diagonal gains
    np.add.at(M, (il, iu),  Cdown)
    np.add.at(M, (iu, il),  Cup)
    return M

def mcmc_fitting(
    data: Any,
    window: Tuple[float, float],
    *,
    pumping: Any,
    line_path: Optional[str] = None,
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
    velocity_kms: float = 0.0,
    delta_lambda_A: float = 0.0,
    fig_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    MCMC fit of CN fluorescence in a wavelength window.

    Shifts are applied to the CN *lines* when computing Jν and emission:
    - v_kms : line Doppler shift [km/s]
    - dlam  : additive line shift [Å]

    Once these are applied in `attach_pumping_and_labels`, the same
    shifted `Wave_vac_AA` is used for both pumping and emission, so no
    extra v_shift/dlam_shift is needed in the synthesis.

    See previous docstring for parameter / return details.
    """
    if priors is None:
        raise ValueError("Please provide a dict of priors for the parameters to fit.")

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
        elif lsf_method == 'Lorentz':
            required = {'fwhm_L'}
            if not required.issubset(param_keys):
                raise ValueError("For Lorentz: prior for fwhm_L required.")
            drop = {"sigma_G", "sigma1", "sigma2", "sigma", "ratio"}
        else:
            raise ValueError(
                "Provide `lsf` or lsf_method in {'2Gauss','Gauss_Lorentz','Gauss'}."
            )
        param_keys = [k for k in param_keys if k not in drop]
        priors = {k: priors[k] for k in param_keys}

    # sanity on priors
    for name in param_keys:
        lo, hi = priors[name]
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f"Bad prior for {name!r}: {priors[name]}")

    # ---------- 1) Static pieces: line list + original pumping ----------
    if line_path is None:
        line_path = str(helper.get_default_mol_linelist_path())

    df_all = helper.load_cn_linelist(line_path)
    lines_brook = filter_AX_BX(
        df_all,
        lambda_min_A=2990.0010,
        lambda_max_A=10009.9980,
        A_min=A_min,
    )
        # 1) Build shifted lines + Jν for this θ
    lines_theta = attach_pumping_and_labels(
        lines_brook,
        pumping,
        use_omega_labels=False,
        line_v_kms=velocity_kms,
        line_dlam_A=delta_lambda_A,
        lsf_for_Jnu=None,
    ) # this line_v_kms applies to pumping & emission set now as 0 when called in the FluorescenceModel class

    # 2) radiative matrix for this theta
    M_rad_theta, idx_to_level_theta, lines_out_theta = build_rate_matrix_nbar(
        lines_theta,
        include_stim_emission=True,
        verbose=False,
    )

    coll_scaf = precompute_cn_collision_scaffold(lines_out_theta, idx_to_level_theta)


    # ---------- 2) Observed data subset ----------
    def _col(obj, name: str) -> np.ndarray:
        if hasattr(obj, "colnames"):
            return np.asarray(obj[name])
        if hasattr(obj, "columns"):
            return np.asarray(obj[name].values)
        return np.asarray(obj[name])

    x_data = _col(data, "WAVE")
    y_data = _col(data, "FLUX_STACK")
    y_err = _col(data, "ERR_STACK")
    cont = _col(data, "CONTINUUM")

    mwin = (x_data >= window[0]) & (x_data <= window[1])
    x_fit = x_data[mwin]
    y_fit = y_data[mwin] - cont[mwin]
    y_err_fit = y_err[mwin]

    # ---------- 3) Helpers ----------
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

        if lsf_method == "2Gauss":
            sigma1 = float(pars["sigma1"])
            sigma2 = float(pars["sigma2"])
            ratio = float(pars["ratio"])
            def lsf_fun(dl: np.ndarray) -> np.ndarray:
                g1 = np.exp(-0.5 * (dl / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
                g2 = np.exp(-0.5 * (dl / sigma2) ** 2) / (sigma2 * np.sqrt(2.0 * np.pi))
                return ratio * g1 + (1.0 - ratio) * g2
            return lsf_fun

        if lsf_method == "Gauss":
            sigma = float(pars["sigma"])
            def lsf_fun(dl: np.ndarray) -> np.ndarray:
                return np.exp(-0.5 * (dl / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
            return lsf_fun

        if lsf_method == "Gauss_Lorentz":
            sigma_G = float(pars["sigma_G"])
            fwhm_L = float(pars["fwhm_L"])
            ratio = float(pars["ratio"])
            gamma = fwhm_L / 2.0
            A = 2.0 / (np.pi * fwhm_L)
            def lsf_fun(dl: np.ndarray) -> np.ndarray:
                gauss = np.exp(-0.5 * (dl / sigma_G) ** 2) / (
                    sigma_G * np.sqrt(2.0 * np.pi)
                )
                lorentz = A * gamma**2 / (gamma**2 + dl**2)
                return ratio * gauss + (1.0 - ratio) * lorentz
            return lsf_fun
        if lsf_method == "Lorentz":
            fwhm_L = float(pars['fwhm_L'])
            gamma = fwhm_L / 2.0
            A = 2.0 / (np.pi * fwhm_L)
            def lsf_fun(dl: np.ndarray) -> np.ndarray:
                return A * gamma**2 / (gamma**2 + dl**2)
            return lsf_fun

        return None

    def model_flux(theta: Sequence[float], wave: np.ndarray) -> np.ndarray:
        pars = theta_to_params(theta)

        logN = float(pars.get("logN", 11.0))
        logQ = float(pars.get("logQ", -3.0))
        T = float(pars.get("T", 300.0))

        # line shifts affecting both pumping & emission
        v_kms = float(pars.get("v_kms", 0.0))
        dlam = float(pars.get("dlam", 0.0))

        lsf_fun = make_lsf_local(pars)
        Q = 10.0**logQ if np.isfinite(logQ) else 0.0

        M = M_rad_theta.copy()

        # 3) collisions
        if Q > 0.0:
            M = apply_collisions_inplace(M, coll_scaf, Q=Q, T=T)

        # 4) populations
        n = solve_with_normalization(M, verbose=False)

        # 5) g-factors
        _, g_en, *_ = g_factors(lines_out_theta, n)

        # 6) synthesize emission on wave
        omega = np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2  # 1" aperture
        _, spec = synth_spectrum_from_lines(
            lines_out_theta,
            g_line_energy=g_en,
            lam_min=float(wave.min()),
            lam_max=float(wave.max()),
            lam_col="Wave_vac_AA",
            N_col_cm2=10.0**logN,
            Omega_sr=omega,
            grid=wave,
            lsf=lsf_fun,
            v_shift_kms=v_kms,
            dlam_shift_A=dlam,
        ) # this are just shifts aplied on the lines centers for synthesis
        return spec

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

    # ---------- 4) Run emcee ----------
    ndim = len(param_keys)
    nburn = nsteps // 2

    print("Number of iterations:", ndim * nwalkers * nsteps)

    p0 = np.array(
        [[np.random.uniform(*priors[name]) for name in param_keys]
         for _ in range(nwalkers)]
    )

    move = emcee.moves.StretchMove(a=a)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, moves=move, threads=threads)
    sampler.run_mcmc(p0, nsteps, progress=progress)
    
    chain = sampler.get_chain()
    lnprob_full = sampler.get_log_prob()

    # ---------- 5) Best-fit (no pruning) ----------
    flat_chain = chain.reshape(-1, ndim)
    flat_lnprob = lnprob_full.reshape(-1)
    best_idx = int(np.argmax(flat_lnprob))
    best_theta = flat_chain[best_idx]
    best_params = theta_to_params(best_theta)

    print("#" * 50)
    print("*** Best fit (no pruning) ***")
    for name in param_keys:
        print(f"{name}: {best_params[name]:.6g}")

    af = sampler.acceptance_fraction
    print("#" * 50)
    print("*** Acceptance Fraction ***")
    print("Mean acceptance fraction:", np.mean(af))
    af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                            between 0.2 and 0.5
            If af < 0.2 decrease the MCMCA parameter
            If af > 0.5 increase the MCMCA parameter
            '''
    print("Mean acceptance fraction:", np.mean(af))
    if np.mean(af)<0.2 or np.mean(af)>0.5:
        print(af_msg)

    samples = chain[nburn:, :, :].reshape(-1, ndim)
    lnprob_burn = lnprob_full[nburn:, :].reshape(-1)

    print("#" * 50)
    print("*** Pruning... ***")
    try:
        samples_pruned, lnprob_pruned = prune(samples, lnprob_burn, quiet=not progress)
    except Exception as exc:
        print("Pruning failed:", exc)
        samples_pruned, lnprob_pruned = samples, lnprob_burn
    # ---------- 7) Posterior summaries ----------
    median_params: Dict[str, float] = {}
    up_errors: Dict[str, float] = {}
    low_errors: Dict[str, float] = {}
    for i, name in enumerate(param_keys):
        p16, p50, p84 = np.percentile(samples_pruned[:, i], [16, 50, 84])
        err = 0.5 * ((p84 - p50) + (p50 - p16))
        median_params[name] = float(p50)
        up_errors[name] = float(p84 - p50)
        low_errors[name] = float(p50 - p16)
        print(f"{name}: {p50:.4f} +/- {err:.4f}  [{p16:.4f}, {p84:.4f}]")

    # ---------- 8) Model ensemble on dense grid ----------
    x_model = np.linspace(window[0], window[1], 20000)
    n_draw = min(200, samples_pruned.shape[0])
    model_stack = np.empty((n_draw, x_model.size))
    for i in range(n_draw):
        model_stack[i] = model_flux(samples_pruned[i], x_model)

    theta_med = [median_params[k] for k in param_keys]
    best_model = model_flux(theta_med, x_model)

    p16, p50, p84 = np.percentile(model_stack, [16, 50, 84], axis=0)
    median_model = p50
    model_p16 = p16
    model_p84 = p84

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
    # ---------- 9) Optional plots ----------
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
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{fig_file}_fit.pdf", dpi=300, format='pdf')
        plt.show()

    # ---------- 10) Return ----------
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
        "best_model": best_model, #should be almost the same as median_model
    }