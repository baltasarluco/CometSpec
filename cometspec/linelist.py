"""Line-list parsing and normalization routines.

Routines
--------
- :func:`normalize_cn_systems_arg` -- Normalize user-friendly CN system selectors to canonical tokens.
- :func:`from_user_linelist` -- Convert a user line list into the normalized transition schema.
- :func:`make_sym` -- Build a symmetry label.
- :func:`from_cn_brooke` -- Convert a Brooke CN line list (e.g. from :func:`~cometspec.helper.load_cn_linelist`) to the normalized schema.
- :func:`filter_cn_systems` -- Filter a Brooke CN line list by system, wavelength, and A (Einstein coefficient) threshold.
- :func:`load_default_transitions` -- Load and normalize packaged transitions per isotopologue.
- :func:`resolve_linelists_with_defaults` -- Resolve user-supplied linelists, filling in defaults for any missing isotopologues.
- :func:`default_linelist_source` -- Return the file path that would be loaded for a given isotopologue from packaged defaults.
- :func:`linelist_origins` -- Return a mapping of isotopologue to source description for a set of linelists.
- :func:`attach_pumping_and_labels` -- Attach pumping information and human-friendly labels to a transition table. **This is important as it ensures the solar pumping information is correctly associated with each transition.**
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Callable, Any

import re
import warnings

import numpy as np
import pandas as pd

from astropy import constants as const
from astropy import units as u
from astropy.table import Table

from . import helper
from .collisions import canonical_diatomic_name


__all__ = [
    "normalize_cn_systems_arg",
    "from_user_linelist",
    "make_sym",
    "from_cn_brooke",
    "filter_cn_systems",
    "load_default_transitions",
    "resolve_linelists_with_defaults",
    "default_linelist_source",
    "linelist_origins",
    "attach_pumping_and_labels",
]


#: Package path
PACKAGE_DIR: Path = Path(__file__).resolve().parent
#: PACKAGE_DIR / "data"
DATA_DIR: Path = PACKAGE_DIR / "data"


def _as_list(x: str | Sequence[str] | None) -> list[str]:
    '''Normalize a string or sequence of strings to a list of strings.

    Parameters
    ----------
    x : str or Sequence[str], optional, default None
        Input string or sequence of strings.

    Returns
    -------
    list[str]
        Normalized list of strings.
    '''

    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def normalize_cn_systems_arg(systems: str | Sequence[str] | None) -> list[str]:
    r"""Translate user-friendly CN band-system selectors into canonical tokens.

    This is the input-parser for any function that needs to know which CN
    band system(s) to operate on. It accepts a variety of human-friendly
    spellings (case-insensitive, with or without dashes/parentheses) and
    maps each one to a fixed set of internal tokens used downstream. A
    sequence of selectors is also accepted; results are flattened and
    deduplicated while preserving order.

    The canonical (output) tokens are:

    * ``"BX00"`` -- :math:`B^{2}\Sigma^{+} \to X^{2}\Sigma^{+}` violet system,
      :math:`(v', v'') = (0, 0)` band (~388 nm).
    * ``"AX_dv0"`` -- :math:`A^{2}\Pi \to X^{2}\Sigma^{+}` red system,
      :math:`\Delta v = |v' - v''| = 0` sequence.
    * ``"AX_dv1"`` -- :math:`A^{2}\Pi \to X^{2}\Sigma^{+}` red system,
      :math:`\Delta v = |v' - v''| = 1` sequence.
    * ``"AX_dv2"`` -- A–X red system, :math:`\Delta v = 2` sequence.
    * ``"AX_dv3"`` -- A–X red system, :math:`\Delta v = 3` sequence.
    * ``"XX"`` -- All X–X transitions.
    * ``"ALL"`` -- This token if used, it will include all transitions, resulting in extremely long computation times.

    Recognized input forms (all matched case-insensitively after stripping):

    * ``None`` -- default selection, returns ``["BX00", "AX_dv1"]``.
    * ``"both"``, ``"bx+ax"``, ``"bxax"`` -- violet plus all three red sequences.
    * ``"all"`` -- returns ``["ALL"]``.
    * ``"bx"``, ``"b-x"``, ``"bx(0,0)"``, ``"bx00"``, ``"bx_00"``, ``"b_x_00"``
      -- the violet :math:`(0,0)` band.
    * ``"ax"``, ``"a-x"`` -- the :math:`\Delta v = 1` and :math:`\Delta v = 2`
      red sequences.
    * ``"ax(dv=0)"``, ``"ax_dv0"`` -- A–X :math:`\Delta v = 0` only.
    * ``"ax(dv=1)"``, ``"ax_dv1"`` -- A–X :math:`\Delta v = 1` only.
    * ``"ax(dv=2)"``, ``"ax_dv2"`` -- A–X :math:`\Delta v = 2` only.
    * ``"ax(dv=3)"``, ``"ax_dv3"`` -- A–X :math:`\Delta v = 3` only.
    * ``"xx"`` -- all X–X transitions.
    * Any other string -- passed through unchanged as a single-element list,
      letting the caller handle (or reject) unknown tokens.
    * A sequence (list, tuple, ...) of any of the above -- each element is
      normalized recursively, results are concatenated, and duplicates are
      removed while preserving first-occurrence order.

    Parameters
    ----------
    systems : str or sequence of str, optional
        Band-system selector(s). See the list of recognized forms above.

    Returns
    -------
    list of str
        Canonical token list. Order matches the order of the input. No results are duplicated.

    Examples
    --------
    .. code-block:: python

        normalize_cn_systems_arg(None)
        ['BX00', 'AX_dv1']
        normalize_cn_systems_arg("both")
        ['BX00', 'AX_dv1', 'AX_dv2', 'AX_dv3']
        normalize_cn_systems_arg("BX")
        ['BX00']
        normalize_cn_systems_arg(["bx", "ax_dv1", "bx"])  # dedup, order preserved
        ['BX00', 'AX_dv1']
        normalize_cn_systems_arg("unknown")
        ['unknown']
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
        if s in ("ax(dv=0)", "ax_dv0"):
            return ["AX_dv0"]
        if s in ("ax(dv=1)", "ax_dv1"):
            return ["AX_dv1"]
        if s in ("ax(dv=2)", "ax_dv2"):
            return ["AX_dv2"]
        if s in ("ax(dv=3)", "ax_dv3"):
            return ["AX_dv3"]
        if s in ('xx',):
            return ['XX']
        else:
            warnings.warn(f"normalize_cn_systems_arg: unrecognized system selector {systems!r}, this will be omitted.")
        return [systems]
    else:
        out: list[str] = []
        for item in systems:
            out.extend(normalize_cn_systems_arg(item))
        seen = set()
        out2 = []
        for t in out:
            if t not in seen:
                seen.add(t)
                out2.append(t)
        return out2


def from_user_linelist(
    df: pd.DataFrame,
    *,
    lam_col: str,
    A_col: str,
    upper_id_col: str,
    lower_id_col: str,
    g_upper_col: str,
    g_lower_col: str,
    lower_es_col: str | None = None,
    lower_v_col: str | None = None,
    lower_J_col: str | None = None,
    lower_sym_col: str | None = None,
    E_lower_cm1_col: str | None = None,
) -> pd.DataFrame:
    r"""Convert a user line list into the normalized transition schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Input line list table.
    lam_col : str
        Wavelength column in vacuum :math:`\AA`.
    A_col : str
        Einstein :math:`A` coefficient column in :math:`\mathrm{s}^{-1}`.
    upper_id_col : str
        Upper-state identifier column.
    lower_id_col : str
        Lower-state identifier column.
    g_upper_col : str
        Upper-state degeneracy column.
    g_lower_col : str
        Lower-state degeneracy column.
    lower_es_col : str, optional, default None
        Optional lower electronic-state column.
    lower_v_col : str, optional, default None
        Optional lower vibrational-level column.
    lower_J_col : str, optional, default None
        Optional lower rotational-level column.
    lower_sym_col : str, optional, default None
        Name of an optional column holding a composite lower-state spin-orbit/parity label. For Brooke-style line lists this is typically the concatenation of the lower-state :math:`F''`, :math:`p''`, and :math:`eS''` columns, which together identify the fine-structure/parity sublevel within its electronic state.
    E_lower_cm1_col : str, optional, default None
        Optional lower-state energy column in :math:`\mathrm{cm}^{-1}`. A pair of levels will use these values to get the :math:`\Delta E` for the collisions.

    Returns
    -------
    pandas.DataFrame
        Normalized transition table. Note that the output has `E_cm1` and optionally `E_lower_cm1`, they are different, the first is the energy corresponding to the transition (energy from the line wavelength) and the second one the energy of a state with respect the ground state.

    Raises
    ------
    ValueError
        If required columns are missing or values are invalid.
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


def make_sym(F, p, use_omega: bool = False, es: Optional[str] = None) -> str:
    """Build a compact CN-style symmetry label.

    Parameters
    ----------
    F : Any
        Spin component or branch label.
    p : Any
        Parity label.
    use_omega : bool, optional, default False
        Whether to emit Omega-style labels for A states.
    es : str, optional, default None
        Electronic-state label.

    Returns
    -------
    str
        Compact symmetry token.
    """
    ptag = str(p).strip().lower()[:1] if p not in (None, "") else "?"
    try:
        Fint = int(F)
    except (ValueError, TypeError):
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
    E_lower_col: str = "E''",
) -> pd.DataFrame:
    """Convert a Brooke CN line list (e.g. the output :func:`~cometspec.helper.load_cn_linelist`) to the normalized schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Brooke-format CN line list.
    lam_col : str, optional, default "lambda_vac_A_from_Cal"
        Wavelength column in vacuum Angstrom.
    A_col : str, optional, default "A"
        Einstein A coefficient column.
    use_omega_labels : bool, optional, default False
        Use Omega labels for A-state symmetry tags.
    E_lower_col : str, optional, default "E''"
        Lower-state energy column in cm^-1.

    Returns
    -------
    pandas.DataFrame
        Normalized transition table. Each row is one rovibronic transition.

        * ``lambda_vac_A`` (:class:`float`) -- Vacuum wavelength in Å.
        * ``A_ul`` (:class:`float`) -- Einstein :math:`A` coefficient (spontaneous emission rate), in s\ :sup:`-1`.
        * ``upper_id`` (:class:`str`) -- String key identifying the upper level, formatted as ``ES|v=V|J=J|sym=S``.
        * ``lower_id`` (:class:`str`) -- String key identifying the lower level, formatted as ``ES|v=V|J=J|sym=S``.
        * ``g_upper`` (:class:`float`) -- Upper-level degeneracy.
        * ``g_lower`` (:class:`float`) -- Lower-level degeneracy.
        * ``E_cm1`` (:class:`float`) -- Transition energy in cm\ :sup:`-1`, computed as :math:`1/\lambda_{\mathrm{vac}}`.
        * ``lower_es`` (:class:`str`) -- Lower electronic state label (e.g. ``X``).
        * ``lower_v`` (:class:`float`) -- Lower vibrational quantum number :math:`v''`.
        * ``lower_J`` (:class:`float`) -- Lower rotational quantum number :math:`J''`.
        * ``lower_sym`` (:class:`str`) -- Lower-level symmetry tag (e-f parity and :math:`\Omega` component).
        * ``E_lower_cm1`` (:class:`float`) -- Lower-state energy in cm\ :sup:`-1`, taken directly from ``E_lower_col``. A pair of levels will use these values to get the :math:`\Delta E` for the collisions.

    Raises
    ------
    ValueError
        If required columns are missing or contain invalid values.
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

    Parameters
    ----------
    df_all : pandas.DataFrame
        Full Brooke/Sneden CN table.
    systems : str or Sequence[str], optional, default None
        System selector(s) accepted by :func:`normalize_cn_systems_arg`.
    lambda_min_A : float, optional, default 2990.001
        Minimum wavelength in Angstrom.
    lambda_max_A : float, optional, default 10009.998
        Maximum wavelength in Angstrom.
    A_min : float, optional, default 1e4
        Minimum Einstein A threshold, or ``None`` to disable.
    lam_col : str, optional, default "lambda_vac_A_from_Cal"
        Wavelength column name.

    Returns
    -------
    pandas.DataFrame
        Filtered CN line list.
    """

    df = df_all.copy()
    tokens = normalize_cn_systems_arg(systems)

    if "ALL" not in tokens:
        df = df[df["eS''"].astype(str).str.upper().str.startswith("X")]
        masks = []
        if "BX00" in tokens:
            masks.append((df["eS'"] == "B") & (df["v'"] == 0) & (df["v''"] == 0))
        if "AX_dv0" in tokens:
            masks.append((df["eS'"] == "A") & (np.abs(df["v'"] - df["v''"]) == 0))
        if "AX_dv1" in tokens:
            masks.append((df["eS'"] == "A") & (np.abs(df["v'"] - df["v''"]) == 1))
        if "AX_dv2" in tokens:
            masks.append((df["eS'"] == "A") & (np.abs(df["v'"] - df["v''"]) == 2))
        if "AX_dv3" in tokens:
            masks.append((df["eS'"] == "A") & (np.abs(df["v'"] - df["v''"]) == 3))
        if 'XX' in tokens:
            masks.append((df["eS'"] == "X"))
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
    
    r"""Load and normalize packaged default transitions per isotopologue. The options are **"12C2", "12C13C", "13C2", "12C14N", "13C14N", "12C15N", "Fe".**
    For CN if the isotopologue is not found it will fall back to "12C14N". Any string with Fe on it will load the fe_normalized.csv file. For C2 if the isotopologue is not found it will fail.
    If CN is choosen, the systems to include can be given as a parameter. Default system is BX(0,0) and AX(Δv=+1), but this can be changed with the ``systems`` argument.
    The options for ``systems`` is list containing one or more of the following str:

    - "both" or "bx+ax": BX(0,0), AX(Δv=±1), AX(Δv=±2) and AX(Δv=±3)
    - "all": all systems in the Brooke linelist (including minor ones, this will lead to extremely high computation times)
    - "bx", "b-x", "bx(0,0)", "bx00", "bx_00", "b_x_00" or "b-x": BX(0,0) only
    - "ax" or "a-x": for "AX_dv1", 'AX_dv2'
    - "ax(dv=0)", "ax_dv0": AX(Δv=0) only
    - "ax(dv=1)", "ax_dv1": AX(Δv=±1) only
    - "ax(dv=2)", "ax_dv2": AX(Δv=±2) only
    - "ax(dv=3)", "ax_dv3": AX(Δv=±3) only
    - 'xx': all X-X transitions

    At the end the references of each line list can be found [1]_ [2]_ [3]_ [4]_ [5]_.
    
    .. note::
        Rows on the line lists with missing or invalid values in any of the necessary columns are dropped.
    
    .. important::
        These are intrinsic filters applied to the default line lists, so lines with values
        beyond these filters will not be retrieved for the default isotopologues, even if the
        corresponding model or function parameters are set.

        * For :math:`\rm CN`, the default line lists are the ones from [1]_ and [2]_, where
          the available systems are those described above. Check the respective references to
          see how they were built. We did not apply an intrinsic :math:`A_{ul}` cut. To use the
          full line lists, you will need to set the corresponding parameters when calling the
          function.
        * For :math:`\rm C_2`, the default line list is the recommended
          `ExoMol <https://www.exomol.com/data/molecules/C2/>`_ compilation
          [3]_ [4]_. The following selection criteria were applied: wavelengths
          in the range :math:`2000`--:math:`10000\,\unicode{x212B}`, upper-level
          energies :math:`< 30\,000\ \mathrm{cm}^{-1}`, vibrational quantum
          number :math:`v < 5`, rotational quantum number :math:`N < 50`, and
          only the :math:`a\,^1\Pi_u - x\,^1\Sigma_g^+`,
          :math:`b\,^3\Sigma_g^- - a\,^3\Pi_u`,
          :math:`d\,^3\Pi_g - a\,^3\Pi_u`,
          :math:`d\,^3\Pi_g - c\,^3\Sigma_u^+`,
          :math:`a\,^3\Pi_u - x\,^1\Sigma_g^+`, and
          :math:`c\,^3\Sigma_u^+ - x\,^1\Sigma_g^+` transitions.
          The minimum intrinsic :math:`A_{ul}` is :math:`10^{3}` for
          :math:`^{12}\mathrm{C}^{13}\mathrm{C}` and :math:`^{12}\mathrm{C}_2`, and
          :math:`10^{-10}` for :math:`^{13}\mathrm{C}_2`. Note that building models with a
          small :math:`A_{\min}` (i.e. including most of the transitions) is computationally
          expensive.
        * For :math:`\rm Fe`, we adopt the line list of [5]_. We retrieved all
          transitions in the :math:`2000`--:math:`10000\,\unicode{x212B}` range
          and retained those with :math:`A_{ul} > 10^{3}\ \mathrm{s}^{-1}` and
          upper-level energies below :math:`40\,000\ \mathrm{cm}^{-1}`. Note that there is an
          intrinsic :math:`A_{ul}` cut of :math:`10^{3}\ \mathrm{s}^{-1}`.
          
    Parameters
    ----------
    isotopologues : str or Sequence[str], optional, default "12C14N"
        One or more isotopologue labels.
    systems : str or Sequence[str], optional, default None
        CN system selector(s).
    A_min : float, optional, default 1e4
        Minimum Einstein A threshold.
    lambda_min_A : float, optional, default 2990.001
        Minimum wavelength in Angstrom.
    lambda_max_A : float, optional, default 10009.998
        Maximum wavelength in Angstrom.
    use_omega_labels : bool, optional, default False
        Use Omega labels for A-state symmetry tags.
    line_paths : dict[str, str], optional, default None
        Optional mapping of isotopologue to explicit file path.

    Returns
    -------
    dict[str, pandas.DataFrame]
       Dictionary mapping isotopologue label to normalized transition table. The keys are exactly the entries in ``isotopologues``; the values are DataFrames with the same schema as described in :func:`from_cn_brooke` or :func:`from_user_linelist`.
    
    References
    ----------
    .. [1] Brooke, J. S. A., Ram, R. S., Western, C. M., et al. 2014, ApJS, 210, 23 (`link <https://doi.org/10.1088/0067-0049/210/2/23>`__).
    .. [2] Sneden, C., Lucatello, S., Ram, R. S., Brooke, J. S. A., & Bernath, P. 2014, ApJS, 214, 26 (`link <https://doi.org/10.1088/0067-0049/214/2/26>`__).
    .. [3] Yurchenko, S. N., Szabó, I., Pyatenko, E., & Tennyson, J. 2018, MNRAS, 480, 3397 (`link <https://doi.org/10.1093/mnras/sty2050>`__).
    .. [4] McKemmish, L. K., Syme, A.-M., Borsovszky, J., et al. 2020, MNRAS, 497, 1081 (`link <https://doi.org/10.1093/mnras/staa1954>`__).
    .. [5] van Hoof, P. A. M. 2018, Galaxies, 6, 63 (`link <https://doi.org/10.3390/galaxies6020063>`__).
    """
    iso_list = _as_list(isotopologues)
    out: dict[str, pd.DataFrame] = {}
    sys_tokens = normalize_cn_systems_arg(systems)
    for iso in iso_list:
        matched = False
        if re.match(r"^\d+C\d+N$", iso):
            matched = True
            if line_paths is not None and iso in line_paths:
                path = line_paths[iso]
            else:
                try:
                    path = str(DATA_DIR / "CN" / f"{iso}.txt")
                except TypeError:
                    path = str(DATA_DIR / "CN" / f"12C14N.txt")

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
            path = DATA_DIR / 'fe_normalized.csv'
            tab = pd.read_csv(path)
            tab = tab[tab['A_ul'] > A_min]
            out[iso] = _drop_invalid_normalized_rows(tab, label=iso)
        if re.match(r"^\d+C\d+C$", iso) or re.match(r"^\d+C\d+$", iso):
            matched = True
            KEY_LINES = "/lines"
            canon = canonical_diatomic_name(iso) or iso
            path = DATA_DIR / f'C2/{canon}.h5'
            if not path.exists():
                path = DATA_DIR / f'C2/{iso}.h5'
            tab = pd.read_hdf(path, key=KEY_LINES)
            tab = tab[tab['A_ul'] > A_min]
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
    """ Function to take a list of linelists and a list of isotopologues. It is going to match all the line lists with their isotopologues, if the len linelists is less than the len of isotopologues, the remaining isotopologues will be loaded with the default linelists.
    Thus if the user wants to mix the default linelists and custome ones the isotopologues should be ordered by first the ones with provided line lists and then the ones without provided line lists, so the function can match them correctly.


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

    Parameters
    ----------
    linelists : pandas.DataFrame or dict[str, pandas.DataFrame] or Sequence[pandas.DataFrame] or None
        User-supplied line list(s). See resolution rules above
    iso_list : Sequence[str]
        Isotopologue labels, in the order they should be returned. Each label is matched against the user-supplied line lists (if any) according to the resolution rules above, and any isotopologue without a user-supplied line list is loaded from the packaged defaults.
    systems : str or Sequence[str], optional, default None
        CN system selector(s) for default CN line lists. See :func:`normalize_cn_systems_arg` for accepted forms.
    A_min : float, optional, default 1e4
        Minimum Einstein A threshold for default line lists, or ``None`` to disable.
    lambda_min_A : float, optional, default 2990.001
        Minimum wavelength in Angstrom for default line lists.
    lambda_max_A : float, optional, default 10009.998
        Maximum wavelength in Angstrom for default line lists.
    use_omega_labels : bool, optional, default False
        Use Omega labels for A-state symmetry tags in default CN line lists.
    line_paths : dict[str, str], optional, default None
    
    Returns
    -------
    dict[str, pandas.DataFrame]
        ``{iso: DataFrame}`` ordered exactly as ``iso_list``.
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

    Parameters
    ----------
    iso : str
        Isotopologue label.
    
    Returns
    -------
    str        
        File path that would be loaded for ``iso`` from packaged defaults.

    Raises
    ------
    ValueError
        If ``iso`` does not match any supported default pattern for the packaged default line lists (12C14N, 13C14N, 12C15N, 12C2, 13C2, 12C13C, or any label containing "Fe").
        (CN-like, C2-like, or containing ``"Fe"``).
    """
    PACKAGE_DIR = Path(__file__).resolve().parent
    DATA_DIR = PACKAGE_DIR / "data"
    if re.match(r"^\d+C\d+N$", iso):
        try:
            path = str(DATA_DIR / "CN" / f"{iso}.txt")
        except TypeError:
            path = str(DATA_DIR / "CN" / f"12C14N.txt")
        return path
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
    """Return a per-isotopologue origin string (file) for the configured line lists.

    Mirrors the resolution rules of :func:`resolve_linelists_with_defaults`:

    - Entries supplied by the user (DataFrame, dict entry, or positional list
      slot) are reported as ``"custom (user-provided)"``.
    - Entries with an explicit override in ``line_paths`` are reported as that
      path.
    - Otherwise the path returned by :func:`default_linelist_source` is used.

    Does not load any data just to determine the origin used.

    Parameters
    ----------
    linelists : pandas.DataFrame or dict[str, pandas.DataFrame] or Sequence[pd.DataFrame] or None
        User-supplied line list(s). See resolution rules in :func:`resolve_linelists_with_defaults`.
    iso_list : Sequence[str]
        Isotopologue labels, in the order they should be returned. Each label is matched against the user-supplied line lists (if any) according to the resolution rules above, and any isotopologue without a user-supplied line list is assigned the origin of the packaged default
    line_paths : dict[str, str], optional, default None
        Optional mapping of isotopologue to explicit file path, used for reporting the origin of any isotopologue without a user-supplied line list. If an isotopologue is present in this dict, its origin is reported as the corresponding path instead of the default path returned by :func:`default_linelist_source`. This is intended to be used when the user has provided a custom path
    
    Returns
    -------
    dict[str, str]
        Mapping of isotopologue label to origin string (e.g. file path). The keys are exactly the entries in ``iso_list``; the values are determined according to the resolution rules above.
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

    Parameters
    ----------
    tab : pandas.DataFrame
        Normalized transition table to check.
    label : str, optional, default ""
        Optional label to include in the warning message for dropped rows.
    
    Returns
    -------
    pandas.DataFrame
        Cleaned table with invalid rows dropped and index reset. If the input table is empty or None, returns an empty table with the same columns and a reset index.
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

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized transition DataFrame.
    pumping : Any
        Pumping spectrum with ``WAVE`` and ``FLUX`` columns.
    line_v_kms : float, optional, default 0.0
        Doppler velocity shift applied to line wavelengths, in km/s.
    line_dlam_A : float, optional, default 0.0
        Additive wavelength shift in Angstrom.
    lsf_for_Jnu : Callable[[numpy.ndarray], numpy.ndarray], optional, default None
        Optional kernel used to average flux around each line.
    lam_col : str, optional, default "lambda_vac_A"
        Input wavelength column name in ``df``.

    Returns
    -------
    astropy.table.Table
        Astropy table with wavelength, frequency, flux-at-line, J_nu and original dataframe columns.
    """
    from .rates import _as_array

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
