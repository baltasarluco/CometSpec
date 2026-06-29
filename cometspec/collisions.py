"""Rotational-collision scaffolds and in-place collision application.

Routines
------------
- :func:`canonical_diatomic_name` -- canonicalize diatomic isotopologue labels.
- :func:`is_homonuclear_diatomic` -- detect homonuclear diatomics.
- :func:`is_atomic_species` -- detect atomic (non-diatomic) species such as Fe.
- :func:`diatomic_symmetry_class` -- classify diatomics for collision selection rules.
- :func:`precompute_collision_scaffold` -- build a collision scaffold from explicit lower-state columns in the transition table.
- :func:`precompute_collision_scaffold_fast` -- same as above but with cached arrays for fast updates.
- :func:`apply_collisions_inplace` -- apply collision rates to a matrix in place using the scaffold.
- :func:`apply_collisions_inplace_fast` -- same as above but using cached arrays and reusable buffers for speed.
"""

from __future__ import annotations

from typing import Dict, Sequence, Any

import re

import numpy as np

from astropy import constants as const
from astropy import units as u


__all__ = [
    "canonical_diatomic_name",
    "is_homonuclear_diatomic",
    "is_atomic_species",
    "diatomic_symmetry_class",
    "precompute_collision_scaffold",
    "precompute_collision_scaffold_fast",
    "apply_collisions_inplace",
    "apply_collisions_inplace_fast",
]


#: Planck constant times speed of light over Boltzmann constant, in K*cm.
HC_OVER_K_B_KCM: float = (const.h * const.c / const.k_B).to_value(u.K * u.cm)


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

    Parameters
    ----------
    iso_name : str, optional
        Isotopologue label, or ``None``.
    
    Returns
    -------
    str or None
        Canonicalized isotopologue label, or ``None`` if the input was ``None``
    """
    if iso_name is None:
        return None
    atoms = _parse_isotopologue_atoms(iso_name)
    if len(atoms) != 2:
        return iso_name
    (m1, e1), (m2, e2) = atoms
    if (m1, e1) == (m2, e2):
        return f"{m1}{e1}2"
    a, b = sorted(atoms, key=lambda x: (x[0], x[1]))
    return f"{a[0]}{a[1]}{b[0]}{b[1]}"


def _parse_isotopologue_atoms(name: str) -> list[tuple[int, str]]:
    """Parse an isotopologue label like ``"12C2"``, ``"12C13C"``, or ``"12C14N"`` into atoms.

    Parameters
    ----------
    name : str
        Isotopologue label using ``<mass><Element>[count]`` tokens
        (e.g., ``"12C2"`` -> two ``(12, "C")``; ``"12C13C"`` -> ``(12, "C")`` and
        ``(13, "C")``; ``"12C14N"`` -> ``(12, "C")`` and ``(14, "N")``).

    Returns
    -------
    list[tuple[int, str]]
        List of ``(mass_number, element)`` tuples. Empty if ``name`` is
        unparseable.
    """
    s = str(name)
    matches = list(re.finditer(r"(\d+)([A-Z][a-z]?)", s))
    atoms: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        mass = int(m.group(1))
        elem = m.group(2)
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

    Parameters
    ----------
    iso_name : str, optional
        Isotopologue label, or ``None``.

    Returns
    -------
    bool
        ``True`` only for diatomics with two identical ``(mass, element)``
        atoms; ``False`` otherwise (including unparseable or non-diatomic labels).
    """
    if iso_name is None:
        return False
    atoms = _parse_isotopologue_atoms(iso_name)
    if len(atoms) != 2:
        return False
    return atoms[0] == atoms[1]


def is_atomic_species(iso_name: str | None) -> bool:
    """Detect whether an isotopologue label denotes an atomic (non-diatomic) species.

    Atomic species have no rotational structure and therefore no rotational
    collision channels in this framework. Labels containing ``"Fe"`` are
    treated as atomic (matching the packaged default Fe line list). Any label
    that parses to a single atom (e.g., ``"56Fe"``) is also classified as atomic.

    Parameters
    ----------
    iso_name : str, optional
        Isotopologue label, or ``None``.

    Returns
    -------
    bool
        ``True`` for atomic species, ``False`` otherwise.
    """
    if iso_name is None:
        return False
    if "Fe" in str(iso_name):
        return True
    atoms = _parse_isotopologue_atoms(iso_name)
    return len(atoms) == 1


def diatomic_symmetry_class(iso_name: str | None) -> str:
    r"""Classify a diatomic isotopologue label for collision selection rules.

    Parameters
    ----------
    iso_name : str, optional
        Isotopologue label such as ``"12C2"``, ``"12C13C"``,
        ``"12C14N"``, or ``None``.

    Returns
    -------
    str
        One of:

        * ``"homonuclear"`` -- two identical ``(mass, element)`` atoms (e.g.,
          ``"12C2"``, ``"13C2"``). Only :math:`|\Delta J| = 2` collisions are physical
          (nuclear-spin manifold preserved).
        * ``"same_element_heteronuclear"`` -- same element, different isotopes
          (e.g., ``"12C13C"``). Symmetry is broken so all J exist,
          but the underlying near-symmetric structure makes both :math:`|\Delta J| = 1`
          and :math:`|\Delta J| = 2` channels relevant.
        * ``"heteronuclear"`` -- different elements (e.g., ``"12C14N"``). Treat as a generic heteronuclear with the caller-provided
          ``dJ_allowed`` (defaults to :math:`|\Delta J| = 1` only).
        * ``"unknown"`` -- label could not be parsed as a diatomic.
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


def precompute_collision_scaffold(
    lines_out: Any,
    idx_to_level: dict,
    *,
    upper_id_col: str = "upper_id",
    lower_id_col: str = "lower_id",

    lower_es_col: str = "lower_es",
    lower_v_col: str = "lower_v",
    lower_J_col: str = "lower_J",
    lower_sym_col: str = "lower_sym",
    E_lower_cm1_col: str = "E_lower_cm1",

    include_deltaJ0_parity_mix: bool = True,
    require_X_only: bool = True,

    iso_name: str | None = None,
    homonuclear: bool | None = None,
    dJ_allowed: Sequence[int] = (1,),
) -> dict:
    r"""Build a rotational-collision scaffold from explicit lower-state columns.

    Parameters
    ----------
    lines_out : Any
        Annotated transition table.
    idx_to_level : dict
        Mapping from matrix index to level identifier.
    upper_id_col : str, optional, default "upper_id"
        Upper-level ID column name.
    lower_id_col : str, optional, default "lower_id"
        Lower-level ID column name.
    lower_es_col : str, optional, default "lower_es"
        Lower electronic-state column name.
    lower_v_col : str, optional, default "lower_v"
        Lower vibrational-state column name.
    lower_J_col : str, optional, default "lower_J"
        Lower rotational-state column name.
    lower_sym_col : str, optional, default "lower_sym"
        Lower-state symmetry/parity column name.
    E_lower_cm1_col : str, optional, default "E_lower_cm1"
        Lower-state energy column name in cm^-1.
    include_deltaJ0_parity_mix : bool, optional, default True
        Allow ``Delta J = 0`` collisions between
        sublevels with different ``sym`` label at the same J. Fires when either
        the molecule is truly heteronuclear (different elements, e.g. CN, OH) OR
        the isotopologue has at least one nucleus with nonzero spin (hyperfine
        structure, e.g. 13C2, 12C13C). Ignored only for strictly zero-hyperfine
        homonuclear species like 12C2.
    require_X_only : bool, optional, default True
        Restrict collisions to the lower electronic state. The
        lower state is auto-detected as the ``lower_es`` label whose minimum
        ``E_lower_cm1`` is smallest, so any spectroscopic notation works
        (``"X"``, ``"X1Sigmag+"``, etc.).
    iso_name : str, optional, default None
        Optional isotopologue label (e.g., ``"12C2"``, ``"13C2"``,
        ``"12C13C"``, ``"12C14N"``) used to auto-classify the diatomic via
        :func:`diatomic_symmetry_class`. Drives the ``|Delta J|`` rule:

        * homonuclear -> :math:`|\Delta J|=2` only
        * same-element heteronuclear (e.g., ``"12C13C"``) -> :math:`|\Delta J|=1` and :math:`|\Delta J|=2`
        * heteronuclear different elements (e.g., ``"12C14N"``) -> uses
          ``dJ_allowed``, which defaults to :math:`|\Delta J|=1` only (Arbitrary choice).

        Ignored when ``homonuclear`` is given explicitly.
    homonuclear : bool, optional, default None
        Explicit override for nuclear symmetry. ``True`` forces
        ``|Delta J| = 2`` only and disables ``include_deltaJ0_parity_mix`` (preserves
        nuclear-spin manifold). ``False`` uses ``dJ_allowed``. ``None`` auto-detects
        from ``iso_name``.
    dJ_allowed : Sequence[int], optional, default (1,)
        Allowed ``|Delta J|`` values for the heteronuclear
        different-element case (e.g., CN). Defaults to ``(1,)`` to match Manfroid 2009 [1]_. Ignored for homonuclear (forced to ``(2,)``) and for
        same-element heteronuclear (forced to ``(1, 2)``).

    Returns
    -------
    dict
        Scaffold dictionary mapping pre-computed collisional pair data to NumPy
        arrays. All arrays share the same length (one entry per allowed
        collisional pair) and are aligned by index. Keys:

        * ``iu`` (:class:`numpy.ndarray` of :class:`int`) -- Matrix index of the upper level of each collisional pair.
        * ``il`` (:class:`numpy.ndarray` of :class:`int`) -- Matrix index of the lower level of each collisional pair.
        * ``gu`` (:class:`numpy.ndarray` of :class:`float`) -- Degeneracy of the upper level.
        * ``gl`` (:class:`numpy.ndarray` of :class:`float`) -- Degeneracy of the lower level.
        * ``dE`` (:class:`numpy.ndarray` of :class:`float`) -- Energy gap :math:`E_u - E_l` between the two levels, in :math:`\mathrm{cm}^{-1}` (always non-negative by construction).

    Raises
    ------
    ValueError
        If required columns are missing.

    References
    ----------
        .. [1] Manfroid, J., Jehin, E., Hutsemékers, D., et al. 2009, A&A, 503, 613 (`link <https://ui.adsabs.harvard.edu/abs/2009A&A...503..613M>`_)
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
        except (KeyError, IndexError, AttributeError, TypeError):
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

    level_id_to_idx: dict[str, int] = {}
    for i, v in idx_to_level.items():
        level_id_to_idx[str(v)] = int(i)

    lower_ids = col_as_array(lines_out, lower_id_col).astype(str)
    les  = col_as_array(lines_out, lower_es_col).astype(str)
    lv   = col_as_array(lines_out, lower_v_col).astype(float)
    lJ   = col_as_array(lines_out, lower_J_col).astype(float)
    lsym = col_as_array(lines_out, lower_sym_col).astype(str)
    Elow = col_as_array(lines_out, E_lower_cm1_col).astype(float)

    good = np.isfinite(lv) & np.isfinite(lJ) & np.isfinite(Elow)
    lower_ids, les, lv, lJ, lsym, Elow = lower_ids[good], les[good], lv[good], lJ[good], lsym[good], Elow[good]

    if require_X_only and lower_ids.size > 0:
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

    has_hyperfine = _has_nonzero_nuclear_spin(iso_name)

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

    key_to_idx = {}
    for k in unique_keys:
        lid = id_by_key[k]
        if lid in level_id_to_idx:
            key_to_idx[k] = level_id_to_idx[lid]

    ground = [k for k in unique_keys if k in key_to_idx]
    if len(ground) < 2:
        return _empty_scaffold()

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
        dE=np.asarray(dE_list, float),
    )

def precompute_collision_scaffold_fast(*args, **kwargs) -> dict:
    """Build collision scaffold and add cached arrays for fast updates.

    Parameters
    ----------
    args : tuple
        Positional arguments forwarded to :func:`precompute_collision_scaffold`.
    kwargs : dict
        Keyword arguments forwarded to :func:`precompute_collision_scaffold`.

    Returns
    -------
    dict
        Collision scaffold with ``dE_over_k_K`` and ``gu_over_gl`` caches.
    """
    sc = precompute_collision_scaffold(*args, **kwargs)

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

def apply_collisions_inplace(M: np.ndarray, scaffold: Dict[str, np.ndarray], *, Q: float, T: float) -> np.ndarray:
    r"""Apply rotational-collision rates to a matrix in place.

    The downward rate :math:`Q \equiv C_{ul}` (in s\ :sup:`-1`) is assumed
    identical for every upper-to-lower pair listed in ``scaffold``. It
    corresponds to :math:`10^{f_{\rm col}}` where the user-facing parameter
    is :math:`f_{\rm col} \equiv \log_{10}(C_{ul}\,/\,\mathrm{s}^{-1})`.

    Parameters
    ----------
    M : numpy.ndarray
        Rate matrix to modify.
    scaffold : dict[str, numpy.ndarray]
        Collision scaffold containing ``iu``, ``il``, ``gu``, ``gl``, and ``dE``.
    Q : float
        Downward collision rate :math:`C_{ul}` in s\ :sup:`-1`, applied
        uniformly to every upper-to-lower pair in ``scaffold``.
    T : float
        Kinetic temperature in K.

    Returns
    -------
    numpy.ndarray
        The modified matrix ``M``.
    """
    if scaffold.get("iu", np.array([])).size == 0 or Q <= 0:
        return M

    iu = scaffold["iu"]
    il = scaffold["il"]
    gu = scaffold["gu"]
    gl = scaffold["gl"]
    dE_cm1 = scaffold["dE"]

    kT = (const.k_B * (T * u.K)).to(u.erg).value

    dE_erg = (const.h * const.c * (dE_cm1 / u.cm)).to(u.erg).value

    Cdown = float(Q)  
    Cup = (gu / gl) * Cdown * np.exp(-dE_erg / kT)

    np.add.at(M, (iu, iu), -Cdown)
    np.add.at(M, (il, il), -Cup)
    np.add.at(M, (il, iu),  Cdown)
    np.add.at(M, (iu, il),  Cup)
    return M

def apply_collisions_inplace_fast(
    M: np.ndarray,
    scaffold: Dict[str, np.ndarray],
    *,
    Q: float,
    T: float,
    Cup_work: np.ndarray,
) -> np.ndarray:
    r"""Apply collisions in place using cached arrays and reusable buffers.

    Same semantics as :func:`apply_collisions_inplace`: the downward rate
    :math:`Q \equiv C_{ul}` (in s\ :sup:`-1`) is assumed identical for every
    upper-to-lower pair in ``scaffold``, and corresponds to :math:`10^{f_{\rm col}}`
    with :math:`f_{\rm col} \equiv \log_{10}(C_{ul}\,/\,\mathrm{s}^{-1})`.

    Parameters
    ----------
    M : numpy.ndarray
        Rate matrix to modify.
    scaffold : dict[str, numpy.ndarray]
        Collision scaffold with ``dE_over_k_K`` and ``gu_over_gl``.
    Q : float
        Downward collision rate :math:`C_{ul}` in s\ :sup:`-1`, applied
        uniformly to every upper-to-lower pair in ``scaffold``.
    T : float
        Kinetic temperature in K.
    Cup_work : numpy.ndarray
        Reusable working array for upward rates.

    Returns
    -------
    numpy.ndarray
        The modified matrix ``M``.
    """
    iu = scaffold.get("iu", None)
    if iu is None or iu.size == 0 or Q <= 0.0 or not np.isfinite(T) or T <= 0.0:
        return M

    il = scaffold["il"]
    dE_over_k_K = scaffold["dE_over_k_K"]
    gu_over_gl = scaffold["gu_over_gl"]

    np.divide(dE_over_k_K, T, out=Cup_work)    
    np.negative(Cup_work, out=Cup_work)         
    np.exp(Cup_work, out=Cup_work)              
    Cup_work *= gu_over_gl                      
    Cup_work *= Q                              

    Cdown = Q

    np.add.at(M, (iu, iu), -Cdown)
    np.add.at(M, (il, il), -Cup_work)
    np.add.at(M, (il, iu),  Cdown)
    np.add.at(M, (iu, il),  Cup_work)
    return M
