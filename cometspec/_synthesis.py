"""Synthesis-related implementations extracted from :class:`FluorescenceModel`.

Provides the bodies of :meth:`FluorescenceModel.update_model` and
:meth:`FluorescenceModel._synthesize_model`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import modeling
from .linelist import normalize_cn_systems_arg
from .rates import (
    solve_with_normalization_fast,
    g_factors_fast_from_cache,
    synth_spectrum_from_lines,
)
from .collisions import apply_collisions_inplace_fast

import astropy.constants as const
H_CGS: float = const.h.cgs.value

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fluorescence import FluorescenceModel


def apply_update_model(
    model: "FluorescenceModel",
    *,
    isotopologues: Union[str, Sequence[str], None] = None,
    systems: Union[str, Sequence[str], None] = None,
    linelists: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], Sequence[pd.DataFrame]]] = None,
    logN: Optional[float] = None,
    logN_by_iso: Optional[Dict[str, float]] = None,
    logQ: Optional[float] = None,
    T: Optional[float] = None,
    T_by_iso: Optional[Dict[str, float]] = None,
    v_kms: Optional[float] = None,
    v_kms_by_iso: Optional[Dict[str, float]] = None,
    dlam: Optional[float] = None,
    dlam_by_iso: Optional[Dict[str, float]] = None,
    A_min: Optional[float] = None,
    lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lsf_method: Optional[str] = None,
    sigma: Optional[float] = None,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    sigma_G: Optional[float] = None,
    fwhm_L: Optional[float] = None,
    ratio: Optional[float] = None,
    window: Optional[Tuple[float, float]] = None,
    pumping: Any = None,
    data: Any = None,
    wave_col: str = None,
    flux_col: str = None,
    error_col: str = None,
    continuum_col: str = None,
    omega: float = np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2,
    N_Model: int = 20000,
) -> None:
    """Implementation of FluorescenceModel.update_model."""
    if data is not None:
        model.data = data
    if pumping is not None:
        model.pumping = pumping
    if window is not None:
        model.window = window

    # --- selection updates ---
    if isotopologues is not None:
        model.isotopologues = isotopologues
    if systems is not None:
        model.systems = systems
    if linelists is not None:
        model.linelists = linelists

    # --- physical updates ---
    if logN is not None:
        model.logN = float(logN)
    if logN_by_iso is not None:
        model.logN_by_iso = dict(logN_by_iso)
    if logQ is not None:
        model.logQ = float(logQ)
    if T is not None:
        model.T = float(T)
    if T_by_iso is not None:
        model.T_by_iso = dict(T_by_iso)
    if v_kms is not None:
        model.v_kms = float(v_kms)
    if v_kms_by_iso is not None:
        model.v_kms_by_iso = dict(v_kms_by_iso)
    if dlam is not None:
        model.dlam = float(dlam)
    if dlam_by_iso is not None:
        model.dlam_by_iso = dict(dlam_by_iso)

    if A_min is not None:
        model.A_min = float(A_min)

    # --- LSF handling (same as your previous logic) ---
    if lsf is not None:
        model.lsf = lsf
        model.lsf_method = "Given"
        for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
            setattr(model, name, None)
    elif lsf_method is not None:
        model.lsf_method = lsf_method
        if lsf_method == "Gauss":
            model.sigma = float(sigma if sigma is not None else 0.01)
            model.lsf = modeling.make_lsf({"sigma": model.sigma}, "Gauss")
            model.sigma1 = model.sigma2 = model.sigma_G = model.fwhm_L = model.ratio = None
        elif lsf_method == "2Gauss":
            if sigma1 is None or sigma2 is None or ratio is None:
                raise ValueError("sigma1, sigma2, ratio required for '2Gauss'.")
            model.sigma1 = float(sigma1)
            model.sigma2 = float(sigma2)
            model.ratio = float(ratio)
            model.lsf = modeling.make_lsf(
                {"sigma1": model.sigma1, "sigma2": model.sigma2, "ratio": model.ratio},
                "2Gauss",
            )
            model.sigma = model.sigma_G = model.fwhm_L = None
        elif lsf_method == "Gauss_Lorentz":
            if sigma_G is None or fwhm_L is None or ratio is None:
                raise ValueError("sigma_G, fwhm_L, ratio required for 'Gauss_Lorentz'.")
            model.sigma_G = float(sigma_G)
            model.fwhm_L = float(fwhm_L)
            model.ratio = float(ratio)
            model.lsf = modeling.make_lsf(
                {"sigma_G": model.sigma_G, "fwhm_L": model.fwhm_L, "ratio": model.ratio},
                "Gauss_Lorentz",
            )
            model.sigma = model.sigma1 = model.sigma2 = None
        elif lsf_method == "Lorentz":
            if fwhm_L is None:
                raise ValueError("fwhm_L required for 'Lorentz'.")
            model.fwhm_L = float(fwhm_L)
            model.lsf = modeling.make_lsf({"fwhm_L": model.fwhm_L}, "Lorentz")
            model.sigma = model.sigma1 = model.sigma2 = model.sigma_G = model.ratio = None
        else:
            raise ValueError(f"Unsupported lsf_method: {lsf_method}")
    else:
        # rebuild from stored params
        if model.lsf_method == "Gauss":
            if model.sigma is None:
                model.sigma = 0.01
            model.lsf = modeling.make_lsf({"sigma": model.sigma}, "Gauss")
        elif model.lsf_method == "2Gauss":
            model.lsf = modeling.make_lsf(
                {"sigma1": model.sigma1, "sigma2": model.sigma2, "ratio": model.ratio},
                "2Gauss",
            )
        elif model.lsf_method == "Gauss_Lorentz":
            model.lsf = modeling.make_lsf(
                {"sigma_G": model.sigma_G, "fwhm_L": model.fwhm_L, "ratio": model.ratio},
                "Gauss_Lorentz",
            )
        elif model.lsf_method == "Lorentz":
            model.lsf = modeling.make_lsf({"fwhm_L": model.fwhm_L}, "Lorentz")

    # NEW: if logN or isotope selection changed, production rate is stale
    if (logN is not None) or (logN_by_iso is not None) or (isotopologues is not None):
        model.q = None
        model.q_err = None

    if omega is not None:
        model.omega = omega
    if wave_col is not None:
        model.wave_col = wave_col
    if flux_col is not None:
        model.flux_col = flux_col
    if error_col is not None:
        model.error_col = error_col
    if continuum_col is not None:
        model.continuum_col = continuum_col

    model._synthesize_model()


def synthesize_model(model: "FluorescenceModel") -> None:
    """Implementation of FluorescenceModel._synthesize_model."""
    if model.pumping is None:
        raise ValueError("Pumping spectrum is required.")
    if model.window is None:
        raise ValueError("window is required.")

    iso_list = model._iso_list()

    # 1) transitions: user-provided isos win; the rest fall back to defaults.
    # Defaults match mcmc.py model_flux path: normalize systems, use pumping
    # wavelength range for default-linelist filtering, no line_paths.
    sys_tokens = normalize_cn_systems_arg(model.systems)
    pump_wave = np.asarray(model.pumping["WAVE"], dtype=float)
    trans_by_iso = modeling.resolve_linelists_with_defaults(
        model.linelists,
        iso_list,
        systems=sys_tokens,
        A_min=model.A_min,
        use_omega_labels=False,
        lambda_min_A=float(pump_wave.min()),
        lambda_max_A=float(pump_wave.max()),
    )
    # 2) per-iso solve + sum spectrum
    lines_by_iso: Dict[str, Any] = {}
    M_by_iso: Dict[str, np.ndarray] = {}
    idx_by_iso: Dict[str, Any] = {}
    n_by_iso: Dict[str, np.ndarray] = {}
    gph_by_iso: Dict[str, np.ndarray] = {}
    gen_by_iso: Dict[str, np.ndarray] = {}
    gphsum_by_iso: Dict[str, float] = {}
    gensum_by_iso: Dict[str, float] = {}
    model_by_iso: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if model.model_wave is None:
        wave = np.arange(model.window[0], model.window[1] + 0.01, 0.01, dtype=float)
    else:
        wave = np.asarray(model.model_wave, float)
    model.model_wave = wave
    spec_total = np.zeros_like(wave, dtype=float)

    for iso, df_trans in trans_by_iso.items():
        # ✅ Pumping shift consistent with fitter (affects J_nu and thus line ratios)
        lines_theta = modeling.attach_pumping_and_labels(
            df_trans,
            model.pumping,
            line_v_kms=float(model.pumping_v_kms),
            line_dlam_A=float(model.pumping_dlam_A),
            lsf_for_Jnu=None,
            lam_col="lambda_vac_A",
        )

        M_rad, idx_to_level, lines_out = modeling.build_rate_matrix_nbar(
            lines_theta,
            include_stim_emission=True,
            verbose=False,
            A_col="A_ul",
            upper_id_col="upper_id",
            lower_id_col="lower_id",
            g_upper_col="g_upper",
            g_lower_col="g_lower",
        )

        if model.include_rotations and not modeling.is_atomic_species(iso):
            coll_scaf = modeling.precompute_collision_scaffold_fast(
                lines_out, idx_to_level, iso_name=iso,
            )
        else:
            coll_scaf = dict(iu=np.array([], int), il=np.array([], int),
                            gu=np.array([]), gl=np.array([]), dE=np.array([]))

        M = M_rad.copy()

        # T fallback mirrors mcmc._T_for_iso: prefer by_iso, else shared T.
        T_i = None
        if model.T_by_iso is not None and iso in model.T_by_iso:
            T_i = float(model.T_by_iso[iso])
        elif model.T is not None:
            T_i = float(model.T)

        # v_kms / dlam fallback mirrors mcmc._v_kms_for_iso / _dlam_for_iso:
        # by_iso wins, else shared, else 0.0 (always float).
        if model.v_kms_by_iso is not None and iso in model.v_kms_by_iso:
            v_kms_i = float(model.v_kms_by_iso[iso])
        elif model.v_kms is not None:
            v_kms_i = float(model.v_kms)
        else:
            v_kms_i = 0.0

        if model.dlam_by_iso is not None and iso in model.dlam_by_iso:
            dlam_i = float(model.dlam_by_iso[iso])
        elif model.dlam is not None:
            dlam_i = float(model.dlam)
        else:
            dlam_i = 0.0

        # logQ fallback mirrors mcmc._logQ_for_iso (one unified branch,
        # independent of how many isotopologues are in iso_list).
        logQ_i = None
        if model.logQ_by_iso is not None and iso in model.logQ_by_iso:
            try:
                logQ_i = float(model.logQ_by_iso[iso])
            except TypeError:
                logQ_i = None
        elif model.logQ is not None:
            logQ_i = float(model.logQ)

        if logQ_i is not None and T_i is not None and model.include_rotations:
            Q_lin = 10.0 ** logQ_i
            if np.isfinite(Q_lin) and Q_lin > 0.0:
                Cup_work = np.empty_like(coll_scaf.get("iu", np.array([], dtype=int)), dtype=float)
                apply_collisions_inplace_fast(M, coll_scaf, Q=Q_lin, T=T_i, Cup_work=Cup_work)

        # Solver + g-factors via the same fast paths used by mcmc.model_flux.
        A_work = np.empty_like(M)
        b_work = np.zeros(M.shape[0], dtype=float)
        n = solve_with_normalization_fast(M, A_work, b_work)

        ui = np.asarray(lines_out["__upper_idx"], dtype=np.int64)
        A_ul = np.asarray(lines_out["A_ul"], dtype=np.float64)
        nu = np.asarray(lines_out["__nu_Hz"], dtype=np.float64)
        hnu = H_CGS * nu
        g_ph = np.empty_like(A_ul, dtype=float)
        g_en = np.empty_like(A_ul, dtype=float)
        g_ph, g_en = g_factors_fast_from_cache(
            ui=ui, A_ul=A_ul, hnu=hnu, n=n, out_g_ph=g_ph, out_g_en=g_en,
        )
        g_ph_sum = float(g_ph.sum())
        g_en_sum = float(g_en.sum())

        if model.logN_by_iso is not None and iso in model.logN_by_iso:
            logN_i = float(model.logN_by_iso[iso])
        elif model.logN is not None:
            logN_i = float(model.logN)
        else:
            raise ValueError(f"No logN available for isotopologue {iso!r}. Set logN or logN_by_iso.")

        # ✅ Emission shift is separate, applied in spectrum synthesis (same as fit model_flux)
        _, spec_i = synth_spectrum_from_lines(
            lines_out,
            g_line_energy=g_en,
            lam_min=float(wave.min()),
            lam_max=float(wave.max()),
            lam_col="Wave_vac_AA",
            N_col_cm2=10.0 ** logN_i,
            Omega_sr=model.omega,
            grid=wave,
            lsf=model.lsf,
            v_shift_kms=v_kms_i,
            dlam_shift_A=dlam_i,
        )

        spec_total += spec_i

        lines_by_iso[iso] = lines_out
        M_by_iso[iso] = M
        idx_by_iso[iso] = idx_to_level
        n_by_iso[iso] = n
        gph_by_iso[iso] = g_ph
        gen_by_iso[iso] = g_en
        gphsum_by_iso[iso] = g_ph_sum
        gensum_by_iso[iso] = g_en_sum
        model_by_iso[iso] = (wave, spec_i)


    model.lines_by_iso = lines_by_iso
    model.M_by_iso = M_by_iso
    model.idx_to_level_by_iso = idx_by_iso
    model.n_by_iso = n_by_iso
    model.g_ph_by_iso = gph_by_iso
    model.g_en_by_iso = gen_by_iso
    model.g_ph_sum_by_iso = gphsum_by_iso
    model.g_en_sum_by_iso = gensum_by_iso
    model.model_by_iso = model_by_iso

    if len(iso_list) == 1:
        iso0 = iso_list[0]
        model.lines = lines_by_iso[iso0]
        model.M = M_by_iso[iso0]
        model.idx_to_level = idx_by_iso[iso0]
        model.n = n_by_iso[iso0]
        model.g_ph = gph_by_iso[iso0]
        model.g_en = gen_by_iso[iso0]
        model.g_ph_sum = gphsum_by_iso[iso0]
        model.g_en_sum = gensum_by_iso[iso0]
    else:
        model.lines = None
        model.M = None
        model.idx_to_level = None
        model.n = None
        model.g_ph = None
        model.g_en = None
        model.g_ph_sum = None
        model.g_en_sum = None

    model.model_wave = wave
    model.best_model = spec_total
