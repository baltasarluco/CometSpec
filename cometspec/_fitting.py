"""Fitting-related implementations extracted from :class:`FluorescenceModel`.

Provides the bodies of :meth:`FluorescenceModel.fit_mcmc` and
:meth:`FluorescenceModel._update_from_result`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import modeling

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fluorescence import FluorescenceModel


def run_fit_mcmc(
    model: "FluorescenceModel",
    data: Optional[Any] = None,
    window: Optional[Tuple[float, float]] = None,
    *,
    pumping: Any = None,
    isotopologues: Union[str, Sequence[str], None] = None,
    systems: Union[str, Sequence[str], None] = None,
    linelists: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], Sequence[pd.DataFrame]]] = None,
    nwalkers: int = 20,
    nsteps: int = 1000,
    n_cores: Optional[int] = None,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lsf_method: Optional[str] = None,
    make_plots: bool = True,
    progress: bool = True,
    A_min: Optional[float] = None,
    a: Optional[float] = None,
    fig_file: str = "mcmc_fit",
    verbose: bool = True,
    pruning: bool = True,
    N_Model: Optional[int] = 20000,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Implementation of FluorescenceModel.fit_mcmc."""
    if data is not None:
        model.data = data
    if model.data is None:
        raise ValueError("No data attached to this FluorescenceModel.")

    if window is None:
        if model.window is None:
            raise ValueError("window must be provided (argument or instance.window).")
        window = model.window
    else:
        model.window = window

    if pumping is None:
        pumping = model.pumping
    else:
        model.pumping = pumping
    if pumping is None:
        raise ValueError("pumping must be provided.")

    if isotopologues is not None:
        model.isotopologues = isotopologues
    if systems is not None:
        model.systems = systems
    if linelists is not None:
        model.linelists = linelists

    if priors is None:
        priors = model.priors or {"logN": (9.0, 15.0), "f_col": (-5.0, 0.0), "T": (10.0, 1000.0)}
    model.priors = priors

    if lsf_method is None:
        lsf_method = model.lsf_method

    if A_min is None:
        A_min = model.A_min
    else:
        model.A_min = float(A_min)

    if a is None:
        a = model.a
    else:
        model.a = float(a)

    if n_cores is None:
        n_cores = getattr(model, "n_cores", None)
    else:
        model.n_cores = n_cores

    # Canonical fallback bridge:
    # If priors do not sample a parameter, pass instance defaults to
    # modeling.mcmc_fitting so this wrapper remains the source of truth.
    init_f_col = float(model.f_col) if model.f_col is not None else None
    init_T = float(model.T) if model.T is not None else 300.0
    init_T_by_iso = dict(model.T_by_iso) if model.T_by_iso is not None else None
    init_v_kms = float(model.v_kms) if model.v_kms is not None else 0.0
    init_v_kms_by_iso = dict(model.v_kms_by_iso) if model.v_kms_by_iso is not None else None
    init_dlam = float(model.dlam) if model.dlam is not None else 0.0
    init_dlam_by_iso = dict(model.dlam_by_iso) if model.dlam_by_iso is not None else None
    init_f_col_by_iso = dict(model.f_col_by_iso) if model.f_col_by_iso is not None else None
    init_logN_by_iso = dict(model.logN_by_iso) if model.logN_by_iso is not None else None
    init_logN = float(model.logN) if model.logN is not None else 11.0
    init_sigma = float(model.sigma) if model.sigma is not None else None
    init_sigma1 = float(model.sigma1) if model.sigma1 is not None else None
    init_sigma2 = float(model.sigma2) if model.sigma2 is not None else None
    init_sigma_G = float(model.sigma_G) if model.sigma_G is not None else None
    init_fwhm_L = float(model.fwhm_L) if model.fwhm_L is not None else None
    init_ratio = float(model.ratio) if model.ratio is not None else None


    # Pumping-shift bridge:
    # These settings control J_nu sampling and are intentionally independent
    # from emission-shift parameters that can also be sampled.
    pumping_v_kms = float(getattr(model, "pumping_v_kms", 0.0))
    pumping_dlam_A = float(getattr(model, "pumping_dlam_A", 0.0))

    result = modeling.mcmc_fitting(
        model.data,
        window,
        pumping=pumping,
        isotopologues=model.isotopologues,
        systems=model.systems,
        linelists=model.linelists,
        nwalkers=nwalkers,
        nsteps=nsteps,
        n_cores=n_cores,
        priors=priors,
        lsf=lsf,
        lsf_method=lsf_method,
        make_plots=make_plots,
        progress=progress,
        A_min=float(A_min),
        a=float(a),

        # ✅ Pumping shift (affects J_nu → line ratios)
        velocity_kms=pumping_v_kms,
        delta_lambda_A=pumping_dlam_A,

        # ✅ Fallbacks if not fit
        init_f_col_by_iso=init_f_col_by_iso,
        init_f_col=init_f_col,
        init_T=init_T,
        init_T_by_iso=init_T_by_iso,
        init_v_kms=init_v_kms,
        init_v_kms_by_iso=init_v_kms_by_iso,
        init_dlam=init_dlam,
        init_dlam_by_iso=init_dlam_by_iso,
        init_logN_by_iso=init_logN_by_iso,
        init_logN=init_logN,
        init_sigma=init_sigma,
        init_sigma1=init_sigma1,
        init_sigma2=init_sigma2,
        init_sigma_G=init_sigma_G,
        init_fwhm_L=init_fwhm_L,
        init_ratio=init_ratio,

        fig_file=fig_file,
        wave_col=model.wave_col,
        flux_col=model.flux_col,
        error_col=model.error_col,
        continuum_col=model.continuum_col,
        omega=model.omega,
        verbose=verbose,
        pruning=pruning,
        include_rotations=model.include_rotations,
        N_Model=N_Model,
        config=config,
    )


    model._update_from_result(result, used_lsf=lsf, used_lsf_method=lsf_method)

    model.pumping_v_kms = pumping_v_kms
    model.pumping_dlam_A = pumping_dlam_A

    return result


def update_from_result(
    model: "FluorescenceModel",
    result: Dict[str, Any],
    *,
    used_lsf: Optional[Callable[[np.ndarray], np.ndarray]],
    used_lsf_method: Optional[str],
) -> None:
    """Implementation of FluorescenceModel._update_from_result."""
    from .fluorescence import FluorescenceModel

    model.param_keys = tuple(result.get("param_keys", ()))
    model.median_params = dict(result.get("median_params", {}))
    model.up_errors_params = dict(result.get("up_errors_params", {}))
    model.low_errors_params = dict(result.get("low_errors_params", {}))

    model.samples_pruned = result.get("samples_pruned")
    model.lnprob_pruned = result.get("lnprob_pruned")

    model.logQ_seeing_corrected = False
    model.logN_seeing_corrected = False

    for name in ("logN", "f_col", "T", "v_kms", "dlam"):
        if name in model.median_params:
            setattr(model, name, float(model.median_params[name]))
            if name == "logN":
                model.logN_err = np.array((
                    float(model.up_errors_params.get(name, 0.0)),
                    float(model.low_errors_params.get(name, 0.0)),
                ))

    iso_list = model._iso_list()
    for iso in iso_list:
        key = f"logN_{iso}"
        if key in model.median_params:
            if model.logN_by_iso is None:
                model.logN_by_iso = {}
            if model.logN_err_by_iso is None:
                model.logN_err_by_iso = {}
            model.logN_by_iso[iso] = float(model.median_params[key])
            model.logN_err_by_iso[iso] = np.array((
                float(model.up_errors_params.get(key, 0.0)),
                float(model.low_errors_params.get(key, 0.0)),
            ))

    for iso in iso_list:
        key = f"T_{iso}"
        if key in model.median_params:
            if model.T_by_iso is None:
                model.T_by_iso = {}
            model.T_by_iso[iso] = float(model.median_params[key])

    for iso in iso_list:
        key = f"v_kms_{iso}"
        if key in model.median_params:
            if model.v_kms_by_iso is None:
                model.v_kms_by_iso = {}
            model.v_kms_by_iso[iso] = float(model.median_params[key])

    for iso in iso_list:
        key = f"dlam_{iso}"
        if key in model.median_params:
            if model.dlam_by_iso is None:
                model.dlam_by_iso = {}
            model.dlam_by_iso[iso] = float(model.median_params[key])

    if used_lsf is not None:
        model.lsf = used_lsf
        model.lsf_method = "Given"
        for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
            setattr(model, name, None)
    else:
        model.lsf_method = used_lsf_method
        if model.lsf_method == "Gauss":
            if "sigma" in model.median_params:
                model.sigma = float(model.median_params["sigma"])
            model.lsf = modeling.make_lsf({"sigma": model.sigma}, "Gauss")
        elif model.lsf_method == "2Gauss":
            vals = {}
            for nm in ("sigma1", "sigma2", "ratio"):
                if nm in model.median_params:
                    vals[nm] = float(model.median_params[nm])
                    setattr(model, nm, vals[nm])
            if len(vals) == 3:
                model.lsf = modeling.make_lsf(vals, "2Gauss")
        elif model.lsf_method == "Gauss_Lorentz":
            vals = {}
            for nm in ("sigma_G", "fwhm_L", "ratio"):
                if nm in model.median_params:
                    vals[nm] = float(model.median_params[nm])
                    setattr(model, nm, vals[nm])
            if len(vals) == 3:
                model.lsf = modeling.make_lsf(vals, "Gauss_Lorentz")
        elif model.lsf_method == "Lorentz":
            if "fwhm_L" in model.median_params:
                model.fwhm_L = float(model.median_params["fwhm_L"])
                model.lsf = modeling.make_lsf({"fwhm_L": model.fwhm_L}, "Lorentz")

    model.median_model = result.get("median_model", None)
    model.best_model = result.get("best_model", None)
    model.model_wave = result.get("model_wave", None)
    model.model_p16 = result.get("model_p16", None)
    model.model_p84 = result.get("model_p84", None)

    logN_keys = {"logN"} | {f"logN_{iso}" for iso in iso_list}
    if any(k in model.median_params for k in logN_keys):
        model.logQ = None
        model.logQ_err = 0

    # Match mcmc.py model_flux cache build: normalize systems, use the pumping
    # spectrum's wavelength range so default linelists contain the same lines
    # the MCMC cache used. Otherwise sum(model_by_iso) won't equal best_model.
    sys_tokens = modeling.normalize_cn_systems_arg(model.systems)
    pump_wave = np.asarray(model.pumping["WAVE"], dtype=float)
    resolved_linelists = modeling.resolve_linelists_with_defaults(
        model.linelists,
        iso_list,
        systems=sys_tokens,
        A_min=model.A_min,
        use_omega_labels=False,
        lambda_min_A=float(pump_wave.min()),
        lambda_max_A=float(pump_wave.max()),
    )

    for i in iso_list:
        non_iso_list = [j for j in iso_list if j != i]
        params_per_iso = {k: v for k, v in model.median_params.items() if not any(k == f"logN_{j}" for j in non_iso_list)}

        sub_model = FluorescenceModel(
            data=model.data,
            window=model.window,
            pumping=model.pumping,
            isotopologues=[i],
            systems=model.systems,
            linelists={i: resolved_linelists[i]},
            line_path=model.line_path,
            model_wave=model.model_wave,
            logN=params_per_iso.get(f"logN_{i}", params_per_iso.get("logN", model.logN_by_iso.get(i, model.logN) if model.logN_by_iso else model.logN)),
            f_col=params_per_iso.get(f"f_col_{i}", params_per_iso.get("f_col", model.f_col_by_iso.get(i, model.f_col) if model.f_col_by_iso else model.f_col)),
            T=params_per_iso.get(f"T_{i}", params_per_iso.get("T", model.T_by_iso.get(i, model.T) if model.T_by_iso else model.T)),
            v_kms=params_per_iso.get(f"v_kms_{i}", params_per_iso.get("v_kms", model.v_kms_by_iso.get(i, model.v_kms) if model.v_kms_by_iso else model.v_kms)),
            dlam=params_per_iso.get(f"dlam_{i}", params_per_iso.get("dlam", model.dlam_by_iso.get(i, model.dlam) if model.dlam_by_iso else model.dlam)),
            A_min=model.A_min,
            lsf=model.lsf,
            lsf_method=model.lsf_method,
            sigma=model.sigma,
            sigma1=model.sigma1,
            sigma2=model.sigma2,
            sigma_G=model.sigma_G,
            fwhm_L=model.fwhm_L,
            ratio=model.ratio,
            omega=model.omega,
            wave_col=model.wave_col,
            flux_col=model.flux_col,
            error_col=model.error_col,
            continuum_col=model.continuum_col,
            include_rotations=model.include_rotations,
        )
        model.model_by_iso[i] = (model.model_wave, sub_model.best_model)
