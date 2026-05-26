"""Serialization implementations extracted from :class:`FluorescenceModel`.

Provides the bodies of :meth:`FluorescenceModel.save` and
:meth:`FluorescenceModel.load`.
"""
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fluorescence import FluorescenceModel


def save(model: "FluorescenceModel", filename: str) -> None:
    """Implementation of FluorescenceModel.save."""
    had_given_lsf = (model.lsf_method == "Given")
    init_kwargs = dict(
        data=model.data,
        window=model.window,
        pumping=model.pumping,
        isotopologues=model.isotopologues,
        systems=model.systems,
        linelists=model.linelists,
        line_path=model.line_path,
        lsf=None,
        lsf_method=model.lsf_method if not had_given_lsf else "Gauss",
        A_min=model.A_min,
        a=model.a,
        name=model.name,
        sigma=model.sigma,
        sigma1=model.sigma1,
        sigma2=model.sigma2,
        sigma_G=model.sigma_G,
        fwhm_L=model.fwhm_L,
        ratio=model.ratio,
        logN=model.logN,
        logN_by_iso=model.logN_by_iso,
        logQ=model.logQ,
        logQ_by_iso=model.logQ_by_iso,
        T=model.T,
        T_by_iso=model.T_by_iso,
        v_kms=model.v_kms,
        v_kms_by_iso=model.v_kms_by_iso,
        dlam=model.dlam,
        dlam_by_iso=model.dlam_by_iso,
        wave_col=model.wave_col,
        flux_col=model.flux_col,
        error_col=model.error_col,
        continuum_col=model.continuum_col,
        omega=model.omega,
        include_rotations=model.include_rotations,
        pumping_v_kms=model.pumping_v_kms,
        pumping_dlam_A=model.pumping_dlam_A,
        model_wave=model.model_wave,
    )

    mcmc_result = dict(
        priors=model.priors,
        param_keys=model.param_keys,
        median_params=model.median_params,
        up_errors_params=model.up_errors_params,
        low_errors_params=model.low_errors_params,
        samples_pruned=model.samples_pruned,
        lnprob_pruned=model.lnprob_pruned,
        model_wave=model.model_wave,
        median_model=model.median_model,
        best_model=model.best_model,
        model_p16=model.model_p16,
        model_p84=model.model_p84,
        model_by_iso=model.model_by_iso,
    )

    derived = dict(
        q=model.q,
        q_err=model.q_err,
        q_seeing_corrected=model.q_seeing_corrected,
        logN_seeing_corrected=model.logN_seeing_corrected,
        logN_err=model.logN_err,
        logN_err_by_iso=model.logN_err_by_iso,
    )

    state = {
        "class": "FluorescenceModel",
        "version": 1,
        "init_kwargs": init_kwargs,
        "mcmc_result": mcmc_result,
        "derived": derived,
        "had_given_lsf": had_given_lsf,
    }

    with open(filename, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(cls, filename: str) -> "FluorescenceModel":
    """Implementation of FluorescenceModel.load."""
    with open(filename, "rb") as f:
        state = pickle.load(f)

    if state.get("class") != "FluorescenceModel":
        raise ValueError("File does not contain a FluorescenceModel state.")
    version = state.get("version")
    if version != 1:
        raise ValueError(
            f"Unsupported FluorescenceModel state version: {version!r} (expected 1)."
        )

    init_kwargs = state["init_kwargs"]
    mcmc_result = state.get("mcmc_result") or {}
    derived = state.get("derived") or {}
    had_given_lsf = state.get("had_given_lsf", False)
    obj = cls(**init_kwargs)

    saved_priors = mcmc_result.get("priors")
    if saved_priors:
        obj.priors = dict(saved_priors)

    if any(v is not None for k, v in mcmc_result.items() if k != "priors"):
        obj._update_from_result(
            mcmc_result,
            used_lsf=None,
            used_lsf_method=init_kwargs.get("lsf_method"),
        )

    obj.q = derived.get("q", None)
    obj.q_err = derived.get("q_err", None)
    obj.q_seeing_corrected = derived.get("q_seeing_corrected", False)
    obj.logN_seeing_corrected = derived.get("logN_seeing_corrected", False)
    if derived.get("logN_err") is not None:
        obj.logN_err = derived["logN_err"]
    if derived.get("logN_err_by_iso") is not None:
        obj.logN_err_by_iso = derived["logN_err_by_iso"]

    if had_given_lsf:
        print(
            "Warning: original model used a custom 'Given' LSF which "
            "was not serialized. Call `obj.update_model(lsf=...)` to restore it."
        )
    return obj
