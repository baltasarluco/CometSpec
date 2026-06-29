"""Optional configuration dataclasses for high-level entry points.

These dataclasses provide a structured alternative to passing many keyword
arguments. They are *additive* and *fully backward compatible*:

- Every field defaults to a sentinel (:data:`UNSET`) meaning "not provided".
- When a config is passed to a function that accepts one, only fields whose
  value is not :data:`UNSET` override the corresponding individual keyword
  argument; all other behavior is unchanged.
- Existing call sites that pass individual keyword arguments continue to work but if they are in the config class they will be overridden by the config values.

Two configs can be used:

- :class:`FluorescenceModelConfig` -- groups :class:`FluorescenceModel`
  constructor options.
- :class:`MCMCFitConfig` -- groups :func:`cometspec.mcmc.mcmc_fitting`
  options.

Check their respective docstrings for details on the parameters, outputs or attributes.
Examples
--------
.. code-block:: python

    from cometspec.config import MCMCFitConfig
    from cometspec.mcmc import mcmc_fitting

    cfg = MCMCFitConfig(nwalkers=80, nsteps=2000)
    mcmc_fitting(data, window, pumping=p, priors=pr, config=cfg)

.. code-block:: python

    from cometspec.config import FluorescenceModelConfig
    from cometspec.fluorescence import FluorescenceModel

    cfg = FluorescenceModelConfig(
        isotopologues=["12C14N", "13C14N"],
        systems="bx",
        T=200.0,
        logN=11.5,
    )
    FluorescenceModel(pumping=p, config=cfg)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


__all__ = [
    "UNSET",
    "FluorescenceModelConfig",
    "MCMCFitConfig",
]


class _Unset:
    """Sentinel type for "field not provided" in optional configs."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Any = _Unset()


@dataclass
class FluorescenceModelConfig:
    """Optional grouped configuration for :class:`FluorescenceModel`.

    Each field mirrors the corresponding constructor keyword argument. Any
    field left at :data:`UNSET` (the default) is ignored, so the constructor
    keeps its existing default. Fields explicitly set on this object override
    the individual keyword arguments.
    """

    data: Any = UNSET
    window: Any = UNSET
    pumping: Any = UNSET
    isotopologues: Any = UNSET
    systems: Any = UNSET
    linelists: Any = UNSET
    line_path: Any = UNSET

    lsf: Any = UNSET
    lsf_method: Any = UNSET
    sigma: Any = UNSET
    sigma1: Any = UNSET
    sigma2: Any = UNSET
    sigma_G: Any = UNSET
    fwhm_L: Any = UNSET
    ratio: Any = UNSET

    A_min: Any = UNSET
    a: Any = UNSET
    name: Any = UNSET

    logN: Any = UNSET
    logN_by_iso: Any = UNSET
    f_col: Any = UNSET
    f_col_by_iso: Any = UNSET
    T: Any = UNSET
    T_by_iso: Any = UNSET
    v_kms: Any = UNSET
    v_kms_by_iso: Any = UNSET
    dlam: Any = UNSET
    dlam_by_iso: Any = UNSET

    wave_col: Any = UNSET
    flux_col: Any = UNSET
    error_col: Any = UNSET
    continuum_col: Any = UNSET
    omega: Any = UNSET

    include_rotations: Any = UNSET
    pumping_v_kms: Any = UNSET
    pumping_dlam_A: Any = UNSET
    model_wave: Any = UNSET
    n_cores: Any = UNSET


@dataclass
class MCMCFitConfig:
    """Optional grouped configuration for :func:`cometspec.mcmc.mcmc_fitting`.

    Each field mirrors the corresponding keyword argument of
    :func:`mcmc_fitting`. Any field left at :data:`UNSET` is ignored. Fields
    explicitly set override the individual keyword arguments passed at the
    call site.
    """

    data: Any = UNSET
    window: Any = UNSET
    pumping: Any = UNSET
    priors: Any = UNSET

    isotopologues: Any = UNSET
    systems: Any = UNSET
    linelists: Any = UNSET

    include_rotations: Any = UNSET
    include_deltaJ0_parity_mix: Any = UNSET
    require_X_only_for_rot: Any = UNSET

    nwalkers: Any = UNSET
    nsteps: Any = UNSET
    n_cores: Any = UNSET

    lsf: Any = UNSET
    lsf_method: Any = UNSET

    make_plots: Any = UNSET
    progress: Any = UNSET

    A_min: Any = UNSET
    a: Any = UNSET

    velocity_kms: Any = UNSET
    delta_lambda_A: Any = UNSET

    init_f_col: Any = UNSET
    init_f_col_by_iso: Any = UNSET
    init_T: Any = UNSET
    init_T_by_iso: Any = UNSET
    init_v_kms: Any = UNSET
    init_v_kms_by_iso: Any = UNSET
    init_dlam: Any = UNSET
    init_dlam_by_iso: Any = UNSET
    init_logN: Any = UNSET
    init_logN_by_iso: Any = UNSET
    init_sigma: Any = UNSET
    init_sigma1: Any = UNSET
    init_sigma2: Any = UNSET
    init_sigma_G: Any = UNSET
    init_fwhm_L: Any = UNSET
    init_ratio: Any = UNSET

    fig_file: Any = UNSET
    wave_col: Any = UNSET
    flux_col: Any = UNSET
    error_col: Any = UNSET
    continuum_col: Any = UNSET
    omega: Any = UNSET
    verbose: Any = UNSET
    pruning: Any = UNSET
    N_Model: Any = UNSET