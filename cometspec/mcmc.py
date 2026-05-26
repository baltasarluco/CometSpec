"""MCMC fitting kernel for the multi-isotopologue fluorescence model."""
from __future__ import annotations

from typing import Dict, Tuple, Sequence, Optional, Callable, Any

import warnings
import multiprocess as _mp
import os as _os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner

from astropy import constants as const

from .linelist import (
    _as_list,
    normalize_cn_systems_arg,
    resolve_linelists_with_defaults,
    attach_pumping_and_labels,
)
from .rates import (
    build_rate_matrix_nbar,
    solve_with_normalization,
    solve_with_normalization_fast,
    g_factors_fast_from_cache,
    synth_spectrum_from_lines,
    make_lsf,
)
from .collisions import (
    _empty_scaffold,
    is_atomic_species,
    precompute_collision_scaffold_fast,
    apply_collisions_inplace_fast,
)
from .config import UNSET, MCMCFitConfig


__all__ = [
    "mcmc_fitting",
]


#: Planck constant, in erg*s.
H_CGS: float = const.h.cgs.value


def _resolve_n_cores(n_cores: Optional[int]) -> int:
    """Normalize ``n_cores`` to a concrete worker count.

    ``None`` or non-positive values map to ``1`` (single-threaded). Otherwise
    the value is clamped to the number of available CPUs.
    """
    if n_cores is None:
        return 1
    try:
        n = int(n_cores)
    except (TypeError, ValueError):
        return 1
    if n <= 1:
        return 1
    cpu = _os.cpu_count() or 1
    return max(1, min(n, cpu))


def _worker_init(lnprob_callable: Optional[Callable[[Sequence[float]], float]] = None) -> None:
    """Initialize a ``spawn``-based pool worker for emcee.

    Each worker (a) installs the pickled ``lnprob_callable`` as the
    module-level :data:`_LNPROB_CALLABLE` so the picklable
    :func:`_lnprob_worker` proxy can dispatch to it, and (b) pins the BLAS
    thread pool to a single thread. Pinning BLAS avoids CPU oversubscription
    when the pool already provides parallelism across walkers.
    """
    global _LNPROB_CALLABLE
    if lnprob_callable is not None:
        _LNPROB_CALLABLE = lnprob_callable
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def _build_mcmc_pool(
    n_cores: Optional[int],
    lnprob_callable: Optional[Callable[[Sequence[float]], float]],
):
    """Create a ``spawn``-based pool for emcee, or ``None``.

    Built on the :mod:`multiprocess` library (drop-in replacement for
    :mod:`multiprocessing` that uses :mod:`dill` instead of :mod:`pickle`),
    so the lnprob callable's user-provided ``lsf`` may be a lambda, a
    closure, or a top-level function defined in a Jupyter notebook cell.
    Uses the ``spawn`` start method on every platform (including macOS,
    where Apple Accelerate's GCD-backed BLAS deadlocks or segfaults when a
    multi-threaded process forks). ``lnprob_callable`` is sent once at pool
    creation through ``initargs`` and installed on each worker as the
    module-level ``_LNPROB_CALLABLE`` so the :func:`_lnprob_worker` proxy
    can dispatch to it. Each worker also pins BLAS threads to 1.

    Returns ``None`` when ``n_cores <= 1``.
    """
    n = _resolve_n_cores(n_cores)
    if n <= 1:
        return None
    ctx = _mp.get_context("spawn")
    return ctx.Pool(
        processes=n,
        initializer=_worker_init,
        initargs=(lnprob_callable,),
    )


def _parent_blas_limiter(n_cores: Optional[int]):
    """Limit parent-process BLAS threads while the pool is alive.

    Workers already pin their own BLAS to one thread in :func:`_worker_init`.
    This context limits the parent the same way for the duration of the
    pool's lifetime, so any incidental BLAS work the parent does while
    walkers run does not oversubscribe CPUs alongside the pool. The previous
    BLAS thread count is restored once the pool is joined.
    """
    n = _resolve_n_cores(n_cores)
    if n <= 1:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        warnings.warn(
            "threadpoolctl is not installed; multi-core MCMC may deadlock on "
            "macOS or oversubscribe BLAS threads on Linux. Install threadpoolctl "
            "or run with n_cores=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return nullcontext()
    return threadpool_limits(limits=1)


_LNPROB_CALLABLE: Optional[Callable[[Sequence[float]], float]] = None


def _lnprob_worker(theta):
    """Module-level proxy that dispatches to the active ``lnprob`` callable.

    ``multiprocessing.Pool.map`` pickles the target callable per call to
    send through the IPC pipe, which would fail for a closure defined inside
    :func:`mcmc_fitting`. This proxy is picklable and forwards to the
    module-level :data:`_LNPROB_CALLABLE`, which is installed once on each
    worker by :func:`_worker_init` via the pool's ``initargs``.
    """
    return _LNPROB_CALLABLE(theta)


class _LnProbCallable:
    """Bundle of state required to evaluate ``lnprob(theta)``.

    Replaces the closure that :func:`mcmc_fitting` previously built locally,
    so the callable can be serialized and shipped to ``spawn`` workers (the
    earlier ``fork``-inheritance trick is unsafe with Apple Accelerate, which
    deadlocks or segfaults when a multi-threaded process forks).

    Serialized via :mod:`dill` (through the :mod:`multiprocess` library), so
    a user-provided ``lsf`` that is a lambda, a closure, or a top-level
    function defined in a Jupyter notebook cell all work transparently.
    """

    def __init__(
        self,
        *,
        priors,
        param_keys,
        cache,
        iso_list,
        x_fit,
        y_fit,
        y_err_fit,
        lsf,
        lsf_method,
        omega,
        include_rotations,
        init_logQ,
        init_logQ_by_iso,
        init_T,
        init_T_by_iso,
        init_v_kms,
        init_v_kms_by_iso,
        init_dlam,
        init_dlam_by_iso,
        init_logN,
        init_logN_by_iso,
        init_sigma,
        init_sigma1,
        init_sigma2,
        init_sigma_G,
        init_fwhm_L,
        init_ratio,
    ):
        self.priors = priors
        self.param_keys = param_keys
        self.cache = cache
        self.iso_list = iso_list
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.y_err_fit = y_err_fit
        self.lsf = lsf
        self.lsf_method = lsf_method
        self.omega = omega
        self.include_rotations = include_rotations
        self.init_logQ = init_logQ
        self.init_logQ_by_iso = init_logQ_by_iso
        self.init_T = init_T
        self.init_T_by_iso = init_T_by_iso
        self.init_v_kms = init_v_kms
        self.init_v_kms_by_iso = init_v_kms_by_iso
        self.init_dlam = init_dlam
        self.init_dlam_by_iso = init_dlam_by_iso
        self.init_logN = init_logN
        self.init_logN_by_iso = init_logN_by_iso
        self.init_sigma = init_sigma
        self.init_sigma1 = init_sigma1
        self.init_sigma2 = init_sigma2
        self.init_sigma_G = init_sigma_G
        self.init_fwhm_L = init_fwhm_L
        self.init_ratio = init_ratio

    def theta_to_params(self, theta: Sequence[float]) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.param_keys, theta)}

    def _ln_prior(self, theta: Sequence[float]) -> float:
        for val, name in zip(theta, self.param_keys):
            lo, hi = self.priors[name]
            if val < lo or val > hi:
                return -np.inf
        return 0.0

    def _make_lsf_local(self, pars: Dict[str, float]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        if self.lsf is not None:
            return self.lsf
        if self.lsf_method is None:
            return None
        return make_lsf(pars, self.lsf_method)

    def _logQ_for_iso(self, iso: str, pars: Dict[str, float]) -> Optional[float]:
        key = f"logQ_{iso}"
        if key in pars:
            return float(pars[key])
        if "logQ" in pars:
            return float(pars["logQ"])
        if self.init_logQ_by_iso is not None and iso in self.init_logQ_by_iso:
            try:
                return float(self.init_logQ_by_iso[iso])
            except TypeError:
                return None
        if self.init_logQ is not None:
            return float(self.init_logQ)
        return None

    def _T_for_iso(self, iso: str, pars: Dict[str, float]) -> float:
        key = f"T_{iso}"
        if key in pars:
            return float(pars[key])
        if "T" in pars:
            return float(pars["T"])
        if self.init_T_by_iso is not None and iso in self.init_T_by_iso:
            return float(self.init_T_by_iso[iso])
        return float(self.init_T)

    def _v_kms_for_iso(self, iso: str, pars: Dict[str, float]) -> float:
        key = f"v_kms_{iso}"
        if key in pars:
            return float(pars[key])
        if "v_kms" in pars:
            return float(pars["v_kms"])
        if self.init_v_kms_by_iso is not None and iso in self.init_v_kms_by_iso:
            return float(self.init_v_kms_by_iso[iso])
        return float(self.init_v_kms)

    def _dlam_for_iso(self, iso: str, pars: Dict[str, float]) -> float:
        key = f"dlam_{iso}"
        if key in pars:
            return float(pars[key])
        if "dlam" in pars:
            return float(pars["dlam"])
        if self.init_dlam_by_iso is not None and iso in self.init_dlam_by_iso:
            return float(self.init_dlam_by_iso[iso])
        return float(self.init_dlam)

    def _logN_for_iso(self, iso: str, pars: Dict[str, float]) -> Optional[float]:
        key = f"logN_{iso}"
        if key in pars:
            return float(pars[key])
        if "logN" in pars:
            return float(pars["logN"])
        if self.init_logN_by_iso is not None and iso in self.init_logN_by_iso:
            try:
                return float(self.init_logN_by_iso[iso])
            except TypeError:
                return None
        if self.init_logN is not None:
            return float(self.init_logN)
        return None

    def model_flux(self, theta: Sequence[float], wave: np.ndarray) -> np.ndarray:
        pars = self.theta_to_params(theta)
        wmin = float(np.min(wave))
        wmax = float(np.max(wave))

        try:
            sigma = float(pars["sigma"]) if "sigma" in pars else float(self.init_sigma)
        except TypeError:
            sigma = None
        try:
            sigma1 = float(pars["sigma1"]) if "sigma1" in pars else float(self.init_sigma1)
        except TypeError:
            sigma1 = None
        try:
            sigma2 = float(pars["sigma2"]) if "sigma2" in pars else float(self.init_sigma2)
        except TypeError:
            sigma2 = None
        try:
            sigma_G = float(pars["sigma_G"]) if "sigma_G" in pars else float(self.init_sigma_G)
        except TypeError:
            sigma_G = None
        try:
            fwhm_L = float(pars["fwhm_L"]) if "fwhm_L" in pars else float(self.init_fwhm_L)
        except TypeError:
            fwhm_L = None
        try:
            ratio = float(pars["ratio"]) if "ratio" in pars else float(self.init_ratio)
        except TypeError:
            ratio = None

        dict_for_lsf = {
            "sigma": sigma, "sigma1": sigma1, "sigma2": sigma2,
            "sigma_G": sigma_G, "fwhm_L": fwhm_L, "ratio": ratio,
        }
        lsf_fun = self._make_lsf_local(dict_for_lsf)

        spec_total = np.zeros_like(wave, dtype=float)

        for iso in self.iso_list:
            C = self.cache[iso]
            M = C["M_work"]
            np.copyto(M, C["M_rad"])

            logQ_i = self._logQ_for_iso(iso, pars)
            T_i = self._T_for_iso(iso, pars)
            v_kms_i = self._v_kms_for_iso(iso, pars)
            dlam_i = self._dlam_for_iso(iso, pars)

            if logQ_i is not None:
                Q_i = 10.0 ** logQ_i if np.isfinite(logQ_i) else 0.0
                if Q_i > 0.0 and self.include_rotations:
                    apply_collisions_inplace_fast(
                        M, C["coll_scaf"], Q=Q_i, T=T_i, Cup_work=C["Cup_work"]
                    )

            n = solve_with_normalization_fast(M, C["A_work"], C["b_work"])

            _, g_en = g_factors_fast_from_cache(
                ui=C["ui"],
                A_ul=C["A_ul"],
                hnu=C["hnu"],
                n=n,
                out_g_ph=C["gph_work"],
                out_g_en=C["gen_work"],
            )

            logN_i = self._logN_for_iso(iso, pars)
            
            _, spec_i = synth_spectrum_from_lines(
                C["lines_out"],
                g_line_energy=g_en,
                lam_min=wmin,
                lam_max=wmax,
                lam_col="Wave_vac_AA",
                N_col_cm2=10.0 ** logN_i,
                Omega_sr=self.omega,
                grid=wave,
                lsf=lsf_fun,
                v_shift_kms=v_kms_i,
                dlam_shift_A=dlam_i,
            )
            spec_total += spec_i

        return spec_total

    def lnlike(self, theta: Sequence[float]) -> float:
        y_model = self.model_flux(theta, self.x_fit)
        if (not np.all(np.isfinite(y_model))) or (y_model.shape != self.x_fit.shape):
            return -np.inf
        inv_sigma2 = 1.0 / (self.y_err_fit ** 2)
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * self.y_err_fit ** 2)
            + (self.y_fit - y_model) ** 2 * inv_sigma2
        )

    def __call__(self, theta: Sequence[float]) -> float:
        lp = self._ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.lnlike(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll


def mcmc_fitting(
    data: Any = None,
    window: Optional[Tuple[float, float]] = None,
    *,
    pumping: Any = None,

    isotopologues: str | Sequence[str] = "12C14N",
    systems: str | Sequence[str] | None = None,

    linelists: pd.DataFrame | dict[str, pd.DataFrame] | Sequence[pd.DataFrame] | None = None,

    include_rotations: bool = True,
    include_deltaJ0_parity_mix: bool = True,
    require_X_only_for_rot: bool = True,

    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    nwalkers: int = 50,
    nsteps: int = 1000,
    n_cores: Optional[int] = None,

    lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lsf_method: Optional[str] = None,

    make_plots: bool = False,
    progress: bool = True,

    A_min: Optional[float] = 1e4,
    a: float = 3,

    # NOTE: these control *pumping* wavelength shift for J_nu (radiative rates)
    velocity_kms: float = 0.0,
    delta_lambda_A: float = 0.0,

    # NOTE: these are fallbacks for parameters not present in priors
    init_logQ: Optional[float] = None,
    init_logQ_by_iso: Optional[Dict[str, Optional[float]]] = None,
    init_T: float = 300.0,
    init_T_by_iso: Optional[Dict[str, float]] = None,
    init_v_kms: float = 0.0,
    init_v_kms_by_iso: Optional[Dict[str, float]] = None,
    init_dlam: float = 0.0,
    init_dlam_by_iso: Optional[Dict[str, float]] = None,
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

    config: Optional[MCMCFitConfig] = None,
) -> Dict[str, Any]:
    """Run MCMC fitting for the fluorescence model in a wavelength window.

    This routine builds the model for one or more isotopologues, applies an
    optional line-spread function (LSF), and samples posterior distributions for
    the parameters provided in ``priors``.

    This is the key function for fitting. It can be used standalone, but is used by
    :class:`cometspec.fluorescence.FluorescenceModel` as its default fitting method.

    Defaults and selector behavior
    ------------------------------
    - ``isotopologues`` defaults to ``"12C14N"``.
    - ``systems`` defaults to ``None``, which maps to ``["BX00", "AX_dv1"]``. Only used for CN; ignored for other species.
    - Accepted string selectors for ``systems`` include:
        - ``"both"``/``"bx+ax"``/``"bxax"`` -> ``["BX00", "AX_dv1", "AX_dv2", "AX_dv3"]``
        - ``"all"`` -> ``["ALL"]`` Extremely high computation cost; not recommended.
        - ``"bx"``, ``"b-x"``, ``"bx(0,0)"``, ``"bx00"``, ``"bx_00"``, ``"b_x_00"`` -> ``["BX00"]``
        - ``"ax"``/``"a-x"`` -> ``["AX_dv1", "AX_dv2"]``
        - ``"ax(dv=1)"``/``"ax_dv1"`` -> ``["AX_dv1"]``
        - ``"ax(dv=2)"``/``"ax_dv2"`` -> ``["AX_dv2"]``
        - ``"ax(dv=3)"``/``"ax_dv3"`` -> ``["AX_dv3"]``
    - emcee ``nwalkers=50`` and ``nsteps=1000`` by default.
    - Collisions are gated by ``logQ``: if neither a per-iso ``logQ_{iso}`` nor a shared ``logQ`` prior is given it will fall back to ``init_logQ``, and if ``init_logQ`` or ``init_logQ_by_iso`` are also not given for an isotopologue, that isotopologue is treated as collisionless. Other collision controls default to ``include_deltaJ0_parity_mix=True`` (to allow :math:`\Delta J = 0`) and ``require_X_only_for_rot=True`` (to enable just collisions to the lower electronic state found).
    - It is recommended to set explicitly ``include_rotations=False`` if no collisions is the desired behavior.
    - Pumping-shift controls default to ``velocity_kms=0.0`` and ``delta_lambda_A=0.0``. These values are used when computing :math:`J_\\nu` for the radiative rates and they are different from ``v_kms`` and ``dlam`` parameters of a model which are shifts of the output spectrum lines.
    - Fallback parameter values starts with ``init_`` check the default behavior here if this function is used directly, check them in :class:`cometspec.fluorescence.FluorescenceModel` if this function is used via the model's default fitting method.
    - If parameters are not present in the priors, they will fall to the ``init_`` default values (if provided) or to the hardcoded defaults in the model_flux function (e.g. T=300K, v_kms=0, etc). This means that if you want to fit for a parameter but don't provide a prior for it, it will not be fitted and instead will use the fallback value. This allows you to control which parameters are fitted and which are fixed without having to change the code of this function.
    
    .. note::
            Priors consist on a dict mapping parameter name to a (min, max) tuple defining the uniform prior range for that parameter. The possible keys are: ``"logN"``, ``"logQ"``, ``"T"``, ``"dlam"``, ``"v_kms"``, 'logN_<isotopologue>', 'logQ_<isotopologue>', 'T_<isotopologue>', 'dlam_<isotopologue>', 'v_kms_<isotopologue>' (any iso not found falls back to <parameter>) and lsf priors, ``sigma``, ``sigma1``, ``sigma2``, ``sigma_G``, ``fwhm_L``, ``ratio``.

    Parameters
    ----------
    data : Any
        Observed spectrum table or DataFrame which is going to be fitted.
    window : tuple[float, float]
        Wavelength fitting window ``(min_A, max_A)``.
    pumping : Any
        Pumping spectrum with ``WAVE`` and ``FLUX`` columns.
    isotopologues : str or Sequence[str], optional, default "12C14N"
        One or more isotopologue labels.
    systems : str or Sequence[str], optional, default None
        CN system selector(s). If ``None``, uses ``["BX00", "AX_dv1"]``.
    linelists : pandas.DataFrame or dict[str, pd.DataFrame] or Sequence[pd.DataFrame], optional, default None
        Optional normalized line-list DataFrame or isotopologue mapping.
    priors : dict[str, tuple[float, float]]
        Parameter prior ranges. Required.
    nwalkers : int, optional, default 50
        Number of walkers.
    nsteps : int, optional, default 1000
        Number of MCMC steps.
    n_cores : int, optional, default None
        Number of CPU cores used to parallelize the per-step walker likelihood
        evaluations through the :mod:`multiprocess` library (a drop-in
        replacement for the standard :mod:`multiprocessing` that uses
        :mod:`dill` for serialization). ``None`` or ``1`` keeps the sampler
        single-threaded (the prior default behavior). Values ``>1`` spawn
        a ``spawn``-based pool that emcee uses to evaluate walkers in
        parallel; the value is clamped to :func:`os.cpu_count`. The ``spawn``
        start method is used on every platform so the pool is safe on macOS
        (Apple Accelerate's GCD-backed BLAS deadlocks or segfaults when a
        multi-threaded process forks) as well as on Linux and Windows. Using
        ``multiprocess`` (with ``dill``) instead of stock ``multiprocessing``
        means user-provided ``lsf`` callables defined as lambdas, closures,
        or top-level functions in a Jupyter notebook cell are serialized
        correctly — stock pickle would silently fail on those and hang the
        sampler. Each worker has its BLAS thread pool pinned to 1 via
        ``threadpoolctl`` to avoid thread oversubscription. The tqdm
        progress bar (when ``progress=True``) keeps working under both modes.

        .. important::

           When calling this function with ``n_cores > 1`` from a ``.py``
           script (not a Jupyter notebook), the call **must** be guarded by
           ``if __name__ == "__main__":``. Each ``spawn`` worker re-imports
           the user's script, and without the guard the call would re-execute
           recursively; Python detects this and aborts with a ``RuntimeError``.
           No guard is needed in notebooks or when ``n_cores`` is ``None``/``1``.

        .. note::

           Whether ``n_cores>1`` is faster than ``n_cores=1`` depends on how
           expensive each likelihood evaluation is, which scales with the
           line-list size — driven by ``A_min``, the number of isotopologues
           and systems, and the window width. Multiprocessing adds a fixed
           IPC overhead per emcee step (pickle + pipe roundtrip per walker
           half). When each ``lnprob`` call is slow (large line list — e.g.
           ``A_min=1e4`` for CN with B-X + A-X, or multi-isotopologue runs)
           the per-step cost dwarfs IPC and ``n_cores=4`` typically gives a
           ~3-4x speedup. When each call is cheap (small line list — e.g.
           ``A_min`` ≥ 1e6 with a single isotopologue, or a narrow window
           with few transitions) IPC overhead can dominate and ``n_cores=1``
           may end up equal or faster. If unsure, time one short run at each
           setting on your problem before committing to a long MCMC.
    lsf : Callable[[numpy.ndarray], numpy.ndarray], optional, default None
        Optional custom LSF callable.
    lsf_method : str, optional, default None
        Built-in LSF method name. It must be one of ``"2Gauss"``, ``"Gauss_Lorentz"``, ``"Gauss"``, or ``"Lorentz"``. Required if ``lsf`` is not provided.
    include_rotations : bool, optional, default True
        Enable rotational collisions in the rate matrix.
    omega : float, optional, default None
        Optional aperture solid angle in sr.
    config : MCMCFitConfig, optional, default None
        Optional grouped configuration. Any field set on this object overrides
        the corresponding individual keyword argument. Fields left at
        :data:`cometspec.config.UNSET` (the default) are ignored, so callers
        that pass individual kwargs not in the config are unaffected. See
        :class:`cometspec.config.MCMCFitConfig`.

    Other Parameters
    ----------------
    include_deltaJ0_parity_mix : bool, optional, default True
        Allow parity-changing ``Delta J = 0`` collisions.
    require_X_only_for_rot : bool, optional, default True
        Restrict collisions to the lower electronic
        state (auto-detected as the ``lower_es`` label with the smallest minimum
        ``E_lower_cm1``; works for any spectroscopic notation).
    make_plots : bool, optional, default False
        Generate diagnostic plots (fit, corner and traces).
    progress : bool, optional, default True
        Show emcee progress output.
    A_min : float, optional, default 1e4
        Minimum Einstein A threshold for line lists. User provided line lists are not filtered by A_ul
    a : float, optional, default 3
        Stretch-move parameter for emcee.
    velocity_kms : float, optional, default 0.0
        Velocity shift used when evaluating pumping J_nu.
    delta_lambda_A : float, optional, default 0.0
        Additive wavelength shift used when evaluating pumping J_nu.
    init_logQ : float, optional, default None
        Fallback ``logQ`` value used by every isotopologue when no
        ``logQ`` prior is sampled and no per-iso entry is given. ``None``
        disables collisions for any isotopologue not covered by ``init_logQ_by_iso``.
    init_logQ_by_iso : dict[str, float or None], optional, default None
        Per-isotopologue fallback ``logQ`` map. Each value may
        be ``None`` to force that isotopologue to be collisionless. Isotopologues
        not present in the map fall back to ``init_logQ``.
    init_T : float, optional, default 300.0
        Global fallback temperature when ``T`` is not sampled and no per-iso entry is given.
    init_T_by_iso : dict[str, float], optional, default None
        Per-isotopologue fallback temperature map. For each isotopologue, ``init_T_by_iso[iso]``
        is used when neither ``T_{iso}`` nor ``T`` appears in the sampled parameters. Isotopologues
        not in this map fall back to ``init_T``.
    init_v_kms : float, optional, default 0.0
        Global fallback emission velocity shift when ``v_kms`` is not sampled and no per-iso entry is given.
    init_v_kms_by_iso : dict[str, float], optional, default None
        Per-isotopologue fallback emission velocity shift. Falls back to ``init_v_kms`` for
        isotopologues not present in the map.
    init_dlam : float, optional, default 0.0
        Global fallback emission wavelength shift when ``dlam`` is not sampled and no per-iso entry is given.
    init_dlam_by_iso : dict[str, float], optional, default None
        Per-isotopologue fallback additive wavelength shift. Falls back to ``init_dlam`` for
        isotopologues not present in the map.
    init_logN : float, optional, default None
        Global fallback ``logN`` when neither ``logN_{iso}`` nor ``logN`` appears in the
        sampled parameters and no per-iso entry is given.
    init_logN_by_iso : dict[str, float], optional, default None
        Per-isotopologue fallback ``logN`` map. For each isotopologue, ``init_logN_by_iso[iso]``
        is used when neither ``logN_{iso}`` nor ``logN`` are sampled. Falls back to ``init_logN``
        for isotopologues not present in the map.
    init_sigma : float, optional, default None
        Fallback Gaussian sigma when ``"sigma"`` is not sampled.
    init_sigma1 : float, optional, default None
        Fallback ``sigma1`` for ``"2Gauss"`` when not sampled.
    init_sigma2 : float, optional, default None
        Fallback ``sigma2`` for ``"2Gauss"`` when not sampled.
    init_sigma_G : float, optional, default None
        Fallback ``sigma_G`` for ``"Gauss_Lorentz"`` when not sampled.
    init_fwhm_L : float, optional, default None
        Fallback Lorentzian FWHM when not sampled.
    init_ratio : float, optional, default None
        Fallback mixture ratio when not sampled.
    fig_file : str, optional, default None
        Base path for output figures.
    wave_col : str, optional, default "WAVE"
        Wavelength column in ``data``.
    flux_col : str, optional, default "FLUX_STACK"
        Flux column in ``data``.
    error_col : str, optional, default "ERR_STACK"
        Uncertainty column in ``data``.
    continuum_col : str, optional, default "CONTINUUM"
        Continuum column in ``data``.
    verbose : bool, optional, default True
        Print diagnostics.
    pruning : bool, optional, default True
        Apply posterior pruning.
    N_Model : int, optional, default 20000
        Number of elements in the model grid.

    Returns
    -------
    dict[str, Any]
        Dictionary with posterior summaries, samples, and model envelopes. Keys:

        * ``param_keys`` (:class:`list` of :class:`str`) -- Ordered list of sampled parameter names, matching the column order of ``samples_pruned``.
        * ``median_params`` (:class:`dict` [:class:`str`, :class:`float`]) -- Posterior median for each parameter.
        * ``up_errors_params`` (:class:`dict` [:class:`str`, :class:`float`]) -- Upper 1-sigma error (p84 - p50) for each parameter.
        * ``low_errors_params`` (:class:`dict` [:class:`str`, :class:`float`]) -- Lower 1-sigma error (p50 - p16) for each parameter.
        * ``samples_pruned`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_samples, ndim)``) -- Pruned and burn-in-removed posterior samples.
        * ``lnprob_pruned`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_samples,)``) -- Log-probability for each pruned sample.
        * ``model_wave`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_Model,)``) -- Wavelength grid over the fitting window.
        * ``median_model`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_Model,)``) -- Model flux evaluated at the posterior median parameters.
        * ``model_p16`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_Model,)``) -- 16th percentile of the model ensemble (lower 1-sigma envelope).
        * ``model_p84`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_Model,)``) -- 84th percentile of the model ensemble (upper 1-sigma envelope).
        * ``best_model`` (:class:`numpy.ndarray` of :class:`float`, shape ``(N_Model,)``) -- Model flux evaluated at the highest-likelihood sample.

    Raises
    ------
    ValueError
        If priors or required parameters are inconsistent.
    ValueError
        If required columns are missing from the data.

    Notes
    -----
    If extreme prior values cause the rate-matrix solver to fail (``LinAlgError``
    from a numerically degenerate matrix, typically due to NaN/Inf collision rates
    at very high Q), ``solve_with_normalization_fast`` returns NaN populations.
    ``lnlike`` then detects a non-finite model and returns ``-inf``, so the walker
    step is rejected gracefully rather than crashing.
    """

    if config is not None:
        if config.data is not UNSET:
            data = config.data
        if config.window is not UNSET:
            window = config.window
        if config.pumping is not UNSET:
            pumping = config.pumping
        if config.priors is not UNSET:
            priors = config.priors
        if config.isotopologues is not UNSET:
            isotopologues = config.isotopologues
        if config.systems is not UNSET:
            systems = config.systems
        if config.linelists is not UNSET:
            linelists = config.linelists
        if config.include_rotations is not UNSET:
            include_rotations = config.include_rotations
        if config.include_deltaJ0_parity_mix is not UNSET:
            include_deltaJ0_parity_mix = config.include_deltaJ0_parity_mix
        if config.require_X_only_for_rot is not UNSET:
            require_X_only_for_rot = config.require_X_only_for_rot
        if config.nwalkers is not UNSET:
            nwalkers = config.nwalkers
        if config.nsteps is not UNSET:
            nsteps = config.nsteps
        if config.n_cores is not UNSET:
            n_cores = config.n_cores
        if config.lsf is not UNSET:
            lsf = config.lsf
        if config.lsf_method is not UNSET:
            lsf_method = config.lsf_method
        if config.make_plots is not UNSET:
            make_plots = config.make_plots
        if config.progress is not UNSET:
            progress = config.progress
        if config.A_min is not UNSET:
            A_min = config.A_min
        if config.a is not UNSET:
            a = config.a
        if config.velocity_kms is not UNSET:
            velocity_kms = config.velocity_kms
        if config.delta_lambda_A is not UNSET:
            delta_lambda_A = config.delta_lambda_A
        if config.init_logQ is not UNSET:
            init_logQ = config.init_logQ
        if config.init_logQ_by_iso is not UNSET:
            init_logQ_by_iso = config.init_logQ_by_iso
        if config.init_T is not UNSET:
            init_T = config.init_T
        if config.init_T_by_iso is not UNSET:
            init_T_by_iso = config.init_T_by_iso
        if config.init_v_kms is not UNSET:
            init_v_kms = config.init_v_kms
        if config.init_v_kms_by_iso is not UNSET:
            init_v_kms_by_iso = config.init_v_kms_by_iso
        if config.init_dlam is not UNSET:
            init_dlam = config.init_dlam
        if config.init_dlam_by_iso is not UNSET:
            init_dlam_by_iso = config.init_dlam_by_iso
        if config.init_logN is not UNSET:
            init_logN = config.init_logN
        if config.init_logN_by_iso is not UNSET:
            init_logN_by_iso = config.init_logN_by_iso
        if config.init_sigma is not UNSET:
            init_sigma = config.init_sigma
        if config.init_sigma1 is not UNSET:
            init_sigma1 = config.init_sigma1
        if config.init_sigma2 is not UNSET:
            init_sigma2 = config.init_sigma2
        if config.init_sigma_G is not UNSET:
            init_sigma_G = config.init_sigma_G
        if config.init_fwhm_L is not UNSET:
            init_fwhm_L = config.init_fwhm_L
        if config.init_ratio is not UNSET:
            init_ratio = config.init_ratio
        if config.fig_file is not UNSET:
            fig_file = config.fig_file
        if config.wave_col is not UNSET:
            wave_col = config.wave_col
        if config.flux_col is not UNSET:
            flux_col = config.flux_col
        if config.error_col is not UNSET:
            error_col = config.error_col
        if config.continuum_col is not UNSET:
            continuum_col = config.continuum_col
        if config.omega is not UNSET:
            omega = config.omega
        if config.verbose is not UNSET:
            verbose = config.verbose
        if config.pruning is not UNSET:
            pruning = config.pruning
        if config.N_Model is not UNSET:
            N_Model = config.N_Model

    if data is None:
        raise ValueError("data must be provided (argument or config.data).")
    if window is None:
        raise ValueError("window must be provided (argument or config.window).")
    if pumping is None:
        raise ValueError("pumping must be provided (argument or config.pumping).")
    if priors is None:
        raise ValueError("priors must be provided (argument or config.priors).")

    required_cols = {wave_col, flux_col, error_col, continuum_col}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(
            f"Data is missing required columns: {missing_cols}. Please specify the correct columns using wave_col, flux_col, error_col, and continuum_col parameters."
        )
    iso_list = _as_list(isotopologues)
    sys_tokens = normalize_cn_systems_arg(systems)

    param_keys = list(priors.keys())

    if lsf is not None:
        drop = {"sigma_G", "fwhm_L", "sigma", "sigma1", "sigma2", "ratio"}
        param_keys = [k for k in param_keys if k not in drop]
        priors = {k: priors[k] for k in param_keys}
    else:
        if lsf_method == "2Gauss":
            drop = {"sigma_G", "fwhm_L", "sigma"}
        elif lsf_method == "Gauss_Lorentz":
            drop = {"sigma1", "sigma2", "sigma"}
        elif lsf_method == "Gauss":
            drop = {"sigma_G", "fwhm_L", "sigma1", "sigma2", "ratio"}
        elif lsf_method == "Lorentz":
            drop = {"sigma_G", "sigma1", "sigma2", "sigma", "ratio"}
        else:
            raise ValueError("Provide `lsf` or lsf_method in {'2Gauss','Gauss_Lorentz','Gauss','Lorentz'}.")

        param_keys = [k for k in param_keys if k not in drop]
        priors = {k: priors[k] for k in param_keys}

    for name in param_keys:
        lo, hi = priors[name]
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f"Bad prior for {name!r}: {priors[name]}")

    trans_by_iso = resolve_linelists_with_defaults(
        linelists,
        iso_list,
        systems=sys_tokens,
        A_min=A_min,
        use_omega_labels=False,
        lambda_min_A=pumping["WAVE"].min(),
        lambda_max_A=pumping["WAVE"].max(),
    )
    def _iso_can_collide(iso: str) -> bool:
        if is_atomic_species(iso):
            return False
        if f"logQ_{iso}" in priors:
            return True
        if "logQ" in priors:
            return True
        if init_logQ_by_iso is not None and iso in init_logQ_by_iso:
            return init_logQ_by_iso[iso] is not None
        return init_logQ is not None

    iso_collides = {iso: _iso_can_collide(iso) for iso in trans_by_iso.keys()}

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
        A_ul = np.asarray(lines_out["A_ul"], dtype=np.float64)        
        nu = np.asarray(lines_out["__nu_Hz"], dtype=np.float64)       
        hnu = H_CGS * nu
        gph_work = np.empty_like(A_ul, dtype=float)
        gen_work = np.empty_like(A_ul, dtype=float)

        if iso_collides[iso]:
            coll_scaf = precompute_collision_scaffold_fast(
                lines_out, idx_to_level,
                include_deltaJ0_parity_mix=include_deltaJ0_parity_mix,
                require_X_only=require_X_only_for_rot,
                iso_name=iso,
            )
        else:
            coll_scaf = _empty_scaffold()

        M_work = np.empty_like(M_rad)              
        A_work = np.empty_like(M_rad)             
        b_work = np.zeros(M_rad.shape[0], float)   
        Cup_work = np.empty_like(coll_scaf.get("iu", np.array([], dtype=int)), dtype=float)


        cache[iso] = dict(
            M_rad=M_rad,
            idx_to_level=idx_to_level,
            lines_out=lines_out,
            coll_scaf=coll_scaf,

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

    lnprob_callable = _LnProbCallable(
        priors=priors,
        param_keys=param_keys,
        cache=cache,
        iso_list=iso_list,
        x_fit=x_fit,
        y_fit=y_fit,
        y_err_fit=y_err_fit,
        lsf=lsf,
        lsf_method=lsf_method,
        omega=omega,
        include_rotations=include_rotations,
        init_logQ=init_logQ,
        init_logQ_by_iso=init_logQ_by_iso,
        init_T=init_T,
        init_T_by_iso=init_T_by_iso,
        init_v_kms=init_v_kms,
        init_v_kms_by_iso=init_v_kms_by_iso,
        init_dlam=init_dlam,
        init_dlam_by_iso=init_dlam_by_iso,
        init_logN=init_logN,
        init_logN_by_iso=init_logN_by_iso,
        init_sigma=init_sigma,
        init_sigma1=init_sigma1,
        init_sigma2=init_sigma2,
        init_sigma_G=init_sigma_G,
        init_fwhm_L=init_fwhm_L,
        init_ratio=init_ratio,
    )

    ndim = len(param_keys)
    nburn = nsteps // 2
    print("Number of iterations:", ndim * nwalkers * nsteps)

    p0 = np.array([[np.random.uniform(*priors[name]) for name in param_keys] for _ in range(nwalkers)])

    move = emcee.moves.StretchMove(a=a)
    global _LNPROB_CALLABLE
    prev_lnprob_callable = _LNPROB_CALLABLE
    _LNPROB_CALLABLE = lnprob_callable
    try:
        with _parent_blas_limiter(n_cores):
            pool = _build_mcmc_pool(n_cores, lnprob_callable)
            log_prob_fn = _lnprob_worker if pool is not None else lnprob_callable
            try:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_prob_fn, moves=move, pool=pool
                )
                sampler.run_mcmc(p0, nsteps, progress=progress)
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
    finally:
        _LNPROB_CALLABLE = prev_lnprob_callable

    chain = sampler.get_chain()
    lnprob_full = sampler.get_log_prob()

    flat_chain = chain.reshape(-1, ndim)
    flat_lnprob = lnprob_full.reshape(-1)
    best_idx = int(np.argmax(flat_lnprob))
    best_theta = flat_chain[best_idx]
    best_params = lnprob_callable.theta_to_params(best_theta)
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
            If af < 0.2 decrease the MCMC a parameter
            If af > 0.5 increase the MCMC a parameter

            actual af = {}'''.format(np.mean(af))
            
    if verbose:
        print("Mean acceptance fraction:", np.mean(af))
    if np.mean(af)<0.2 or np.mean(af)>0.5:
        print(af_msg)
        warnings.warn("Acceptance fraction out of bounds.", UserWarning)

    samples = chain[nburn:, :, :].reshape(-1, ndim)
    lnprob_burn = lnprob_full[nburn:, :].reshape(-1)

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

    x_model = np.linspace(window[0], window[1], N_Model)
    n_draw = min(200, samples_pruned.shape[0])
    model_stack = np.empty((n_draw, x_model.size))
    for i in range(n_draw):
        model_stack[i] = lnprob_callable.model_flux(samples_pruned[i], x_model)

    theta_med = [median_params[k] for k in param_keys]
    best_model = lnprob_callable.model_flux(theta_med, x_model)

    p16_m, p50_m, p84_m = np.percentile(model_stack, [16, 50, 84], axis=0)
    median_model = p50_m
    model_p16 = p16_m
    model_p84 = p84_m

    param_labels = {
        "logN": r"$\mathrm{log}_{10}(N)$",
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
        param_lab_by_iso = {
            f'{k}_{iso}': (rf"$\mathrm{{log}}_{{10}}(N_{{{iso}}})$" if k == "logN" else f"{param_labels[k]}_{iso}")
            for k in param_labels for iso in iso_list
        }
        param_labels = {**param_labels, **param_lab_by_iso}

    _logN_units_note = r"$N\ [\mathrm{molecules\ cm}^{-2}]$"
    _has_logN = any(k == "logN" or k.startswith("logN_") for k in param_keys)
    if make_plots:
        trace_ylabel_fontsize = max(13 - 0.4 * ndim, 8)
        trace_tick_fontsize = max(11 - 0.3 * ndim, 7)
        fig, axes = plt.subplots(ndim, 1, figsize=(10, max(1.8 * ndim, 4)), sharex=True)
        if ndim == 1:
            axes = [axes]

        nsteps_chain, nwalkers_chain, _ = chain.shape
        steps = np.arange(nsteps_chain)

        for j, name in enumerate(param_keys):
            for w in range(nwalkers_chain):
                axes[j].plot(steps, chain[:, w, j], alpha=0.7, lw=0.8)
            axes[j].set_ylabel(param_labels[name], fontsize=trace_ylabel_fontsize, labelpad=2)
            axes[j].tick_params(axis='both', labelsize=trace_tick_fontsize)
            if name == "logN" or name.startswith("logN_"):
                axes[j].text(0.98, 0.97, _logN_units_note, fontsize=trace_ylabel_fontsize,
                             va='top', ha='right', transform=axes[j].transAxes,
                             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
        axes[-1].set_xlabel("iteration", fontsize=trace_ylabel_fontsize)
        fig.tight_layout(h_pad=0.3)

        plt.savefig(f"{fig_file}_mcmc_traces.pdf", dpi=300, format='pdf')
        plt.show()

        fig_size = max(3.0 * ndim, 10)

        label_fontsize = max(16 - 0.7 * ndim, 9)
        title_fontsize = max(14 - 0.7 * ndim, 8)
        tick_fontsize = max(12 - 0.4 * ndim, 7)

        fig = corner.corner(
            samples_pruned,
            labels=[param_labels[k] for k in param_keys],
            title_kwargs={'fontsize': title_fontsize, 'y': 1.05},
            title_fmt=".2f",
            use_math_text=True,
            bins=15,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            color='lightseagreen',
            hist_kwargs={'color': 'black', 'linewidth': 1.5},
            contour_kwargs={'linewidths': 1, 'colors': 'black'},
            label_kwargs={'fontsize': label_fontsize},
            fig=plt.figure(figsize=(fig_size, fig_size)), 
        )

        margin = max(0.08 + 0.008 * ndim, 0.12)
        fig.subplots_adjust(wspace=0.05, hspace=0.05, left=margin, bottom=margin, right=0.97, top=0.95)

        elements = [param_labels[k] for k in param_keys]
        axes = np.array(fig.axes).reshape((ndim, ndim))

        for i in range(ndim):
            ax = axes[i, i]
            name_i = param_keys[i]
            q16, q50, q84 = np.quantile(samples[:, i], [0.16, 0.5, 0.84])
            q_minus, q_plus = q50 - q16, q84 - q50
            value_str = rf"${q50:.2f}_{{-{q_minus:.2f}}}^{{+{q_plus:.2f}}}$"
            if name_i == "logN" or name_i.startswith("logN_"):
                title = f"{_logN_units_note}\n{elements[i]}\n{value_str}"
                ax.set_title(title, fontsize=title_fontsize, y=1.12)
            else:
                title = f"{elements[i]}\n{value_str}"
                ax.set_title(title, fontsize=title_fontsize, y=1.05)

        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            for formatter in (ax.xaxis.get_major_formatter(), ax.yaxis.get_major_formatter()):
                if hasattr(formatter, 'set_useOffset'):
                    formatter.set_useOffset(False)

        if ndim > 3:
            for ax in fig.get_axes():
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')

        plt.savefig(f"{fig_file}_corner.pdf", dpi=300, format='pdf')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(x_fit, y_fit, label="Data (cont-sub)", color="black", alpha=0.8)
        plt.fill_between(
            x_fit,
            y_fit - y_err_fit,
            y_fit + y_err_fit,
            color="k",
            alpha=0.25,
            label=r"1$\sigma$ error",
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
