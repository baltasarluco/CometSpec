"""High-level fluorescence model and production-rate workflow.

This module exposes :class:`FluorescenceModel`, the user-facing wrapper that
ties together line-list selection, physical parameters, the line-spread
function (LSF), and observed-data column names into a single object. It is
the preferred entry point for fitting, synthesis, and profile- or
Haser-based production-rate estimation.

The low-level numerical routines (transition normalization, rate matrices,
collisions, MCMC kernel, spectrum synthesis) live in
:mod:`cometspec.modeling`. This module is the orchestration layer where models can be created or fitted. When
fitting through :meth:`FluorescenceModel.fit_mcmc` attributes are overridden. The effective fallback
values for parameters that are not going to be sampled are read from the model instance and forwarded explicitly to
:func:`cometspec.modeling.mcmc_fitting`.

Notes
-----
- Multi-isotopologue fitting is supported; when more than one isotopologue
  is active, ``logN`` is replaced by per-isotopologue ``logN_{iso}`` keys
  (and likewise for ``logQ_{iso}`` if collisions are sampled per iso). If not all the isotopologues have their own ``logN_{iso}`` key, the missing ones fall back to ``logN``; the same applies to ``logQ``.
- :meth:`FluorescenceModel.save` and :meth:`FluorescenceModel.load` provide
  a way to save the models as pickled files. They can be loaded later in the case the user wants to fit just one long MCMC run and then build the models afterwards.
- Check :func:`cometspec.linelist.load_default_transitions` for default supported isotopologue, and the default line lists that are loaded when ``linelists`` is not provided and their corresponding references.
- It is recommended to call the isotopologues with the following nomenclature <number><Element><number><Element> for instance 12C14N, and it will recognize homonuclear, diatomic heteronuclear of the same element or diatomic heteronuclear of different elements. If the isotopologue name does not follow any of the previous categories, it will fall in the default category, which is to include :math:`|\Delta J| = \pm 1, 0`. Check :func:`cometspec.collisions.precompute_collision_scaffold` for the collision selection rules.

Classes
-------
FluorescenceModel
    High-level wrapper over fitting, synthesis, and production-rate computation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Callable, Sequence, Union

import numpy as np
import pandas as pd

from astropy import units as u

from . import modeling
from .config import UNSET, FluorescenceModelConfig, MCMCFitConfig


__all__ = ["FluorescenceModel"]


class FluorescenceModel:
    """High-level fluorescence workflow wrapper.

    This class centralizes model state (line selection, physical parameters,
    LSF, and data columns) and is the preferred entry point for fitting and
    synthesis.

    IMPORTANT:
    When ``fit_mcmc`` calls ``modeling.mcmc_fitting``, fallback values for
    parameters that are not sampled are taken from this instance (e.g. ``self.logQ``,
    ``self.T``, ``self.v_kms``, ``self.dlam``) and passed explicitly.

    The object exposes a stable set of attributes, even before fitting.

    **Always available** (initialized in ``__init__``):

    **Selection / config**

    - ``data`` (:class:`Any` or :obj:`None`) -- Observed spectrum container (DataFrame or astropy Table).
    - ``window`` (``tuple[float, float]`` or :obj:`None`) -- Wavelength interval ``(min_A, max_A)`` used for synthesis.
    - ``pumping`` (:class:`Any`) -- Solar irradiance spectrum used to evaluate :math:`J_\\nu`. Must have WAVE and FLUX columns in :math:`\mathrm{Å}` and :math:`\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Å}^{-1}` units.
    - ``isotopologues`` (:class:`str` or ``Sequence[str]``) -- Isotopologue label(s). Default labels are ``"12C14N"``, ``"13C14N"``, ``"12C15N"``, ``"12C13C"``, ``"12C2"``, ``"13C2"`` or any label containing ``Fe``. They can also be custom labels for custom line lists.
    - ``systems`` (:class:`str` or ``Sequence[str]`` or :obj:`None`) -- CN system selector(s). Check :func:`cometspec.linelist.normalize_cn_systems_arg` for supported values.
    - ``linelists`` (:class:`pandas.DataFrame` or ``dict[str, pandas.DataFrame]`` or ``Sequence[pandas.DataFrame]`` or :obj:`None`) -- Custom normalized line list(s).
    - ``line_path`` (:class:`str` or :obj:`None`) -- Custom path for single-isotopologue line list loading. Just use it if the default line lists are located in non standard locations.
    - ``name`` (:class:`str` or :obj:`None`) -- Model label. Defaults to ``"Fluorescence"`` if not provided.

    **Physical / model controls**

    - ``logN`` (:class:`float` or :obj:`None`) -- :math:`\log_{10}` column density in molecules cm\ :sup:`-2`.
    - ``logN_by_iso`` (``dict[str, float] or None``) -- Per-isotopologue :math:`\log_{10}` column density. Same units as above. Falls back to ``logN`` for isotopologues not in this dict.
    - ``logQ`` (:class:`float` or :obj:`None`) -- :math:`\log_{10}` collisional rate in s\ :sup:`-1`.
    - ``logQ_by_iso`` (``dict[str, float] or None``) -- Per-isotopologue :math:`\log_{10}` collisional rate. Same units as above. Falls back to ``logQ`` for isotopologues not in this dict.
    - ``T`` (:class:`float` or :obj:`None`) -- Kinetic temperature in K.
    - ``T_by_iso`` (``dict[str, float] or None``) -- Per-isotopologue kinetic temperature in K. Falls back to ``T`` for isotopologues not in this dict.
    - ``v_kms`` (:class:`float` or :obj:`None`) -- Emission-frame Doppler velocity shift in km/s.
    - ``v_kms_by_iso`` (``dict[str, float] or None``) -- Per-isotopologue emission-frame velocity shift in km/s. Falls back to ``v_kms`` for isotopologues not in this dict.
    - ``dlam`` (:class:`float` or :obj:`None`) -- Additive emission-frame wavelength shift in Å.
    - ``dlam_by_iso`` (``dict[str, float] or None``) -- Per-isotopologue additive wavelength shift in Å. Falls back to ``dlam`` for isotopologues not in this dict.
    - ``omega`` (:class:`float`) -- Aperture solid angle in sr.

    **LSF config**

    - ``lsf`` (callable or :obj:`None`) -- Custom LSF callable; ``lsf_method`` is stored as ``"Given"`` when set.
    - ``lsf_method`` (:class:`str` or :obj:`None`) -- Built-in LSF mode: ``"Gauss"``, ``"2Gauss"``, ``"Gauss_Lorentz"``, or ``"Lorentz"``.
    - ``sigma`` (:class:`float` or :obj:`None`) -- Gaussian sigma in Å for ``"Gauss"`` mode.
    - ``sigma1`` (:class:`float` or :obj:`None`) -- Component 1 sigma in Å for ``"2Gauss"`` mode.
    - ``sigma2`` (:class:`float` or :obj:`None`) -- Component 2 sigma in Å for ``"2Gauss"`` mode.
    - ``sigma_G`` (:class:`float` or :obj:`None`) -- Gaussian sigma in Å for ``"Gauss_Lorentz"`` mode.
    - ``fwhm_L`` (:class:`float` or :obj:`None`) -- Lorentzian FWHM in Å for ``"Gauss_Lorentz"`` or ``"Lorentz"`` mode.
    - ``ratio`` (:class:`float` or :obj:`None`) -- Mixture ratio for ``"2Gauss"`` and ``"Gauss_Lorentz"`` modes. Gaussian / Lorentzian for ``"Gauss_Lorentz"``; Gaussian 1 / Gaussian 2 for ``"2Gauss"``.

    **Pumping controls**

    - ``pumping_v_kms`` (:class:`float`) -- Velocity shift in km/s applied to line wavelengths when sampling :math:`J_\\nu`.
    - ``pumping_dlam_A`` (:class:`float`) -- Additive wavelength shift in Å applied to line wavelengths when sampling :math:`J_\\nu`.

    .. note::

       The minimum and maximum wavelengths used to filter the default
       transition linelists are taken from the minimum and maximum wavelengths
       of the supplied solar irradiance (pumping) spectrum.

    **Data column names**

    - ``wave_col`` (:class:`str`) -- Wavelength column name in ``data``.
    - ``flux_col`` (:class:`str`) -- Flux column name in ``data``.
    - ``error_col`` (:class:`str`) -- Uncertainty column name in ``data``.
    - ``continuum_col`` (:class:`str`) -- Continuum column name in ``data``.

    **Runtime knobs**

    - ``A_min`` (:class:`float` or :obj:`None`) -- Minimum Einstein-A threshold for line list. User provided line lists are not filtered by A_ul
    - ``a`` (:class:`float`) -- emcee stretch-move parameter forwarded to :meth:`fit_mcmc`.
    - ``n_cores`` (:class:`int` or :obj:`None`) -- Number of CPU cores used to parallelize the MCMC sampler walker evaluations through ``multiprocessing`` (``spawn`` start method, safe on macOS/Linux/Windows). ``None`` or ``1`` keeps the sampler single-threaded. Forwarded to :meth:`fit_mcmc` as the default; can be overridden per-call. The tqdm progress bar (when ``progress=True``) keeps working under both modes. When ``n_cores > 1`` and running from a ``.py`` script (not a Jupyter notebook), the :meth:`fit_mcmc` call must be wrapped in ``if __name__ == "__main__":`` (no guard needed in notebooks or when ``n_cores`` is ``None``/``1``).
    - ``include_rotations`` (:class:`bool`) -- Enable rotational collisions in synthesis and fitting. Recommended to explicitly set this to ``False`` when fitting with no collisions. To see collisions to be included go to :func:`cometspec.collisions.precompute_collision_scaffold`.
    - ``model_wave`` (:class:`numpy.ndarray` or :obj:`None`) -- Optional pre-defined synthesis wavelength grid passed at construction; if ``None`` it is built from ``window``.

    **Synthesis outputs** (set on every :meth:`_synthesize_model` call, including ``__init__`` and :meth:`update_model`; overridden by :meth:`fit_mcmc`):

    - ``model_wave`` (:class:`numpy.ndarray` of :class:`float`) -- Actual wavelength grid used for the current model. After :meth:`fit_mcmc` this is replaced by the fit evaluation grid.
    - ``best_model`` (:class:`numpy.ndarray` of :class:`float`) -- Total synthesized flux at the current parameters. After :meth:`fit_mcmc` this is replaced by the highest-likelihood-sample model. If the rate-matrix solver fails (``LinAlgError``, typically caused by NaN/Inf collision rates at extreme Q values), ``best_model`` will contain NaN — this signals that the chosen parameters are unphysical.

    **Fitting outputs** (initialized empty/``None``; populated after :meth:`fit_mcmc`):

    - ``priors`` (``dict[str, tuple[float, float]]``) -- Parameter prior ranges. Empty dict until set.
    - ``param_keys`` (``tuple[str, ...]``) -- Ordered sampled parameter names. Empty until fit.
    - ``median_params`` (``dict[str, float]``) -- Posterior median for each sampled parameter.
    - ``up_errors_params`` (``dict[str, float]``) -- Upper 1-sigma error (p84 - p50) for each parameter.
    - ``low_errors_params`` (``dict[str, float]``) -- Lower 1-sigma error (p50 - p16) for each parameter.
    - ``samples_pruned`` (:class:`numpy.ndarray` of :class:`float` or :obj:`None`) -- Pruned posterior samples, shape ``(N_samples, ndim)``.
    - ``lnprob_pruned`` (:class:`numpy.ndarray` of :class:`float` or :obj:`None`) -- Log-probability of each pruned sample.
    - ``median_model`` (:class:`numpy.ndarray` of :class:`float` or :obj:`None`) -- Model flux evaluated at the posterior median parameters.
    - ``model_p16`` (:class:`numpy.ndarray` of :class:`float` or :obj:`None`) -- 16th-percentile model envelope.
    - ``model_p84`` (:class:`numpy.ndarray` of :class:`float` or :obj:`None`) -- 84th-percentile model envelope.
    - ``logN_err`` (:class:`numpy.ndarray` of :class:`float` or :obj:`None`) -- ``[upper, lower]`` 1-sigma errors on ``logN``.
    - ``logN_err_by_iso`` (``dict[str, numpy.ndarray] or None``) -- Per-isotopologue ``logN`` 1-sigma errors.

    **Per-isotopologue containers** (created at initialization; recomputed on each fit or update):

    - ``lines_by_iso`` (``dict[str, pandas.DataFrame] or None``) -- Annotated transition tables per isotopologue.
    - ``M_by_iso`` (``dict[str, numpy.ndarray] or None``) -- Radiative rate matrices per isotopologue.
    - ``idx_to_level_by_iso`` (``dict[str, dict] or None``) -- Matrix-index-to-level mappings per isotopologue.
    - ``n_by_iso`` (``dict[str, numpy.ndarray] or None``) -- Steady-state level populations per isotopologue.
    - ``g_ph_by_iso`` (``dict[str, numpy.ndarray] or None``) -- Photon g-factors per line per isotopologue.
    - ``g_en_by_iso`` (``dict[str, numpy.ndarray] or None``) -- Energy g-factors per line per isotopologue.
    - ``g_ph_sum_by_iso`` (``dict[str, float] or None``) -- Total photon g-factor per isotopologue.
    - ``g_en_sum_by_iso`` (``dict[str, float] or None``) -- Total energy g-factor per isotopologue.
    - ``model_by_iso`` (``dict[str, tuple[numpy.ndarray, numpy.ndarray]] or None``) -- Synthesized spectrum per isotopologue as ``(wave, flux)`` tuples.

    **Derived production-rate fields** (set by :meth:`compute_production_rate`):

    - ``q`` (``float or dict[str, float] or None``) -- :math:`\log_{10}` production rate(s).
    - ``q_err`` (``float or dict or None``) -- 1-sigma uncertainty on ``q``.
    - ``q_seeing_corrected`` (:class:`bool`) -- Whether ``q`` has been corrected for seeing losses.
    - ``logN_seeing_corrected`` (:class:`bool`) -- Whether ``logN`` has been corrected for seeing losses.
    """

    def __init__(
        self,
        *,
        data: Optional[Any] = None,
        window: Optional[Tuple[float, float]] = (3850.0, 3900.0),
        pumping: Any = None,
        isotopologues: Union[str, Sequence[str]] = "12C14N",
        systems: Union[str, Sequence[str], None] = None,
        linelists: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], Sequence[pd.DataFrame]]] = None,
        line_path: Optional[str] = None,
        lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lsf_method: Optional[str] = "Gauss",
        A_min: Optional[float] = 1e4,
        a: float = 3.0,
        name: Optional[str] = None,
        sigma: Optional[float] = 0.01,
        sigma1: Optional[float] = None,
        sigma2: Optional[float] = None,
        sigma_G: Optional[float] = None,
        fwhm_L: Optional[float] = None,
        ratio: Optional[float] = None,
        logN: Optional[float] = 11.0,
        logN_by_iso: Optional[Dict[str, float]] = None,
        logQ: Optional[float] = None,
        logQ_by_iso: Optional[Dict[str, float]] = None,
        T: Optional[float] = 300.0,
        T_by_iso: Optional[Dict[str, float]] = None,
        v_kms: Optional[float] = 0.0,
        v_kms_by_iso: Optional[Dict[str, float]] = None,
        dlam: Optional[float] = 0.0,
        dlam_by_iso: Optional[Dict[str, float]] = None,
        wave_col: str = "WAVE",
        flux_col: str = "FLUX_STACK",
        error_col: str = "ERR_STACK",
        continuum_col: str = "CONTINUUM",
        omega: float = np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2,
        include_rotations: bool = True,
        pumping_v_kms: float = 0.0,
        pumping_dlam_A: float = 0.0,
        model_wave: Optional[np.ndarray] = None,
        n_cores: Optional[int] = None,

        config: Optional[FluorescenceModelConfig] = None,
    ) -> None:
        """Initialize a fluorescence model instance and synthesize the first model.

        The constructor stores configuration/state and immediately calls
        :meth:`_synthesize_model` to populate model products.

        .. note::
            If multiple isotopologues are provided, it is recommended to use the format `<parameter>_<iso>` for each isotopologue instead of ``<parameter>`` (``logN``, ``logQ``, ``T``, ``v_kms``, ``dlam``).

        Parameters
        ----------
        pumping : Any
            Solar irradiance spectrum at the comet, used to evaluate ``J_nu``
            for transitions. Required.
        data : Any, optional, default None
            Observed spectrum container (typically pandas DataFrame or astropy Table).
        window : tuple[float, float], optional, default (3850.0, 3900.0)
            Wavelength interval ``(lambda_min_A, lambda_max_A)`` used for synthesis.
        isotopologues : str or Sequence[str], optional, default "12C14N"
            Isotopologue label or list of labels.
        systems : str or Sequence[str], optional, default None
            CN system selector(s) forwarded to
            :func:`modeling.load_default_transitions`.
        linelists : pandas.DataFrame or dict[str, pandas.DataFrame] or Sequence[pandas.DataFrame], optional, default None
            Optional normalized line list(s). Accepted forms:

            * ``None`` -> every isotopologue loaded from packaged defaults.
            * Single :class:`pandas.DataFrame` -> assigned to the first isotopologue;
              the remaining isotopologues fall back to packaged defaults.
            * :class:`dict` mapping iso label to DataFrame -> isotopologues missing
              from the dict fall back to defaults; extra keys are ignored.
            * Sequence of DataFrames -> positionally paired with the first
              ``len(linelists)`` isotopologues; the remainder fall back to defaults.

            Loading a default for an unsupported isotopologue label (e.g.
            ``"COH"``) raises ``ValueError``.
        lsf : Callable[[numpy.ndarray], numpy.ndarray], optional, default None
            Optional custom LSF callable. If provided, ``lsf_method`` parameterization
            values are ignored and ``lsf_method`` is stored as ``"Given"``.
        lsf_method : str, optional, default "Gauss"
            Built-in LSF mode when ``lsf`` is not provided. Supported values are
            ``"Gauss"``, ``"2Gauss"``, ``"Gauss_Lorentz"``, and ``"Lorentz"``.
        logN : float, optional, default 11.0
            Default column density ``log10(N / cm^-2)`` used for synthesis and fallback behavior.
        logN_by_iso : dict[str, float], optional, default None
            Optional per-isotopologue ``log10(N / cm^-2)`` map. Any isotopologue missing from the map falls back to ``logN``.
        logQ : float, optional, default None
            Collisional rate (log10 scale) default fallback.
            ``Q`` is in s^-1.
        logQ_by_iso : dict[str, float], optional, default None
            Optional per-isotopologue ``log10(Q/s^-1)`` map. When provided, each isotopologue uses its own collisional rate; any isotopologue missing from the map falls back to ``logQ``.
        T : float, optional, default 300.0
            Kinetic temperature in Kelvin. Global fallback for all isotopologues.
        T_by_iso : dict[str, float], optional, default None
            Per-isotopologue kinetic temperature in K. For each isotopologue, ``T_by_iso[iso]`` is used if present; otherwise falls back to ``T``.
        v_kms : float, optional, default 0.0
            Emission-frame Doppler velocity shift in km/s. Global fallback for all isotopologues.
        v_kms_by_iso : dict[str, float], optional, default None
            Per-isotopologue emission-frame velocity shift in km/s. Falls back to ``v_kms`` for isotopologues not in this dict.
        dlam : float, optional, default 0.0
            Additive emission-frame wavelength shift in Å. Global fallback for all isotopologues.
        dlam_by_iso : dict[str, float], optional, default None
            Per-isotopologue additive wavelength shift in Å. Falls back to ``dlam`` for isotopologues not in this dict.
        include_rotations : bool, optional, default True
            Enable rotational collisions in synthesis/fitting. Recommended to explicitly set this to ``False`` when fitting with no collisions, as it reduces the number of levels and speeds up the computation. To see collisions to be included go to :func:`cometspec.collisions.precompute_collision_scaffold`.
        config : FluorescenceModelConfig, optional, default None
            Optional grouped configuration. Any field set on this object
            overrides the corresponding individual keyword argument. Fields
            left at :data:`cometspec.config.UNSET` (the default) are ignored,
            so callers that pass individual kwargs are unaffected. See
            :class:`cometspec.config.FluorescenceModelConfig`.

        Other Parameters
        ----------------
        line_path : str, optional, default None
            Optional custom path for single-isotopologue line list loading. Use it just if the default line lists are located in non standard locations.
        sigma : float, optional, default 0.01
            Gaussian sigma for ``lsf_method="Gauss"``.
        sigma1 : float, optional, default None
            Component 1 sigma for ``lsf_method="2Gauss"``.
        sigma2 : float, optional, default None
            Component 2 sigma for ``lsf_method="2Gauss"``.
        sigma_G : float, optional, default None
            Gaussian sigma for ``lsf_method="Gauss_Lorentz"``.
        fwhm_L : float, optional, default None
            Lorentzian FWHM for ``lsf_method="Gauss_Lorentz"`` or ``"Lorentz"``.
        ratio : float, optional, default None
            Mixture ratio for ``"2Gauss"`` and ``"Gauss_Lorentz"`` modes.
            For ``"Gauss_Lorentz"`` it is gauss / lorentz; for ``"2Gauss"`` it is gauss1 / gauss2.
        A_min : float, optional, default 1e4
            Minimum Einstein-A threshold used by default transition loading. User provided line lists are not filtered by A_ul
        a : float, optional, default 3.0
            Stored ``emcee`` stretch-move parameter used by :meth:`fit_mcmc`.
        name : str, optional, default None
            Optional model label. If ``None``, it is set to ``"Fluorescence"``.

            .. note::

               The minimum and maximum wavelengths used to filter the default
               transition linelists are taken from the minimum and maximum
               wavelengths of the supplied solar irradiance (pumping) spectrum.
        wave_col : str, optional, default "WAVE"
            Wavelength column name in ``data``.
        flux_col : str, optional, default "FLUX_STACK"
            Flux column name in ``data``.
        error_col : str, optional, default "ERR_STACK"
            Uncertainty column name in ``data``.
        continuum_col : str, optional, default "CONTINUUM"
            Continuum column name in ``data``.
        omega : float, optional, default np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2
            Aperture solid angle in sr. The default assumes a circular aperture of radius 0.5".
        pumping_v_kms : float, optional, default 0.0
            Velocity shift (km/s) applied to line wavelengths when sampling pumping irradiance ``J_nu``.
        pumping_dlam_A : float, optional, default 0.0
            Additive shift (A) applied to line wavelengths when sampling pumping irradiance ``J_nu``.
        model_wave : numpy.ndarray, optional, default None
            Optional pre-defined synthesis grid. If ``None``, a grid is built from
            ``window`` with 0.01 A spacing.
        n_cores : int, optional, default None
            Default number of CPU cores used by :meth:`fit_mcmc` to parallelize
            walker likelihood evaluations through ``multiprocessing`` (``spawn``
            start method, safe on macOS/Linux/Windows). ``None`` or ``1`` keeps
            the sampler single-threaded. Stored on the instance and overridable
            per :meth:`fit_mcmc` call.

        Raises
        ------
        ValueError
            If ``pumping`` is not provided or if LSF parameterization is inconsistent.
        """
        
        if config is not None:
            if config.data is not UNSET:
                data = config.data
            if config.window is not UNSET:
                window = config.window
            if config.pumping is not UNSET:
                pumping = config.pumping
            if config.isotopologues is not UNSET:
                isotopologues = config.isotopologues
            if config.systems is not UNSET:
                systems = config.systems
            if config.linelists is not UNSET:
                linelists = config.linelists
            if config.line_path is not UNSET:
                line_path = config.line_path
            if config.lsf is not UNSET:
                lsf = config.lsf
            if config.lsf_method is not UNSET:
                lsf_method = config.lsf_method
            if config.sigma is not UNSET:
                sigma = config.sigma
            if config.sigma1 is not UNSET:
                sigma1 = config.sigma1
            if config.sigma2 is not UNSET:
                sigma2 = config.sigma2
            if config.sigma_G is not UNSET:
                sigma_G = config.sigma_G
            if config.fwhm_L is not UNSET:
                fwhm_L = config.fwhm_L
            if config.ratio is not UNSET:
                ratio = config.ratio
            if config.A_min is not UNSET:
                A_min = config.A_min
            if config.a is not UNSET:
                a = config.a
            if config.name is not UNSET:
                name = config.name
            if config.logN is not UNSET:
                logN = config.logN
            if config.logN_by_iso is not UNSET:
                logN_by_iso = config.logN_by_iso
            if config.logQ is not UNSET:
                logQ = config.logQ
            if config.logQ_by_iso is not UNSET:
                logQ_by_iso = config.logQ_by_iso
            if config.T is not UNSET:
                T = config.T
            if config.T_by_iso is not UNSET:
                T_by_iso = config.T_by_iso
            if config.v_kms is not UNSET:
                v_kms = config.v_kms
            if config.v_kms_by_iso is not UNSET:
                v_kms_by_iso = config.v_kms_by_iso
            if config.dlam is not UNSET:
                dlam = config.dlam
            if config.dlam_by_iso is not UNSET:
                dlam_by_iso = config.dlam_by_iso
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
            if config.include_rotations is not UNSET:
                include_rotations = config.include_rotations
            if config.pumping_v_kms is not UNSET:
                pumping_v_kms = config.pumping_v_kms
            if config.pumping_dlam_A is not UNSET:
                pumping_dlam_A = config.pumping_dlam_A
            if config.model_wave is not UNSET:
                model_wave = config.model_wave
            if config.n_cores is not UNSET:
                n_cores = config.n_cores

        self.data = data
        self.wave_col = wave_col
        self.flux_col = flux_col
        self.error_col = error_col
        self.continuum_col = continuum_col
        self.omega = omega
        self.include_rotations = include_rotations
        
        self.pumping_v_kms = float(pumping_v_kms)
        self.pumping_dlam_A = float(pumping_dlam_A)

        self.window = window
        self.pumping = pumping

        self.isotopologues = isotopologues
        self.systems = systems
        self.linelists = linelists
        self.line_path = line_path

        self.A_min = float(A_min)
        self.name = name or "Fluorescence"
        self.a = a
        self.n_cores = n_cores

        self.logN = logN
        self.logN_by_iso = dict(logN_by_iso) if logN_by_iso is not None else None
        self.logQ = logQ
        self.logQ_by_iso = dict(logQ_by_iso) if logQ_by_iso is not None else None
        self.T = T
        self.T_by_iso = dict(T_by_iso) if T_by_iso is not None else None
        self.v_kms = v_kms
        self.v_kms_by_iso = dict(v_kms_by_iso) if v_kms_by_iso is not None else None
        self.dlam = dlam
        self.dlam_by_iso = dict(dlam_by_iso) if dlam_by_iso is not None else None

        self.q: Optional[Union[float, Dict[str, float]]] = None
        self.q_err: Optional[Union[float, Dict[str, float]]] = None
        self.logN_err: Optional[np.ndarray] = None
        self.logN_err_by_iso: Optional[Dict[str, np.ndarray]] = None

        if lsf is not None:
            self.lsf = lsf
            self.lsf_method = "Given"
            self.sigma = None
            self.sigma1 = None
            self.sigma2 = None
            self.sigma_G = None
            self.fwhm_L = None
            self.ratio = None
        else:
            self.lsf_method = lsf_method

            if lsf_method == "Gauss":
                self.sigma = 0.01 if sigma is None else float(sigma)
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, lsf_method)
                self.sigma1 = None
                self.sigma2 = None
                self.ratio = None
                self.sigma_G = None
                self.fwhm_L = None

            elif lsf_method == "2Gauss":
                if sigma1 is None or sigma2 is None or ratio is None:
                    raise ValueError("sigma1, sigma2, ratio required for '2Gauss'.")
                self.sigma1 = float(sigma1)
                self.sigma2 = float(sigma2)
                self.ratio = float(ratio)
                self.lsf = modeling.make_lsf(
                    {"sigma1": self.sigma1, "sigma2": self.sigma2, "ratio": self.ratio},
                    lsf_method,
                )
                self.sigma = None
                self.sigma_G = None
                self.fwhm_L = None

            elif lsf_method == "Gauss_Lorentz":
                if sigma_G is None or fwhm_L is None or ratio is None:
                    raise ValueError("sigma_G, fwhm_L, ratio required for 'Gauss_Lorentz'.")
                self.sigma_G = float(sigma_G)
                self.fwhm_L = float(fwhm_L)
                self.ratio = float(ratio)
                self.lsf = modeling.make_lsf(
                    {"sigma_G": self.sigma_G, "fwhm_L": self.fwhm_L, "ratio": self.ratio},
                    lsf_method,
                )
                self.sigma = None
                self.sigma1 = None
                self.sigma2 = None

            elif lsf_method == "Lorentz":
                if fwhm_L is None:
                    raise ValueError("fwhm_L required for 'Lorentz'.")
                self.fwhm_L = float(fwhm_L)
                self.lsf = modeling.make_lsf({"fwhm_L": self.fwhm_L}, lsf_method)
                self.sigma = None
                self.sigma1 = None
                self.sigma2 = None
                self.sigma_G = None
                self.ratio = None
            else:
                raise ValueError(f"Unsupported lsf_method: {lsf_method}")

        self.q_seeing_corrected: bool = False
        self.logN_seeing_corrected: bool = False

        self.priors: Dict[str, Tuple[float, float]] = {}
        self.param_keys: Tuple[str, ...] = ()
        self.median_params: Dict[str, float] = {}
        self.up_errors_params: Dict[str, float] = {}
        self.low_errors_params: Dict[str, float] = {}
        self.samples_pruned: Optional[np.ndarray] = None
        self.lnprob_pruned: Optional[np.ndarray] = None

        self.lines_by_iso: Optional[Dict[str, Any]] = None
        self.M_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.idx_to_level_by_iso: Optional[Dict[str, Any]] = None
        self.n_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.g_ph_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.g_en_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.g_ph_sum_by_iso: Optional[Dict[str, float]] = None
        self.g_en_sum_by_iso: Optional[Dict[str, float]] = None
        self.model_by_iso: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

        self.lines = None
        self.M = None
        self.idx_to_level = None
        self.n = None
        self.g_ph = None
        self.g_en = None
        self.g_ph_sum = None
        self.g_en_sum = None

        self.model_wave = model_wave
        self.median_model: Optional[np.ndarray] = None
        self.best_model: Optional[np.ndarray] = None
        self.model_p16: Optional[np.ndarray] = None
        self.model_p84: Optional[np.ndarray] = None

        self._synthesize_model()


    def _iso_list(self) -> list[str]:
        """Return isotopologues as a concrete list.

        Returns
        -------
        list[str]
            List of isotopologue labels from ``self.isotopologues``.
        """
        return [self.isotopologues] if isinstance(self.isotopologues, str) else list(self.isotopologues)

    @staticmethod
    def _km_per_arcsec(delta_au: float) -> float:
        """Convert angular scale to projected distance.

        Parameters
        ----------
        delta_au : float
            Observer-target distance in astronomical units.

        Returns
        -------
        float
            Kilometers subtended by 1 arcsec at ``delta_au``.
        """
        from . import _geometry
        return _geometry.km_per_arcsec(delta_au)

    @classmethod
    def _aperture_area_cm2(cls, aperture: dict, *, delta_au: float) -> u.Quantity:
        """Convert aperture definition into projected collecting area.

        Supported formats are:
        - ``{"type": "circular", "radius_arcsec": R}``
        - ``{"type": "rectangular", "width_arcsec": W, "length_arcsec": L}``

        Parameters
        ----------
        aperture : dict
            Aperture definition dictionary.
        delta_au : float
            Observer-target distance in AU.

        Returns
        -------
        astropy.units.Quantity
            Projected aperture area in ``cm^2``.

        Raises
        ------
        ValueError
            If ``aperture['type']`` is unsupported.
        """
        from . import _geometry
        return _geometry.aperture_area_cm2(cls, aperture, delta_au=delta_au)

    @classmethod
    def _sbpy_aperture(cls, aperture: dict, *, delta_au: float):
        """Build an ``sbpy`` aperture object.

        Parameters
        ----------
        aperture : dict
            Aperture definition dictionary.
        delta_au : float
            Observer-target distance in AU.

        Returns
        -------
        CircularAperture or RectangularAperture
            ``sbpy`` aperture instance for ``Haser.total_number``.

        Raises
        ------
        ValueError
            If ``aperture['type']`` is unsupported.
        """
        from . import _geometry
        return _geometry.sbpy_aperture(cls, aperture, delta_au=delta_au)

   
    def fit_mcmc(
        self,
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
        config: Optional["MCMCFitConfig"] = None,
    ) -> Dict[str, Any]:
        """Run MCMC using the current model state.

        This method forwards current wrapper defaults/state into
        :func:`modeling.mcmc_fitting`, updates instance attributes with posterior
        summaries, and keeps pumping-shift settings synchronized.

        .. note::
            Priors consist on a dict mapping parameter name to a (min, max) tuple defining the uniform prior range for that parameter. The possible keys are: ``"logN"``, ``"logQ"``, ``"T"``, ``"dlam"``, ``"v_kms"``, 'logN_<isotopologue>', 'logQ_<isotopologue>', 'T_<isotopologue>', 'dlam_<isotopologue>', 'v_kms_<isotopologue>' (any iso not found falls back to <parameter>) and lsf priors, ``sigma``, ``sigma1``, ``sigma2``, ``sigma_G``, ``fwhm_L``, ``ratio``.

        Parameters
        ----------
        data : Any, optional, default None
            Optional observed spectrum. If provided, updates ``self.data``.
        window : tuple[float, float], optional, default None
            Optional fit window ``(lambda_min_A, lambda_max_A)``.
            If provided, updates ``self.window``.
        pumping : Any, optional, default None
            Optional solar irradiance spectrum. If provided, updates ``self.pumping``.
        isotopologues : str or Sequence[str], optional, default None
            Optional isotopologue override for this fit.
            If provided, updates ``self.isotopologues``.
        systems : str or Sequence[str], optional, default None
            Optional CN system selector override. If provided,
            updates ``self.systems``.
        linelists : pandas.DataFrame or dict[str, pandas.DataFrame] or Sequence[pandas.DataFrame], optional, default None
            Optional line-list override. If provided, updates
            ``self.linelists``.
        nwalkers : int, optional, default 20
            Number of MCMC walkers. Default is ``20``.
        nsteps : int, optional, default 1000
            Number of MCMC steps. Default is ``1000``.
        n_cores : int, optional, default None
            Number of CPU cores used to parallelize the MCMC walker likelihood
            evaluations through ``multiprocessing`` (``spawn`` start method,
            safe on macOS/Linux/Windows). ``None`` falls back to
            ``self.n_cores``; if that is also ``None`` or ``1`` the sampler
            runs single-threaded. Values ``>1`` are clamped to
            :func:`os.cpu_count`. The tqdm progress bar (when
            ``progress=True``) keeps working under both modes. When provided,
            this value also updates ``self.n_cores``.

            .. important::

               When calling :meth:`fit_mcmc` with ``n_cores > 1`` from a
               ``.py`` script (not a Jupyter notebook), the call **must** be
               wrapped in ``if __name__ == "__main__":``. Each ``spawn``
               worker re-imports the user's script, and without the guard
               the call re-executes on import; Python aborts with a
               ``RuntimeError``. No guard is needed in notebooks or when
               ``n_cores`` is ``None``/``1``.
        priors : dict[str, tuple[float, float]], optional, default None
            Prior ranges for sampled parameters. If ``None``, uses
            stored priors or ``{"logN": (9.0, 15.0), "logQ": (-5.0, 0.0), "T": (10.0, 1000.0)}``.
        lsf : Callable[[numpy.ndarray], numpy.ndarray], optional, default None
            Optional custom LSF callable for fit-time synthesis.
        lsf_method : str, optional, default None
            Optional built-in LSF method override for this fit.
        make_plots : bool, optional, default True
            Generate diagnostic plots inside modeling fitter.
            Default is ``True``.
        progress : bool, optional, default True
            Show sampler progress output. Default is ``True``.
        A_min : float, optional, default None
            Optional threshold override; if provided updates ``self.A_min``.
        a : float, optional, default None
            Optional stretch-move parameter override; if provided updates ``self.a``.
        fig_file : str, optional, default "mcmc_fit"
            Figure file prefix passed to modeling fitter.
            Default is ``"mcmc_fit"``.
        verbose : bool, optional, default True
            Enable verbose output in modeling fitter. Default is ``True``.
        pruning : bool, optional, default True
            Enable posterior pruning in modeling fitter, check our publication for details. Default is ``True``.
        N_Model : int, optional, default 20000
            Number of elements in the model grid. Default is ``20000``.
        config : MCMCFitConfig, optional, default None
            Optional grouped configuration forwarded to
            :func:`cometspec.mcmc.mcmc_fitting`. Any field set on this object
            overrides the corresponding individual keyword argument.

        Returns
        -------
        dict[str, Any]
            Fit result dictionary from :func:`modeling.mcmc_fitting`.

        Raises
        ------
        ValueError
            If no data/pumping/window are available.

        Notes
        -----
        Result keys and mirrored attributes:

        - ``param_keys`` -> ``self.param_keys``
        - ``median_params`` -> ``self.median_params``
        - ``up_errors_params`` -> ``self.up_errors_params``
        - ``low_errors_params`` -> ``self.low_errors_params``
        - ``samples_pruned`` -> ``self.samples_pruned``
        - ``lnprob_pruned`` -> ``self.lnprob_pruned``
        - ``model_wave``, ``median_model``, ``best_model``, ``model_p16``, ``model_p84``
          -> corresponding instance attributes

        Side effects:
        Updates fit state and posterior products through :meth:`_update_from_result`,
        and stores the pumping-shift values used during the fit in
        ``self.pumping_v_kms`` and ``self.pumping_dlam_A``.

        If extreme prior values cause the rate-matrix solver to fail (``LinAlgError``
        from a numerically degenerate matrix, typically due to NaN/Inf collision rates
        at very high Q), ``solve_with_normalization_fast`` returns NaN populations.
        ``lnlike`` then detects a non-finite model and returns ``-inf``, so the walker
        step is rejected gracefully rather than crashing.
        """
        from . import _fitting
        return _fitting.run_fit_mcmc(
            self,
            data,
            window,
            pumping=pumping,
            isotopologues=isotopologues,
            systems=systems,
            linelists=linelists,
            nwalkers=nwalkers,
            nsteps=nsteps,
            n_cores=n_cores,
            priors=priors,
            lsf=lsf,
            lsf_method=lsf_method,
            make_plots=make_plots,
            progress=progress,
            A_min=A_min,
            a=a,
            fig_file=fig_file,
            verbose=verbose,
            pruning=pruning,
            N_Model=N_Model,
            config=config,
        )


    def update_model(
        self,
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
        """Update instance parameters and re-synthesize the model. 

        Any non-``None`` argument is applied to the instance. LSF settings are
        rebuilt according to ``lsf``/``lsf_method`` and then
        :meth:`_synthesize_model` is called. Notice that you can give the parameters 
        as keywords or you can update them directly, however this function should
        always be called after any parameter update to keep the model products consistent with the parameters.

        Parameters
        ----------
        isotopologues : str or Sequence[str], optional, default None
            Optional isotopologue selection update.
        systems : str or Sequence[str], optional, default None
            Optional CN systems selector update.
        linelists : pandas.DataFrame or dict[str, pandas.DataFrame], optional, default None
            Optional line-list update.
        logN : float, optional, default None
            Optional global ``log10(N/cm^2)`` update.
        logN_by_iso : dict[str, float], optional, default None
            Optional per-isotopologue ``log10(N/cm^2)`` update.
        logQ : float, optional, default None
            Optional ``logQ`` update.
        T : float, optional, default None
            Optional global temperature update.
        T_by_iso : dict[str, float], optional, default None
            Optional per-isotopologue temperature update. Falls back to ``T`` for isotopologues not present.
        v_kms : float, optional, default None
            Optional global emission velocity shift update.
        v_kms_by_iso : dict[str, float], optional, default None
            Optional per-isotopologue emission velocity shift update. Falls back to ``v_kms``.
        dlam : float, optional, default None
            Optional global emission wavelength shift update.
        dlam_by_iso : dict[str, float], optional, default None
            Optional per-isotopologue additive wavelength shift update. Falls back to ``dlam``.
        A_min : float, optional, default None
            Optional transition threshold update.
        lsf : Callable[[numpy.ndarray], numpy.ndarray], optional, default None
            Optional custom LSF callable update.
        lsf_method : str, optional, default None
            Optional LSF mode update.
        sigma : float, optional, default None
            Optional Gaussian sigma update.
        sigma1 : float, optional, default None
            Optional 2-Gaussian component sigma1 update.
        sigma2 : float, optional, default None
            Optional 2-Gaussian component sigma2 update.
        sigma_G : float, optional, default None
            Optional Gaussian sigma for Gauss-Lorentz update.
        fwhm_L : float, optional, default None
            Optional Lorentzian FWHM update.
        ratio : float, optional, default None
            Optional mixture ratio update.
        window : tuple[float, float], optional, default None
            Optional synthesis window update.
        pumping : Any, optional, default None
            Optional pumping spectrum update.
        data : Any, optional, default None
            Optional observed data update.
        wave_col : str, optional, default None
            Optional wavelength column-name update.
        flux_col : str, optional, default None
            Optional flux column-name update.
        error_col : str, optional, default None
            Optional uncertainty column-name update.
        continuum_col : str, optional, default None
            Optional continuum column-name update.
        omega : float, optional, default np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2
            Optional aperture solid-angle update.
            Default argument value matches constructor default.

        Raises
        ------
        ValueError
            If LSF parameters are inconsistent for the selected method.

        Notes
        -----
        Side effects:

        - Always calls :meth:`_synthesize_model`.
        - Resets ``self.q`` and ``self.q_err`` to ``None`` when ``logN``,
          ``logN_by_iso``, or isotopologue selection changes.
        """
        from . import _synthesis
        return _synthesis.apply_update_model(
            self,
            isotopologues=isotopologues,
            systems=systems,
            linelists=linelists,
            logN=logN,
            logN_by_iso=logN_by_iso,
            logQ=logQ,
            T=T,
            T_by_iso=T_by_iso,
            v_kms=v_kms,
            v_kms_by_iso=v_kms_by_iso,
            dlam=dlam,
            dlam_by_iso=dlam_by_iso,
            A_min=A_min,
            lsf=lsf,
            lsf_method=lsf_method,
            sigma=sigma,
            sigma1=sigma1,
            sigma2=sigma2,
            sigma_G=sigma_G,
            fwhm_L=fwhm_L,
            ratio=ratio,
            window=window,
            pumping=pumping,
            data=data,
            wave_col=wave_col,
            flux_col=flux_col,
            error_col=error_col,
            continuum_col=continuum_col,
            omega=omega,
            N_Model=N_Model,
        )


    def _synthesize_model(self) -> None:
        """Recompute internal line/rate/spectrum products from current state.

        This method builds transitions, rate matrices, level populations,
        g-factors, and synthesized spectra for each isotopologue, then stores
        both per-isotopologue containers and single-isotopologue convenience
        attributes.

        Raises
        ------
        ValueError
            If ``self.pumping`` or ``self.window`` is missing, or
            if line-list/isotopologue combinations are inconsistent.

        Notes
        -----
        Side effects:
        Updates ``self.lines_by_iso``, ``self.M_by_iso``, ``self.idx_to_level_by_iso``,
        ``self.n_by_iso``, ``self.g_ph_by_iso``, ``self.g_en_by_iso``,
        ``self.g_ph_sum_by_iso``, ``self.g_en_sum_by_iso``, ``self.model_by_iso``,
        ``self.model_wave``, and ``self.best_model`` plus flat single-iso shortcuts.
        """
        from . import _synthesis
        return _synthesis.synthesize_model(self)


    def n_lines(self) -> Union[int, Dict[str, int]]:
        """Print and return the number of transitions in the synthesized model.

        For a single isotopologue, prints and returns a single integer. For
        multiple isotopologues, prints one count per isotopologue plus the total
        and returns a ``{iso: count}`` dict.

        Returns
        -------
        int or dict[str, int]
            Line count per isotopologue, or a single int for one iso.

        Raises
        ------
        RuntimeError
            If the model has not been synthesized yet (i.e.
            :meth:`_synthesize_model` has not run, so ``lines_by_iso`` is unset).
        """
        lbi = getattr(self, "lines_by_iso", None)
        if not lbi:
            raise RuntimeError(
                "No synthesized lines available. Run update_model()/fit_mcmc() first."
            )

        counts = {iso: int(len(lines)) for iso, lines in lbi.items()}
        if len(counts) == 1:
            iso, n = next(iter(counts.items()))
            print(f"{iso}: {n} lines")
            return n

        width = max(len(iso) for iso in counts)
        for iso, n in counts.items():
            print(f"{iso:<{width}} : {n} lines")
        print(f"{'total':<{width}} : {sum(counts.values())} lines")
        return counts

    def print_linelist_origins(self) -> Dict[str, str]:
        """Print and return the source of each isotopologue's line list.

        For each isotopologue, prints either ``"custom (user-provided)"`` (when
        the line list came from the ``linelists`` argument) or the file path
        that would be / was loaded from the packaged defaults. Resolution
        mirrors :meth:`_synthesize_model`, so this can be called before or
        after synthesis.

        Returns
        -------
        dict[str, str]
            ``{iso: origin}`` ordered as ``self.isotopologues``.
        """
        iso_list = self._iso_list()
        line_paths = None
        if self.line_path is not None and iso_list:
            line_paths = {iso_list[0]: self.line_path}

        origins = modeling.linelist_origins(
            self.linelists, iso_list, line_paths=line_paths
        )

        width = max(len(iso) for iso in origins)
        for iso, src in origins.items():
            print(f"{iso:<{width}} : {src}")
        return origins


    def _update_from_result(
        self,
        result: Dict[str, Any],
        *,
        used_lsf: Optional[Callable[[np.ndarray], np.ndarray]],
        used_lsf_method: Optional[str],
    ) -> None:
        """Apply fit outputs to instance state.

        Parameters
        ----------
        result : dict[str, Any]
            Output dictionary produced by :func:`modeling.mcmc_fitting`.
        used_lsf : Callable[[numpy.ndarray], numpy.ndarray], optional
            LSF callable used in the fit call, if any.
        used_lsf_method : str, optional
            LSF method string used in the fit call when
            ``used_lsf`` is ``None``.

        Notes
        -----
        Side effects:

        - Updates posterior summaries and chains (``param_keys``, ``median_params``,
          uncertainty dictionaries, and pruned samples).
        - Updates core physical parameters when present in median parameters.
        - Updates LSF representation (either custom given callable or rebuilt method form).
        - Updates model grid and best/median model arrays.
        - Rebuilds ``self.model_by_iso`` entries through per-isotopologue temporary
          model synthesis.
        - Resets ``self.q``/``self.q_err`` when fitted ``logN`` values are present.
        """
        from . import _fitting
        return _fitting.update_from_result(
            self,
            result,
            used_lsf=used_lsf,
            used_lsf_method=used_lsf_method,
        )


    def compute_production_rate(
        self,
        *,
        delta_au: float,
        aperture: dict,
        parent_length_km: float,
        daughter_length_km: float,
        v_outflow_km_s: float,
        use_samples: bool = True,
        N_total_coma_km: float = 1e7,
    ) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
        """Estimate production rate ``log10(Q)`` from fitted column densities using a Haser model.

        The method uses a Haser model to convert aperture column-density
        constraints into total production rates, either from posterior samples or
        from current median ``logN`` values.

        .. note::
            This method assumes a Haser coma model. If you have a spatial
            flux profile instead of integrated column densities, use
            :meth:`compute_production_rate_from_profile` instead.

        Parameters
        ----------
        delta_au : float
            Geocentric distance in AU (for arcsec-to-km projection).
        aperture : dict
            Aperture geometry definition. Supported forms are
            ``{"type": "circular", "radius_arcsec": R}`` and
            ``{"type": "rectangular", "width_arcsec": W, "length_arcsec": L}``.
        parent_length_km : float
            Parent scale length in km.
        daughter_length_km : float
            Daughter scale length in km.
        v_outflow_km_s : float
            Gas outflow velocity in km/s.
        use_samples : bool, optional, default True
            If ``True``, uses ``self.samples_pruned`` chains; if
            ``False``, uses current median values in ``self.logN``/``self.logN_by_iso``.
            Default is ``True``. Without samples, there is no error estimate.
        N_total_coma_km : float, optional, default 1e7
            Radius in km used to approximate total coma count
            in the Haser model. Default is ``1e7``.

        Returns
        -------
        tuple[float, float] or dict[str, tuple[float, float]]
            For single isotopologue, returns ``(logQ50, logQerr)``. For
            multi-isotopologue models, returns ``dict[iso] = (logQ50, logQerr)``.

        Raises
        ------
        ValueError
            If required chains/values are missing or Haser aperture
            fraction is invalid.
        KeyError
            If expected isotopologue ``logN`` chains are missing.

        Notes
        -----
        Stores computed values in ``self.q`` and ``self.q_err``.
        """
        from . import _production
        return _production.compute_production_rate(
            self,
            delta_au=delta_au,
            aperture=aperture,
            parent_length_km=parent_length_km,
            daughter_length_km=daughter_length_km,
            v_outflow_km_s=v_outflow_km_s,
            use_samples=use_samples,
            N_total_coma_km=N_total_coma_km,
        )

    def add_slit_loss_error(
        self,
        *,
        lambda_nm: float,
        aperture: dict,
        correct: str = "both",  
        eps_min_arcsec_500: float = 0.7,
        eps_max_arcsec_500: float = 1.2,
        zmin_deg: float = 45.0,
        zmax_deg: float = 45.0,
        n_points: int = 2000,
    ) -> Union[float, Dict[str, float]]:
        """Add seeing/slit-loss systematic uncertainty to ``q_err``, ``logN_err``, or both.

        This method can be called after :meth:`compute_production_rate` (for ``correct="q"`` or ``correct="both"``) or before (for ``correct="logN"``). Each quantity tracks its own correction state, so they can be corrected independently or together. The final error is the quadrature sum of the original error and a slit-loss error. Check :func:`cometspec.helper.add_slit_loss_error_scalar` for details on the slit-loss error calculation.

        Parameters
        ----------
        lambda_nm : float
            Wavelength in nm used for seeing scaling.
        aperture : dict
            Aperture geometry definition dictionary. Same format as
            :meth:`compute_production_rate`.
        correct : str, optional, default "both"
            Which quantity to correct. One of ``"q"`` (default),
            ``"logN"``, or ``"both"``. Note that ``"q"`` requires ``self.q`` and
            ``self.q_err`` to be set; ``"logN"`` requires ``self.logN`` and
            ``self.logN_err`` to be set.
        eps_min_arcsec_500 : float, optional, default 0.7
            Minimum seeing FWHM at 500 nm and zenith, in arcsec.
            Default is ``0.7``.
        eps_max_arcsec_500 : float, optional, default 1.2
            Maximum seeing FWHM at 500 nm and zenith, in arcsec.
            Default is ``1.2``.
        zmin_deg : float, optional, default 45.0
            Minimum (best) zenith angle during observations, in degrees.
            Default is ``45.0``.
        zmax_deg : float, optional, default 45.0
            Maximum (worst) zenith angle during observations, in degrees.
            Default is ``45.0``.
        n_points : int, optional, default 2000
            Number of wavelength points used to sample the narrow window
            around ``lambda_nm``. Must be >= 2. Default is ``2000``.

        Raises
        ------
        ValueError
            If ``correct`` is not one of ``"q"``, ``"logN"``, ``"both"``;
            or if the required attributes (``self.q``, ``self.q_err``, ``self.logN``,
            ``self.logN_err``) are not set for the requested correction.
        KeyError
            If an isotopologue key is missing from ``self.q``,
            ``self.q_err``, ``self.logN_by_iso``, or ``self.logN_err_by_iso``.

        Notes
        -----
        Side effects:

        - Updates ``self.q_err`` / ``self.q_err_by_iso`` and sets
          ``self.q_seeing_corrected = True`` when correcting ``q``.
        - Updates ``self.logN_err`` / ``self.logN_err_by_iso`` and sets
          ``self.logN_seeing_corrected = True`` when correcting ``logN``.
        - If a quantity was already corrected, prints a warning and skips it
          without raising an error.
        """
        from . import _production
        return _production.add_slit_loss_error(
            self,
            lambda_nm=lambda_nm,
            aperture=aperture,
            correct=correct,
            eps_min_arcsec_500=eps_min_arcsec_500,
            eps_max_arcsec_500=eps_max_arcsec_500,
            zmin_deg=zmin_deg,
            zmax_deg=zmax_deg,
            n_points=n_points,
        )

    def compute_aperture_integral(
        self,
        *,
        aperture: dict,
        delta_au: float,
    ) -> u.Quantity:
        """Compute the geometric aperture integral **G** for a N(ρ) ∝ ρ⁻¹
        column-density profile (pure radial outflow), such that ``Q = v * N_ap / G``.

        Circular:     G = π ρ_ap / 2
        Rectangular:  G = [L·asinh(W/L) + W·asinh(L/W)] / 2

        Parameters
        ----------
        aperture : dict
            Aperture geometry. Same format as :meth:`compute_production_rate`.
        delta_au : float
            Geocentric distance in AU (for arcsec → km conversion).

        Returns
        -------
        astropy.units.Quantity
            G in cm.

        Raises
        ------
        ValueError
            If the aperture type is unsupported.
        """
        from . import _production
        return _production.compute_aperture_integral(
            self, aperture=aperture, delta_au=delta_au,
        )

    def compute_production_rate_from_profile(
        self,
        *,
        G: u.Quantity,
        delta_au: float,
        aperture: dict,
        v_outflow_km_s: float,
        use_samples: bool = True,
    ) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
        """Estimate ``log10(Q)`` from a column-density profile via a pre-computed
        aperture integral **G**.

        Examples
        --------
        .. code-block:: python

            G = model.compute_aperture_integral(
                    aperture=aperture, delta_au=delta_au, profile="rho^-1"
                )
            logQ, logQ_err = model.compute_production_rate_from_profile(
                    G=G, aperture=aperture, delta_au=delta_au,
                    v_outflow_km_s=0.85,
                )

        For a custom profile, compute G yourself (cm) and pass it directly,
        skipping :meth:`compute_aperture_integral` entirely::

            import astropy.units as u
            import numpy as np
            G_custom = my_numerical_integral(...) * u.cm
            logQ, logQ_err = model.compute_production_rate_from_profile(
                    G=G_custom, ...
                )

        Parameters
        ----------
        G : astropy.units.Quantity
            Geometric aperture integral in cm, from
            :meth:`compute_aperture_integral` or a custom calculation.
            Must satisfy ``Q = v * N_ap / G``.
        delta_au : float
            Geocentric distance in AU.
        aperture : dict
            Aperture geometry (needed to compute the collecting area A_cm²).
        v_outflow_km_s : float
            Gas outflow velocity in km/s.
        use_samples : bool, optional, default True
            If ``True``, propagates full MCMC chains; if ``False``, uses the
            median ``logN`` only. If ``False``, no error estimate is produced.

        Returns
        -------
        tuple[float, float]
            ``(logQ50, logQ_err)`` for single-isotopologue models.
        dict[str, tuple[float, float]]
            ``{iso: (logQ50, logQ_err)}`` for multi-isotopologue models.

        Raises
        ------
        ValueError
            If MCMC samples are missing and ``use_samples=True``.
        """
        from . import _production
        return _production.compute_production_rate_from_profile(
            self,
            G=G,
            delta_au=delta_au,
            aperture=aperture,
            v_outflow_km_s=v_outflow_km_s,
            use_samples=use_samples,
        )

  
    @staticmethod
    def _arcsec_to_km(*, delta_au: float) -> float:
        """Convert 1 arcsec at geocentric distance delta_au to km at the comet."""
        from . import _geometry
        return _geometry.arcsec_to_km(delta_au=delta_au)
    
   
    def save(self, filename: str) -> None:
        """Serialize model state to a pickle file.

        The saved state includes constructor kwargs, MCMC products, and derived
        production-rate fields ``q`` and ``q_err``. If the model uses a custom
        callable LSF (``lsf_method == "Given"``), that callable is not serialized.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        from . import _io
        return _io.save(self, filename)

    @classmethod
    def load(cls, filename: str) -> "FluorescenceModel":
        """Load a serialized :class:`FluorescenceModel` from disk.

        Parameters
        ----------
        filename : str
            Input pickle file path created by :meth:`save`.

        Returns
        -------
        FluorescenceModel
            Reconstructed fluorescence model instance.

        Raises
        ------
        ValueError
            If the file does not contain a compatible
            ``FluorescenceModel`` state version.

        Notes
        -----
        Side effects:
        If the original model used a custom callable LSF, a warning is printed because
        that callable is not serialized.
        """
        from . import _io
        return _io.load(cls, filename)
    

    