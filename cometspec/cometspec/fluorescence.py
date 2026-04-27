from __future__ import annotations

"""
High-level fluorescence model wrapper (UPDATED for isotopologues + systems)
+ Production rate + slit-loss systematic error support.

IMPORTANT ARCHITECTURE NOTE
---------------------------
This module is the main orchestration layer and should be treated as the
canonical source of runtime defaults for the fitting workflow. The low-level
numerical routines live in ``modeling.py``, but when fitting through
``FluorescenceModel.fit_mcmc`` the effective defaults for fallback parameters
(``logQ``, ``T``, ``v_kms``, ``dlam``) are taken from the current
``FluorescenceModel`` instance and forwarded explicitly to
``modeling.mcmc_fitting``.
"""

from typing import Any, Dict, Optional, Tuple, Callable, Sequence, Union

import numpy as np
import pickle
import pandas as pd

# used by slit-loss error helper (erf)
import math

# optional: astropy/sbpy only needed for production rate
from astropy import units as u
from sbpy.activity import Haser, CircularAperture, RectangularAperture

from . import helper, modeling


# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# Fluorescence Model
# ---------------------------------------------------------------------
class FluorescenceModel:
    """High-level CN fluorescence workflow wrapper.

    This class centralizes model state (line selection, physical parameters,
    LSF, and data columns) and is the preferred entry point for fitting and
    synthesis.

    IMPORTANT:
    When ``fit_mcmc`` calls ``modeling.mcmc_fitting``, fallback values for
    parameters that are not sampled are taken from this instance (``self.logQ``,
    ``self.T``, ``self.v_kms``, ``self.dlam``) and passed explicitly.

        Where to find parameters and results
        ------------------------------------
        The object exposes a stable set of attributes, even before fitting.

        Always available (initialized in ``__init__``):
        - Selection/config: ``isotopologues``, ``systems``, ``linelists``, ``line_path``
        - Physical/model controls: ``logN``, ``logN_by_iso``, ``logQ``, ``T``, ``v_kms``, ``dlam``
        - LSF config: ``lsf``, ``lsf_method``, ``sigma``, ``sigma1``, ``sigma2``,
            ``sigma_G``, ``fwhm_L``, ``ratio``
        - Data column names: ``wave_col``, ``flux_col``, ``error_col``, ``continuum_col``
        - Runtime knobs: ``A_min``, ``a``, ``threads``, ``include_rotations``

        Fitting outputs (exist before fit but start empty/None):
        - ``priors`` (empty dict until set)
        - ``param_keys`` (empty tuple)
        - ``median_params``, ``up_errors_params``, ``low_errors_params`` (empty dicts)
        - ``samples_pruned``, ``lnprob_pruned`` (None until MCMC is run)
        - ``median_model``, ``best_model``, ``model_p16``, ``model_p84``

        Per-isotopologue containers (created by synthesis/fitting):
        - ``lines_by_iso``, ``M_by_iso``, ``idx_to_level_by_iso``, ``n_by_iso``
        - ``g_ph_by_iso``, ``g_en_by_iso``, ``g_ph_sum_by_iso``, ``g_en_sum_by_iso``
        - ``model_by_iso``

        Derived production-rate fields:
        - ``q`` and ``q_err`` (set by :meth:`compute_production_rate`)
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
        threads: int = 1,
        name: Optional[str] = None,
        sigma: Optional[float] = 0.01,
        sigma1: Optional[float] = None,
        sigma2: Optional[float] = None,
        sigma_G: Optional[float] = None,
        fwhm_L: Optional[float] = None,
        ratio: Optional[float] = None,
        pumping_min_wave: Optional[float] = 2990.0010,
        pumping_max_wave: Optional[float] = 10009.9980,
        logN: Optional[float] = 11.0,
        logN_by_iso: Optional[Dict[str, float]] = None,
        logQ: Optional[float] = None,
        logQ_by_iso: Optional[Dict[str, float]] = None,
        T: Optional[float] = 300.0,
        v_kms: Optional[float] = 0.0,
        dlam: Optional[float] = 0.0,
        wave_col: str = "WAVE",
        flux_col: str = "FLUX_STACK",
        error_col: str = "ERR_STACK",
        continuum_col: str = "CONTINUUM",
        omega: float = np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2,
        include_rotations: bool = True,
        pumping_v_kms: float = 0.0,
        pumping_dlam_A: float = 0.0,
        model_wave: Optional[np.ndarray] = None,

    ) -> None:
        """Initialize a fluorescence model instance and synthesize the first model.

        The constructor stores configuration/state and immediately calls
        :meth:`_synthesize_model` to populate model products.

        :param data: Observed spectrum container (typically pandas DataFrame or astropy Table).
            Default is ``None``.
        :type data: Any or None
        :param window: Wavelength interval ``(lambda_min_A, lambda_max_A)`` used for synthesis.
            Default is ``(3850.0, 3900.0)``.
        :type window: tuple[float, float] or None
        :param pumping: Solar Irradiance spectrum at the comet, used to evaluate ``J_nu`` for transitions.
            This argument is required.
        :type pumping: Any
        :param isotopologues: Isotopologue label or list of labels. Default is ``"12C14N"``.
        :type isotopologues: str or Sequence[str]
        :param systems: CN system selector(s) forwarded to
            :func:`modeling.load_default_transitions`. Default is ``None``.
        :type systems: str or Sequence[str] or None
        :param linelists: Optional normalized line list(s). Accepted forms:

            * ``None`` -> every isotopologue loaded from packaged defaults.
            * Single :class:`pandas.DataFrame` -> assigned to the first isotopologue;
              the remaining isotopologues fall back to packaged defaults.
            * :class:`dict` mapping iso label to DataFrame -> isotopologues missing
              from the dict fall back to defaults; extra keys are ignored.
            * Sequence of DataFrames -> positionally paired with the first
              ``len(linelists)`` isotopologues; the remainder fall back to defaults.

            Loading a default for an unsupported isotopologue label (e.g.
            ``"COH"``) raises ``ValueError``. Default is ``None``.
        :type linelists: pandas.DataFrame or dict[str, pandas.DataFrame] or
            Sequence[pandas.DataFrame] or None
        :param line_path: Optional custom path for single-isotopologue line list loading.
            Default is ``None``.
        :type line_path: str or None
        :param lsf: Optional custom LSF callable. If provided, ``lsf_method`` parameterization
            values are ignored and ``lsf_method`` is stored as ``"Given"``. Default is ``None``.
        :type lsf: Callable[[numpy.ndarray], numpy.ndarray] or None
        :param lsf_method: Built-in LSF mode when ``lsf`` is not provided. Supported values are
            ``"Gauss"``, ``"2Gauss"``, ``"Gauss_Lorentz"``, and ``"Lorentz"``.
            Default is ``"Gauss"``.
        :type lsf_method: str or None
        :param A_min: Minimum Einstein-A threshold used by default CN transition loading.
            Default is ``1e4``.
        :type A_min: float
        :param a: Stored ``emcee`` stretch-move parameter used by :meth:`fit_mcmc`.
            Default is ``3.0``.
        :type a: float
        :param threads: Stored thread count forwarded to :func:`modeling.mcmc_fitting`.
            Default is ``1``.
        :type threads: int
        :param name: Optional model label. If ``None``, it is set to ``"Fluorescence"``.
            Default is ``None``.
        :type name: str or None
        :param sigma: Gaussian sigma for ``lsf_method="Gauss"``. Default is ``0.01``.
        :type sigma: float or None
        :param sigma1: Component 1 sigma for ``lsf_method="2Gauss"``.
            Default is ``None``.
        :type sigma1: float or None
        :param sigma2: Component 2 sigma for ``lsf_method="2Gauss"``.
            Default is ``None``.
        :type sigma2: float or None
        :param sigma_G: Gaussian sigma for ``lsf_method="Gauss_Lorentz"``.
            Default is ``None``.
        :type sigma_G: float or None
        :param fwhm_L: Lorentzian FWHM for ``lsf_method="Gauss_Lorentz"`` or ``"Lorentz"``.
            Default is ``None``.
        :type fwhm_L: float or None
        :param ratio: Mixture ratio for ``"2Gauss"`` and ``"Gauss_Lorentz"`` modes.
            Default is ``None``. It is gauss / lorentz for ``"Gauss_Lorentz"`` and gauss1 / gauss2 for ``"2Gauss"``.
        :type ratio: float or None
        :param pumping_min_wave: Minimum wavelength bound (A) used when loading default
            the solar irradiance. Default is ``2990.0010``.
        :type pumping_min_wave: float or None
        :param pumping_max_wave: Maximum wavelength bound (A) used when loading default
            the solar irradiance. Default is ``10009.9980``.
        :type pumping_max_wave: float or None
        :param logN: Default column density ``log10(N/cm^2)`` used for synthesis and fallback
            behavior. Default is ``11.0``.
        :type logN: float or None
        :param logN_by_iso: Optional per-isotopologue ``log10(N/cm^2)`` map.
            Default is ``None``.
        :type logN_by_iso: dict[str, float] or None
        :param logQ: Collisional production-rate proxy (log10 scale) default fallback.
            Default is ``-3.0``. Q in s^-1
        :type logQ: float or None
        :param logQ_by_iso: Optional per-isotopologue ``log10(Q/s^-1)`` map.
            When provided, each isotopologue uses its own collisional rate; any
            isotopologue missing from the map falls back to ``logQ``. Default is
            ``None``.
        :type logQ_by_iso: dict[str, float] or None
        :param T: Temperature in Kelvin used in synthesis/fallback behavior, see paper for meaning.
            Default is ``300.0``.
        :type T: float or None
        :param v_kms: Emission-frame Doppler velocity shift in km/s for spectral synthesis.
            Default is ``0.0``. This parameter moves the line list centers when synthesizing the model.
        :type v_kms: float or None
        :param dlam: Additive emission-frame wavelength shift in Angstrom.
            Default is ``0.0``. This parameter moves the line list centers when synthesizing the model.
        :type dlam: float or None
        :param wave_col: Wavelength column name in ``data``. Default is ``"WAVE"``.
        :type wave_col: str
        :param flux_col: Flux column name in ``data``. Default is ``"FLUX_STACK"``.
        :type flux_col: str
        :param error_col: Uncertainty column name in ``data``. Default is ``"ERR_STACK"``.
        :type error_col: str
        :param continuum_col: Continuum column name in ``data``. Default is ``"CONTINUUM"``.
        :type continuum_col: str
        :param omega: Aperture solid angle in sr. Default is
            ``np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2``, which assumes a circular apperture of radius 0.5".
        :type omega: float
        :type logN_seeing_corrected: bool
        :param include_rotations: Enable rotational collisions in synthesis/fitting.
            Default is ``True``.
        :type include_rotations: bool
        :param pumping_v_kms: Velocity shift (km/s) applied to line wavelengths when sampling
            pumping irradiance ``J_nu``. Default is ``0.0``.
        :type pumping_v_kms: float
        :param pumping_dlam_A: Additive shift (A) applied to line wavelengths when sampling
            pumping irradiance ``J_nu``. Default is ``0.0``.
        :type pumping_dlam_A: float
        :param model_wave: Optional pre-defined synthesis grid. If ``None``, a grid is built from
            ``window`` with 0.01 A spacing. Default is ``None``.
        :type model_wave: numpy.ndarray or None
        :raises ValueError: If ``pumping`` is not provided or if LSF parameterization is inconsistent.
        """
        if pumping is None:
            raise ValueError("Pumping spectrum must be provided to FluorescenceModel.")

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

        self.pumping_min_wave = pumping_min_wave
        self.pumping_max_wave = pumping_max_wave

        self.A_min = float(A_min)
        self.name = name or "Fluorescence"
        self.a = a
        self.threads = threads

        self.logN = logN
        self.logN_by_iso = dict(logN_by_iso) if logN_by_iso is not None else None
        self.logQ = logQ
        self.logQ_by_iso = dict(logQ_by_iso) if logQ_by_iso is not None else None
        self.T = T
        self.v_kms = v_kms
        self.dlam = dlam

        # NEW: derived quantities (production rate and its uncertainty)
        self.q: Optional[Union[float, Dict[str, float]]] = None
        self.q_err: Optional[Union[float, Dict[str, float]]] = None
        self.logN_err: Optional[Union[float, Dict[str, float]]] = None
        self.logN_err_by_iso: Optional[Dict[str, float]] = None

        # --- LSF setup ---
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

        # bools for slit loss
        self.q_seeing_corrected: bool = False
        self.logN_seeing_corrected: bool = False

        # --- Fit-related containers ---
        self.priors: Dict[str, Tuple[float, float]] = {}
        self.param_keys: Tuple[str, ...] = ()
        self.median_params: Dict[str, float] = {}
        self.up_errors_params: Dict[str, float] = {}
        self.low_errors_params: Dict[str, float] = {}
        self.samples_pruned: Optional[np.ndarray] = None
        self.lnprob_pruned: Optional[np.ndarray] = None

        # Derived model containers (now potentially per-iso internally)
        self.lines_by_iso: Optional[Dict[str, Any]] = None
        self.M_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.idx_to_level_by_iso: Optional[Dict[str, Any]] = None
        self.n_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.g_ph_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.g_en_by_iso: Optional[Dict[str, np.ndarray]] = None
        self.g_ph_sum_by_iso: Optional[Dict[str, float]] = None
        self.g_en_sum_by_iso: Optional[Dict[str, float]] = None
        self.model_by_iso: Optional[Dict[str, np.ndarray]] = None

        # "flat" convenience (single iso uses these)
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

    # -------------------------
    # Small internal helpers
    # -------------------------
    def _iso_list(self) -> list[str]:
        """Return isotopologues as a concrete list.

        :returns: List of isotopologue labels from ``self.isotopologues``.
        :rtype: list[str]
        """
        return [self.isotopologues] if isinstance(self.isotopologues, str) else list(self.isotopologues)

    @staticmethod
    def _km_per_arcsec(delta_au: float) -> float:
        """Convert angular scale to projected distance.

        :param delta_au: Observer-target distance in astronomical units.
        :type delta_au: float
        :returns: Kilometers subtended by 1 arcsec at ``delta_au``.
        :rtype: float
        """
        delta_km = (float(delta_au) * u.au).to(u.km).value
        return float(delta_km * np.tan(1.0 * u.arcsec.to(u.rad)))

    @classmethod
    def _aperture_area_cm2(cls, aperture: dict, *, delta_au: float) -> u.Quantity:
        """Convert aperture definition into projected collecting area.

        Supported formats are:
        - ``{"type": "circular", "radius_arcsec": R}``
        - ``{"type": "rectangular", "width_arcsec": W, "length_arcsec": L}``

        :param aperture: Aperture definition dictionary.
        :type aperture: dict
        :param delta_au: Observer-target distance in AU.
        :type delta_au: float
        :returns: Projected aperture area in ``cm^2``.
        :rtype: astropy.units.Quantity
        :raises ValueError: If ``aperture['type']`` is unsupported.
        """
        ap_type = aperture.get("type", "").lower().strip()
        km_per_arcsec = cls._km_per_arcsec(delta_au)

        if ap_type == "circular":
            R_arcsec = float(aperture["radius_arcsec"])
            R_km = R_arcsec * km_per_arcsec
            R_cm = R_km * 1e5
            return (np.pi * R_cm**2) * u.cm**2

        if ap_type == "rectangular":
            W_arcsec = float(aperture["width_arcsec"])
            L_arcsec = float(aperture["length_arcsec"])
            W_km = W_arcsec * km_per_arcsec
            L_km = L_arcsec * km_per_arcsec
            W_cm = W_km * 1e5
            L_cm = L_km * 1e5
            return (W_cm * L_cm) * u.cm**2

        raise ValueError("aperture['type'] must be 'circular' or 'rectangular'")

    @classmethod
    def _sbpy_aperture(cls, aperture: dict, *, delta_au: float):
        """Build an ``sbpy`` aperture object in kilometers.

        :param aperture: Aperture definition dictionary.
        :type aperture: dict
        :param delta_au: Observer-target distance in AU.
        :type delta_au: float
        :returns: ``sbpy`` aperture instance for ``Haser.total_number``.
        :rtype: CircularAperture or RectangularAperture
        :raises ValueError: If ``aperture['type']`` is unsupported.
        """
        ap_type = aperture.get("type", "").lower().strip()
        km_per_arcsec = cls._km_per_arcsec(delta_au)

        if ap_type == "circular":
            R_arcsec = float(aperture["radius_arcsec"])
            R_km = R_arcsec * km_per_arcsec
            return CircularAperture((R_km * u.km))

        if ap_type == "rectangular":
            W_arcsec = float(aperture["width_arcsec"])
            L_arcsec = float(aperture["length_arcsec"])
            W_km = W_arcsec * km_per_arcsec
            L_km = L_arcsec * km_per_arcsec
            # sbpy RectangularAperture uses (width, height)
            return RectangularAperture((W_km * u.km), (L_km * u.km))

        raise ValueError("aperture['type'] must be 'circular' or 'rectangular'")

    # ------------------------------------------------------------------
    # Public: run MCMC (UNCHANGED BODY, except it still calls _update_from_result)
    # ------------------------------------------------------------------
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
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lsf_method: Optional[str] = None,
        make_plots: bool = True,
        progress: bool = True,
        A_min: Optional[float] = None,
        a: Optional[float] = None,
        threads: Optional[int] = None,
        fig_file: str = "mcmc_fit",
        verbose: bool = True,
        pruning: bool = True,
        N_Model: Optional[int] = 20000,
    ) -> Dict[str, Any]:
        """Run MCMC using the current model state.

        This method forwards current wrapper defaults/state into
        :func:`modeling.mcmc_fitting`, updates instance attributes with posterior
        summaries, and keeps pumping-shift settings synchronized.

        :param data: Optional observed spectrum. If provided, updates ``self.data``.
        :type data: Any or None
        :param window: Optional fit window ``(lambda_min_A, lambda_max_A)``.
            If provided, updates ``self.window``.
        :type window: tuple[float, float] or None
        :param pumping: Optional pumping spectrum. If provided, updates ``self.pumping``.
        :type pumping: Any or None
        :param isotopologues: Optional isotopologue override for this fit.
            If provided, updates ``self.isotopologues``.
        :type isotopologues: str or Sequence[str] or None
        :param systems: Optional CN system selector override. If provided,
            updates ``self.systems``.
        :type systems: str or Sequence[str] or None
        :param linelists: Optional line-list override. If provided, updates
            ``self.linelists``.
        :type linelists: pandas.DataFrame or dict[str, pandas.DataFrame] or None
        :param nwalkers: Number of MCMC walkers. Default is ``20``.
        :type nwalkers: int
        :param nsteps: Number of MCMC steps. Default is ``1000``.
        :type nsteps: int
        :param priors: Prior ranges for sampled parameters. If ``None``, uses
            stored priors or ``{"logN": (9.0, 15.0), "logQ": (-5.0, 0.0), "T": (10.0, 1000.0)}``.
        :type priors: dict[str, tuple[float, float]] or None
        :param lsf: Optional custom LSF callable for fit-time synthesis.
        :type lsf: Callable[[numpy.ndarray], numpy.ndarray] or None
        :param lsf_method: Optional built-in LSF method override for this fit.
        :type lsf_method: str or None
        :param make_plots: Generate diagnostic plots inside modeling fitter.
            Default is ``True``.
        :type make_plots: bool
        :param progress: Show sampler progress output. Default is ``True``.
        :type progress: bool
        :param A_min: Optional threshold override; if provided updates ``self.A_min``.
        :type A_min: float or None
        :param a: Optional stretch-move parameter override; if provided updates ``self.a``.
        :type a: float or None
        :param threads: Optional thread-count override; if provided updates ``self.threads``.
        :type threads: int or None
        :param fig_file: Figure file prefix passed to modeling fitter.
            Default is ``"mcmc_fit"``.
        :type fig_file: str
        :param verbose: Enable verbose output in modeling fitter. Default is ``True``.
        :type verbose: bool
        :param pruning: Enable posterior pruning in modeling fitter. Default is ``True``.
        :type pruning: bool
        :param N_Model: Number of elements in the model grid. Default is ``20000``.
        :type N_Model: int
        :returns: Fit result dictionary from :func:`modeling.mcmc_fitting`.
        :rtype: dict[str, Any]
        :raises ValueError: If no data/pumping/window are available.

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
        """

        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("No data attached to this FluorescenceModel.")

        if window is None:
            if self.window is None:
                raise ValueError("window must be provided (argument or instance.window).")
            window = self.window
        else:
            self.window = window

        if pumping is None:
            pumping = self.pumping
        else:
            self.pumping = pumping
        if pumping is None:
            raise ValueError("pumping must be provided.")

        if isotopologues is not None:
            self.isotopologues = isotopologues
        if systems is not None:
            self.systems = systems
        if linelists is not None:
            self.linelists = linelists

        if priors is None:
            priors = self.priors or {"logN": (9.0, 15.0), "logQ": (-5.0, 0.0), "T": (10.0, 1000.0)}
            print("No priors provided, using default priors: logN, logQ, T.")
        self.priors = priors

        if lsf_method is None:
            lsf_method = self.lsf_method

        if A_min is None:
            A_min = self.A_min
        else:
            self.A_min = float(A_min)

        if a is None:
            a = self.a
        else:
            self.a = float(a)

        if threads is None:
            threads = self.threads
        else:
            self.threads = int(threads)

        # Canonical fallback bridge:
        # If priors do not sample a parameter, pass instance defaults to
        # modeling.mcmc_fitting so this wrapper remains the source of truth.
        init_logQ = float(self.logQ) if self.logQ is not None else None
        init_T = float(self.T) if self.T is not None else 300.0
        init_v_kms = float(self.v_kms) if self.v_kms is not None else 0.0
        init_dlam = float(self.dlam) if self.dlam is not None else 0.0
        init_logQ_by_iso = dict(self.logQ_by_iso) if self.logQ_by_iso is not None else None
        _init_logN_by_iso = dict(self.logN_by_iso) if self.logN_by_iso is not None else None
        init_logN = float(self.logN) if self.logN is not None else 11.0
        init_sigma = float(self.sigma) if self.sigma is not None else None
        init_sigma1 = float(self.sigma1) if self.sigma1 is not None else None
        init_sigma2 = float(self.sigma2) if self.sigma2 is not None else None
        init_sigma_G = float(self.sigma_G) if self.sigma_G is not None else None
        init_fwhm_L = float(self.fwhm_L) if self.fwhm_L is not None else None
        init_ratio = float(self.ratio) if self.ratio is not None else None


        # Pumping-shift bridge:
        # These settings control J_nu sampling and are intentionally independent
        # from emission-shift parameters that can also be sampled.
        pumping_v_kms = float(getattr(self, "pumping_v_kms", 0.0))
        pumping_dlam_A = float(getattr(self, "pumping_dlam_A", 0.0))

        result = modeling.mcmc_fitting(
            self.data,
            window,
            pumping=pumping,
            isotopologues=self.isotopologues,
            systems=self.systems,
            linelists=self.linelists,
            nwalkers=nwalkers,
            nsteps=nsteps,
            priors=priors,
            lsf=lsf,
            lsf_method=lsf_method,
            make_plots=make_plots,
            progress=progress,
            A_min=float(A_min),
            a=float(a),
            threads=int(threads),
            
            # ✅ Pumping shift (affects J_nu → line ratios)
            velocity_kms=pumping_v_kms,
            delta_lambda_A=pumping_dlam_A,

            # ✅ Fallbacks if not fit
            init_logQ_by_iso=init_logQ_by_iso,
            init_logQ=init_logQ,
            init_T=init_T,
            init_v_kms=init_v_kms,
            init_dlam=init_dlam,
            init_logN_by_iso=_init_logN_by_iso,
            init_logN=init_logN,
            init_sigma=init_sigma,
            init_sigma1=init_sigma1,
            init_sigma2=init_sigma2,
            init_sigma_G=init_sigma_G,
            init_fwhm_L=init_fwhm_L,
            init_ratio=init_ratio,

            fig_file=fig_file,
            wave_col=self.wave_col,
            flux_col=self.flux_col,
            error_col=self.error_col,
            continuum_col=self.continuum_col,
            omega=self.omega,
            verbose=verbose,
            pruning=pruning,
            include_rotations=self.include_rotations,
            N_Model=N_Model,
        )

        
        self._update_from_result(result, used_lsf=lsf, used_lsf_method=lsf_method)

        # ✅ Ensure wrapper synthesis uses the same pumping shift used by the fit
        self.pumping_v_kms = pumping_v_kms
        self.pumping_dlam_A = pumping_dlam_A

        return result


    # ------------------------------------------------------------------
    # Public: update params & resynthesize (ONLY ADD: q/q_err reset logic)
    # ------------------------------------------------------------------
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
        v_kms: Optional[float] = None,
        dlam: Optional[float] = None,
        A_min: Optional[float] = None,
        pumping_min_wave: Optional[float] = None,
        pumping_max_wave: Optional[float] = None,
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
        :meth:`_synthesize_model` is called.

        :param isotopologues: Optional isotopologue selection update.
        :type isotopologues: str or Sequence[str] or None
        :param systems: Optional CN systems selector update.
        :type systems: str or Sequence[str] or None
        :param linelists: Optional line-list update.
        :type linelists: pandas.DataFrame or dict[str, pandas.DataFrame] or None
        :param logN: Optional global ``log10(N/cm^2)`` update.
        :type logN: float or None
        :param logN_by_iso: Optional per-isotopologue ``log10(N/cm^2)`` update.
        :type logN_by_iso: dict[str, float] or None
        :param logQ: Optional ``logQ`` update.
        :type logQ: float or None
        :param T: Optional temperature update.
        :type T: float or None
        :param v_kms: Optional emission velocity shift update.
        :type v_kms: float or None
        :param dlam: Optional emission wavelength shift update.
        :type dlam: float or None
        :param A_min: Optional transition threshold update.
        :type A_min: float or None
        :param pumping_min_wave: Optional minimum transition wavelength update.
        :type pumping_min_wave: float or None
        :param pumping_max_wave: Optional maximum transition wavelength update.
        :type pumping_max_wave: float or None
        :param lsf: Optional custom LSF callable update.
        :type lsf: Callable[[numpy.ndarray], numpy.ndarray] or None
        :param lsf_method: Optional LSF mode update.
        :type lsf_method: str or None
        :param sigma: Optional Gaussian sigma update.
        :type sigma: float or None
        :param sigma1: Optional 2-Gaussian component sigma1 update.
        :type sigma1: float or None
        :param sigma2: Optional 2-Gaussian component sigma2 update.
        :type sigma2: float or None
        :param sigma_G: Optional Gaussian sigma for Gauss-Lorentz update.
        :type sigma_G: float or None
        :param fwhm_L: Optional Lorentzian FWHM update.
        :type fwhm_L: float or None
        :param ratio: Optional mixture ratio update.
        :type ratio: float or None
        :param window: Optional synthesis window update.
        :type window: tuple[float, float] or None
        :param pumping: Optional pumping spectrum update.
        :type pumping: Any or None
        :param data: Optional observed data update.
        :type data: Any or None
        :param wave_col: Optional wavelength column-name update.
        :type wave_col: str or None
        :param flux_col: Optional flux column-name update.
        :type flux_col: str or None
        :param error_col: Optional uncertainty column-name update.
        :type error_col: str or None
        :param continuum_col: Optional continuum column-name update.
        :type continuum_col: str or None
        :param omega: Optional aperture solid-angle update.
            Default argument value matches constructor default.
        :type omega: float
        :raises ValueError: If LSF parameters are inconsistent for the selected method.

        Side effects:
        - Always calls :meth:`_synthesize_model`.
        - Resets ``self.q`` and ``self.q_err`` to ``None`` when ``logN``,
          ``logN_by_iso``, or isotopologue selection changes.
        """
        if data is not None:
            self.data = data
        if pumping is not None:
            self.pumping = pumping
        if window is not None:
            self.window = window

        # --- selection updates ---
        if isotopologues is not None:
            self.isotopologues = isotopologues
        if systems is not None:
            self.systems = systems
        if linelists is not None:
            self.linelists = linelists

        # --- physical updates ---
        if logN is not None:
            self.logN = float(logN)
        if logN_by_iso is not None:
            self.logN_by_iso = dict(logN_by_iso)
        if logQ is not None:
            self.logQ = float(logQ)
        if T is not None:
            self.T = float(T)
        if v_kms is not None:
            self.v_kms = float(v_kms)
        if dlam is not None:
            self.dlam = float(dlam)

        if A_min is not None:
            self.A_min = float(A_min)
        if pumping_min_wave is not None:
            self.pumping_min_wave = float(pumping_min_wave)
        if pumping_max_wave is not None:
            self.pumping_max_wave = float(pumping_max_wave)

        # --- LSF handling (same as your previous logic) ---
        if lsf is not None:
            self.lsf = lsf
            self.lsf_method = "Given"
            for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
                setattr(self, name, None)
        elif lsf_method is not None:
            self.lsf_method = lsf_method
            if lsf_method == "Gauss":
                self.sigma = float(sigma if sigma is not None else 0.01)
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")
                self.sigma1 = self.sigma2 = self.sigma_G = self.fwhm_L = self.ratio = None
            elif lsf_method == "2Gauss":
                if sigma1 is None or sigma2 is None or ratio is None:
                    raise ValueError("sigma1, sigma2, ratio required for '2Gauss'.")
                self.sigma1 = float(sigma1)
                self.sigma2 = float(sigma2)
                self.ratio = float(ratio)
                self.lsf = modeling.make_lsf(
                    {"sigma1": self.sigma1, "sigma2": self.sigma2, "ratio": self.ratio},
                    "2Gauss",
                )
                self.sigma = self.sigma_G = self.fwhm_L = None
            elif lsf_method == "Gauss_Lorentz":
                if sigma_G is None or fwhm_L is None or ratio is None:
                    raise ValueError("sigma_G, fwhm_L, ratio required for 'Gauss_Lorentz'.")
                self.sigma_G = float(sigma_G)
                self.fwhm_L = float(fwhm_L)
                self.ratio = float(ratio)
                self.lsf = modeling.make_lsf(
                    {"sigma_G": self.sigma_G, "fwhm_L": self.fwhm_L, "ratio": self.ratio},
                    "Gauss_Lorentz",
                )
                self.sigma = self.sigma1 = self.sigma2 = None
            elif lsf_method == "Lorentz":
                if fwhm_L is None:
                    raise ValueError("fwhm_L required for 'Lorentz'.")
                self.fwhm_L = float(fwhm_L)
                self.lsf = modeling.make_lsf({"fwhm_L": self.fwhm_L}, "Lorentz")
                self.sigma = self.sigma1 = self.sigma2 = self.sigma_G = self.ratio = None
            else:
                raise ValueError(f"Unsupported lsf_method: {lsf_method}")
        else:
            # rebuild from stored params
            if self.lsf_method == "Gauss":
                if self.sigma is None:
                    self.sigma = 0.01
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")
            elif self.lsf_method == "2Gauss":
                self.lsf = modeling.make_lsf(
                    {"sigma1": self.sigma1, "sigma2": self.sigma2, "ratio": self.ratio},
                    "2Gauss",
                )
            elif self.lsf_method == "Gauss_Lorentz":
                self.lsf = modeling.make_lsf(
                    {"sigma_G": self.sigma_G, "fwhm_L": self.fwhm_L, "ratio": self.ratio},
                    "Gauss_Lorentz",
                )
            elif self.lsf_method == "Lorentz":
                self.lsf = modeling.make_lsf({"fwhm_L": self.fwhm_L}, "Lorentz")

        # NEW: if logN or isotope selection changed, production rate is stale
        if (logN is not None) or (logN_by_iso is not None) or (isotopologues is not None):
            self.q = None
            self.q_err = None

        if omega is not None:
            self.omega = omega
        if wave_col is not None:
            self.wave_col = wave_col
        if flux_col is not None:
            self.flux_col = flux_col
        if error_col is not None:
            self.error_col = error_col
        if continuum_col is not None:
            self.continuum_col = continuum_col

        self._synthesize_model()

    # ------------------------------------------------------------------
    # Internal: rebuild model from current parameters (UNCHANGED BODY)
    # ------------------------------------------------------------------
    def _synthesize_model(self) -> None:
        """Recompute internal line/rate/spectrum products from current state.

        This method builds transitions, rate matrices, level populations,
        g-factors, and synthesized spectra for each isotopologue, then stores
        both per-isotopologue containers and single-isotopologue convenience
        attributes.

        :raises ValueError: If ``self.pumping`` or ``self.window`` is missing, or
            if line-list/isotopologue combinations are inconsistent.

        Side effects:
        Updates ``self.lines_by_iso``, ``self.M_by_iso``, ``self.idx_to_level_by_iso``,
        ``self.n_by_iso``, ``self.g_ph_by_iso``, ``self.g_en_by_iso``,
        ``self.g_ph_sum_by_iso``, ``self.g_en_sum_by_iso``, ``self.model_by_iso``,
        ``self.model_wave``, and ``self.best_model`` plus flat single-iso shortcuts.
        """
        if self.pumping is None:
            raise ValueError("Pumping spectrum is required.")
        if self.window is None:
            raise ValueError("window is required.")

        iso_list = self._iso_list()

        # 1) transitions: user-provided isos win; the rest fall back to defaults.
        line_paths = None
        if self.line_path is not None and iso_list:
            line_paths = {iso_list[0]: self.line_path}

        trans_by_iso = modeling.resolve_linelists_with_defaults(
            self.linelists,
            iso_list,
            systems=self.systems,
            A_min=self.A_min,
            lambda_min_A=float(self.pumping_min_wave),
            lambda_max_A=float(self.pumping_max_wave),
            use_omega_labels=False,
            line_paths=line_paths,
        )
        if self.linelists is not None and self.A_min is not None:
            trans_by_iso = {
                iso: df[df["A_ul"] >= self.A_min].reset_index(drop=True)
                for iso, df in trans_by_iso.items()
            }

        # 2) per-iso solve + sum spectrum
        lines_by_iso: Dict[str, Any] = {}
        M_by_iso: Dict[str, np.ndarray] = {}
        idx_by_iso: Dict[str, Any] = {}
        n_by_iso: Dict[str, np.ndarray] = {}
        gph_by_iso: Dict[str, np.ndarray] = {}
        gen_by_iso: Dict[str, np.ndarray] = {}
        gphsum_by_iso: Dict[str, float] = {}
        gensum_by_iso: Dict[str, float] = {}
        model_by_iso: Dict[str, np.ndarray] = {}

        if self.model_wave is None:
            wave = np.arange(self.window[0], self.window[1] + 0.01, 0.01, dtype=float)
            self.model_wave = wave
        else:
            wave = np.asarray(self.model_wave, float)
            # check if the window is different from the wave range, if so, update it
            if (wave.min() < self.window[0]) or (wave.max() > self.window[1]):
                wave = np.arange(self.window[0], self.window[1] + 0.01, 0.01, dtype=float)
            if (wave.min() > self.window[0]) or (wave.max() < self.window[1]):
                wave = np.arange(self.window[0], self.window[1] + 0.01, 0.01, dtype=float)
            self.model_wave = wave
        spec_total = np.zeros_like(wave, dtype=float)

        for iso, df_trans in trans_by_iso.items():
            # ✅ Pumping shift consistent with fitter (affects J_nu and thus line ratios)
            lines_theta = modeling.attach_pumping_and_labels(
                df_trans,
                self.pumping,
                line_v_kms=float(self.pumping_v_kms),
                line_dlam_A=float(self.pumping_dlam_A),
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

            if self.include_rotations:
                coll_scaf = modeling.precompute_cn_collision_scaffold_fast(
                    lines_out, idx_to_level, iso_name=iso,
                )
            else:
                coll_scaf = dict(iu=np.array([], int), il=np.array([], int),
                                gu=np.array([]), gl=np.array([]), dE=np.array([]))

            M = M_rad.copy()
            if len(iso_list) == 1:
            # ✅ Collisions only if logQ/T are defined and include_rotations
                if self.logQ is not None and self.T is not None and self.include_rotations:
                    Q_lin = 10.0 ** float(self.logQ)
                    if np.isfinite(Q_lin) and Q_lin > 0.0:
                        Cup_work = np.empty_like(coll_scaf.get("iu", np.array([], dtype=int)), dtype=float)
                        modeling.apply_collisions_inplace_fast(M, coll_scaf, Q=Q_lin, T=float(self.T), Cup_work=Cup_work)
            else:
                # multi-iso: only apply collisions if logQ_by_iso is defined for this iso
                logQ_i = None
                if self.logQ_by_iso is not None and iso in self.logQ_by_iso:
                    try:
                        logQ_i = float(self.logQ_by_iso[iso])
                    except TypeError:
                        pass  # logQ is not a valid float, so we skip collisions for this iso
                elif self.logQ is not None:
                    logQ_i = float(self.logQ)

                if logQ_i is not None:
                    Q_lin = 10.0 ** logQ_i
                    if np.isfinite(Q_lin) and Q_lin > 0.0 and self.include_rotations:
                        Cup_work = np.empty_like(coll_scaf.get("iu", np.array([], dtype=int)), dtype=float)
                        modeling.apply_collisions_inplace_fast(M, coll_scaf, Q=Q_lin, T=float(self.T), Cup_work=Cup_work)
            n = modeling.solve_with_normalization(M, verbose=False)
            g_ph, g_en, g_ph_sum, g_en_sum = modeling.g_factors(lines_out, n, A_col="A_ul")

            # logN for this iso
            if len(iso_list) == 1:
                logN_i = float(self.logN)
            else:
                if self.logN_by_iso is not None and iso in self.logN_by_iso:
                    logN_i = float(self.logN_by_iso[iso])
                else:
                    if self.logN is None:
                        raise ValueError("For multi-iso, provide logN_by_iso or set logN as a common value.")
                    logN_i = float(self.logN)

            # ✅ Emission shift is separate, applied in spectrum synthesis (same as fit model_flux)
            _, spec_i = modeling.synth_spectrum_from_lines(
                lines_out,
                g_line_energy=g_en,
                lam_min=float(wave.min()),
                lam_max=float(wave.max()),
                lam_col="Wave_vac_AA",
                N_col_cm2=10.0 ** logN_i,
                Omega_sr=self.omega,
                grid=wave,
                lsf=self.lsf,  # you said you provide this -> fixed LSF, good
                v_shift_kms=float(self.v_kms or 0.0),
                dlam_shift_A=float(self.dlam or 0.0),
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


        self.lines_by_iso = lines_by_iso
        self.M_by_iso = M_by_iso
        self.idx_to_level_by_iso = idx_by_iso
        self.n_by_iso = n_by_iso
        self.g_ph_by_iso = gph_by_iso
        self.g_en_by_iso = gen_by_iso
        self.g_ph_sum_by_iso = gphsum_by_iso
        self.g_en_sum_by_iso = gensum_by_iso
        self.model_by_iso = model_by_iso

        if len(iso_list) == 1:
            iso0 = iso_list[0]
            self.lines = lines_by_iso[iso0]
            self.M = M_by_iso[iso0]
            self.idx_to_level = idx_by_iso[iso0]
            self.n = n_by_iso[iso0]
            self.g_ph = gph_by_iso[iso0]
            self.g_en = gen_by_iso[iso0]
            self.g_ph_sum = gphsum_by_iso[iso0]
            self.g_en_sum = gensum_by_iso[iso0]
        else:
            self.lines = None
            self.M = None
            self.idx_to_level = None
            self.n = None
            self.g_ph = None
            self.g_en = None
            self.g_ph_sum = None
            self.g_en_sum = None

        self.model_wave = wave
        self.best_model = spec_total


    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def n_lines(self) -> Union[int, Dict[str, int]]:
        """Print and return the number of transitions in the synthesized model.

        For a single isotopologue, prints and returns a single integer. For
        multiple isotopologues, prints one count per isotopologue plus the total
        and returns a ``{iso: count}`` dict.

        :raises RuntimeError: If the model has not been synthesized yet (i.e.
            :meth:`_synthesize_model` has not run, so ``lines_by_iso`` is unset).
        :returns: Line count per isotopologue, or a single int for one iso.
        :rtype: int or dict[str, int]
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

        :returns: ``{iso: origin}`` ordered as ``self.isotopologues``.
        :rtype: dict[str, str]
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


    # ------------------------------------------------------------------
    # Internal: apply MCMC result (ONLY ADD: q/q_err reset if logN changed)
    # ------------------------------------------------------------------
    def _update_from_result(
        self,
        result: Dict[str, Any],
        *,
        used_lsf: Optional[Callable[[np.ndarray], np.ndarray]],
        used_lsf_method: Optional[str],
    ) -> None:
        """Apply fit outputs to instance state.

                :param result: Output dictionary produced by :func:`modeling.mcmc_fitting`.
                :type result: dict[str, Any]
                :param used_lsf: LSF callable used in the fit call, if any.
                :type used_lsf: Callable[[numpy.ndarray], numpy.ndarray] or None
                :param used_lsf_method: LSF method string used in the fit call when
                        ``used_lsf`` is ``None``.
                :type used_lsf_method: str or None

                Side effects:
                - Updates posterior summaries and chains (``param_keys``, ``median_params``,
                    uncertainty dictionaries, and pruned samples).
                - Updates core physical parameters when present in median parameters.
                - Updates LSF representation (either custom given callable or rebuilt method form).
                - Updates model envelopes and best/median model arrays.
                - Rebuilds ``self.model_by_iso`` entries through per-isotopologue temporary
                    model synthesis.
                - Resets ``self.q``/``self.q_err`` when fitted ``logN`` values are present.
            """
        self.param_keys = tuple(result.get("param_keys", ()))
        self.median_params = dict(result.get("median_params", {}))
        self.up_errors_params = dict(result.get("up_errors_params", {}))
        self.low_errors_params = dict(result.get("low_errors_params", {}))

        self.samples_pruned = result.get("samples_pruned")
        self.lnprob_pruned = result.get("lnprob_pruned")

        self.q_seeing_corrected = False
        self.logN_seeing_corrected = False
        
        for name in ("logN", "logQ", "T", "v_kms", "dlam"):
            if name in self.median_params:
                setattr(self, name, float(self.median_params[name]))
                if name == "logN":
                    self.logN_err = np.array((
                        float(self.up_errors_params.get(name, 0.0)),
                        float(self.low_errors_params.get(name, 0.0)),
                    ))

        iso_list = self._iso_list()
        any_isoN = any((f"logN_{iso}" in self.median_params) for iso in iso_list)
        if any_isoN:
            self.logN_by_iso = {}
            self.logN_err_by_iso = {}
            for iso in iso_list:
                key = f"logN_{iso}"
                if key in self.median_params:
                    self.logN_by_iso[iso] = float(self.median_params[key])
                    self.logN_err_by_iso[iso] = np.array((
                        float(self.up_errors_params.get(key, 0.0)),
                        float(self.low_errors_params.get(key, 0.0)),
                    ))

        # LSF update (same as your current)
        if used_lsf is not None:
            self.lsf = used_lsf
            self.lsf_method = "Given"
            for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
                setattr(self, name, None)
        else:
            self.lsf_method = used_lsf_method
            if self.lsf_method == "Gauss":
                self.sigma = float(self.median_params.get("sigma", 0.01))
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")
            elif self.lsf_method == "2Gauss":
                vals = {}
                for nm in ("sigma1", "sigma2", "ratio"):
                    if nm in self.median_params:
                        vals[nm] = float(self.median_params[nm])
                        setattr(self, nm, vals[nm])
                if len(vals) == 3:
                    self.lsf = modeling.make_lsf(vals, "2Gauss")
            elif self.lsf_method == "Gauss_Lorentz":
                vals = {}
                for nm in ("sigma_G", "fwhm_L", "ratio"):
                    if nm in self.median_params:
                        vals[nm] = float(self.median_params[nm])
                        setattr(self, nm, vals[nm])
                if len(vals) == 3:
                    self.lsf = modeling.make_lsf(vals, "Gauss_Lorentz")
            elif self.lsf_method == "Lorentz":
                if "fwhm_L" in self.median_params:
                    self.fwhm_L = float(self.median_params["fwhm_L"])
                    self.lsf = modeling.make_lsf({"fwhm_L": self.fwhm_L}, "Lorentz")

        self.median_model = result.get("median_model", None)
        self.best_model = result.get("best_model", None)
        self.model_wave = result.get("model_wave", None)
        self.model_p16 = result.get("model_p16", None)
        self.model_p84 = result.get("model_p84", None)

        # NEW: if fitted logN changed, production rate is stale
        logN_keys = {"logN"} | {f"logN_{iso}" for iso in iso_list}
        if any(k in self.median_params for k in logN_keys):
            self.q = None
            self.q_err = 0
        
        for i in iso_list:
            non_iso_list = [j for j in iso_list if j != i]
            params_per_iso = {k: v for k, v in self.median_params.items() if not any(k == f"logN_{j}" for j in non_iso_list)}
            
            # we need to synthetize a model just with this iso's logN and the pther parameters
            sub_model = FluorescenceModel(
                data=self.data,
                window=self.window,
                pumping=self.pumping,
                isotopologues=[i],
                systems=self.systems,
                linelists=self.linelists,
                line_path=self.line_path,
                logN=params_per_iso.get(f"logN_{i}", self.logN),
                logQ=params_per_iso.get(f"logQ_{i}", self.logQ),
                T=params_per_iso.get("T", self.T),
                v_kms=params_per_iso.get("v_kms", self.v_kms),
                dlam=params_per_iso.get("dlam", self.dlam),
                A_min=self.A_min,
                pumping_min_wave=self.pumping_min_wave,
                pumping_max_wave=self.pumping_max_wave,
                lsf=self.lsf,
                lsf_method=self.lsf_method,
                sigma=self.sigma,
                sigma1=self.sigma1,
                sigma2=self.sigma2,
                sigma_G=self.sigma_G,
                fwhm_L=self.fwhm_L,
                ratio=self.ratio,
                omega=self.omega,
                wave_col=self.wave_col,
                flux_col=self.flux_col,
                error_col=self.error_col,
                continuum_col=self.continuum_col,
                include_rotations=self.include_rotations,
            )
            self.model_by_iso[i] = (sub_model.model_wave, sub_model.best_model)

    # ------------------------------------------------------------------
    # NEW: Production rate from fitted chains (single or multi-iso)
    # ------------------------------------------------------------------
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
        """Estimate production rate ``log10(Q)`` from fitted column densities.

        The method uses a Haser model to convert aperture column-density
        constraints into total production rates, either from posterior samples or
        from current median ``logN`` values.

        :param delta_au: Geocentric distance in AU (for arcsec-to-km projection).
        :type delta_au: float
        :param aperture: Aperture geometry definition. Supported forms are
            ``{"type": "circular", "radius_arcsec": R}`` and
            ``{"type": "rectangular", "width_arcsec": W, "length_arcsec": L}``.
        :type aperture: dict
        :param parent_length_km: Parent scale length in km.
        :type parent_length_km: float
        :param daughter_length_km: Daughter scale length in km.
        :type daughter_length_km: float
        :param v_outflow_km_s: Gas outflow velocity in km/s.
        :type v_outflow_km_s: float
        :param use_samples: If ``True``, uses ``self.samples_pruned`` chains; if
            ``False``, uses current median values in ``self.logN``/``self.logN_by_iso``.
            Default is ``True``.
        :type use_samples: bool
        :param N_total_coma_km: Radius in km used to approximate total coma count
            in the Haser model. Default is ``1e7``.
        :type N_total_coma_km: float
        :returns: For single isotopologue, returns ``(logQ50, logQerr)``. For
            multi-isotopologue models, returns ``dict[iso] = (logQ50, logQerr)``.
        :rtype: tuple[float, float] or dict[str, tuple[float, float]]
        :raises ValueError: If required chains/values are missing or Haser aperture
            fraction is invalid.
        :raises KeyError: If expected isotopologue ``logN`` chains are missing.

        Side effects:
        Stores computed values in ``self.q`` and ``self.q_err``.
        """
        iso_list = self._iso_list()

        # Build Haser model
        haser = Haser(
            Q=1 * u.s**-1,
            v=float(v_outflow_km_s) * u.km / u.s,
            parent=float(parent_length_km) * u.km,
            daughter=float(daughter_length_km) * u.km,
        )

        # aperture objects / area
        A_cm2 = self._aperture_area_cm2(aperture, delta_au=float(delta_au))
        ap_sbpy = self._sbpy_aperture(aperture, delta_au=float(delta_au))

        # fraction in aperture
        N_in = haser.total_number(ap_sbpy)
        N_tot = haser.total_number(float(N_total_coma_km) * u.km)
        ratio = N_in / N_tot
        if hasattr(ratio, "to_value"):
            frac = float(ratio.to_value(u.dimensionless_unscaled))
        elif hasattr(ratio, "value"):
            frac = float(ratio.value)
        else:
            frac = float(ratio)

        if not np.isfinite(frac) or frac <= 0:
            raise ValueError("Haser aperture fraction is invalid (<=0 or non-finite). Check aperture/delta.")

        # daughter lifetime
        daughter_lifetime_s = (float(daughter_length_km) * u.km) / (float(v_outflow_km_s) * u.km / u.s)

        # helper: logN chain extraction
        def get_logN_chain_for_iso(iso: str) -> np.ndarray:
            if self.samples_pruned is None or self.param_keys is None:
                raise ValueError("No MCMC samples available (samples_pruned/param_keys missing). Fit first or set use_samples=False.")
            pkeys = list(self.param_keys)
            if len(iso_list) == 1:
                key = "logN"
            else:
                key = f"logN_{iso}"
            if key not in pkeys:
                raise KeyError(f"Missing parameter '{key}' in chains. param_keys={self.param_keys}")
            j = pkeys.index(key)
            return np.asarray(self.samples_pruned[:, j], float)


        def compute_from_logN(logN_vals: np.ndarray) -> Tuple[float, float]:
            """
            logN_vals: log10(column density) in molecules / cm^2

            Steps:
            Ncol  = 10^logN  [molecules cm^-2]
            N_ap  = Ncol * A_cm2        [molecules]
            N_tot = N_ap / frac         [molecules]
            Q     = N_tot / tau         [molecules s^-1]
            """
            # column density in molecules / cm^2
            Ncol = (10.0 ** np.asarray(logN_vals, float)) / (u.cm**2)

            # molecules in the aperture
            N_ap = Ncol * A_cm2

            # total molecules in coma (Haser fraction correction)
            N_tot = N_ap / frac
  
            # production rate
            tau = daughter_lifetime_s.to(u.s)
            Q = (N_tot / tau).to(1 / u.s)  # molecules/s (dimensionally 1/s)

            logQ = np.log10(Q.value)

            p16, p50, p84 = np.percentile(logQ, [16, 50, 84])
            err = 0.5 * ((p84 - p50) + (p50 - p16))
            return float(p50), float(err)


        if len(iso_list) == 1:
            iso = iso_list[0]
            if use_samples:
                logN_chain = get_logN_chain_for_iso(iso)
            else:
                logN_chain = np.array([float(self.logN)], dtype=float)
            q50, qerr = compute_from_logN(logN_chain)
            self.q = q50
            self.q_err = qerr
            return q50, qerr

        # multi-iso
        out: Dict[str, Tuple[float, float]] = {}
        for iso in iso_list:
            if use_samples:
                logN_chain = get_logN_chain_for_iso(iso)
            else:
                if self.logN_by_iso is None or iso not in self.logN_by_iso:
                    raise ValueError(f"Missing logN_by_iso[{iso}] and use_samples=False.")
                logN_chain = np.array([float(self.logN_by_iso[iso])], dtype=float)

            q50, qerr = compute_from_logN(logN_chain)
            out[iso] = (q50, qerr)

        self.q = {k: v[0] for k, v in out.items()}
        self.q_err = {k: v[1] for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    # NEW: add slit-loss systematic error to existing q_err
    # ------------------------------------------------------------------
    def add_slit_loss_error(
        self,
        *,
        lambda_nm: float,
        aperture: dict,
        correct: str = "both",  # "q", "logN", or "both"
        eps_min_arcsec_500: float = 0.7,
        eps_max_arcsec_500: float = 1.2,
        zmin_deg: float = 45.0,
        zmax_deg: float = 45.0,
        n_points: int = 2000,
    ) -> Union[float, Dict[str, float]]:
        """Add seeing/slit-loss systematic uncertainty to ``q_err``, ``logN_err``, or both.

            This method can be called after :meth:`compute_production_rate` (for ``correct="q"``
            or ``correct="both"``) or after fitting (for ``correct="logN"``). Each quantity
            tracks its own correction state, so they can be corrected independently or together.

            The final error is teh quadrature sum of the original error and a slit-loss error. 
            For details check the paper


            :param lambda_nm: Wavelength in nm used for seeing scaling.
            :type lambda_nm: float
            :param aperture: Aperture geometry definition dictionary. Same format as
                :meth:`compute_production_rate`.
            :type aperture: dict
            :param correct: Which quantity to correct. One of ``"q"`` (default),
                ``"logN"``, or ``"both"``. Note that ``"q"`` requires ``self.q`` and
                ``self.q_err`` to be set; ``"logN"`` requires ``self.logN`` and
                ``self.logN_err`` to be set.
            :type correct: str
            :param eps_min_arcsec_500: Minimum seeing FWHM at 500 nm and zenith, in arcsec.
                Default is ``0.7``.
            :type eps_min_arcsec_500: float
            :param eps_max_arcsec_500: Maximum seeing FWHM at 500 nm and zenith, in arcsec.
                Default is ``1.2``.
            :type eps_max_arcsec_500: float
            :param zmin_deg: Minimum (best) zenith angle during observations, in degrees.
                Default is ``45.0``.
            :type zmin_deg: float
            :param zmax_deg: Maximum (worst) zenith angle during observations, in degrees.
                Default is ``45.0``.
            :type zmax_deg: float
            :param n_points: Number of wavelength points used to sample the narrow window
                around ``lambda_nm``. Must be >= 2. Default is ``2000``.
            :type n_points: int
            :raises ValueError: If ``correct`` is not one of ``"q"``, ``"logN"``, ``"both"``;
                or if the required attributes (``self.q``, ``self.q_err``, ``self.logN``,
                ``self.logN_err``) are not set for the requested correction.
            :raises KeyError: If an isotopologue key is missing from ``self.q``,
                ``self.q_err``, ``self.logN_by_iso``, or ``self.logN_err_by_iso``.

            Side effects:
                - Updates ``self.q_err`` / ``self.q_err_by_iso`` and sets
                ``self.q_seeing_corrected = True`` when correcting ``q``.
                - Updates ``self.logN_err`` / ``self.logN_err_by_iso`` and sets
                ``self.logN_seeing_corrected = True`` when correcting ``logN``.
                - If a quantity was already corrected, prints a warning and skips it
                without raising an error.
            """
        if correct not in ("q", "logN", "both"):
            raise ValueError("correct must be 'q', 'logN', or 'both'.")

        do_q    = correct in ("q", "both")
        do_logN = correct in ("logN", "both")

        if do_q and (self.q is None or self.q_err is None):
            raise ValueError("self.q and self.q_err must be set before correcting q. Recomended to fit first or set it manually to 0")
        if do_logN and (self.logN is None or self.logN_err is None) and (self.logN_by_iso is None or self.logN_err_by_iso is None):
            raise ValueError("self.logN and self.logN_err must be set before correcting logN.")
        if do_q and self.q_seeing_corrected:
            print('q was already corrected, skipping.')
            do_q = False
        if do_logN and self.logN_seeing_corrected:
            print('logN was already corrected, skipping.')
            do_logN = False

        if not do_q and not do_logN:
            return

        _slitloss_kwargs = dict(
            lambda_nm=float(lambda_nm),
            aperture=aperture,
            eps_min_arcsec_500=float(eps_min_arcsec_500),
            eps_max_arcsec_500=float(eps_max_arcsec_500),
            zmin_deg=float(zmin_deg),
            zmax_deg=float(zmax_deg),
            n_points=int(n_points),
        )

        iso_list = self._iso_list()

        if len(iso_list) == 1:
            if do_q:
                self.q_err = helper.add_slit_loss_error_scalar(
                    float(self.q), float(self.q_err), **_slitloss_kwargs
                )
                self.q_seeing_corrected = True
            if do_logN:
                self.logN_err = np.array([
                    helper.add_slit_loss_error_scalar(
                        float(self.logN), float(self.logN_err[0]), **_slitloss_kwargs
                    ),
                    helper.add_slit_loss_error_scalar(
                        float(self.logN), float(self.logN_err[1]), **_slitloss_kwargs
                    ),
                ])
                self.logN_seeing_corrected = True
            return 

        # multi-iso
        if not isinstance(self.q, dict) or not isinstance(self.q_err, dict):
            raise ValueError("For multi-isotopologue models, self.q and self.q_err must be dicts keyed by iso.")

        new_errs: Dict[str, float] = {}
        for iso in iso_list:
            if do_q:
                if iso not in self.q or iso not in self.q_err:
                    raise KeyError(f"Missing q/q_err for iso='{iso}'.")
                new_errs[iso] = helper.add_slit_loss_error_scalar(
                    float(self.q[iso]), float(self.q_err[iso]), **_slitloss_kwargs
                )
            if do_logN:
                self.logN_err_by_iso[iso] = np.array([
                    helper.add_slit_loss_error_scalar(
                        float(self.logN_by_iso[iso]), float(self.logN_err_by_iso[iso][0]), **_slitloss_kwargs
                    ),
                    helper.add_slit_loss_error_scalar(
                        float(self.logN_by_iso[iso]), float(self.logN_err_by_iso[iso][1]), **_slitloss_kwargs
                    ),
                ])

        if do_q:
            self.q_err = new_errs
            self.q_seeing_corrected = True
        if do_logN:
            self.logN_seeing_corrected = True
        return 

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
        km_per_arcsec = self._arcsec_to_km(delta_au=float(delta_au))
        ap_type = aperture.get("type", "circular").lower()

        if ap_type == "circular":
            rho_ap = (float(aperture["radius_arcsec"]) * km_per_arcsec * u.km).to(u.cm)
            return np.pi * rho_ap / 2.0

        elif ap_type == "rectangular":
            W = (float(aperture["width_arcsec"])  * km_per_arcsec * u.km).to(u.cm)
            L = (float(aperture["length_arcsec"]) * km_per_arcsec * u.km).to(u.cm)
            return (
                L * np.arcsinh((W / L).to_value(""))
                + W * np.arcsinh((L / W).to_value(""))
            ) / 2.0

        else:
            raise ValueError(f"Unsupported aperture type '{ap_type}'.")
   
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

        Intended workflow::

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
        use_samples : bool
            If ``True``, propagates full MCMC chains; if ``False``, uses the
            median ``logN`` only.

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
        iso_list = self._iso_list()
        A_cm2 = self._aperture_area_cm2(aperture, delta_au=float(delta_au))
        v = float(v_outflow_km_s) * u.km / u.s

        # Validate G has the right dimension
        try:
            G = G.to(u.cm)
        except u.UnitConversionError:
            raise ValueError(f"G must be convertible to cm, got units '{G.unit}'.")

        def get_logN_chain(iso: str) -> np.ndarray:
            if self.samples_pruned is None or self.param_keys is None:
                raise ValueError(
                    "No MCMC samples available. Fit first or set use_samples=False."
                )
            pkeys = list(self.param_keys)
            key = "logN" if len(iso_list) == 1 else f"logN_{iso}"
            if key not in pkeys:
                raise KeyError(f"Missing parameter '{key}' in chains. param_keys={self.param_keys}")
            return np.asarray(self.samples_pruned[:, pkeys.index(key)], float)

        def compute_from_logN(logN_vals: np.ndarray) -> Tuple[float, float]:
            Ncol = (10.0 ** np.asarray(logN_vals, float)) / u.cm**2
            N_ap = Ncol * A_cm2          # molecules
            Q    = (v * N_ap / G).to(1 / u.s)
            logQ = np.log10(Q.value)
            p16, p50, p84 = np.percentile(logQ, [16, 50, 84])
            return float(p50), float(0.5 * ((p84 - p50) + (p50 - p16)))

        # --- single isotopologue ---
        if len(iso_list) == 1:
            iso = iso_list[0]
            chain = get_logN_chain(iso) if use_samples else np.array([float(self.logN)])
            q50, qerr = compute_from_logN(chain)
            self.q, self.q_err = q50, qerr
            return q50, qerr

        # --- multi-isotopologue ---
        out: Dict[str, Tuple[float, float]] = {}
        for iso in iso_list:
            if use_samples:
                chain = get_logN_chain(iso)
            else:
                if self.logN_by_iso is None or iso not in self.logN_by_iso:
                    raise ValueError(f"Missing logN_by_iso['{iso}'] and use_samples=False.")
                chain = np.array([float(self.logN_by_iso[iso])])
            out[iso] = compute_from_logN(chain)

        self.q     = {k: v[0] for k, v in out.items()}
        self.q_err = {k: v[1] for k, v in out.items()}
        return out

    # ---------------------------------------------------------------------
    # Helper: arcsec -> km conversion at geocentric distance delta_au.
    # Add this if it doesn't already exist in your class.
    # ---------------------------------------------------------------------
    @staticmethod
    def _arcsec_to_km(*, delta_au: float) -> float:
        """Convert 1 arcsec at geocentric distance delta_au to km at the comet."""
        AU_KM = 1.495978707e8
        ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
        return float(delta_au) * AU_KM * ARCSEC_TO_RAD
    
    # ------------------------------------------------------------------
    # Serialization: include q/q_err
    # ------------------------------------------------------------------
    def save(self, filename: str) -> None:
        """Serialize model state to a pickle file.

        The saved state includes constructor kwargs, MCMC products, and derived
        production-rate fields ``q`` and ``q_err``. If the model uses a custom
        callable LSF (``lsf_method == "Given"``), that callable is not serialized.

        :param filename: Output file path.
        :type filename: str
        """
        had_given_lsf = (self.lsf_method == "Given")
        init_kwargs = dict(
            data=self.data,
            window=self.window,
            pumping=self.pumping,
            isotopologues=self.isotopologues,
            systems=self.systems,
            linelists=self.linelists,
            line_path=self.line_path,
            lsf=None,
            lsf_method=self.lsf_method if not had_given_lsf else "Gauss",
            A_min=self.A_min,
            a=self.a,
            threads=self.threads,
            name=self.name,
            sigma=self.sigma,
            sigma1=self.sigma1,
            sigma2=self.sigma2,
            sigma_G=self.sigma_G,
            fwhm_L=self.fwhm_L,
            ratio=self.ratio,
            pumping_min_wave=self.pumping_min_wave,
            pumping_max_wave=self.pumping_max_wave,
            logN=self.logN,
            logN_by_iso=self.logN_by_iso,
            logQ=self.logQ,
            logQ_by_iso=self.logQ_by_iso,
            T=self.T,
            v_kms=self.v_kms,
            dlam=self.dlam,
            include_rotations=self.include_rotations,
        )

        mcmc_result = dict(
            param_keys=self.param_keys,
            median_params=self.median_params,
            up_errors_params=self.up_errors_params,
            low_errors_params=self.low_errors_params,
            samples_pruned=self.samples_pruned,
            lnprob_pruned=self.lnprob_pruned,
            model_wave=self.model_wave,
            median_model=self.median_model,
            best_model=self.best_model,
            model_p16=self.model_p16,
            model_p84=self.model_p84,
            model_by_iso=self.model_by_iso,
        )

        # NEW: persist q/q_err and correction flags
        derived = dict(
            q=self.q,
            q_err=self.q_err,
            q_seeing_corrected=self.q_seeing_corrected,
            logN_seeing_corrected=self.logN_seeing_corrected,
            logN_err=self.logN_err,
            logN_err_by_iso=self.logN_err_by_iso,
        )

        state = {
            "class": "FluorescenceModel",
            "version": 4,
            "init_kwargs": init_kwargs,
            "mcmc_result": mcmc_result,
            "derived": derived,
            "had_given_lsf": had_given_lsf,
            "linelists": self.linelists,  # explicit backup in case init_kwargs is from older format
        }

        with open(filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str) -> "FluorescenceModel":
        """Load a serialized :class:`FluorescenceModel` from disk.

        :param filename: Input pickle file path created by :meth:`save`.
        :type filename: str
        :returns: Reconstructed fluorescence model instance.
        :rtype: FluorescenceModel
        :raises ValueError: If the file does not contain a compatible
            ``FluorescenceModel`` state version.

        Side effects:
        Restores fit products and derived ``q``/``q_err`` when present. If the
        original model used a custom callable LSF, a warning is printed because
        that callable is not serialized.
        """
        with open(filename, "rb") as f:
            state = pickle.load(f)

        if state.get("class") != "FluorescenceModel":
            raise ValueError("File does not contain a FluorescenceModel state.")
        version = state.get("version", 1)
        if version not in (1, 2, 3, 4):
            raise ValueError("Unsupported FluorescenceModel state version.")

        init_kwargs = state["init_kwargs"]

        # If linelists was saved as a top-level key (version 4+), use it to
        # patch init_kwargs in case it was missing or None there (e.g. old pickles).
        if state.get("linelists") is not None and init_kwargs.get("linelists") is None:
            init_kwargs["linelists"] = state["linelists"]

        mcmc_result = state.get("mcmc_result") or {}
        derived = state.get("derived") or {}
        had_given_lsf = state.get("had_given_lsf", False)

        obj = cls(**init_kwargs)

        if any(v is not None for v in mcmc_result.values()):
            obj._update_from_result(
                mcmc_result,
                used_lsf=None,
                used_lsf_method=init_kwargs.get("lsf_method"),
            )

        # NEW: restore q/q_err, correction flags, and logN errors
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
    

    