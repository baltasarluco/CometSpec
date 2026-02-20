from __future__ import annotations

"""
High-level fluorescence model wrapper (UPDATED for isotopologues + systems)
+ Production rate + slit-loss systematic error support.

Key additions vs your last FluorescenceModel
--------------------------------------------
1) self.q and self.q_err
   - single-iso: float or None
   - multi-iso: dict[str, float] or None (keyed by isotopologue)

2) New public methods:
   - compute_production_rate(...)
   - add_slit_loss_error(...)

3) Serialization:
   - save/load now persist q and q_err (safe even if None)

4) Safety:
   - if fitted logN changed in _update_from_result => reset q/q_err to None
   - if update_model changes logN/logN_by_iso/isotopologues => reset q/q_err to None
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
    def __init__(
        self,
        *,
        data: Optional[Any] = None,
        window: Optional[Tuple[float, float]] = (3850.0, 3900.0),
        pumping: Any = None,
        isotopologues: Union[str, Sequence[str]] = "12C14N",
        systems: Union[str, Sequence[str], None] = None,
        linelists: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        line_path: Optional[str] = None,
        lsf: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lsf_method: Optional[str] = "Gauss",
        A_min: float = 1e4,
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
        logQ: Optional[float] = -3.0,
        T: Optional[float] = 300.0,
        v_kms: Optional[float] = 0.0,
        dlam: Optional[float] = 0.0,
        wave_col: str = "WAVE",
        flux_col: str = "FLUX_STACK",
        error_col: str = "ERR_STACK",
        continuum_col: str = "CONTINUUM",
        omega: float = np.pi * (0.5 * np.pi / (180.0 * 3600.0)) ** 2,
        seeing_corrected: bool = False,
        include_rotations: bool = True,
        pumping_v_kms: float = 0.0,
        pumping_dlam_A: float = 0.0,
        model_wave: Optional[np.ndarray] = None,

    ) -> None:
        if pumping is None:
            raise ValueError("Pumping spectrum must be provided to FluorescenceModel.")

        self.data = data
        self.wave_col = wave_col
        self.flux_col = flux_col
        self.error_col = error_col
        self.continuum_col = continuum_col
        self.seeing_corrected = seeing_corrected
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
        self.T = T
        self.v_kms = v_kms
        self.dlam = dlam

        # NEW: derived quantities (production rate and its uncertainty)
        self.q: Optional[Union[float, Dict[str, float]]] = None
        self.q_err: Optional[Union[float, Dict[str, float]]] = None

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
        return [self.isotopologues] if isinstance(self.isotopologues, str) else list(self.isotopologues)

    @staticmethod
    def _km_per_arcsec(delta_au: float) -> float:
        delta_km = (float(delta_au) * u.au).to(u.km).value
        return float(delta_km * np.tan(1.0 * u.arcsec.to(u.rad)))

    @classmethod
    def _aperture_area_cm2(cls, aperture: dict, *, delta_au: float) -> u.Quantity:
        """
        Convert aperture definition into collecting area on the sky in cm^2.
        aperture:
          {"type":"circular", "radius_arcsec": R}
          {"type":"rectangular", "width_arcsec": W, "length_arcsec": L}
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
        """
        Build sbpy aperture object in km units for Haser.total_number().
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
        linelists: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
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
    ) -> Dict[str, Any]:
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

        # ------------------------------
        # ✅ NEW: fallbacks for parameters NOT being fit
        # If priors does not include them, modeling.mcmc_fitting must use these,
        # instead of magic defaults (-99/300/0/0).
        # ------------------------------
        init_logQ = float(self.logQ) if self.logQ is not None else -3.0
        init_T = float(self.T) if self.T is not None else 300.0
        init_v_kms = float(self.v_kms) if self.v_kms is not None else 0.0
        init_dlam = float(self.dlam) if self.dlam is not None else 0.0

        # ------------------------------
        # ✅ NEW: pumping shift (J_nu sampling) consistency
        # These control where J_nu is evaluated (Fraunhofer structure matters!)
        # They are separate from emission shift v_kms/dlam which can still be fit.
        # ------------------------------
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
            init_logQ=init_logQ,
            init_T=init_T,
            init_v_kms=init_v_kms,
            init_dlam=init_dlam,

            fig_file=fig_file,
            wave_col=self.wave_col,
            flux_col=self.flux_col,
            error_col=self.error_col,
            continuum_col=self.continuum_col,
            omega=self.omega,
            verbose=verbose,
            pruning=pruning,
            include_rotations=self.include_rotations,
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
        linelists: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
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
    ) -> None:
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
        if self.pumping is None:
            raise ValueError("Pumping spectrum is required.")
        if self.window is None:
            raise ValueError("window is required.")

        iso_list = self._iso_list()

        # 1) transitions
        if self.linelists is None:
            line_paths = None
            if self.line_path is not None:
                line_paths = {iso_list[0]: self.line_path}

            trans_by_iso = modeling.load_default_cn_transitions(
                isotopologues=iso_list,
                systems=self.systems,
                A_min=self.A_min,
                lambda_min_A=float(self.pumping_min_wave),
                lambda_max_A=float(self.pumping_max_wave),
                use_omega_labels=False,
                line_paths=line_paths,
            )
        else:
            if isinstance(self.linelists, pd.DataFrame):
                if len(iso_list) != 1:
                    raise ValueError("If linelists is a single DataFrame, isotopologues must be a single iso.")
                trans_by_iso = {iso_list[0]: self.linelists}
            else:
                trans_by_iso = {iso: self.linelists[iso] for iso in iso_list}

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
                coll_scaf = modeling.precompute_cn_collision_scaffold_fast(lines_out, idx_to_level)
            else:
                coll_scaf = dict(iu=np.array([], int), il=np.array([], int),
                                gu=np.array([]), gl=np.array([]), dE=np.array([]))

            M = M_rad.copy()

            # ✅ Collisions only if logQ/T are defined and include_rotations
            if self.logQ is not None and self.T is not None and self.include_rotations:
                Q_lin = 10.0 ** float(self.logQ)
                if np.isfinite(Q_lin) and Q_lin > 0.0:
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
    # Internal: apply MCMC result (ONLY ADD: q/q_err reset if logN changed)
    # ------------------------------------------------------------------
    def _update_from_result(
        self,
        result: Dict[str, Any],
        *,
        used_lsf: Optional[Callable[[np.ndarray], np.ndarray]],
        used_lsf_method: Optional[str],
    ) -> None:
        self.param_keys = tuple(result.get("param_keys", ()))
        self.median_params = dict(result.get("median_params", {}))
        self.up_errors_params = dict(result.get("up_errors_params", {}))
        self.low_errors_params = dict(result.get("low_errors_params", {}))

        self.samples_pruned = result.get("samples_pruned")
        self.lnprob_pruned = result.get("lnprob_pruned")

        for name in ("logN", "logQ", "T", "v_kms", "dlam"):
            if name in self.median_params:
                setattr(self, name, float(self.median_params[name]))

        iso_list = self._iso_list()
        any_isoN = any((f"logN_{iso}" in self.median_params) for iso in iso_list)
        if any_isoN:
            self.logN_by_iso = {}
            for iso in iso_list:
                key = f"logN_{iso}"
                if key in self.median_params:
                    self.logN_by_iso[iso] = float(self.median_params[key])

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
            self.q_err = None
        
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
                logQ=params_per_iso.get("logQ", self.logQ),
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
        """
        Compute log10(Q) and its (symmetrized) uncertainty from the fitted logN chains,
        using a Haser model and the chosen aperture.

        Parameters
        ----------
        delta_au : float
            Geocentric distance [AU] (needed to convert arcsec -> km).
        aperture : dict
            {"type":"circular","radius_arcsec":R} or
            {"type":"rectangular","width_arcsec":W,"length_arcsec":L}
        parent_length_km, daughter_length_km : float
            Haser scale lengths (km).
        v_outflow_km_s : float
            Outflow speed (km/s).
        use_samples : bool
            If True uses self.samples_pruned chains; if False uses median logN(s) only.
        N_total_coma_km : float
            Radius (km) to approximate total coma for the Haser "total" number.

        Returns
        -------
        single iso: (logQ50, logQerr)
        multi iso: dict[iso] = (logQ50, logQerr)

        Also sets self.q and self.q_err accordingly.
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
        eps_min_arcsec_500: float = 0.7,
        eps_max_arcsec_500: float = 1.2,
        zmin_deg: float = 45.0,
        zmax_deg: float = 45.0,
        n_points: int = 2000,
    ) -> Union[float, Dict[str, float]]:
        """
        Inflate q_err by adding a slit-loss systematic based on seeing bounds.

        Requires self.q and self.q_err to already be set (e.g. by compute_production_rate()).

        Returns the updated q_err (float or dict).
        """
        if self.q is None or self.q_err is None:
            raise ValueError("self.q and self.q_err must be set before calling add_slit_loss_error().")
        if self.seeing_corrected:
            print('It was already corrected, there is no need to apply it again.')
            return self.q_err
        iso_list = self._iso_list()

        if len(iso_list) == 1:
            q = float(self.q)  # log10(Q)
            qerr = float(self.q_err)
            new_err = helper.add_slit_loss_error_scalar(
                q,
                qerr,
                lambda_nm=float(lambda_nm),
                aperture=aperture,
                eps_min_arcsec_500=float(eps_min_arcsec_500),
                eps_max_arcsec_500=float(eps_max_arcsec_500),
                zmin_deg=float(zmin_deg),
                zmax_deg=float(zmax_deg),
                n_points=int(n_points),
            )
            self.q_err = new_err
            self.seeing_corrected = True
            return new_err

        # multi-iso dict
        if not isinstance(self.q, dict) or not isinstance(self.q_err, dict):
            raise ValueError("For multi-isotopologue models, self.q and self.q_err must be dicts keyed by iso.")

        new_errs: Dict[str, float] = {}
        for iso in iso_list:
            if iso not in self.q or iso not in self.q_err:
                raise KeyError(f"Missing q/q_err for iso='{iso}'.")
            new_errs[iso] = helper.add_slit_loss_error_scalar(
                float(self.q[iso]),
                float(self.q_err[iso]),
                lambda_nm=float(lambda_nm),
                aperture=aperture,
                eps_min_arcsec_500=float(eps_min_arcsec_500),
                eps_max_arcsec_500=float(eps_max_arcsec_500),
                zmin_deg=float(zmin_deg),
                zmax_deg=float(zmax_deg),
                n_points=int(n_points),
            )

        self.q_err = new_errs
        self.seeing_corrected = True
        return new_errs

    # ------------------------------------------------------------------
    # Serialization: include q/q_err
    # ------------------------------------------------------------------
    def save(self, filename: str) -> None:
        had_given_lsf = (self.lsf_method == "Given")

        init_kwargs = dict(
            data=self.data,
            window=self.window,
            pumping=self.pumping,
            isotopologues=self.isotopologues,
            systems=self.systems,
            linelists=None,  # do not serialize
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
            T=self.T,
            v_kms=self.v_kms,
            dlam=self.dlam,
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

        # NEW: persist q/q_err
        derived = dict(q=self.q, q_err=self.q_err)

        state = {
            "class": "FluorescenceModel",
            "version": 3,
            "init_kwargs": init_kwargs,
            "mcmc_result": mcmc_result,
            "derived": derived,
            "had_given_lsf": had_given_lsf,
        }

        with open(filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str) -> "FluorescenceModel":
        with open(filename, "rb") as f:
            state = pickle.load(f)

        if state.get("class") != "FluorescenceModel":
            raise ValueError("File does not contain a FluorescenceModel state.")
        version = state.get("version", 1)
        if version not in (1, 2, 3):
            raise ValueError("Unsupported FluorescenceModel state version.")

        init_kwargs = state["init_kwargs"]
        mcmc_result = state.get("mcmc_result") or {}
        derived = state.get("derived") or {}
        had_given_lsf = state.get("had_given_lsf", False)

        obj = cls(**init_kwargs)

        if any(mcmc_result.values()):
            obj._update_from_result(
                mcmc_result,
                used_lsf=None,
                used_lsf_method=init_kwargs.get("lsf_method"),
            )

        # NEW: restore q/q_err
        obj.q = derived.get("q", None)
        obj.q_err = derived.get("q_err", None)

        if had_given_lsf:
            print(
                "Warning: original model used a custom 'Given' LSF which "
                "was not serialized. Call `obj.update_model(lsf=...)` to restore it."
            )
        return obj