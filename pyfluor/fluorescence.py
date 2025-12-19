from __future__ import annotations

"""
High-level fluorescence model wrapper.
"""

from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
import pickle


from . import helper, modeling


class FluorescenceModel:
    """
    Container for a single CN fluorescence model + its MCMC fit.

    Notes
    -----
    - Without a user-provided LSF, a Gaussian LSF is used by default,
      with sigma=0.01 AA, and all other LSF params are kept None.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        data: Optional[Any] = None,
        window: Optional[Tuple[float, float]] = (3850.0, 3900.0),
        pumping: Any = None,
        line_path: Optional[str] = helper.get_default_mol_linelist_path(),
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
        logQ: Optional[float] = -3.0,
        T: Optional[float] = 300.0,
        v_kms: Optional[float] = 0.0,
        dlam: Optional[float] = 0.0,
    ) -> None:
        if pumping is None:
            raise ValueError("Pumping spectrum must be provided to FluorescenceModel.")

        # Observed data + config
        self.data = data
        self.window = window
        self.pumping = pumping
        self.line_path = line_path
        self.pumping_min_wave = pumping_min_wave
        self.pumping_max_wave = pumping_max_wave

        self.A_min = float(A_min)
        self.name = name or "CN fluorescence"
        self.a = a
        self.threads = threads

        # --- Physical / excitation parameters ---
        self.logN = logN
        self.logQ = logQ
        self.T = T
        self.v_kms = v_kms
        self.dlam = dlam

        # --- LSF setup ---
        if lsf is not None:
            # Fixed external LSF
            self.lsf = lsf
            self.lsf_method = "Given"
            # all parametric LSF params irrelevant
            self.sigma = None
            self.sigma1 = None
            self.sigma2 = None
            self.sigma_G = None
            self.fwhm_L = None
            self.ratio = None

        else:
            # Parametric LSF
            self.lsf_method = lsf_method

            if lsf_method == "Gauss":
                # default sigma if not provided
                self.sigma = 0.01 if sigma is None else float(sigma)
                dic_params = {"sigma": self.sigma}
                self.lsf = modeling.make_lsf(dic_params, lsf_method)

                self.sigma1 = None
                self.sigma2 = None
                self.ratio = None
                self.sigma_G = None
                self.fwhm_L = None

            elif lsf_method == "2Gauss":
                if sigma1 is None or sigma2 is None or ratio is None:
                    raise ValueError(
                        "sigma1, sigma2, and ratio must be provided for '2Gauss' LSF method."
                    )
                self.sigma1 = float(sigma1)
                self.sigma2 = float(sigma2)
                self.ratio = float(ratio)

                dic_params = {
                    "sigma1": self.sigma1,
                    "sigma2": self.sigma2,
                    "ratio": self.ratio,
                }
                self.lsf = modeling.make_lsf(dic_params, lsf_method)

                self.sigma = None
                self.sigma_G = None
                self.fwhm_L = None

            elif lsf_method == "Gauss_Lorentz":
                if sigma_G is None or fwhm_L is None or ratio is None:
                    raise ValueError(
                        "sigma_G, fwhm_L, and ratio must be provided for 'Gauss_Lorentz' LSF method."
                    )
                self.sigma_G = float(sigma_G)
                self.fwhm_L = float(fwhm_L)
                self.ratio = float(ratio)

                dic_params = {
                    "sigma_G": self.sigma_G,
                    "fwhm_L": self.fwhm_L,
                    "ratio": self.ratio,
                }
                self.lsf = modeling.make_lsf(dic_params, lsf_method)

                self.sigma = None
                self.sigma1 = None
                self.sigma2 = None
            elif lsf_method == "Lorentz":
                if fwhm_L is None:
                    raise ValueError(
                        "fwhm_L must be provided for 'Lorentz' LSF method."
                    )
                self.fwhm_L = float(fwhm_L)

                dic_params = {
                    "fwhm_L": self.fwhm_L,
                }
                self.lsf = modeling.make_lsf(dic_params, lsf_method)

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
        self.v_up_error: float = None
        self.v_low_error: float = None
        self.v_samples: Optional[np.ndarray] = None


        self.lines = None
        self.M = None
        self.idx_to_level = None
        self.n = None
        self.g_ph = None
        self.g_en = None
        self.g_ph_sum = None
        self.g_en_sum = None

        self.model_wave: Optional[np.ndarray] = None
        self.median_model: Optional[np.ndarray] = None
        self.best_model: Optional[np.ndarray] = None
        self.model_p16: Optional[np.ndarray] = None
        self.model_p84: Optional[np.ndarray] = None

        # Build initial synthetic model
        self._synthesize_model()

    # ------------------------------------------------------------------
    # Public: run MCMC
    # ------------------------------------------------------------------
    def fit_mcmc(
        self,
        data: Optional[Any] = None,
        window: Optional[Tuple[float, float]] = None,
        *,
        pumping: Any = None,
        line_path: Optional[str] = None,
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
    ) -> Dict[str, Any]:
        # allow overrides
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("No data attached to this FluorescenceModel.")

        if window is None:
            if self.window is None:
                raise ValueError(
                    "window must be provided (argument or FluorescenceModel.window)."
                )
            window = self.window
        else:
            self.window = window

        if pumping is None:
            pumping = self.pumping
        else:
            self.pumping = pumping
        if pumping is None:
            raise ValueError(
                "pumping must be provided (argument or FluorescenceModel.pumping)."
            )

        if line_path is None:
            line_path = self.line_path
        else:
            self.line_path = line_path

        if priors is None:
            priors = self.priors or {
                "logN": (9.0, 15.0),
                "logQ": (-5.0, 0.0),
                "T": (10.0, 1000.0),}
            
            print(
                "No priors provided, using default priors: logN, logQ, T, v_kms."
            )
        self.priors = priors

        if lsf is not None:
            print("An lsf was given, so it will be assumed as fixed.")
        # if None, we'll use the instance LSF / lsf_method

        if lsf_method is None:
            print(
                "No lsf_method was given, so the one stored in the instance will be used."
            )
            lsf_method = self.lsf_method

        if A_min is None:
            A_min = self.A_min
        else:
            self.A_min = A_min

        if a is None:
            a = self.a
        else:
            self.a = a

        if threads is None:
            threads = self.threads
        else:
            self.threads = threads

        print('Lets go with the full fit')
        # Delegate to underlying implementation
        result = modeling.mcmc_fitting(
            self.data,
            window,
            pumping=pumping,
            line_path=line_path,
            nwalkers=nwalkers,
            nsteps=nsteps,
            priors=priors,
            lsf=lsf,
            lsf_method=lsf_method,
            make_plots=make_plots,
            progress=progress,
            A_min=A_min,
            a=a,
            threads=threads,
            fig_file=fig_file,
        )

        # Update instance from fit result (this will also resynthesize)
        self._update_from_result(
            result,
            used_lsf=lsf,
            used_lsf_method=lsf_method,
        )

        return result

    # ------------------------------------------------------------------
    # Public: update params & resynthesize
    # ------------------------------------------------------------------
    def update_model(
        self,
        *,
        logN: Optional[float] = None,
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
    ) -> None:
        """
        Update one or more *model-defining* parameters and immediately
        recompute the synthetic spectrum.

        Call like:
            model.update_model(logN=12.0, T=400.0)
        """

        # Basic fields
        if data is not None:
            self.data = data
        if pumping is not None:
            self.pumping = pumping
        if window is not None:
            self.window = window

        if logN is not None:
            self.logN = float(logN)
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

        # LSF: either a fully given kernel or parametric
        if lsf is not None:
            # external fixed LSF overrides everything
            self.lsf = lsf
            self.lsf_method = "Given"
            for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
                setattr(self, name, None)

        elif lsf_method is not None:
            # switching LSF model
            self.lsf_method = lsf_method
            if lsf_method == "Gauss":
                self.sigma = float(sigma if sigma is not None else 0.01)
                self.sigma1 = None
                self.sigma2 = None
                self.sigma_G = None
                self.fwhm_L = None
                self.ratio = None
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")

            elif lsf_method == "2Gauss":
                if sigma1 is None or sigma2 is None or ratio is None:
                    raise ValueError(
                        "sigma1, sigma2, and ratio must be provided for '2Gauss'."
                    )
                self.sigma1 = float(sigma1)
                self.sigma2 = float(sigma2)
                self.ratio = float(ratio)
                self.sigma = None
                self.sigma_G = None
                self.fwhm_L = None
                self.lsf = modeling.make_lsf(
                    {
                        "sigma1": self.sigma1,
                        "sigma2": self.sigma2,
                        "ratio": self.ratio,
                    },
                    "2Gauss",
                )

            elif lsf_method == "Gauss_Lorentz":
                if sigma_G is None or fwhm_L is None or ratio is None:
                    raise ValueError(
                        "sigma_G, fwhm_L, and ratio must be provided for 'Gauss_Lorentz'."
                    )
                self.sigma_G = float(sigma_G)
                self.fwhm_L = float(fwhm_L)
                self.ratio = float(ratio)
                self.sigma = None
                self.sigma1 = None
                self.sigma2 = None
                self.lsf = modeling.make_lsf(
                    {
                        "sigma_G": self.sigma_G,
                        "fwhm_L": self.fwhm_L,
                        "ratio": self.ratio,
                    },
                    "Gauss_Lorentz",
                )
            elif lsf_method == "Lorentz":
                if fwhm_L is None:
                    raise ValueError(
                        "fwhm_L must be provided for 'Lorentz'."
                    )
                self.fwhm_L = float(fwhm_L)
                self.sigma = None
                self.sigma1 = None
                self.sigma2 = None
                self.sigma_G = None
                self.ratio = None
                self.lsf = modeling.make_lsf(
                    {
                        "fwhm_L": self.fwhm_L,
                    },
                    "Lorentz",
                )
            else:
                raise ValueError(f"Unsupported lsf_method: {lsf_method}")

        # If only individual sigma* / ratio changed for same lsf_method:
        elif self.lsf_method == "Gauss" and sigma is not None:
            self.sigma = float(sigma)
            self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")
            self.sigma1 = None
            self.sigma2 = None
            self.sigma_G = None
            self.fwhm_L = None
            self.ratio = None

        elif self.lsf_method == "2Gauss" and any(
            v is not None for v in (sigma1, sigma2, ratio)
        ):
            self.sigma1 = float(sigma1) if sigma1 is not None else self.sigma1
            self.sigma2 = float(sigma2) if sigma2 is not None else self.sigma2
            self.ratio = float(ratio) if ratio is not None else self.ratio
            self.lsf = modeling.make_lsf(
                {
                    "sigma1": self.sigma1,
                    "sigma2": self.sigma2,
                    "ratio": self.ratio,
                },
                "2Gauss",
            )
            self.sigma = None
            self.sigma_G = None
            self.fwhm_L = None

        elif self.lsf_method == "Gauss_Lorentz" and any(
            v is not None for v in (sigma_G, fwhm_L, ratio)
        ):
            self.sigma_G = float(sigma_G) if sigma_G is not None else self.sigma_G
            self.fwhm_L = float(fwhm_L) if fwhm_L is not None else self.fwhm_L
            self.ratio = float(ratio) if ratio is not None else self.ratio
            self.lsf = modeling.make_lsf(
                {
                    "sigma_G": self.sigma_G,
                    "fwhm_L": self.fwhm_L,
                    "ratio": self.ratio,
                },
                "Gauss_Lorentz",
            )
            self.sigma = None
            self.sigma1 = None
            self.sigma2 = None
        elif self.lsf_method == "Lorentz" and fwhm_L is not None:
            self.fwhm_L = float(fwhm_L)
            self.lsf = modeling.make_lsf(
                {
                    "fwhm_L": self.fwhm_L,
                },
                "Lorentz",
            )
            self.sigma = None
            self.sigma1 = None
            self.sigma2 = None
            self.sigma_G = None
            self.ratio = None
        else:
            #update the lsf to the current self values
            if self.lsf_method == "Gauss":
                if self.sigma is None:
                    self.sigma = 0.01
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")
                self.sigma1 = None
                self.sigma2 = None
                self.sigma_G = None
            elif self.lsf_method == "2Gauss":
                self.lsf = modeling.make_lsf(
                    {
                        "sigma1": self.sigma1,
                        "sigma2": self.sigma2,
                        "ratio": self.ratio,
                    },
                    "2Gauss",
                )
                self.sigma = None
                self.sigma_G = None
                self.fwhm_L = None
            elif self.lsf_method == "Gauss_Lorentz":
                self.lsf = modeling.make_lsf(
                    {
                        "sigma_G": self.sigma_G,
                        "fwhm_L": self.fwhm_L,
                        "ratio": self.ratio,
                    },
                    "Gauss_Lorentz",
                )
                self.sigma = None
                self.sigma1 = None
                self.sigma2 = None
            
        # Finally, recompute everything
        self._synthesize_model()

    # ------------------------------------------------------------------
    # Internal: rebuild model from current parameters
    # ------------------------------------------------------------------
    def _synthesize_model(self) -> None:
        """
        Build rate matrix, level populations, g-factors, and synthetic spectrum
        using the current attributes.
        """
        if self.pumping is None:
            raise ValueError("Pumping spectrum is required to synthesize the model.")
        if self.line_path is None:
            self.line_path = helper.get_default_cn_linelist_path()


        df_all = modeling.load_cn_linelist(self.line_path)

        lines_brook = modeling.filter_AX_BX(
            df_all,
            lambda_min_A=self.pumping_min_wave,
            lambda_max_A=self.pumping_max_wave,
            A_min=self.A_min,
        )

        lines_theta = modeling.attach_pumping_and_labels(
            lines_brook,
            self.pumping,
            use_omega_labels=False,
            lsf_for_Jnu=None, 
        )

        M_rad_theta, idx_to_level_theta, lines_out_theta = (
            modeling.build_rate_matrix_nbar(
                lines_theta,
                include_stim_emission=True,
                verbose=False,
            )
        )
        
        coll_scaf = modeling.precompute_cn_collision_scaffold(lines_out_theta, idx_to_level_theta)

        self.lines = lines_out_theta

        M = M_rad_theta.copy()

        # collisions
        if self.logQ is not None and self.T is not None:
            M = modeling.apply_collisions_inplace(M, coll_scaf, Q=10**self.logQ, T=self.T)

        self.M = M
        self.idx_to_level = idx_to_level_theta

        n = modeling.solve_with_normalization(M, verbose=False)
        self.n = n

        g_ph, g_en, g_ph_sum, g_en_sum = modeling.g_factors(self.lines, self.n)
        self.g_ph = g_ph
        self.g_en = g_en
        self.g_ph_sum = g_ph_sum
        self.g_en_sum = g_en_sum

        # spectrum grid
        if self.model_wave is None:
            wave = np.arange(
                self.window[0],
                self.window[1] + 0.01,
                0.01,
            )
            self.model_wave = wave
        else:
            wave = self.model_wave

        grid, spec = modeling.synth_spectrum_from_lines(
            self.lines,
            g_line_energy=self.g_en,
            lam_min=float(wave.min()),
            lam_max=float(wave.max()),
            lam_col="Wave_vac_AA",
            N_col_cm2=10.0 ** self.logN,
            Omega_sr=np.pi
            * (0.5 * np.pi / (180.0 * 3600.0)) ** 2,
            grid=self.model_wave,
            lsf=self.lsf,
            v_shift_kms=self.v_kms,
            dlam_shift_A=self.dlam,
        )

        self.model_wave = grid
        self.best_model = spec
        # These are only filled by MCMC; keep None here
        self.model_p16 = None
        self.model_p84 = None

    # ------------------------------------------------------------------
    # Internal: apply MCMC result and resynthesize
    # ------------------------------------------------------------------
    def _update_from_result(
        self,
        result: Dict[str, Any],
        *,
        used_lsf: Optional[Callable[[np.ndarray], np.ndarray]],
        used_lsf_method: Optional[str],
    ) -> None:
        """Update parameters and cached attributes from an mcmc_fitting result."""

        # --- Store sampling outputs / metadata ---
        self.param_keys = tuple(result.get("param_keys", ()))
        self.median_params = dict(result.get("median_params", {}))
        self.up_errors_params = dict(result.get("up_errors_params", {}))
        self.low_errors_params = dict(result.get("low_errors_params", {}))

        self.samples_pruned = result.get("samples_pruned")
        self.lnprob_pruned = result.get("lnprob_pruned")

        # --- Update physical parameters from medians (if present) ---
        for name in ("logN", "logQ", "T", "v_kms", "dlam"):
            if name in self.median_params:
                setattr(self, name, float(self.median_params[name]))

        # --- LSF handling ---
        if used_lsf is not None:
            # User-supplied fixed LSF during fit
            self.lsf = used_lsf
            self.lsf_method = "Given"
            for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
                setattr(self, name, None)

        else:
            # Rebuild parametric LSF from median_params + used_lsf_method
            self.lsf_method = used_lsf_method

            for name in ("sigma", "sigma1", "sigma2", "sigma_G", "fwhm_L", "ratio"):
                setattr(self, name, None)

            if self.lsf_method == "Gauss":
                if "sigma" in self.median_params:
                    self.sigma = float(self.median_params["sigma"])
                if self.sigma is None:
                    self.sigma = 0.01
                self.lsf = modeling.make_lsf({"sigma": self.sigma}, "Gauss")

            elif self.lsf_method == "2Gauss":
                vals = {}
                for nm in ("sigma1", "sigma2", "ratio"):
                    if nm in self.median_params:
                        val = float(self.median_params[nm])
                        setattr(self, nm, val)
                        vals[nm] = val
                if len(vals) == 3:
                    self.lsf = modeling.make_lsf(vals, "2Gauss")

            elif self.lsf_method == "Gauss_Lorentz":
                vals = {}
                for nm in ("sigma_G", "fwhm_L", "ratio"):
                    if nm in self.median_params:
                        val = float(self.median_params[nm])
                        setattr(self, nm, val)
                        vals[nm] = val
                if len(vals) == 3:
                    self.lsf = modeling.make_lsf(vals, "Gauss_Lorentz")
            elif self.lsf_method == "Lorentz":
                if "fwhm_L" in self.median_params:
                    self.fwhm_L = float(self.median_params["fwhm_L"])
                if self.fwhm_L is not None:
                    self.lsf = modeling.make_lsf(
                        {
                            "fwhm_L": self.fwhm_L,
                        },
                        "Lorentz",
                    )

        # --- Rebuild CN model from these updated parameters ---
        # This recomputes self.model_wave and self.median_model consistently.
        self.median_model = result.get("median_model", None)
        self.best_model = result.get("best_model", None)
        self.model_wave = result.get("model_wave", None)
        # --- Attach MCMC envelopes if provided ---
        # They should correspond to the same grid as model_wave.
        if "model_p16" in result and result["model_p16"] is not None:
            self.model_p16 = result["model_p16"]
        if "model_p84" in result and result["model_p84"] is not None:
            self.model_p84 = result["model_p84"]

    def _get_init_state(self) -> Dict[str, Any]:
            """
            Build a dict of kwargs that can be fed back into __init__
            to reconstruct this instance's configuration.

            Note: uses the parametric LSF description when possible.
            """
            # For a Given LSF, we try to pickle the callable directly.
            lsf_for_init = None
            lsf_method_for_init = self.lsf_method

            if self.lsf_method == "Given":
                lsf_for_init = self.lsf
                # keep lsf_method as "Given" so __init__ understands it
            else:
                # parametric: we only need method + its params
                lsf_for_init = None

            return dict(
                data=self.data,
                window=self.window,
                pumping=self.pumping,
                line_path=self.line_path,
                lsf=lsf_for_init,
                lsf_method=lsf_method_for_init,
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
                logQ=self.logQ,
                T=self.T,
                v_kms=self.v_kms,
                dlam=self.dlam,
            )

    def _get_fit_state(self) -> Dict[str, Any]:
        """
        State related to the MCMC / derived products.
        Safe to restore by direct attribute assignment.
        """
        return dict(
            priors=self.priors,
            param_keys=self.param_keys,
            median_params=self.median_params,
            up_errors_params=self.up_errors_params,
            low_errors_params=self.low_errors_params,
            samples_pruned=self.samples_pruned,
            lnprob_pruned=self.lnprob_pruned,
            v_up_error=self.v_up_error,
            v_low_error=self.v_low_error,
            v_samples=self.v_samples,
            model_wave=self.model_wave,
            median_model=self.median_model,
            best_model=self.best_model,
            model_p16=self.model_p16,
            model_p84=self.model_p84,
        )

    def save(self, filename: str) -> None:
        had_given_lsf = (self.lsf_method == "Given")

        init_kwargs = dict(
            data=self.data,
            window=self.window,
            pumping=self.pumping,
            line_path=self.line_path,
            lsf=None,  # never serialize the callable
            # if it was 'Given', fall back to a safe default on load
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
        )

        state = {
            "class": "FluorescenceModel",
            "version": 1,
            "init_kwargs": init_kwargs,
            "mcmc_result": mcmc_result,
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
        if state.get("version", 1) != 1:
            raise ValueError("Unsupported FluorescenceModel state version.")

        init_kwargs = state["init_kwargs"]
        mcmc_result = state.get("mcmc_result") or {}
        had_given_lsf = state.get("had_given_lsf", False)

        # Build base object (runs _synthesize_model)
        obj = cls(**init_kwargs)

        # Re-apply MCMC result if present
        if any(mcmc_result.values()):
            obj._update_from_result(
                mcmc_result,
                used_lsf=None,
                used_lsf_method=init_kwargs.get("lsf_method"),
            )

        if had_given_lsf:
            print(
                "Warning: original model used a custom 'Given' LSF which "
                "was not serialized. Please call `obj.update_model(lsf=...)` "
                "to restore the exact LSF."
            )

        return obj
