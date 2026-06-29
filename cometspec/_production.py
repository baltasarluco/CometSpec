"""Production-rate implementations extracted from :class:`FluorescenceModel`.

Provides the bodies of :meth:`FluorescenceModel.compute_production_rate`,
:meth:`FluorescenceModel.add_slit_loss_error`,
:meth:`FluorescenceModel.compute_aperture_integral`, and
:meth:`FluorescenceModel.compute_production_rate_from_profile`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Union

import numpy as np
from astropy import units as u
from sbpy.activity import Haser

from . import helper

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fluorescence import FluorescenceModel


def compute_production_rate(
    model: "FluorescenceModel",
    *,
    delta_au: float,
    aperture: dict,
    parent_length_km: float,
    daughter_length_km: float,
    v_outflow_km_s: float,
    use_samples: bool = True,
    N_total_coma_km: float = 1e7,
) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
    """Implementation of FluorescenceModel.compute_production_rate."""
    iso_list = model._iso_list()

    # Build Haser model
    haser = Haser(
        Q=1 * u.s**-1,
        v=float(v_outflow_km_s) * u.km / u.s,
        parent=float(parent_length_km) * u.km,
        daughter=float(daughter_length_km) * u.km,
    )

    # aperture objects / area
    A_cm2 = model._aperture_area_cm2(aperture, delta_au=float(delta_au))
    ap_sbpy = model._sbpy_aperture(aperture, delta_au=float(delta_au))

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
        if model.samples_pruned is None or model.param_keys is None:
            raise ValueError("No MCMC samples available (samples_pruned/param_keys missing). Fit first or set use_samples=False.")
        pkeys = list(model.param_keys)
        if len(iso_list) == 1:
            key = "logN"
        else:
            key = f"logN_{iso}"
        if key not in pkeys:
            raise KeyError(f"Missing parameter '{key}' in chains. param_keys={model.param_keys}")
        j = pkeys.index(key)
        return np.asarray(model.samples_pruned[:, j], float)


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
            logN_chain = np.array([float(model.logN)], dtype=float)
        q50, qerr = compute_from_logN(logN_chain)
        model.logQ = q50
        model.logQ_err = qerr
        return q50, qerr

    # multi-iso
    out: Dict[str, Tuple[float, float]] = {}
    for iso in iso_list:
        if use_samples:
            logN_chain = get_logN_chain_for_iso(iso)
        else:
            if model.logN_by_iso is None or iso not in model.logN_by_iso:
                raise ValueError(f"Missing logN_by_iso[{iso}] and use_samples=False.")
            logN_chain = np.array([float(model.logN_by_iso[iso])], dtype=float)

        q50, qerr = compute_from_logN(logN_chain)
        out[iso] = (q50, qerr)

    model.logQ = {k: v[0] for k, v in out.items()}
    model.logQ_err = {k: v[1] for k, v in out.items()}
    return out


def add_slit_loss_error(
    model: "FluorescenceModel",
    *,
    lambda_nm: float,
    aperture: dict,
    correct: str = "both",  # "logQ", "logN", or "both"
    eps_min_arcsec_500: float = 0.7,
    eps_max_arcsec_500: float = 1.2,
    zmin_deg: float = 45.0,
    zmax_deg: float = 45.0,
    n_points: int = 2000,
) -> Union[float, Dict[str, float]]:
    """Implementation of FluorescenceModel.add_slit_loss_error."""
    if correct not in ("logQ", "logN", "both"):
        raise ValueError("correct must be 'logQ', 'logN', or 'both'.")

    do_logQ = correct in ("logQ", "both")
    do_logN = correct in ("logN", "both")

    if do_logQ and (model.logQ is None or model.logQ_err is None):
        raise ValueError("self.logQ and self.logQ_err must be set before correcting logQ. Recomended to fit first or set it manually to 0")
    if do_logN and (model.logN is None or model.logN_err is None) and (model.logN_by_iso is None or model.logN_err_by_iso is None):
        raise ValueError("self.logN and self.logN_err must be set before correcting logN.")
    if do_logQ and model.logQ_seeing_corrected:
        print('logQ was already corrected, skipping.')
        do_logQ = False
    if do_logN and model.logN_seeing_corrected:
        print('logN was already corrected, skipping.')
        do_logN = False

    if not do_logQ and not do_logN:
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

    iso_list = model._iso_list()

    if len(iso_list) == 1:
        if do_logQ:
            model.logQ_err = helper.add_slit_loss_error_scalar(float(model.logQ_err), **_slitloss_kwargs
            )
            model.logQ_seeing_corrected = True
        if do_logN:
            model.logN_err = np.array([
                helper.add_slit_loss_error_scalar(float(model.logN_err[0]), **_slitloss_kwargs
                ),
                helper.add_slit_loss_error_scalar(float(model.logN_err[1]), **_slitloss_kwargs
                ),
            ])
            model.logN_seeing_corrected = True
        return

    # multi-iso
    if not isinstance(model.logQ, dict) or not isinstance(model.logQ_err, dict):
        raise ValueError("For multi-isotopologue models, self.logQ and self.logQ_err must be dicts keyed by iso.")

    new_errs: Dict[str, float] = {}
    for iso in iso_list:
        if do_logQ:
            if iso not in model.logQ or iso not in model.logQ_err:
                raise KeyError(f"Missing logQ/logQ_err for iso='{iso}'.")
            new_errs[iso] = helper.add_slit_loss_error_scalar(float(model.logQ_err[iso]), **_slitloss_kwargs
            )
        if do_logN:
            model.logN_err_by_iso[iso] = np.array([
                helper.add_slit_loss_error_scalar(float(model.logN_err_by_iso[iso][0]), **_slitloss_kwargs
                ),
                helper.add_slit_loss_error_scalar(float(model.logN_err_by_iso[iso][1]), **_slitloss_kwargs
                ),
            ])

    if do_logQ:
        model.logQ_err = new_errs
        model.logQ_seeing_corrected = True
    if do_logN:
        model.logN_seeing_corrected = True
    return


def compute_aperture_integral(
    model: "FluorescenceModel",
    *,
    aperture: dict,
    delta_au: float,
) -> u.Quantity:
    """Implementation of FluorescenceModel.compute_aperture_integral."""
    km_per_arcsec = model._arcsec_to_km(delta_au=float(delta_au))
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
    model: "FluorescenceModel",
    *,
    G: u.Quantity,
    delta_au: float,
    aperture: dict,
    v_outflow_km_s: float,
    use_samples: bool = True,
) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
    """Implementation of FluorescenceModel.compute_production_rate_from_profile."""
    iso_list = model._iso_list()
    A_cm2 = model._aperture_area_cm2(aperture, delta_au=float(delta_au))
    v = float(v_outflow_km_s) * u.km / u.s

    # Validate G has the right dimension
    try:
        G = G.to(u.cm)
    except u.UnitConversionError:
        raise ValueError(f"G must be convertible to cm, got units '{G.unit}'.")

    def get_logN_chain(iso: str) -> np.ndarray:
        if model.samples_pruned is None or model.param_keys is None:
            raise ValueError(
                "No MCMC samples available. Fit first or set use_samples=False."
            )
        pkeys = list(model.param_keys)
        key = "logN" if len(iso_list) == 1 else f"logN_{iso}"
        if key not in pkeys:
            raise KeyError(f"Missing parameter '{key}' in chains. param_keys={model.param_keys}")
        return np.asarray(model.samples_pruned[:, pkeys.index(key)], float)

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
        chain = get_logN_chain(iso) if use_samples else np.array([float(model.logN)])
        q50, qerr = compute_from_logN(chain)
        model.logQ, model.logQ_err = q50, qerr
        return q50, qerr

    # --- multi-isotopologue ---
    out: Dict[str, Tuple[float, float]] = {}
    for iso in iso_list:
        if use_samples:
            chain = get_logN_chain(iso)
        else:
            if model.logN_by_iso is None or iso not in model.logN_by_iso:
                raise ValueError(f"Missing logN_by_iso['{iso}'] and use_samples=False.")
            chain = np.array([float(model.logN_by_iso[iso])])
        out[iso] = compute_from_logN(chain)

    model.logQ     = {k: v[0] for k, v in out.items()}
    model.logQ_err = {k: v[1] for k, v in out.items()}
    return out
