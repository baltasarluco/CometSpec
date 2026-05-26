"""Geometry helpers extracted from :class:`FluorescenceModel`.

Pure aperture/projection utilities. None of these functions need access
to a :class:`FluorescenceModel` instance.
"""
from __future__ import annotations

import numpy as np
from astropy import units as u
from sbpy.activity import CircularAperture, RectangularAperture


def km_per_arcsec(delta_au: float) -> float:
    """Implementation of FluorescenceModel._km_per_arcsec."""
    delta_km = (float(delta_au) * u.au).to(u.km).value
    return float(delta_km * np.tan(1.0 * u.arcsec.to(u.rad)))


def aperture_area_cm2(cls, aperture: dict, *, delta_au: float) -> u.Quantity:
    """Implementation of FluorescenceModel._aperture_area_cm2."""
    ap_type = aperture.get("type", "").lower().strip()
    km_per_arcsec_val = cls._km_per_arcsec(delta_au)

    if ap_type == "circular":
        R_arcsec = float(aperture["radius_arcsec"])
        R_km = R_arcsec * km_per_arcsec_val
        R_cm = R_km * 1e5
        return (np.pi * R_cm**2) * u.cm**2

    if ap_type == "rectangular":
        W_arcsec = float(aperture["width_arcsec"])
        L_arcsec = float(aperture["length_arcsec"])
        W_km = W_arcsec * km_per_arcsec_val
        L_km = L_arcsec * km_per_arcsec_val
        W_cm = W_km * 1e5
        L_cm = L_km * 1e5
        return (W_cm * L_cm) * u.cm**2

    raise ValueError("aperture['type'] must be 'circular' or 'rectangular'")


def sbpy_aperture(cls, aperture: dict, *, delta_au: float):
    """Implementation of FluorescenceModel._sbpy_aperture."""
    ap_type = aperture.get("type", "").lower().strip()
    km_per_arcsec_val = cls._km_per_arcsec(delta_au)

    if ap_type == "circular":
        R_arcsec = float(aperture["radius_arcsec"])
        R_km = R_arcsec * km_per_arcsec_val
        return CircularAperture((R_km * u.km))

    if ap_type == "rectangular":
        W_arcsec = float(aperture["width_arcsec"])
        L_arcsec = float(aperture["length_arcsec"])
        W_km = W_arcsec * km_per_arcsec_val
        L_km = L_arcsec * km_per_arcsec_val
        return RectangularAperture((W_km, L_km) * u.km)

    raise ValueError("aperture['type'] must be 'circular' or 'rectangular'")


def arcsec_to_km(*, delta_au: float) -> float:
    """Implementation of FluorescenceModel._arcsec_to_km."""
    AU_KM = 1.495978707e8
    ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
    return float(delta_au) * AU_KM * ARCSEC_TO_RAD
