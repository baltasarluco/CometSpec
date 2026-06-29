Production Rate
===============

After fitting a fluorescence model, the column density
:math:`\log_{10} N` (:math:`N` in molecules cm\ :sup:`-2`) can be converted to a molecular
production rate :math:`Q` (molecules s\ :sup:`-1`) using one of two
approaches:

- **Haser model** — models a spherically expanding coma with parent and
  daughter photodissociation scale lengths.
- **ρ**\ :sup:`-1` **profile** — assumes a pure radial-outflow column-density
  profile :math:`N(\rho) \propto \rho^{-1}` and uses a pre-computed
  aperture integral.

The examples below build a short MCMC run first so that posterior samples are
available. To skip the sampling step, pass ``use_samples=False`` to use the
median ``logN`` directly (no error estimate is returned in that case); if you use
this approach, there is no need to fit.

.. rubric:: Sections

- :ref:`gs-pr-haser`
- :ref:`gs-pr-rho`
- :ref:`gs-pr-multi`
- :ref:`gs-pr-seeing`
- :ref:`gs-pr-ref`

----

Let's prepare a fitted model from which we will compute production rates.

.. note::

   The collisional parameter ``f_col`` is defined as
   :math:`f_{\rm col} \equiv \log_{10}(C_{ul}\,/\,\mathrm{s}^{-1})`, where the
   same downward rate :math:`C_{ul}` is assumed for every upper-to-lower pair
   included in the collision scaffold.

.. code-block:: python
    
    from cometspec.helper import open_kurucz_irradiance, open_hall_anderson_irradiance
    from cometspec import FluorescenceModel
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd

    irradiance = open_kurucz_irradiance()

    model = FluorescenceModel(
        window=(3870, 3885),
        pumping=irradiance,
        isotopologues="12C14N",
        systems=['BX', 'AX_dv1'],
        sigma=0.05,
        f_col=-1,
        logN=14.0,
        T=200,
        v_kms=0,
        lsf_method="Gauss",
        wave_col="WAVE",
        flux_col="FLUX",
        error_col="ERR",
        continuum_col="CONTINUUM",
    )

    # add noise
    data_flux = model.best_model + np.random.normal(0, np.max(model.best_model)*0.05, 
                                                        size=model.model_wave.shape)
    data = pd.DataFrame({'WAVE': model.model_wave, 
                        'FLUX': data_flux, 
                        'ERR':np.max(model.best_model)*0.05, 
                        'CONTINUUM': np.zeros_like(model.model_wave)})

    results = model.fit_mcmc(
                    data=data,
                    nwalkers=10,
                    nsteps=500,
                    priors={
                        "logN":  (12.0, 16.0),
                        "T":     (50.0, 300.0),
                        "v_kms": (-2.0, 2.0),
                    },
                    make_plots=False,
                    )
.. _gs-pr-haser:

Haser model
-----------

:meth:`~cometspec.fluorescence.FluorescenceModel.compute_production_rate` uses the
`sbpy <https://sbpy.readthedocs.io>`_ Haser implementation. You must supply
the geocentric distance, aperture geometry, CN scale lengths, and the outflow
velocity.

Typical CN scale lengths at 1 AU are: parent ~ 1.3 × 10\ :sup:`4` km,
daughter ~ 2.1 × 10\ :sup:`5` km [1]_. Both scale as :math:`r_h^2`.
The outflow velocity can be estimated as :math:`v = 0.85 r_h^{-0.5}` km s\ :sup:`-1` [2]_, where :math:`r_h` is the heliocentric distance in AU.

.. code-block:: python

   rh = 1.0   # your heliocentric distance [AU]

   logQ, logQ_err = model.compute_production_rate(
       delta_au=1.0, # your geocentric distance [AU]
       aperture={"type": "circular", "radius_arcsec": 5.0},
       parent_length_km=1.3e4 * rh**2,
       daughter_length_km=2.1e5 * rh**2,
       v_outflow_km_s=0.85 * rh**(-0.5),
   )
   print(f"log Q(CN) = {logQ:.2f} ± {logQ_err:.2f}  [s^-1]")

Rectangular (slit) apertures are also supported:

.. code-block:: python

   logQ_slit, logQ_slit_err = model.compute_production_rate(
       delta_au=1.0,
       aperture={"type": "rectangular", "width_arcsec": 2.0, "length_arcsec": 10.0},
       parent_length_km=1.3e4 * rh**2,
       daughter_length_km=2.1e5 * rh**2,
       v_outflow_km_s=0.85 * rh**(-0.5),
   )
   print(f"log Q(CN, slit) = {logQ_slit:.2f} ± {logQ_slit_err:.2f}")

The production rate and its error are stored in the attributes ``model.logQ`` and ``model.logQ_err`` after calling ``compute_production_rate``.

----

.. _gs-pr-rho:

ρ\ :sup:`-1` profile via aperture integral
------------------------------------------

For a :math:`N(\rho) \propto \rho^{-1}` profile where :math:`\rho` is the radial distance from the nucleus, first compute the
geometric aperture integral **G** with
:meth:`~cometspec.fluorescence.FluorescenceModel.compute_aperture_integral`, then pass it
to :meth:`~cometspec.fluorescence.FluorescenceModel.compute_production_rate_from_profile`.

.. code-block:: python

   G = model.compute_aperture_integral(
       aperture={"type": "circular", "radius_arcsec": 5.0},
       delta_au=1.0,
   )
   print("Aperture integral G:", G)

   logQ_prof, logQ_prof_err = model.compute_production_rate_from_profile(
       G=G,
       delta_au=1.0,
       aperture={"type": "circular", "radius_arcsec": 5.0},
       v_outflow_km_s=0.85 * rh**(-0.5),
   )
   print(f"log Q (rho^-1 profile) = {logQ_prof:.2f} ± {logQ_prof_err:.2f}")

----

.. _gs-pr-multi:

Multi-isotopologue production rates
------------------------------------

When multiple isotopologues are fitted,
:meth:`~cometspec.fluorescence.FluorescenceModel.compute_production_rate` returns a
``dict`` keyed by isotopologue name. You can also use the attribute ``model.logQ`` to access the production rates and errors for each isotopologue after calling ``compute_production_rate``.

.. code-block:: python

    # Two-iso fit
    model = FluorescenceModel(
        pumping=irradiance,
        isotopologues=["12C14N", "13C14N"], systems="BX",
        logN_by_iso={"12C14N": 14, "13C14N": 13.0},
        T=100.0, sigma=0.02, f_col=-1, lsf_method="Gauss",
        wave_col="WAVE", flux_col="FLUX", error_col="ERR", 
        continuum_col="CONTINUUM",
    )
    data_flux = model.best_model + np.random.normal(0, np.max(model.best_model)*0.05,
                                                        size=model.model_wave.shape)
    data = pd.DataFrame({'WAVE': model.model_wave,
                        'FLUX': data_flux,
                        'ERR':np.max(model.best_model)*0.05,
                        'CONTINUUM': np.zeros_like(model.model_wave)})


    model.fit_mcmc(
        data=data,
        nwalkers=20, nsteps=300,
        priors={
            "logN_12C14N": (12.0, 16.0),
            "logN_13C14N": (11.0, 14.0),
            "T":           (50.0, 300.0),
        },
        make_plots=False, progress=False,
    )

    rates = model.compute_production_rate(
        delta_au=1.0,
        aperture={"type": "circular", "radius_arcsec": 5.0},
        parent_length_km=1.3e4 * rh**2,
        daughter_length_km=2.1e5 * rh**2,
        v_outflow_km_s=0.85 * rh**(-0.5),
    )
    for iso, (logQ_i, logQ_i_err) in rates.items():
        print(f"log Q({iso}) = {logQ_i:.2f} ± {logQ_i_err:.2f}")

.. note::
    
    If you have multiple isotopologues and want to use ``use_samples=False``, you need to have both present in ``logN_by_iso``.
    On the other hand, if multiple isotopologues are present and you want to use ``use_samples=True``, you will need to fit both isotopologues;
    otherwise the code raises an error. However, you can work around this by adding the missing isotopologue later to ``logN_by_iso``, appending
    the missing ``logN_<isotopologue>`` at the end of ``param_keys``, and adding an array of the repeated value to ``samples_pruned``.
    An example of this can be found in the Jupyter notebook **WorkFlow.ipynb** of the GitHub repository (`link <https://github.com/baltasarluco/CometSpec>`__).

----

.. _gs-pr-seeing:

Seeing-corrected production rates
-----------------------------------

:meth:`~cometspec.fluorescence.FluorescenceModel.add_slit_loss_error` inflates the
column-density uncertainty to account for seeing-induced slit losses. Call it on the fitted model after
``compute_production_rate``. This will also compute the slit-loss error of ``logN``.

.. code-block:: python

    print('Before adding slit-loss error:')
    for iso in model.logQ:
        print(f"log Q({iso}) = {model.logQ[iso]:.3f} ± {model.logQ_err[iso]:.3f}")
    for iso in model.logN_by_iso:
        print(f"log N({iso}) = {model.logN_by_iso[iso]:.2f} ± {model.logN_err_by_iso[iso][0]:.2f}") # for N you have up and low error, we just show 1

    model.add_slit_loss_error(
    lambda_nm=388.3, # the wavelength of the fitted band or line [nm]
    aperture={"type": "circular", "radius_arcsec": .5}, 
                # small aperture to illustrate the effect
    eps_min_arcsec_500=0.7,   # best-case seeing at 500 nm [arcsec]
    eps_max_arcsec_500=1.5,   # worst-case seeing at 500 nm [arcsec]
    zmin_deg=20, # minimum zenith angle of observations [degrees]
    zmax_deg=60, # maximum zenith angle of observations [degrees]
    )

    print('After adding slit-loss error:')
    for iso in model.logQ:
        print(f"log Q({iso}) = {model.logQ[iso]:.3f} ± {model.logQ_err[iso]:.3f}")
    for iso in model.logN_by_iso:
        print(f"log N({iso}) = {model.logN_by_iso[iso]:.2f} ± {model.logN_err_by_iso[iso][0]:.2f}") # for N you have up and low error, we just show 1


If only one isotopologue is fitted, you have ``logN`` and ``logN_err``. We encourage you to read the documentation of :func:`~cometspec.fluorescence.FluorescenceModel.add_slit_loss_error`
for more details on the method and its assumptions. Since this method adds the initial error in quadrature with the slit-loss error,
the slit-loss error is not recalculated if ``add_slit_loss_error`` is called multiple times, in order to avoid accumulating it.
If you want to recalculate it, you will need to either 1) fit again, or 2) reset the attributes ``logQ_seeing_corrected = False`` and ``logN_seeing_corrected = False``
and run the production rate calculation again (if the latter is not applied, the error will start accumulating repeatedly).

.. _gs-pr-ref:

References
----------

.. [1] A’Hearn, M. F., Millis, R. C., Schleicher, D. O., Osip, D. J., & Birch, P. V. 1995, Icarus, 118, 223 (`link <https://ui.adsabs.harvard.edu/abs/1995Icar..118..223A/abstract>`__)
.. [2] Cochran, A. L. & Schleicher, D. G. 1993, Icarus, 105, 235 (`link <https://ui.adsabs.harvard.edu/abs/1993Icar..105..235C/abstract>`__)