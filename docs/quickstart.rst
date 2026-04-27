Quick Start
===========

This guide walks through a minimal example: building a CN fluorescence model
and fitting it to an observed cometary spectrum.

1. Load the solar pumping spectrum
----------------------------------

CometSpec ships with the Kurucz solar irradiance. Load it as the pumping
source for the fluorescence model:

.. code-block:: python

   import pandas as pd
   from cometspec import helper

   # Load the bundled Kurucz irradiance
   kurucz_path = helper.get_kurucz_irradiance_path()
   pumping = helper.open_kurucz_irradiance(kurucz_path)

2. Create a fluorescence model
------------------------------

Build a model for the CN B–X and A–X systems in a given wavelength window:

.. code-block:: python

   from cometspec.fluorescence import FluorescenceModel

   model = FluorescenceModel(
       pumping=pumping,
       isotopologues="12C14N",
       window=(3850, 3900),          # Angstroms
       logN=12.0,                    # log10 column density (cm^-2)
       T=300.0,                      # rotational temperature (K)
   )

The model automatically:

- Loads the packaged CN line list (Brooke et al.)
- Builds the rate matrix with solar pumping rates
- Solves level populations
- Synthesizes a spectrum convolved with a Gaussian LSF

3. Inspect the synthetic spectrum
---------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 4))
   plt.plot(model.model_wave, model.best_model)
   plt.xlabel("Wavelength (Å)")
   plt.ylabel("Flux")
   plt.title("Synthetic CN fluorescence spectrum")
   plt.tight_layout()
   plt.show()

4. Fit observed data with MCMC
------------------------------

If you have an observed spectrum, pass it as a ``DataFrame`` with wavelength,
flux, error, and continuum columns:

.. code-block:: python

   model = FluorescenceModel(
       data=observed_df,
       pumping=pumping,
       isotopologues="12C14N",
       window=(3850, 3900),
       wave_col="WAVE",
       flux_col="FLUX_STACK",
       error_col="ERR_STACK",
       continuum_col="CONTINUUM",
   )

   # Run the MCMC fit
   model.fit_mcmc()

   # Results
   print("Median parameters:", model.median_params)
   print("Upper errors:", model.up_errors_params)
   print("Lower errors:", model.low_errors_params)

5. Multi-isotopologue fitting
-----------------------------

Fit multiple CN isotopologues simultaneously:

.. code-block:: python

   model = FluorescenceModel(
       data=observed_df,
       pumping=pumping,
       isotopologues=["12C14N", "13C14N", "12C15N"],
       window=(3850, 3900),
   )

   model.fit_mcmc()

6. Compute production rates
---------------------------

After fitting, derive the CN production rate using the Haser coma model:

.. code-block:: python

   model.compute_production_rate()
   print(f"log Q = {model.q} ± {model.q_err}")

7. Save and load models
------------------------

.. code-block:: python

   # Save
   model.save("my_fit.pkl")

   # Load
   loaded = FluorescenceModel.load("my_fit.pkl")

Next steps
----------

- See the :doc:`api/index` for the full API reference.
- Explore the ``WorkFlow_Example/`` directory in the repository for a
  complete Jupyter notebook walkthrough.
