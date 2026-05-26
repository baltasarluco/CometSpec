API Reference
=============

CometSpec is organized into the following modules:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`fluorescence`
     - High-level ``FluorescenceModel`` class — the main entry point for
       building models and running MCMC fits.
   * - :doc:`modeling`
     - Core engine roadmap. Split across :doc:`linelist`, :doc:`rates`,
       :doc:`collisions`, and :doc:`mcmc`. ``cometspec.modeling`` itself is a
       backwards-compatible re-export shim.
   * - :doc:`helper`
     - Simple utilities for common tasks as loading data and to calculate slit-losses errors (which ``FluorescenceModel`` uses).
   * - :doc:`config`
     - Optional grouped-configuration dataclasses
       (``FluorescenceModelConfig``, ``MCMCFitConfig``).

.. toctree::
   :maxdepth: 2
   :hidden:

   fluorescence
   modeling
   helper
   config
