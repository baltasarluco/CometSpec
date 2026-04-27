API Reference
=============

CometSpec is organized into three modules:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`fluorescence`
     - High-level ``FluorescenceModel`` class — the main entry point for
       building models and running MCMC fits.
   * - :doc:`modeling`
     - Core engine: line-list normalization, rate matrices, level populations,
       g-factors, spectrum synthesis, and MCMC fitting.
   * - :doc:`helper`
     - File I/O utilities, solar irradiance loading, line-list access,
       slit-loss corrections, and seeing calculations.

.. toctree::
   :maxdepth: 2
   :hidden:

   fluorescence
   modeling
   helper
