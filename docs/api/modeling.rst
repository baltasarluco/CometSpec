``cometspec.modeling``
======================

Backwards-compatible re-export shim. The implementation has been split into
focused submodules; You can import names from ``cometspec.modeling`` or from
each submodule directly.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Submodule
     - Description
   * - :doc:`linelist`
     - Line-list parsing, normalization, defaults, and label attachment.
   * - :doc:`rates`
     - Rate matrix, populations, g-factors, spectrum synthesis, and LSFs.
   * - :doc:`collisions`
     - Rotational-collision scaffolds and diatomic-symmetry helpers.
   * - :doc:`mcmc`
     - MCMC fitting kernel.

.. toctree::
   :maxdepth: 1
   :hidden:

   linelist
   rates
   collisions
   mcmc
