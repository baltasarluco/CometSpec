"""Backwards-compatible re-export shim.

The implementation has been split into focused submodules:

- :mod:`cometspec.linelist`   -- line-list parsing & normalization
- :mod:`cometspec.rates`      -- rate-matrix, g-factors, synthesis
- :mod:`cometspec.collisions` -- rotational-collision scaffolds
- :mod:`cometspec.mcmc`       -- MCMC fitting

All functions in linelist, rates, collisions, and mcmc are re-exported here for backwards compatibility. You can import from modeling or from each submodule.
"""

from .linelist import *  # noqa: F401, F403
from .rates import *     # noqa: F401, F403
from .collisions import *  # noqa: F401, F403
from .mcmc import *      # noqa: F401, F403
