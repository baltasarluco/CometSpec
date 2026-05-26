"""
CometSpec: fluorescence modeling and MCMC fitting for cometary spectra.

Submodules:
- cometspec.helper       : file I/O & utilities
- cometspec.modeling     : line lists, rate matrix, g-factors, synthesis, MCMC
- cometspec.fluorescence : FluorescenceModel class
- cometspec.config       : optional grouped-configuration dataclasses
"""

from . import helper, modeling, fluorescence, config
from .config import FluorescenceModelConfig, MCMCFitConfig
from .fluorescence import FluorescenceModel

__all__ = [
    "helper",
    "modeling",
    "fluorescence",
    "config",
    "FluorescenceModel",
    "FluorescenceModelConfig",
    "MCMCFitConfig",
]

__version__ = "0.1.0"
