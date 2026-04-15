"""
CometSpec: fluorescence modeling and MCMC fitting for cometary spectra.

Submodules:
- cometspec.helper   : file I/O & utilities
- cometspec.modeling : line lists, rate matrix, g-factors, synthesis, MCMC
- cometspec.fluorescence     :  FluorescenceModel class
"""

from . import helper, modeling, fluorescence

__all__ = ["helper", "modeling", "fluorescence", ]

__version__ = "0.1.0"
