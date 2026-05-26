.. py:module:: cometspec

.. raw:: html

   <div class="hero-section">
      <h1>CometSpec</h1>
      <p class="hero-tagline">Fluorescence modeling &amp; MCMC fitting for cometary spectra</p>
      <div class="hero-badges">
         <a href="https://github.com/baltasarluco/CometSpec">
            <img src="https://img.shields.io/badge/GitHub-CometSpec-blue?logo=github" alt="GitHub">
         </a>
         <a href="https://pypi.org/project/cometspec/">
            <img src="https://img.shields.io/pypi/v/cometspec?color=blue" alt="PyPI">
         </a>
         <a href="https://opensource.org/licenses/MIT">
            <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
         </a>
         <a href="https://www.python.org/">
            <img src="https://img.shields.io/badge/Python-≥3.9-blue?logo=python&logoColor=white" alt="Python ≥3.9">
         </a>
      </div>
      <img src="_static/comet-swirl-twin.svg" alt="comet animation" class="hero-comet"/>
   </div>


**CometSpec** is a Python package for modeling fluorescence emission in
cometary spectra and fitting observed data using Markov Chain Monte Carlo
(MCMC) methods.


.. raw:: html

   <div class="feature-grid">
      <div class="feature-card">
         <div class="feature-icon">&#9734;</div>
         <h3>Fluorescence Modeling</h3>
         <p>Physics-based fluorescence equilibrium model with solar radiative pumping, spontaneous and stimulated emission, and collisional excitation.</p>
      </div>
      <div class="feature-card">
         <div class="feature-icon">&#9881;</div>
         <h3>MCMC Fitting</h3>
         <p>Bayesian parameter estimation via <code>emcee</code> with flexible priors and multi-isotopologue support.</p>
      </div>
      <div class="feature-card">
         <div class="feature-icon">&#9878;</div>
         <h3>Multi-Isotopologue</h3>
         <p>Simultaneous modeling of <sup>12</sup>C<sup>14</sup>N, <sup>13</sup>C<sup>14</sup>N, and <sup>12</sup>C<sup>15</sup>N, <sup>12</sup>C<sub>2</sub>, <sup>13</sup>C<sub>2</sub>, <sup>12</sup>C<sup>13</sup>C, and Fe. Additinally custom line list are supported.</p>
      </div>
      <div class="feature-card">
         <div class="feature-icon">&#9788;</div>
         <h3>Production Rates</h3>
         <p>Derive  production rates from fitted column densities using the Haser coma model or asumming a &rho;<sup>-1</sup> profile</p>
      </div>
   </div>


Roadmap
-------

CometSpec is under active development. Planned additions include:

- **Dust and solar continuum** — model the underlying continuum alongside emission lines
- **More molecules** — extend beyond CN, C\ :sub:`2`, Fe to other cometary species and polishing of existing line lists.
- **Non-equilibrium and non-LTE models** — go beyond fluorescence equilibrium to model more complex excitation conditions
- **Nested sampling** — add nested sampling as an alternative or complement to MCMC for posterior estimation

If you are interested in collaborating, feel free to reach out at
`baltasarluco@uc.cl <mailto:baltasarluco@uc.cl>`_.

Getting Started
---------------

Install CometSpec and run your first model in minutes:

.. code-block:: bash

   pip install cometspec

Then follow the :doc:`getting_started/index` guide or explore the full :doc:`api/index`.

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   installation
   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: About

   citation
