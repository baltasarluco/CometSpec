# CometSpec

Fluorescence modeling and MCMC fitting for cometary spectra.

- **Documentation:** [https://cometspec.readthedocs.io/en/latest/](https://cometspec.readthedocs.io/en/latest/)
- **Source code:** [https://github.com/baltasarluco/CometSpec](https://github.com/baltasarluco/CometSpec)

A usage example is provided in `WorkFlow_Example/` as a Jupyter notebook called `WorkFlow.ipynb`, and `Test_getting_started.ipynb` are the cells on the documentation.

For citation information, see the [Citation page in the documentation](https://cometspec.readthedocs.io/en/latest/citation.html).

Data DOI: [10.5281/zenodo.19801569](https://zenodo.org/records/19801569)

Repo DOI: [10.5281/zenodo.21534317](https://zenodo.org/records/21534317)

Paper: [`Luco et al. 2026 (ADS)`](https://ui.adsabs.harvard.edu/abs/2026arXiv260715355L/abstract)

---

## Install

```bash
pip install cometspec
```

Or install from source:

```bash
git clone https://github.com/baltasarluco/CometSpec.git
cd CometSpec
pip install .
```

---

## Roadmap

CometSpec is under active development. Planned additions include:

- **Dust and solar continuum** — model the underlying continuum alongside emission lines
- **More molecules** — extend beyond CN, C₂, Fe to other cometary species and polishing of existing line lists.
- **Non-equilibrium and non-LTE models** — go beyond fluorescence equilibrium to model more complex excitation conditions
- **Nested sampling** — add nested sampling as an alternative or complement to MCMC for posterior estimation

If you are interested in collaborating, feel free to reach out at [baltasarluco@uc.cl](mailto:baltasarluco@uc.cl).

---

## Contact

For questions, suggestions, or bug reports, open an issue on [GitHub](https://github.com/baltasarluco/CometSpec/issues) or contact [baltasarluco@uc.cl](mailto:baltasarluco@uc.cl).

## Version 0.1.1 what's new:

- We add the possibility to include the $\rm X^2 \Sigma^+$ rovibrational transitions of CN.
- We improve the mcmc plotting, which was previously hidding some labels for some corner cases.
- We changed two parameter names: 
    - `logQ` is now `f_col`. In order to avoid confusion between collisions and production rates.
    - `q` is now `logQ` to clarify that the production rate is in log scale.
    - In relation top both changes, any `_err`, `_iso` or other parameter directly related to the old`logQ` or `q` has been renamed accordingly.
    