# CometSpec

Fluorescence modeling and MCMC fitting for cometary spectra.

**Documentation:** [https://cometspec.readthedocs.io/en/latest/](https://cometspec.readthedocs.io/en/latest/)

A usage example is provided in `WorkFlow_Example/` as a Jupyter notebook called `WorkFlow.ipynb`, and `Test_getting_started.ipynb` are the cells on the documentation.

If you use this code, please cite: *waiting for publication.*

Repository DOI: *waiting for DOI.*  
Data DOI: [10.5281/zenodo.19801570](https://doi.org/10.5281/zenodo.19801570)


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
