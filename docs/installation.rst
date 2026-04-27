Installation
============

Requirements
------------

CometSpec requires **Python 3.9** or later and depends on:

- `NumPy <https://numpy.org>`_ (≥1.22)
- `Pandas <https://pandas.pydata.org>`_ (≥1.5)
- `Matplotlib <https://matplotlib.org>`_ (≥3.5)
- `Astropy <https://www.astropy.org>`_ (≥5.0)
- `sbpy <https://sbpy.org>`_ (≥0.4)
- `emcee <https://emcee.readthedocs.io>`_ (≥3.1)
- `corner <https://corner.readthedocs.io>`_ (≥2.2)
- `specutils <https://specutils.readthedocs.io>`_ (≥1.9)
- `tqdm <https://tqdm.github.io>`_ (≥4.60)
- `PyTables <https://www.pytables.org>`_ (``tables`` ≥3.7) — required to read the
  packaged C\ :sub:`2` HDF5 line lists via :func:`pandas.read_hdf`.

Install from source
-------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/baltasarluco/CometSpec.git
   cd CometSpec/cometspec
   pip install .

For development (editable install with test dependencies):

.. code-block:: bash

   pip install -e ".[dev]"

Verify installation
-------------------

.. code-block:: python

   import cometspec
   print(cometspec.__version__)
   # 0.1.0
