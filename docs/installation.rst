Installation
============

Requirements
------------

CometSpec requires **Python 3.9** or later and depends on:

- `NumPy <https://numpy.org>`_ (≥1.22)
- `SciPy <https://scipy.org>`_ (≥1.7)
- `Pandas <https://pandas.pydata.org>`_ (≥1.5)
- `Matplotlib <https://matplotlib.org>`_ (≥3.5)
- `Astropy <https://www.astropy.org>`_ (≥5.0)
- `specutils <https://specutils.readthedocs.io>`_ (≥1.10)
- `sbpy <https://sbpy.org>`_ (≥0.4)
- `emcee <https://emcee.readthedocs.io>`_ (≥3.1)
- `corner <https://corner.readthedocs.io>`_ (≥2.2)
- `tqdm <https://tqdm.github.io>`_ (≥4.60)
- `PyTables <https://www.pytables.org>`_ (``tables`` ≥3.7)
- `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ (≥3.0)
- `multiprocess <https://github.com/uqfoundation/multiprocess>`_ (≥0.70)

Install from PyPI
-----------------
.. code-block:: bash

   pip install cometspec

Install from source
-------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/baltasarluco/CometSpec.git
   cd CometSpec
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
