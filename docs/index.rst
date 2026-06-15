pyLACEpostHoc
=============

Database, post-hoc analysis, and plotting layer for the LACE pose
estimator family (Geurten 2022, *Frontiers in Behavioural
Neuroscience*).  This package consumes the LACE tracker's outputs
(CSV / SQLite / HDF5 + figure triplets) and produces publication-ready
statistics and figures.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Installation
------------

.. code-block:: bash

   pip install -e ".[full]"      # core + I/O readers + survival stats
   # or, lightweight:
   pip install -e .

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
