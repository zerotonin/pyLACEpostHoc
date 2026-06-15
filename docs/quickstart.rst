Quickstart
==========

Install
-------

.. code-block:: bash

   pip install -e ".[full]"   # core + I/O readers + survival stats
   # or, lightweight (pure-Python helpers + plotting only):
   pip install -e .

Configure machine-specific paths
--------------------------------

Data roots are **not** hardcoded. Before running anything, copy the
template and fill in the real paths for your machine:

.. code-block:: bash

   cp local_paths.template.json local_paths.json   # gitignored
   $EDITOR local_paths.json                          # set data_root, database_path, figure_root

Paths resolve in three steps: a ``PYLACE_POSTHOC_<KEY>`` environment
variable wins, then the active profile in ``local_paths.json`` (``local``
by default, or ``hpc``; select with ``PYLACE_POSTHOC_PROFILE``). A missing
``local_paths.json`` fails loudly and names the template to copy.

.. code-block:: python

   import config

   db_root = config.get_path("database_path")          # active profile
   scratch = config.get_path("data_root", profile="hpc")

Build the database
------------------

Sort a folder of raw recordings and run each one through the analysers:

.. code-block:: python

   from fish_data_base.fishDataBase import FishDataBase

   db = FishDataBase()                       # database_path from config
   db.run_multi_trace_folder(
       folder_position=raw_folder,
       gene_name="rei",
       experiment_str="CCur",                # CCur | Ta | Unt | cst
       birth_date="11-2018",
   )

Analyse and plot
----------------

.. code-block:: python

   import pandas as pd
   from data_base_analyser.CounterCurrentAnalyser import CounterCurrentAnalyser

   df = pd.read_csv(config.get_path("database_path") / "fishDataBase.csv")
   analyser = CounterCurrentAnalyser(df)
   analyser.main(fig_path=f"{config.get_path('figure_root')}/",
                 data_path=f"{config.get_path('figure_root')}/")

Save a figure with the lab triple output (SVG + PNG + CSV):

.. code-block:: python

   import matplotlib.pyplot as plt
   from constants import save_figure

   fig, ax = plt.subplots()
   ax.plot([0, 1], [0, 1])
   save_figure(fig, "demo", config.get_path("figure_root"), data=df)

Ready-to-run entry points live in ``run_scripts/`` (not packaged); each
reads its paths from ``config``.
