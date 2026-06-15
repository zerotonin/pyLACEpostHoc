Overview
========

pyLACEpostHoc is the database, post-hoc analysis, and plotting layer for
the LACE pose estimator family (Garg et al. 2022, *Frontiers in
Behavioural Neuroscience*). It consumes the LACE tracker's outputs and
turns them into publication-ready kinematic statistics and figures.

It is the downstream member of a three-repository pipeline:

- `LACE <https://github.com/zerotonin/LACE>`_ — the original MATLAB
  tracker accompanying the 2022 paper.
- `pyLACE <https://github.com/zerotonin/pyLACE>`_ — the active Python
  rewrite of the tracker; the upstream producer of the data analysed here.
- **pyLACEpostHoc** (this package) — analysis and plotting downstream of
  the tracker.

Module map
----------

``constants``
    Wong (2011) colourblind-safe palette, figure defaults, the
    ``save_figure`` triple-output helper (SVG + PNG + CSV), arena
    geometry, and shared type aliases.

``config``
    Resolves machine-specific paths (environment variable →
    ``local_paths.json`` → in-repo ``data/``), so no paths are hardcoded.

``data_handlers``
    Readers for the raw inputs: MATLAB LACE result files, video
    (OpenCV / Norpix / image stacks), and Spike2 electrophysiology.

``trace_analysis``
    Per-recording computation: pixel→millimetre trajectories, mid-line
    curvature, swimming speed, spike detection, and interactive
    frame/coordinate calibration.

``fish_data_base``
    Sorts raw recording folders, runs each recording through the
    analysers, and assembles the per-fish CSV-backed database.

``data_base_analyser``
    Cross-animal statistics — the counter-current rheotaxis analyser.

``plotting``
    Publication figures: trace overlays, c-start contour/spike panels and
    animations, and day-wise occupancy heatmaps.

``index_tools``
    Small pure-numpy helpers for turning boolean masks into index runs.

Conventions
-----------

Every module carries type hints and Google-style docstrings, figures use
the Wong palette via ``constants``, and the camelCase identifiers from the
original codebase remain available as deprecation aliases that warn and
forward to their snake_case replacements.
