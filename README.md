# pyLACEpostHoc

Database, post-hoc analysis, and plotting layer for the LACE pose
estimator family (originally Geurten 2022 *Frontiers in Behavioural
Neuroscience*).

This repository holds the analysis code that previously lived in
[pyLACE](https://github.com/zerotonin/pyLACE). pyLACE is now reserved
for the active rewrite of the LACE tracker itself; pyLACEpostHoc
consumes the tracker's outputs (CSV / SQLite / HDF5 + figure triplets)
and produces publication-ready statistics and figures.

## Modules

- `fish_data_base/` — per-fish database (CSV-backed pandas) and
  multi-trial analysis orchestration.
- `trace_analysis/` — kinematic / curvature / spike-detection routines
  on already-extracted traces.
- `plotting/` — publication-grade figures (c-start movies, fish
  overlays, daywise heatmaps).
- `data_base_analyser/` — counter-current rheotaxis statistical analyser.
- `data_handlers/` — media / MATLAB / Spike2 readers.
- `run_scripts/` — experiment-specific entry points.

## Status

Stable but pre-modernisation: still on Python 3.7 (`pylace.yaml`), no
`src/` layout, no test suite. Use as-is; modernisation against the
Geurten Lab coding whitepaper is a separate follow-up.

## License

GPL-3.0-or-later.
