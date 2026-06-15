# pyLACEpostHoc

[![tests](https://github.com/zerotonin/pyLACEpostHoc/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/pyLACEpostHoc/actions/workflows/tests.yml)
[![docs](https://github.com/zerotonin/pyLACEpostHoc/actions/workflows/docs.yml/badge.svg)](https://github.com/zerotonin/pyLACEpostHoc/actions/workflows/docs.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![DOI](https://img.shields.io/badge/DOI-10.3389%2Ffnbeh.2022.819146-blue.svg)](https://doi.org/10.3389/fnbeh.2022.819146)

Database, post-hoc analysis, and plotting layer for the LACE pose
estimator family (originally Geurten 2022 *Frontiers in Behavioural
Neuroscience*).

This repository holds the analysis code that previously lived in
[pyLACE](https://github.com/zerotonin/pyLACE). pyLACE is now reserved
for the active rewrite of the LACE tracker itself; pyLACEpostHoc
consumes the tracker's outputs (CSV / SQLite / HDF5 + figure triplets)
and produces publication-ready statistics and figures.

## Related repositories

- [**LACE**](https://github.com/zerotonin/LACE) — the original tracker
  accompanying Geurten 2022.
- [**pyLACE**](https://github.com/zerotonin/pyLACE) — the active Python
  rewrite of the LACE tracker; the upstream producer of the outputs
  this package analyses.
- **pyLACEpostHoc** (this repo) — the database, post-hoc analysis, and
  plotting layer downstream of the tracker.

## Installation

The core install is light (numpy / pandas / scipy / matplotlib /
seaborn). Raw-data readers and survival statistics ship as optional
extras so they never block a basic install.

```bash
# Core only (pure-Python helpers, plotting stack)
pip install -e .

# With raw-data readers (video, MATLAB, Spike2, HDF5/parquet)
pip install -e ".[io]"

# With survival / permutation statistics
pip install -e ".[stats]"

# Everything needed to run the full pipelines
pip install -e ".[full]"

# Test + lint tooling
pip install -e ".[dev]"
```

Conda users can build the full environment directly:

```bash
conda env create -f environment.yml
conda activate pylaceposthoc
```

### Configure machine-specific paths (step one)

Data roots are **not** hardcoded. Before running anything, copy the
template and fill in the real paths for your machine:

```bash
cp local_paths.template.json local_paths.json   # local_paths.json is gitignored
$EDITOR local_paths.json                          # set data_root, database_path, figure_root
```

Paths resolve in three steps: a `PYLACE_POSTHOC_<KEY>` environment
variable wins, then the active profile in `local_paths.json` (`local`
by default, or `hpc`; select with `PYLACE_POSTHOC_PROFILE`). Shell and
SLURM consumers can source the values via `python config.py --export`.
A missing `local_paths.json` fails loudly and names the template to
copy.

## Modules

- `index_tools.py` — boolean-sequence → index helpers (pure numpy).
- `fish_data_base/` — per-fish database (CSV-backed pandas) and
  multi-trial analysis orchestration.
- `trace_analysis/` — kinematic / curvature / spike-detection routines
  on already-extracted traces.
- `plotting/` — publication-grade figures (c-start movies, fish
  overlays, daywise heatmaps).
- `data_base_analyser/` — counter-current rheotaxis statistical analyser.
- `data_handlers/` — media / MATLAB / Spike2 readers.
- `run_scripts/` — experiment-specific entry points (not packaged).

## Development

```bash
pip install -e ".[dev]"
pytest                      # unit tests
ruff check tests index_tools.py constants.py config.py deprecation.py data_handlers trace_analysis
```

Tests, docs, and releases are automated via GitHub Actions
(`.github/workflows/`). Docs build with Sphinx + Furo and deploy to
GitHub Pages on push to `master`; tagging `vX.Y.Z` cuts a release.

## Status

Stable but mid-modernisation. The package is now pip-installable with a
flat-layout (`pyproject.toml` + `setuptools-scm`), a unit-test suite,
and CI/CD. The legacy research modules still predate the Geurten Lab
coding whitepaper (Python 3.7-era style); they are linted and
hardened incrementally — `index_tools.py` is the first fully
modernised module.

## Citation

If you use this software, please cite the LACE paper:

> Garg, V., André, S., Giraldo, D., Heyer, L., Göpfert, M. C., Dosch,
> R., & Geurten, B. R. H. (2022). A markerless pose estimator applicable
> to limbless animals. *Frontiers in Behavioral Neuroscience*, 16,
> 819146. <https://doi.org/10.3389/fnbeh.2022.819146>

Machine-readable metadata (and GitHub's "Cite this repository" button)
is in [`CITATION.cff`](CITATION.cff), which carries this article as its
`preferred-citation`.

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
