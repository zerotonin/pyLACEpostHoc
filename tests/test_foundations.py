# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_foundations                          ║
# ║  « constants, config resolver, deprecation shim »               ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
import constants  # noqa: E402
from deprecation import deprecated_alias  # noqa: E402


# ── constants ───────────────────────────────────────────────────────
def test_wong_palette_has_eight_colours():
    assert len(constants.WONG) == 8
    assert all(v.startswith("#") for v in constants.WONG.values())


def test_save_figure_writes_triple(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    written = constants.save_figure(fig, "demo", tmp_path, data=data)
    plt.close(fig)
    assert written["svg"].exists()
    assert written["png"].exists()
    assert written["csv"].exists()


def test_save_figure_skips_csv_without_data(tmp_path):
    fig, _ = plt.subplots()
    written = constants.save_figure(fig, "nodata", tmp_path)
    plt.close(fig)
    assert "csv" not in written


# ── config ──────────────────────────────────────────────────────────
@pytest.fixture
def paths_file(tmp_path):
    pf = tmp_path / "local_paths.json"
    pf.write_text(
        json.dumps({"local": {"data_root": "/data/local"}, "hpc": {"data_root": "/scratch"}}),
        encoding="utf-8",
    )
    return pf


def test_get_path_from_profile(paths_file):
    # Compare Path objects, not strings: separators differ across OSes.
    assert config.get_path("data_root", profile="hpc", paths_file=paths_file) == Path("/scratch")


def test_env_var_overrides_file(paths_file, monkeypatch):
    monkeypatch.setenv("PYLACE_POSTHOC_DATA_ROOT", "/override")
    assert config.get_path("data_root", paths_file=paths_file) == Path("/override")


def test_missing_file_names_template(tmp_path):
    with pytest.raises(config.PathConfigError, match="local_paths.template.json"):
        config.get_path("data_root", paths_file=tmp_path / "absent.json")


def test_unknown_key_raises(paths_file):
    with pytest.raises(config.PathConfigError, match="nope"):
        config.get_path("nope", profile="local", paths_file=paths_file)


def test_export_env(paths_file):
    assert config.export_env(profile="local", paths_file=paths_file) == (
        "PYLACE_POSTHOC_DATA_ROOT=/data/local"
    )


# ── deprecation ─────────────────────────────────────────────────────
def test_deprecated_alias_warns_and_forwards():
    def compute_thing(a, b):
        return a + b

    old = deprecated_alias(compute_thing)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = old(2, 3)
    assert result == 5
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "compute_thing" in str(caught[-1].message)
