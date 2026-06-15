# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_plotting                             ║
# ║  « time axis, histogram helpers, deprecation »                   ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import plotting.fishPlot as fishPlot  # noqa: E402
from plotting.DaywiseAnalysis import DaywiseAnalysis  # noqa: E402


# ── fishPlot.make_time_axis ─────────────────────────────────────────
def test_make_time_axis_units():
    np.testing.assert_allclose(fishPlot.make_time_axis(5, 10, "s"),
                               [0, 0.125, 0.25, 0.375, 0.5])
    np.testing.assert_allclose(fishPlot.make_time_axis(5, 10, "ms"),
                               [0, 125, 250, 375, 500])


def test_make_time_axis_unknown_unit():
    with pytest.raises(ValueError, match="unknown unit"):
        fishPlot.make_time_axis(5, 10, "fortnight")


def test_make_time_axis_deprecated_alias_warns():
    with pytest.warns(DeprecationWarning, match="make_time_axis"):
        fishPlot.makeTimeAxis(3, 1)


# ── DaywiseAnalysis pure helpers (bypassing heavy __init__) ─────────
def _bare_daywise():
    return DaywiseAnalysis.__new__(DaywiseAnalysis)


def test_normalise_histograms_each_day_sums_to_one():
    hist = np.ones((3, 2, 2))
    normed = _bare_daywise().normalise_histograms(hist)
    np.testing.assert_allclose(normed.sum(axis=(1, 2)), [1, 1, 1])


def test_normalise_histograms_rejects_non_3d():
    with pytest.raises(ValueError, match="3D numpy array"):
        _bare_daywise().normalise_histograms(np.ones((2, 2)))


def test_adjust_histogram_shape_trims_and_pads():
    analyser = _bare_daywise()
    hist = np.ones((3, 2, 2))
    assert analyser.adjust_histogram_shape(hist, 2).shape == (2, 2, 2)
    padded = analyser.adjust_histogram_shape(hist, 5)
    assert padded.shape == (5, 2, 2)
    assert np.isnan(padded[0]).all()  # NaN padding goes in front


def test_extract_fishid_tanknumber_parses_folder():
    path = "/data/tankNum_7__fishID_AB12/day.npy"
    assert _bare_daywise().extract_fishID_tanknumber(path) == (7, "AB12")
