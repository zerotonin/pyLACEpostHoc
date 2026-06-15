# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_trace_analysis                       ║
# ║  « curvature, speed, spikes, body length »                       ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trace_analysis.CurvatureAnalyser import CurvatureAnalyser
from trace_analysis.speed_analyser import SpeedAnalyser
from trace_analysis.SpikeDetector import SpikeDetector
from trace_analysis.traceAnalyser import TraceAnalyser


# ── CurvatureAnalyser ───────────────────────────────────────────────
def _curvature_reference(df, k):
    """Original iterrows implementation, kept as an oracle for the vector one."""
    out = []
    for _, row in df.iterrows():
        pts = np.array([(row[f"x_coord_{i}"], row[f"y_coord_{i}"]) for i in range(k)])
        tan = np.diff(pts, axis=0)
        unit = tan / np.linalg.norm(tan, axis=1, keepdims=True)
        out.append(np.linalg.norm(np.diff(unit, axis=0), axis=1).sum())
    return np.array(out)


def test_straight_midline_has_zero_curvature():
    df = pd.DataFrame({f"x_coord_{i}": [float(i)] for i in range(3)}
                      | {f"y_coord_{i}": [0.0] for i in range(3)})
    np.testing.assert_allclose(
        CurvatureAnalyser(df).calculate_total_curvature(number_of_coordinates=3), [0.0]
    )


def test_vectorised_curvature_matches_reference():
    rng = np.random.default_rng(0)
    k = 10
    data = {f"x_coord_{i}": rng.normal(size=5) for i in range(k)}
    data |= {f"y_coord_{i}": rng.normal(size=5) for i in range(k)}
    df = pd.DataFrame(data)
    np.testing.assert_allclose(
        CurvatureAnalyser(df).calculate_total_curvature(number_of_coordinates=k),
        _curvature_reference(df, k),
    )


# ── SpeedAnalyser ───────────────────────────────────────────────────
def test_analyse_fish_speed_returns_expected_keys():
    df = pd.DataFrame({
        "thrust_m/s": [0.0, 0.1, 0.0, 0.2],
        "slip_m/s": [0.0, 0.05, 0.0, 0.0],
        "yaw_deg/s": [0.0, 200.0, 0.0, 150.0],
    })
    result = SpeedAnalyser(fps=100, dataframe=df).analyse_fish_speed()
    assert result["activity_duration_s"] == pytest.approx(2 / 100)
    assert result["activity_fraction"] == pytest.approx(0.5)
    assert "torque" in result


# ── SpikeDetector ───────────────────────────────────────────────────
def test_instantaneous_freq_is_inverse_isi():
    df = pd.DataFrame({"spike_peak_s": [0.0, 0.5, 1.0]})
    np.testing.assert_allclose(
        SpikeDetector.calculate_instantaneous_spike_freq(df), [2.0, 2.0]
    )


def test_separate_m_units_alias_warns_and_tags():
    sd = SpikeDetector(pd.DataFrame({"Signal stream 0": [0.0]}))
    sd.spike_train_df = pd.DataFrame({"amplitude_muV": [1.0, 1.0, 1.0, 100.0]})
    with pytest.warns(DeprecationWarning, match="separate_m_units"):
        sd.separate_M_units()
    assert list(sd.spike_train_df.spike_category) == ["Other", "Other", "Other", "Mauthner"]


# ── TraceAnalyser (leaf method without full construction) ────────────
def test_body_length_of_unit_steps():
    analyser = TraceAnalyser.__new__(TraceAnalyser)  # bypass heavy __init__
    mid_line = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    body_len, body_axis = analyser.calculate_body_length(mid_line)
    assert body_len == pytest.approx(2.0)
    np.testing.assert_allclose(body_axis, [0.0, 1.0, 2.0])
