# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_coverage_extra                       ║
# ║  « assorted edge cases broadening coverage »                     ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import constants
import index_tools
from deprecation import deprecated_alias, deprecated_class_alias
from trace_analysis.CurvatureAnalyser import CurvatureAnalyser
from trace_analysis.speed_analyser import SpeedAnalyser


# ── constants ───────────────────────────────────────────────────────
def test_arena_geometry_lookup():
    arena = constants.ARENAS["counter_current"]
    assert arena.extent == (45.0, 167.0)
    assert arena.name == "counter_current"


def test_semantic_colours_are_from_wong():
    assert set(constants.SEMANTIC.values()) <= set(constants.WONG.values())


# ── config ──────────────────────────────────────────────────────────
def test_active_profile_default_and_env(monkeypatch):
    monkeypatch.delenv(config.PROFILE_ENV, raising=False)
    assert config.active_profile() == "local"
    monkeypatch.setenv(config.PROFILE_ENV, "hpc")
    assert config.active_profile() == "hpc"
    assert config.active_profile("override") == "override"  # explicit wins


# ── deprecation ─────────────────────────────────────────────────────
def test_deprecated_alias_preserves_return():
    def add(a, b):
        return a + b

    with pytest.warns(DeprecationWarning):
        assert deprecated_alias(add, "oldAdd")(2, 3) == 5


def test_deprecated_class_alias_keeps_isinstance():
    class New:
        def __init__(self, x):
            self.x = x

    OldName = deprecated_class_alias(New, "OldName")
    with pytest.warns(DeprecationWarning, match="New"):
        obj = OldName(7)
    assert isinstance(obj, New)
    assert obj.x == 7


# ── index_tools ─────────────────────────────────────────────────────
def test_indices_to_start_end_single_run():
    np.testing.assert_array_equal(
        index_tools.indices_to_start_end([0, 1, 2, 3, 4]), np.array([[0, 4]])
    )


# ── trace_analysis ──────────────────────────────────────────────────
def test_speed_analyser_torque_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        SpeedAnalyser(fps=100).calculate_torque(mode="sideways")


def test_curvature_find_peak_amplitudes():
    analyser = CurvatureAnalyser(pd.DataFrame())
    curvature = np.array([0.0, 1.0, 0.0, 3.0, 0.0])
    amps = analyser.find_peak_amplitudes(curvature, prominence_threshold=0.5)
    np.testing.assert_array_equal(np.sort(amps), [1.0, 3.0])
