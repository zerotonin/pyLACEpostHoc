# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_data_handlers                        ║
# ║  « loaders: transforms, mode guards, deprecation aliases »      ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np
import pytest

# matLabResultLoader needs only scipy + numpy (both core deps).
from data_handlers.matLabResultLoader import MatlabResultLoader, matLabResultLoader


# ── matLabResultLoader ──────────────────────────────────────────────
def test_ndarray_to_np_array_2d_flips_xy():
    nd = np.empty(2, dtype=object)
    nd[0] = [[1.0, 2.0]]
    nd[1] = [[3.0, 4.0]]
    loader = MatlabResultLoader("dummy.mat")
    np.testing.assert_array_equal(
        loader.ndarray_to_np_array_2d(nd),
        np.array([[2.0, 1.0], [4.0, 3.0]]),  # x and y swapped by fliplr
    )


def test_get_data_rejects_unknown_mode():
    loader = MatlabResultLoader("dummy.mat", mode="nope")
    with pytest.raises(ValueError, match="Unknown mode"):
        loader.get_data()


def test_deprecated_class_alias_warns_and_is_subclass():
    with pytest.warns(DeprecationWarning, match="MatlabResultLoader"):
        loader = matLabResultLoader("dummy.mat")
    assert isinstance(loader, MatlabResultLoader)


def test_deprecated_method_alias_warns():
    loader = MatlabResultLoader("dummy.mat", mode="nope")
    with pytest.warns(DeprecationWarning, match="get_data"):
        with pytest.raises(ValueError):
            loader.getData()


# ── spike2SimpleIO (needs neo + quantities) ─────────────────────────
def test_events_to_bool_signal_maps_to_nearest_sample():
    pytest.importorskip("neo")
    pytest.importorskip("quantities")
    from data_handlers.spike2SimpleIO import SegmentSaver

    saver = SegmentSaver(spike2_reader=None, save_pos="out.csv")
    index = np.array([0.0, 1.0, 2.0, 3.0])
    events = np.array([1.1, 2.9])
    np.testing.assert_array_equal(
        saver.events_to_bool_signal(index, events),
        np.array([False, True, False, True]),
    )
