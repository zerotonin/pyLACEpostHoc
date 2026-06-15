# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — tests/test_index_tools                          ║
# ║  « pure-numpy boolean→index helpers »                            ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import numpy as np

import index_tools


def test_bool2indice_matches_docstring_example():
    """boolean 1,1,1,0,1,1,0,1 -> indices 0,1,2,4,5,7."""
    bools = [1, 1, 1, 0, 1, 1, 0, 1]
    np.testing.assert_array_equal(
        index_tools.bool2indice(bools),
        np.array([0, 1, 2, 4, 5, 7]),
    )


def test_bool2indice_empty():
    assert index_tools.bool2indice([0, 0, 0]).size == 0


def test_indice_seq2start_end_splits_runs():
    indices = [0, 1, 2, 4, 5, 7]
    np.testing.assert_array_equal(
        index_tools.indice_seq2start_end(indices),
        np.array([[0, 2], [4, 5], [7, 7]]),
    )


def test_bool_seq2start_end_indices_end_to_end():
    bools = [1, 1, 1, 0, 1, 1, 0, 1]
    np.testing.assert_array_equal(
        index_tools.bool_Seq2start_end_indices(bools),
        np.array([[0, 2], [4, 5], [7, 7]]),
    )


def test_bracket_bools_grows_isolated_true():
    bools = [0, 0, 1, 0, 0]
    np.testing.assert_array_equal(
        index_tools.bracket_bools(bools),
        np.array([0, 1, 1, 1, 0]),
    )


def test_get_duration_from_start_end():
    start_end = np.array([[0, 2], [4, 5], [7, 7]])
    np.testing.assert_array_equal(
        index_tools.get_duration_from_start_end(start_end),
        np.array([[2], [1], [0]]),
    )
