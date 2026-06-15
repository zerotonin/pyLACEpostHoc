# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — index_tools                                     ║
# ║  « boolean signals to index runs »                               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Helpers that turn a boolean mask into the indices of its True   ║
# ║  runs and back into (start, end) bracket pairs.                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Convert boolean signals into index runs and (start, end) bracket pairs."""
from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from deprecation import deprecated_alias


def bool_to_indices(bool_signal: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Return the indices of the True entries in a boolean signal.

    Example::

        boolean array  1, 1, 1, 0, 1, 1, 0, 1
        output         0, 1, 2,    4, 5,    7

    Args:
        bool_signal: Sequence of truthy/falsy values.

    Returns:
        Integer array of the indices where the signal is True.
    """
    return np.array([i for i, x in enumerate(bool_signal) if x])


def indices_to_start_end(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    """Collapse a sorted index list into (start, end) pairs of runs.

    Consecutive indices form one run; a gap starts a new run.

    Args:
        indices: Sorted integer indices (e.g. from :func:`bool_to_indices`).

    Returns:
        ``(n_runs, 2)`` array of inclusive (start, end) index pairs.
    """
    index_diff = np.diff(indices)
    starts = [indices[0]]
    ends: list[int] = []
    for i in range(1, len(indices)):
        if index_diff[i - 1] != 1:
            ends.append(indices[i - 1])
            starts.append(indices[i])
    ends.append(indices[-1])
    return np.array(list(zip(starts, ends)))


def bool_to_start_end_indices(bool_signal: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Return inclusive (start, end) pairs for each True run of a mask."""
    indices = bool_to_indices(bool_signal)
    return indices_to_start_end(indices)


def bracket_bools(bool_signal: Sequence[bool]) -> Sequence[bool]:
    """Grow every isolated True so its immediate neighbours are also True.

    Args:
        bool_signal: Sequence of truthy/falsy values (returned type matches).

    Returns:
        A copy with each True's neighbours set True.
    """
    bracketed = copy.deepcopy(bool_signal)
    for i in range(1, len(bracketed) - 1):
        if bool_signal[i] == 1:
            bracketed[i - 1], bracketed[i + 1] = (True, True)
    return bracketed


def bracket_start_end_of_sequence(start_end_indices: np.ndarray, seq_len: int) -> np.ndarray:
    """Widen each (start, end) run by one frame on both sides, within bounds.

    A run is only widened when it neither starts at 0 nor ends before
    ``seq_len``, so the brackets never run off the recording.

    Args:
        start_end_indices: ``(n_runs, 2)`` array of (start, end) pairs.
        seq_len:           Length the end must reach to be widened.

    Returns:
        A copy of the array with eligible runs widened by one frame.
    """
    widened = copy.deepcopy(start_end_indices)
    for run in range(widened.shape[0]):
        if widened[run, 0] != 0 and widened[run, 1] >= seq_len:
            widened[run, 0] = widened[run, 0] - 1
            widened[run, 1] = widened[run, 1] + 1
    return widened


def get_duration_from_start_end(start_end_indices: np.ndarray) -> np.ndarray:
    """Return the per-run duration as ``end - start`` for each (start, end) pair."""
    return np.diff(start_end_indices)


# ─────────────────────────────────────────────────────────────────
#  Deprecated camelCase aliases  « removed in a future release »
# ─────────────────────────────────────────────────────────────────
bool2indice = deprecated_alias(bool_to_indices, "bool2indice")
indice_seq2start_end = deprecated_alias(indices_to_start_end, "indice_seq2start_end")
bool_Seq2start_end_indices = deprecated_alias(
    bool_to_start_end_indices, "bool_Seq2start_end_indices"
)
bracket_starts_end_of_sequence = deprecated_alias(
    bracket_start_end_of_sequence, "bracket_starts_end_of_sequence"
)
