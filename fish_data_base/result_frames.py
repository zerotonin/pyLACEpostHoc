# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — fish_data_base.result_frames                    ║
# ║  « trace arrays to tidy DataFrames »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Pure helpers that turn analysed trace arrays into the 2D/3D and ║
# ║  histogram DataFrames written to the per-fish database.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build the tidy result DataFrames written to the per-fish database."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Counter-current arena axes in millimetres (stream length, orthogonal width).
HIST_STREAM_MM: int = 162
HIST_ORTHO_MM: int = 43


def prep_df_3d(col1_name: str, col2_name: str, reps: int) -> tuple[pd.DataFrame, list[str]]:
    """Return an empty frame and the interleaved ``name_i`` column labels."""
    indices = np.linspace(0, reps - 1, reps, dtype=int)
    labels = [[f"{col1_name}_{i}", f"{col2_name}_{i}"] for i in indices]
    labels = [name for pair in labels for name in pair]
    return pd.DataFrame([], columns=labels), labels


def to_time_index(data_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Convert a frame-based index to seconds in place and return the frame."""
    data_df.index = data_df.index / fps
    data_df.index.name = "time sec"
    return data_df


def make_df_3d(
    data: list, col1_name: str, col2_name: str, fps: float | None = None, index: str | None = None
) -> pd.DataFrame:
    """Stack per-detection arrays into a wide ``name_i`` DataFrame."""
    reps = max(len(entry) for entry in data)
    _, labels = prep_df_3d(col1_name, col2_name, reps)
    rows = [dict(zip(labels, detection.flatten())) for detection in data]
    data_df = pd.DataFrame(rows, columns=labels)
    if index == "Time":
        data_df = to_time_index(data_df, fps)
    return data_df


def make_df_2d(
    data, col1_name: str, col2_name: str, fps: float | None = None, index: str | None = None
) -> pd.DataFrame:
    """Build a two-column DataFrame, optionally with a time index."""
    data_df = pd.DataFrame(data, columns=[col1_name, col2_name])
    if index == "Time":
        data_df = to_time_index(data_df, fps)
    return data_df


def make_df_hist(data: np.ndarray) -> pd.DataFrame:
    """Build the occupancy-histogram DataFrame with a millimetre stream axis."""
    stream_axis = np.linspace(0, HIST_STREAM_MM, data.shape[1] + 2)
    ortho_axis = np.linspace(0, HIST_ORTHO_MM, data.shape[0] + 2)
    labels = ["streamAxisMM " + str(int(x)) for x in stream_axis[1:-1]]
    labels = ["orthoIndexMM"] + labels
    data = np.vstack((ortho_axis[1:-1], data.T)).T
    return pd.DataFrame(data, columns=labels)
