# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — trace_analysis.CurvatureAnalyser                ║
# ║  « total mid-line curvature peaks »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Computes per-frame total mid-line curvature and summarises its  ║
# ║  peak amplitudes (median, mean, maximum).                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Total mid-line curvature per frame and its peak-amplitude statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class CurvatureAnalyser:
    """Analyse the curvature of a fish mid-line from a coordinate DataFrame.

    Args:
        midline_df: DataFrame with ``x_coord_i`` / ``y_coord_i`` columns.
    """

    def __init__(self, midline_df: pd.DataFrame) -> None:
        self.midline_df = midline_df

    def calculate_total_curvature(self, number_of_coordinates: int = 10) -> np.ndarray:
        """Return the total mid-line curvature for every frame (row).

        Vectorised over all frames: the turning between successive unit
        tangents is summed along the body for each row.

        Args:
            number_of_coordinates: Number of mid-line points per frame.

        Returns:
            1D array of total curvature, one value per frame.
        """
        x = self.midline_df[[f"x_coord_{i}" for i in range(number_of_coordinates)]].to_numpy()
        y = self.midline_df[[f"y_coord_{i}" for i in range(number_of_coordinates)]].to_numpy()
        points = np.stack([x, y], axis=-1)                       # (n_frames, n_pts, 2)
        tangents = np.diff(points, axis=1)                       # (n_frames, n_pts-1, 2)
        unit_tangents = tangents / np.linalg.norm(tangents, axis=2, keepdims=True)
        tangent_change = np.diff(unit_tangents, axis=1)          # (n_frames, n_pts-2, 2)
        magnitudes = np.linalg.norm(tangent_change, axis=2)      # (n_frames, n_pts-2)
        return magnitudes.sum(axis=1)

    def find_peak_amplitudes(
        self, curvature_vector: np.ndarray, prominence_threshold: float
    ) -> np.ndarray:
        """Return the amplitudes of curvature peaks above a prominence."""
        peaks, _ = find_peaks(curvature_vector, prominence=prominence_threshold)
        return curvature_vector[peaks]

    def get_total_curvature_amps(self, prominence_threshold: float = 0.5) -> dict[str, float]:
        """Summarise total-curvature peak amplitudes.

        Args:
            prominence_threshold: Minimum prominence for a peak to count.

        Returns:
            Dict with median, mean, max peak amplitude and the max index.
        """
        total_curv = self.calculate_total_curvature()
        total_curv_amps = self.find_peak_amplitudes(total_curv, prominence_threshold)
        return {
            "median_curv_amp": np.nanmedian(total_curv_amps),
            "mean_curv_amp": np.nanmean(total_curv_amps),
            "max_curv_amp": np.nanmax(total_curv_amps),
            "max_curv_amp_index": np.nanargmax(total_curv_amps),
        }
