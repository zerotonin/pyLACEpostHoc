# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — trace_analysis.speed_analyser                   ║
# ║  « thrust, slip, and yaw speed metrics »                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Derives activity, cruise speed, torque, and central speed values ║
# ║  from a millimetre trace DataFrame.                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Speed, activity, cruise, and torque metrics from a fish trace DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ACTIVITY_THRESHOLD: tuple[float, float, float] = (0.025, 0.025, 100.0)


class SpeedAnalyser:
    """Analyse thrust/slip/yaw speed of a fish from a trace DataFrame.

    Args:
        fps:       Frames per second of the trace.
        dataframe: Trace data with ``thrust_m/s``, ``slip_m/s``, ``yaw_deg/s``.
    """

    def __init__(self, fps: float, dataframe: pd.DataFrame | None = None) -> None:
        self.fps = fps
        self.trace_df = dataframe if dataframe is not None else pd.DataFrame()
        self.all_speed = pd.DataFrame()
        self.speed_analysis_df = pd.DataFrame()
        self.activity = pd.Series(dtype=bool)
        self.cruise_speed = pd.DataFrame()

    def extract_fish_speeds(self) -> None:
        """Extract the thrust, slip, and yaw speed columns from the trace."""
        self.all_speed = self.trace_df[["thrust_m/s", "slip_m/s", "yaw_deg/s"]]

    def set_activity_array(
        self, activity_threshold: tuple[float, float, float] = DEFAULT_ACTIVITY_THRESHOLD
    ) -> None:
        """Flag frames where any of thrust, slip, or yaw exceeds its threshold.

        Args:
            activity_threshold: Thresholds for thrust, slip, and yaw.
        """
        self.activity = pd.DataFrame(
            [
                self.all_speed["thrust_m/s"].abs() > activity_threshold[0],
                self.all_speed["slip_m/s"].abs() > activity_threshold[1],
                self.all_speed["yaw_deg/s"].abs() > activity_threshold[2],
            ]
        ).transpose().any(axis="columns")

    def extract_cruise_speed(self) -> None:
        """Keep only the active frames as cruise speed."""
        self.cruise_speed = self.all_speed[self.activity]

    def calculate_torque(self, mode: str = "cruise") -> float:
        """Return median (abs thrust + abs slip) / abs yaw over the chosen frames.

        Args:
            mode: ``"cruise"`` (active frames) or ``"all"`` (every frame).

        Raises:
            ValueError: If ``mode`` is neither ``"cruise"`` nor ``"all"``.
        """
        if mode == "cruise":
            torque_data = self.cruise_speed
        elif mode == "all":
            torque_data = self.all_speed
        else:
            raise ValueError(f"calculate_torque: unknown mode: {mode}")

        return np.median(
            (torque_data["thrust_m/s"].abs() + torque_data["slip_m/s"].abs())
            / torque_data["yaw_deg/s"].abs()
        )

    def calculate_central_speed_values(self, speed_df: pd.DataFrame) -> list[float]:
        """Return mean then median of absolute thrust, slip, yaw for a frame set."""
        data = speed_df.abs().mean().tolist()
        data += speed_df.abs().median().tolist()
        return data

    def analyse_fish_speed(self) -> dict[str, float]:
        """Run the full speed analysis and return the metric dictionary."""
        self.extract_fish_speeds()
        self.set_activity_array()
        self.extract_cruise_speed()

        data = self.calculate_central_speed_values(self.all_speed)
        data += self.calculate_central_speed_values(self.cruise_speed)

        data.append(self.activity.sum() / self.fps)
        data.append(self.activity.sum() / self.activity.shape[0])
        data.append(self.activity[::-1].idxmax() / self.fps)
        data.append(self.calculate_torque())

        keys = [
            "thrust_mean_m/s", "slip_mean_m/s", "yaw_mean_m/s",
            "thrust_median_m/s", "slip_median_m/s", "yaw_median_m/s",
            "cruising_thrust_mean_m/s", "cruising_slip_mean_m/s", "cruising_yaw_mean_m/s",
            "cruising_thrust_median_m/s", "cruising_slip_median_m/s", "cruising_yaw_median_m/s",
            "activity_duration_s", "activity_fraction", "sec_to_first_stop", "torque",
        ]
        return dict(zip(keys, data))
