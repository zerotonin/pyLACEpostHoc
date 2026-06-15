# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.fishExample                         ║
# ║  « mid-line and kinematics demo »                                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  A small example plotting mid-lines and yaw kinematics from one  ║
# ║  MATLAB result file.                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Example: plot mid-lines and yaw kinematics from one MATLAB result file."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plotting.fishPlot as fishPlot
from data_handlers.matLabResultLoader import MatlabResultLoader

FPS = 200


def main(result_mat: Path) -> None:
    """Plot a time-coloured mid-line series and yaw angle/velocity panels."""
    loader = MatlabResultLoader(result_mat)
    (_trace_info, _contour, trace_midline, _head, _tail, trace,
     bendability, *_rest) = loader.get_data()

    fig, ax = plt.subplots(1)
    fishPlot.mid_line_plot(ax, trace_midline, 0, 6000, 1, "cividis", FPS)
    plt.show()

    fig, ax = plt.subplots(3, 1)
    time_ax = fishPlot.make_time_axis(trace.shape[0], FPS, "s")
    ax[0].plot(time_ax, trace[:, 3])
    fishPlot.plot_angle_vel_abs(fig, ax[1], time_ax, np.rad2deg(trace[:, 2]), trace[:, 5], "yaw")
    bend = np.array([np.mean(x[:, 1], axis=0) for x in bendability])
    ax[2].plot(time_ax, bend)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mid-line and kinematics demo.")
    parser.add_argument("result_mat", type=Path, help="a *_result_ana.mat file")
    main(parser.parse_args().result_mat)
