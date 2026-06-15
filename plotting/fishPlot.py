# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — plotting.fishPlot                               ║
# ║  « trace overlays and kinematic plots »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Plot helpers for frame overlays, spatial histograms, mid-line   ║
# ║  time series, colour bars, and angle/velocity panels.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Matplotlib helpers for fish trace overlays and kinematic figures."""
from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from constants import WONG
from deprecation import deprecated_alias


def frame_overlay(ax: Axes, frame, contour, mid_line, head, tail, box_coords,
                  frame_cmap: str = "gray") -> None:
    """Show ``frame`` and overlay the contour, mid-line, head, tail, and box."""
    ax.imshow(frame, cmap=frame_cmap)
    plot_trace_result(ax, contour, mid_line, head, tail, box_coords)


def plot_trace_result(ax: Axes, contour, mid_line, head, tail, box_coords) -> None:
    """Overlay the trace result (mid-line, contour, head, tail, box) on ``ax``."""
    ax.plot(mid_line[:, 0], mid_line[:, 1], ".-", color=WONG["bluish_green"])
    ax.plot(contour[:, 0], contour[:, 1], "-", color=WONG["yellow"])
    ax.plot(head[0], head[1], "o", color=WONG["blue"])
    ax.plot(tail[0], tail[1], "s", color=WONG["blue"])
    if box_coords is not None:
        ax.plot(box_coords[:, 0], box_coords[:, 1], "-", color=WONG["yellow"])
        ax.plot(box_coords[[0, -1], 0], box_coords[[0, -1], 1], "-", color=WONG["yellow"])


def simple_spatial_hist(ax: Axes, prob_density, cmap: str = "PuBuGn") -> None:
    """Show a probability-density occupancy map on ``ax``."""
    ax.imshow(prob_density, origin="lower", interpolation="gaussian", cmap=cmap)


def seaborn_spatial_hist(mid_line) -> None:
    """Plot a mid-line occupancy density as a seaborn joint KDE with marginals."""
    all_mid_line = np.vstack((mid_line[:]))
    df = pd.DataFrame(
        data={"x-coordinate, mm": all_mid_line[:, 0], "y-coordinate, mm": all_mid_line[:, 1]}
    )
    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=1.66666, light=1, as_cmap=True)
    grid = sns.JointGrid(data=df, x="x-coordinate, mm", y="y-coordinate, mm", space=0)
    grid.plot_joint(sns.kdeplot, fill=True, cmap=cmap)
    grid.ax_joint.set_aspect("equal")
    grid.plot_marginals(sns.histplot, color=WONG["bluish_green"], alpha=0.75, bins=25)


def add_color_bar(ax: Axes, cmap, vmin: float, vmax: float, orientation: str,
                  axis_label: str) -> None:
    """Add a horizontal (``'h'``) or vertical (``'v'``) colour bar to ``ax``."""
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    if orientation == "h":
        cbar = plt.colorbar(sm, orientation="horizontal", ax=ax)
        cbar.ax.set_xlabel(axis_label, rotation=0)
    if orientation == "v":
        cbar = plt.colorbar(sm, orientation="vertical", ax=ax)
        cbar.ax.set_xlabel(axis_label, rotation=90)


def mid_line_plot(ax: Axes, trace_mid_line, start: int, stop: int, step: int,
                  colormap: str, fps: float) -> None:
    """Plot every ``step``-th mid-line, coloured by time, with a colour bar."""
    cmap = plt.get_cmap(colormap)
    for i in range(start, stop):
        if i % step == 0:
            mid_line = trace_mid_line[i]
            c = (i - start) / (stop - start)
            ax.plot(mid_line[:, 0], mid_line[:, 1], ".-", color=cmap(c))
            ax.plot(mid_line[-1, 0], mid_line[-1, 1], "k.")
    add_color_bar(ax, cmap, 0, (stop - start) / fps, "h", "time, s")
    plt.gca().set_aspect("equal", adjustable="box")


def make_time_axis(length: int, fps: float, unit: str = "s") -> np.ndarray:
    """Return a time axis of ``length`` samples in s, ms, min, or h.

    Raises:
        ValueError: If ``unit`` is not one of ``s``, ``ms``, ``min``, ``h``.
    """
    time_s = np.linspace(0, length / fps, length)
    factors = {"s": 1.0, "ms": 1000.0, "min": 1 / 60, "h": 1 / 3600}
    if unit not in factors:
        raise ValueError(f"make_time_axis: unknown unit: {unit}")
    return time_s * factors[unit]


def plot_angle_vel_abs(fig: Figure, ax: Axes, time_ax, angle_deg, vel_deg_s,
                       angle_str: str) -> None:
    """Plot an angle and its angular velocity on twin y-axes."""
    color = WONG["blue"]
    ax.set_xlabel("time, s")
    ax.set_ylabel(f"{angle_str} angle, deg", color=color)
    ax.plot(time_ax, angle_deg, color=color)
    ax.tick_params(axis="y", labelcolor=color)

    ax2 = ax.twinx()
    color = WONG["sky_blue"]
    ax2.set_ylabel(f"{angle_str} velocity, deg*s-1", color=color)
    ax2.plot(time_ax, vel_deg_s, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout()


# Deprecated camelCase function names.
frameOverlay = deprecated_alias(frame_overlay, "frameOverlay")
plotTraceResult = deprecated_alias(plot_trace_result, "plotTraceResult")
simpleSpatialHist = deprecated_alias(simple_spatial_hist, "simpleSpatialHist")
seabornSpatialHist = deprecated_alias(seaborn_spatial_hist, "seabornSpatialHist")
addColorBar = deprecated_alias(add_color_bar, "addColorBar")
midLinePlot = deprecated_alias(mid_line_plot, "midLinePlot")
makeTimeAxis = deprecated_alias(make_time_axis, "makeTimeAxis")
plotAngleVelAbs = deprecated_alias(plot_angle_vel_abs, "plotAngleVelAbs")
