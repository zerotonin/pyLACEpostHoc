# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — plotting.cStartPlotter                          ║
# ║  « c-start contour and spike figures »                           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Builds the combined contour, spike-raster, and parameter figure ║
# ║  (and its animation) for the c-start startle experiments.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Combined contour, spike-raster, and parameter figures for c-start data."""
from __future__ import annotations

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from constants import FIGURE_DPI, WONG
from data_handlers.mediaHandler import MediaHandler
from deprecation import deprecated_class_alias

CONTOUR_XLIM: tuple[int, int] = (0, 280)   # arena width in pixels
CONTOUR_YLIM: tuple[int, int] = (0, 130)   # arena height in pixels
SPIKE_RASTER_XLIM: tuple[int, int] = (0, 5)  # seconds shown in the raster
THRUST_YLIM: tuple[float, float] = (0.0, 2.5)
DEFAULT_NUM_CONTOURS: int = 200
DEFAULT_COMETS_TAIL: int = 25
ANIMATION_FIGSIZE: tuple[float, float] = (4.16, 3.35)
ANIMATION_FPS: int = 25


class CStartPlotter:
    """Plot contours, spike occurrences, and parameters in one figure.

    Used for Garg et al. 2023 A and B.
    """

    def __init__(self) -> None:
        pass

    def create_vertical_axes(self):
        """Create the figure with three stacked axes plus a colour-bar axis."""
        fig = plt.figure()
        gs_main = gridspec.GridSpec(6, 1)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[:3, :], width_ratios=[9, 1], wspace=0.01
        )
        ax1 = plt.subplot(gs_inner[0, 0])   # contours (top 3/6)
        cax1 = plt.subplot(gs_inner[0, 1])  # colour bar beside ax1
        ax2 = plt.subplot(gs_main[3:5, :])  # parameters (middle 2/6)
        ax3 = plt.subplot(gs_main[5, :])    # spike raster (bottom 1/6)
        return fig, (ax1, cax1, ax2, ax3)

    def plot_spike_occurrences(self, spike_df, ax) -> None:
        """Draw each spike time as a short vertical tick on ``ax``."""
        ax.set_xlim(*SPIKE_RASTER_XLIM)
        for spike_time in spike_df["spike_peak_s"]:
            ax.axvline(x=spike_time, ymin=0.45, ymax=0.55, linewidth=1, color=WONG["black"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spike Occurrences")
        ax.set_yticks([])

    def plot_two_parameters(self, fig, ax, time_ax, param1, param2, param1_label,
                            param2_label, x_lim=None) -> None:
        """Plot two parameters against time on twin (log-left) y-axes."""
        if not x_lim:
            x_lim = (time_ax[0], time_ax[-1])

        color = WONG["blue"]
        ax.plot(time_ax, param1, color=color)
        ax.set_xlabel("time, s")
        ax.set_ylabel(param1_label, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_xlim(x_lim)
        ax.set_yscale("log")

        ax2 = ax.twinx()
        color = WONG["sky_blue"]
        ax2.plot(time_ax, param2, color=color)
        ax2.set_ylabel(param2_label, color=color)
        ax2.set_ylim(THRUST_YLIM)
        ax2.tick_params(axis="y", labelcolor=color)

    def plot_contours(self, ax, cax, trace_contour, fps, num_contours=DEFAULT_NUM_CONTOURS,
                      colormap="viridis", alpha=0.5, outline=True, background_image=None,
                      contour_offset=None) -> None:
        """Plot translucent, time-coloured smoothed contours with a colour bar."""
        if background_image is not None:
            ax.imshow(background_image, zorder=0)

        contour_indices = np.linspace(0, len(trace_contour) - 1, num_contours, dtype=int)
        cmap = plt.get_cmap(colormap)

        for i, idx in enumerate(contour_indices):
            contour = np.array(trace_contour[idx])
            num_points = len(contour)
            t = np.linspace(0, 1, num_points)
            new_t = np.linspace(0, 1, num_points * 5)  # upsample for a smooth outline
            offset = contour_offset if contour_offset is not None else (0, 0)
            x_spline = make_interp_spline(t, contour[:, 0], k=3)(new_t) + offset[0]
            y_spline = make_interp_spline(t, contour[:, 1], k=3)(new_t) + offset[1]
            polygon = patches.Polygon(
                np.column_stack((x_spline, y_spline)),
                closed=True,
                facecolor=cmap(i / len(contour_indices)),
                alpha=alpha,
                edgecolor="black" if outline else None,
            )
            ax.add_patch(polygon)

        ax.set_xlim(*CONTOUR_XLIM)
        ax.set_ylim(*CONTOUR_YLIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        norm = plt.Normalize(vmin=0, vmax=(len(trace_contour) - 1) / fps * 1000)
        cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
        cbar.set_label("Time (ms)")

    def create_final_plot(self, spike_df, time_ax, trace, interp_instant_freq, trace_contour,
                          fps, background_image=None):
        """Assemble the static contour/parameter/spike figure."""
        fig, ax_list = self.create_vertical_axes()
        self.plot_spike_occurrences(spike_df, ax_list[3])
        self.plot_two_parameters(
            fig, ax_list[2], time_ax, interp_instant_freq, np.abs(trace[:, 3]),
            "instant. spike frequency, Hz", "thrust, m/s",
        )
        self.plot_contours(
            ax_list[0], ax_list[1], trace_contour, fps,
            num_contours=DEFAULT_NUM_CONTOURS, colormap="viridis", alpha=0.5,
            background_image=background_image,
        )
        return fig, ax_list

    def _render_animation_frame(self, spike_df, time_ax, frame, trace, interp_instant_freq,
                                trace_contour, fps, background_image, contour_offset, x_lim):
        """Render one animation frame to an RGB image array."""
        fig, ax_list = self.create_vertical_axes()
        self.plot_spike_occurrences(spike_df[spike_df["spike_peak_s"] <= time_ax[frame]], ax_list[3])
        for side in ("right", "left", "top"):
            ax_list[3].spines[side].set_visible(False)
        ax_list[3].set_yticks([])
        ax_list[3].set_ylabel("Spikes")

        self.plot_two_parameters(
            fig, ax_list[2], time_ax[:frame + 1], interp_instant_freq[:frame + 1],
            np.abs(trace[:frame + 1, 3]), "instant. spike frequency, Hz", "thrust, m/s",
            x_lim=x_lim,
        )
        ax_list[2].set_xticks([])
        ax_list[2].set_xlabel("")
        ax_list[2].set_xticklabels([])

        self.plot_contours(
            ax_list[0], ax_list[1], trace_contour, fps, num_contours=DEFAULT_NUM_CONTOURS,
            colormap="viridis", alpha=0.025, background_image=background_image,
            contour_offset=contour_offset,
        )

        fig.set_size_inches(*ANIMATION_FIGSIZE)
        fig.set_dpi(FIGURE_DPI)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return image

    def create_animated_plot(self, spike_df, time_ax, trace, interp_instant_freq, trace_contour,
                             fps, path_to_media_file, animation_file_position,
                             comets_tail=DEFAULT_COMETS_TAIL, contour_offset=None,
                             round_robin_offset=None):
        """Render a comet-tail contour animation synced to spikes and thrust."""
        plt.rcParams.update({"font.size": 8})
        frames = []
        media = MediaHandler(path_to_media_file, "movie", fps)

        frame_indices = self.get_frame_indices(time_ax, media.fps)
        video_index = frame_indices
        if round_robin_offset is not None:
            video_index = np.roll(video_index, round_robin_offset)

        x_lim = (time_ax[0], time_ax[-1])
        for frame in tqdm(range(len(time_ax)), desc="Making frames"):
            current_frame_index = frame_indices[frame]
            if current_frame_index < comets_tail:
                continue
            try:
                current_contour = trace_contour[current_frame_index - comets_tail:current_frame_index]
                background_image = media.get_frame(int(video_index[frame]))
                frames.append(self._render_animation_frame(
                    spike_df, time_ax, frame, trace, interp_instant_freq, current_contour,
                    fps, background_image, contour_offset, x_lim,
                ))
            except Exception:
                pass

        animation = cv2.VideoWriter(
            animation_file_position, cv2.VideoWriter_fourcc(*"mp4v"),
            ANIMATION_FPS, frames[0].shape[:2][::-1],
        )
        for frame_image in frames:
            animation.write(cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR))
        animation.release()
        return animation

    def get_frame_indices(self, time_ax, fps) -> np.ndarray:
        """Return the integer video frame index for each point in ``time_ax``."""
        return (time_ax * fps).astype(int)


# Deprecated lower-camelCase class name.
cStartPlotter = deprecated_class_alias(CStartPlotter, "cStartPlotter")
