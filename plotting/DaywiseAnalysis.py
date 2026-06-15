# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — plotting.DaywiseAnalysis                        ║
# ║  « day-by-day occupancy heatmaps »                               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads per-day occupancy histograms, splits them by sex, and     ║
# ║  plots day-wise heatmap grids and box/strip summaries.           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Day-wise occupancy heatmaps and box/strip summaries of fish movement."""
from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from constants import WONG

# Two-colour Wong palette for the male/female hue.
SEX_PALETTE: list[str] = [WONG["sky_blue"], WONG["orange"]]


class DaywiseAnalysis:
    """Plot day-wise occupancy heatmaps and box/strip summaries.

    Args:
        df:               Day-wise analysis metadata (sex, IDs, metrics).
        parent_directory: Directory tree holding the per-fish ``.npy`` histograms.
    """

    def __init__(self, df: pd.DataFrame, parent_directory: str | Path) -> None:
        self.df = df
        self.parent_directory = parent_directory
        self.histogram_file_positions = self.find_npy_files(self.parent_directory)
        self.fishID, self.hists = self.load_normed_histograms(self.histogram_file_positions)

    def get_day_data(self, data_3d: np.ndarray, day: int) -> np.ndarray:
        """Return the 2D slice for a given 1-based day."""
        return data_3d[day - 1]

    def plot_histogram(self, ax, data: np.ndarray, cmap, norm, day: int) -> None:
        """Plot one smoothed day heatmap on ``ax``."""
        data_smooth = scipy.ndimage.zoom(data, 3)
        sns.heatmap(data=data_smooth, cmap=cmap, norm=norm, ax=ax)
        ax.set_title(f"Day {day}")
        ax.set_axis_off()

    def create_daywise_histograms(self, data_3d: np.ndarray) -> tuple[Figure, Figure]:
        """Build a 4x6 grid of day heatmaps plus a standalone colour-bar figure."""
        vmin, vmax = np.nanmin(data_3d[data_3d > 0]), np.nanmax(data_3d)
        cmap = "viridis"
        norm = LogNorm(vmin=vmin if vmin > 0 else 0.01, vmax=vmax)

        # Scope the dark style so it does not leak into the caller's later plots.
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(4, 6, figsize=(24, 16), sharex=True, sharey=True,
                                     facecolor="white")
            axes = axes.flatten()
            for day, ax in enumerate(axes):
                if day < data_3d.shape[0]:
                    self.plot_histogram(ax, data_3d[day], cmap, norm, day + 1)
                    if day == 18:
                        ax.set_xlabel("X (cm)")
                        ax.set_ylabel("Y (cm)")
                else:
                    ax.axis("off")

            fig_cbar = plt.figure(figsize=(3, 8), facecolor="white")
            cbar_ax = fig_cbar.add_axes([0.1, 0.2, 0.3, 0.6])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=14, colors="black")
        return fig, fig_cbar

    def create_vertical_box_stripplot(self, x_col: str, y_col: str, hue_col: str | None = None,
                                      hue_order=None) -> Figure:
        """Draw a box + strip plot of ``y_col`` by ``x_col``, split by ``hue_col``."""
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x=x_col, y=y_col, hue=hue_col, hue_order=hue_order, data=self.df,
                    whis=[0, 100], width=0.6, palette=SEX_PALETTE)
        sns.stripplot(x=x_col, y=y_col, hue=hue_col, hue_order=hue_order, data=self.df,
                      size=4, color=".3", linewidth=0)
        ax.yaxis.grid(True)
        ax.set(xlabel="")
        sns.despine(trim=True, left=True)
        return fig

    def find_npy_files(self, directory: str | Path) -> list[Path]:
        """Return every ``.npy`` file under ``directory`` (recursively)."""
        return sorted(Path(directory).rglob("*.npy"))

    def extract_fishID_tanknumber(self, file_path: str | Path) -> tuple[int, str]:
        """Parse (tank number, fish ID) from a histogram file's parent folder."""
        parts = Path(file_path).parent.name.split("__")
        tanknumber = int(parts[0].replace("tankNum_", ""))
        fish_id = parts[1].replace("fishID_", "")
        return tanknumber, fish_id

    def load_npy_file(self, file_path: str | Path) -> np.ndarray:
        """Load a ``.npy`` histogram file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"No .npy file found at {path}")
        return np.load(path)

    def normalise_histograms(self, histogram: np.ndarray) -> np.ndarray:
        """Normalise each 2D day slice of a 3D histogram to sum to one.

        Raises:
            ValueError: If ``histogram`` is not a 3D array.
        """
        if not isinstance(histogram, np.ndarray) or histogram.ndim != 3:
            raise ValueError("The input histogram should be a 3D numpy array.")
        return histogram / histogram.sum(axis=(1, 2), keepdims=True)

    def adjust_histogram_shape(self, hist: np.ndarray, max_days: int) -> np.ndarray:
        """Trim or NaN-pad a histogram's day axis to exactly ``max_days``."""
        hist_days = hist.shape[0]
        if hist_days > max_days:
            return hist[-max_days:]
        if hist_days < max_days:
            padding = np.full((max_days - hist_days,) + hist.shape[1:], np.nan)
            return np.concatenate((padding, hist), axis=0)
        return hist

    def load_normed_histograms(
        self, histogram_file_positions: list, max_days: int = 22
    ) -> tuple[list, np.ndarray]:
        """Load, normalise, and day-align every histogram into one 4D array."""
        fishes = []
        histograms = []
        for file_position in histogram_file_positions:
            hist = self.load_npy_file(file_position)
            hist = self.normalise_histograms(hist)
            hist = self.adjust_histogram_shape(hist, max_days)
            fishes.append(self.extract_fishID_tanknumber(file_position))
            histograms.append(hist)
        return fishes, np.stack(histograms, axis=3)

    def sort_hists_by_sex(self) -> tuple[np.ndarray, np.ndarray]:
        """Split the loaded histograms into male and female 4D arrays."""
        sex_map = {(row["Tank_number"], row["ID"]): row["Sex"] for _, row in self.df.iterrows()}
        male_hists = []
        female_hists = []
        for i, (tank_num, fish_id) in enumerate(self.fishID):
            sex = sex_map.get((tank_num, fish_id))
            if sex == "M":
                male_hists.append(self.hists[:, :, :, i])
            elif sex == "F":
                female_hists.append(self.hists[:, :, :, i])
        return np.stack(male_hists, axis=3), np.stack(female_hists, axis=3)

    def create_spatial_histograms(self) -> tuple[Figure, Figure, Figure, Figure]:
        """Build day-wise heatmap grids for the median male and female fish."""
        male_hists, female_hists = self.sort_hists_by_sex()
        male_hists = self.normalise_histograms(np.nanmedian(male_hists, axis=3))
        female_hists = self.normalise_histograms(np.nanmedian(female_hists, axis=3))
        male_figure, male_figure_cbar = self.create_daywise_histograms(male_hists)
        female_figure, female_figure_cbar = self.create_daywise_histograms(female_hists)
        return male_figure, male_figure_cbar, female_figure, female_figure_cbar

    def create_box_strip_plots(self) -> list[Figure]:
        """Return one box/strip figure per behavioural metric, split by sex."""
        topics = [
            "Median_speed_cmPs", "Gross_speed_cmPs", "Median_activity_duration_s",
            "Activity_fraction", "Median_freezing_duration_s", "Freezing_fraction",
            "Median_top_duration_s", "Top_fraction", "Median_bottom_duration_s",
            "Bottom_fraction", "Median_tigmotaxis_duration_s", "Tigmotaxis_fraction",
            "Tigmotaxis_transition_freq", "Latency_to_top_s", "Distance_travelled_cm",
        ]
        return [
            self.create_vertical_box_stripplot("Day_number", topic, "Sex", ("M", "F"))
            for topic in topics
        ]
