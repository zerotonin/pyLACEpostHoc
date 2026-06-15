# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — data_base_analyser.CounterCurrentAnalyser       ║
# ║  « rheotaxis distance statistics »                               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads occupancy densities from the database and compares how far ║
# ║  each group sits from the stream centre (histograms + boxplots). ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Compare how far each group sits from the counter-current stream centre."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp2d  # deprecated in scipy; see FIXME below

# Arena centre used for distance-to-stream calculations, in pixel-bin units.
ORTHO_CENTRE: float = 4 * 4.77
# FIXME(flagged): the stream-axis centre differs between the two distance
# methods — calculate_distances_to_core uses 12*10 (=120) while
# calculate_weighted_distances_to_core uses 9*10 (=90). At most one matches the
# real arena centre, and the choice changes the published distances. Both are
# preserved verbatim below pending confirmation of the correct value.
STREAM_CENTRE_MAX: float = 12 * 10
STREAM_CENTRE_WEIGHTED: float = 9 * 10


class CounterCurrentAnalyser:
    """Analyse counter-current occupancy: distance to the stream centre.

    Args:
        df:        Database DataFrame with occupancy-density file paths.
        genotypes: Genotypes to keep.
        sexes:     Sexes to keep.
    """

    def __init__(self, df, genotypes=("rei-INT", "rei-HM", "rei-HT"), sexes=("F", "M")) -> None:
        self.df = df
        self.df = self.filter_dataframe(genotypes, "counter current", sexes)

    # ── data processing ─────────────────────────────────────────────
    def filter_dataframe(self, genotypes, exp: str, sexes) -> pd.DataFrame:
        """Keep only rows matching the genotypes, experiment, and sexes."""
        return self.df[
            self.df["genotype"].isin(genotypes)
            & (self.df["expType"] == exp)
            & self.df["sex"].isin(sexes)
        ]

    def extract_numbers_from_columnnames(self, columns) -> np.ndarray:
        """Return the first integer found in each column name."""
        numbers = []
        for col in columns:
            number = re.findall(r"\d+", col)
            if number:
                numbers.append(int(number[0]))
        return np.array(numbers)

    def read_prob_density_csv(self, file_position) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read and normalise an occupancy-density CSV into (data, x, y)."""
        pd_df = pd.read_csv(file_position)
        if "Unnamed: 0" in pd_df.columns:
            pd_df = pd_df.drop(columns="Unnamed: 0")
        x_axis = self.extract_numbers_from_columnnames(pd_df.columns)
        y_axis = pd_df.orthoIndexMM.values
        data = pd_df.iloc[:, 1:].to_numpy()
        data = data / np.sum(np.sum(data, axis=1), axis=0)  # normalise histogram
        return data, x_axis, y_axis

    def generate_grouped_data(self) -> dict:
        """Stack each (sex, genotype) group's density arrays into a 3D array."""
        grouped = self.df.groupby(["sex", "genotype"])
        result = {}
        for (sex, genotype), group in grouped:
            data_arrays = [self.read_prob_density_csv(row["path2_probDensity"])[0]
                           for _, row in group.iterrows()]
            result[(sex, genotype)] = np.stack(data_arrays)
        return result

    # ── distance calculations ───────────────────────────────────────
    def calculate_distances_to_core(self, data: np.ndarray) -> np.ndarray:
        """Distance from the stream centre to each density's peak occupancy."""
        distance_to_core = []
        for i in range(data.shape[0]):
            current_data = data[i, :, :]
            ortho_idx, stream_idx = np.unravel_index(
                np.argmax(current_data, axis=None), current_data.shape
            )
            y_diff_mm = ortho_idx - ORTHO_CENTRE
            x_diff_mm = stream_idx - STREAM_CENTRE_MAX
            distance_to_core.append(np.sqrt(x_diff_mm**2 + y_diff_mm**2))
        return np.array(distance_to_core)

    def calculate_weighted_distances_to_core(self, data: np.ndarray) -> np.ndarray:
        """Occupancy-weighted mean distance from the stream centre."""
        weighted_distance_to_core = []
        for i in range(data.shape[0]):
            current_data = data[i, :, :]
            y_diff_mm, x_diff_mm = np.mgrid[0:current_data.shape[0], 0:current_data.shape[1]]
            y_diff_mm = y_diff_mm.astype(float) - ORTHO_CENTRE
            x_diff_mm = x_diff_mm.astype(float) - STREAM_CENTRE_WEIGHTED
            distances = np.sqrt(x_diff_mm**2 + y_diff_mm**2)
            weighted_distance_to_core.append(np.sum(distances * current_data))
        return np.array(weighted_distance_to_core)

    def generate_boxplot_data(self, result: dict, mode: str = "all") -> pd.DataFrame:
        """Build the long-form distance-to-centre DataFrame for boxplots.

        Args:
            result: (sex, genotype) → 3D density array.
            mode:   ``"all"`` (weighted) or ``"max"`` (peak-based) distance.

        Raises:
            ValueError: If ``mode`` is neither ``"all"`` nor ``"max"``.
        """
        boxplot_list = []
        for key, data in result.items():
            if mode == "all":
                dist = self.calculate_weighted_distances_to_core(data)
            elif mode == "max":
                dist = self.calculate_distances_to_core(data)
            else:
                raise ValueError(f"generate_boxplot_data: unknown mode: {mode}")
            core_df = pd.DataFrame(dist, columns=["distance to center of stream, mm"])
            core_df["id"] = key[0] + key[1]
            core_df["sex"] = key[0]
            core_df["genotype"] = key[1]
            boxplot_list.append(core_df)
        return pd.concat(boxplot_list)

    # ── plotting ────────────────────────────────────────────────────
    def plot_2d_histogram_with_marginals(
        self, x_axis, y_axis, data, interp_factor: int = 2, clim=None
    ):
        """Plot an interpolated 2D occupancy histogram with marginal curves."""
        fig = plt.figure(figsize=(10, 7))
        gs = plt.GridSpec(6, 6, wspace=0.3, hspace=0.3)
        ax_main = fig.add_subplot(gs[1:-1, :-1])
        ax_main.set_aspect("equal")
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:-1, -1], sharey=ax_main)

        # FIXME(flagged): scipy.interpolate.interp2d is deprecated (>=1.10) and
        # removed in scipy 1.14; migrate to RegularGridInterpolator. Kept for now
        # to preserve the existing interpolation result.
        f = interp2d(x_axis, y_axis, data, kind="cubic")
        x_axis_new = np.linspace(x_axis.min(), x_axis.max(), x_axis.size * interp_factor)
        y_axis_new = np.linspace(y_axis.min(), y_axis.max(), y_axis.size * interp_factor)
        data_new = f(x_axis_new, y_axis_new)

        x_grid, y_grid = np.meshgrid(
            np.append(x_axis_new, x_axis_new[-1] * 2 - x_axis_new[-2]),
            np.append(y_axis_new, y_axis_new[-1] * 2 - y_axis_new[-2]),
        )
        im = ax_main.pcolormesh(
            x_grid, y_grid, data_new, cmap="viridis",
            vmin=clim[0] if clim else None, vmax=clim[1] if clim else None,
        )

        ax_top.plot(x_axis, data.sum(axis=0), color="darkgray", alpha=0.7)
        ax_top.set_ylim(0, 0.125)
        ax_right.plot(data.sum(axis=1), y_axis, color="darkgray", alpha=0.7)
        ax_right.set_xlim(0, 0.275)
        ax_top.set_xticks([])
        ax_right.set_yticks([])
        ax_main.set_xlabel("X-axis (mm)")
        ax_main.set_ylabel("Y-axis (mm)")

        divider = make_axes_locatable(ax_main)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cbar = fig.colorbar(im, cax=cax, label="Colorbar", orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("bottom")
        return fig

    def create_all_histograms(self, result, x_ax, y_ax, interp_factor=3, clim=(0.0, 0.03)) -> list:
        """Return one summed-occupancy histogram figure per (sex, genotype)."""
        fig_list = []
        for key, data in result.items():
            com_data = np.sum(data, axis=0)
            com_data = com_data / com_data.sum()
            fig = self.plot_2d_histogram_with_marginals(
                x_ax, y_ax, com_data, interp_factor=interp_factor, clim=clim
            )
            fig.suptitle(key[0] + " " + key[1])
            fig_list.append(fig)
        return fig_list

    def plot_boxplot(self, box_df):
        """Plot the distance-to-stream-centre boxplot by genotype and sex."""
        fig = plt.figure(figsize=(10, 10))
        sns.boxplot(
            data=box_df, x="genotype", y="distance to center of stream, mm", hue="sex",
            notch=False, order=["rei-INT", "rei-HT", "rei-HM"], hue_order=["M", "F"],
        )
        return fig

    # ── main ────────────────────────────────────────────────────────
    def save_result(self, boxplot_df, box_fig, hist_fig, fig_path, data_path) -> None:
        """Save the boxplot figure, histogram figures, and boxplot CSV."""
        box_fig.savefig(f"{fig_path}CC_distance2stream.svg")
        for i, fig in enumerate(hist_fig, start=1):
            fig.savefig(f"{fig_path}CC_histogram_f{i}.svg")
        boxplot_df.to_csv(f"{data_path}counter_current.csv")

    def main(self, fig_path, data_path, distance_mode="all", interp_factor=3, clim=(0.0, 0.03)) -> None:
        """Generate and save all counter-current plots and statistics."""
        _, x_ax, y_ax = self.read_prob_density_csv(self.df.path2_probDensity[0])
        prob_density = self.generate_grouped_data()
        boxplot_df = self.generate_boxplot_data(prob_density, distance_mode)
        box_fig = self.plot_boxplot(boxplot_df)
        hist_fig = self.create_all_histograms(
            prob_density, x_ax, y_ax, interp_factor=interp_factor, clim=clim
        )
        self.save_result(boxplot_df, box_fig, hist_fig, fig_path, data_path)
