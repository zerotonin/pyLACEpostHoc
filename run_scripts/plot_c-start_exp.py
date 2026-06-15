# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.plot_c-start_exp                    ║
# ║  « interactive c-start figures »                                 ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Builds the combined c-start contour/spike figure per recording  ║
# ║  and boxplots the spike metrics by genotype.                     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build per-recording c-start figures and boxplot the spike metrics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import pandas as pd
import seaborn as sns

import config
import plotting.fishPlot as fishPlot
from data_handlers.matLabResultLoader import MatlabResultLoader
from fish_data_base.fishDataBase import FishDataBase
from plotting.cStartPlotter import CStartPlotter

SPIKE_METRICS = [
    "latency_to_m_cell", "latency_to_others", "m_cell_spikes",
    "median_spike_instFreq_Hz", "other_spikes",
]


def build_figure(plotter: CStartPlotter, row) -> tuple:
    """Build the combined contour/spike figure for one recording row."""
    loader = MatlabResultLoader(row["path2_anaMat"])
    _info, contour, _mid, _head, _tail, trace, *_rest = loader.get_data()
    spike_df = pd.read_csv(row.path2_spike_train_df)
    time_ax = fishPlot.make_time_axis(trace.shape[0], row.fps)
    interp_freq = np.interp(
        time_ax, spike_df["spike_peak_s"].to_numpy(), spike_df["instant_freq"].to_numpy()
    )
    fig, ax_list = plotter.create_final_plot(spike_df, time_ax, trace, interp_freq, contour, row.fps)
    ax_list[0].set_title(f"{row.genotype} {row.sex} ")
    fig.tight_layout()
    return fig, ax_list


def review_interactively(df: pd.DataFrame, plotter: CStartPlotter, out_dir: Path) -> None:
    """Show each figure with Close / Save-and-Close buttons."""
    for i, row in df.iterrows():
        if i <= 0:
            continue
        try:
            fig, _ax_list = build_figure(plotter, row)
            ax_close = plt.axes([0.35, 0.05, 0.1, 0.075])
            ax_save = plt.axes([0.55, 0.05, 0.1, 0.075])
            button_close = widgets.Button(ax_close, "Close")
            button_save = widgets.Button(ax_save, "Save and Close")
            save_filename = out_dir / f"{i}_{row.genotype}_{row.sex}.svg"
            button_close.on_clicked(lambda event, f=fig: plt.close(f))
            button_save.on_clicked(
                lambda event, f=fig, name=save_filename: (f.savefig(name), plt.close(f))
            )
            plt.show()
        except Exception:
            pass


def plot_spike_metric_boxplots(df: pd.DataFrame) -> None:
    """Boxplot each spike metric by genotype and sex."""
    sns.set_theme(style="ticks")
    for category in SPIKE_METRICS:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x="genotype", y=category, data=df, hue="sex",
                    whis=[0, 100], width=0.6, notch=True)
        sns.stripplot(x="genotype", y=category, data=df, hue="sex",
                      size=4, color=".3", linewidth=0)
        ax.xaxis.grid(True)
        ax.set(ylabel=category)
        sns.despine(trim=True, left=True)


def main() -> None:
    """Review c-start figures interactively, then plot the spike-metric boxplots."""
    database_path = config.get_path("database_path")
    out_dir = config.get_path("figure_root")
    db = FishDataBase(database_path, database_path / "fishDataBase_cstart.csv")
    plotter = CStartPlotter()

    review_interactively(db.database, plotter, out_dir)
    plot_spike_metric_boxplots(db.database)
    plt.show()


if __name__ == "__main__":
    main()
