# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.c_start_movies                      ║
# ║  « c-start contour animations »                                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Renders comet-tail contour animations and raw-ephys figures for ║
# ║  selected c-start recordings.                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Render comet-tail contour animations and raw-ephys figures for c-starts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import plotting.fishPlot as fishPlot
from data_handlers.matLabResultLoader import MatlabResultLoader
from data_handlers.spike2SimpleIO import SegmentSaver, Spike2SimpleReader
from fish_data_base.fishDataBase import FishDataBase
from plotting.cStartPlotter import CStartPlotter
from trace_analysis.SpikeDetector import SpikeDetector

# Hand-picked recordings and their per-recording alignment corrections.
GOOD_TRIALS = [75, 164, 261, 326, 345, 378]
OFFSETS = [(0, 0), (0, 0), (0, 0), (-10, 0), (-10, 0), (-10, 0)]
TRACE_VIDEO_DISPARITY = [0, 0, 0, 0, -523, 0]
ROUND_ROBIN_CORRECTION = [0, 0, 0, -3715, -4027, -3866]


def render_animations(df: pd.DataFrame, plotter: CStartPlotter, out_dir: Path) -> None:
    """Render a comet-tail contour animation for each selected recording."""
    for c, (_i, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="movies")):
        loader = MatlabResultLoader(row["path2_anaMat"])
        _info, contour, _mid, _head, _tail, trace, *_rest = loader.get_data()

        for shift in (TRACE_VIDEO_DISPARITY[c], ROUND_ROBIN_CORRECTION[c]):
            trace = np.roll(trace, shift, axis=0)
            contour = np.roll(contour, shift, axis=0)

        spike_df = pd.read_csv(row.path2_spike_train_df)
        time_ax = fishPlot.make_time_axis(trace.shape[0], row.fps)
        interp_freq = np.interp(
            time_ax, spike_df["spike_peak_s"].to_numpy(), spike_df["instant_freq"].to_numpy()
        )
        animation_filepath = str(out_dir / f"{row.genotype}_{Path(row.avi).name.replace(' ', '_')}")
        plotter.create_animated_plot(
            spike_df, time_ax, trace, interp_freq, contour, row.fps, row.avi,
            animation_filepath, contour_offset=OFFSETS[c],
            round_robin_offset=ROUND_ROBIN_CORRECTION[c],
        )


def render_raw_ephys(df: pd.DataFrame, plotter: CStartPlotter, out_dir: Path) -> None:
    """Render the raw field-potential trace with the spike raster for each row."""
    for _i, row in tqdm(df.iterrows(), total=df.shape[0], desc="raw ephys"):
        reader = Spike2SimpleReader(row.path2_smr)
        reader.main()
        signal_df = SegmentSaver(reader, "no csv file will be produced").main()[0]
        detector = SpikeDetector(signal_df)
        detector.main()
        spike_df = pd.read_csv(row.path2_spike_train_df)

        fig = plt.figure(figsize=(8, 5))
        gs = gridspec.GridSpec(10, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:9, :])
        ax2 = fig.add_subplot(gs[9:, :])
        detector.df_signal.plot(ax=ax1, legend=False)
        ax1.set_xlabel("")
        ax1.set_xlim((0, 5))
        ax1.set_ylim((-2, 2))
        ax1.set_xticklabels([])
        ax1.plot([0, 5], [detector.threshold, detector.threshold], "k--")
        ax1.plot([0, 5], [-detector.threshold, -detector.threshold], "k--")
        ax1.set_ylabel("field potential, muV ")
        plotter.plot_spike_occurrences(spike_df, ax2)
        for side in ("right", "left", "top"):
            ax2.spines[side].set_visible(False)
        ax2.set_ylabel("")
        plt.tight_layout()
        stem = Path(row.avi).name.replace(" ", "_").split(".")[0]
        fig.savefig(out_dir / f"{row.genotype}_{stem}.svg")


def main() -> None:
    """Render animations and raw-ephys figures for the selected recordings."""
    database_path = config.get_path("database_path")
    out_dir = config.get_path("figure_root")
    db = FishDataBase(database_path, database_path / "fishDataBase_cstart.csv")
    selected = db.database.iloc[GOOD_TRIALS, :]
    plotter = CStartPlotter()

    render_animations(selected, plotter, out_dir)
    render_raw_ephys(selected, plotter, out_dir)


if __name__ == "__main__":
    main()
