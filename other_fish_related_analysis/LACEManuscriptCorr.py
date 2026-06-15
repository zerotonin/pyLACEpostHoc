# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — other_fish_related_analysis.LACEManuscriptCorr  ║
# ║  « LACE auto-correction statistics »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Summarises how often the LACE tracker auto-corrected detections ║
# ║  across a folder of MATLAB result files, for the manuscript.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Summarise LACE tracker auto-correction frequency across result files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from data_handlers.matLabResultLoader import MatlabResultLoader

# Trace-result columns: 11 = detection quality, 12 = auto-correction flag.
QUALITY_COL: int = 11
CORRECTION_COL: int = 12


def summarise_corrections(source_dir: Path) -> pd.DataFrame:
    """Return per-file frame counts, correction counts, rate, and quality."""
    files = sorted(Path(source_dir).rglob("*.mat"))
    results = []
    for file in tqdm(files, desc="read matlab files"):
        loader = MatlabResultLoader(file)
        loader.get_data()
        quality = [entry[0][0][QUALITY_COL] for entry in loader.traceResult]
        auto_corrector = [entry[0][0][CORRECTION_COL] for entry in loader.traceResult]
        frames = len(auto_corrector)
        num_corr = np.sum(auto_corrector)
        results.append([frames, num_corr, num_corr / frames, np.mean(quality)])
    return pd.DataFrame(
        np.array(results),
        columns=["frame", "number of corrections", "corrections per frame", "median quality, au"],
    )


def main(source_dir: Path | None = None, save_position: Path | None = None) -> None:
    """Summarise corrections under the configured data root and plot them."""
    data_root = config.get_path("data_root")
    source_dir = source_dir or data_root / "combinedData/traceResultsAna"
    save_position = save_position or config.get_path("figure_root") / "fishDf.h5"

    df = summarise_corrections(Path(source_dir))
    df.to_hdf(save_position, key="df")
    sns.displot(df, x="corrections per frame", binwidth=0.01, kde=False, rug=True,
                log_scale=(False, True))
    plt.show()


if __name__ == "__main__":
    main()
