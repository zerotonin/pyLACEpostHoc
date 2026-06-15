# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — other_fish_related_analysis.cholesterolAna      ║
# ║  « cholesterol assay statistics »                                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Confidence intervals, Fisher tests, and scatter plots for the   ║
# ║  freeze-dried cholesterol fluorescence assay.                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Confidence intervals, Fisher tests, and plots for the cholesterol assay."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from deprecation import deprecated_alias

CONFIDENCE_LEVEL: float = 0.95


def get_ci(data: np.ndarray) -> tuple[float, float]:
    """Return the 95% t confidence interval of ``data`` (NaNs dropped)."""
    data = data[np.logical_not(np.isnan(data))]
    return st.t.interval(CONFIDENCE_LEVEL, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))


def get_ci_from_2d(data: np.ndarray, axis: int = 0) -> list[tuple[float, float]]:
    """Return per-column confidence intervals of a 2D array."""
    shape_range = 1 if axis == 0 else 0
    return [get_ci(data[:, col]) for col in range(data.shape[shape_range])]


def read_file(file_position: str | Path) -> tuple[np.ndarray, list]:
    """Read a transposed assay CSV, returning its values and column labels."""
    df = pd.read_csv(file_position, index_col=0, header=None).T
    return df.to_numpy(), list(df.columns)


def scatter_plot(data, med, ci, data_labels, ylim: tuple[float, float] = (0, 30.0)) -> None:
    """Scatter each group's points with a median marker and CI error bar."""
    for set_i in range(data.shape[1]):
        plt.scatter(np.ones(shape=data.shape[0]) * set_i, data[:, set_i])
        bounds = [[ci[set_i][0]], [ci[set_i][1]]]
        plt.errorbar(set_i, med[set_i], yerr=bounds, marker="s", mec="k")
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, data.shape[1] - 1, data.shape[1]))
    ax.set_xticklabels(data_labels, rotation=45, ha="right")
    plt.ylabel(r"Cholestrol, µg")
    plt.ylim(ylim)


def add_significance(x_coords, y_coords, offset, significance_str: str = "ns",
                     color_str: str = "k") -> None:
    """Draw a significance bracket with a label between two x positions."""
    x1, x2 = x_coords
    y1, y2 = y_coords
    y1, y2 = y1 + offset, y2 + offset
    y3, h = np.max(y_coords) + 2 * offset, 2 * offset
    plt.plot([x1, x1, x2, x2], [y1, y3 + h, y3 + h, y2], lw=1.5, c=color_str)
    plt.text((x1 + x2) * 0.5, y3 + h, significance_str, ha="center", va="bottom", color=color_str)


def fisher_test_for_fluorescence(group_a, group_b) -> tuple[float, float]:
    """Fisher exact test on (positive, total) counts of two groups."""
    return st.fisher_exact(
        [[group_a[0], group_b[0]], [group_a[1] - group_a[0], group_b[1] - group_b[0]]]
    )


def main(csv_path: str | Path) -> None:
    """Plot the three-panel cholesterol scatter from an assay CSV."""
    data, data_labels = read_file(csv_path)
    med = np.nanmedian(data, axis=0)
    ci = get_ci_from_2d(data, axis=0)
    for panel in range(3):
        lo, hi = panel * 3, panel * 3 + 3
        plt.subplot(1, 3, panel + 1)
        scatter_plot(data[:, lo:hi], med[lo:hi], ci[lo:hi], data_labels[lo:hi])
    plt.show()


# Deprecated camelCase function names.
getCI = deprecated_alias(get_ci, "getCI")
getCIfrom2D = deprecated_alias(get_ci_from_2d, "getCIfrom2D")
readFile = deprecated_alias(read_file, "readFile")
addSignificance = deprecated_alias(add_significance, "addSignificance")
fisherTest4Fluo = deprecated_alias(fisher_test_for_fluorescence, "fisherTest4Fluo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the cholesterol assay scatter.")
    parser.add_argument("csv_path", type=Path, help="assay CSV (normalised fluorescence)")
    main(parser.parse_args().csv_path)
