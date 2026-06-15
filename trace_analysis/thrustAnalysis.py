# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — trace_analysis.thrustAnalysis                   ║
# ║  « thrust-triggered average by sex »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Extracts swimming thrust strokes, builds a thrust-triggered     ║
# ║  average, and compares males and females by permutation test.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Thrust-triggered average of swimming strokes, compared by sex."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

import config
from data_handlers.matLabResultLoader import MatlabResultLoader

THRUST_THRESHOLD: float = 0.1     # m/s, minimum peak height for a stroke
FPS: int = 200                    # acquisition frame rate
FRAMES_BEFORE: int = 20           # frames kept before each stroke peak
FRAMES_AFTER: int = 100           # frames kept after each stroke peak
PEAK_DISTANCE: int = 100          # minimum frames between stroke peaks
N_PERMUTATIONS: int = 20000       # Monte-Carlo permutations
THRUST_COLUMN: int = 3            # column of `trace` holding thrust


def exact_mc_perm_test(xs: np.ndarray, ys: np.ndarray, n_permutations: int) -> float:
    """Monte-Carlo permutation p-value for a difference between two samples.

    FIXME(flagged): the observed statistic uses the *median* difference while
    the permuted statistic uses the *mean* difference, which is inconsistent.
    Preserved as-is pending confirmation; the lab standard is to run this
    through reRandomStats. See issue #4.

    Args:
        xs, ys:         The two samples to compare.
        n_permutations: Number of Monte-Carlo shuffles.

    Returns:
        Fraction of permutations at least as extreme as the observed value.
    """
    n, k = len(xs), 0
    diff = np.abs(np.median(xs) - np.median(ys))
    zs = np.concatenate([xs, ys])
    for _ in range(n_permutations):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / n_permutations


def extract_thrust_by_sex(
    meta: pd.DataFrame, collection_dir: Path
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load the thrust trace of every animal, split into males and females.

    Args:
        meta:           Metadata rows (with ``matFileName`` and ``sex``).
        collection_dir: Directory holding the per-animal result files.

    Returns:
        ``(male_thrust, female_thrust)`` lists of 1D thrust arrays.
    """
    male_thrust: list[np.ndarray] = []
    female_thrust: list[np.ndarray] = []
    for i in tqdm(range(meta.shape[0]), desc="Loading thrust traces"):
        file_names = glob.glob(str(collection_dir / (meta["matFileName"].iloc[i][:-4] + "*")))
        loader = MatlabResultLoader(file_names[0])
        trace = loader.get_data()[5]
        if meta["sex"].iloc[i] == "female":
            female_thrust.append(trace[:, THRUST_COLUMN])
        else:
            male_thrust.append(trace[:, THRUST_COLUMN])
    return male_thrust, female_thrust


def thrust_triggered_average(
    thrust_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the thrust-triggered average across a group of animals.

    Args:
        thrust_list: Per-animal thrust traces.

    Returns:
        ``(mean, sd, stroke_frequency_per_min)`` across animals.
    """
    mean_thrusts: list[np.ndarray] = []
    freq: list[float] = []
    for thrust in thrust_list:
        peaks, _ = find_peaks(thrust, height=THRUST_THRESHOLD, distance=PEAK_DISTANCE)
        trig_ave = [
            thrust[peak - FRAMES_BEFORE : peak + FRAMES_AFTER]
            for peak in peaks
            if peak > FRAMES_BEFORE and peak < thrust.shape[0] - FRAMES_AFTER
        ]
        if trig_ave:
            freq.append(len(trig_ave) / (thrust.shape[0] / FPS))
            mean_thrusts.append(np.nanmean(np.array(trig_ave), axis=0))
    mean_thrusts = np.array(mean_thrusts)
    return (
        np.nanmean(mean_thrusts, axis=0),
        np.nanstd(mean_thrusts, axis=0),
        np.array(freq) * 60,
    )


def plot_results(
    inter_ind_mean: list[np.ndarray],
    inter_ind_sd: list[np.ndarray],
    thrust_freq: list[np.ndarray],
) -> None:
    """Plot the thrust-triggered averages and stroke-frequency boxplots."""
    import matplotlib.pyplot as plt

    p_mean = exact_mc_perm_test(inter_ind_mean[0], inter_ind_mean[1], N_PERMUTATIONS)
    p_freq = exact_mc_perm_test(thrust_freq[0], thrust_freq[1], N_PERMUTATIONS)

    x = np.linspace(-1 * FRAMES_BEFORE / FPS * 1000, FRAMES_AFTER / FPS * 1000,
                    FRAMES_BEFORE + FRAMES_AFTER)
    fig, ax = plt.subplots()
    for series in (0, 1):
        ax.plot(x, inter_ind_mean[series])
        ax.fill_between(
            x,
            inter_ind_mean[series] - inter_ind_sd[series],
            inter_ind_mean[series] + inter_ind_sd[series],
            alpha=0.5,
        )
    ax.legend(["male", "female"])
    ax.set_xlabel("time, ms")
    ax.set_ylabel("thrust, m*s-1")
    ax.set_title(f"thrust triggered average | perm. median p = {p_mean}")

    fig2, ax2 = plt.subplots()
    ax2.boxplot(thrust_freq)
    ax2.set_ylabel("frequency of thrust strokes, min-1")
    ax2.set_title(f"stroke frequency per minute | perm. median p = {p_freq}")
    plt.show()


def main(collection_dir: Path | None = None, meta_path: Path | None = None) -> None:
    """Run the full thrust analysis for the ABTLF tapped data set.

    Paths default to the configured ``data_root`` (see ``local_paths.json``).
    """
    data_root = config.get_path("data_root")
    collection_dir = collection_dir or data_root / "combinedData/traceResultsAna/ABTLF"
    meta_path = meta_path or data_root / "combinedData/traceResultsAna_meta_pandasPickle.pkl"

    meta = pd.read_pickle(meta_path)
    meta = meta.loc[(meta["genoType"] == "ABTLF") & (meta["experimentType"] == "tapped")]

    male_thrust, female_thrust = extract_thrust_by_sex(meta, Path(collection_dir))

    inter_ind_mean, inter_ind_sd, thrust_freq = [], [], []
    for thrust_list in (male_thrust, female_thrust):
        mean, sd, freq = thrust_triggered_average(thrust_list)
        inter_ind_mean.append(mean)
        inter_ind_sd.append(sd)
        thrust_freq.append(freq)

    plot_results(inter_ind_mean, inter_ind_sd, thrust_freq)


if __name__ == "__main__":
    main()
