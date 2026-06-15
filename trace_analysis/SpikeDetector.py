# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — trace_analysis.SpikeDetector                    ║
# ║  « detect and classify ephys spikes »                            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Finds positive/negative spikes in a single ephys channel,       ║
# ║  splits Mauthner from other units, and quantifies latency/rate.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Detect, classify, and quantify spikes in a single ephys channel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from deprecation import deprecated_alias

SIGNAL_COLUMN: str = "Signal stream 0"
# Mauthner spikes are categorised by amplitude relative to the median rather
# than a fixed threshold, because amplitude depends on electrode-fish geometry.
MAUTHNER_AMPLITUDE_FACTOR: float = 4.0


class SpikeDetector:
    """Detect spikes in a single-channel electrophysiology signal.

    Args:
        df_signal:     DataFrame holding the ephys signal and a Keyboard
                       column marking stimulus activation.
        min_latency_s: Minimum latency between stimulus and a counted spike.
    """

    def __init__(self, df_signal: pd.DataFrame, min_latency_s: float = 0.003) -> None:
        self.df_signal = df_signal
        self.min_latency_s = min_latency_s
        self.spike_train_df: pd.DataFrame | None = None
        self.stimulus_occurence_s = None

    def find_peaks_in_df(self, signal: pd.Series, threshold: float) -> pd.DataFrame:
        """Find peaks in ``signal`` above a prominence ``threshold``.

        Args:
            signal:    Signal series indexed by time (shares ``df_signal``'s index).
            threshold: Prominence threshold for :func:`scipy.signal.find_peaks`.

        Returns:
            DataFrame of spike peak/start/stop times and amplitudes.
        """
        peak_pos, peak_data = find_peaks(signal, prominence=threshold)
        peak_pos_time = self.df_signal.index[peak_pos].to_numpy()
        peak_start_time = self.df_signal.index[peak_data["left_bases"]].to_numpy()
        peak_end_time = self.df_signal.index[peak_data["right_bases"]].to_numpy()
        peak_amplitude = peak_data["prominences"]
        return pd.DataFrame(
            np.stack([peak_pos_time, peak_start_time, peak_end_time, peak_amplitude]).T,
            columns=["spike_peak_s", "spike_start_s", "spike_stop_s", "amplitude_muV"],
        )

    def get_peak_time_and_amp(self, noise_std_factor: float = 1.5) -> None:
        """Detect positive and negative spikes into ``spike_train_df``.

        Args:
            noise_std_factor: Multiplier on the signal std for the threshold.
        """
        signal = self.df_signal[SIGNAL_COLUMN]
        self.threshold = signal.std() * noise_std_factor

        spike_df_positive = self.find_peaks_in_df(signal, self.threshold)
        # Negate the signal to find troughs, without mutating df_signal.
        spike_df_negative = self.find_peaks_in_df(-signal, self.threshold)
        # FIXME(flagged): this assigns a DataFrame *attribute* 'amplitude_mV'
        # rather than the 'amplitude_muV' column, so negative-spike amplitudes
        # are never actually negated. Preserved as-is pending confirmation —
        # see issue #2. (separate_m_units uses .abs(), so spike
        # categorisation is unaffected; only the stored amplitude sign is.)
        spike_df_negative.amplitude_mV = spike_df_negative.amplitude_muV * -1

        spike_df = pd.concat([spike_df_positive, spike_df_negative])
        spike_df = spike_df.sort_values(by="spike_peak_s")
        spike_df = spike_df.reset_index(drop=True)
        self.spike_train_df = spike_df

    @staticmethod
    def calculate_instantaneous_spike_freq(spike_df: pd.DataFrame) -> np.ndarray:
        """Return instantaneous spike frequency from inter-spike intervals."""
        inter_spike_intervals = np.diff(spike_df.spike_peak_s.to_numpy())
        return 1 / inter_spike_intervals

    def separate_m_units(self) -> None:
        """Tag each spike as ``"Mauthner"`` or ``"Other"`` by amplitude.

        Mauthner spikes (the giant flight-reflex fibres) are much larger;
        they are tagged when their absolute amplitude exceeds the median
        absolute amplitude by :data:`MAUTHNER_AMPLITUDE_FACTOR`, not a fixed cutoff,
        because amplitude varies with electrode-fish distance and orientation.
        """
        amplitude = self.spike_train_df.amplitude_muV.abs()
        self.spike_train_df["spike_category"] = "Other"
        self.spike_train_df.loc[
            amplitude > amplitude.median() * MAUTHNER_AMPLITUDE_FACTOR, "spike_category"
        ] = "Mauthner"

    def get_timing_from_keyboard(self) -> None:
        """Record the times at which the stimulus keyboard trigger fired."""
        self.stimulus_occurence_s = self.df_signal.index[self.df_signal.Keyboard]

    def calculate_latency(self) -> tuple[float, float]:
        """Return latency to the first Mauthner and first other spike.

        Spikes earlier than ``min_latency_s`` after a stimulus are ignored.

        Returns:
            ``(mauthner_latency, other_latency)`` in seconds.
        """
        mauthner_latency = self.df_signal.index[-1]
        other_latency = self.df_signal.index[-1]

        for stimulus_time in self.stimulus_occurence_s:
            mauthner_candidates = self.spike_train_df[
                (self.spike_train_df.spike_category == "Mauthner")
                & (self.spike_train_df.spike_peak_s >= stimulus_time + self.min_latency_s)
            ]
            other_candidates = self.spike_train_df[
                (self.spike_train_df.spike_category == "Other")
                & (self.spike_train_df.spike_peak_s >= stimulus_time + self.min_latency_s)
            ]

            if not mauthner_candidates.empty:
                mauthner_latency = mauthner_candidates.iloc[0].spike_peak_s - stimulus_time
            if not other_candidates.empty:
                other_latency = other_candidates.iloc[0].spike_peak_s - stimulus_time

        return mauthner_latency, other_latency

    def quantify_spike_properties(self) -> dict:
        """Return spike counts, latencies, and median instantaneous frequency."""
        mauthner_latency, other_latency = self.calculate_latency()
        spike_counts = self.spike_train_df.spike_category.value_counts()
        return {
            "m_cell_spikes": spike_counts["Mauthner"],
            "other_spikes": spike_counts["Other"],
            "latency_to_m_cell": mauthner_latency,
            "latency_to_others": other_latency,
            "median_spike_instFreq_Hz": self.spike_train_df.instant_freq.median(),
        }

    def main(self, noise_factor: float = 1.5) -> tuple[pd.DataFrame, dict]:
        """Run detection end-to-end, returning the spike train and properties.

        Args:
            noise_factor: Noise std multiplier for the detection threshold.

        Returns:
            ``(spike_train_df, spike_properties)``.
        """
        self.get_peak_time_and_amp(noise_factor)
        self.separate_m_units()
        instant_freq = self.calculate_instantaneous_spike_freq(self.spike_train_df)
        instant_freq = np.insert(instant_freq, 0, 0)
        self.spike_train_df["instant_freq"] = instant_freq
        self.get_timing_from_keyboard()
        spike_properties = self.quantify_spike_properties()
        return self.spike_train_df, spike_properties

    # Deprecated camelCase method name.
    separate_M_units = deprecated_alias(separate_m_units, "separate_M_units")
