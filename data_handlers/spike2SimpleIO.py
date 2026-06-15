# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — data_handlers.spike2SimpleIO                    ║
# ║  « read and tabulate Spike2 smr files »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Wraps neo to read Cambridge Electronics Spike2 recordings into  ║
# ║  per-segment analog + event dictionaries, then to pandas frames. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Read Cambridge Electronics Spike2 ``.smr`` files via neo into pandas."""
from __future__ import annotations

from pathlib import Path

import neo
import numpy as np
import pandas as pd
import quantities as pq

from deprecation import deprecated_alias, deprecated_class_alias


class Spike2SimpleReader:
    """Read a simple Spike2 ``.smr`` file into per-segment analog/event data.

    Wraps the neo library [1]_. The result (``self.outPutData``) is a list
    with one entry per Spike2 segment. Each entry is a ``(analog, events)``
    tuple of dictionaries keyed by channel name; event values hold the
    times an event occurred, analog values hold the time vector and
    channel magnitudes.

    Args:
        file_name: Path to the ``.smr`` file to read.

    References:
        .. [1] Garcia S. et al. (2014) Neo: an object model for handling
           electrophysiology data in multiple formats. Front. Neuroinform.
           8:10. doi:10.3389/fninf.2014.00010
    """

    def __init__(self, file_name: str | Path) -> None:
        self.fileName = str(file_name)
        self.eventList: list = []
        self.analogSigList: list = []
        self.neoReader = False
        self.dataBlock = False
        self.outPutData: list = []

    def read_by_neo(self) -> None:
        """Initialise the neo Spike2 reader and read the data block."""
        self.neoReader = neo.io.Spike2IO(filename=self.fileName)
        self.dataBlock = self.neoReader.read(lazy=False)[0]

    def read_segments(self) -> None:
        """Read every segment of the data block into ``self.outPutData``.

        Some ``.smr`` files hold several recordings at different times;
        each such division is a segment.
        """
        self.outPutData = []
        for seg in self.dataBlock.segments:
            self.outPutData.append(self.read_single_seg(seg))

    def read_single_seg(self, seg) -> tuple[dict, dict]:
        """Read one segment, returning ``(analog_signals, events)``."""
        events = self.read_events(seg)
        analog_sigs = self.read_analog_signals(seg)
        return (analog_sigs, events)

    def read_events(self, seg) -> dict:
        """Return a segment's event channels as ``{name: occurrence times}``."""
        event_data = {}
        for event in seg.events:
            event_data[event.name] = event.times
        return event_data

    def read_analog_signals(self, seg) -> dict:
        """Return a segment's analog channels as ``{name: magnitudes}``.

        The first entry, ``"time_s"``, holds the shared time vector.
        """
        analog_data = {}
        analog_data["time_s"] = seg.analogsignals[0].times
        for a_signal in seg.analogsignals:
            analog_data[a_signal.name] = a_signal.magnitude
        return analog_data

    def main(self) -> None:
        """Read the file: initialise the reader, then read all segments."""
        self.read_by_neo()
        self.read_segments()

    # Deprecated camelCase method names.
    readByNeo = deprecated_alias(read_by_neo, "readByNeo")
    readSegments = deprecated_alias(read_segments, "readSegments")
    readSingleSeg = deprecated_alias(read_single_seg, "readSingleSeg")
    readEvents = deprecated_alias(read_events, "readEvents")
    readAnalogSignals = deprecated_alias(read_analog_signals, "readAnalogSignals")


class SegmentSaver:
    """Write Spike2 reader data to pandas, mapping events to boolean columns.

    Event channels are not bound to the analog sample frequency, so each
    event is placed at the nearest analog sample point.

    Args:
        spike2_reader: A :class:`Spike2SimpleReader` that has been read.
        save_pos:      Output path, e.g. ``/folder/name.ext``; segment
                       index and count are inserted as ``name_1_3.ext``.
    """

    def __init__(self, spike2_reader: Spike2SimpleReader, save_pos: str | Path) -> None:
        self.s2sr = spike2_reader
        self.savePos = str(save_pos)

    def main(self, save_mode: bool = False) -> list[pd.DataFrame]:
        """Build one DataFrame per segment, optionally saving each as CSV.

        Returns:
            List of per-segment DataFrames.
        """
        seg_num = len(self.s2sr.outPutData)
        dataframe_list = []
        for index, segment in enumerate(self.s2sr.outPutData):
            df = self.analog_signal_dict_to_pandas(segment)
            df = self.event_dict_to_pandas(segment, df)
            if save_mode:
                df.to_csv(self.savePos[:-4] + f"_{index}_{seg_num}" + self.savePos[-4:])
            dataframe_list.append(df)
        return dataframe_list

    def analog_signal_dict_to_pandas(self, segment: tuple[dict, dict]) -> pd.DataFrame:
        """Turn a segment's analog signals into a time-indexed DataFrame."""
        a_sig_dict = segment[0]
        a_sig_dict["time_s"] = a_sig_dict["time_s"].rescale(pq.s)
        for channel in list(a_sig_dict.keys()):
            a_sig_dict[channel] = np.array(a_sig_dict[channel]).flatten()
        df = pd.DataFrame(a_sig_dict)
        df.set_index("time_s", inplace=True)
        return df

    def event_dict_to_pandas(self, segment: tuple[dict, dict], df: pd.DataFrame) -> pd.DataFrame:
        """Add a segment's event channels to ``df`` as boolean columns."""
        event_dict = segment[1]
        for channel in list(event_dict.keys()):
            times = event_dict[channel].rescale(pq.s)
            times = np.array(times.flatten())
            df[channel] = self.events_to_bool_signal(df.index.to_numpy(), times)
        return df

    def events_to_bool_signal(self, index_array: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Map event times onto the nearest analog samples as a boolean mask.

        Args:
            index_array: Time index as a NumPy array.
            events:      Event times in seconds.

        Returns:
            Boolean array, True at the closest sample to each event.
        """
        bool_array = np.full(index_array.shape, False)
        for occurrence in events:
            abs_diff = np.abs(index_array - occurrence)
            pos = abs_diff.argmin()
            bool_array[pos] = True
        return bool_array

    # Deprecated camelCase method names.
    analogSignalDict2Pandas = deprecated_alias(
        analog_signal_dict_to_pandas, "analogSignalDict2Pandas"
    )
    eventDict2Pandas = deprecated_alias(event_dict_to_pandas, "eventDict2Pandas")
    events2boolSignal = deprecated_alias(events_to_bool_signal, "events2boolSignal")


# Deprecated lower-camelCase class names.
spike2SimpleReader = deprecated_class_alias(Spike2SimpleReader, "spike2SimpleReader")
segmentSaver = deprecated_class_alias(SegmentSaver, "segmentSaver")
