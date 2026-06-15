# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.readMetaData                        ║
# ║  « convert MATLAB metadata to CSV »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Reads the combined LACE metadata .mat file and writes it out as ║
# ║  a CSV and a pandas pickle.                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Convert the combined LACE metadata .mat file to CSV and pickle."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import scipy.io

import config

# Per-file metadata field index in the MATLAB metaAll struct.
FIELDS = {
    "matFileName": 1, "genoType": 2, "sex": 3, "experimentType": 4,
    "pix2mm": 5, "bwFilterSet": 6, "fps": 7, "saccThresh": 8,
}


def convert_metadata(mat_path: Path) -> pd.DataFrame:
    """Read the metaAll struct from a .mat file into a tidy DataFrame."""
    meta_all = scipy.io.loadmat(mat_path)["metaAll"]
    columns: dict[str, list] = {name: [] for name in FIELDS}
    date_time = []
    for file_i in range(meta_all.shape[1]):
        entry = meta_all[0, file_i]
        for name, idx in FIELDS.items():
            value = os.path.basename(entry[idx][0]) if name == "matFileName" else entry[idx][0]
            columns[name].append(value)
        date_time.append([int(s) for s in re.findall(r"\d+", os.path.basename(entry[1][0]))])
    return pd.DataFrame({"dateTime": date_time, **columns})


def main() -> None:
    """Convert the configured metadata .mat file to CSV and pickle."""
    combined = config.get_path("data_root") / "combinedData"
    meta_data = convert_metadata(combined / "traceResultsAna_meta.mat")
    meta_data.to_csv(combined / "traceResultsAna_meta.csv", index=False)
    meta_data.to_pickle(combined / "traceResultsAna_meta_pandasPickle.pkl")


if __name__ == "__main__":
    main()
