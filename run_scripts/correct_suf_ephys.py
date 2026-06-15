# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.correct_suf_ephys                   ║
# ║  « one-off round-robin correction »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  One-time cyclic-shift correction of mis-aligned sufge1 c-start  ║
# ║  tracking data (guarded; do not re-run on corrected data).       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""One-time cyclic-shift correction of mis-aligned sufge1 c-start tracking.

WARNING: this rewrites tracking CSVs in place. Running it twice on the same
data corrupts it. The correction is guarded behind an explicit ``--apply``
flag and a typed confirmation, and writes nothing in the default dry run.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd
from tqdm import tqdm

import config
from fish_data_base.fishDataBase import FishDataBase

STRAIN_MAP = {"Ht": "sufge1-HT", "Hm": "sufge1-HM", "Int": "sufge1-INT"}
TRACKING_FIELDS = [
    "path2_trace_mm", "path2_midLineUniform_mm", "path2_midLineUniform_pix",
    "path2_head_mm", "path2_tail_mm",
]


def extract_info_from_id_text(identifier_text: str) -> tuple[str, str, int]:
    """Parse (strain, sex, fish number) from a jump-table identifier string.

    Example:
        >>> extract_info_from_id_text("sample_Hm123M4IIII")
        ('sufge1-HM', 'M', 4)
    """
    identifier = identifier_text.split("_")[1]
    if identifier[1] == "n":
        strain, rest = identifier[:3], identifier[3:]
    else:
        strain, rest = identifier[:2], identifier[2:]
    strain = STRAIN_MAP[strain]
    sex, rest = rest[:1], rest[1:]
    fish_no = int(re.findall(r"\d+", rest)[0])
    return strain, sex, fish_no


def get_val(df: pd.DataFrame, field: str):
    """Return the first value in a DataFrame column."""
    return df[field].iloc[0]


def shift_dataframe(df: pd.DataFrame, y: int, correct_timing: bool = True) -> pd.DataFrame:
    """Cyclically rotate a DataFrame so it starts at row ``y + 1``."""
    upper_part = df.iloc[y + 1:].copy()
    lower_part = df.iloc[:y + 1].copy()
    if correct_timing:
        frame_dur = df["time sec"].diff().median()
        upper_part.loc[:, "time sec"] = upper_part["time sec"] - upper_part["time sec"].iloc[0]
        lower_part.loc[:, "time sec"] = (
            lower_part["time sec"] + upper_part["time sec"].iloc[-1] + frame_dur
        )
    return pd.concat([upper_part, lower_part]).reset_index(drop=True)


def correct_rr_error(fish_df: pd.DataFrame, rr_offset: int, write: bool) -> None:
    """Cyclic-shift every tracking CSV for one fish, writing only if ``write``."""
    for field in TRACKING_FIELDS:
        path = get_val(fish_df, field)
        if not isinstance(path, str):
            continue
        df = pd.read_csv(path)
        df = shift_dataframe(df, rr_offset, correct_timing=(field != "path2_trace_mm"))
        if write:
            df.to_csv(path, index=False)


def main(apply: bool = False) -> None:
    """Apply (or dry-run) the round-robin correction over the jump table."""
    if apply:
        confirm = input("This rewrites tracking CSVs in place. Type 'APPLY' to proceed: ")
        if confirm != "APPLY":
            print("Aborted; no files written.")
            return

    database_path = config.get_path("database_path")
    db = FishDataBase(database_path, database_path / "fishDataBase_cstart.csv")
    df = db.database
    df_jump = pd.read_csv(database_path / "suf_cstart_round_robin_jumps.csv")

    for _, row in tqdm(df_jump.iterrows(), desc="round robin update"):
        strain, sex, fish_no = extract_info_from_id_text(row[1])
        fish = df.loc[
            (df["genotype"] == strain) & (df["sex"] == sex) & (df["animalNo"] == fish_no), :
        ]
        if len(fish) == 1:
            correct_rr_error(fish, int(row[2].split("-")[0]), write=apply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-robin c-start correction (one-time).")
    parser.add_argument("--apply", action="store_true", help="write corrections (default: dry run)")
    main(parser.parse_args().apply)
