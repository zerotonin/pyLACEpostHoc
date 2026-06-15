# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.run_curvature_ana                   ║
# ║  « c-start curvature boxplots »                                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Computes total mid-line curvature per recording and boxplots it ║
# ║  by genotype and sex.                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Boxplot total mid-line curvature amplitudes by genotype and sex."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from fish_data_base.fishDataBase import FishDataBase
from trace_analysis.CurvatureAnalyser import CurvatureAnalyser

META_COLUMNS = ["genotype", "sex", "animalNo", "expType", "birthDate"]


def collect_curvature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute curvature amplitudes for every recording with a mid-line."""
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="curvature"):
        if not isinstance(row.path2_midLineUniform_pix, str):
            continue
        try:
            midline_df = pd.read_csv(row.path2_midLineUniform_pix)
            amps = CurvatureAnalyser(midline_df).get_total_curvature_amps()
            records.append({**row[META_COLUMNS].to_dict(), **amps})
        except Exception:
            print(f"!! {row.path2_midLineUniform_pix} did not produce output")
    return pd.DataFrame(records)


def main() -> None:
    """Load the c-start database, compute curvature, and boxplot it."""
    database_path = config.get_path("database_path")
    db = FishDataBase(database_path, database_path / "fishDataBase_cstart.csv")
    curv_df = collect_curvature(db.database)

    for _tag, exp_name in [("cst", "c-start"), ("Unt", "free swimming")]:
        for parameter in ["median_curv_amp", "mean_curv_amp", "max_curv_amp"]:
            plt.figure()
            sns.boxplot(
                x="genotype", y=parameter, order=["rei-INT", "rei-HT", "rei-HM"],
                hue="sex", hue_order=["M", "F"],
                data=curv_df.loc[curv_df["expType"] == exp_name, :],
            ).set_title(exp_name)
    plt.show()


if __name__ == "__main__":
    main()
