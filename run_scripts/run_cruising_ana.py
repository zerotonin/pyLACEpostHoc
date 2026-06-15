# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.run_cruising_ana                    ║
# ║  « cruising speed boxplots »                                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Runs the speed analyser over every cruising recording and       ║
# ║  boxplots the kinematic metrics by genotype and sex.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Boxplot cruising kinematic metrics by genotype and sex."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from fish_data_base.fishDataBase import FishDataBase
from trace_analysis.speed_analyser import SpeedAnalyser

META_COLUMNS = ["genotype", "sex", "animalNo", "expType", "birthDate"]
SPEED_METRICS = [
    "thrust_mean_m/s", "slip_mean_m/s", "yaw_mean_m/s", "thrust_median_m/s",
    "slip_median_m/s", "yaw_median_m/s", "cruising_thrust_mean_m/s", "cruising_slip_mean_m/s",
    "cruising_yaw_mean_m/s", "cruising_thrust_median_m/s", "cruising_slip_median_m/s",
    "cruising_yaw_median_m/s", "activity_duration_s", "activity_fraction",
    "sec_to_first_stop", "torque",
]


def collect_speeds(df: pd.DataFrame) -> pd.DataFrame:
    """Run the speed analyser over every recording with a trace."""
    analyser = SpeedAnalyser(fps=501)
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="speed"):
        path = row.path2_trace_mm if isinstance(row.path2_trace_mm, str) else row.path2_head_mm
        analyser.fps = row.fps
        analyser.trace_df = pd.read_csv(path)
        records.append({**row[META_COLUMNS].to_dict(), **analyser.analyse_fish_speed()})
    return pd.DataFrame(records)


def main() -> None:
    """Load the cruising database, compute speeds, and boxplot the metrics."""
    database_path = config.get_path("database_path")
    figure_root = config.get_path("figure_root")
    db = FishDataBase(database_path, database_path / "fishDataBase_cruise.csv")
    speed_df = collect_speeds(db.database)

    for _tag, exp_name in [("Ta", "motivated swimming"), ("Unt", "free swimming")]:
        for parameter in SPEED_METRICS:
            plt.figure()
            sns.boxplot(
                x="genotype", y=parameter, order=["rei-INT", "rei-HT", "rei-HM"],
                hue="sex", hue_order=["M", "F"],
                data=speed_df.loc[speed_df["expType"] == exp_name, :],
            ).set_title(exp_name)
            safe = f"{exp_name}--{parameter}".replace(" ", "_").replace("/", "_per_")
            plt.savefig(figure_root / f"{safe}.svg")
    plt.show()


if __name__ == "__main__":
    main()
