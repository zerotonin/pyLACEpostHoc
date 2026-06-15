# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.rei_ana                             ║
# ║  « run the counter-current analyser »                            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads the fish database and runs the counter-current distance   ║
# ║  statistics, saving figures and the boxplot CSV.                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Run the counter-current distance analyser over the fish database."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import config
from data_base_analyser.CounterCurrentAnalyser import CounterCurrentAnalyser


def main() -> None:
    """Load the database and run the counter-current analysis."""
    database_path = config.get_path("database_path")
    figure_root = config.get_path("figure_root")

    df = pd.read_csv(database_path / "fishDataBase.csv")
    analyser = CounterCurrentAnalyser(df)
    # save_result joins these prefixes with filenames, so end them with "/".
    analyser.main(f"{figure_root}/", f"{figure_root}/")
    plt.show()


if __name__ == "__main__":
    main()
