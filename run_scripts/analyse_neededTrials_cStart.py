# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — run_scripts.analyse_neededTrials_cStart         ║
# ║  « c-start survival analysis »                                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Kaplan-Meier and Cox survival analysis of behavioural response  ║
# ║  across trials by genotype and sex.                              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Survival analysis of c-start behavioural response across trials.

Kaplan-Meier curves, pairwise log-rank tests, and a Cox proportional-hazards
model over genotype and sex. The original analysis was not statistically
significant, possibly from too few repetitions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

import config

GENOTYPE_CODES = {"Int": 0, "Ht": 1, "Hm": 2}


def load_trials(csv_path) -> pd.DataFrame:
    """Load and clean the c-start trial CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Sex"] = df["Sex"].str.strip()
    df["Genotype"] = df["Genotype"].astype("category")
    df["Sex"] = df["Sex"].astype("category")
    return df


def plot_survival_curves(df: pd.DataFrame) -> None:
    """Plot Kaplan-Meier survival curves per sex and genotype."""
    kmf = KaplanMeierFitter()
    for sex in ["M", "F"]:
        for label, grouped in df[df["Sex"] == sex].groupby("Genotype"):
            kmf.fit(grouped["Trial"], event_observed=grouped["Behav_Resp"], label=f"{label}_{sex}")
            kmf.plot()


def print_logrank_tests(df: pd.DataFrame) -> None:
    """Print pairwise log-rank tests between genotypes."""
    groups = {name: df[df["Genotype"] == name] for name in ("Int", "Ht", "Hm")}
    for a, b in [("Int", "Ht"), ("Int", "Hm"), ("Hm", "Ht")]:
        result = logrank_test(
            groups[a]["Trial"], groups[b]["Trial"],
            event_observed_A=groups[a]["Behav_Resp"], event_observed_B=groups[b]["Behav_Resp"],
        )
        print(f"\n{a} vs {b}")
        result.print_summary()


def fit_cox_model(df: pd.DataFrame) -> None:
    """Fit and summarise a Cox proportional-hazards model."""
    df = df.copy()
    df["Genotype"] = df["Genotype"].map(GENOTYPE_CODES).apply(int)
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="Trial", event_col="Behav_Resp")
        cph.print_summary()
    except Exception as exc:
        print("Error occurred:", exc)


def main() -> None:
    """Run the full c-start survival analysis."""
    df = load_trials(config.get_path("database_path") / "c-start_trial.csv")
    plot_survival_curves(df)
    print_logrank_tests(df)
    fit_cox_model(df)
    plt.show()


if __name__ == "__main__":
    main()
