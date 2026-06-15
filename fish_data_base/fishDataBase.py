# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — fish_data_base.fishDataBase                     ║
# ║  « the per-fish CSV-backed database »                            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads, grows, and saves the master fish database, running each  ║
# ║  new recording folder through fishRecAnalysis.                   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Create, grow, and persist the master CSV-backed fish database."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from deprecation import deprecated_alias, deprecated_class_alias
from fish_data_base.counterCurrentAna import SortMultiFileFolder
from fish_data_base.fishRecAnalysis import FishRecAnalysis

DATABASE_FIELDS: list[str] = [
    "genotype", "sex", "animalNo", "expType", "birthDate", "fps", "traceLenFrame",
    "traceLenSec", "inZoneFraction", "inZoneDuration", "inZoneMedDiverg_Deg",
    "path2_inZoneBendability", "path2_midLineUniform_mm", "path2_midLineUniform_pix",
    "path2_head_mm", "path2_tail_mm", "path2_probDensity", "path2_smr", "path2_s2r",
    "path2_seq", "path2_csv", "path2_mat", "path2_anaMat",
]


class FishDataBase:
    """Load, extend, and persist the master fish-recording database.

    Args:
        database_path:          Database root; defaults to ``database_path``
                                from the configured paths.
        database_file_position: CSV file; defaults to ``<root>/fishDataBase.csv``.
    """

    def __init__(
        self,
        database_path: str | Path | None = None,
        database_file_position: str | Path | None = None,
    ) -> None:
        if database_path is None:
            database_path = config.get_path("database_path")
        self.database_path = Path(database_path)
        self.data_paths = [
            "path2_inZoneBendability", "path2_midLineUniform_mm", "path2_midLineUniform_pix",
            "path2_head_mm", "path2_tail_mm", "path2_probDensity", "path2_trace_mm",
            "path2_probDensity", "path2_smr", "path2_csv",
        ]
        if database_file_position is None:
            self.database_file_position = self.database_path / "fishDataBase.csv"
        else:
            self.database_file_position = Path(database_file_position)
        self.load_database()

    def load_database(self) -> None:
        """Load the database CSV, offering to create one if it is missing."""
        try:
            self.database = pd.read_csv(self.database_file_position)
            self.database = self.database.drop(columns="Unnamed: 0", errors="ignore")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            answer = "?"
            while answer not in ("y", "n"):
                print(f"Fish data base cannot be read at position: {self.database_file_position}")
                answer = input(
                    f"Do you want to create a fish data base at "
                    f"{self.database_file_position}? (y)es or (n)o"
                )
            if answer == "n":
                raise ValueError("Cannot read fish data base") from None
            self.init_database()

    def init_database(self) -> None:
        """Create and persist an empty database with the standard fields."""
        self.database = pd.DataFrame([], columns=DATABASE_FIELDS)
        self.save_database()

    def run_multi_trace_folder(
        self,
        folder_position: str | Path,
        gene_name: str,
        experiment_str: str,
        birth_date: str,
        start_at: int = 0,
        gui_correction: bool = True,
        default_wrong_arena: str = "",
    ) -> None:
        """Analyse every not-yet-processed recording in a folder into the database.

        Args:
            folder_position: Folder of raw recording files.
            gene_name:       Gene name for the recordings.
            experiment_str:  Experiment tag.
            birth_date:      Birth date of the fish.
            start_at:        Index to start from in the sorted file list.
            gui_correction:  Whether to run interactive correction.
            default_wrong_arena: Default answer for the wrong-arena prompt.
        """
        sorter = SortMultiFileFolder(folder_position, experiment_str)
        file_dictionary = sorter.run()
        keys = list(file_dictionary.keys())

        already_analysed = [
            Path(x).name for x in self.database.path2_anaMat if isinstance(x, str)
        ]

        for key in tqdm(keys[start_at:], desc="analyse files"):
            data_dictionary = file_dictionary[key]
            if Path(data_dictionary["anaMat"]).name in already_analysed:
                continue
            analysis = FishRecAnalysis(
                data_dictionary, gene_name, experiment_str, birth_date, self.database_path
            )
            analysis.main(correction_mode=gui_correction)
            self.add_database_entry(analysis.save_data_frames())
            self.save_database()

    def add_database_entry(self, database_entry: dict) -> None:
        """Append one recording's entry to the database."""
        self.database = pd.concat(
            [self.database, pd.DataFrame([database_entry])], ignore_index=True
        )

    def integrate_database(self, file_path: str | Path) -> None:
        """Append the rows of another database CSV into this one."""
        new_dataframe = pd.read_csv(file_path)
        new_dataframe = new_dataframe.drop(columns="Unnamed: 0", errors="ignore")
        self.database = pd.concat([self.database, new_dataframe], ignore_index=True)

    def rebase_paths(self, default_path: str | Path | None = None) -> None:
        """Rewrite stored file paths from ``default_path`` to this database root.

        Args:
            default_path: Old root baked into the stored paths; defaults to the
                          configured ``database_path``.
        """
        if default_path is None:
            default_path = config.get_path("database_path")
        for path_column in self.data_paths:
            self.database[path_column] = self.database[path_column].replace(
                {str(default_path): str(self.database_path)}, regex=True
            )

    def save_database(self) -> None:
        """Persist the database to its CSV file."""
        self.database.to_csv(self.database_file_position, index=False)

    # Deprecated camelCase method names.
    addDataBase = deprecated_alias(add_database_entry, "addDataBase")
    saveDataBase = deprecated_alias(save_database, "saveDataBase")


# Deprecated lower-camelCase class name.
fishDataBase = deprecated_class_alias(FishDataBase, "fishDataBase")
