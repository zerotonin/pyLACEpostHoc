# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — fish_data_base.counterCurrentAna                ║
# ║  « sort a folder of recording files »                            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Groups a folder of raw files by genotype, sex, and animal number ║
# ║  into per-recording dictionaries for the analyser.               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Group a folder of raw recording files into per-recording dictionaries."""
from __future__ import annotations

import os
import re
from pathlib import Path

from deprecation import deprecated_alias, deprecated_class_alias

EXPERIMENT_NAMES: dict[str, str] = {
    "CCur": "counter current",
    "Ta": "motivated swimming",
    "Unt": "free swiming",
    "cst": "c-start",
}


class SortMultiFileFolder:
    """Sort a folder of raw files into one dictionary per recording.

    Args:
        source_path:       Folder containing the raw recording files.
        experiment_string: Experiment tag used to name the experiment type.
    """

    def __init__(self, source_path: str | Path, experiment_string: str) -> None:
        self.source_path = source_path
        self.file_dict: dict = {}
        self.experiment_string = experiment_string

    def extract_genotype_number_sex(self, file_name: str, tag: str) -> tuple[str, int, str]:
        """Parse genotype, animal number, and sex from a 2-letter-tag file name."""
        index = file_name.find(tag)
        genotype = file_name[index:index + 2]
        sex = file_name[index + 2:index + 3]
        number = re.sub("[^0-9]", "", file_name[index + 3:index + 6])
        return genotype, int(number), sex

    def extract_genotype_number_sex_4int_wt(self, file_name: str, tag: str) -> tuple[str, int, str]:
        """Parse genotype, animal number, and sex for internal wild-type names."""
        index = file_name.find(tag)
        genotype = file_name[index:index + 3]
        sex = file_name[index + 3:index + 4]
        number = re.sub("[^0-9]", "", file_name[index + 4:index + 7])
        return genotype, int(number), sex

    def get_file_type(self, extension: str) -> str:
        """Return the file extension without the dot, lower-cased."""
        return extension[1:].lower()

    def make_dataset_key(self, genotype: str, animal_no: int, sex: str) -> str:
        """Build a unique dataset key from genotype, sex, and animal number."""
        return genotype + sex + str(animal_no)

    def classify_file(self, file_name: str, ext: str) -> tuple[str, int, str, str]:
        """Classify a file into (genotype, animal number, sex, file type)."""
        file_name_upper = file_name.upper()
        if "HMF" in file_name_upper:
            genotype, animal_no, sex = self.extract_genotype_number_sex(file_name_upper, "HMF")
        elif "HMM" in file_name_upper:
            genotype, animal_no, sex = self.extract_genotype_number_sex(file_name_upper, "HMM")
        elif "HTF" in file_name_upper:
            genotype, animal_no, sex = self.extract_genotype_number_sex(file_name_upper, "HTF")
        elif "HTM" in file_name_upper:
            genotype, animal_no, sex = self.extract_genotype_number_sex(file_name_upper, "HTM")
        elif "INTF" in file_name_upper:
            genotype, animal_no, sex = self.extract_genotype_number_sex_4int_wt(file_name_upper, "INTF")
        elif "INTM" in file_name_upper:
            genotype, animal_no, sex = self.extract_genotype_number_sex_4int_wt(file_name_upper, "INTM")
        elif "INTWF" in file_name_upper:
            file_name_upper = file_name_upper.replace("INTW", "INT")
            genotype, animal_no, sex = self.extract_genotype_number_sex_4int_wt(file_name_upper, "INTF")
        elif "INTWM" in file_name_upper:
            file_name_upper = file_name_upper.replace("INTW", "INT")
            genotype, animal_no, sex = self.extract_genotype_number_sex_4int_wt(file_name_upper, "INTM")
        else:
            genotype, animal_no, sex = ("N/A", -1, "N/A")
            print("file seems wrongly named: ", file_name)
        return genotype, animal_no, sex, self.get_file_type(ext)

    def update_file_dict(self, file_data_tuple: tuple, data_set_key: str, file_path) -> None:
        """Ensure a dataset entry exists, then record this file's path."""
        if data_set_key not in self.file_dict:
            self.file_dict[data_set_key] = self.initialise_data_dict(file_data_tuple)
        self.update_data_dict(data_set_key, file_data_tuple, file_path)

    def initialise_data_dict(self, file_data_tuple: tuple) -> dict:
        """Build an empty per-recording dictionary from a classified tuple.

        The file-position keys are: ``smr`` (Mauthner setup file), ``s2r``
        (Mauthner data file), ``seq`` (Norpix movie), ``csv`` (tank bounding
        box), ``mat`` (LACE trace), and ``anaMat`` (LACE analysis).
        """
        return {
            "genotype": file_data_tuple[0],
            "sex": file_data_tuple[2],
            "animalNo": file_data_tuple[1],
            "expType": self.get_full_experiment_name(),
            "smr": "", "s2r": "", "seq": "", "csv": "", "mat": "", "anaMat": "",
        }

    def get_full_experiment_name(self) -> str:
        """Return the full experiment name for the experiment tag."""
        try:
            return EXPERIMENT_NAMES[self.experiment_string]
        except KeyError:
            raise ValueError(
                "sortMultiFileFolder: get_full_experiment_name: "
                f"unknown experiment string: {self.experiment_string}"
            ) from None

    def update_data_dict(self, data_set_key: str, file_data_tuple: tuple, file_path) -> None:
        """Record a file's path under its dataset, splitting ana/raw MATLAB files."""
        if file_data_tuple[3] == "mat":
            if str(file_path)[-7:-4].lower() == "ana":
                self.file_dict[data_set_key]["anaMat"] = str(file_path)
            else:
                self.file_dict[data_set_key]["mat"] = str(file_path)
        else:
            self.file_dict[data_set_key][file_data_tuple[3]] = str(file_path)

    def run(self) -> dict:
        """Scan the source folder and return the per-recording dictionaries."""
        result = list(Path(self.source_path).rglob("*.*"))
        self.file_dict = {}
        for file_path in result:
            file_name, ext = os.path.splitext(os.path.basename(file_path))
            file_data_tuple = self.classify_file(file_name, ext)
            data_set_key = self.make_dataset_key(*file_data_tuple[:3])
            self.update_file_dict(file_data_tuple, data_set_key, file_path)
        return self.file_dict

    # Deprecated dunder-style entry point.
    __main__ = deprecated_alias(run, "__main__")


# Deprecated lower-camelCase class name.
sortMultiFileFolder = deprecated_class_alias(SortMultiFileFolder, "sortMultiFileFolder")
