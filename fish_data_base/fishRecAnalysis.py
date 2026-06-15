# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — fish_data_base.fishRecAnalysis                  ║
# ║  « analyse one recording into the database »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Orchestrates correction, trajectory, and spike analysis of one  ║
# ║  recording and writes its result frames and a database entry.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Analyse a single zebrafish recording into per-fish database frames."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_handlers.spike2SimpleIO import SegmentSaver, Spike2SimpleReader
from deprecation import deprecated_alias, deprecated_class_alias
from fish_data_base import result_frames
from trace_analysis.SpikeDetector import SpikeDetector
from trace_analysis.traceAnalyser import TraceAnalyser
from trace_analysis.traceCorrector import TraceCorrector

# Known arena sizes per experiment type; each tuple is (y, x) in millimetres.
ARENA_SIZES: dict[str, tuple[int, int]] = {
    "cruise": (114, 248),
    "c_start": (40, 80),
    "counter_current": (45, 167),
}


class FishRecAnalysis:
    """Analyse one recording and assemble its database entry and result frames.

    Args:
        dataDict:     File positions and metadata for the recording.
        genName:      Gene name, prepended to the genotype.
        expStr:       Experiment tag (``CCur``, ``Ta``, ``Unt``, ``cst``).
        birthDate:    Birth date of the fish.
        dataBasePath: Database root; defaults to the configured ``database_path``.
    """

    def __init__(
        self,
        dataDict: dict,
        genName: str,
        expStr: str,
        birthDate: str,
        dataBasePath: str | Path | None = None,
    ) -> None:
        if dataBasePath is None:
            dataBasePath = config.get_path("database_path")
        self.dbPath = Path(dataBasePath)
        self.dataDict = dataDict
        self.genName = genName
        self.dataDict["genotype"] = self.genName + "-" + self.dataDict["genotype"]
        self.dataDict["birthDate"] = birthDate
        self.expStr = expStr
        self.arena_sizes = ARENA_SIZES
        self.dataList: list = []

    def make_save_folder(self) -> None:
        """Create the per-recording output folder named from its metadata."""
        rec_number = len([p for p in self.dbPath.iterdir() if p.is_dir()])
        folder_name = (
            f"{self.expStr}_{self.dataDict['genotype']}_{self.dataDict['birthDate']}_"
            f"{self.dataDict['sex']}_{self.dataDict['animalNo']}_ID#{rec_number}"
        )
        self.savePath = self.dbPath / folder_name
        self.savePath.mkdir()

    def correction_analysis(self, correction_mode: bool) -> None:
        """Load the trace corrector and, if needed, calibrate interactively."""
        matlab_files_loaded = True
        try:
            self.traCor = TraceCorrector(self.dataDict)
        except Exception:
            matlab_files_loaded = False
            self.traCor = None

        if matlab_files_loaded and not self.traCor.mmTraceAvailable and correction_mode:
            self.traCor.calibrate_tracking()
        try:
            self.traCor.close_figure()
        except Exception:
            pass

    def analyse_trajectory(self) -> None:
        """Convert to millimetres and run the experiment-specific analysis."""
        self.traAna = TraceAnalyser(self.traCor, self.get_arena_size_by_experiment_tag())
        if not self.traCor.mmTraceAvailable:
            self.traAna.pixel_trajectories_to_mm()
        if self.expStr == "CCur":
            self.traAna.calculate_spatial_histogram()
            self.traAna.in_zone_analyse()
        try:
            self.traAna.get_uniform_mid_line()
        except Exception:
            self.traAna.midLineUniform_mm = None
        self.traAna.export_meta_dict()
        self.dataList = self.traAna.export_data_list()

    def analyse_spiketrain(self) -> None:
        """Detect spikes from the Spike2 file and append the results."""
        try:
            spike_train_df, spike_properties = self.process_spike_data()
            self.dataList.append(["spike_train_df", spike_train_df, 2])
            self.dataList.append(["spike_properties", spike_properties, 2])
        except Exception:
            self.dataList.append(["spike_train_df", None, 2])
            self.dataList.append(["spike_properties", None, 2])

    def main(self, correction_mode: bool = True) -> None:
        """Run correction, trajectory, and (for c-start) spike-train analysis."""
        self.correction_analysis(correction_mode)
        if self.traCor is not None:
            self.analyse_trajectory()
        if self.expStr == "cst":
            self.analyse_spiketrain()

    def process_spike_data(self) -> tuple[pd.DataFrame, dict]:
        """Read the Spike2 file and detect spikes into a train and properties."""
        reader = Spike2SimpleReader(self.dataDict["smr"])
        reader.main()
        saver = SegmentSaver(reader, "no csv file will be produced")
        df = saver.main()[0]
        detector = SpikeDetector(df)
        return detector.main()

    # ── result DataFrame builders (thin wrappers over result_frames) ────
    def prep_df_3d(self, col1_name: str, col2_name: str, reps: int):
        """Empty 3D frame and its column labels."""
        return result_frames.prep_df_3d(col1_name, col2_name, reps)

    def get_time_index(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Convert a frame index to seconds using the trace fps."""
        return result_frames.to_time_index(data_df, self.traAna.fps)

    def make_pandas_df_3d(self, data, col1_name, col2_name, index=None) -> pd.DataFrame:
        """Build a 3D result frame (uses the trace fps for a time index)."""
        return result_frames.make_df_3d(data, col1_name, col2_name, self.traAna.fps, index)

    def make_pandas_df_2d(self, data, col1_name, col2_name, index=None) -> pd.DataFrame:
        """Build a 2D result frame (uses the trace fps for a time index)."""
        return result_frames.make_df_2d(data, col1_name, col2_name, self.traAna.fps, index)

    def make_pandas_df_4hist(self, data) -> pd.DataFrame:
        """Build the occupancy-histogram result frame."""
        return result_frames.make_df_hist(data)

    def make_result_dfs(self) -> dict:
        """Assemble all analysed arrays into their named result DataFrames."""
        return_dict = {
            "inZoneBendability": None, "midLineUniform_mm": None, "midLineUniform_pix": None,
            "head_mm": None, "tail_mm": None, "probDensity": None, "trace_mm": None,
            "spike_train_df": None, "spike_properties": None,
        }
        for data in self.dataList:
            if data[0] == "inZoneBendability":
                return_dict["inZoneBendability"] = self.make_pandas_df_3d(data[1], "bodyAxis", "angle")
            elif data[0] == "midLineUniform_mm":
                return_dict["midLineUniform_mm"] = self.make_pandas_df_3d(data[1], "x_coord", "y_coord", "Time")
            elif data[0] == "midLineUniform_pix":
                return_dict["midLineUniform_pix"] = self.make_pandas_df_3d(data[1], "x_coord", "y_coord", "Time")
            elif data[0] == "head_mm":
                return_dict["head_mm"] = self.make_pandas_df_2d(data[1], "x_coord", "y_coord", "Time")
            elif data[0] == "tail_mm":
                return_dict["tail_mm"] = self.make_pandas_df_2d(data[1], "x_coord", "y_coord", "Time")
            elif data[0] == "trace_mm":
                return_dict["trace_mm"] = pd.DataFrame(
                    self.traAna.trace_mm,
                    columns=["x_position_mm", "y_position_mm", "yaw_rad",
                             "thrust_m/s", "slip_m/s", "yaw_deg/s"],
                )
            elif data[0] == "probDensity":
                return_dict["probDensity"] = self.make_pandas_df_4hist(data[1])
            elif data[0] == "spike_train_df":
                return_dict["spike_train_df"] = data[1]
        return return_dict

    def save_data_frames(self) -> dict:
        """Write every result frame to CSV and return the database entry."""
        data_frames = self.make_result_dfs()
        self.make_save_folder()
        for key, frame in data_frames.items():
            if frame is not None:
                save_pos = self.savePath / f"{key}.csv"
                self.dataDict["path2_" + key] = str(save_pos)
                frame.to_csv(save_pos)
        return self.make_database_entry()

    def make_database_entry(self) -> dict:
        """Build the database row: prune helper keys and rename file paths."""
        db_entry = self.dataDict.copy()
        for tag in ["movieFrameIDX", "probDensity_xCenters", "probDensity_yCenters"]:
            db_entry.pop(tag, None)
        for new_key, old_key in [
            ("path2_smr", "smr"), ("path2_s2r", "s2r"), ("path2_seq", "seq"),
            ("path2_csv", "csv"), ("path2_mat", "mat"), ("path2_anaMat", "anaMat"),
        ]:
            if old_key in db_entry:
                db_entry[new_key] = db_entry.pop(old_key)
            else:
                db_entry[new_key] = None
        if self.expStr == "cst":
            spike_properties = self.dataList[-1][1]
            if spike_properties is not None:
                db_entry = {**db_entry, **spike_properties}
        return db_entry

    def save_2d_matrix(self, data_list_entry: list) -> None:
        """Save a 2D matrix to a text file and record its path."""
        tag, mat = data_list_entry[0], data_list_entry[1]
        file_position = self.savePath / f"{tag}.txt"
        np.savetxt(file_position, mat)
        self.dataDict["path2_" + tag] = str(file_position)

    def save_3d_matrix(self, data_list_entry: list) -> None:
        """Save a 3D matrix by reshaping it to 2D first."""
        temp = deepcopy(data_list_entry)
        temp[1] = temp[1].reshape(temp[1].shape[0], -1)
        self.save_2d_matrix(temp)

    def load_2d_matrix(self, file_position: str | Path) -> np.ndarray:
        """Load a 2D matrix from a text file."""
        return np.loadtxt(file_position)

    def load_3d_matrix(self, file_position: str | Path) -> np.ndarray:
        """Load a 3D matrix saved as 2D and restore its original shape."""
        temp = self.load_2d_matrix(file_position)
        return temp.reshape(temp.shape[0], temp.shape[1] // temp.shape[2], temp.shape[2])

    def check_mm_trace(self, default_answer: str = "x") -> None:
        """Warn (and offer a fix) if the trace exceeds the expected arena."""
        size = self.get_arena_size_by_experiment_tag()
        if (np.max(self.traAna.trace_mm[:, 0]) > size[0]
                or np.max(self.traAna.trace_mm[:, 1]) > size[1]):
            self.wrong_arena_dlg(size, default_answer)

    def wrong_arena_dlg(self, expected_size: tuple, default_answer: str = "x") -> None:
        """Prompt the user about an out-of-bounds trace and optionally rescale."""
        print("=" * 79)
        print("| The current file has trajectory coordinates outside the experimental setup! |")
        print("=" * 79)
        print(f"\nAnalysed MatLab file: {self.dataDict['anaMat']}")
        print(f"Experiment string: {self.expStr} | Expected arena size (y, x): {expected_size}")
        print(
            "Found maximal coordinates (y, x): "
            f"({np.max(self.traAna.trace_mm[:, 0])}, {np.max(self.traAna.trace_mm[:, 1])})"
        )

        while default_answer not in "ACTSN":
            default_answer = input(
                "Which arena was WRONGLY used? (A)bort, (C)ounter current, cruise (T)ank, "
                "C-(S)tart, or (N)one all is fine: "
            ).upper()

        if default_answer == "A":
            raise ValueError(f"Aborted file due to user input: {self.dataDict['anaMat']}")
        if default_answer == "C":
            self.interp_trace_mm(*expected_size, *self.arena_sizes["counter_current"])
        elif default_answer == "T":
            self.interp_trace_mm(*expected_size, *self.arena_sizes["cruise"])
        elif default_answer == "S":
            self.interp_trace_mm(*expected_size, *self.arena_sizes["c_start"])
        else:
            print("Nothing was changed")

    def interp_trace_mm(self, y_length: float, x_length: float, y_old: float, x_old: float) -> None:
        """Rescale a millimetre trace computed with the wrong arena dimensions.

        Translational velocities become approximations after rescaling.
        """
        y_factor = y_length / y_old
        x_factor = x_length / x_old
        mix_factor = (x_factor + y_factor) / 2.0

        # MATLAB traces store x first, y second.
        self.traAna.trace_mm[:, 0] = self.traAna.trace_mm[:, 0] * x_factor
        self.traAna.trace_mm[:, 1] = self.traAna.trace_mm[:, 1] * y_factor
        self.traAna.trace_mm[:, 3] = self.traAna.trace_mm[:, 3] * mix_factor
        self.traAna.trace_mm[:, 4] = self.traAna.trace_mm[:, 4] * mix_factor
        self.traAna.medMaxVelocities[:, 0:2] = self.traAna.medMaxVelocities[:, 0:2] * mix_factor

    def get_arena_size_by_experiment_tag(self) -> tuple[int, int]:
        """Return the (y, x) arena size for this recording's experiment tag."""
        if self.expStr == "CCur":
            return self.arena_sizes["counter_current"]
        if self.expStr in ("Ta", "Unt"):
            return self.arena_sizes["cruise"]
        if self.expStr == "cst":
            return self.arena_sizes["c_start"]
        raise ValueError(
            f"fishRecAnalysis:get_arena_size_by_experiment_tag: "
            f"Unknown experiment string: {self.expStr}"
        )

    # Deprecated camelCase method names.
    makeSaveFolder = deprecated_alias(make_save_folder, "makeSaveFolder")
    correctionAnalysis = deprecated_alias(correction_analysis, "correctionAnalysis")
    prepDf_3D = deprecated_alias(prep_df_3d, "prepDf_3D")
    getTimeIndex = deprecated_alias(get_time_index, "getTimeIndex")
    makePandasDF_3D = deprecated_alias(make_pandas_df_3d, "makePandasDF_3D")
    makePandasDF_2D = deprecated_alias(make_pandas_df_2d, "makePandasDF_2D")
    makePandasDF4Hist = deprecated_alias(make_pandas_df_4hist, "makePandasDF4Hist")
    makeResultDFs = deprecated_alias(make_result_dfs, "makeResultDFs")
    saveDataFrames = deprecated_alias(save_data_frames, "saveDataFrames")
    makeDataBaseEntry = deprecated_alias(make_database_entry, "makeDataBaseEntry")
    save2DMatrix = deprecated_alias(save_2d_matrix, "save2DMatrix")
    save3DMatrix = deprecated_alias(save_3d_matrix, "save3DMatrix")
    load2DMatrix = deprecated_alias(load_2d_matrix, "load2DMatrix")
    load3DMatrix = deprecated_alias(load_3d_matrix, "load3DMatrix")


# Deprecated lower-camelCase class name.
fishRecAnalysis = deprecated_class_alias(FishRecAnalysis, "fishRecAnalysis")
