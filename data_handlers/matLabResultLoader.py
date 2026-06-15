# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — data_handlers.matLabResultLoader                ║
# ║  « read LACE MATLAB result files »                               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Loads the .mat output of the MATLAB LACE tracker and splits it  ║
# ║  into trace, contour, mid-line, head/tail, and kinematic arrays. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Load and unpack the MATLAB LACE tracker's ``.mat`` result files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from deprecation import deprecated_alias, deprecated_class_alias


class MatlabResultLoader:
    """Load a MATLAB LACE result file and expose its arrays as attributes.

    Args:
        file_position: Path to the ``.mat`` result file.
        mode:          Loader mode; only ``"anaMat"`` is supported.
    """

    def __init__(self, file_position: str | Path, mode: str = "anaMat") -> None:
        self.file_position = Path(file_position)
        self.mode = mode

    def read_ana_mat_file(self) -> None:
        """Load the analysis ``.mat`` file into meta/analysed/trace arrays."""
        mat = scipy.io.loadmat(self.file_position)
        self.metaData = mat["metaData"]
        self.analysedData = mat["analysedData"]
        self.traceResult = self.analysedData[0][0][0]

    def ndarray_to_np_array_2d(self, nd_array: np.ndarray) -> np.ndarray:
        """Convert a MATLAB cell column into a 2D NumPy array (x first).

        Args:
            nd_array: MATLAB array column to convert.

        Returns:
            2D array with x and y columns swapped so x comes first.
        """
        temp = nd_array.tolist()
        return np.fliplr(np.array([x[0][:] for x in temp]))  # fliplr so x is first

    def flatten_ndarray(self, nd_array: np.ndarray) -> list[np.ndarray]:
        """Flatten a MATLAB cell column into a list of 2D arrays (x first).

        Args:
            nd_array: MATLAB array column to flatten.

        Returns:
            List of 2D arrays, one per cell, x and y columns swapped.
        """
        temp = nd_array.tolist()
        return [np.fliplr(np.array(x[0][0].tolist())) for x in temp]  # fliplr so x is first

    def split_results_to_variables(self) -> None:
        """Split the loaded trace result into the individual data arrays.

        Trace-info columns (``traceInfo``):

        1.  x-position in pixels
        2.  y-position in pixels
        3.  major axis length of the fitted ellipse
        4.  minor axis length of the fitted ellipse
        5.  ellipse angle in degrees
        6.  quality of the fit
        7.  number of animals trusted after final evaluation
        8.  number of animals in the ellipse by surface area
        9.  number of animals in the ellipse by contour length
        10. whether the animal is close to a previously traced animal (1 == yes)
        11. evaluation weighted mean
        12. detection quality [a.u.]
        13. correction index (1 if the area was corrected automatically)
        """
        self.traceInfo = self.ndarray_to_np_array_2d(self.traceResult[:, 0])
        self.traceContour = self.flatten_ndarray(self.traceResult[:, 1])
        self.traceMidline = self.flatten_ndarray(self.traceResult[:, 2])
        self.traceHead = self.ndarray_to_np_array_2d(self.traceResult[:, 3])
        self.traceTail = self.ndarray_to_np_array_2d(self.traceResult[:, 4])
        self.trace = self.analysedData[0][0][1]
        self.bendability = [x[0][:] for x in self.analysedData[0][0][2].tolist()]
        self.binnedBend = self.analysedData[0][0][3]
        self.saccs = self.analysedData[0][0][4]
        self.trigAveSacc = self.analysedData[0][0][5]
        self.medMaxVelocities = self.analysedData[0][0][6]

    def get_data(self) -> tuple:
        """Read and unpack the file, returning all extracted arrays.

        Returns:
            Tuple of (traceInfo, traceContour, traceMidline, traceHead,
            traceTail, trace, bendability, binnedBend, saccs, trigAveSacc,
            medMaxVelocities).

        Raises:
            ValueError: If ``mode`` is not ``"anaMat"``.
        """
        if self.mode != "anaMat":
            raise ValueError(f"Unknown mode for MatlabResultLoader: {self.mode!r}")
        self.read_ana_mat_file()
        self.split_results_to_variables()
        return (
            self.traceInfo,
            self.traceContour,
            self.traceMidline,
            self.traceHead,
            self.traceTail,
            self.trace,
            self.bendability,
            self.binnedBend,
            self.saccs,
            self.trigAveSacc,
            self.medMaxVelocities,
        )

    # Deprecated camelCase method names — forward to the snake_case versions.
    readAnaMatFile = deprecated_alias(read_ana_mat_file, "readAnaMatFile")
    ndArray2npArray2D = deprecated_alias(ndarray_to_np_array_2d, "ndArray2npArray2D")
    flattenNDarray = deprecated_alias(flatten_ndarray, "flattenNDarray")
    splitResults2Variables = deprecated_alias(
        split_results_to_variables, "splitResults2Variables"
    )
    getData = deprecated_alias(get_data, "getData")


# Deprecated lower-camelCase class name.
matLabResultLoader = deprecated_class_alias(MatlabResultLoader, "matLabResultLoader")
