# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — trace_analysis.traceAnalyser                    ║
# ║  « pixel traces to millimetre kinematics »                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Turns a corrected pixel trace into millimetre trajectories,     ║
# ║  spatial histograms, in-zone metrics, and uniform mid-lines.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Convert a corrected pixel trace into millimetre-space kinematics."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d

from deprecation import deprecated_alias, deprecated_class_alias

# Rectangular in-zone margins in millimetres: [[x_min, y_min], [x_max, y_max]].
ZONE_MARGINS: np.ndarray = np.array([[40, 11.5], [163, 31.5]])
DEFAULT_SPATIAL_BINS: list[int] = [16, 8]
DEFAULT_MIDLINE_POINTS: int = 10


class TraceAnalyser:
    """Analyse a fish trace, producing millimetre-space kinematic measures.

    Takes a ``traceCorrector``-like object and derives trajectories, spatial
    occupancy, in-zone behaviour, and uniform mid-lines.

    Args:
        traceCorrectorObj:  Source of corrected pixel traces and metadata.
        default_arena_size: (y, x) arena extent in millimetres for the
                            experiment.
    """

    def __init__(self, traceCorrectorObj, default_arena_size) -> None:
        self.mm_tra_available = traceCorrectorObj.mmTraceAvailable
        self._load_trace_data(traceCorrectorObj)
        self._load_movie_meta(traceCorrectorObj)
        self.make_movie_idx()
        self._load_metadata(traceCorrectorObj, default_arena_size)
        self._setup_arena(traceCorrectorObj)
        self.zoneMargins = ZONE_MARGINS
        self._preallocate(traceCorrectorObj)

    def _load_trace_data(self, obj) -> None:
        """Copy the pixel-based and body-based trace arrays off the source."""
        # fish data -> pixel based
        self.head_pix = obj.head
        self.tail_pix = obj.tail
        self.contour_pix = obj.contour
        self.midLine_pix = obj.midLine
        # fish data -> body based
        self.bendability = obj.matLabLoader.bendability
        self.binnedBend = obj.matLabLoader.binnedBend
        self.saccs = obj.matLabLoader.saccs
        self.trigAveSacc = obj.matLabLoader.trigAveSacc
        self.medMaxVelocities = obj.matLabLoader.medMaxVelocities

    def _load_movie_meta(self, obj) -> None:
        """Copy movie timing metadata and derive the trace length in seconds."""
        self.headerDict = obj.headerDict
        self.pixelOffset = obj.pixelOffset
        self.frameOffset = obj.frameShift
        self.traceLenFrame = obj.allocated_frames
        self.originFrame = obj.originFrame
        self.fps = obj.fps
        self.traceLenSec = self.traceLenFrame / self.fps

    def _load_metadata(self, obj, default_arena_size) -> None:
        """Copy genotype/sex/animal metadata and the arena size."""
        self.genotype = obj.dataDict["genotype"]
        self.sex = obj.dataDict["sex"]
        self.animalNo = obj.dataDict["animalNo"]
        self.arena_size_by_experiment = default_arena_size
        self.dataList = []

    def _setup_arena(self, obj) -> None:
        """Build arena coordinates and the pixel→mm interpolator, or use mm."""
        if not self.mm_tra_available:
            # FIXME(flagged): the 4th corner uses arena_size[1] for its y value;
            # for a rectangle it is expected to be arena_size[0]. Preserved
            # as-is pending confirmation — see the Sprint 2 notes.
            self.arenaCoords_mm = np.array([
                [0, 0],
                [self.arena_size_by_experiment[1], 0],
                [self.arena_size_by_experiment[1], self.arena_size_by_experiment[0]],
                [0, self.arena_size_by_experiment[1]],
            ])
            self.arenaCoords_pix = obj.boxCoords
            self.sort_coords_arena_pix()
            self.make_interpolator()
            self.yaw = obj.matLabLoader.trace[:, 2]
        else:
            self.trace_mm = obj.matLabLoader.trace

    def _preallocate(self, obj) -> None:
        """Initialise the export dict and the result placeholders to None."""
        self.exportDict = obj.dataDict
        self.inZoneFraction = None
        self.inZoneDuration = None
        self.probDensity_xCenters = None
        self.probDensity_yCenters = None
        self.inZoneBendability = None
        self.midLineUniform_mm = None
        self.midLineUniform_pix = None
        self.head_mm = None
        self.tail_mm = None
        self.contour_mm = None
        self.midLine_mm = None
        self.probDensity = None
        self.medianDivergenceFromStraightInZone_DEG = None

    def export_meta_dict(self) -> dict:
        """Return the metadata dict augmented with the scalar trace results."""
        self.exportDict["movieFrameIDX"] = self.movieIDX
        self.exportDict["fps"] = self.fps
        self.exportDict["traceLenFrame"] = self.traceLenFrame
        self.exportDict["traceLenSec"] = self.traceLenSec
        self.exportDict["inZoneFraction"] = self.inZoneFraction
        self.exportDict["inZoneDuration"] = self.inZoneDuration
        self.exportDict["inZoneMedDiverg_Deg"] = self.medianDivergenceFromStraightInZone_DEG
        self.exportDict["probDensity_xCenters"] = self.probDensity_xCenters
        self.exportDict["probDensity_yCenters"] = self.probDensity_yCenters
        self.exportDict["path2_inZoneBendability"] = None
        self.exportDict["path2_midLineUniform_mm"] = None
        self.exportDict["path2_midLineUniform_pix"] = None
        self.exportDict["path2_head_mm"] = None
        self.exportDict["path2_tail_mm"] = None
        self.exportDict["path2_probDensity"] = None
        return self.exportDict

    def export_data_list(self) -> list:
        """Return the array results as ``[name, array, ndim]`` rows for saving."""
        if self.inZoneBendability is not None:
            self.dataList.append(["inZoneBendability", self.inZoneBendability, 3])
        if self.midLineUniform_mm is not None:
            self.dataList.append(["midLineUniform_mm", np.array(self.midLineUniform_mm), 3])
        if self.midLineUniform_pix is not None:
            self.dataList.append(["midLineUniform_pix", np.array(self.midLineUniform_pix), 3])
        if self.head_mm is not None:
            self.dataList.append(["head_mm", self.head_mm, 2])
        if self.tail_mm is not None:
            self.dataList.append(["tail_mm", self.tail_mm, 2])
        if self.probDensity is not None:
            self.dataList.append(["probDensity", self.probDensity, 2])
        if self.mm_tra_available:
            self.dataList.append(["trace_mm", self.trace_mm, 2])
        return self.dataList

    def make_movie_idx(self) -> None:
        """Build the wrapped movie frame index aligned to the trace origin."""
        if self.frameOffset < 0:
            frame_shift = self.frameOffset + self.traceLenFrame
        else:
            frame_shift = self.frameOffset
        self.movieIDX = (
            np.arange(self.traceLenFrame) + self.originFrame + frame_shift
        ) % self.traceLenFrame

    def sort_coords_arena_pix(self) -> None:
        """Order the four arena pixel corners consistently (by y then x)."""
        desc_y = np.flipud(self.arenaCoords_pix[np.argsort(self.arenaCoords_pix[:, 1])])
        low_row = desc_y[0:2, :]
        high_row = desc_y[2:, :]
        self.arenaCoords_pix = np.vstack(
            (low_row[np.argsort(low_row[:, 0])], np.flipud(high_row[np.argsort(high_row[:, 0])]))
        )

    def make_interpolator(self) -> None:
        """Build x and y pixel→millimetre interpolators from the arena corners."""
        x = self.arenaCoords_pix[:, 0]
        y = self.arenaCoords_pix[:, 1]
        self.interpX = LinearNDInterpolator(list(zip(x, y)), self.arenaCoords_mm[:, 0])
        self.interpY = LinearNDInterpolator(list(zip(x, y)), self.arenaCoords_mm[:, 1])

    def interpolate_to_mm(self, coords_2d: np.ndarray) -> np.ndarray:
        """Map an (N, 2) array of pixel coordinates into millimetre space."""
        return np.vstack((self.interpX(coords_2d), self.interpY(coords_2d))).T

    def pixel_trajectories_to_mm(self) -> None:
        """Convert head/tail/contour/mid-line pixel traces to millimetres.

        Each conversion is attempted independently; one that fails (e.g. an
        out-of-hull point the interpolator cannot map) leaves that result as
        ``None`` rather than aborting the whole trace.
        """
        try:
            self.head_mm = self.interpolate_to_mm(self.head_pix)
        except Exception:
            self.head_mm = None
        try:
            self.tail_mm = self.interpolate_to_mm(self.tail_pix)
        except Exception:
            self.tail_mm = None
        try:
            self.contour_mm = [self.interpolate_to_mm(x) for x in self.contour_pix]
        except Exception:
            self.contour_mm = None
        try:
            self.midLine_mm = [self.interpolate_to_mm(x) for x in self.midLine_pix]
        except Exception:
            self.midLine_mm = None

        if not self.mm_tra_available and self.head_mm is not None and self.tail_mm is not None:
            self.create_trace_mm_denovo()
            self.mm_tra_available = True

    def create_trace_mm_denovo(self) -> None:
        """Build ``trace_mm`` from head/tail midpoints, yaw, and speeds."""
        trace_mm = (self.head_mm + self.tail_mm) / 2.0
        yaw = self.yaw.reshape(-1, 1)
        trace_mm = np.hstack((trace_mm, yaw))

        speeds = np.diff(trace_mm, axis=0) * self.fps
        speeds = np.vstack((speeds, np.full((1, 3), np.nan)))
        speeds[:, 2] = np.rad2deg(speeds[:, 2])

        self.trace_mm = np.hstack((trace_mm, speeds))

    def calculate_spatial_histogram(self, bins: list[int] | None = None) -> None:
        """Compute the 2D occupancy probability density of the trace."""
        if bins is None:
            bins = DEFAULT_SPATIAL_BINS
        if self.mm_tra_available:
            # MATLAB trajectories are x then y, so the indices are flipped here.
            temp = np.histogram2d(self.trace_mm[:, 1], self.trace_mm[:, 0], bins, density=True)
        else:
            all_mid_line = np.vstack((self.midLine_mm[:]))
            temp = np.histogram2d(all_mid_line[:, 0], all_mid_line[:, 1], bins, density=True)
        self.probDensity = temp[0].T
        self.probDensity_xCenters = temp[1]
        self.probDensity_yCenters = temp[2]

    def calculate_in_zone_idx(self) -> None:
        """Flag each frame where the fish lies fully within the zone margins."""
        self.zoneIDX = []
        for frame_i in range(self.traceLenFrame):
            mmt = self.trace_mm[frame_i, :]
            if self.mm_tra_available:
                bool_tests = [
                    mmt[0] >= self.zoneMargins[0, 0],
                    mmt[1] >= self.zoneMargins[0, 1],
                    mmt[0] <= self.zoneMargins[1, 0],
                    mmt[1] <= self.zoneMargins[1, 1],
                ]
            else:
                # whole body must be inside the margins
                mid_line = self.midLine_mm[frame_i]
                bool_tests = [
                    (mid_line[:, 0] >= self.zoneMargins[0, 0]).all(),
                    (mid_line[:, 1] >= self.zoneMargins[0, 1]).all(),
                    (mid_line[:, 0] <= self.zoneMargins[1, 0]).all(),
                    (mid_line[:, 1] <= self.zoneMargins[1, 1]).all(),
                ]
            self.zoneIDX.append(all(bool_tests))

    def in_zone_analyse(self) -> None:
        """Compute in-zone fraction, duration, bendability, and divergence."""
        self.calculate_in_zone_idx()
        self.inZoneFraction = sum(self.zoneIDX) / self.traceLenFrame
        self.inZoneDuration = self.inZoneFraction * self.traceLenSec
        self.inZoneBendability = [
            value for idx, value in enumerate(self.bendability) if self.zoneIDX[idx]
        ]
        self.medianDivergenceFromStraightInZone_DEG = np.median(
            [np.sum(np.abs(x[:, 1] - 180)) for x in self.inZoneBendability]
        )

    def calculate_body_length(self, mid_line: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the mid-line body length and the cumulative length axis."""
        vector_norms = np.linalg.norm(np.diff(mid_line, axis=0), axis=1)
        body_len = vector_norms.sum()
        body_axis = np.cumsum(np.insert(vector_norms, 0, 0.0, axis=0))
        return body_len, body_axis

    def interp_mid_line(self, mid_line: np.ndarray, step: int = DEFAULT_MIDLINE_POINTS) -> np.ndarray:
        """Resample a mid-line to ``step`` points evenly spaced along the body."""
        _body_len, body_axis = self.calculate_body_length(mid_line)
        f_x = interp1d(body_axis, mid_line[:, 0], kind="cubic")
        f_y = interp1d(body_axis, mid_line[:, 1], kind="cubic")
        new_body_axis = np.linspace(0, body_axis[-1], step)
        return np.vstack((f_x(new_body_axis), f_y(new_body_axis))).T

    def get_uniform_mid_line(self, mid_line_points: int = DEFAULT_MIDLINE_POINTS) -> None:
        """Build uniform mid-lines for pixel (and mm, if needed) data."""
        self.midLineUniform_pix = self.get_uniform_midline_subroutine(
            self.midLine_pix, mid_line_points
        )
        if not self.mm_tra_available:
            self.midLineUniform_mm = self.get_uniform_midline_subroutine(
                self.midLine_mm, mid_line_points
            )

    def get_uniform_midline_subroutine(self, mid_line, mid_line_points: int) -> np.ndarray:
        """Resample every frame's mid-line and stack into one 3D array."""
        mid_line_result = [self.interp_mid_line(mid_line_frame, mid_line_points)
                           for mid_line_frame in mid_line]
        return np.array(mid_line_result)

    # Deprecated camelCase method names.
    exportMetaDict = deprecated_alias(export_meta_dict, "exportMetaDict")
    exportDataList = deprecated_alias(export_data_list, "exportDataList")
    makeMovieIDX = deprecated_alias(make_movie_idx, "makeMovieIDX")
    sortCoordsArenaPix = deprecated_alias(sort_coords_arena_pix, "sortCoordsArenaPix")
    makeInterpolator = deprecated_alias(make_interpolator, "makeInterpolator")
    interpolate2mm = deprecated_alias(interpolate_to_mm, "interpolate2mm")
    pixelTrajectories2mmTrajectories = deprecated_alias(
        pixel_trajectories_to_mm, "pixelTrajectories2mmTrajectories"
    )
    calculateSpatialHistogram = deprecated_alias(
        calculate_spatial_histogram, "calculateSpatialHistogram"
    )
    calculateInZoneIDX = deprecated_alias(calculate_in_zone_idx, "calculateInZoneIDX")
    inZoneAnalyse = deprecated_alias(in_zone_analyse, "inZoneAnalyse")
    calculateBodyLength = deprecated_alias(calculate_body_length, "calculateBodyLength")
    interpMidLine = deprecated_alias(interp_mid_line, "interpMidLine")
    getUniformMidLine = deprecated_alias(get_uniform_mid_line, "getUniformMidLine")


# Deprecated lower-camelCase class name.
traceAnalyser = deprecated_class_alias(TraceAnalyser, "traceAnalyser")
