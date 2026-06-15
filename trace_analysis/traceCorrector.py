# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — trace_analysis.traceCorrector                   ║
# ║  « interactive frame/coordinate calibration »                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Overlays tracking on movie frames and lets the user nudge the   ║
# ║  frame offset and pixel shift to align detection with the video. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Interactively align LACE tracking with the source movie frames."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data_handlers.matLabResultLoader import MatlabResultLoader
from data_handlers.mediaHandler import MediaHandler
from deprecation import deprecated_alias, deprecated_class_alias
from plotting.fishPlot import frameOverlay

PAUSE_S: float = 0.001  # per-frame pause during the visual self-test

# key → frame-offset delta (a/d step one frame, A/D step ten)
_FRAME_STEP: dict[str, int] = {"a": -1, "d": 1, "A": -10, "D": 10}
# key → (dx, dy) coordinate nudge of the detection overlay
_COORD_STEP: dict[str, tuple[int, int]] = {
    "right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1),
}


class TraceCorrector:
    """Display tracking over movie frames and apply frame/pixel corrections.

    Args:
        dataDict: Metadata and file positions (``csv``, ``anaMat``, ``seq``,
                  ``avi``) for one recording.
    """

    def __init__(self, dataDict: dict) -> None:
        self.dataDict = dataDict

        # read arena box (absent for already-millimetre traces)
        if self.dataDict["csv"] == "":
            self.mmTraceAvailable = True
        else:
            self.mmTraceAvailable = False
            self.boxCoords = np.genfromtxt(self.dataDict["csv"], delimiter=",")

        # load MATLAB data
        self.matLabLoader = MatlabResultLoader(self.dataDict["anaMat"])
        self.matLabLoader.get_data()

        # load movie file
        if self.dataDict["seq"] != "":
            self.mH = MediaHandler(self.dataDict["seq"], "norpix")
        elif self.dataDict["avi"] != "":
            self.mH = MediaHandler(self.dataDict["avi"], "movie")

        # shorthands
        self.contour = self.matLabLoader.traceContour
        self.head = self.matLabLoader.traceHead
        self.tail = self.matLabLoader.traceTail
        self.midLine = self.matLabLoader.traceMidline
        if self.mH.modus == "norpix":
            self.headerDict = self.mH.media.header_dict
            self.originFrame = self.headerDict["origin"]
            self.allocated_frames = self.headerDict["allocated_frames"]
            self.fps = self.headerDict["suggested_frame_rate"]
        elif self.mH.modus == "movie":
            self.headerDict = None
            self.originFrame = 0
            self.allocated_frames = self.mH.length
            self.fps = self.mH.fps

        # preallocations
        self.currentFrame = None
        self.frameI = 0

        # calibration state
        self.calibrationOngoing = False
        self.frameShift = 0
        self.pixelOffset = np.array([0.0, 0.0])
        self.coordShift = np.zeros(shape=(1, 2))

        # matplotlib
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

    def close_figure(self) -> None:
        """Close the figure displayed by the corrector."""
        plt.close(self.fig)

    def calculate_coord_shift(self, buffer_shift: int) -> np.ndarray:
        """Convert a linear buffer shift into an (x, y) pixel shift."""
        image_width = self.mH.width
        x_shift = int(buffer_shift % image_width)
        y_shift = buffer_shift // image_width
        return np.array([x_shift, y_shift])

    def shift_frame_coords(self) -> None:
        """Apply the current coordinate shift to all tracking arrays."""
        self.head = self.head + self.coordShift
        self.tail = self.tail + self.coordShift
        self.contour = [x + self.coordShift for x in self.contour]
        self.midLine = [x + self.coordShift for x in self.midLine]

    def plot_frame_overlay(self) -> None:
        """Draw the current frame with the tracking data overlaid."""
        frameOverlay(
            self.ax, self.currentFrame, self.contour[self.frameI], self.midLine[self.frameI],
            self.head[self.frameI, :], self.tail[self.frameI, :], self.boxCoords,
        )
        if self.calibrationOngoing:
            self.ax.set_title(
                "q = quit | f = fullscreen | a = -1 frame | A -10 frames | "
                "c = +1 frame | D +10 frames | w = negative origin | "
                "e = origin frame| cursor moves detection | s = save frame"
            )
            self.ax.set_xlabel(
                f"frame offSet: {self.frameShift} | origin frame: {self.originFrame} "
                f"| pixelShift (x,y): {self.pixelOffset}"
            )
        else:
            self.ax.set_xlabel(
                f"frame: {self.frameI} | dur: {np.round(self.frameI / self.fps, 2)}"
            )
        plt.draw()

    def get_frame_no_for_norpix(self, correction_shift: int) -> int:
        """Return the wrapped Norpix frame number for a correction shift."""
        if correction_shift < 0:
            correction_shift = self.allocated_frames + correction_shift
        return int(
            np.abs((self.frameI + self.originFrame + correction_shift) % self.allocated_frames)
        )

    def load_norpix_frame(self, frame_shift: int) -> np.ndarray:
        """Load the Norpix frame for a given frame shift."""
        return self.mH.get_frame(self.get_frame_no_for_norpix(frame_shift))

    def on_press(self, event) -> None:
        """Handle key presses: step frames, reset origin, or nudge detection."""
        load_new_img = False
        shift_coord = False
        key = event.key

        if key in _FRAME_STEP:
            self.frameShift += _FRAME_STEP[key]
            load_new_img = True
        elif key == "w":
            self.frameShift = self.originFrame * -1
            load_new_img = True
        elif key == "e":
            self.frameShift = 0
            load_new_img = True
        elif key in ("q", "Q"):
            self.calibrationOngoing = False
        elif key in _COORD_STEP:
            self.coordShift = np.array(_COORD_STEP[key])
            self.pixelOffset = self.pixelOffset + self.coordShift
            shift_coord = True

        if shift_coord:
            self.shift_frame_coords()
        if self.calibrationOngoing:
            self.refresh_image(load_new_img)

    def refresh_image(self, new_img_flag: bool) -> None:
        """Redraw the overlay, loading a new frame first if requested."""
        if new_img_flag:
            self.currentFrame = self.load_norpix_frame(self.frameShift)
        plt.cla()
        self.plot_frame_overlay()
        self.fig.canvas.draw()

    def calibrate_tracking(self) -> None:
        """Show the interactive calibration interface from the first frame."""
        self.frameI = 0
        self.calibrationOngoing = True
        self.refresh_image(True)
        plt.show()

    def run_test(self, length_in_frames: int = 100) -> None:
        """Step through evenly spaced frames as a visual self-test."""
        self.fig, self.ax = plt.subplots()
        plt.ion()
        for frame_i in np.linspace(0, self.allocated_frames - 1, length_in_frames, dtype=int):
            self.frameI = frame_i
            self.refresh_image(True)
            plt.pause(PAUSE_S)

    # Deprecated camelCase method names.
    calculateCoordShift = deprecated_alias(calculate_coord_shift, "calculateCoordShift")
    shiftFrameCoords = deprecated_alias(shift_frame_coords, "shiftFrameCoords")
    plotFrameOverlay = deprecated_alias(plot_frame_overlay, "plotFrameOverlay")
    getFrameNo4Norpix = deprecated_alias(get_frame_no_for_norpix, "getFrameNo4Norpix")
    loadNorPixFrame = deprecated_alias(load_norpix_frame, "loadNorPixFrame")
    refreshImage = deprecated_alias(refresh_image, "refreshImage")
    calibrateTracking = deprecated_alias(calibrate_tracking, "calibrateTracking")
    runTest = deprecated_alias(run_test, "runTest")


# Deprecated lower-camelCase class name.
traceCorrector = deprecated_class_alias(TraceCorrector, "traceCorrector")
