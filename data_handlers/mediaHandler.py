# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — data_handlers.mediaHandler                      ║
# ║  « uniform frame access across video formats »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  One interface over OpenCV movies, Norpix sequences, and image   ║
# ║  stacks, with a small frame buffer and stabilisation helpers.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Uniform frame access over OpenCV movies, Norpix sequences, and images."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pims
import tqdm
from vidstab import VidStab

from deprecation import deprecated_alias, deprecated_class_alias

DEFAULT_BUFFER_SIZE: int = 2000   # frames kept in the LRU frame buffer
DEFAULT_IMAGE_FPS: int = 25       # assumed fps for image-sequence input
DEFAULT_REGISTER_BORDER: int = 50  # stabilisation border, in pixels
WAIT_KEY_MS: int = 5              # cv2.waitKey poll interval
ESC_KEY: int = 27                 # ASCII escape, breaks the register loop


class MediaHandler:
    """Read frames from movies, Norpix sequences, or image stacks alike.

    Args:
        filename:    Path to the media file or image-sequence glob.
        modus:       One of ``"movie"``, ``"norpix"``, or ``"image"``.
        fps:         Frames per second; ignored where the format reports it.
        buffer_size: Maximum number of frames to keep buffered.

    Raises:
        ValueError: If ``modus`` is not a recognised mode.
    """

    def __init__(
        self,
        filename: str | Path,
        modus: str,
        fps: int = 0,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        self.activeFrame: np.ndarray | list = []
        self.frameNo = 0
        self.modus = modus
        self.buffer: dict[int, np.ndarray] = {}
        self.bufferLog: list[int] = []
        self.bufferSize = buffer_size
        self.fileName = str(filename)
        self.fps = fps

        if modus == "movie":
            self.media = cv2.VideoCapture(self.fileName)
            self.length = self.media.get(cv2.CAP_PROP_FRAME_COUNT)
            self.height = self.media.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width = self.media.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.colorDim = 3 if self.media.get(cv2.CAP_PROP_MONOCHROME) == 0 else 1
            self.fps = self.media.get(cv2.CAP_PROP_FPS)
            self._make_int_parameters()
        elif modus == "norpix":
            self.media = pims.NorpixSeq(self.fileName)
            self.length = len(self.media) - 1
            if len(self.media.frame_shape) == 2:
                self.height, self.width = self.media.frame_shape
            else:
                self.height, self.width, self.colorDim = self.media.frame_shape
            self.fps = self.media.frame_rate
            self._make_int_parameters()
        elif modus == "image":
            self.media = pims.ImageSequence(self.fileName)
            self.length = len(self.media) - 1
            self.height, self.width, self.colorDim = self.media.frame_shape
            self.fps = DEFAULT_IMAGE_FPS
            self._make_int_parameters()
        else:
            raise ValueError(f"MediaHandler: unknown modus {modus!r}")

    def _make_int_parameters(self) -> None:
        """Cast the frame-count and frame-size parameters to integers."""
        self.length = int(self.length)
        self.height = int(self.height)
        self.width = int(self.width)
        self.size = (self.width, self.height)

    def _clamp_frame_no(self, frame_no: int) -> int:
        """Clamp a requested frame number into ``[0, length]``."""
        if frame_no < 0:
            return 0
        if frame_no > self.length:
            return self.length
        return frame_no

    def get_frame(self, frame_no: int) -> np.ndarray:
        """Return the frame at ``frame_no``, serving from the buffer if cached.

        Args:
            frame_no: Requested frame index (clamped into range).

        Returns:
            The frame as a NumPy array.

        Raises:
            ValueError: If the handler's modus is unknown.
        """
        frame_no = self._clamp_frame_no(frame_no)

        if frame_no in self.bufferLog:
            self.activeFrame = np.array(self.buffer[frame_no], copy=True)
            self.frameNo = frame_no
            return self.activeFrame

        if self.modus == "movie":
            self._get_frame_movie(frame_no)
        elif self.modus == "norpix":
            self._get_frame_norpix(frame_no)
        elif self.modus == "image":
            self._get_frame_image(frame_no)
        else:
            raise ValueError(f"MediaHandler: unknown modus {self.modus!r}")

        if len(self.bufferLog) > self.bufferSize:
            self.buffer.pop(self.bufferLog[0])
            self.bufferLog.pop(0)

        self.buffer[frame_no] = np.array(self.activeFrame, copy=True)
        self.bufferLog.append(frame_no)
        return self.activeFrame

    def get_frame_no(self) -> int:
        """Return the current frame number."""
        return self.frameNo

    def _get_frame_movie(self, frame_no: int) -> None:
        """Read a frame from an OpenCV movie into ``activeFrame``."""
        self.frameNo = frame_no
        self.media.set(1, frame_no)
        flag, self.activeFrame = self.media.read(frame_no)
        if not flag or self.activeFrame is None:
            raise OSError(f"Frame {frame_no} unreadable in {self.fileName}")

    def _get_frame_norpix(self, frame_no: int) -> None:
        """Read a frame from a Norpix sequence into ``activeFrame``."""
        self.frameNo = frame_no
        self.activeFrame = self.media.get_frame(frame_no)

    def _get_frame_image(self, frame_no: int) -> None:
        """Read a frame from an image sequence into ``activeFrame``."""
        self.frameNo = frame_no
        self.activeFrame = self.media.get_frame(frame_no)

    def get_time(self) -> float:
        """Return the time in seconds of the current frame."""
        return self.frameNo / self.fps

    def transcode_seq_to_avi(self, target_file: str | Path) -> None:
        """Transcode a Norpix sequence to an XVID-encoded AVI file."""
        if self.modus != "norpix":
            raise ValueError("transcode_seq_to_avi only works with Norpix files")

        source_fps = round(self.fps)
        frame_shape = self.size
        allocated_frames = self.media.header_dict["allocated_frames"]

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        out = cv2.VideoWriter(str(target_file), fourcc, source_fps, frame_shape)

        for frame_no in tqdm.tqdm(range(allocated_frames), desc="transcoding " + self.fileName):
            frame = self.get_frame(frame_no)
            gray_3c = cv2.merge([frame, frame, frame])
            out.write(gray_3c)
            cv2.imshow("frame", gray_3c)

        out.release()

    def register_movie(
        self,
        source_file: str | Path,
        target_file: str | Path,
        border: int = DEFAULT_REGISTER_BORDER,
    ) -> None:
        """Stabilise a movie and let the user track an object interactively.

        Args:
            source_file: Source movie path (informational).
            target_file: Output path for the stabilised, registered movie.
            border:      Stabilisation border in pixels.

        Raises:
            ValueError: If the handler's modus is not ``"movie"``.
        """
        if self.modus != "movie":
            raise ValueError("register_movie only works with OpenCV movie files")

        object_tracker = cv2.TrackerCSRT_create()
        stabilizer = VidStab()
        object_bounding_box = None

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        source_fps = round(self.fps)
        frame_shape = (self.size[0] + 2 * border, self.size[1] + 2 * border)
        out = cv2.VideoWriter(str(target_file), fourcc, source_fps, frame_shape)

        while True:
            _grabbed, frame = self.media.read()
            stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=border)
            if stabilized_frame is None:
                break

            if object_bounding_box is not None:
                success, object_bounding_box = object_tracker.update(stabilized_frame)
                if success:
                    (x, y, w, h) = [int(v) for v in object_bounding_box]
                    cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Frame", stabilized_frame)
            out.write(stabilized_frame)
            key = cv2.waitKey(WAIT_KEY_MS)

            # A non-zero frame means stabilisation has warmed up; let the
            # user pick a ROI, then track it from then on.
            if stabilized_frame.sum() > 0 and object_bounding_box is None:
                object_bounding_box = cv2.selectROI(
                    "Frame", stabilized_frame, fromCenter=False, showCrosshair=True
                )
                object_tracker.init(stabilized_frame, object_bounding_box)
            elif key == ESC_KEY:
                break

        out.release()
        cv2.destroyAllWindows()

    # Deprecated camelCase method names.
    SR_makeIntParameters = deprecated_alias(_make_int_parameters, "SR_makeIntParameters")
    SR_setFrameNoInBounds = deprecated_alias(_clamp_frame_no, "SR_setFrameNoInBounds")
    getFrame = deprecated_alias(get_frame, "getFrame")
    get_frameNo = deprecated_alias(get_frame_no, "get_frameNo")
    getFrameMov = deprecated_alias(_get_frame_movie, "getFrameMov")
    getFrameNorpix = deprecated_alias(_get_frame_norpix, "getFrameNorpix")
    getFrameImage = deprecated_alias(_get_frame_image, "getFrameImage")
    transcode_seq2avis = deprecated_alias(transcode_seq_to_avi, "transcode_seq2avis")


# Deprecated lower-camelCase class name.
mediaHandler = deprecated_class_alias(MediaHandler, "mediaHandler")
