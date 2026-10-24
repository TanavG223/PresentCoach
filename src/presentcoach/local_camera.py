"""Built-in/USB UVC webcam access for the local Face Vault workflow."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalCameraInfo:
    index: int
    width: int
    height: int
    backend: str


class LocalCamera:
    """Small, testable wrapper around an OpenCV camera capture."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        capture_factory: Callable[..., cv2.VideoCapture] | None = None,
    ) -> None:
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> LocalCameraInfo:
        if self._capture is not None:
            raise RuntimeError("Camera is already open")
        if sys.platform == "darwin":
            capture = self._capture_factory(self.index, cv2.CAP_AVFOUNDATION)
        else:
            capture = self._capture_factory(self.index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera {self.index}. On macOS, allow "
                "camera access for Terminal/Codex in System Settings > "
                "Privacy & Security > Camera."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        backend = "unknown"
        try:
            backend = capture.getBackendName()
        except cv2.error:
            pass
        return LocalCameraInfo(self.index, width, height, backend)

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Camera stopped returning frames")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "LocalCamera":
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
