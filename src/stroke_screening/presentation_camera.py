"""Low-load, single-owner local camera service for PresentCoach."""

from __future__ import annotations

import secrets
import threading
import time
from typing import Sequence

import cv2
import numpy as np

from .local_camera import LocalCamera


class CameraBusyError(RuntimeError):
    """Raised when another recording already owns the camera."""


class CameraSessionError(RuntimeError):
    """Raised when a camera owner is missing, stale, or stopped."""


class PresentationCameraService:
    """Own one camera thread and expose copied frames plus an MJPEG preview."""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        width: int = 960,
        height: int = 540,
        timeout_seconds: float = 30 * 60 + 30,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.timeout_seconds = timeout_seconds
        self._condition = threading.Condition()
        self._owner: str | None = None
        self._camera: LocalCamera | None = None
        self._thread: threading.Thread | None = None
        self._timer: threading.Timer | None = None
        self._stop_event = threading.Event()
        self._latest: np.ndarray | None = None
        self._generation = -1
        self._error: str | None = None

    def start(self) -> str:
        with self._condition:
            if self._owner is not None:
                raise CameraBusyError("A presentation recording is already using the camera")
            owner = secrets.token_urlsafe(24)
            camera = LocalCamera(self.camera_index, width=self.width, height=self.height)
            camera.open()
            self._owner = owner
            self._camera = camera
            self._latest = None
            self._generation = -1
            self._error = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._reader, name="presentcoach-camera", daemon=True
            )
            self._thread.start()
            self._timer = threading.Timer(self.timeout_seconds, self.stop)
            self._timer.daemon = True
            self._timer.start()
            return owner

    def _reader(self) -> None:
        try:
            while not self._stop_event.is_set():
                camera = self._camera
                if camera is None:
                    break
                frame = camera.read()
                with self._condition:
                    self._latest = frame
                    self._generation += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = str(error)
                self._condition.notify_all()
        finally:
            if self._camera is not None:
                self._camera.close()

    def _assert_owner(self, owner: str) -> None:
        if not owner or not secrets.compare_digest(owner, self._owner or ""):
            raise CameraSessionError("The local camera session expired")
        if self._error:
            raise CameraSessionError("The local camera stopped unexpectedly")

    def next_frame(
        self, owner: str, after: int = -1, timeout: float = 3.0
    ) -> tuple[np.ndarray, int]:
        deadline = time.monotonic() + timeout
        with self._condition:
            self._assert_owner(owner)
            while self._generation <= after and not self._error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraSessionError("The camera did not return a frame")
                self._condition.wait(timeout=remaining)
                self._assert_owner(owner)
            if self._latest is None:
                raise CameraSessionError("The camera preview is not ready")
            return self._latest.copy(), self._generation

    def preview_jpeg(
        self,
        owner: str,
        overlay_points: Sequence[tuple[float, float]] | None = None,
    ) -> bytes:
        frame, _generation = self.next_frame(owner)
        height, width = frame.shape[:2]
        preview_width = min(512, width)
        preview_height = max(1, round(height * preview_width / width))
        preview = cv2.resize(
            frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA
        )
        preview = cv2.flip(preview, 1)
        for x_value, y_value in overlay_points or ():
            x = round((1.0 - float(x_value)) * preview_width)
            y = round(float(y_value) * preview_height)
            if 0 <= x < preview_width and 0 <= y < preview_height:
                cv2.circle(preview, (x, y), 2, (171, 255, 95), -1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if not ok:
            raise CameraSessionError("The camera preview could not be encoded")
        return bytes(encoded)

    def stop(self, owner: str | None = None) -> None:
        with self._condition:
            if owner is not None:
                self._assert_owner(owner)
            if self._owner is None:
                return
            self._stop_event.set()
            thread = self._thread
            timer = self._timer
            self._timer = None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._condition:
            self._owner = None
            self._camera = None
            self._thread = None
            self._latest = None
            self._error = None
            self._condition.notify_all()
