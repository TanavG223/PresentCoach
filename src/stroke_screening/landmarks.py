"""MediaPipe Tasks Face Landmarker wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """One frame of model output."""

    landmarks: np.ndarray | None
    blendshapes: dict[str, float]
    face_count: int
    inference_ms: float
    transformation_matrix: np.ndarray | None = None


class FaceLandmarkEngine:
    """Synchronous video-mode inference with monotonically increasing time."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {model_path}. "
                "Run scripts/setup.sh first."
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            # Detect up to two so a crowded frame is rejected instead of
            # silently measuring whichever person happened to win detection.
            num_faces=2,
            min_face_detection_confidence=0.55,
            min_face_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )
        self._last_timestamp_ms = -1
        self._smoothed_points: np.ndarray | None = None
        self._last_face_timestamp_ms = -1

    def detect(
        self, bgr_frame: np.ndarray, timestamp_ms: int | None = None
    ) -> FaceDetection:
        now_ms = (
            timestamp_ms
            if timestamp_ms is not None
            else time.monotonic_ns() // 1_000_000
        )
        now_ms = max(int(now_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = now_ms

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, now_ms)
        inference_ms = (time.perf_counter() - started) * 1000.0
        face_count = len(result.face_landmarks)
        if face_count != 1:
            self._smoothed_points = None
            self._last_face_timestamp_ms = -1
            return FaceDetection(None, {}, face_count, inference_ms, None)

        raw_points = np.array(
            [
                (landmark.x, landmark.y, landmark.z)
                for landmark in result.face_landmarks[0]
            ],
            dtype=np.float64,
        )
        gap_ms = now_ms - self._last_face_timestamp_ms
        if self._smoothed_points is None or gap_ms > 250:
            points = raw_points
        else:
            alpha = 1.0 - np.exp(-max(gap_ms, 1) / 80.0)
            points = self._smoothed_points + alpha * (
                raw_points - self._smoothed_points
            )
        self._smoothed_points = points
        self._last_face_timestamp_ms = now_ms
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            blendshapes = {
                category.category_name: float(category.score)
                for category in result.face_blendshapes[0]
            }
        transformation_matrix = None
        if result.facial_transformation_matrixes:
            transformation_matrix = np.asarray(
                result.facial_transformation_matrixes[0],
                dtype=np.float64,
            ).copy()
        return FaceDetection(
            points,
            blendshapes,
            face_count,
            inference_ms,
            transformation_matrix,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkEngine":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
