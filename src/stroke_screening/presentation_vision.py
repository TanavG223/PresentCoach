"""Real-time MediaPipe presentation metrics and landmark-overlay state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
from statistics import fmean
from typing import Iterable

import cv2
import numpy as np

from .landmarks import FaceLandmarkEngine
from .presentation_core import VisionSample


# A sparse face mesh is enough to make live detection visible without making
# every preview frame expensive to encode.
OVERLAY_INDICES = tuple(sorted(set(
    [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 152, 162,
     172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378,
     33, 133, 159, 145, 362, 263, 386, 374, 468, 473,
     61, 78, 13, 14, 308, 291, 0, 17, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
)))


@dataclass(frozen=True)
class _FrameMetric:
    timestamp_seconds: float
    face_detected: bool
    eye_contact: bool | None
    gaze_horizontal: float | None
    gaze_vertical: float | None
    yaw_degrees: float | None
    pitch_degrees: float | None
    roll_degrees: float | None
    face_center_x: float | None
    face_center_y: float | None
    mouth_activity: float | None
    brow_activity: float | None
    expression_change: float | None
    inference_ms: float


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(finite) if finite else None


def _eye_ratio(points: np.ndarray, corner_a: int, corner_b: int, iris: int) -> float:
    start = points[corner_a, :2]
    direction = points[corner_b, :2] - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-9:
        return 0.5
    return float(np.dot(points[iris, :2] - start, direction) / denominator)


def _eye_vertical(points: np.ndarray, iris: int, lids: tuple[int, int]) -> float:
    upper = float(points[lids[0], 1])
    lower = float(points[lids[1], 1])
    span = lower - upper
    if abs(span) <= 1e-9:
        return 0.5
    return (float(points[iris, 1]) - upper) / span


def _head_angles(matrix: np.ndarray | None) -> tuple[float | None, float | None, float | None]:
    if matrix is None or matrix.shape[0] < 3 or matrix.shape[1] < 3:
        return None, None, None
    rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
    try:
        angles = cv2.RQDecomp3x3(rotation)[0]
    except cv2.error:
        return None, None, None
    # OpenCV returns rotations around x, y, z: pitch, yaw, roll.
    return float(angles[1]), float(angles[0]), float(angles[2])


def _blendshape_mean(blendshapes: dict[str, float], names: tuple[str, ...]) -> float:
    values = [float(blendshapes.get(name, 0.0)) for name in names]
    return fmean(values) if values else 0.0


class PresentationVisionAnalyzer:
    """Process live frames at a caller-controlled cadence and bucket to 1 Hz."""

    def __init__(self, model_path: Path) -> None:
        self._engine = FaceLandmarkEngine(model_path)
        self._lock = threading.Lock()
        self._frames: list[_FrameMetric] = []
        self._overlay_points: tuple[tuple[float, float], ...] = ()
        self._previous_expression: tuple[float, float] | None = None

    def process(self, frame: np.ndarray, timestamp_seconds: float) -> None:
        detection = self._engine.detect(frame)
        points = detection.landmarks
        if points is None:
            metric = _FrameMetric(
                timestamp_seconds, False, None, None, None, None, None, None,
                None, None, None, None, None, detection.inference_ms,
            )
            with self._lock:
                self._frames.append(metric)
                self._overlay_points = ()
            return

        left_horizontal = _eye_ratio(points, 33, 133, 468)
        right_horizontal = _eye_ratio(points, 362, 263, 473)
        gaze_horizontal = fmean((left_horizontal, right_horizontal))
        gaze_vertical = fmean((
            _eye_vertical(points, 468, (159, 145)),
            _eye_vertical(points, 473, (386, 374)),
        ))
        yaw, pitch, roll = _head_angles(detection.transformation_matrix)
        head_centered = (
            yaw is not None and pitch is not None
            and abs(yaw) <= 14.0 and abs(pitch) <= 15.0
        )
        eye_contact = bool(
            head_centered
            and 0.34 <= gaze_horizontal <= 0.66
            and 0.18 <= gaze_vertical <= 0.82
        )
        mouth = _blendshape_mean(detection.blendshapes, (
            "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
            "mouthFrownRight", "mouthPucker", "mouthPressLeft", "mouthPressRight",
        ))
        brow = _blendshape_mean(detection.blendshapes, (
            "browInnerUp", "browDownLeft", "browDownRight",
            "browOuterUpLeft", "browOuterUpRight",
        ))
        previous = self._previous_expression
        change = 0.0 if previous is None else math.hypot(mouth - previous[0], brow - previous[1])
        self._previous_expression = (mouth, brow)
        center = np.mean(points[[10, 152, 234, 454], :2], axis=0)
        overlay = tuple(
            (float(points[index, 0]), float(points[index, 1]))
            for index in OVERLAY_INDICES
            if index < points.shape[0]
        )
        metric = _FrameMetric(
            timestamp_seconds=float(timestamp_seconds),
            face_detected=True,
            eye_contact=eye_contact,
            gaze_horizontal=gaze_horizontal,
            gaze_vertical=gaze_vertical,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
            face_center_x=float(center[0]),
            face_center_y=float(center[1]),
            mouth_activity=mouth,
            brow_activity=brow,
            expression_change=change,
            inference_ms=detection.inference_ms,
        )
        with self._lock:
            self._frames.append(metric)
            self._overlay_points = overlay

    def overlay_points(self) -> tuple[tuple[float, float], ...]:
        with self._lock:
            return tuple(self._overlay_points)

    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def finish(self, duration_seconds: float) -> tuple[VisionSample, ...]:
        with self._lock:
            frames = tuple(self._frames)
        buckets: list[VisionSample] = []
        bucket_count = max(1, math.ceil(max(duration_seconds, 0.01)))
        for second in range(bucket_count):
            selected = [
                frame for frame in frames
                if second <= frame.timestamp_seconds < second + 1.0
            ]
            if not selected:
                buckets.append(VisionSample(
                    timestamp_seconds=float(second), frame_count=0,
                    face_detected=False, eye_contact=None,
                    gaze_horizontal=None, gaze_vertical=None,
                    yaw_degrees=None, pitch_degrees=None, roll_degrees=None,
                    face_center_x=None, face_center_y=None,
                    mouth_activity=None, brow_activity=None,
                    expression_change=None, inference_ms=None,
                    detected_frame_count=0, contact_frame_count=0,
                    contact_eligible_frame_count=0,
                ))
                continue
            detected = [frame for frame in selected if frame.face_detected]
            eye_values = [frame.eye_contact for frame in detected if frame.eye_contact is not None]
            buckets.append(VisionSample(
                timestamp_seconds=float(second),
                frame_count=len(selected),
                face_detected=len(detected) * 2 >= len(selected),
                eye_contact=(sum(value is True for value in eye_values) >= max(1, len(eye_values) / 2)) if eye_values else None,
                gaze_horizontal=_mean_or_none(frame.gaze_horizontal for frame in detected),
                gaze_vertical=_mean_or_none(frame.gaze_vertical for frame in detected),
                yaw_degrees=_mean_or_none(frame.yaw_degrees for frame in detected),
                pitch_degrees=_mean_or_none(frame.pitch_degrees for frame in detected),
                roll_degrees=_mean_or_none(frame.roll_degrees for frame in detected),
                face_center_x=_mean_or_none(frame.face_center_x for frame in detected),
                face_center_y=_mean_or_none(frame.face_center_y for frame in detected),
                mouth_activity=_mean_or_none(frame.mouth_activity for frame in detected),
                brow_activity=_mean_or_none(frame.brow_activity for frame in detected),
                expression_change=_mean_or_none(frame.expression_change for frame in detected),
                inference_ms=_mean_or_none(frame.inference_ms for frame in selected),
                detected_frame_count=len(detected),
                contact_frame_count=sum(value is True for value in eye_values),
                contact_eligible_frame_count=len(eye_values),
            ))
        return tuple(buckets)

    def close(self) -> None:
        self._engine.close()
