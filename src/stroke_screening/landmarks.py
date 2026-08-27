"""MediaPipe Tasks Face Landmarker wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


MAX_FACE_COUNT = 2
FACE_DETECTION_CONFIDENCE = 0.55
FACE_PRESENCE_CONFIDENCE = 0.55
FACE_TRACKING_CONFIDENCE = 0.55
LANDMARK_SMOOTHING_TIME_CONSTANT_MS = 80.0
LANDMARK_REACQUISITION_GAP_MS = 250
LANDMARK_TEMPORAL_HOLD_MS = 80
MULTISCALE_FALLBACK_ENABLED = True
MULTISCALE_TILE_SIZE = 0.5
MULTISCALE_TILE_OFFSETS = (0.0, 0.25, 0.5)
MULTISCALE_DEDUPLICATION_IOU = 0.25
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_SCORE_THRESHOLD = 0.85
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 50
YUNET_ROI_SCALES = (1.25,)
YUNET_LIVENESS_WINDOW_MS = 800
YUNET_LIVENESS_HOLD_MS = 2_000
YUNET_LIVENESS_RANGE_THRESHOLD = 0.11
YUNET_LIVENESS_MIN_IOU = 0.01
YUNET_APPEARANCE_SIZE = 48
YUNET_APPEARANCE_LAG_MIN_MS = 150
YUNET_APPEARANCE_LAG_MAX_MS = 300
YUNET_APPEARANCE_MOTION_THRESHOLD = 0.18
YUNET_PROVISIONAL_TRACK_MS = 500
YUNET_PROVISIONAL_REARM_GAP_MS = 1_000
YUNET_PROVISIONAL_LUMINANCE_CHANGE_RATIO = 0.35
SMALL_FACE_LIVENESS_AREA_THRESHOLD = 0.015
MIN_EYE_TO_LOWER_FACE_LUMINANCE_RATIO = 0.50


@dataclass(frozen=True)
class FaceDetection:
    """One frame of model output."""

    landmarks: np.ndarray | None
    blendshapes: dict[str, float]
    face_count: int
    inference_ms: float
    transformation_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class _FaceCandidate:
    landmarks: np.ndarray
    blendshapes: dict[str, float]
    transformation_matrix: np.ndarray | None
    bounds: tuple[float, float, float, float]
    area: float


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _deduplicate_candidates(
    candidates: list[_FaceCandidate],
) -> list[_FaceCandidate]:
    unique: list[_FaceCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.area, reverse=True):
        if any(
            _intersection_over_union(candidate.bounds, item.bounds)
            >= MULTISCALE_DEDUPLICATION_IOU
            for item in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _multiscale_crops(
    frame_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the bounded legacy half-frame crop grid."""

    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return ()
    crops: list[tuple[int, int, int, int]] = []
    for top_ratio in MULTISCALE_TILE_OFFSETS:
        for left_ratio in MULTISCALE_TILE_OFFSETS:
            x0 = min(max(round(left_ratio * frame_width), 0), frame_width - 1)
            y0 = min(max(round(top_ratio * frame_height), 0), frame_height - 1)
            x1 = min(
                max(round((left_ratio + MULTISCALE_TILE_SIZE) * frame_width), x0 + 1),
                frame_width,
            )
            y1 = min(
                max(round((top_ratio + MULTISCALE_TILE_SIZE) * frame_height), y0 + 1),
                frame_height,
            )
            crops.append((x0, y0, x1, y1))
    return tuple(crops)


def _proposal_crop(
    detection: np.ndarray,
    frame_size: tuple[int, int],
    scale: float,
) -> tuple[int, int, int, int] | None:
    """Expand one anonymous YuNet face box into a square landmark ROI."""

    frame_width, frame_height = frame_size
    values = np.asarray(detection, dtype=np.float64).reshape(-1)
    if (
        values.size < 4
        or frame_width <= 0
        or frame_height <= 0
        or not np.isfinite(values[:4]).all()
        or not math.isfinite(scale)
        or scale <= 1.0
    ):
        return None
    x, y, width, height = (float(value) for value in values[:4])
    if width <= 1.0 or height <= 1.0:
        return None
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    side = max(width, height) * scale
    x0 = max(0, math.floor(center_x - side / 2.0))
    y0 = max(0, math.floor(center_y - side / 2.0))
    x1 = min(frame_width, math.ceil(center_x + side / 2.0))
    y1 = min(frame_height, math.ceil(center_y + side / 2.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _eye_visibility_ratio(
    bgr_frame: np.ndarray, detection: np.ndarray
) -> float | None:
    """Measure whether both proposed eye regions remain visibly illuminated."""

    values = np.asarray(detection, dtype=np.float64).reshape(-1)
    if (
        bgr_frame.ndim != 3
        or bgr_frame.shape[2] != 3
        or values.size < 14
        or not np.isfinite(values[:14]).all()
    ):
        return None
    height, width = bgr_frame.shape[:2]
    box_width, box_height = float(values[2]), float(values[3])
    if width <= 0 or height <= 0 or box_width <= 1.0 or box_height <= 1.0:
        return None
    radius = max(2, round(min(box_width, box_height) * 0.10))
    grayscale = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    def patch_mean(x: float, y: float) -> float | None:
        center_x, center_y = round(x), round(y)
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return float(np.mean(grayscale[y0:y1, x0:x1]))

    eye_values = [patch_mean(values[4], values[5]), patch_mean(values[6], values[7])]
    lower_values = [
        patch_mean(values[8], values[9]),
        patch_mean(values[10], values[11]),
        patch_mean(values[12], values[13]),
    ]
    if any(value is None for value in (*eye_values, *lower_values)):
        return None
    eye_luminance = float(np.mean(eye_values))
    lower_luminance = float(np.mean(lower_values))
    if lower_luminance <= 1e-6:
        return None
    ratio = eye_luminance / lower_luminance
    return ratio if math.isfinite(ratio) else None


def _face_box_luminance(
    bgr_frame: np.ndarray, detection: np.ndarray
) -> float | None:
    values = np.asarray(detection, dtype=np.float64).reshape(-1)
    if (
        bgr_frame.ndim != 3
        or bgr_frame.shape[2] != 3
        or values.size < 4
        or not np.isfinite(values[:4]).all()
    ):
        return None
    height, width = bgr_frame.shape[:2]
    x, y, box_width, box_height = (float(value) for value in values[:4])
    x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
    x1 = min(width, math.ceil(x + box_width))
    y1 = min(height, math.ceil(y + box_height))
    if x1 <= x0 or y1 <= y0:
        return None
    value = float(
        np.mean(cv2.cvtColor(bgr_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    )
    return value if math.isfinite(value) else None


class FaceLandmarkEngine:
    """Synchronous video-mode inference with monotonically increasing time."""

    def __init__(
        self,
        model_path: Path,
        detector_model_path: Path | None = None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model not found at {model_path}. "
                "Run scripts/setup.sh first."
            )
        detector_path = detector_model_path or model_path.with_name(
            YUNET_MODEL_FILENAME
        )
        if not detector_path.is_file():
            raise FileNotFoundError(
                f"Anonymous face proposal model not found at {detector_path}. "
                "Run scripts/setup_presentcoach.sh first."
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            # Detect up to two so a crowded frame is rejected instead of
            # silently measuring whichever person happened to win detection.
            num_faces=MAX_FACE_COUNT,
            min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )
        fallback_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=MAX_FACE_COUNT,
            min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._fallback_landmarker = (
                mp.tasks.vision.FaceLandmarker.create_from_options(
                    fallback_options
                )
            )
        except BaseException:
            self._landmarker.close()
            raise
        try:
            self._proposal_detector = cv2.FaceDetectorYN.create(
                str(detector_path),
                "",
                (320, 320),
                YUNET_SCORE_THRESHOLD,
                YUNET_NMS_THRESHOLD,
                YUNET_TOP_K,
            )
        except BaseException:
            self._fallback_landmarker.close()
            self._landmarker.close()
            raise
        self._last_timestamp_ms = -1
        self._smoothed_points: np.ndarray | None = None
        self._last_face_timestamp_ms = -1
        self._proposal_history: deque[tuple[int, np.ndarray]] = deque()
        self._proposal_appearance_history: deque[
            tuple[int, np.ndarray]
        ] = deque()
        self._proposal_motion_history: deque[tuple[int, float]] = deque()
        self._proposal_bounds: tuple[float, float, float, float] | None = None
        self._proposal_live_until_ms = -1
        self._last_small_proposal_ms = -1
        self._proposal_provisional_until_ms = -1
        self._last_small_proposal_luminance: float | None = None
        self._proposal_confirms_single_face = False
        self._last_fresh_candidate: _FaceCandidate | None = None
        self._last_fresh_candidate_timestamp_ms = -1

    @staticmethod
    def _candidate_from_result(
        result: object,
        face_index: int,
        *,
        crop: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> _FaceCandidate | None:
        x0, y0, x1, y1 = crop
        frame_width, frame_height = frame_size
        crop_width = x1 - x0
        crop_height = y1 - y0
        try:
            face = result.face_landmarks[face_index]
            points = np.asarray(
                [
                    (
                        (float(landmark.x) * crop_width + x0) / frame_width,
                        (float(landmark.y) * crop_height + y0) / frame_height,
                        float(landmark.z) * crop_width / frame_width,
                    )
                    for landmark in face
                ],
                dtype=np.float64,
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if (
            points.ndim != 2
            or points.shape[0] < 455
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            return None
        left = float(points[:, 0].min())
        top = float(points[:, 1].min())
        right = float(points[:, 0].max())
        bottom = float(points[:, 1].max())
        area = max(0.0, right - left) * max(0.0, bottom - top)
        if area <= 1e-8:
            return None
        blendshapes: dict[str, float] = {}
        raw_blendshapes = getattr(result, "face_blendshapes", ())
        if face_index < len(raw_blendshapes):
            try:
                blendshapes = {
                    str(category.category_name): float(category.score)
                    for category in raw_blendshapes[face_index]
                    if np.isfinite(float(category.score))
                }
            except (AttributeError, TypeError, ValueError):
                blendshapes = {}
        transformation_matrix = None
        matrices = getattr(result, "facial_transformation_matrixes", ())
        if face_index < len(matrices):
            try:
                candidate_matrix = np.asarray(
                    matrices[face_index], dtype=np.float64
                )
                if (
                    candidate_matrix.shape == (4, 4)
                    and np.isfinite(candidate_matrix).all()
                ):
                    transformation_matrix = candidate_matrix.copy()
            except (TypeError, ValueError):
                transformation_matrix = None
        return _FaceCandidate(
            landmarks=points,
            blendshapes=blendshapes,
            transformation_matrix=transformation_matrix,
            bounds=(left, top, right, bottom),
            area=area,
        )

    @staticmethod
    def _proposal_liveness_features(candidate: _FaceCandidate) -> np.ndarray | None:
        points = candidate.landmarks[:, :2]
        if points.shape[0] <= 300 or not np.isfinite(points).all():
            return None
        eye_span = float(np.linalg.norm(points[33] - points[263]))
        if not math.isfinite(eye_span) or eye_span <= 1e-8:
            return None
        eye_midpoint = (points[33] + points[263]) / 2.0
        features = np.asarray(
            [
                np.linalg.norm(points[13] - points[14]) / eye_span,
                np.linalg.norm(points[61] - points[291]) / eye_span,
                np.linalg.norm(points[70] - points[300]) / eye_span,
                (points[1, 0] - eye_midpoint[0]) / eye_span,
            ],
            dtype=np.float64,
        )
        return features if np.isfinite(features).all() else None

    def _reset_proposal_state(self) -> None:
        self._proposal_history.clear()
        self._proposal_appearance_history.clear()
        self._proposal_motion_history.clear()
        self._proposal_bounds = None
        self._proposal_live_until_ms = -1

    @staticmethod
    def _proposal_appearance(
        candidate: _FaceCandidate, bgr_frame: np.ndarray
    ) -> np.ndarray | None:
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            return None
        height, width = bgr_frame.shape[:2]
        left, top, right, bottom = candidate.bounds
        if not all(math.isfinite(value) for value in candidate.bounds):
            return None
        box_width, box_height = right - left, bottom - top
        if box_width <= 0.0 or box_height <= 0.0:
            return None
        padding_x, padding_y = box_width * 0.15, box_height * 0.15
        x0 = max(0, math.floor((left - padding_x) * width))
        x1 = min(width, math.ceil((right + padding_x) * width))
        y0 = max(0, math.floor((top - padding_y) * height))
        y1 = min(height, math.ceil((bottom + padding_y) * height))
        if x1 <= x0 or y1 <= y0:
            return None
        grayscale = cv2.cvtColor(bgr_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        appearance = cv2.resize(
            cv2.GaussianBlur(grayscale, (3, 3), 0),
            (YUNET_APPEARANCE_SIZE, YUNET_APPEARANCE_SIZE),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        deviation = float(np.std(appearance))
        if not math.isfinite(deviation) or deviation <= 1e-6:
            return None
        appearance = (appearance - float(np.mean(appearance))) / deviation
        return appearance if np.isfinite(appearance).all() else None

    def _proposal_is_live(
        self,
        candidate: _FaceCandidate,
        timestamp_ms: int,
        bgr_frame: np.ndarray,
    ) -> bool:
        features = self._proposal_liveness_features(candidate)
        appearance = self._proposal_appearance(candidate, bgr_frame)
        previous_bounds = self._proposal_bounds
        if (
            features is None
            or appearance is None
            or (
                previous_bounds is not None
                and _intersection_over_union(previous_bounds, candidate.bounds)
                < YUNET_LIVENESS_MIN_IOU
            )
            or (
                self._proposal_history
                and timestamp_ms - self._proposal_history[-1][0]
                > YUNET_PROVISIONAL_REARM_GAP_MS
            )
        ):
            self._reset_proposal_state()
        if features is None or appearance is None:
            return False
        self._proposal_bounds = candidate.bounds
        self._proposal_history.append((timestamp_ms, features))
        previous_appearance = next(
            (
                value
                for previous_timestamp, value in reversed(
                    self._proposal_appearance_history
                )
                if YUNET_APPEARANCE_LAG_MIN_MS
                <= timestamp_ms - previous_timestamp
                <= YUNET_APPEARANCE_LAG_MAX_MS
            ),
            None,
        )
        self._proposal_appearance_history.append((timestamp_ms, appearance))
        if previous_appearance is not None:
            motion = float(np.mean(np.abs(appearance - previous_appearance)))
            if math.isfinite(motion):
                self._proposal_motion_history.append((timestamp_ms, motion))
        while (
            self._proposal_history
            and timestamp_ms - self._proposal_history[0][0]
            > YUNET_LIVENESS_WINDOW_MS
        ):
            self._proposal_history.popleft()
        while (
            self._proposal_appearance_history
            and timestamp_ms - self._proposal_appearance_history[0][0]
            > YUNET_LIVENESS_WINDOW_MS
        ):
            self._proposal_appearance_history.popleft()
        while (
            self._proposal_motion_history
            and timestamp_ms - self._proposal_motion_history[0][0]
            > YUNET_LIVENESS_WINDOW_MS
        ):
            self._proposal_motion_history.popleft()
        if len(self._proposal_history) >= 3:
            values = np.asarray(
                [item[1] for item in self._proposal_history], dtype=np.float64
            )
            landmark_motion = float(np.max(np.ptp(values, axis=0)))
            appearance_motion = max(
                (item[1] for item in self._proposal_motion_history),
                default=0.0,
            )
            if (
                landmark_motion >= YUNET_LIVENESS_RANGE_THRESHOLD
                and appearance_motion >= YUNET_APPEARANCE_MOTION_THRESHOLD
            ):
                self._proposal_live_until_ms = timestamp_ms + YUNET_LIVENESS_HOLD_MS
        return (
            timestamp_ms <= self._proposal_provisional_until_ms
            or timestamp_ms <= self._proposal_live_until_ms
        )

    def _observe_small_proposal(
        self, timestamp_ms: int, luminance: float | None
    ) -> None:
        previous_luminance = self._last_small_proposal_luminance
        luminance_change = bool(
            luminance is not None
            and previous_luminance is not None
            and abs(luminance - previous_luminance)
            / max(previous_luminance, 1.0)
            >= YUNET_PROVISIONAL_LUMINANCE_CHANGE_RATIO
        )
        if (
            self._last_small_proposal_ms < 0
            or timestamp_ms - self._last_small_proposal_ms
            > YUNET_PROVISIONAL_REARM_GAP_MS
            or luminance_change
        ):
            self._proposal_provisional_until_ms = (
                timestamp_ms + YUNET_PROVISIONAL_TRACK_MS
            )
        self._last_small_proposal_ms = timestamp_ms
        self._last_small_proposal_luminance = luminance

    def _temporal_hold_candidate(self, timestamp_ms: int) -> _FaceCandidate | None:
        """Bridge at most one 15 FPS frame when one proposal remains visible."""

        age_ms = timestamp_ms - self._last_fresh_candidate_timestamp_ms
        if (
            self._proposal_confirms_single_face
            and self._last_fresh_candidate is not None
            and 0 < age_ms <= LANDMARK_TEMPORAL_HOLD_MS
        ):
            return self._last_fresh_candidate
        return None

    def _multiscale_candidates(
        self, bgr_frame: np.ndarray, rgb_frame: np.ndarray, timestamp_ms: int
    ) -> tuple[list[_FaceCandidate], float]:
        self._proposal_confirms_single_face = False
        if not MULTISCALE_FALLBACK_ENABLED:
            return [], 0.0
        frame_height, frame_width = rgb_frame.shape[:2]
        started = time.perf_counter()
        self._proposal_detector.setInputSize((frame_width, frame_height))
        _status, detections = self._proposal_detector.detect(bgr_frame)
        if detections is None or len(detections) == 0:
            self._reset_proposal_state()
            return [], (time.perf_counter() - started) * 1000.0
        # More than one credible face must remain an abstention. The system has
        # no identity model and must not guess which person is the presenter.
        if len(detections) > 1:
            self._reset_proposal_state()
            return [], (time.perf_counter() - started) * 1000.0
        self._observe_small_proposal(
            timestamp_ms, _face_box_luminance(bgr_frame, detections[0])
        )
        eye_ratio = _eye_visibility_ratio(bgr_frame, detections[0])
        if (
            eye_ratio is None
            or eye_ratio < MIN_EYE_TO_LOWER_FACE_LUMINANCE_RATIO
        ):
            self._reset_proposal_state()
            return [], (time.perf_counter() - started) * 1000.0
        self._proposal_confirms_single_face = True

        candidates: list[_FaceCandidate] = []
        for scale in YUNET_ROI_SCALES:
            crop_bounds = _proposal_crop(
                detections[0], (frame_width, frame_height), scale
            )
            if crop_bounds is None:
                continue
            x0, y0, x1, y1 = crop_bounds
            crop = np.ascontiguousarray(rgb_frame[y0:y1, x0:x1])
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            result = self._fallback_landmarker.detect_for_video(
                image, timestamp_ms
            )
            for face_index in range(len(result.face_landmarks)):
                candidate = self._candidate_from_result(
                    result,
                    face_index,
                    crop=crop_bounds,
                    frame_size=(frame_width, frame_height),
                )
                if candidate is not None:
                    candidates.append(candidate)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return _deduplicate_candidates(candidates), inference_ms

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
        self._proposal_confirms_single_face = False

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        started = time.perf_counter()
        result = self._landmarker.detect_for_video(image, now_ms)
        inference_ms = (time.perf_counter() - started) * 1000.0
        face_count = len(result.face_landmarks)
        candidate = None
        candidate_is_held = False
        proposal_quality_checked = False
        if face_count == 1:
            frame_height, frame_width = rgb_frame.shape[:2]
            candidate = self._candidate_from_result(
                result,
                0,
                crop=(0, 0, frame_width, frame_height),
                frame_size=(frame_width, frame_height),
            )
        elif face_count == 0:
            candidates, fallback_ms = self._multiscale_candidates(
                bgr_frame, rgb_frame, now_ms
            )
            inference_ms += fallback_ms
            proposal_quality_checked = True
            face_count = min(len(candidates), MAX_FACE_COUNT)
            if face_count == 1:
                candidate = candidates[0]
        if face_count == 1 and candidate is not None:
            if candidate.area < SMALL_FACE_LIVENESS_AREA_THRESHOLD:
                if not proposal_quality_checked:
                    proposal_started = time.perf_counter()
                    frame_height, frame_width = bgr_frame.shape[:2]
                    self._proposal_detector.setInputSize((frame_width, frame_height))
                    _status, proposal_detections = self._proposal_detector.detect(
                        bgr_frame
                    )
                    inference_ms += (
                        time.perf_counter() - proposal_started
                    ) * 1000.0
                    eye_ratio = (
                        None
                        if proposal_detections is None
                        or len(proposal_detections) != 1
                        else _eye_visibility_ratio(
                            bgr_frame, proposal_detections[0]
                        )
                    )
                    if (
                        proposal_detections is not None
                        and len(proposal_detections) == 1
                    ):
                        self._observe_small_proposal(
                            now_ms,
                            _face_box_luminance(
                                bgr_frame, proposal_detections[0]
                            ),
                        )
                    if (
                        eye_ratio is None
                        or eye_ratio < MIN_EYE_TO_LOWER_FACE_LUMINANCE_RATIO
                    ):
                        self._reset_proposal_state()
                        candidate = None
                        face_count = 0
                    else:
                        self._proposal_confirms_single_face = True
                if candidate is not None and not self._proposal_is_live(
                    candidate, now_ms, bgr_frame
                ):
                    candidate = None
                    face_count = 0
            else:
                self._reset_proposal_state()
        held_candidate = self._temporal_hold_candidate(now_ms)
        if (face_count != 1 or candidate is None) and held_candidate is not None:
            candidate = held_candidate
            candidate_is_held = True
            face_count = 1
        if face_count != 1 or candidate is None:
            if face_count > 1:
                self._reset_proposal_state()
                self._last_fresh_candidate = None
                self._last_fresh_candidate_timestamp_ms = -1
            self._smoothed_points = None
            self._last_face_timestamp_ms = -1
            return FaceDetection(None, {}, face_count, inference_ms, None)

        if not candidate_is_held:
            self._last_fresh_candidate = candidate
            self._last_fresh_candidate_timestamp_ms = now_ms

        raw_points = candidate.landmarks
        gap_ms = now_ms - self._last_face_timestamp_ms
        if self._smoothed_points is None or gap_ms > LANDMARK_REACQUISITION_GAP_MS:
            points = raw_points
        else:
            alpha = 1.0 - np.exp(
                -max(gap_ms, 1) / LANDMARK_SMOOTHING_TIME_CONSTANT_MS
            )
            points = self._smoothed_points + alpha * (
                raw_points - self._smoothed_points
            )
        self._smoothed_points = points
        self._last_face_timestamp_ms = now_ms
        return FaceDetection(
            points,
            candidate.blendshapes,
            face_count,
            inference_ms,
            candidate.transformation_matrix,
        )

    def close(self) -> None:
        try:
            self._landmarker.close()
        finally:
            self._fallback_landmarker.close()

    def __enter__(self) -> "FaceLandmarkEngine":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
