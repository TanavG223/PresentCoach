"""Deterministic, identity-free evaluation of the face-landmark tracker.

This module deliberately evaluates the existing MediaPipe pipeline rather than
training on the people in the clips.  It keeps only anonymous geometry long
enough to calculate aggregate tracking measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import time
from typing import Mapping, Sequence

import cv2
import numpy as np

from .landmarks import FaceLandmarkEngine
from .presentation_video_timing import frame_timestamp_seconds


MANIFEST_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 256 * 1024
SUPPORTED_TRANSFORMS = frozenset({"identity", "duplicate_fixed_crop"})
RUNTIME_ONLY_METRICS = frozenset({
    "mean_inference_ms",
    "p95_inference_ms",
    "wall_clock_seconds",
    "processing_throughput_fps",
})
RIGID_LANDMARK_INDICES = np.asarray(
    (1, 6, 10, 109, 152, 168, 234, 338, 454), dtype=np.int64
)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


class TrackingEvaluationError(ValueError):
    """Raised when an evaluation input or manifest is invalid."""


@dataclass(frozen=True)
class TrackingObservation:
    """Anonymous detector result for one sampled frame."""

    timestamp_seconds: float
    face_count: int
    inference_ms: float
    landmarks: np.ndarray | None = None


@dataclass(frozen=True)
class TrackingClipSpec:
    """Validated evaluation case loaded from the checked-in manifest."""

    clip_id: str
    filename: str
    sha256: str
    source_url: str
    source_sha1: str | None
    license_name: str
    purpose: str
    dataset_role: str
    start_seconds: float
    duration_seconds: float
    transform: Mapping[str, object]
    expectations: Mapping[str, float]


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise TrackingEvaluationError(f"{label} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TrackingEvaluationError(f"{label} must be a number") from error
    if not math.isfinite(number) or number < minimum:
        raise TrackingEvaluationError(f"{label} must be finite and >= {minimum}")
    return number


def _validate_transform(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TrackingEvaluationError(f"{label} must be an object")
    transform_type = value.get("type", "identity")
    if transform_type not in SUPPORTED_TRANSFORMS:
        raise TrackingEvaluationError(f"{label}.type is unsupported")
    if transform_type == "identity":
        return {"type": "identity"}
    crop = value.get("crop_normalized")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in crop)
    ):
        raise TrackingEvaluationError(
            f"{label}.crop_normalized must be [left, top, right, bottom]"
        )
    left, top, right, bottom = (float(item) for item in crop)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise TrackingEvaluationError(f"{label}.crop_normalized is out of range")
    return {
        "type": "duplicate_fixed_crop",
        "crop_normalized": [left, top, right, bottom],
    }


def load_tracking_manifest(path: Path) -> tuple[dict[str, object], tuple[TrackingClipSpec, ...]]:
    """Load and strictly validate a bounded tracking-evaluation manifest."""

    try:
        if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
            raise TrackingEvaluationError("Tracking manifest is missing or too large")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrackingEvaluationError("Tracking manifest is not valid JSON") from error
    if not isinstance(document, dict):
        raise TrackingEvaluationError("Tracking manifest must be an object")
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TrackingEvaluationError("Unsupported tracking manifest schema")
    target_fps = _finite_number(document.get("target_fps"), "target_fps", minimum=1.0)
    if target_fps > 60:
        raise TrackingEvaluationError("target_fps must be <= 60")
    model = document.get("model")
    if not isinstance(model, dict):
        raise TrackingEvaluationError("model must be an object")
    model_filename = model.get("filename")
    model_sha256 = model.get("sha256")
    if (
        not isinstance(model_filename, str)
        or Path(model_filename).name != model_filename
        or not isinstance(model_sha256, str)
        or len(model_sha256) != 64
    ):
        raise TrackingEvaluationError("model filename or SHA-256 is invalid")
    try:
        int(model_sha256, 16)
    except ValueError as error:
        raise TrackingEvaluationError("model filename or SHA-256 is invalid") from error
    clips_value = document.get("clips")
    if not isinstance(clips_value, list) or not clips_value:
        raise TrackingEvaluationError("clips must be a non-empty list")
    if len(clips_value) > 50:
        raise TrackingEvaluationError("clips exceeds the 50-case limit")

    specs: list[TrackingClipSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(clips_value):
        label = f"clips[{index}]"
        if not isinstance(item, dict):
            raise TrackingEvaluationError(f"{label} must be an object")
        clip_id = item.get("id")
        filename = item.get("filename")
        digest = item.get("sha256")
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise TrackingEvaluationError(f"{label}.id must be unique and non-empty")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise TrackingEvaluationError(f"{label}.filename must be a base filename")
        if not isinstance(digest, str) or len(digest) != 64:
            raise TrackingEvaluationError(f"{label}.sha256 is invalid")
        try:
            int(digest, 16)
        except ValueError as error:
            raise TrackingEvaluationError(f"{label}.sha256 is invalid") from error
        source_url = item.get("source_url")
        source_sha1 = item.get("source_sha1")
        license_name = item.get("license")
        purpose = item.get("purpose")
        if not all(isinstance(value, str) and value for value in (source_url, license_name, purpose)):
            raise TrackingEvaluationError(f"{label} needs source_url, license, and purpose")
        if source_sha1 is not None:
            if not isinstance(source_sha1, str) or len(source_sha1) != 40:
                raise TrackingEvaluationError(f"{label}.source_sha1 is invalid")
            try:
                int(source_sha1, 16)
            except ValueError as error:
                raise TrackingEvaluationError(f"{label}.source_sha1 is invalid") from error
        dataset_role = item.get("dataset_role", "regression")
        if dataset_role not in {"regression", "development", "holdout", "derived"}:
            raise TrackingEvaluationError(f"{label}.dataset_role is invalid")
        expectations = item.get("expectations")
        if not isinstance(expectations, dict) or not expectations:
            raise TrackingEvaluationError(f"{label}.expectations must be a non-empty object")
        parsed_expectations = {
            str(key): _finite_number(value, f"{label}.expectations.{key}")
            for key, value in expectations.items()
        }
        unknown = set(parsed_expectations) - set(_CHECK_DEFINITIONS)
        if unknown:
            raise TrackingEvaluationError(
                f"{label}.expectations contains unsupported checks: {sorted(unknown)}"
            )
        seen.add(clip_id)
        specs.append(TrackingClipSpec(
            clip_id=clip_id,
            filename=filename,
            sha256=digest.lower(),
            source_url=source_url,
            source_sha1=source_sha1.lower() if source_sha1 is not None else None,
            license_name=license_name,
            purpose=purpose,
            dataset_role=dataset_role,
            start_seconds=_finite_number(item.get("start_seconds", 0), f"{label}.start_seconds"),
            duration_seconds=_finite_number(
                item.get("duration_seconds"), f"{label}.duration_seconds", minimum=0.1
            ),
            transform=_validate_transform(item.get("transform", {"type": "identity"}), f"{label}.transform"),
            expectations=parsed_expectations,
        ))
    normalized = dict(document)
    normalized["target_fps"] = target_fps
    return normalized, tuple(specs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_metric_projection(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """Remove hardware/runtime timing while retaining every model output metric."""

    return {
        str(key): value
        for key, value in metrics.items()
        if key not in RUNTIME_ONLY_METRICS
    }


def _valid_points(observation: TrackingObservation) -> np.ndarray | None:
    points = observation.landmarks
    if observation.face_count != 1 or points is None:
        return None
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] <= int(RIGID_LANDMARK_INDICES.max()) or array.shape[1] < 2:
        return None
    if not np.isfinite(array[:, :2]).all():
        return None
    return array


def _shape_state(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    left = points[LEFT_EYE_OUTER, :2]
    right = points[RIGHT_EYE_OUTER, :2]
    eye_vector = right - left
    scale = float(np.linalg.norm(eye_vector))
    if scale <= 1e-6:
        return None
    center = (left + right) / 2.0
    angle = math.atan2(float(eye_vector[1]), float(eye_vector[0]))
    cosine = math.cos(-angle)
    sine = math.sin(-angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    shape = ((points[RIGID_LANDMARK_INDICES, :2] - center) @ rotation.T) / scale
    return shape, center, scale


def _percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _dropout_runs(
    observations: Sequence[TrackingObservation], sample_interval: float
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    start = 0
    while start < len(observations):
        if _valid_points(observations[start]) is not None:
            start += 1
            continue
        end = start
        while end + 1 < len(observations) and _valid_points(observations[end + 1]) is None:
            end += 1
        counts = {observations[index].face_count for index in range(start, end + 1)}
        if counts == {0}:
            kind = "no_face"
        elif counts and min(counts) > 1:
            kind = "multiple_faces"
        else:
            kind = "mixed_or_invalid"
        previous_valid = start > 0 and _valid_points(observations[start - 1]) is not None
        next_valid = end + 1 < len(observations) and _valid_points(observations[end + 1]) is not None
        reacquisition = None
        if previous_valid and next_valid:
            reacquisition = max(
                0.0,
                observations[end + 1].timestamp_seconds
                - observations[start].timestamp_seconds,
            )
        runs.append({
            "kind": kind,
            "start_seconds": round(observations[start].timestamp_seconds, 4),
            "end_seconds": round(observations[end].timestamp_seconds, 4),
            "duration_seconds": round(
                max(sample_interval, observations[end].timestamp_seconds - observations[start].timestamp_seconds + sample_interval),
                4,
            ),
            "frame_count": end - start + 1,
            "bracketed_by_valid_track": bool(previous_valid and next_valid),
            "reacquisition_seconds": None if reacquisition is None else round(reacquisition, 4),
        })
        start = end + 1
    return runs


def compute_tracking_metrics(
    observations: Sequence[TrackingObservation],
    *,
    evaluation_duration_seconds: float,
    target_fps: float,
    stable_min_frames: int = 8,
) -> dict[str, object]:
    """Calculate aggregate anonymous tracking measurements.

    Landmark jitter is only reported for continuous single-face runs of at
    least ``stable_min_frames`` whose frame-to-frame face center moves no more
    than 2% of the image and whose inter-eye scale changes no more than 8%.
    Geometry is centered, eye-line aligned, and inter-eye normalized first.
    """

    if evaluation_duration_seconds <= 0 or target_fps <= 0:
        raise TrackingEvaluationError("Evaluation duration and target FPS must be positive")
    if stable_min_frames < 2:
        raise TrackingEvaluationError("stable_min_frames must be >= 2")
    ordered = tuple(sorted(observations, key=lambda item: item.timestamp_seconds))
    sample_interval = 1.0 / target_fps
    valid = [_valid_points(item) is not None for item in ordered]
    valid_count = sum(valid)
    no_face_count = sum(item.face_count == 0 for item in ordered)
    multi_face_count = sum(item.face_count > 1 for item in ordered)
    inconsistent_count = sum(
        (item.face_count == 1 and not is_valid)
        or (item.face_count != 1 and item.landmarks is not None)
        for item, is_valid in zip(ordered, valid)
    )
    multi_face_abstention_violations = sum(
        item.face_count > 1 and item.landmarks is not None for item in ordered
    )
    inference_values = [
        float(item.inference_ms)
        for item in ordered
        if math.isfinite(float(item.inference_ms)) and item.inference_ms >= 0
    ]

    stable_run: list[tuple[np.ndarray, np.ndarray, float]] = []
    stable_jitters: list[float] = []
    stable_segment_count = 0

    def finish_stable_run() -> None:
        nonlocal stable_segment_count
        if len(stable_run) < stable_min_frames:
            stable_run.clear()
            return
        stable_segment_count += 1
        for previous, current in zip(stable_run, stable_run[1:]):
            previous_shape = previous[0]
            current_shape = current[0]
            stable_jitters.append(float(np.sqrt(np.mean((current_shape - previous_shape) ** 2))))
        stable_run.clear()

    for item in ordered:
        points = _valid_points(item)
        state = _shape_state(points) if points is not None else None
        if state is None:
            finish_stable_run()
            continue
        if stable_run:
            _, previous_center, previous_scale = stable_run[-1]
            _, current_center, current_scale = state
            center_delta = float(np.linalg.norm(current_center - previous_center))
            scale_delta = abs(current_scale - previous_scale) / max(previous_scale, 1e-9)
            if center_delta > 0.02 or scale_delta > 0.08:
                finish_stable_run()
        stable_run.append(state)
    finish_stable_run()

    runs = _dropout_runs(ordered, sample_interval)
    reacquisitions = [
        float(run["reacquisition_seconds"])
        for run in runs
        if run["reacquisition_seconds"] is not None
    ]
    longest_dropout = max((float(run["duration_seconds"]) for run in runs), default=0.0)
    count = len(ordered)
    return {
        "analyzed_frame_count": count,
        "analyzed_fps": round(count / evaluation_duration_seconds, 3),
        "valid_face_frame_count": valid_count,
        "valid_face_frame_ratio": round(valid_count / count, 6) if count else 0.0,
        "no_face_frame_count": no_face_count,
        "no_face_frame_ratio": round(no_face_count / count, 6) if count else 0.0,
        "multi_face_frame_count": multi_face_count,
        "multi_face_frame_ratio": round(multi_face_count / count, 6) if count else 0.0,
        "inconsistent_detection_count": inconsistent_count,
        "multi_face_abstention_violations": multi_face_abstention_violations,
        "dropout_run_count": len(runs),
        "dropout_runs": runs,
        "longest_dropout_seconds": round(longest_dropout, 4),
        "reacquisition_event_count": len(reacquisitions),
        "mean_reacquisition_seconds": None if not reacquisitions else round(fmean(reacquisitions), 4),
        "max_reacquisition_seconds": None if not reacquisitions else round(max(reacquisitions), 4),
        "stable_segment_count": stable_segment_count,
        "stable_transition_count": len(stable_jitters),
        "landmark_jitter_median": None if not stable_jitters else round(float(_percentile(stable_jitters, 50) or 0.0), 6),
        "landmark_jitter_p95": None if not stable_jitters else round(float(_percentile(stable_jitters, 95) or 0.0), 6),
        "mean_inference_ms": None if not inference_values else round(fmean(inference_values), 3),
        "p95_inference_ms": None if not inference_values else round(float(_percentile(inference_values, 95) or 0.0), 3),
    }


def _apply_transform(frame: np.ndarray, transform: Mapping[str, object]) -> np.ndarray:
    transform_type = transform.get("type", "identity")
    if transform_type == "identity":
        return frame
    if transform_type != "duplicate_fixed_crop":
        raise TrackingEvaluationError("Unsupported frame transform")
    crop = transform["crop_normalized"]
    if not isinstance(crop, list) or len(crop) != 4:
        raise TrackingEvaluationError("Invalid duplicate crop")
    height, width = frame.shape[:2]
    left, top, right, bottom = (float(value) for value in crop)
    x0, y0 = int(round(left * width)), int(round(top * height))
    x1, y1 = int(round(right * width)), int(round(bottom * height))
    source = frame[y0:y1, x0:x1]
    if source.size == 0:
        raise TrackingEvaluationError("Duplicate crop is empty")
    tile = cv2.resize(source, (400, 450), interpolation=cv2.INTER_AREA)
    output = np.full((480, 900, 3), 220, dtype=np.uint8)
    output[15:465, 25:425] = tile
    output[15:465, 475:875] = tile
    return output


def analyze_tracking_clip(
    path: Path,
    *,
    model_path: Path,
    start_seconds: float,
    duration_seconds: float,
    target_fps: float,
    transform: Mapping[str, object],
) -> tuple[dict[str, object], float]:
    """Run the production detector over one bounded clip excerpt."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise TrackingEvaluationError("Video could not be decoded")
    source_duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / max(
        float(capture.get(cv2.CAP_PROP_FPS)), 1.0
    )
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise TrackingEvaluationError("Video duration is invalid")
    if start_seconds >= source_duration:
        raise TrackingEvaluationError("Evaluation start is beyond the video")
    evaluation_duration = min(duration_seconds, source_duration - start_seconds)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
    end_seconds = start_seconds + evaluation_duration
    next_sample = start_seconds
    observations: list[TrackingObservation] = []
    engine = FaceLandmarkEngine(model_path)
    wall_started = time.perf_counter()
    decoded_index = 0
    previous_timestamp: float | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_timestamp_seconds(
                float(capture.get(cv2.CAP_PROP_POS_MSEC)),
                frame_index=decoded_index,
                source_fps=source_fps,
                previous_seconds=previous_timestamp,
                origin_seconds=start_seconds,
            )
            previous_timestamp = timestamp
            decoded_index += 1
            if timestamp < start_seconds - (0.5 / source_fps):
                continue
            if timestamp >= end_seconds:
                break
            if timestamp + (0.5 / source_fps) < next_sample:
                continue
            evaluated_frame = _apply_transform(frame, transform)
            if evaluated_frame.shape[1] > 960:
                scale = 960.0 / evaluated_frame.shape[1]
                evaluated_frame = cv2.resize(
                    evaluated_frame,
                    (960, max(1, round(evaluated_frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            relative_timestamp = max(0.0, timestamp - start_seconds)
            detection = engine.detect(
                evaluated_frame, timestamp_ms=round(relative_timestamp * 1000.0)
            )
            observations.append(TrackingObservation(
                timestamp_seconds=relative_timestamp,
                face_count=detection.face_count,
                inference_ms=detection.inference_ms,
                landmarks=None if detection.landmarks is None else detection.landmarks.copy(),
            ))
            next_sample += 1.0 / target_fps
            if timestamp > next_sample:
                next_sample = timestamp + 1.0 / target_fps
    finally:
        wall_seconds = time.perf_counter() - wall_started
        engine.close()
        capture.release()
    if not observations:
        raise TrackingEvaluationError("No frames were analyzed")
    metrics = compute_tracking_metrics(
        observations,
        evaluation_duration_seconds=evaluation_duration,
        target_fps=target_fps,
    )
    metrics["wall_clock_seconds"] = round(wall_seconds, 3)
    metrics["processing_throughput_fps"] = round(len(observations) / max(wall_seconds, 1e-9), 3)
    return metrics, evaluation_duration


@dataclass(frozen=True)
class _CheckDefinition:
    metric: str
    comparison: str
    label: str


_CHECK_DEFINITIONS: dict[str, _CheckDefinition] = {
    "min_analyzed_fps": _CheckDefinition("analyzed_fps", "min", "Analyzed frame cadence"),
    "max_analyzed_fps": _CheckDefinition("analyzed_fps", "max", "Analyzed frame cadence upper bound"),
    "min_valid_face_frame_ratio": _CheckDefinition("valid_face_frame_ratio", "min", "Valid single-face frames"),
    "max_valid_face_frame_ratio": _CheckDefinition("valid_face_frame_ratio", "max", "Expected difficult-face abstention"),
    "min_dropout_run_count": _CheckDefinition("dropout_run_count", "min", "Expected tracking dropout runs"),
    "max_dropout_run_count": _CheckDefinition("dropout_run_count", "max", "Tracking dropout runs"),
    "max_longest_dropout_seconds": _CheckDefinition("longest_dropout_seconds", "max", "Longest tracking dropout"),
    "min_reacquisition_event_count": _CheckDefinition("reacquisition_event_count", "min", "Track reacquisition events"),
    "max_reacquisition_seconds": _CheckDefinition("max_reacquisition_seconds", "max", "Track reacquisition time"),
    "min_stable_transition_count": _CheckDefinition("stable_transition_count", "min", "Defensible stable-landmark transitions"),
    "max_landmark_jitter_p95": _CheckDefinition("landmark_jitter_p95", "max", "Stable-segment landmark jitter p95"),
    "min_multi_face_frame_ratio": _CheckDefinition("multi_face_frame_ratio", "min", "Multiple-face challenge frames"),
    "max_multi_face_abstention_violations": _CheckDefinition("multi_face_abstention_violations", "max", "Multiple-face abstention violations"),
    "max_inconsistent_detection_count": _CheckDefinition("inconsistent_detection_count", "max", "Detector output inconsistencies"),
}


def build_tracking_checks(
    metrics: Mapping[str, object], expectations: Mapping[str, float]
) -> list[dict[str, object]]:
    """Apply only declared, deterministic manifest expectations."""

    checks: list[dict[str, object]] = []
    for key, expected in expectations.items():
        definition = _CHECK_DEFINITIONS.get(key)
        if definition is None:
            raise TrackingEvaluationError(f"Unsupported expectation: {key}")
        actual = metrics.get(definition.metric)
        comparable = isinstance(actual, (int, float)) and not isinstance(actual, bool)
        if comparable:
            actual_number = float(actual)
            passed = actual_number >= expected if definition.comparison == "min" else actual_number <= expected
        else:
            passed = False
        operator = ">=" if definition.comparison == "min" else "<="
        checks.append({
            "id": key,
            "label": definition.label,
            "passed": passed,
            "actual": actual,
            "expected": f"{operator} {expected:g}",
        })
    return checks
