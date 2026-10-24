"""Ground-truth scoring for anonymous presentation face tracking.

This module intentionally does not train MediaPipe, associate identities across
clips, or infer gaze truth from natural video.  It compares existing
``TrackingObservation`` values with bounded, independently supplied interval
annotations and fails closed when a metric does not have enough evidence.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Mapping, Sequence

import numpy as np

from .presentation_tracking_eval import TrackingObservation


ANNOTATION_SCHEMA_VERSION = 1
MAX_ANNOTATION_BYTES = 2 * 1024 * 1024
MAX_ANNOTATION_INTERVALS = 10_000
MAX_EVALUATION_SECONDS = 1_800.0
MAX_OBSERVATIONS = 108_000
MIN_REQUIRED_LANDMARKS = 455

FACE_STATES = frozenset({
    "single_trackable",
    "no_face",
    "multiple_faces",
    "occluded_or_tiny",
    "uncertain",
})
NONTRACKABLE_STATES = frozenset({
    "no_face", "multiple_faces", "occluded_or_tiny",
})
MOTION_LABELS = frozenset({
    "static", "translate", "scale", "head_turn", "mixed", "unknown",
})
GAZE_LABELS = frozenset({"contact", "away", "ineligible", "unknown"})
SCORABLE_GAZE_LABELS = frozenset({"contact", "away"})
DATASET_ROLES = frozenset({
    "development", "model_output_holdout", "derived_regression",
})
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?")


class GroundTruthValidationError(ValueError):
    """Raised when annotation, split, policy, or observation input is invalid."""


@dataclass(frozen=True)
class GroundTruthInterval:
    """One half-open, anonymous ground-truth interval."""

    start_seconds: float
    end_seconds: float
    face_state: str
    motion: str
    gaze_truth: str
    gaze_label_explicit: bool = False


@dataclass(frozen=True)
class GroundTruthAnnotations:
    """Strictly validated annotations for one bounded clip excerpt."""

    clip_id: str
    split_group_id: str
    duration_seconds: float
    intervals: tuple[GroundTruthInterval, ...]

    @property
    def has_explicit_gaze_labels(self) -> bool:
        return any(
            interval.gaze_label_explicit
            and interval.gaze_truth in SCORABLE_GAZE_LABELS
            for interval in self.intervals
        )


@dataclass(frozen=True)
class GroundTruthPowerPolicy:
    """Minimum evidence required before a scalar metric may be reported."""

    minimum_trackable_frames: int = 15
    minimum_nontrackable_frames: int = 15
    minimum_multi_face_frames: int = 15
    minimum_trackable_seconds: float = 10.0
    minimum_reacquisition_events: int = 3
    minimum_sampling_coverage_ratio: float = 0.80

    def validate(self) -> None:
        integer_fields = (
            ("minimum_trackable_frames", self.minimum_trackable_frames),
            ("minimum_nontrackable_frames", self.minimum_nontrackable_frames),
            ("minimum_multi_face_frames", self.minimum_multi_face_frames),
            ("minimum_reacquisition_events", self.minimum_reacquisition_events),
        )
        for label, value in integer_fields:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= MAX_OBSERVATIONS
            ):
                raise GroundTruthValidationError(
                    f"{label} must be an integer between 1 and {MAX_OBSERVATIONS}"
                )
        seconds = self.minimum_trackable_seconds
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or not 0 < float(seconds) <= MAX_EVALUATION_SECONDS
        ):
            raise GroundTruthValidationError(
                "minimum_trackable_seconds must be finite and within the evaluation limit"
            )
        ratio = self.minimum_sampling_coverage_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0 < float(ratio) <= 1
        ):
            raise GroundTruthValidationError(
                "minimum_sampling_coverage_ratio must be finite and within (0, 1]"
            )


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float = MAX_EVALUATION_SECONDS,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundTruthValidationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise GroundTruthValidationError(
            f"{label} must be finite and between {minimum:g} and {maximum:g}"
        )
    return number


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise GroundTruthValidationError(f"{label} is invalid")
    return value


def parse_ground_truth_annotations(document: object) -> GroundTruthAnnotations:
    """Validate a bounded annotation document without reading arbitrary media."""

    if not isinstance(document, dict):
        raise GroundTruthValidationError("Annotation document must be an object")
    allowed_document_keys = {
        "schema_version", "clip_id", "split_group_id", "duration_seconds", "intervals",
    }
    unknown_document_keys = set(document) - allowed_document_keys
    if unknown_document_keys:
        raise GroundTruthValidationError(
            f"Annotation document has unsupported fields: {sorted(unknown_document_keys)}"
        )
    if document.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise GroundTruthValidationError("Unsupported ground-truth annotation schema")
    clip_id = _identifier(document.get("clip_id"), "clip_id")
    split_group_id = _identifier(document.get("split_group_id"), "split_group_id")
    duration = _finite_number(
        document.get("duration_seconds"), "duration_seconds", minimum=0.001
    )
    raw_intervals = document.get("intervals")
    if (
        not isinstance(raw_intervals, list)
        or not raw_intervals
        or len(raw_intervals) > MAX_ANNOTATION_INTERVALS
    ):
        raise GroundTruthValidationError(
            f"intervals must contain between 1 and {MAX_ANNOTATION_INTERVALS} entries"
        )

    intervals: list[GroundTruthInterval] = []
    previous_end = 0.0
    for index, raw in enumerate(raw_intervals):
        label = f"intervals[{index}]"
        if not isinstance(raw, dict):
            raise GroundTruthValidationError(f"{label} must be an object")
        allowed_interval_keys = {
            "start_seconds", "end_seconds", "face_state", "motion", "gaze_truth",
        }
        unknown_interval_keys = set(raw) - allowed_interval_keys
        if unknown_interval_keys:
            raise GroundTruthValidationError(
                f"{label} has unsupported fields: {sorted(unknown_interval_keys)}"
            )
        start = _finite_number(
            raw.get("start_seconds"), f"{label}.start_seconds", maximum=duration
        )
        end = _finite_number(
            raw.get("end_seconds"), f"{label}.end_seconds", maximum=duration
        )
        if end <= start:
            raise GroundTruthValidationError(
                f"{label} must have end_seconds > start_seconds"
            )
        if intervals and start < previous_end - 1e-9:
            raise GroundTruthValidationError(
                "Annotation intervals must be ordered and non-overlapping"
            )
        face_state = raw.get("face_state")
        motion = raw.get("motion")
        gaze_explicit = "gaze_truth" in raw
        gaze_truth = raw.get("gaze_truth", "unknown")
        if face_state not in FACE_STATES:
            raise GroundTruthValidationError(f"{label}.face_state is unsupported")
        if motion not in MOTION_LABELS:
            raise GroundTruthValidationError(f"{label}.motion is unsupported")
        if gaze_truth not in GAZE_LABELS:
            raise GroundTruthValidationError(f"{label}.gaze_truth is unsupported")
        if face_state != "single_trackable" and motion != "unknown":
            raise GroundTruthValidationError(
                f"{label}.motion must be unknown outside single_trackable intervals"
            )
        if face_state != "single_trackable" and gaze_truth in SCORABLE_GAZE_LABELS:
            raise GroundTruthValidationError(
                f"{label}.gaze_truth cannot claim contact outside a trackable interval"
            )
        intervals.append(GroundTruthInterval(
            start_seconds=start,
            end_seconds=end,
            face_state=str(face_state),
            motion=str(motion),
            gaze_truth=str(gaze_truth),
            gaze_label_explicit=gaze_explicit,
        ))
        previous_end = end

    return GroundTruthAnnotations(
        clip_id=clip_id,
        split_group_id=split_group_id,
        duration_seconds=duration,
        intervals=tuple(intervals),
    )


def load_ground_truth_annotations(path: Path) -> GroundTruthAnnotations:
    """Load a size-bounded UTF-8 JSON annotation file and validate it strictly."""

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise GroundTruthValidationError(
                    f"Annotation file contains duplicate field {key!r}"
                )
            value[key] = item
        return value

    try:
        if not path.is_file() or path.stat().st_size > MAX_ANNOTATION_BYTES:
            raise GroundTruthValidationError("Annotation file is missing or too large")
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundTruthValidationError("Annotation file is not valid UTF-8 JSON") from error
    return parse_ground_truth_annotations(document)


def validate_split_group_isolation(cases: Sequence[Mapping[str, object]]) -> None:
    """Reject source/presenter groups that leak into the model-output holdout."""

    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise GroundTruthValidationError("Split cases must be a sequence")
    if not cases or len(cases) > 1_000:
        raise GroundTruthValidationError("Split cases must contain between 1 and 1000 entries")
    seen_ids: set[str] = set()
    roles_by_group: dict[str, set[str]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise GroundTruthValidationError(f"cases[{index}] must be an object")
        if set(case) != {"clip_id", "dataset_role", "split_group_id"}:
            raise GroundTruthValidationError(
                f"cases[{index}] must contain only clip_id, dataset_role, and split_group_id"
            )
        clip_id = _identifier(case.get("clip_id"), f"cases[{index}].clip_id")
        if clip_id in seen_ids:
            raise GroundTruthValidationError("clip_id values must be unique")
        seen_ids.add(clip_id)
        group = _identifier(
            case.get("split_group_id"), f"cases[{index}].split_group_id"
        )
        role = case.get("dataset_role")
        if role not in DATASET_ROLES:
            raise GroundTruthValidationError(f"cases[{index}].dataset_role is unsupported")
        roles_by_group.setdefault(group, set()).add(str(role))

    for group, roles in roles_by_group.items():
        if "model_output_holdout" in roles and roles != {"model_output_holdout"}:
            raise GroundTruthValidationError(
                f"split_group_id {group!r} leaks into the model-output holdout"
            )


def _valid_landmarks(observation: TrackingObservation) -> bool:
    if observation.face_count != 1 or observation.landmarks is None:
        return False
    try:
        points = np.asarray(observation.landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        points.ndim == 2
        and points.shape[0] >= MIN_REQUIRED_LANDMARKS
        and points.shape[0] <= 1_000
        and points.shape[1] >= 2
        and points.shape[1] <= 4
        and np.isfinite(points[:, :2]).all()
    )


def _validated_observations(
    observations: Sequence[TrackingObservation], duration: float
) -> tuple[TrackingObservation, ...]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise GroundTruthValidationError("observations must be a sequence")
    if len(observations) > MAX_OBSERVATIONS:
        raise GroundTruthValidationError("observations exceed the bounded evaluation limit")
    ordered = tuple(observations)
    previous = -1.0
    for index, observation in enumerate(ordered):
        if not isinstance(observation, TrackingObservation):
            raise GroundTruthValidationError(
                f"observations[{index}] must be a TrackingObservation"
            )
        timestamp = observation.timestamp_seconds
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or not 0 <= float(timestamp) < duration
        ):
            raise GroundTruthValidationError(
                f"observations[{index}].timestamp_seconds is out of range"
            )
        if float(timestamp) <= previous:
            raise GroundTruthValidationError(
                "Observation timestamps must be strictly increasing"
            )
        if (
            isinstance(observation.face_count, bool)
            or not isinstance(observation.face_count, int)
            or observation.face_count < 0
        ):
            raise GroundTruthValidationError(
                f"observations[{index}].face_count must be a non-negative integer"
            )
        previous = float(timestamp)
    return ordered


def _interval_index(
    timestamp: float,
    intervals: Sequence[GroundTruthInterval],
    starts: Sequence[float],
) -> int | None:
    index = bisect_right(starts, timestamp) - 1
    if index < 0 or timestamp >= intervals[index].end_seconds:
        return None
    return index


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6)


def _sampling_coverage(count: int, seconds: float, target_fps: float) -> float:
    expected = seconds * target_fps
    return round(min(1.0, count / expected), 6) if expected > 0 else 0.0


def score_tracking_ground_truth(
    observations: Sequence[TrackingObservation],
    annotations: GroundTruthAnnotations,
    *,
    target_fps: float,
    power_policy: GroundTruthPowerPolicy | None = None,
) -> dict[str, object]:
    """Score model output against interval ground truth with explicit abstention.

    Reacquisition succeeds only when the scorer finds three consecutive valid
    observations in the new ``single_trackable`` interval.  Any missing sample
    or timestamp gap larger than 1.5 expected frame intervals breaks a run.
    """

    if not isinstance(annotations, GroundTruthAnnotations):
        raise GroundTruthValidationError("annotations must be validated first")
    if (
        isinstance(target_fps, bool)
        or not isinstance(target_fps, (int, float))
        or not math.isfinite(float(target_fps))
        or not 1 <= float(target_fps) <= 60
    ):
        raise GroundTruthValidationError("target_fps must be between 1 and 60")
    policy = power_policy or GroundTruthPowerPolicy()
    if not isinstance(policy, GroundTruthPowerPolicy):
        raise GroundTruthValidationError("power_policy is invalid")
    policy.validate()
    ordered = _validated_observations(observations, annotations.duration_seconds)
    intervals = annotations.intervals
    starts = tuple(interval.start_seconds for interval in intervals)
    labeled: list[tuple[TrackingObservation, int, bool]] = []
    unlabeled_count = 0
    for observation in ordered:
        interval_index = _interval_index(
            float(observation.timestamp_seconds), intervals, starts
        )
        if interval_index is None:
            unlabeled_count += 1
            continue
        labeled.append((observation, interval_index, _valid_landmarks(observation)))

    trackable = [
        item
        for item in labeled
        if intervals[item[1]].face_state == "single_trackable"
    ]
    nontrackable = [
        item
        for item in labeled
        if intervals[item[1]].face_state in NONTRACKABLE_STATES
    ]
    multiple_faces = [
        item
        for item in labeled
        if intervals[item[1]].face_state == "multiple_faces"
    ]
    uncertain_count = sum(
        intervals[item[1]].face_state == "uncertain" for item in labeled
    )
    motion_observation_counts = {
        motion: sum(
            intervals[item[1]].face_state == "single_trackable"
            and intervals[item[1]].motion == motion
            for item in labeled
        )
        for motion in sorted(MOTION_LABELS - {"unknown"})
    }
    trackable_seconds = sum(
        interval.end_seconds - interval.start_seconds
        for interval in intervals
        if interval.face_state == "single_trackable"
    )
    nontrackable_seconds = sum(
        interval.end_seconds - interval.start_seconds
        for interval in intervals
        if interval.face_state in NONTRACKABLE_STATES
    )
    multi_face_seconds = sum(
        interval.end_seconds - interval.start_seconds
        for interval in intervals
        if interval.face_state == "multiple_faces"
    )
    trackable_sampling_coverage = _sampling_coverage(
        len(trackable), trackable_seconds, float(target_fps)
    )
    nontrackable_sampling_coverage = _sampling_coverage(
        len(nontrackable), nontrackable_seconds, float(target_fps)
    )
    multi_face_sampling_coverage = _sampling_coverage(
        len(multiple_faces), multi_face_seconds, float(target_fps)
    )

    reasons: dict[str, str] = {}
    trackable_coverage: float | None
    if (
        len(trackable) < policy.minimum_trackable_frames
        or trackable_sampling_coverage < policy.minimum_sampling_coverage_ratio
    ):
        trackable_coverage = None
        reasons["trackable_coverage"] = (
            f"needs at least {policy.minimum_trackable_frames} labeled trackable frames; "
            f"found {len(trackable)} with {trackable_sampling_coverage:.1%} sampling coverage"
        )
    else:
        trackable_coverage = _rounded_ratio(
            sum(item[2] for item in trackable), len(trackable)
        )

    false_valid_rate: float | None
    if (
        len(nontrackable) < policy.minimum_nontrackable_frames
        or nontrackable_sampling_coverage < policy.minimum_sampling_coverage_ratio
    ):
        false_valid_rate = None
        reasons["false_valid_rate"] = (
            f"needs at least {policy.minimum_nontrackable_frames} labeled non-trackable frames; "
            f"found {len(nontrackable)} with {nontrackable_sampling_coverage:.1%} sampling coverage"
        )
    else:
        false_valid_rate = _rounded_ratio(
            sum(item[2] for item in nontrackable), len(nontrackable)
        )

    multi_face_abstention: float | None
    if (
        len(multiple_faces) < policy.minimum_multi_face_frames
        or multi_face_sampling_coverage < policy.minimum_sampling_coverage_ratio
    ):
        multi_face_abstention = None
        reasons["multi_face_abstention"] = (
            f"needs at least {policy.minimum_multi_face_frames} labeled multiple-face frames; "
            f"found {len(multiple_faces)} with {multi_face_sampling_coverage:.1%} sampling coverage"
        )
    else:
        multi_face_abstention = _rounded_ratio(
            sum(not item[2] for item in multiple_faces), len(multiple_faces)
        )

    maximum_gap = 1.5 / float(target_fps)
    dropout_run_count = 0
    run_length = 0
    previous_item: tuple[TrackingObservation, int, bool] | None = None
    for item in trackable:
        observation, interval_index, valid = item
        consecutive = bool(
            previous_item is not None
            and previous_item[1] == interval_index
            and float(observation.timestamp_seconds)
            - float(previous_item[0].timestamp_seconds) <= maximum_gap + 1e-9
        )
        if not consecutive:
            if run_length >= 2:
                dropout_run_count += 1
            run_length = 0
        if valid:
            if run_length >= 2:
                dropout_run_count += 1
            run_length = 0
        else:
            run_length += 1
        previous_item = item
    if run_length >= 2:
        dropout_run_count += 1

    dropouts_per_annotated_minute: float | None
    if (
        trackable_seconds < float(policy.minimum_trackable_seconds)
        or len(trackable) < policy.minimum_trackable_frames
        or trackable_sampling_coverage < policy.minimum_sampling_coverage_ratio
    ):
        dropouts_per_annotated_minute = None
        reasons["dropouts_per_annotated_minute"] = (
            f"needs at least {float(policy.minimum_trackable_seconds):g} "
            "annotated trackable seconds with sufficient samples; "
            f"found {trackable_seconds:.3f} seconds and "
            f"{trackable_sampling_coverage:.1%} sampling coverage"
        )
    else:
        dropouts_per_annotated_minute = round(
            dropout_run_count * 60.0 / trackable_seconds, 6
        )

    reacquisition_intervals = [
        index
        for index, interval in enumerate(intervals)
        if (
            index > 0
            and interval.face_state == "single_trackable"
            and intervals[index - 1].face_state in NONTRACKABLE_STATES
            and abs(intervals[index - 1].end_seconds - interval.start_seconds) <= 1e-9
        )
    ]
    by_interval: dict[int, list[tuple[TrackingObservation, bool]]] = {}
    for observation, interval_index, valid in labeled:
        by_interval.setdefault(interval_index, []).append((observation, valid))
    reacquisition_events: list[dict[str, object]] = []
    successful_latencies: list[float] = []
    for interval_index in reacquisition_intervals:
        candidates = by_interval.get(interval_index, [])
        first_valid_timestamp: float | None = None
        for start_index in range(max(0, len(candidates) - 2)):
            window = candidates[start_index : start_index + 3]
            if len(window) != 3 or not all(item[1] for item in window):
                continue
            if any(
                float(right[0].timestamp_seconds) - float(left[0].timestamp_seconds)
                > maximum_gap + 1e-9
                for left, right in zip(window, window[1:])
            ):
                continue
            first_valid_timestamp = float(window[0][0].timestamp_seconds)
            break
        onset = intervals[interval_index].start_seconds
        latency = (
            None
            if first_valid_timestamp is None
            else round(max(0.0, first_valid_timestamp - onset), 6)
        )
        if latency is not None:
            successful_latencies.append(latency)
        reacquisition_events.append({
            "onset_seconds": round(onset, 6),
            "reacquired": latency is not None,
            "latency_seconds": latency,
        })

    event_count = len(reacquisition_events)
    success_count = len(successful_latencies)
    reacquisition_success_rate: float | None
    mean_reacquisition_latency_seconds: float | None
    max_reacquisition_latency_seconds: float | None
    if event_count < policy.minimum_reacquisition_events:
        reacquisition_success_rate = None
        mean_reacquisition_latency_seconds = None
        max_reacquisition_latency_seconds = None
        reason = (
            f"needs at least {policy.minimum_reacquisition_events} annotated reacquisition events; "
            f"found {event_count}"
        )
        reasons["reacquisition_success_rate"] = reason
        reasons["mean_reacquisition_latency_seconds"] = reason
        reasons["max_reacquisition_latency_seconds"] = reason
    else:
        reacquisition_success_rate = _rounded_ratio(success_count, event_count)
        if success_count != event_count:
            mean_reacquisition_latency_seconds = None
            max_reacquisition_latency_seconds = None
            reason = (
                f"{event_count - success_count} of {event_count} annotated events never produced "
                "three consecutive valid frames"
            )
            reasons["mean_reacquisition_latency_seconds"] = reason
            reasons["max_reacquisition_latency_seconds"] = reason
        else:
            mean_reacquisition_latency_seconds = round(
                fmean(successful_latencies), 6
            )
            max_reacquisition_latency_seconds = round(
                max(successful_latencies), 6
            )

    gaze_reason = (
        "tracking annotations contain no explicit contact/away labels"
        if not annotations.has_explicit_gaze_labels
        else "TrackingObservation contains no gaze prediction; use a separate gaze scorer"
    )
    reasons["gaze_accuracy"] = gaze_reason

    return {
        "schema_version": 1,
        "clip_id": annotations.clip_id,
        "split_group_id": annotations.split_group_id,
        "target_fps": float(target_fps),
        "observation_count": len(ordered),
        "labeled_observation_count": len(labeled),
        "unlabeled_observation_count": unlabeled_count,
        "uncertain_observation_count": uncertain_count,
        "trackable_observation_count": len(trackable),
        "trackable_valid_observation_count": sum(item[2] for item in trackable),
        "nontrackable_observation_count": len(nontrackable),
        "false_valid_observation_count": sum(item[2] for item in nontrackable),
        "multi_face_observation_count": len(multiple_faces),
        "multi_face_abstained_observation_count": sum(
            not item[2] for item in multiple_faces
        ),
        "motion_observation_counts": motion_observation_counts,
        "annotated_trackable_seconds": round(trackable_seconds, 6),
        "trackable_sampling_coverage": trackable_sampling_coverage,
        "nontrackable_sampling_coverage": nontrackable_sampling_coverage,
        "multi_face_sampling_coverage": multi_face_sampling_coverage,
        "trackable_coverage": trackable_coverage,
        "false_valid_rate": false_valid_rate,
        "multi_face_abstention": multi_face_abstention,
        "dropout_run_count": dropout_run_count,
        "dropouts_per_annotated_minute": dropouts_per_annotated_minute,
        "reacquisition_event_count": event_count,
        "reacquisition_success_count": success_count,
        "reacquisition_success_rate": reacquisition_success_rate,
        "mean_reacquisition_latency_seconds": mean_reacquisition_latency_seconds,
        "max_reacquisition_latency_seconds": max_reacquisition_latency_seconds,
        "reacquisition_events": reacquisition_events,
        "explicit_gaze_labels": annotations.has_explicit_gaze_labels,
        "gaze_accuracy": None,
        "power_policy": {
            "minimum_trackable_frames": policy.minimum_trackable_frames,
            "minimum_nontrackable_frames": policy.minimum_nontrackable_frames,
            "minimum_multi_face_frames": policy.minimum_multi_face_frames,
            "minimum_trackable_seconds": float(policy.minimum_trackable_seconds),
            "minimum_reacquisition_events": policy.minimum_reacquisition_events,
            "minimum_sampling_coverage_ratio": float(
                policy.minimum_sampling_coverage_ratio
            ),
        },
        "underpowered_reasons": reasons,
    }
