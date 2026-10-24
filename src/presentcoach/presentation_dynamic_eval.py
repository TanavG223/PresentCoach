"""Fail-closed evaluation of anonymous tracking on moving-speaker videos.

The evaluator is deliberately split-aware.  A normal run can only inspect the
development corpus; the model-output holdout must be named explicitly by the
caller.  Face landmarks exist only as short-lived values returned by the
shared production sampler and are reduced to aggregate measurements before a
report is assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence

from .landmarks import (
    FACE_DETECTION_CONFIDENCE,
    FACE_PRESENCE_CONFIDENCE,
    FACE_TRACKING_CONFIDENCE,
    LANDMARK_REACQUISITION_GAP_MS,
    LANDMARK_SMOOTHING_TIME_CONSTANT_MS,
    LANDMARK_TEMPORAL_HOLD_MS,
    MAX_FACE_COUNT,
    MIN_EYE_TO_LOWER_FACE_LUMINANCE_RATIO,
    SMALL_FACE_LIVENESS_AREA_THRESHOLD,
    MULTISCALE_DEDUPLICATION_IOU,
    MULTISCALE_FALLBACK_ENABLED,
    MULTISCALE_TILE_OFFSETS,
    MULTISCALE_TILE_SIZE,
    YUNET_MODEL_FILENAME,
    YUNET_MODEL_SHA256,
    YUNET_LIVENESS_HOLD_MS,
    YUNET_LIVENESS_MIN_IOU,
    YUNET_APPEARANCE_LAG_MAX_MS,
    YUNET_APPEARANCE_LAG_MIN_MS,
    YUNET_APPEARANCE_MOTION_THRESHOLD,
    YUNET_APPEARANCE_SIZE,
    YUNET_LIVENESS_RANGE_THRESHOLD,
    YUNET_LIVENESS_WINDOW_MS,
    YUNET_NMS_THRESHOLD,
    YUNET_PROVISIONAL_TRACK_MS,
    YUNET_PROVISIONAL_REARM_GAP_MS,
    YUNET_PROVISIONAL_LUMINANCE_CHANGE_RATIO,
    YUNET_ROI_SCALES,
    YUNET_SCORE_THRESHOLD,
    YUNET_TOP_K,
)
from .presentation_tracking_eval import (
    TrackingEvaluationError,
    TrackingObservation,
    compute_tracking_metrics,
    deterministic_metric_projection,
    sample_tracking_clip,
    sha256_file,
)
from .presentation_tracking_ground_truth import (
    GroundTruthAnnotations,
    GroundTruthPowerPolicy,
    GroundTruthValidationError,
    NONTRACKABLE_STATES,
    load_ground_truth_annotations,
    score_tracking_ground_truth,
    validate_split_group_isolation,
)


SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
ALLOWED_SPLITS = frozenset({"development", "model_output_holdout"})
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MEDIA_BYTES = 1024 * 1024 * 1024
MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_CLIPS = 50
SHA256_LENGTH = 64

_MANIFEST_KEYS = frozenset({
    "schema_version",
    "description",
    "evaluation_only",
    "never_training_data",
    "identity_recognition",
    "face_embeddings_allowed",
    "landmarks_persisted",
    "gaze_ground_truth_available",
    "gaze_annotation_policy",
    "binary_distribution",
    "target_fps",
    "model",
    "power_policy",
    "scoring_policy",
    "annotation_contract",
    "canonical_excerpt",
    "split_policy",
    "clips",
})
_MODEL_KEYS = frozenset({"filename", "sha256"})
_POWER_POLICY_KEYS = frozenset({
    "minimum_trackable_frames",
    "minimum_nontrackable_frames",
    "minimum_multi_face_frames",
    "minimum_trackable_seconds",
    "minimum_reacquisition_events",
    "minimum_sampling_coverage_ratio",
})
_THRESHOLD_RULES: Mapping[str, tuple[str, str]] = {
    "trackable_coverage_min": ("trackable_coverage", "min"),
    "false_valid_rate_max": ("false_valid_rate", "max"),
    "multi_face_abstention_min": ("multi_face_abstention", "min"),
    "dropouts_per_annotated_minute_max": (
        "dropouts_per_annotated_minute",
        "max",
    ),
    "reacquisition_success_rate_min": ("reacquisition_success_rate", "min"),
    "max_reacquisition_latency_seconds_max": (
        "max_reacquisition_latency_seconds",
        "max",
    ),
}
_ACCEPTANCE_POLICY_KEYS = frozenset(_THRESHOLD_RULES)
_SCORING_POLICY_KEYS = frozenset({
    "scope",
    "thresholds_frozen_before_model_output_holdout",
    "aggregation",
    "uncertain_and_unlabeled_intervals",
    "underpowered_metric_behavior",
    "gaze_metric_behavior",
    "required_metrics",
    "acceptance_thresholds",
    "claim_boundary",
})
_ANNOTATION_CONTRACT_KEYS = frozenset({
    "method",
    "curator_count",
    "annotation_date",
    "manual_sampling_interval_seconds",
    "boundary_refinement_interval_seconds",
    "boundary_uncertainty_seconds",
    "frame_exhaustive",
    "independently_adjudicated",
    "accuracy_claim_allowed",
    "single_trackable",
    "no_face",
    "multiple_faces",
    "occluded_or_tiny",
    "uncertain",
    "motion",
    "gaze_truth",
    "limitations",
    "pre_release_curator_qa",
})
_PRE_RELEASE_CURATOR_QA_KEYS = frozenset({
    "completed_date",
    "status",
    "canonical_excerpts_reviewed",
    "method",
    "model_or_evaluator_output_viewed",
    "results_generated_from_draft_annotations",
    "corrections",
})
_CLIP_KEYS = frozenset({
    "id",
    "filename",
    "sha256",
    "dataset_role",
    "split_group_id",
    "excerpt_start_seconds",
    "excerpt_duration_seconds",
    "annotation_file",
    "annotation_sha256",
    "manual_visual_inspection_only",
    "model_output_examined_during_scouting",
    "source",
    "attribution",
})
_SOURCE_KEYS = frozenset({
    "title",
    "page_url",
    "derivative_url",
    "derivative_filename",
    "derivative_sha256",
    "original_sha1",
    "original_duration_seconds",
    "original_resolution",
    "derivative_resolution",
})
_ATTRIBUTION_KEYS = frozenset({
    "creator",
    "license_name",
    "license_spdx",
    "license_url",
    "credit_line",
    "change_notice",
    "redistribution_constraints",
})
_CANONICAL_EXCERPT_KEYS = frozenset({
    "container",
    "video_codec",
    "width",
    "height",
    "frame_rate",
    "pixel_format",
    "audio_codec",
    "audio_sample_rate_hz",
    "audio_channels",
    "duration_seconds",
    "metadata_removed",
    "reference_encoder",
    "reproducibility_note",
})
_SPLIT_POLICY_KEYS = frozenset({"development", "model_output_holdout", "grouping"})


class DynamicEvaluationError(ValueError):
    """Raised when evaluation evidence cannot be trusted."""


@dataclass(frozen=True)
class DynamicClipSpec:
    """One validated, immutable moving-speaker evaluation case."""

    clip_id: str
    filename: str
    media_sha256: str
    dataset_role: str
    split_group_id: str
    duration_seconds: float
    annotation_file: str
    annotation_sha256: str
    source_title: str
    source_page_url: str
    license_spdx: str


@dataclass(frozen=True)
class DynamicManifest:
    """Validated execution contract from the checked-in corpus manifest."""

    target_fps: float
    model_filename: str
    model_sha256: str
    power_policy: GroundTruthPowerPolicy
    acceptance_policy: Mapping[str, object]
    scoring_policy: Mapping[str, object]
    clips: tuple[DynamicClipSpec, ...]
    document: Mapping[str, object]


Sampler = Callable[..., tuple[tuple[TrackingObservation, ...], float, dict[str, float]]]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DynamicEvaluationError(f"JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _load_bounded_json(path: Path, *, maximum_bytes: int, label: str) -> object:
    try:
        if not path.is_file():
            raise DynamicEvaluationError(f"{label} is missing")
        size = path.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise DynamicEvaluationError(f"{label} is empty or exceeds {maximum_bytes} bytes")
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DynamicEvaluationError(f"{label} is not valid UTF-8 JSON") from error


def _exact_keys(value: object, expected: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DynamicEvaluationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise DynamicEvaluationError(
            f"{label} fields do not match the schema; missing={missing}, unknown={unknown}"
        )
    return value


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float = 1_800.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicEvaluationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise DynamicEvaluationError(
            f"{label} must be finite and between {minimum:g} and {maximum:g}"
        )
    return number


def _positive_integer(value: object, label: str, *, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise DynamicEvaluationError(f"{label} must be an integer between 1 and {maximum}")
    return value


def _nonempty_string(value: object, label: str, *, maximum: int = 10_000) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise DynamicEvaluationError(f"{label} must be a non-empty bounded string")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise DynamicEvaluationError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise DynamicEvaluationError(f"{label} must be a SHA-256 digest") from error
    return value.lower()


def _safe_relative_path(value: object, label: str, *, base_filename: bool = False) -> str:
    text = _nonempty_string(value, label, maximum=255)
    candidate = Path(text)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise DynamicEvaluationError(f"{label} must be a safe relative path")
    if base_filename and candidate.name != text:
        raise DynamicEvaluationError(f"{label} must be a base filename")
    return text


def _resolution(value: object, label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise DynamicEvaluationError(f"{label} must be [positive width, positive height]")


def _validate_metadata(document: Mapping[str, object]) -> None:
    for key, expected in (
        ("evaluation_only", True),
        ("never_training_data", True),
        ("identity_recognition", False),
        ("face_embeddings_allowed", False),
        ("landmarks_persisted", False),
        ("gaze_ground_truth_available", False),
    ):
        if document.get(key) is not expected:
            raise DynamicEvaluationError(f"Manifest privacy contract requires {key}={expected!r}")
    for key in (
        "description",
        "gaze_annotation_policy",
        "binary_distribution",
    ):
        _nonempty_string(document.get(key), key)

    canonical = _exact_keys(
        document.get("canonical_excerpt"),
        _CANONICAL_EXCERPT_KEYS,
        "canonical_excerpt",
    )
    for key in (
        "container",
        "video_codec",
        "pixel_format",
        "audio_codec",
        "reference_encoder",
        "reproducibility_note",
    ):
        _nonempty_string(canonical.get(key), f"canonical_excerpt.{key}")
    for key in ("width", "height", "frame_rate", "audio_sample_rate_hz", "audio_channels"):
        _positive_integer(canonical.get(key), f"canonical_excerpt.{key}")
    _finite_number(
        canonical.get("duration_seconds"),
        "canonical_excerpt.duration_seconds",
        minimum=0.1,
    )
    if canonical.get("metadata_removed") is not True:
        raise DynamicEvaluationError("canonical_excerpt.metadata_removed must be true")

    split_policy = _exact_keys(
        document.get("split_policy"), _SPLIT_POLICY_KEYS, "split_policy"
    )
    for key in _SPLIT_POLICY_KEYS:
        _nonempty_string(split_policy.get(key), f"split_policy.{key}")

    annotation = _exact_keys(
        document.get("annotation_contract"),
        _ANNOTATION_CONTRACT_KEYS,
        "annotation_contract",
    )
    for key in (
        "method",
        "annotation_date",
        "single_trackable",
        "no_face",
        "multiple_faces",
        "occluded_or_tiny",
        "uncertain",
        "motion",
        "gaze_truth",
        "limitations",
    ):
        _nonempty_string(annotation.get(key), f"annotation_contract.{key}")
    _positive_integer(annotation.get("curator_count"), "annotation_contract.curator_count")
    manual_interval = _finite_number(
        annotation.get("manual_sampling_interval_seconds"),
        "annotation_contract.manual_sampling_interval_seconds",
        minimum=0.001,
    )
    refinement_interval = _finite_number(
        annotation.get("boundary_refinement_interval_seconds"),
        "annotation_contract.boundary_refinement_interval_seconds",
        minimum=0.001,
    )
    if refinement_interval > manual_interval:
        raise DynamicEvaluationError(
            "annotation_contract boundary refinement cannot be coarser than manual sampling"
        )
    _finite_number(
        annotation.get("boundary_uncertainty_seconds"),
        "annotation_contract.boundary_uncertainty_seconds",
    )
    for key in ("frame_exhaustive", "independently_adjudicated", "accuracy_claim_allowed"):
        if annotation.get(key) is not False:
            raise DynamicEvaluationError(f"annotation_contract.{key} must be false")
    curator_qa = _exact_keys(
        annotation.get("pre_release_curator_qa"),
        _PRE_RELEASE_CURATOR_QA_KEYS,
        "annotation_contract.pre_release_curator_qa",
    )
    for key in ("completed_date", "status", "method"):
        _nonempty_string(
            curator_qa.get(key), f"annotation_contract.pre_release_curator_qa.{key}"
        )
    _positive_integer(
        curator_qa.get("canonical_excerpts_reviewed"),
        "annotation_contract.pre_release_curator_qa.canonical_excerpts_reviewed",
        maximum=MAX_CLIPS,
    )
    if curator_qa.get("model_or_evaluator_output_viewed") is not False:
        raise DynamicEvaluationError(
            "pre-release curator QA must be independent of model or evaluator output"
        )
    if curator_qa.get("results_generated_from_draft_annotations") != "invalidated":
        raise DynamicEvaluationError(
            "pre-release curator QA must invalidate results from draft annotations"
        )
    corrections = curator_qa.get("corrections")
    if (
        not isinstance(corrections, list)
        or not corrections
        or len(corrections) > MAX_CLIPS
        or any(not isinstance(item, str) or not item or len(item) > 1_000 for item in corrections)
    ):
        raise DynamicEvaluationError(
            "annotation_contract.pre_release_curator_qa.corrections is invalid"
        )


def _parse_power_policy(value: object) -> GroundTruthPowerPolicy:
    document = _exact_keys(value, _POWER_POLICY_KEYS, "power_policy")
    policy = GroundTruthPowerPolicy(
        minimum_trackable_frames=_positive_integer(
            document.get("minimum_trackable_frames"),
            "power_policy.minimum_trackable_frames",
            maximum=108_000,
        ),
        minimum_nontrackable_frames=_positive_integer(
            document.get("minimum_nontrackable_frames"),
            "power_policy.minimum_nontrackable_frames",
            maximum=108_000,
        ),
        minimum_multi_face_frames=_positive_integer(
            document.get("minimum_multi_face_frames"),
            "power_policy.minimum_multi_face_frames",
            maximum=108_000,
        ),
        minimum_trackable_seconds=_finite_number(
            document.get("minimum_trackable_seconds"),
            "power_policy.minimum_trackable_seconds",
            minimum=0.001,
        ),
        minimum_reacquisition_events=_positive_integer(
            document.get("minimum_reacquisition_events"),
            "power_policy.minimum_reacquisition_events",
            maximum=108_000,
        ),
        minimum_sampling_coverage_ratio=_finite_number(
            document.get("minimum_sampling_coverage_ratio"),
            "power_policy.minimum_sampling_coverage_ratio",
            minimum=0.000001,
            maximum=1.0,
        ),
    )
    try:
        policy.validate()
    except GroundTruthValidationError as error:
        raise DynamicEvaluationError(str(error)) from error
    return policy


def _parse_scoring_policy(
    value: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    document = _exact_keys(value, _SCORING_POLICY_KEYS, "scoring_policy")
    for key in (
        "scope",
        "aggregation",
        "uncertain_and_unlabeled_intervals",
        "underpowered_metric_behavior",
        "gaze_metric_behavior",
        "claim_boundary",
    ):
        _nonempty_string(document.get(key), f"scoring_policy.{key}")
    if document.get("thresholds_frozen_before_model_output_holdout") is not True:
        raise DynamicEvaluationError(
            "scoring_policy.thresholds_frozen_before_model_output_holdout must be true"
        )
    required = document.get("required_metrics")
    allowed_metrics = {metric for metric, _ in _THRESHOLD_RULES.values()}
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(item, str) for item in required)
        or len(required) != len(set(required))
        or any(item not in allowed_metrics for item in required)
    ):
        raise DynamicEvaluationError(
            "scoring_policy.required_metrics must be a unique non-empty metric list"
        )
    thresholds = _exact_keys(
        document.get("acceptance_thresholds"),
        _ACCEPTANCE_POLICY_KEYS,
        "scoring_policy.acceptance_thresholds",
    )
    parsed: dict[str, object] = {"required_metrics": list(required)}
    for key in _THRESHOLD_RULES:
        maximum = (
            1.0
            if key in {
                "trackable_coverage_min",
                "false_valid_rate_max",
                "multi_face_abstention_min",
                "reacquisition_success_rate_min",
            }
            else 10_000.0
        )
        parsed[key] = _finite_number(
            thresholds.get(key),
            f"scoring_policy.acceptance_thresholds.{key}",
            maximum=maximum,
        )
    return document, parsed


def load_dynamic_manifest(path: Path) -> DynamicManifest:
    """Load an exact-schema moving-speaker manifest with duplicate-key rejection."""

    raw = _load_bounded_json(path, maximum_bytes=MAX_MANIFEST_BYTES, label="Dynamic manifest")
    document = _exact_keys(raw, _MANIFEST_KEYS, "manifest")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise DynamicEvaluationError("Unsupported dynamic manifest schema")
    _validate_metadata(document)
    target_fps = _finite_number(
        document.get("target_fps"), "target_fps", minimum=15.0, maximum=60.0
    )
    model = _exact_keys(document.get("model"), _MODEL_KEYS, "model")
    model_filename = _safe_relative_path(
        model.get("filename"), "model.filename", base_filename=True
    )
    model_sha256 = _digest(model.get("sha256"), "model.sha256")
    power_policy = _parse_power_policy(document.get("power_policy"))
    scoring_policy, acceptance_policy = _parse_scoring_policy(
        document.get("scoring_policy")
    )

    raw_clips = document.get("clips")
    if not isinstance(raw_clips, list) or not 1 <= len(raw_clips) <= MAX_CLIPS:
        raise DynamicEvaluationError(f"clips must contain between 1 and {MAX_CLIPS} entries")
    reviewed_count = document["annotation_contract"]["pre_release_curator_qa"][
        "canonical_excerpts_reviewed"
    ]
    if reviewed_count != len(raw_clips):
        raise DynamicEvaluationError(
            "pre-release curator QA count must match the frozen clip count"
        )
    clips: list[DynamicClipSpec] = []
    seen_ids: set[str] = set()
    split_cases: list[dict[str, object]] = []
    for index, raw_clip in enumerate(raw_clips):
        label = f"clips[{index}]"
        clip = _exact_keys(raw_clip, _CLIP_KEYS, label)
        clip_id = _nonempty_string(clip.get("id"), f"{label}.id", maximum=128)
        if clip_id in seen_ids:
            raise DynamicEvaluationError(f"{label}.id must be unique")
        seen_ids.add(clip_id)
        filename = _safe_relative_path(
            clip.get("filename"), f"{label}.filename", base_filename=True
        )
        role = clip.get("dataset_role")
        if role not in ALLOWED_SPLITS:
            raise DynamicEvaluationError(f"{label}.dataset_role is unsupported")
        split_group = _nonempty_string(
            clip.get("split_group_id"), f"{label}.split_group_id", maximum=128
        )
        duration = _finite_number(
            clip.get("excerpt_duration_seconds"),
            f"{label}.excerpt_duration_seconds",
            minimum=0.1,
        )
        _finite_number(
            clip.get("excerpt_start_seconds"), f"{label}.excerpt_start_seconds"
        )
        annotation_file = _safe_relative_path(
            clip.get("annotation_file"), f"{label}.annotation_file"
        )
        annotation_parts = Path(annotation_file).parts
        if not annotation_parts or annotation_parts[0] != "annotations":
            raise DynamicEvaluationError(f"{label}.annotation_file must be under annotations/")
        manual_only = clip.get("manual_visual_inspection_only")
        output_examined = clip.get("model_output_examined_during_scouting")
        if not isinstance(manual_only, bool) or not isinstance(output_examined, bool):
            raise DynamicEvaluationError(f"{label} scouting flags must be booleans")
        if role == "model_output_holdout" and (not manual_only or output_examined):
            raise DynamicEvaluationError(
                f"{label} violates the untouched model-output holdout contract"
            )

        source = _exact_keys(clip.get("source"), _SOURCE_KEYS, f"{label}.source")
        for key in (
            "title",
            "page_url",
            "derivative_url",
            "derivative_filename",
            "original_sha1",
        ):
            _nonempty_string(source.get(key), f"{label}.source.{key}")
        _safe_relative_path(
            source.get("derivative_filename"),
            f"{label}.source.derivative_filename",
            base_filename=True,
        )
        _digest(source.get("derivative_sha256"), f"{label}.source.derivative_sha256")
        original_sha1 = str(source["original_sha1"])
        if len(original_sha1) != 40:
            raise DynamicEvaluationError(f"{label}.source.original_sha1 must be a SHA-1 digest")
        try:
            int(original_sha1, 16)
        except ValueError as error:
            raise DynamicEvaluationError(
                f"{label}.source.original_sha1 must be a SHA-1 digest"
            ) from error
        _finite_number(
            source.get("original_duration_seconds"),
            f"{label}.source.original_duration_seconds",
            minimum=duration,
            maximum=100_000.0,
        )
        _resolution(source.get("original_resolution"), f"{label}.source.original_resolution")
        _resolution(source.get("derivative_resolution"), f"{label}.source.derivative_resolution")

        attribution = _exact_keys(
            clip.get("attribution"), _ATTRIBUTION_KEYS, f"{label}.attribution"
        )
        for key in _ATTRIBUTION_KEYS - {"redistribution_constraints"}:
            _nonempty_string(attribution.get(key), f"{label}.attribution.{key}")
        constraints = attribution.get("redistribution_constraints")
        if (
            not isinstance(constraints, list)
            or not constraints
            or any(not isinstance(item, str) or not item for item in constraints)
        ):
            raise DynamicEvaluationError(
                f"{label}.attribution.redistribution_constraints must be a string list"
            )
        clips.append(DynamicClipSpec(
            clip_id=clip_id,
            filename=filename,
            media_sha256=_digest(clip.get("sha256"), f"{label}.sha256"),
            dataset_role=str(role),
            split_group_id=split_group,
            duration_seconds=duration,
            annotation_file=annotation_file,
            annotation_sha256=_digest(
                clip.get("annotation_sha256"), f"{label}.annotation_sha256"
            ),
            source_title=str(source["title"]),
            source_page_url=str(source["page_url"]),
            license_spdx=str(attribution["license_spdx"]),
        ))
        split_cases.append({
            "clip_id": clip_id,
            "dataset_role": str(role),
            "split_group_id": split_group,
        })
    try:
        validate_split_group_isolation(split_cases)
    except GroundTruthValidationError as error:
        raise DynamicEvaluationError(str(error)) from error
    if not all(any(item.dataset_role == role for item in clips) for role in ALLOWED_SPLITS):
        raise DynamicEvaluationError("Manifest must contain both development and holdout clips")
    return DynamicManifest(
        target_fps=target_fps,
        model_filename=model_filename,
        model_sha256=model_sha256,
        power_policy=power_policy,
        acceptance_policy=acceptance_policy,
        scoring_policy=scoring_policy,
        clips=tuple(clips),
        document=document,
    )


def _ensure_bounded_file(path: Path, *, maximum_bytes: int, label: str) -> None:
    try:
        if not path.is_file() or path.is_symlink():
            raise DynamicEvaluationError(f"{label} is missing or is a symbolic link")
        size = path.stat().st_size
    except OSError as error:
        raise DynamicEvaluationError(f"{label} could not be inspected") from error
    if size <= 0 or size > maximum_bytes:
        raise DynamicEvaluationError(f"{label} is empty or exceeds {maximum_bytes} bytes")


def _verify_digest(path: Path, expected: str, *, maximum_bytes: int, label: str) -> str:
    _ensure_bounded_file(path, maximum_bytes=maximum_bytes, label=label)
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise DynamicEvaluationError(f"{label} could not be hashed") from error
    if actual != expected:
        raise DynamicEvaluationError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _power_policy_document(policy: GroundTruthPowerPolicy) -> dict[str, object]:
    return {
        "minimum_trackable_frames": policy.minimum_trackable_frames,
        "minimum_nontrackable_frames": policy.minimum_nontrackable_frames,
        "minimum_multi_face_frames": policy.minimum_multi_face_frames,
        "minimum_trackable_seconds": float(policy.minimum_trackable_seconds),
        "minimum_reacquisition_events": policy.minimum_reacquisition_events,
        "minimum_sampling_coverage_ratio": float(policy.minimum_sampling_coverage_ratio),
    }


def _deterministic_run_projection(run: Mapping[str, object]) -> dict[str, object]:
    tracking = run.get("tracking_metrics")
    scored = run.get("ground_truth_metrics")
    if not isinstance(tracking, Mapping) or not isinstance(scored, Mapping):
        raise DynamicEvaluationError("Evaluation run did not produce aggregate metrics")
    return {
        "evaluation_duration_seconds": run.get("evaluation_duration_seconds"),
        "tracking_metrics": deterministic_metric_projection(tracking),
        "ground_truth_metrics": dict(scored),
    }


def _annotation_seconds(annotations: GroundTruthAnnotations, states: frozenset[str]) -> float:
    return sum(
        interval.end_seconds - interval.start_seconds
        for interval in annotations.intervals
        if interval.face_state in states
    )


def _evaluate_once(
    media_path: Path,
    *,
    model_path: Path,
    spec: DynamicClipSpec,
    annotations: GroundTruthAnnotations,
    target_fps: float,
    power_policy: GroundTruthPowerPolicy,
    sampler: Sampler,
) -> dict[str, object]:
    try:
        observations, evaluated_duration, runtime = sampler(
            media_path,
            model_path=model_path,
            start_seconds=0.0,
            duration_seconds=spec.duration_seconds,
            target_fps=target_fps,
            transform={"type": "identity"},
        )
        tracking_metrics = compute_tracking_metrics(
            observations,
            evaluation_duration_seconds=evaluated_duration,
            target_fps=target_fps,
        )
        tracking_metrics.update(runtime)
        ground_truth = score_tracking_ground_truth(
            observations,
            annotations,
            target_fps=target_fps,
            power_policy=power_policy,
        )
    except (OSError, TrackingEvaluationError, GroundTruthValidationError) as error:
        raise DynamicEvaluationError(f"{spec.clip_id} analysis failed: {error}") from error
    finally:
        # The report builder never receives or serializes raw landmark arrays.
        if "observations" in locals():
            del observations
    return {
        "evaluation_duration_seconds": round(float(evaluated_duration), 6),
        "tracking_metrics": tracking_metrics,
        "ground_truth_metrics": ground_truth,
    }


def _sampling_coverage(count: int, seconds: float, target_fps: float) -> float:
    denominator = seconds * target_fps
    return round(min(1.0, count / denominator), 6) if denominator > 0 else 0.0


def aggregate_ground_truth_metrics(
    cases: Sequence[Mapping[str, object]],
    *,
    annotations: Sequence[GroundTruthAnnotations],
    target_fps: float,
    power_policy: GroundTruthPowerPolicy,
) -> dict[str, object]:
    """Pool declared numerators and denominators within exactly one split."""

    if len(cases) != len(annotations) or not cases:
        raise DynamicEvaluationError("Aggregate inputs must be non-empty and aligned")
    metrics: list[Mapping[str, object]] = []
    for index, value in enumerate(cases):
        if not isinstance(value, Mapping):
            raise DynamicEvaluationError(
                f"cases[{index}] must contain ground-truth measurements"
            )
        metrics.append(value)

    def total(key: str) -> int:
        values = [item.get(key) for item in metrics]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise DynamicEvaluationError(f"Aggregate count {key} is malformed")
        return sum(int(value) for value in values)

    observation_count = total("observation_count")
    labeled_count = total("labeled_observation_count")
    trackable_count = total("trackable_observation_count")
    trackable_valid = total("trackable_valid_observation_count")
    nontrackable_count = total("nontrackable_observation_count")
    false_valid = total("false_valid_observation_count")
    multi_count = total("multi_face_observation_count")
    multi_abstained = total("multi_face_abstained_observation_count")
    dropout_count = total("dropout_run_count")
    reacquisition_count = total("reacquisition_event_count")
    reacquisition_success = total("reacquisition_success_count")
    trackable_seconds = sum(
        _annotation_seconds(item, frozenset({"single_trackable"}))
        for item in annotations
    )
    nontrackable_seconds = sum(
        _annotation_seconds(item, NONTRACKABLE_STATES) for item in annotations
    )
    multi_seconds = sum(
        _annotation_seconds(item, frozenset({"multiple_faces"}))
        for item in annotations
    )
    trackable_sampling = _sampling_coverage(trackable_count, trackable_seconds, target_fps)
    nontrackable_sampling = _sampling_coverage(
        nontrackable_count, nontrackable_seconds, target_fps
    )
    multi_sampling = _sampling_coverage(multi_count, multi_seconds, target_fps)
    reasons: dict[str, str] = {}

    if (
        trackable_count >= power_policy.minimum_trackable_frames
        and trackable_sampling >= power_policy.minimum_sampling_coverage_ratio
    ):
        trackable_coverage: float | None = round(trackable_valid / trackable_count, 6)
    else:
        trackable_coverage = None
        reasons["trackable_coverage"] = (
            f"needs {power_policy.minimum_trackable_frames} trackable frames and "
            f"{power_policy.minimum_sampling_coverage_ratio:.1%} sampling coverage; "
            f"found {trackable_count} and {trackable_sampling:.1%}"
        )
    if (
        nontrackable_count >= power_policy.minimum_nontrackable_frames
        and nontrackable_sampling >= power_policy.minimum_sampling_coverage_ratio
    ):
        false_valid_rate: float | None = round(false_valid / nontrackable_count, 6)
    else:
        false_valid_rate = None
        reasons["false_valid_rate"] = (
            f"needs {power_policy.minimum_nontrackable_frames} non-trackable frames and "
            f"{power_policy.minimum_sampling_coverage_ratio:.1%} sampling coverage; "
            f"found {nontrackable_count} and {nontrackable_sampling:.1%}"
        )
    if (
        multi_count >= power_policy.minimum_multi_face_frames
        and multi_sampling >= power_policy.minimum_sampling_coverage_ratio
    ):
        multi_face_abstention: float | None = round(multi_abstained / multi_count, 6)
    else:
        multi_face_abstention = None
        reasons["multi_face_abstention"] = (
            f"needs {power_policy.minimum_multi_face_frames} multiple-face frames and "
            f"{power_policy.minimum_sampling_coverage_ratio:.1%} sampling coverage; "
            f"found {multi_count} and {multi_sampling:.1%}"
        )
    if (
        trackable_seconds >= power_policy.minimum_trackable_seconds
        and trackable_count >= power_policy.minimum_trackable_frames
        and trackable_sampling >= power_policy.minimum_sampling_coverage_ratio
    ):
        dropouts_per_minute: float | None = round(
            dropout_count * 60.0 / trackable_seconds, 6
        )
    else:
        dropouts_per_minute = None
        reasons["dropouts_per_annotated_minute"] = (
            f"needs {power_policy.minimum_trackable_seconds:g} annotated trackable seconds "
            "with sufficient sampled frames"
        )

    successful_latencies: list[float] = []
    for item in metrics:
        events = item.get("reacquisition_events")
        if not isinstance(events, list):
            raise DynamicEvaluationError("reacquisition_events is malformed")
        for event in events:
            if not isinstance(event, Mapping):
                raise DynamicEvaluationError("reacquisition event is malformed")
            latency = event.get("latency_seconds")
            if latency is not None:
                if isinstance(latency, bool) or not isinstance(latency, (int, float)):
                    raise DynamicEvaluationError("reacquisition latency is malformed")
                successful_latencies.append(float(latency))
    if reacquisition_count >= power_policy.minimum_reacquisition_events:
        reacquisition_rate: float | None = round(
            reacquisition_success / reacquisition_count, 6
        )
        if reacquisition_success == reacquisition_count:
            mean_latency: float | None = round(
                sum(successful_latencies) / len(successful_latencies), 6
            )
            max_latency: float | None = round(max(successful_latencies), 6)
        else:
            mean_latency = None
            max_latency = None
            latency_reason = (
                f"{reacquisition_count - reacquisition_success} of {reacquisition_count} "
                "events never produced three consecutive valid frames"
            )
            reasons["mean_reacquisition_latency_seconds"] = latency_reason
            reasons["max_reacquisition_latency_seconds"] = latency_reason
    else:
        reacquisition_rate = None
        mean_latency = None
        max_latency = None
        reason = (
            f"needs {power_policy.minimum_reacquisition_events} reacquisition events; "
            f"found {reacquisition_count}"
        )
        reasons["reacquisition_success_rate"] = reason
        reasons["mean_reacquisition_latency_seconds"] = reason
        reasons["max_reacquisition_latency_seconds"] = reason
    reasons["gaze_accuracy"] = (
        "natural multi-camera presentation clips have no reliable camera-contact ground truth"
    )

    return {
        "observation_count": observation_count,
        "labeled_observation_count": labeled_count,
        "trackable_valid_observation_count": trackable_valid,
        "trackable_observation_count": trackable_count,
        "nontrackable_false_valid_observation_count": false_valid,
        "nontrackable_observation_count": nontrackable_count,
        "multi_face_abstained_observation_count": multi_abstained,
        "multi_face_observation_count": multi_count,
        "dropout_run_count": dropout_count,
        "annotated_trackable_seconds": round(trackable_seconds, 6),
        "annotated_nontrackable_seconds": round(nontrackable_seconds, 6),
        "annotated_multi_face_seconds": round(multi_seconds, 6),
        "trackable_sampling_coverage": trackable_sampling,
        "nontrackable_sampling_coverage": nontrackable_sampling,
        "multi_face_sampling_coverage": multi_sampling,
        "trackable_coverage": trackable_coverage,
        "false_valid_rate": false_valid_rate,
        "multi_face_abstention": multi_face_abstention,
        "dropouts_per_annotated_minute": dropouts_per_minute,
        "reacquisition_success_count": reacquisition_success,
        "reacquisition_event_count": reacquisition_count,
        "reacquisition_success_rate": reacquisition_rate,
        "mean_reacquisition_latency_seconds": mean_latency,
        "max_reacquisition_latency_seconds": max_latency,
        "gaze_accuracy": None,
        "underpowered_reasons": reasons,
    }


def apply_acceptance_policy(
    aggregate: Mapping[str, object], policy: Mapping[str, object]
) -> dict[str, object]:
    """Apply global split-level thresholds without treating missing data as a pass."""

    required = policy.get("required_metrics")
    if not isinstance(required, list):
        raise DynamicEvaluationError("Acceptance policy was not validated")
    checks: list[dict[str, object]] = []
    for threshold_key, (metric, comparison) in _THRESHOLD_RULES.items():
        threshold = policy.get(threshold_key)
        actual = aggregate.get(metric)
        required_metric = metric in required
        if actual is None:
            status = "not_evaluated"
            passed: bool | None = None
        elif isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise DynamicEvaluationError(f"Aggregate metric {metric} is malformed")
        else:
            passed = float(actual) >= float(threshold) if comparison == "min" else float(actual) <= float(threshold)
            status = "pass" if passed else "fail"
        checks.append({
            "metric": metric,
            "required": required_metric,
            "status": status,
            "passed": passed,
            "actual": actual,
            "comparison": ">=" if comparison == "min" else "<=",
            "threshold": threshold,
        })
    required_checks = [item for item in checks if item["required"]]
    underpowered_required = [
        str(item["metric"])
        for item in required_checks
        if item["status"] == "not_evaluated"
    ]
    failed_required = [
        str(item["metric"])
        for item in required_checks
        if item["status"] == "fail"
    ]
    status = "fail" if failed_required or underpowered_required else "pass"
    return {
        "status": status,
        "failed_required_metrics": failed_required,
        "underpowered_required_metrics": underpowered_required,
        "checks": checks,
    }


def _git_state(project_root: Path) -> dict[str, object]:
    def command(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "status_sha256": None if status is None else hashlib.sha256(status.encode()).hexdigest(),
    }


def build_dynamic_evaluation_report(
    *,
    manifest_path: Path,
    project_root: Path,
    split: str = "development",
    sampler: Sampler = sample_tracking_clip,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Evaluate exactly one split and return an aggregate-only evidence report."""

    if split not in ALLOWED_SPLITS:
        raise DynamicEvaluationError(
            "split must be development or model_output_holdout"
        )
    manifest_path = manifest_path.resolve()
    project_root = project_root.resolve()
    manifest = load_dynamic_manifest(manifest_path)
    manifest_digest = _verify_digest(
        manifest_path,
        sha256_file(manifest_path),
        maximum_bytes=MAX_MANIFEST_BYTES,
        label="Dynamic manifest",
    )
    model_path = project_root / "models" / manifest.model_filename
    model_digest = _verify_digest(
        model_path,
        manifest.model_sha256,
        maximum_bytes=MAX_MODEL_BYTES,
        label="MediaPipe model",
    )
    _verify_digest(
        project_root / "models" / YUNET_MODEL_FILENAME,
        YUNET_MODEL_SHA256,
        maximum_bytes=MAX_MODEL_BYTES,
        label="Anonymous proposal model",
    )
    selected = tuple(item for item in manifest.clips if item.dataset_role == split)
    if not selected:
        raise DynamicEvaluationError(f"No clips exist for split {split}")

    repeats = 2 if split == "development" else 1
    cases: list[dict[str, object]] = []
    annotation_objects: list[GroundTruthAnnotations] = []
    for spec in selected:
        media_path = project_root / "test_media" / spec.filename
        annotation_path = project_root / "test_media" / spec.annotation_file
        media_digest = _verify_digest(
            media_path,
            spec.media_sha256,
            maximum_bytes=MAX_MEDIA_BYTES,
            label=f"Media {spec.clip_id}",
        )
        annotation_digest = _verify_digest(
            annotation_path,
            spec.annotation_sha256,
            maximum_bytes=2 * 1024 * 1024,
            label=f"Annotation {spec.clip_id}",
        )
        try:
            annotations = load_ground_truth_annotations(annotation_path)
        except GroundTruthValidationError as error:
            raise DynamicEvaluationError(
                f"Annotation {spec.clip_id} is invalid: {error}"
            ) from error
        if (
            annotations.clip_id != spec.clip_id
            or annotations.split_group_id != spec.split_group_id
            or abs(annotations.duration_seconds - spec.duration_seconds) > 1e-9
        ):
            raise DynamicEvaluationError(
                f"Annotation {spec.clip_id} does not match its manifest contract"
            )
        annotation_objects.append(annotations)
        runs = [
            _evaluate_once(
                media_path,
                model_path=model_path,
                spec=spec,
                annotations=annotations,
                target_fps=manifest.target_fps,
                power_policy=manifest.power_policy,
                sampler=sampler,
            )
            for _ in range(repeats)
        ]
        first_projection = _deterministic_run_projection(runs[0])
        mismatches = [
            index + 1
            for index, run in enumerate(runs[1:], start=1)
            if _deterministic_run_projection(run) != first_projection
        ]
        if mismatches:
            raise DynamicEvaluationError(
                f"{spec.clip_id} deterministic metrics changed on repeat runs {mismatches}"
            )
        first = runs[0]
        cases.append({
            "clip_id": spec.clip_id,
            "dataset_role": spec.dataset_role,
            "split_group_id": spec.split_group_id,
            "filename": spec.filename,
            "media_sha256": media_digest,
            "annotation_file": spec.annotation_file,
            "annotation_sha256": annotation_digest,
            "source_title": spec.source_title,
            "source_page_url": spec.source_page_url,
            "license_spdx": spec.license_spdx,
            "evaluation_duration_seconds": first["evaluation_duration_seconds"],
            "tracking_metrics": first["tracking_metrics"],
            "ground_truth_metrics": first["ground_truth_metrics"],
            "repeatability": {
                "runs": repeats,
                "status": "pass" if repeats > 1 else "not_run_for_holdout",
                "exact_non_runtime_metrics": True if repeats > 1 else None,
            },
        })

    aggregate = aggregate_ground_truth_metrics(
        [case["ground_truth_metrics"] for case in cases],
        annotations=annotation_objects,
        target_fps=manifest.target_fps,
        power_policy=manifest.power_policy,
    )
    gate = apply_acceptance_policy(aggregate, manifest.acceptance_policy)
    configuration = {
        "target_fps": manifest.target_fps,
        "start_seconds": 0.0,
        "transform": {"type": "identity"},
        "sampler": "presentation_tracking_eval.sample_tracking_clip",
        "development_repeatability_runs": 2,
        "holdout_repeatability_runs": 1,
        "repeatability_projection": "all aggregate model outputs excluding runtime-only metrics",
        "face_landmarker": {
            "max_face_count": MAX_FACE_COUNT,
            "min_face_detection_confidence": FACE_DETECTION_CONFIDENCE,
            "min_face_presence_confidence": FACE_PRESENCE_CONFIDENCE,
            "min_tracking_confidence": FACE_TRACKING_CONFIDENCE,
            "landmark_smoothing_time_constant_ms": LANDMARK_SMOOTHING_TIME_CONSTANT_MS,
            "reacquisition_gap_ms": LANDMARK_REACQUISITION_GAP_MS,
            "temporal_hold_ms": LANDMARK_TEMPORAL_HOLD_MS,
            "multiscale_fallback_enabled": MULTISCALE_FALLBACK_ENABLED,
            "multiscale_tile_size": MULTISCALE_TILE_SIZE,
            "multiscale_tile_offsets": list(MULTISCALE_TILE_OFFSETS),
            "multiscale_tile_count": len(MULTISCALE_TILE_OFFSETS) ** 2,
            "proposal_model_filename": YUNET_MODEL_FILENAME,
            "proposal_model_sha256": YUNET_MODEL_SHA256,
            "proposal_score_threshold": YUNET_SCORE_THRESHOLD,
            "proposal_nms_threshold": YUNET_NMS_THRESHOLD,
            "proposal_top_k": YUNET_TOP_K,
            "proposal_roi_scales": list(YUNET_ROI_SCALES),
            "proposal_liveness_window_ms": YUNET_LIVENESS_WINDOW_MS,
            "proposal_liveness_hold_ms": YUNET_LIVENESS_HOLD_MS,
            "proposal_liveness_min_iou": YUNET_LIVENESS_MIN_IOU,
            "proposal_appearance_size": YUNET_APPEARANCE_SIZE,
            "proposal_appearance_lag_min_ms": YUNET_APPEARANCE_LAG_MIN_MS,
            "proposal_appearance_lag_max_ms": YUNET_APPEARANCE_LAG_MAX_MS,
            "proposal_appearance_motion_threshold": (
                YUNET_APPEARANCE_MOTION_THRESHOLD
            ),
            "proposal_liveness_range_threshold": YUNET_LIVENESS_RANGE_THRESHOLD,
            "proposal_provisional_track_ms": YUNET_PROVISIONAL_TRACK_MS,
            "proposal_provisional_rearm_gap_ms": (
                YUNET_PROVISIONAL_REARM_GAP_MS
            ),
            "proposal_provisional_luminance_change_ratio": (
                YUNET_PROVISIONAL_LUMINANCE_CHANGE_RATIO
            ),
            "small_face_liveness_area_threshold": SMALL_FACE_LIVENESS_AREA_THRESHOLD,
            "minimum_eye_to_lower_face_luminance_ratio": (
                MIN_EYE_TO_LOWER_FACE_LUMINANCE_RATIO
            ),
            "multiscale_deduplication_iou": MULTISCALE_DEDUPLICATION_IOU,
        },
    }
    power_document = _power_policy_document(manifest.power_policy)
    policy_document = {
        "power_policy": power_document,
        "scoring_policy": dict(manifest.scoring_policy),
    }
    module_path = Path(__file__).resolve()
    tracking_path = module_path.with_name("presentation_tracking_eval.py")
    scorer_path = module_path.with_name("presentation_tracking_ground_truth.py")
    landmarks_path = module_path.with_name("landmarks.py")
    timing_path = module_path.with_name("presentation_video_timing.py")
    source_hashes = {
        "presentation_dynamic_eval.py": sha256_file(module_path),
        "presentation_tracking_eval.py": sha256_file(tracking_path),
        "presentation_tracking_ground_truth.py": sha256_file(scorer_path),
        "landmarks.py": sha256_file(landmarks_path),
        "presentation_video_timing.py": sha256_file(timing_path),
    }
    script_path = module_path.parents[2] / "scripts" / "evaluate_dynamic_presentations.py"
    if not script_path.is_file():
        raise DynamicEvaluationError("Dynamic evaluation entrypoint is missing")
    source_hashes["scripts/evaluate_dynamic_presentations.py"] = sha256_file(script_path)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split": split,
        "split_isolation": "one report contains exactly one dataset_role",
        "case_count": len(cases),
        "manifest_sha256": manifest_digest,
        "model": {
            "filename": manifest.model_filename,
            "sha256": model_digest,
            "matches_manifest": True,
        },
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "policy": policy_document,
        "policy_sha256": _canonical_sha256(policy_document),
        "source_sha256": source_hashes,
        "git": _git_state(project_root),
        "evaluation_only": True,
        "used_for_training": False,
        "identity_recognition": False,
        "face_embeddings_created": False,
        "landmarks_persisted": False,
        "gaze_accuracy_evaluated": False,
        "annotation_evidence": "single-curator coarse observed-state intervals; not independently adjudicated accuracy",
        "aggregate": aggregate,
        "release_gate": gate,
        "cases": cases,
    }


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Durably replace a report without exposing a partial JSON document."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
