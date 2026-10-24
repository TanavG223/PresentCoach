#!/usr/bin/env python3
"""Run the real PresentCoach upload API against the frozen dynamic corpus.

The default invocation evaluates only the development split.  The
``model_output_holdout`` split is deliberately opt-in so routine development
runs cannot silently inspect held-out PresentCoach output.

This evaluator exercises the production Flask route, MediaPipe pipeline,
whisper.cpp adapter, Ollama adapter, encrypted store, signed byte-range
playback, and bootstrap reload.  It stores only bounded aggregate evidence;
transcripts, landmarks, uploaded videos, keys, and access capabilities never
enter the JSON report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile
import time
from typing import Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

from presentcoach.presentation_ai import OllamaPresentationLLM
from presentcoach.presentation_audio import WhisperCppTranscriber
from presentcoach.presentation_core import (
    FEEDBACK_METRIC_QUALITY_FLAGS,
    FEEDBACK_SCHEMA,
    PROHIBITED_FEEDBACK_TERMS,
)
from presentcoach.presentation_server import create_app
from presentcoach.presentation_store import MEDIA_MAGIC, PresentationStore
from presentcoach.presentation_tracking_ground_truth import (
    GroundTruthPowerPolicy,
    GroundTruthValidationError,
    load_ground_truth_annotations,
    validate_split_group_isolation,
)
from presentcoach.presentation_video import LocalVideoAnalyzer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "test_media" / "dynamic_presentation_manifest.json"
DEFAULT_DEVELOPMENT_OUTPUT = ROOT / "reports" / "presentcoach_e2e_development.json"
DEFAULT_HOLDOUT_OUTPUT = ROOT / "reports" / "presentcoach_e2e_holdout.json"
SCHEMA_VERSION = "presentcoach.api-e2e.v1"
DATASET_ROLES = ("development", "model_output_holdout")
QUALITY_FLAGS = (
    "face_detected",
    "eye_contact",
    "head_stability",
    "expression_variety",
    "audio_clear",
)
QUALITY_VALUES = frozenset({"good", "bad"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?\Z"
)
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_CASES = 32
MIN_ANALYZED_FPS = 14.5
MAX_ANALYZED_FPS = 15.5
DURATION_TOLERANCE_SECONDS = 0.35
RANGE_BYTES = 64


class E2EContractError(ValueError):
    """Raised when frozen inputs or API evidence violate the evaluator contract."""


@dataclass(frozen=True)
class DynamicClip:
    clip_id: str
    filename: str
    sha256: str
    dataset_role: str
    split_group_id: str
    excerpt_start_seconds: float
    excerpt_duration_seconds: float
    annotation_file: str
    annotation_sha256: str
    source_page_url: str
    source_title: str
    license_spdx: str


@dataclass(frozen=True)
class DynamicManifest:
    path: Path
    sha256: str
    target_fps: float
    model_filename: str
    model_sha256: str
    canonical_duration_seconds: float
    power_policy: GroundTruthPowerPolicy
    clips: tuple[DynamicClip, ...]


@dataclass
class MemoryKeyStore:
    """Ephemeral KeyStore implementation used only by this evaluation run."""

    values: dict[str, bytes]

    def __init__(self) -> None:
        self.values = {}

    def save(self, profile: str, key: bytes) -> None:
        self.values[profile] = bytes(key)

    def load(self, profile: str) -> bytes:
        try:
            return self.values[profile]
        except KeyError as error:
            raise E2EContractError("The ephemeral profile key is missing") from error

    def delete(self, profile: str) -> None:
        self.values.pop(profile, None)


class _InactiveRecorder:
    """The upload-only evaluator never opens a camera or microphone."""

    @staticmethod
    def is_active() -> bool:
        return False


class ObservedOllama:
    """Hash the exact prompt contract while delegating to the real Ollama adapter."""

    def __init__(self, delegate: OllamaPresentationLLM) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    def status(self) -> dict[str, object]:
        return self.delegate.status()

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, object]
    ) -> dict[str, object]:
        call: dict[str, object] = {
            "system_sha256": _sha256_bytes(system.encode("utf-8")),
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "schema_sha256": _canonical_json_sha256(schema),
        }
        try:
            response = self.delegate.complete_json(
                system=system, prompt=prompt, schema=schema
            )
        except Exception as error:
            call["completed"] = False
            call["error_type"] = type(error).__name__
            self.calls.append(call)
            raise
        call["completed"] = True
        call["response_sha256"] = _canonical_json_sha256(response)
        self.calls.append(call)
        return response


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise E2EContractError(f"JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path, *, maximum_bytes: int) -> tuple[dict[str, object], bytes]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise E2EContractError(f"Required JSON file is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise E2EContractError(f"Required JSON must be a regular file: {path}")
    if not 1 <= metadata.st_size <= maximum_bytes:
        raise E2EContractError(f"Required JSON size is outside its bound: {path}")
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_duplicate_rejecting_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E2EContractError(f"Required JSON is unreadable: {path}") from error
    if not isinstance(document, dict):
        raise E2EContractError(f"Required JSON root must be an object: {path}")
    return document, raw


def _exact_keys(
    document: Mapping[str, object],
    *,
    required: set[str],
    label: str,
) -> None:
    missing = required - set(document)
    extra = set(document) - required
    if missing or extra:
        raise E2EContractError(
            f"{label} fields do not match the frozen schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise E2EContractError(f"{label} is invalid")
    return value


def _bounded_text(value: object, label: str, *, maximum: int = 4_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise E2EContractError(f"{label} must be non-empty bounded text")
    return value


def _finite(
    value: object,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float = 10_000.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E2EContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise E2EContractError(f"{label} is outside its numeric bound")
    return number


def _https_url(value: object, label: str) -> str:
    text = _bounded_text(value, label)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise E2EContractError(f"{label} must be an HTTPS URL")
    return text


def _relative_file(value: object, label: str, *, suffix: str) -> str:
    text = _bounded_text(value, label, maximum=240)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix != suffix:
        raise E2EContractError(f"{label} is not a safe relative {suffix} path")
    return text


def _resolution(value: object, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or not all(1 <= item <= 7680 for item in value)
    ):
        raise E2EContractError(f"{label} must contain two bounded integers")
    return int(value[0]), int(value[1])


def _parse_clip(raw: object, index: int) -> DynamicClip:
    label = f"clips[{index}]"
    if not isinstance(raw, dict):
        raise E2EContractError(f"{label} must be an object")
    _exact_keys(
        raw,
        required={
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
        },
        label=label,
    )
    clip_id = _identifier(raw["id"], f"{label}.id")
    filename = _relative_file(raw["filename"], f"{label}.filename", suffix=".mp4")
    if Path(filename).parent != Path("."):
        raise E2EContractError(f"{label}.filename must live directly in test_media")
    digest = str(raw["sha256"])
    annotation_digest = str(raw["annotation_sha256"])
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise E2EContractError(f"{label}.sha256 must be lowercase SHA-256")
    if SHA256_PATTERN.fullmatch(annotation_digest) is None:
        raise E2EContractError(
            f"{label}.annotation_sha256 must be lowercase SHA-256"
        )
    role = str(raw["dataset_role"])
    if role not in DATASET_ROLES:
        raise E2EContractError(f"{label}.dataset_role is unsupported")
    split_group_id = _identifier(raw["split_group_id"], f"{label}.split_group_id")
    start = _finite(raw["excerpt_start_seconds"], f"{label}.excerpt_start_seconds")
    duration = _finite(
        raw["excerpt_duration_seconds"],
        f"{label}.excerpt_duration_seconds",
        minimum=30.0,
        maximum=300.0,
    )
    annotation_file = _relative_file(
        raw["annotation_file"], f"{label}.annotation_file", suffix=".json"
    )
    if Path(annotation_file).parts[:1] != ("annotations",):
        raise E2EContractError(f"{label}.annotation_file must be under annotations/")
    manual_only = raw["manual_visual_inspection_only"]
    output_examined = raw["model_output_examined_during_scouting"]
    if not isinstance(manual_only, bool) or not isinstance(output_examined, bool):
        raise E2EContractError(f"{label} scouting flags must be booleans")
    if role == "model_output_holdout" and (
        manual_only is not True or output_examined is not False
    ):
        raise E2EContractError(
            f"{label} violates the sealed model-output holdout contract"
        )

    source = raw["source"]
    if not isinstance(source, dict):
        raise E2EContractError(f"{label}.source must be an object")
    _exact_keys(
        source,
        required={
            "title",
            "page_url",
            "derivative_url",
            "derivative_filename",
            "derivative_sha256",
            "original_sha1",
            "original_duration_seconds",
            "original_resolution",
            "derivative_resolution",
        },
        label=f"{label}.source",
    )
    source_title = _bounded_text(source["title"], f"{label}.source.title")
    page_url = _https_url(source["page_url"], f"{label}.source.page_url")
    _https_url(source["derivative_url"], f"{label}.source.derivative_url")
    _relative_file(
        source["derivative_filename"],
        f"{label}.source.derivative_filename",
        suffix=".webm",
    )
    if SHA256_PATTERN.fullmatch(str(source["derivative_sha256"])) is None:
        raise E2EContractError(f"{label}.source.derivative_sha256 is invalid")
    if SHA1_PATTERN.fullmatch(str(source["original_sha1"])) is None:
        raise E2EContractError(f"{label}.source.original_sha1 is invalid")
    original_duration = _finite(
        source["original_duration_seconds"],
        f"{label}.source.original_duration_seconds",
        minimum=duration,
        maximum=8 * 60 * 60,
    )
    if start + duration > original_duration + 0.1:
        raise E2EContractError(f"{label} excerpt exceeds its source duration")
    _resolution(source["original_resolution"], f"{label}.source.original_resolution")
    _resolution(
        source["derivative_resolution"], f"{label}.source.derivative_resolution"
    )

    attribution = raw["attribution"]
    if not isinstance(attribution, dict):
        raise E2EContractError(f"{label}.attribution must be an object")
    _exact_keys(
        attribution,
        required={
            "creator",
            "license_name",
            "license_spdx",
            "license_url",
            "credit_line",
            "change_notice",
            "redistribution_constraints",
        },
        label=f"{label}.attribution",
    )
    for key in ("creator", "license_name", "license_spdx", "credit_line", "change_notice"):
        _bounded_text(attribution[key], f"{label}.attribution.{key}")
    _https_url(attribution["license_url"], f"{label}.attribution.license_url")
    constraints = attribution["redistribution_constraints"]
    if (
        not isinstance(constraints, list)
        or not constraints
        or len(constraints) > 16
        or any(not isinstance(item, str) or not item or len(item) > 500 for item in constraints)
    ):
        raise E2EContractError(
            f"{label}.attribution.redistribution_constraints is invalid"
        )
    return DynamicClip(
        clip_id=clip_id,
        filename=filename,
        sha256=digest,
        dataset_role=role,
        split_group_id=split_group_id,
        excerpt_start_seconds=start,
        excerpt_duration_seconds=duration,
        annotation_file=annotation_file,
        annotation_sha256=annotation_digest,
        source_page_url=page_url,
        source_title=source_title,
        license_spdx=str(attribution["license_spdx"]),
    )


def load_dynamic_manifest(path: Path) -> DynamicManifest:
    """Load and fully validate the frozen dynamic-presentation manifest."""

    resolved = path.resolve(strict=True)
    document, raw = _strict_json(resolved, maximum_bytes=MAX_MANIFEST_BYTES)
    _exact_keys(
        document,
        required={
            "schema_version",
            "description",
            "target_fps",
            "model",
            "evaluation_only",
            "never_training_data",
            "identity_recognition",
            "face_embeddings_allowed",
            "landmarks_persisted",
            "gaze_ground_truth_available",
            "gaze_annotation_policy",
            "binary_distribution",
            "canonical_excerpt",
            "split_policy",
            "annotation_contract",
            "power_policy",
            "scoring_policy",
            "clips",
        },
        label="manifest",
    )
    if document["schema_version"] != 1:
        raise E2EContractError("Unsupported dynamic manifest schema")
    expected_safety = {
        "evaluation_only": True,
        "never_training_data": True,
        "identity_recognition": False,
        "face_embeddings_allowed": False,
        "landmarks_persisted": False,
        "gaze_ground_truth_available": False,
    }
    for key, expected in expected_safety.items():
        if document[key] is not expected:
            raise E2EContractError(f"manifest.{key} violates the evaluation boundary")
    if document["gaze_annotation_policy"] != "unknown_for_every_manual_interval":
        raise E2EContractError("The dynamic corpus must not claim natural-video gaze truth")
    _bounded_text(document["description"], "manifest.description")
    _bounded_text(document["binary_distribution"], "manifest.binary_distribution")

    target_fps = _finite(
        document["target_fps"], "manifest.target_fps", minimum=1, maximum=60
    )
    if abs(target_fps - 15.0) > 1e-9:
        raise E2EContractError("The API evaluator requires the frozen 15 FPS target")
    model = document["model"]
    if not isinstance(model, dict):
        raise E2EContractError("manifest.model must be an object")
    _exact_keys(model, required={"filename", "sha256"}, label="manifest.model")
    model_filename = _relative_file(
        model["filename"], "manifest.model.filename", suffix=".task"
    )
    model_sha256 = str(model["sha256"])
    if SHA256_PATTERN.fullmatch(model_sha256) is None:
        raise E2EContractError("manifest.model.sha256 is invalid")

    canonical = document["canonical_excerpt"]
    if not isinstance(canonical, dict):
        raise E2EContractError("manifest.canonical_excerpt must be an object")
    _exact_keys(
        canonical,
        required={
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
        },
        label="manifest.canonical_excerpt",
    )
    if (
        canonical["container"] != "MP4"
        or canonical["pixel_format"] != "yuv420p"
        or canonical["metadata_removed"] is not True
    ):
        raise E2EContractError("Canonical excerpt encoding contract changed")
    canonical_duration = _finite(
        canonical["duration_seconds"],
        "manifest.canonical_excerpt.duration_seconds",
        minimum=30,
        maximum=300,
    )
    for field in ("video_codec", "audio_codec", "reference_encoder", "reproducibility_note"):
        _bounded_text(canonical[field], f"manifest.canonical_excerpt.{field}")
    for field, maximum in (
        ("width", 7680),
        ("height", 7680),
        ("frame_rate", 120),
        ("audio_sample_rate_hz", 384000),
        ("audio_channels", 16),
    ):
        _finite(canonical[field], f"manifest.canonical_excerpt.{field}", minimum=1, maximum=maximum)

    for section in ("split_policy", "annotation_contract", "scoring_policy"):
        if not isinstance(document[section], dict) or not document[section]:
            raise E2EContractError(f"manifest.{section} must be a non-empty object")
    scoring = document["scoring_policy"]
    if scoring.get("thresholds_frozen_before_model_output_holdout") is not True:
        raise E2EContractError("Holdout thresholds were not frozen before output")

    power = document["power_policy"]
    if not isinstance(power, dict):
        raise E2EContractError("manifest.power_policy must be an object")
    _exact_keys(
        power,
        required={
            "minimum_trackable_frames",
            "minimum_nontrackable_frames",
            "minimum_multi_face_frames",
            "minimum_trackable_seconds",
            "minimum_reacquisition_events",
            "minimum_sampling_coverage_ratio",
        },
        label="manifest.power_policy",
    )
    try:
        power_policy = GroundTruthPowerPolicy(**power)
        power_policy.validate()
    except (TypeError, GroundTruthValidationError) as error:
        raise E2EContractError("manifest.power_policy is invalid") from error

    raw_clips = document["clips"]
    if not isinstance(raw_clips, list) or not 1 <= len(raw_clips) <= MAX_CASES:
        raise E2EContractError(f"manifest.clips must contain 1-{MAX_CASES} cases")
    clips = tuple(_parse_clip(raw_clip, index) for index, raw_clip in enumerate(raw_clips))
    if len({clip.clip_id for clip in clips}) != len(clips):
        raise E2EContractError("Dynamic clip ids must be unique")
    if len({clip.filename for clip in clips}) != len(clips):
        raise E2EContractError("Dynamic clip filenames must be unique")
    if not all(any(clip.dataset_role == role for clip in clips) for role in DATASET_ROLES):
        raise E2EContractError("Both development and model-output holdout clips are required")
    try:
        validate_split_group_isolation([
            {
                "clip_id": clip.clip_id,
                "dataset_role": clip.dataset_role,
                "split_group_id": clip.split_group_id,
            }
            for clip in clips
        ])
    except GroundTruthValidationError as error:
        raise E2EContractError("Dynamic split groups leak across roles") from error

    media_dir = resolved.parent
    for clip in clips:
        if abs(clip.excerpt_duration_seconds - canonical_duration) > 1e-9:
            raise E2EContractError(
                f"{clip.clip_id} duration differs from the canonical excerpt"
            )
        annotation_path = (media_dir / clip.annotation_file).resolve()
        if media_dir not in annotation_path.parents:
            raise E2EContractError(f"{clip.clip_id} annotation escapes test_media")
        if sha256_file(annotation_path) != clip.annotation_sha256:
            raise E2EContractError(f"{clip.clip_id} annotation hash mismatch")
        try:
            annotation = load_ground_truth_annotations(annotation_path)
        except GroundTruthValidationError as error:
            raise E2EContractError(f"{clip.clip_id} annotation is invalid") from error
        if (
            annotation.clip_id != clip.clip_id
            or annotation.split_group_id != clip.split_group_id
            or abs(annotation.duration_seconds - canonical_duration) > 1e-9
        ):
            raise E2EContractError(
                f"{clip.clip_id} annotation identity or duration mismatch"
            )
    return DynamicManifest(
        path=resolved,
        sha256=_sha256_bytes(raw),
        target_fps=target_fps,
        model_filename=model_filename,
        model_sha256=model_sha256,
        canonical_duration_seconds=canonical_duration,
        power_policy=power_policy,
        clips=clips,
    )


def selected_clips(
    manifest: DynamicManifest, dataset_role: str
) -> tuple[DynamicClip, ...]:
    if dataset_role not in DATASET_ROLES:
        raise E2EContractError("Unsupported dataset split")
    selected = tuple(
        clip for clip in manifest.clips if clip.dataset_role == dataset_role
    )
    if not selected:
        raise E2EContractError(f"No cases are declared for {dataset_role}")
    return selected


def default_output_for_role(dataset_role: str) -> Path:
    """Return a split-specific default so held-out evidence cannot overwrite dev."""

    if dataset_role == "development":
        return DEFAULT_DEVELOPMENT_OUTPUT
    if dataset_role == "model_output_holdout":
        return DEFAULT_HOLDOUT_OUTPUT
    raise E2EContractError("Unsupported dataset split")


def _check(
    label: str, passed: bool, actual: object, expected: str
) -> dict[str, object]:
    return {
        "label": label[:160],
        "passed": bool(passed),
        "actual": _bounded_report_value(actual),
        "expected": expected[:300],
    }


def _bounded_report_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, Mapping):
        return {
            str(key)[:80]: _bounded_report_value(item)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_report_value(item) for item in list(value)[:24]]
    return str(value)[:500]


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _timestamps_are_monotonic(
    items: object,
    *,
    start_key: str,
    end_key: str | None,
    duration: float,
    strictly: bool,
) -> bool:
    if not isinstance(items, list) or not items:
        return False
    previous = -math.inf
    for item in items:
        if not isinstance(item, Mapping):
            return False
        start = _numeric(item.get(start_key))
        end = start if end_key is None else _numeric(item.get(end_key))
        if start is None or end is None or not 0 <= start <= end <= duration + 0.11:
            return False
        if start < previous or (strictly and start <= previous):
            return False
        previous = start
    return True


_FACT_UNITS = {
    "eye_contact_percent": "%",
    "head_rotation_std_degrees": "degrees",
    "head_position_std_percent": "% frame",
    "expression_variety_index": "index",
    "face_presence_percent": "%",
    "overall_words_per_minute": "WPM",
    "filler_count": "fillers",
    "filler_rate_per_minute": "fillers/min",
    "strict_filler_count": "um/uh fillers",
    "strict_filler_rate_per_minute": "um/uh per min",
    "pauses_over_2_seconds": "pauses",
    "pauses_over_3_seconds": "pauses",
    "long_pause_rate_per_minute": "pauses/min",
    "longest_pause_seconds": "seconds",
}


def _allowed_feedback_facts(metrics: Mapping[str, object]) -> list[tuple[str, float, str, float]]:
    facts: list[tuple[str, float, str, float]] = []
    duration = _numeric(metrics.get("duration_seconds")) or 0.0
    aggregate = metrics.get("aggregate")
    if isinstance(aggregate, Mapping):
        for metric, unit in _FACT_UNITS.items():
            value = _numeric(aggregate.get(metric))
            if value is not None:
                facts.append((metric, value, unit, duration))
    timeline = metrics.get("timeline")
    if not isinstance(timeline, Mapping):
        return facts
    gaze = timeline.get("longest_gaze_break")
    if isinstance(gaze, Mapping):
        value = _numeric(gaze.get("duration_seconds"))
        timestamp = _numeric(gaze.get("start_seconds"))
        if value is not None and timestamp is not None:
            facts.append(("longest_gaze_break_seconds", value, "seconds", timestamp))
    pace = timeline.get("pace_windows")
    if isinstance(pace, list):
        for item in pace:
            if isinstance(item, Mapping):
                value = _numeric(item.get("words_per_minute"))
                timestamp = _numeric(item.get("start_seconds"))
                if value is not None and timestamp is not None:
                    facts.append(("window_words_per_minute", value, "WPM", timestamp))
    clusters = timeline.get("filler_clusters")
    if isinstance(clusters, list):
        for item in clusters:
            if isinstance(item, Mapping):
                value = _numeric(item.get("count"))
                timestamp = _numeric(item.get("start_seconds"))
                if value is not None and timestamp is not None:
                    facts.append((
                        "strict_filler_cluster_count",
                        value,
                        "um/uh fillers",
                        timestamp,
                    ))
    pause = timeline.get("longest_pause")
    if isinstance(pause, Mapping):
        value = _numeric(pause.get("duration_seconds"))
        timestamp = _numeric(pause.get("start_seconds"))
        if value is not None and timestamp is not None:
            facts.append(("longest_pause_seconds", value, "seconds", timestamp))
    return facts


def feedback_is_metric_grounded(
    feedback: object, metrics: Mapping[str, object]
) -> tuple[bool, str]:
    if not isinstance(feedback, Mapping):
        return False, "feedback is not an object"
    if feedback.get("status") != "ready":
        return False, f"feedback status is {feedback.get('status')!r}"
    if not str(feedback.get("source", "")).startswith("local_llm_verified_"):
        return False, "feedback source is not verified local LLM output"
    quality = metrics.get("quality_flags")
    if not isinstance(quality, Mapping):
        return False, "quality flags are absent"
    facts = _allowed_feedback_facts(metrics)
    claims: list[Mapping[str, object]] = []
    for role, maximum in (("strengths", 2), ("improvements", 3)):
        values = feedback.get(role)
        if not isinstance(values, list) or len(values) > maximum:
            return False, f"{role} violates its claim bound"
        if any(not isinstance(item, Mapping) for item in values):
            return False, f"{role} contains a malformed claim"
        claims.extend(values)  # type: ignore[arg-type]
    if not claims:
        return False, "the local LLM returned no verified claims"
    seen: set[str] = set()
    for claim in claims:
        metric = str(claim.get("metric", ""))
        value = _numeric(claim.get("value"))
        unit = str(claim.get("unit", ""))
        timestamp = _numeric(claim.get("timestamp_seconds"))
        text = str(claim.get("text", ""))
        if metric in seen:
            return False, f"duplicate feedback metric {metric}"
        seen.add(metric)
        required_quality = FEEDBACK_METRIC_QUALITY_FLAGS.get(metric)
        if required_quality is None or quality.get(required_quality) != "good":
            return False, f"claim uses quality-insufficient metric {metric}"
        if not text or any(term in text.casefold() for term in PROHIBITED_FEEDBACK_TERMS):
            return False, f"claim text for {metric} is prohibited or empty"
        if value is None or timestamp is None:
            return False, f"claim {metric} lacks a finite number or timestamp"
        matched = any(
            metric == fact_metric
            and unit == fact_unit
            and abs(value - fact_value) <= 0.11
            and abs(timestamp - fact_timestamp) <= 0.11
            for fact_metric, fact_value, fact_unit, fact_timestamp in facts
        )
        if not matched:
            return False, f"claim {metric} does not match stored metrics"
        if f"{value:g}" not in text or f"{timestamp:g}" not in text:
            return False, f"claim {metric} text omits its value or timestamp"
    bad = {key for key, value in quality.items() if value != "good"}
    insufficient = feedback.get("insufficient_data")
    if not isinstance(insufficient, list) or set(map(str, insufficient)) != bad:
        return False, "feedback misstated insufficient quality flags"
    return True, f"{len(claims)} claims matched exact stored facts"


def coaching_cues_are_bounded(
    cues: object, *, session_id: str, duration: float
) -> tuple[bool, str]:
    if not isinstance(cues, Mapping):
        return False, "coaching cues are not an object"
    if cues.get("schema_version") != "presentcoach.review-cues.v1":
        return False, "coaching cue schema changed"
    if cues.get("session_id") != session_id:
        return False, "coaching cues reference a different session"
    items = cues.get("cues")
    if not isinstance(items, list) or len(items) > 10_000:
        return False, "coaching cue list is malformed or unbounded"
    previous: tuple[float, str] | None = None
    for item in items:
        if not isinstance(item, Mapping):
            return False, "coaching cue is not an object"
        start = _numeric(item.get("start_seconds"))
        end = _numeric(item.get("end_seconds"))
        seek = _numeric(item.get("seek_seconds"))
        cue_id = str(item.get("cue_id", ""))
        if (
            start is None
            or end is None
            or seek is None
            or not 0 <= start <= end <= duration + 0.11
            or not 0 <= seek <= duration + 0.11
        ):
            return False, f"cue {cue_id!r} has out-of-bounds time"
        order = (start, cue_id)
        if previous is not None and order < previous:
            return False, "coaching cues are not time ordered"
        previous = order
    limitations = cues.get("limitations")
    expected_limitations = {
        "descriptive_measured_behavior_only": True,
        "infers_confidence": False,
        "infers_reading": False,
    }
    if limitations != expected_limitations:
        return False, "coaching cue inference limitations changed"
    return True, f"{len(items)} cues stayed inside 0-{duration:g} seconds"


def validate_api_document(
    document: object,
    *,
    expected_duration: float,
    observed_llm_calls: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate one successful API response without trusting its prose."""

    if not isinstance(document, Mapping):
        return [_check("API JSON object", False, type(document).__name__, "object")]
    session = document.get("session")
    metrics = document.get("metrics")
    if not isinstance(session, Mapping) or not isinstance(metrics, Mapping):
        return [_check(
            "Session and metrics objects",
            False,
            {"session": type(session).__name__, "metrics": type(metrics).__name__},
            "both objects",
        )]
    duration = _numeric(session.get("duration_seconds"))
    session_id = str(session.get("session_id", ""))
    aggregate = metrics.get("aggregate")
    quality = metrics.get("quality_flags")
    transcript = session.get("transcript")
    vision = session.get("vision_metrics")
    checks: list[dict[str, object]] = []
    checks.append(_check(
        "Imported session kind",
        session.get("session_kind") == "imported",
        session.get("session_kind"),
        "imported",
    ))
    duration_ok = duration is not None and abs(duration - expected_duration) <= DURATION_TOLERANCE_SECONDS
    checks.append(_check(
        "Canonical excerpt duration",
        duration_ok,
        duration,
        f"{expected_duration:g} ± {DURATION_TOLERANCE_SECONDS:g} seconds",
    ))
    if duration is None:
        return checks
    metric_duration = _numeric(metrics.get("duration_seconds"))
    checks.append(_check(
        "Metric/session duration identity",
        metric_duration is not None and abs(metric_duration - duration) <= 0.01,
        metric_duration,
        f"{duration:g} seconds",
    ))
    cadence = (
        _numeric(aggregate.get("analyzed_vision_fps"))
        if isinstance(aggregate, Mapping)
        else None
    )
    checks.append(_check(
        "Analyzed media cadence",
        cadence is not None and MIN_ANALYZED_FPS <= cadence <= MAX_ANALYZED_FPS,
        cadence,
        f"{MIN_ANALYZED_FPS:g}-{MAX_ANALYZED_FPS:g} sampled media FPS",
    ))
    vision_ok = _timestamps_are_monotonic(
        vision,
        start_key="timestamp_seconds",
        end_key=None,
        duration=duration,
        strictly=True,
    )
    checks.append(_check(
        "Vision timestamps monotonic and bounded",
        vision_ok,
        len(vision) if isinstance(vision, list) else type(vision).__name__,
        ">= 1 strictly increasing second buckets",
    ))
    transcript_ok = _timestamps_are_monotonic(
        transcript,
        start_key="start_seconds",
        end_key="end_seconds",
        duration=duration,
        strictly=False,
    )
    checks.append(_check(
        "Whisper timestamps monotonic and bounded",
        transcript_ok,
        len(transcript) if isinstance(transcript, list) else type(transcript).__name__,
        ">= 1 ordered words within the session",
    ))
    expected_quality = set(QUALITY_FLAGS)
    quality_ok = (
        isinstance(quality, Mapping)
        and set(quality) == expected_quality
        and all(value in QUALITY_VALUES for value in quality.values())
        and session.get("quality_flags") == quality
    )
    checks.append(_check(
        "Five derived quality flags",
        quality_ok,
        dict(quality) if isinstance(quality, Mapping) else type(quality).__name__,
        "exact face/contact/stability/expression/audio good|bad flags",
    ))
    detected_buckets = 0
    vision_decision_count = 0
    if isinstance(vision, list):
        detected_buckets = sum(
            item.get("face_detected") is True
            for item in vision
            if isinstance(item, Mapping)
        )
        vision_decision_count = sum(
            isinstance(item, Mapping)
            and isinstance(item.get("face_detected"), bool)
            for item in vision
        )
    checks.append(_check(
        "Anonymous face tracking path executed",
        isinstance(vision, list)
        and bool(vision)
        and vision_decision_count == len(vision),
        {
            "bucket_decisions": vision_decision_count,
            "detected_buckets": detected_buckets,
        },
        "every one-second bucket has an explicit face/no-face decision; zero detections is valid abstention",
    ))
    grounded, grounding_detail = feedback_is_metric_grounded(
        document.get("feedback"), metrics
    )
    checks.append(_check(
        "Metric-grounded local-LLM feedback",
        grounded,
        grounding_detail,
        "ready verified claims exactly match stored facts",
    ))
    calls_ok = (
        len(observed_llm_calls) == 1
        and observed_llm_calls[0].get("completed") is True
        and all(
            SHA256_PATTERN.fullmatch(str(observed_llm_calls[0].get(key, "")))
            for key in (
                "system_sha256",
                "prompt_sha256",
                "schema_sha256",
                "response_sha256",
            )
        )
    )
    checks.append(_check(
        "One successful real Ollama call",
        calls_ok,
        len(observed_llm_calls),
        "exactly 1 hashed request/response",
    ))
    cue_ok, cue_detail = coaching_cues_are_bounded(
        document.get("coaching_cues"), session_id=session_id, duration=duration
    )
    checks.append(_check(
        "Coaching cue bounds",
        cue_ok,
        cue_detail,
        "ordered cue timestamps stay within the video",
    ))
    media = document.get("media")
    media_ok = (
        isinstance(media, Mapping)
        and media.get("available") is True
        and media.get("mime_type") == "video/mp4"
        and media.get("source") == "upload"
        and isinstance(media.get("playback_url"), str)
    )
    checks.append(_check(
        "Encrypted replay grant returned",
        media_ok,
        (
            {
                "available": media.get("available"),
                "mime_type": media.get("mime_type"),
                "source": media.get("source"),
            }
            if isinstance(media, Mapping)
            else type(media).__name__
        ),
        "available signed upload replay as video/mp4",
    ))
    return checks


def _filesystem_security_checks(
    *,
    store: PresentationStore,
    profile_id: str,
    profile_name: str,
    session_id: str,
    source_path: Path,
) -> tuple[list[dict[str, object]], bytes]:
    data_dir = store.data_dir
    profile_path = data_dir / f"{profile_id}.presentcoach"
    media_path = data_dir / f"{profile_id}-{session_id}.presentcoach-media"
    checks: list[dict[str, object]] = []
    modes = {
        "directory": stat.S_IMODE(data_dir.stat().st_mode),
        "profile": stat.S_IMODE(profile_path.stat().st_mode),
        "media": stat.S_IMODE(media_path.stat().st_mode),
    }
    checks.append(_check(
        "Encrypted storage permissions",
        modes == {"directory": 0o700, "profile": 0o600, "media": 0o600},
        {key: oct(value) for key, value in modes.items()},
        "directory 0700; files 0600",
    ))
    profile_ciphertext = profile_path.read_bytes()
    media_ciphertext = media_path.read_bytes()
    source_head = source_path.read_bytes()[:RANGE_BYTES]
    known_plaintext_signatures = (
        b"\x00\x00\x00\x18ftyp",
        b"\x00\x00\x00\x1cftyp",
        b"\x1a\x45\xdf\xa3",
        b"OggS",
    )
    no_plaintext_signature = (
        media_ciphertext.startswith(MEDIA_MAGIC)
        and not any(media_ciphertext.startswith(item) for item in known_plaintext_signatures)
        and (not source_head or source_head not in media_ciphertext)
        and profile_name.encode("utf-8") not in profile_ciphertext
    )
    checks.append(_check(
        "No plaintext media/profile signature at rest",
        no_plaintext_signature,
        {
            "encrypted_container_magic": media_ciphertext[: len(MEDIA_MAGIC)].decode("ascii", "replace"),
            "source_probe_present": bool(source_head and source_head in media_ciphertext),
            "profile_name_present": profile_name.encode("utf-8") in profile_ciphertext,
        },
        "PCMEDIA2 container only; no MP4/WebM probe or profile name",
    ))
    reader = store.open_session_media(profile_id, session_id)
    try:
        end = min(RANGE_BYTES, reader.metadata.plaintext_bytes)
        plaintext_head = b"".join(reader.iter_range(0, end))
    finally:
        reader.close()
    return checks, plaintext_head


def _playback_checks(
    client,
    *,
    playback_url: str,
    profile_id: str,
    session_id: str,
    expected_head: bytes,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    parsed = urlsplit(playback_url)
    token = parse_qs(parsed.query).get("token", [""])[0]
    grant_shape = (
        parsed.path
        == f"/api/profiles/{profile_id}/sessions/{session_id}/video"
        and bool(token)
    )
    checks.append(_check(
        "Playback URL is session-scoped and signed",
        grant_shape,
        {"path_matches": parsed.path.endswith(f"/{session_id}/video"), "token_present": bool(token)},
        "session path plus non-empty signed token",
    ))
    ranged = client.get(
        playback_url,
        headers={"Range": f"bytes=0-{max(0, len(expected_head) - 1)}"},
    )
    ranged_data = ranged.get_data()
    range_ok = (
        ranged.status_code == 206
        and ranged_data == expected_head
        and ranged.headers.get("Accept-Ranges") == "bytes"
        and ranged.headers.get("Content-Range", "").startswith("bytes 0-")
        and ranged.headers.get("Cache-Control") == "no-store"
    )
    checks.append(_check(
        "Signed HTTP Range replay",
        range_ok,
        {
            "status": ranged.status_code,
            "bytes": len(ranged_data),
            "content_range": ranged.headers.get("Content-Range"),
        },
        f"206 with exact first {len(expected_head)} authenticated plaintext bytes",
    ))
    ranged.close()
    unsigned = client.get(parsed.path)
    unsigned_ok = unsigned.status_code == 403 and (
        unsigned.get_json(silent=True) or {}
    ).get("code") == "media_access_denied"
    checks.append(_check(
        "Unsigned replay denied",
        unsigned_ok,
        unsigned.status_code,
        "403 media_access_denied",
    ))
    unsigned.close()
    return checks


def _case_summary(
    clip: DynamicClip,
    *,
    document: Mapping[str, object] | None,
    checks: list[dict[str, object]],
    llm_calls: Sequence[Mapping[str, object]],
    annotation_sha256: str,
    wall_seconds: float,
) -> dict[str, object]:
    measurements: dict[str, object] = {}
    feedback_summary: dict[str, object] = {}
    cue_summary: dict[str, object] = {}
    session_id: str | None = None
    if isinstance(document, Mapping):
        session = document.get("session")
        metrics = document.get("metrics")
        feedback = document.get("feedback")
        cues = document.get("coaching_cues")
        if isinstance(session, Mapping):
            session_id = str(session.get("session_id", "")) or None
            measurements["duration_seconds"] = session.get("duration_seconds")
            transcript = session.get("transcript")
            vision = session.get("vision_metrics")
            measurements["transcript_word_count"] = (
                len(transcript) if isinstance(transcript, list) else 0
            )
            measurements["vision_bucket_count"] = (
                len(vision) if isinstance(vision, list) else 0
            )
        if isinstance(metrics, Mapping):
            aggregate = metrics.get("aggregate")
            if isinstance(aggregate, Mapping):
                measurements["aggregate"] = {
                    key: aggregate.get(key)
                    for key in (
                        "analyzed_vision_fps",
                        "face_presence_percent",
                        "eye_contact_percent",
                        "overall_words_per_minute",
                        "strict_filler_count",
                        "pauses_over_3_seconds",
                    )
                }
            quality = metrics.get("quality_flags")
            measurements["quality_flags"] = (
                dict(quality) if isinstance(quality, Mapping) else {}
            )
        if isinstance(feedback, Mapping):
            feedback_summary = {
                "status": feedback.get("status"),
                "source": feedback.get("source"),
                "strength_metrics": [
                    item.get("metric")
                    for item in feedback.get("strengths", [])
                    if isinstance(item, Mapping)
                ],
                "improvement_metrics": [
                    item.get("metric")
                    for item in feedback.get("improvements", [])
                    if isinstance(item, Mapping)
                ],
                "insufficient_data": feedback.get("insufficient_data"),
            }
        if isinstance(cues, Mapping):
            cue_summary = {
                "status": cues.get("status"),
                "counts": cues.get("counts"),
                "cue_count": len(cues.get("cues", []))
                if isinstance(cues.get("cues"), list)
                else 0,
            }
    return {
        "case_id": clip.clip_id,
        "dataset_role": clip.dataset_role,
        "split_group_id": clip.split_group_id,
        "media": {
            "filename": clip.filename,
            "sha256": clip.sha256,
            "annotation_sha256": annotation_sha256,
            "source_page_url": clip.source_page_url,
            "license_spdx": clip.license_spdx,
            "excerpt_start_seconds_in_source": clip.excerpt_start_seconds,
            "excerpt_duration_seconds": clip.excerpt_duration_seconds,
        },
        "session_id": session_id,
        "wall_seconds": round(max(0.0, wall_seconds), 3),
        "measurements": measurements,
        "feedback": feedback_summary,
        "coaching_cues": cue_summary,
        "llm_calls": [_bounded_report_value(item) for item in llm_calls],
        "passed": all(bool(item.get("passed")) for item in checks),
        "checks": checks,
    }


def _preflight_selected_media(
    manifest: DynamicManifest, clips: Sequence[DynamicClip]
) -> dict[str, dict[str, str]]:
    evidence: dict[str, dict[str, str]] = {}
    media_dir = manifest.path.parent
    for clip in clips:
        path = (media_dir / clip.filename).resolve()
        if media_dir not in path.parents or not path.is_file():
            raise E2EContractError(
                f"Missing canonical clip {clip.filename}; run the corpus downloader"
            )
        digest = sha256_file(path)
        if digest != clip.sha256:
            raise E2EContractError(f"Canonical media hash mismatch for {clip.clip_id}")
        annotation_path = (media_dir / clip.annotation_file).resolve()
        evidence[clip.clip_id] = {
            "media_sha256": digest,
            "annotation_sha256": sha256_file(annotation_path),
        }
    return evidence


def _artifact_preflight(
    manifest: DynamicManifest, *, model_name: str
) -> tuple[
    LocalVideoAnalyzer,
    ObservedOllama,
    dict[str, object],
    dict[str, str],
]:
    model_path = ROOT / "models" / manifest.model_filename
    whisper_model = ROOT / "models" / "whisper" / "ggml-base.en-q5_1.bin"
    if sha256_file(model_path) != manifest.model_sha256:
        raise E2EContractError("Pinned MediaPipe model hash mismatch")
    transcriber = WhisperCppTranscriber(model_path=whisper_model)
    analyzer = LocalVideoAnalyzer(
        model_path=model_path,
        transcriber=transcriber,
    )
    real_llm = OllamaPresentationLLM(model=model_name)
    model_status = real_llm.status()
    model_digest = str(model_status.get("digest", ""))
    if (
        model_status.get("available") is not True
        or model_status.get("installed") is not True
        or SHA256_PATTERN.fullmatch(model_digest) is None
    ):
        raise E2EContractError("Exact local PresentCoach Ollama model is unavailable")
    observed = ObservedOllama(real_llm)
    artifacts = {
        "manifest_sha256": manifest.sha256,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "mediapipe_model_sha256": manifest.model_sha256,
        "whisper_model_sha256": sha256_file(whisper_model),
        "feedback_schema_sha256": _canonical_json_sha256(FEEDBACK_SCHEMA),
    }
    return analyzer, observed, model_status, artifacts


def _authorize(client, app) -> None:
    capability = app.extensions["presentation_access_capability"]
    response = client.get("/", query_string={"access": capability})
    if response.status_code != 303 or response.headers.get("Location") != "/":
        raise E2EContractError("Private launch capability exchange failed")
    if "access=" in "\n".join(response.headers.values()):
        raise E2EContractError("Private launch capability leaked into response headers")
    response.close()


def _split_result(role: str, cases: list[dict[str, object]]) -> dict[str, object]:
    if not cases:
        return {
            "dataset_role": role,
            "status": "not_run",
            "case_count": 0,
            "passed": 0,
            "pass_rate_percent": None,
            "cases": [],
        }
    passed = sum(item.get("passed") is True for item in cases)
    return {
        "dataset_role": role,
        "status": "passed" if passed == len(cases) else "failed",
        "case_count": len(cases),
        "passed": passed,
        "pass_rate_percent": round(100.0 * passed / len(cases), 1),
        "cases": cases,
    }


def build_report(
    *,
    selected_role: str,
    manifest: DynamicManifest,
    model_status: Mapping[str, object],
    artifacts: Mapping[str, str],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    """Build a split-explicit report that cannot imply an unrun holdout passed."""

    results = {
        role: _split_result(role, cases if role == selected_role else [])
        for role in DATASET_ROLES
    }
    selected = results[selected_role]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evaluation_only": True,
        "used_for_training": False,
        "identity_recognition": False,
        "landmarks_persisted": False,
        "selected_dataset_role": selected_role,
        "holdout_default": "sealed_not_run",
        "target_analyzed_fps": manifest.target_fps,
        "local_model": {
            "name": model_status.get("model"),
            "digest": model_status.get("digest"),
            "available": model_status.get("available") is True,
        },
        "artifacts": dict(artifacts),
        "overall": {
            "status": selected["status"],
            "case_count": selected["case_count"],
            "passed": selected["passed"],
            "pass_rate_percent": selected["pass_rate_percent"],
        },
        "results": results,
        "limitations": [
            "This is a bounded scenario/API evaluation, not population accuracy or model training.",
            "Natural presentation video has no independently labeled camera-contact gaze truth.",
            "The report stores aggregates and hashes only; no transcript, landmark array, video, key, or access capability is retained.",
        ],
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(payload) > MAX_REPORT_BYTES:
        raise E2EContractError("The bounded E2E report exceeded 2 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def run_evaluation(
    *,
    manifest: DynamicManifest,
    selected_role: str,
    model_name: str,
) -> dict[str, object]:
    clips = selected_clips(manifest, selected_role)
    selected_hashes = _preflight_selected_media(manifest, clips)
    analyzer, observed_llm, model_status, artifacts = _artifact_preflight(
        manifest, model_name=model_name
    )
    cases: list[dict[str, object]] = []
    profile_name = "PresentCoach API E2E"
    with tempfile.TemporaryDirectory(prefix="presentcoach-api-e2e-") as temporary:
        data_dir = Path(temporary) / "encrypted-store"
        keys = MemoryKeyStore()
        store = PresentationStore(data_dir=data_dir, key_store=keys)
        capability = "e2e-local-capability-" + secrets.token_urlsafe(48)
        app = create_app(
            store=store,
            llm=observed_llm,
            recorder=_InactiveRecorder(),
            video_analyzer=analyzer,
            access_capability=capability,
            testing=True,
        )
        client = app.test_client()
        _authorize(client, app)
        bootstrap_response = client.get("/api/bootstrap")
        if bootstrap_response.status_code != 200:
            raise E2EContractError("Authorized bootstrap failed")
        bootstrap = bootstrap_response.get_json()
        bootstrap_response.close()
        csrf = str(bootstrap.get("csrf_token", ""))
        created_response = client.post(
            "/api/profiles",
            json={"name": profile_name},
            headers={"X-CSRF-Token": csrf},
        )
        if created_response.status_code != 201:
            raise E2EContractError("Temporary encrypted profile creation failed")
        profile = created_response.get_json()["profile"]
        profile_id = str(profile["id"])
        created_response.close()

        session_ids: dict[str, str] = {}
        for clip in clips:
            print(f"API E2E {clip.clip_id}", flush=True)
            case_started = time.monotonic()
            source_path = manifest.path.parent / clip.filename
            before_calls = len(observed_llm.calls)
            response = None
            document: Mapping[str, object] | None = None
            checks: list[dict[str, object]] = []
            try:
                with source_path.open("rb") as upload:
                    response = client.post(
                        "/api/videos/analyze",
                        data={
                            "profile_id": profile_id,
                            "note": "Frozen dynamic-corpus API evaluation; never training data.",
                            "video": (upload, clip.filename),
                        },
                        content_type="multipart/form-data",
                        headers={"X-CSRF-Token": csrf},
                    )
                status = response.status_code
                raw_document = response.get_json(silent=True)
                checks.append(_check(
                    "POST /api/videos/analyze",
                    status == 201,
                    status,
                    "201 Created",
                ))
                if isinstance(raw_document, Mapping):
                    document = raw_document
                if status == 201 and document is not None:
                    call_slice = observed_llm.calls[before_calls:]
                    checks.extend(validate_api_document(
                        document,
                        expected_duration=clip.excerpt_duration_seconds,
                        observed_llm_calls=call_slice,
                    ))
                    session = document.get("session")
                    media = document.get("media")
                    if not isinstance(session, Mapping) or not isinstance(media, Mapping):
                        raise E2EContractError("Successful upload omitted session media")
                    session_id = str(session.get("session_id", ""))
                    session_ids[clip.clip_id] = session_id
                    storage_checks, plaintext_head = _filesystem_security_checks(
                        store=store,
                        profile_id=profile_id,
                        profile_name=profile_name,
                        session_id=session_id,
                        source_path=source_path,
                    )
                    checks.extend(storage_checks)
                    checks.extend(_playback_checks(
                        client,
                        playback_url=str(media.get("playback_url", "")),
                        profile_id=profile_id,
                        session_id=session_id,
                        expected_head=plaintext_head,
                    ))
            except Exception as error:
                checks.append(_check(
                    "Evaluator completed case without exception",
                    False,
                    f"{type(error).__name__}: {error}"[:500],
                    "no exception",
                ))
            finally:
                if response is not None:
                    response.close()
            call_slice = observed_llm.calls[before_calls:]
            case = _case_summary(
                clip,
                document=document,
                checks=checks,
                llm_calls=call_slice,
                annotation_sha256=selected_hashes[clip.clip_id]["annotation_sha256"],
                wall_seconds=time.monotonic() - case_started,
            )
            cases.append(case)
            print(f"{'PASS' if case['passed'] else 'FAIL'} {clip.clip_id}", flush=True)

        # A new Flask app has a new signer and browser session but shares the
        # encrypted store/key objects, simulating a local application reload.
        restarted = create_app(
            store=store,
            llm=observed_llm,
            recorder=_InactiveRecorder(),
            video_analyzer=analyzer,
            access_capability="reloaded-e2e-capability-" + secrets.token_urlsafe(48),
            testing=True,
        )
        fresh_client = restarted.test_client()
        _authorize(fresh_client, restarted)
        reload_response = fresh_client.get(
            "/api/bootstrap", query_string={"profile": profile_id}
        )
        reload_document = reload_response.get_json(silent=True)
        reload_status = reload_response.status_code
        reload_response.close()
        loaded_by_id: dict[str, Mapping[str, object]] = {}
        if isinstance(reload_document, Mapping):
            sessions = reload_document.get("sessions")
            if isinstance(sessions, list):
                for item in sessions:
                    if not isinstance(item, Mapping):
                        continue
                    raw_session = item.get("session")
                    if isinstance(raw_session, Mapping):
                        loaded_by_id[str(raw_session.get("session_id", ""))] = item
        for case in cases:
            clip_id = str(case["case_id"])
            expected_session_id = session_ids.get(clip_id, "")
            loaded = loaded_by_id.get(expected_session_id)
            media = loaded.get("media") if isinstance(loaded, Mapping) else None
            persisted = (
                reload_status == 200
                and isinstance(loaded, Mapping)
                and isinstance(media, Mapping)
                and media.get("available") is True
                and isinstance(media.get("playback_url"), str)
                and bool(parse_qs(urlsplit(str(media["playback_url"])).query).get("token"))
            )
            case["checks"].append(_check(
                "Bootstrap reload persistence",
                persisted,
                {
                    "bootstrap_status": reload_status,
                    "session_found": isinstance(loaded, Mapping),
                    "fresh_signed_media": isinstance(media, Mapping)
                    and media.get("available") is True,
                },
                "session reloads from encrypted history with a fresh signed replay grant",
            ))
            case["passed"] = all(
                bool(item.get("passed")) for item in case["checks"]
            )

    return build_report(
        selected_role=selected_role,
        manifest=manifest,
        model_status=model_status,
        artifacts=artifacts,
        cases=cases,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional split-specific report path.",
    )
    parser.add_argument(
        "--dataset-role",
        choices=DATASET_ROLES,
        default="development",
        help=(
            "Development is the safe default. Selecting model_output_holdout is "
            "an explicit one-way evaluation decision after thresholds are frozen."
        ),
    )
    parser.add_argument("--model", default="presentcoach-local")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_dynamic_manifest(args.manifest)
        report = run_evaluation(
            manifest=manifest,
            selected_role=args.dataset_role,
            model_name=args.model,
        )
        write_report(
            args.output or default_output_for_role(args.dataset_role), report
        )
    except (E2EContractError, OSError, ValueError) as error:
        print(f"E2E evaluator failed closed: {error}", file=sys.stderr)
        return 2
    overall = report["overall"]
    print(json.dumps(overall, sort_keys=True), flush=True)
    return 0 if overall["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
