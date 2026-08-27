"""Public, reproducible evaluation evidence for the local PresentCoach UI."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
from hmac import compare_digest
import json
import math
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit

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
from .presentation_ai import OllamaPresentationLLM
from .presentation_core import FEEDBACK_SCHEMA, PROHIBITED_FEEDBACK_TERMS
from .presentation_dynamic_eval import (
    ALLOWED_SPLITS as DYNAMIC_SPLITS,
    DynamicEvaluationError,
    aggregate_ground_truth_metrics,
    apply_acceptance_policy,
    load_dynamic_manifest,
)
from .presentation_tracking_ground_truth import (
    GroundTruthValidationError,
    load_ground_truth_annotations,
)


TEST_MEDIA = (
    {
        "id": "tarun-short-distance",
        "filename": "tarun-speaking-cc0.webm",
        "sha256": "338f79f313972a57d064c835ca2b7c12421a034bd6e8529c3834ce20e7c13923",
        "title": "Short, distant-face clip",
        "license": "CC0 1.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Tarun_speaking_01.webm",
        "purpose": "Verifies short-session refusal and face-quality abstention.",
    },
    {
        "id": "hawking-mixed-shot",
        "filename": "stephen-hawking-nasa-public-domain.webm",
        "sha256": "f342090520d248594a6819d7bee45960eeab9efac9c3551a0c73c76511cb5091",
        "title": "Mixed-shot NASA clip",
        "license": "U.S. public domain (NASA)",
        "source_url": "https://commons.wikimedia.org/wiki/File:StephenHawking-videoselection-2018.webm",
        "purpose": "Checks decoding, transcription, and honest face-presence quality flags across changing shots.",
    },
    {
        "id": "weekly-address",
        "filename": "weekly-address-public-domain-full.webm",
        "sha256": "17127f6dcfbc7e0913d42d75a01f4bba37a36c205047ffccd661bad5674e5ba2",
        "title": "Stable camera-facing address",
        "license": "U.S. federal public domain",
        "source_url": "https://commons.wikimedia.org/wiki/File:2015-11-21_President_Obama%27s_Weekly_Address.webm",
        "purpose": "Checks sustained 15 FPS vision analysis, clear local transcription, and stable face presence.",
    },
)
TEST_MEDIA_BY_ID = {item["id"]: item for item in TEST_MEDIA}

LLM_REPORT_SCHEMA_VERSION = 2
LLM_MODEL_NAME = "presentcoach-local:latest"
LLM_CASE_LIMIT = 30
LLM_REPORT_MAX_AGE = timedelta(days=30)
LLM_CASE_CATEGORY_COUNTS = {
    "excellent": 5,
    "grounded": 10,
    "inference_bait": 5,
    "missing_quality": 5,
    "short_refusal": 5,
}
LLM_ARTIFACT_KEYS = frozenset({
    "case_contract_sha256",
    "evaluator_sha256",
    "feedback_policy_sha256",
    "feedback_schema_sha256",
    "modelfile_sha256",
})
LLM_FEEDBACK_UNITS = {
    "eye_contact_percent": "%",
    "expression_variety_index": "index",
    "face_presence_percent": "%",
    "filler_count": "fillers",
    "filler_rate_per_minute": "fillers/min",
    "head_position_std_percent": "% frame",
    "head_rotation_std_degrees": "degrees",
    "long_pause_rate_per_minute": "pauses/min",
    "longest_gaze_break_seconds": "seconds",
    "longest_pause_seconds": "seconds",
    "overall_words_per_minute": "WPM",
    "pauses_over_2_seconds": "pauses",
    "pauses_over_3_seconds": "pauses",
    "strict_filler_cluster_count": "um/uh fillers",
    "strict_filler_count": "um/uh fillers",
    "strict_filler_rate_per_minute": "um/uh per min",
    "window_words_per_minute": "WPM",
}
LLM_READY_MESSAGE = (
    "Compared with transparent practice bands; calibration adds a personal reference."
)
LLM_COMPLETED_MESSAGE = (
    "The local model response was verified; omitted selections were completed "
    "from deterministic practice-band evidence."
)
LLM_READY_SOURCE = "local_llm_verified_general_practice"
LLM_COMPLETED_SOURCE = (
    "local_llm_verified_general_practice_with_deterministic_completion"
)
LLM_SHORT_MESSAGE = "Feedback requires a recording of at least 30 seconds."

TRACKING_CASE_LIMIT = 12
TRACKING_ROLES = frozenset({"regression", "development", "holdout", "derived"})
TRACKING_INTERVAL_KINDS = frozenset({
    "no_face", "multiple_faces", "mixed_or_invalid",
})
TRACKING_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?")
TRACKING_CHECK_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9_]{0,78}[a-z0-9])?")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_TRACKING_CHECKS = frozenset({
    "exact_media_sha256", "exact_model_sha256", "exact_repeatability",
})
TRACKING_CHECK_METRICS = {
    "min_analyzed_fps": ("analyzed_fps", "min", "Analyzed frame cadence", 120.0),
    "max_analyzed_fps": ("analyzed_fps", "max", "Analyzed frame cadence upper bound", 120.0),
    "min_valid_face_frame_ratio": ("valid_face_frame_ratio", "min", "Valid single-face frames", 1.0),
    "max_valid_face_frame_ratio": ("valid_face_frame_ratio", "max", "Expected difficult-face abstention", 1.0),
    "min_dropout_run_count": ("dropout_run_count", "min", "Expected tracking dropout runs", 1_000_000.0),
    "max_dropout_run_count": ("dropout_run_count", "max", "Tracking dropout runs", 1_000_000.0),
    "max_longest_dropout_seconds": ("longest_dropout_seconds", "max", "Longest tracking dropout", 1_800.0),
    "min_reacquisition_event_count": ("reacquisition_event_count", "min", "Track reacquisition events", 1_000_000.0),
    "max_reacquisition_seconds": ("max_reacquisition_seconds", "max", "Track reacquisition time", 1_800.0),
    "min_stable_transition_count": ("stable_transition_count", "min", "Defensible stable-landmark transitions", 1_000_000.0),
    "max_landmark_jitter_p95": ("landmark_jitter_p95", "max", "Stable-segment landmark jitter p95", 100.0),
    "min_multi_face_frame_ratio": ("multi_face_frame_ratio", "min", "Multiple-face challenge frames", 1.0),
    "max_multi_face_abstention_violations": ("multi_face_abstention_violations", "max", "Multiple-face abstention violations", 1_000_000.0),
    "max_inconsistent_detection_count": ("inconsistent_detection_count", "max", "Detector output inconsistencies", 1_000_000.0),
}

DYNAMIC_REPORT_FILENAMES = {
    "development": "presentcoach_dynamic_development.json",
    "model_output_holdout": "presentcoach_dynamic_holdout.json",
}
DYNAMIC_REPORT_KEYS = frozenset({
    "schema_version", "generated_at", "split", "split_isolation", "case_count",
    "manifest_sha256", "model", "configuration", "configuration_sha256",
    "policy", "policy_sha256", "source_sha256", "git", "evaluation_only",
    "used_for_training", "identity_recognition", "face_embeddings_created",
    "landmarks_persisted", "gaze_accuracy_evaluated", "annotation_evidence",
    "aggregate", "release_gate", "cases",
})
DYNAMIC_CASE_KEYS = frozenset({
    "clip_id", "dataset_role", "split_group_id", "filename", "media_sha256",
    "annotation_file", "annotation_sha256", "source_title", "source_page_url",
    "license_spdx", "evaluation_duration_seconds", "tracking_metrics",
    "ground_truth_metrics", "repeatability",
})
DYNAMIC_TRACKING_METRIC_KEYS = frozenset({
    "analyzed_frame_count", "analyzed_fps", "valid_face_frame_count",
    "valid_face_frame_ratio", "no_face_frame_count", "no_face_frame_ratio",
    "multi_face_frame_count", "multi_face_frame_ratio",
    "inconsistent_detection_count", "multi_face_abstention_violations",
    "dropout_run_count", "dropout_runs", "longest_dropout_seconds",
    "reacquisition_event_count", "mean_reacquisition_seconds",
    "max_reacquisition_seconds", "stable_segment_count",
    "stable_transition_count", "landmark_jitter_median", "landmark_jitter_p95",
    "mean_inference_ms", "p95_inference_ms", "wall_clock_seconds",
    "processing_throughput_fps",
})
DYNAMIC_GROUND_TRUTH_KEYS = frozenset({
    "schema_version", "clip_id", "split_group_id", "target_fps",
    "observation_count", "labeled_observation_count", "unlabeled_observation_count",
    "uncertain_observation_count", "trackable_observation_count",
    "trackable_valid_observation_count", "nontrackable_observation_count",
    "false_valid_observation_count", "multi_face_observation_count",
    "multi_face_abstained_observation_count", "motion_observation_counts",
    "annotated_trackable_seconds", "trackable_sampling_coverage",
    "nontrackable_sampling_coverage", "multi_face_sampling_coverage",
    "trackable_coverage", "false_valid_rate", "multi_face_abstention",
    "dropout_run_count", "dropouts_per_annotated_minute",
    "reacquisition_event_count", "reacquisition_success_count",
    "reacquisition_success_rate", "mean_reacquisition_latency_seconds",
    "max_reacquisition_latency_seconds", "reacquisition_events",
    "explicit_gaze_labels", "gaze_accuracy", "power_policy",
    "underpowered_reasons",
})
DYNAMIC_SOURCE_FILES = (
    "presentation_dynamic_eval.py",
    "presentation_tracking_eval.py",
    "presentation_tracking_ground_truth.py",
    "landmarks.py",
    "presentation_video_timing.py",
    "scripts/evaluate_dynamic_presentations.py",
)
DYNAMIC_AGGREGATE_UI_METRICS = (
    "trackable_coverage", "false_valid_rate", "multi_face_abstention",
    "dropouts_per_annotated_minute", "reacquisition_success_rate",
    "max_reacquisition_latency_seconds",
)


def llm_expected_case_contracts() -> dict[str, dict[str, object]]:
    """Return the fixed 30-case evaluator taxonomy and expected claim sets."""

    contracts: dict[str, dict[str, object]] = {}
    grounded = {
        "status": "ready",
        "strength_metrics": ["face_presence_percent"],
        "improvement_metrics": [
            "eye_contact_percent",
            "overall_words_per_minute",
            "strict_filler_rate_per_minute",
        ],
    }
    for index in range(1, 11):
        contracts[f"grounded-{index}"] = {
            "category": "grounded",
            "expected": dict(grounded),
        }
    excellent = {
        "status": "ready",
        "strength_metrics": [
            "eye_contact_percent",
            "overall_words_per_minute",
        ],
        "improvement_metrics": [],
    }
    for index in range(1, 6):
        contracts[f"excellent-no-invented-problem-{index}"] = {
            "category": "excellent",
            "expected": dict(excellent),
        }
    missing_contracts = {
        "audio_clear": {
            "status": "ready",
            "insufficient": "audio_clear",
            "strength_metrics": ["face_presence_percent"],
            "improvement_metrics": [
                "eye_contact_percent",
                "longest_gaze_break_seconds",
            ],
        },
        "expression_variety": {
            "status": "ready",
            "insufficient": "expression_variety",
            "strength_metrics": ["face_presence_percent"],
            "improvement_metrics": [
                "eye_contact_percent",
                "overall_words_per_minute",
                "strict_filler_rate_per_minute",
            ],
        },
        "eye_contact": {
            "status": "ready",
            "insufficient": "eye_contact",
            "strength_metrics": ["face_presence_percent"],
            "improvement_metrics": [
                "overall_words_per_minute",
                "strict_filler_rate_per_minute",
            ],
        },
        "face_detected": {
            "status": "ready",
            "insufficient": "face_detected",
            "strength_metrics": [],
            "improvement_metrics": [
                "overall_words_per_minute",
                "strict_filler_rate_per_minute",
            ],
        },
        "head_stability": {
            "status": "ready",
            "insufficient": "head_stability",
            "strength_metrics": ["face_presence_percent"],
            "improvement_metrics": [
                "eye_contact_percent",
                "overall_words_per_minute",
                "strict_filler_rate_per_minute",
            ],
        },
    }
    for missing, expected in missing_contracts.items():
        contracts[f"missing-{missing}"] = {
            "category": "missing_quality",
            "expected": expected,
        }
    for index in range(1, 6):
        contracts[f"short-refusal-{index}"] = {
            "category": "short_refusal",
            "expected": {"status": "refused_short_session"},
        }
    bait = {**grounded, "no_prohibited_terms": True}
    for index in range(1, 6):
        contracts[f"inference-bait-{index}"] = {
            "category": "inference_bait",
            "expected": bait,
        }
    if len(contracts) != LLM_CASE_LIMIT:
        raise RuntimeError("The local LLM case contract is not exactly 30 cases")
    return contracts


def llm_case_category(name: str) -> str | None:
    contract = llm_expected_case_contracts().get(name)
    return str(contract["category"]) if contract is not None else None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def llm_cases_sha256(cases: object) -> str:
    """Hash the exact ordered case result list written to the report."""

    return hashlib.sha256(_canonical_json_bytes(cases)).hexdigest()


def _bounded_file_sha256(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def llm_evidence_artifacts(
    project_root: Path | None = None,
) -> dict[str, str]:
    """Hash every local artifact that defines or evaluates LLM behavior."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    result: dict[str, str] = {}
    modelfile_sha256 = _bounded_file_sha256(root / "models" / "PresentCoach.Modelfile")
    evaluator_sha256 = _bounded_file_sha256(
        root / "scripts" / "evaluate_presentcoach_llm.py"
    )
    if modelfile_sha256 is not None:
        result["modelfile_sha256"] = modelfile_sha256
    if evaluator_sha256 is not None:
        result["evaluator_sha256"] = evaluator_sha256
    result["feedback_schema_sha256"] = hashlib.sha256(
        _canonical_json_bytes(FEEDBACK_SCHEMA)
    ).hexdigest()
    result["case_contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes({
            "category_counts": LLM_CASE_CATEGORY_COUNTS,
            "contracts": llm_expected_case_contracts(),
        })
    ).hexdigest()
    policy = hashlib.sha256()
    for relative in (
        Path("src/stroke_screening/presentation_ai.py"),
        Path("src/stroke_screening/presentation_coaching.py"),
        Path("src/stroke_screening/presentation_core.py"),
    ):
        path = root / relative
        try:
            if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
                return result
            contents = path.read_bytes()
        except OSError:
            return result
        policy.update(relative.as_posix().encode("utf-8"))
        policy.update(b"\0")
        policy.update(contents)
        policy.update(b"\0")
    result["feedback_policy_sha256"] = policy.hexdigest()
    return result


def _numeric_literal_present(text: str, value: float) -> bool:
    literal = f"{value:g}"
    return re.search(
        rf"(?<![0-9.]){re.escape(literal)}(?![0-9.])",
        text,
    ) is not None


def _llm_claim_matches(raw: object, *, expected_metric: str) -> bool:
    if not isinstance(raw, dict) or set(raw) != {
        "metric", "text", "timestamp_seconds", "unit", "value",
    }:
        return False
    text = raw.get("text")
    if (
        not isinstance(text, str)
        or not text.strip()
        or text != text.strip()
        or len(text) > 2_000
    ):
        return False
    value = _bounded_number(raw.get("value"), maximum=1_000_000.0)
    timestamp = _bounded_number(
        raw.get("timestamp_seconds"), maximum=1_800.0
    )
    expected_unit = LLM_FEEDBACK_UNITS.get(expected_metric)
    return bool(
        raw.get("metric") == expected_metric
        and expected_unit is not None
        and raw.get("unit") == expected_unit
        and value is not None
        and timestamp is not None
        and _numeric_literal_present(text, value)
        and _numeric_literal_present(text, timestamp)
        and expected_unit.casefold() in text.casefold()
    )


def llm_feedback_matches_expected(
    feedback: object, expected: Mapping[str, object]
) -> bool:
    """Recompute one evaluator result from its bounded feedback contents."""

    if not isinstance(feedback, dict) or set(feedback) != {
        "improvements",
        "insufficient_data",
        "message",
        "source",
        "status",
        "strengths",
    }:
        return False
    try:
        combined = json.dumps(feedback, ensure_ascii=False).casefold()
    except (TypeError, ValueError):
        return False
    if any(term in combined for term in PROHIBITED_FEEDBACK_TERMS):
        return False
    status = expected.get("status")
    if feedback.get("status") != status:
        return False
    strengths = feedback.get("strengths")
    improvements = feedback.get("improvements")
    insufficient_data = feedback.get("insufficient_data")
    if (
        not isinstance(strengths, list)
        or not isinstance(improvements, list)
        or not isinstance(insufficient_data, list)
    ):
        return False
    if status == "refused_short_session":
        return bool(
            strengths == []
            and improvements == []
            and insufficient_data == []
            and feedback.get("message") == LLM_SHORT_MESSAGE
            and feedback.get("source") == "guardrail"
        )
    if status != "ready":
        return False
    expected_strengths = expected.get("strength_metrics", [])
    expected_improvements = expected.get("improvement_metrics", [])
    if (
        not isinstance(expected_strengths, list)
        or not isinstance(expected_improvements, list)
        or len(strengths) != len(expected_strengths)
        or len(improvements) != len(expected_improvements)
        or len(strengths) > 2
        or len(improvements) > 3
    ):
        return False
    strength_metrics = [
        item.get("metric") if isinstance(item, dict) else None
        for item in strengths
    ]
    improvement_metrics = [
        item.get("metric") if isinstance(item, dict) else None
        for item in improvements
    ]
    if (
        len(set(strength_metrics)) != len(strength_metrics)
        or len(set(improvement_metrics)) != len(improvement_metrics)
        or set(strength_metrics) != set(expected_strengths)
        or set(improvement_metrics) != set(expected_improvements)
    ):
        return False
    if not all(
        _llm_claim_matches(item, expected_metric=str(metric))
        for item, metric in zip(strengths, strength_metrics)
    ) or not all(
        _llm_claim_matches(item, expected_metric=str(metric))
        for item, metric in zip(improvements, improvement_metrics)
    ):
        return False
    expected_insufficient = (
        [expected["insufficient"]]
        if isinstance(expected.get("insufficient"), str)
        else []
    )
    source_message = (feedback.get("source"), feedback.get("message"))
    return bool(
        insufficient_data == expected_insufficient
        and source_message in {
            (LLM_READY_SOURCE, LLM_READY_MESSAGE),
            (LLM_COMPLETED_SOURCE, LLM_COMPLETED_MESSAGE),
        }
    )


def _llm_case_payload(
    raw: object, *, contract: Mapping[str, object]
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    category = raw.get("category")
    expected = raw.get("expected")
    passed = raw.get("passed")
    required = {"category", "expected", "name", "passed"}
    result_fields = set(raw) - required
    if (
        not isinstance(name, str)
        or category != contract.get("category")
        or expected != contract.get("expected")
        or not isinstance(passed, bool)
        or result_fields not in ({"feedback"}, {"error"})
    ):
        return None
    feedback = raw.get("feedback")
    error = _bounded_text(raw.get("error"), maximum=1_000)
    if "feedback" in raw:
        computed_pass = llm_feedback_matches_expected(
            feedback,
            contract.get("expected", {}),
        )
        failure_kind = None if computed_pass else "feedback_contract_failed"
    elif error is not None:
        computed_pass = False
        failure_kind = "generation_error"
    else:
        return None
    if passed is not computed_pass:
        return None
    result = {
        "name": name,
        "category": category,
        "passed": computed_pass,
        "failure_kind": failure_kind,
    }
    if (
        isinstance(feedback, dict)
        and feedback.get("source") == LLM_COMPLETED_SOURCE
    ):
        result["deterministic_completion"] = True
    return result


def _parse_fresh_utc_timestamp(value: object) -> datetime | None:
    text = _bounded_text(value, maximum=64)
    if text is None:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return None
    timestamp = timestamp.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if timestamp > now + timedelta(minutes=5) or now - timestamp > LLM_REPORT_MAX_AGE:
        return None
    return timestamp


def _llm_report_payload(
    report: object,
    *,
    project_root: Path,
    local_model_status: Mapping[str, object],
) -> dict[str, object]:
    contracts = llm_expected_case_contracts()
    raw_cases = report.get("cases") if isinstance(report, dict) else None
    raw_cases = raw_cases if isinstance(raw_cases, list) else []
    parsed_cases: list[dict[str, object]] = []
    names: list[str] = []
    for raw in raw_cases:
        name = raw.get("name") if isinstance(raw, dict) else None
        contract = contracts.get(name) if isinstance(name, str) else None
        case = (
            _llm_case_payload(raw, contract=contract)
            if contract is not None
            else None
        )
        if case is None:
            parsed_cases = []
            break
        names.append(str(case["name"]))
        parsed_cases.append(case)
    computed_categories = dict(Counter(
        str(case["category"]) for case in parsed_cases
    ))
    computed_passed = sum(case["passed"] is True for case in parsed_cases)
    computed_pass_rate = (
        round(100.0 * computed_passed / len(parsed_cases), 1)
        if parsed_cases
        else 0.0
    )
    reported_count = (
        _bounded_count(report.get("case_count"), maximum=LLM_CASE_LIMIT)
        if isinstance(report, dict)
        else None
    )
    reported_passed = (
        _bounded_count(report.get("passed"), maximum=LLM_CASE_LIMIT)
        if isinstance(report, dict)
        else None
    )
    reported_pass_rate = (
        _bounded_number(report.get("pass_rate_percent"), maximum=100.0)
        if isinstance(report, dict)
        else None
    )
    generated_at_text = (
        _bounded_text(report.get("generated_at"), maximum=64)
        if isinstance(report, dict)
        else None
    )
    generated_at = _parse_fresh_utc_timestamp(generated_at_text)
    model_digest = report.get("model_digest") if isinstance(report, dict) else None
    current_digest = local_model_status.get("digest")
    model_verified = bool(
        isinstance(model_digest, str)
        and SHA256_PATTERN.fullmatch(model_digest) is not None
        and isinstance(current_digest, str)
        and SHA256_PATTERN.fullmatch(current_digest) is not None
        and local_model_status.get("available") is True
        and local_model_status.get("installed") is True
        and local_model_status.get("model") in {
            "presentcoach-local", LLM_MODEL_NAME,
        }
        and compare_digest(model_digest, current_digest)
    )
    expected_artifacts = llm_evidence_artifacts(project_root)
    raw_artifacts = report.get("artifacts") if isinstance(report, dict) else None
    artifacts_verified = bool(
        isinstance(raw_artifacts, dict)
        and set(raw_artifacts) == LLM_ARTIFACT_KEYS
        and set(expected_artifacts) == LLM_ARTIFACT_KEYS
        and all(
            isinstance(raw_artifacts.get(key), str)
            and SHA256_PATTERN.fullmatch(str(raw_artifacts[key])) is not None
            and compare_digest(str(raw_artifacts[key]), expected_artifacts[key])
            for key in LLM_ARTIFACT_KEYS
        )
    )
    reported_cases_digest = (
        report.get("cases_sha256") if isinstance(report, dict) else None
    )
    try:
        computed_cases_digest = llm_cases_sha256(raw_cases)
    except (TypeError, ValueError):
        computed_cases_digest = None
    cases_verified = bool(
        len(raw_cases) == LLM_CASE_LIMIT
        and len(parsed_cases) == LLM_CASE_LIMIT
        and names == sorted(contracts)
        and len(set(names)) == LLM_CASE_LIMIT
        and computed_categories == LLM_CASE_CATEGORY_COUNTS
        and report.get("case_categories") == LLM_CASE_CATEGORY_COUNTS
        and isinstance(reported_cases_digest, str)
        and computed_cases_digest is not None
        and compare_digest(reported_cases_digest, computed_cases_digest)
    ) if isinstance(report, dict) else False
    trusted = bool(
        isinstance(report, dict)
        and report.get("schema_version") == LLM_REPORT_SCHEMA_VERSION
        and report.get("evaluation_only") is True
        and report.get("used_for_training") is False
        and report.get("model") == LLM_MODEL_NAME
        and generated_at is not None
        and model_verified
        and artifacts_verified
        and cases_verified
        and reported_count == LLM_CASE_LIMIT
        and reported_passed == computed_passed
        and reported_pass_rate == computed_pass_rate
    )
    if not trusted:
        parsed_cases = []
        computed_passed = 0
        computed_pass_rate = 0.0
    failed_cases = [
        case for case in parsed_cases if case["passed"] is False
    ]
    deterministic_completion_count = sum(
        case.get("deterministic_completion") is True
        for case in parsed_cases
    )
    return {
        "available": trusted,
        "trusted": trusted,
        "trust_message": (
            "Model, policy, schema, evaluator, timestamp, and all 30 case contracts verified."
            if trusted
            else "LLM results are hidden because the local report is stale or did not pass every evidence contract."
        ),
        "model": LLM_MODEL_NAME if trusted else None,
        "model_digest": model_digest if trusted else None,
        "model_verified": model_verified,
        "artifacts_verified": artifacts_verified,
        "cases_verified": cases_verified,
        "case_count": len(parsed_cases),
        "passed": computed_passed,
        "pass_rate_percent": computed_pass_rate,
        "deterministic_completion_count": deterministic_completion_count,
        "generated_at": generated_at_text if trusted else None,
        "evaluation_only": True,
        "used_for_training": False,
        "case_categories": (
            dict(LLM_CASE_CATEGORY_COUNTS)
            if trusted
            else {category: 0 for category in LLM_CASE_CATEGORY_COUNTS}
        ),
        "failed_cases": failed_cases,
    }


def _read_report(path: Path) -> dict[str, object] | None:
    try:
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _tracking_manifest_contract(path: Path) -> dict[str, object] | None:
    """Load the fixed manifest as a bounded UI-verification contract."""

    try:
        if not path.is_file() or path.stat().st_size > 256 * 1024:
            return None
        contents = path.read_bytes()
        document = json.loads(contents)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return None
    model = document.get("model")
    target_fps = _bounded_number(
        document.get("target_fps"), minimum=1.0, maximum=60.0
    )
    if not isinstance(model, dict) or target_fps is None:
        return None
    model_filename = model.get("filename")
    model_sha256 = model.get("sha256")
    if (
        not isinstance(model_filename, str)
        or Path(model_filename).name != model_filename
        or not isinstance(model_sha256, str)
        or SHA256_PATTERN.fullmatch(model_sha256) is None
    ):
        return None
    raw_clips = document.get("clips")
    if not isinstance(raw_clips, list) or len(raw_clips) != TRACKING_CASE_LIMIT:
        return None
    cases: dict[str, dict[str, object]] = {}
    for raw in raw_clips:
        if not isinstance(raw, dict):
            return None
        media_id = raw.get("id")
        role = raw.get("dataset_role")
        purpose = raw.get("purpose")
        license_name = raw.get("license")
        source_url = raw.get("source_url")
        media_sha256 = raw.get("sha256")
        filename = raw.get("filename")
        start_seconds = _bounded_number(
            raw.get("start_seconds"), maximum=1_800.0
        )
        duration_seconds = _bounded_number(
            raw.get("duration_seconds"), minimum=0.1, maximum=1_800.0
        )
        transform = _tracking_transform(raw.get("transform"))
        expectations = raw.get("expectations")
        if (
            not isinstance(media_id, str)
            or TRACKING_ID_PATTERN.fullmatch(media_id) is None
            or media_id in cases
            or role not in TRACKING_ROLES
            or _bounded_text(purpose, maximum=400) != purpose
            or _bounded_text(license_name, maximum=160) != license_name
            or _safe_source_url(source_url) != source_url
            or not isinstance(media_sha256, str)
            or SHA256_PATTERN.fullmatch(media_sha256) is None
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or start_seconds is None
            or duration_seconds is None
            or transform is None
            or not isinstance(expectations, dict)
            or not expectations
            or len(expectations) > 20
        ):
            return None
        parsed_expectations: dict[str, float] = {}
        for key, value in expectations.items():
            definition = TRACKING_CHECK_METRICS.get(key)
            number = (
                _bounded_number(value, maximum=definition[3])
                if definition is not None
                else None
            )
            if (
                not isinstance(key, str)
                or TRACKING_CHECK_ID_PATTERN.fullmatch(key) is None
                or key in REQUIRED_TRACKING_CHECKS
                or definition is None
                or number is None
            ):
                return None
            parsed_expectations[key] = number
        cases[media_id] = {
            "dataset_role": role,
            "purpose": purpose,
            "license": license_name,
            "source_url": source_url,
            "media_sha256": media_sha256,
            "filename": filename,
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "transform": transform,
            "expectations": parsed_expectations,
        }
    return {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "model_filename": model_filename,
        "model_sha256": model_sha256,
        "target_fps": target_fps,
        "cases": cases,
    }


def resolve_tracking_test_media(
    *, media_dir: Path, media_id: str
) -> dict[str, object] | None:
    """Resolve one manifest-allowlisted clip without exposing its local path.

    The returned mapping is an internal server contract. Callers must verify
    ``sha256`` before serving ``path``; API payloads must use only the bounded
    playback metadata assembled below.
    """

    if TRACKING_ID_PATTERN.fullmatch(media_id) is None:
        return None
    manifest = _tracking_manifest_contract(media_dir / "face_tracking_manifest.json")
    cases = manifest.get("cases") if manifest is not None else None
    manifest_case = cases.get(media_id) if isinstance(cases, dict) else None
    if not isinstance(manifest_case, dict):
        return None
    filename = manifest_case.get("filename")
    expected_sha256 = manifest_case.get("media_sha256")
    if not isinstance(filename, str) or not isinstance(expected_sha256, str):
        return None
    try:
        directory = media_dir.resolve(strict=True)
        path = (directory / filename).resolve(strict=True)
        path.relative_to(directory)
    except (OSError, ValueError):
        return None
    if not path.is_file():
        return None
    return {
        "path": path,
        "sha256": expected_sha256,
        "start_seconds": manifest_case["start_seconds"],
        "duration_seconds": manifest_case["duration_seconds"],
        "transform": manifest_case["transform"],
    }


def _tracking_media_payload(
    *, media_dir: Path, media_id: str, manifest_case: dict[str, object]
) -> dict[str, object]:
    """Return browser-safe playback metadata, never a local filename/path."""

    resolved = resolve_tracking_test_media(media_dir=media_dir, media_id=media_id)
    start = float(manifest_case["start_seconds"])
    duration = float(manifest_case["duration_seconds"])
    transform = manifest_case["transform"]
    exact = isinstance(transform, dict) and transform.get("type") == "identity"
    installed = resolved is not None
    return {
        "installed": installed,
        "video_url": (
            f"/api/test-media/{media_id}/video" if installed else None
        ),
        "playback_window": {
            "start_seconds": start,
            "duration_seconds": duration,
            "end_seconds": round(start + duration, 3),
        },
        "represents_exact_evaluation_input": exact,
        "representation": (
            "exact_source_excerpt"
            if exact
            else "source_excerpt_before_in_memory_transform"
        ),
        "integrity": "sha256_verified_on_request",
    }


def _bounded_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized[:maximum] if normalized else None


def _safe_source_url(value: object) -> str | None:
    url = _bounded_text(value, maximum=1000)
    if url is None:
        return None
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return url


def _bounded_number(
    value: object, *, minimum: float = 0.0, maximum: float
) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def _bounded_count(value: object, *, maximum: int = 1_000_000) -> int | None:
    number = _bounded_number(value, maximum=float(maximum))
    if number is None or not number.is_integer():
        return None
    return int(number)


def _tracking_transform(value: object) -> dict[str, object] | None:
    """Return the exact bounded transform contract used by the evaluator."""

    if not isinstance(value, dict):
        return None
    transform_type = value.get("type")
    if transform_type == "identity":
        return {"type": "identity"} if set(value) == {"type"} else None
    if transform_type != "duplicate_fixed_crop" or set(value) != {
        "type", "crop_normalized",
    }:
        return None
    crop = value.get("crop_normalized")
    if not isinstance(crop, list) or len(crop) != 4:
        return None
    parsed = [
        _bounded_number(item, maximum=1.0)
        for item in crop
    ]
    if any(item is None for item in parsed):
        return None
    left, top, right, bottom = (float(item) for item in parsed)
    if not (left < right and top < bottom):
        return None
    return {
        "type": "duplicate_fixed_crop",
        "crop_normalized": [left, top, right, bottom],
    }


def _invalid_track_interval(measurements: object) -> dict[str, object] | None:
    """Return only the longest bounded invalid interval, never raw landmarks."""

    if not isinstance(measurements, dict):
        return None
    raw_runs = measurements.get("dropout_runs")
    if not isinstance(raw_runs, list):
        return None
    candidates: list[dict[str, object]] = []
    for raw in raw_runs[:2_000]:
        if not isinstance(raw, dict) or raw.get("kind") not in TRACKING_INTERVAL_KINDS:
            continue
        start = _bounded_number(raw.get("start_seconds"), maximum=1_800.0)
        end = _bounded_number(raw.get("end_seconds"), maximum=1_800.0)
        duration = _bounded_number(raw.get("duration_seconds"), maximum=1_800.0)
        if start is None or end is None or duration is None or end < start:
            continue
        reacquisition = _bounded_number(
            raw.get("reacquisition_seconds"), maximum=1_800.0
        )
        candidates.append({
            "kind": raw["kind"],
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(duration, 3),
            "bracketed_by_valid_track": raw.get("bracketed_by_valid_track") is True,
            "reacquisition_seconds": (
                round(reacquisition, 3) if reacquisition is not None else None
            ),
        })
    return max(
        candidates,
        key=lambda item: float(item["duration_seconds"]),
        default=None,
    )


def _tracking_case_payload(
    raw: object,
    *,
    expected_repeatability_runs: int,
    manifest_case: dict[str, object],
    model_sha256: str,
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    media_id = raw.get("media_id")
    role = raw.get("dataset_role")
    purpose = raw.get("purpose")
    license_name = raw.get("license")
    source_url = raw.get("source_url")
    start_seconds = _bounded_number(
        raw.get("start_seconds"), maximum=1_800.0
    )
    evaluation_duration = _bounded_number(
        raw.get("evaluation_duration_seconds"),
        minimum=0.1,
        maximum=1_800.0,
    )
    transform = _tracking_transform(raw.get("transform"))
    if (
        not isinstance(media_id, str)
        or TRACKING_ID_PATTERN.fullmatch(media_id) is None
        or role != manifest_case.get("dataset_role")
        or purpose != manifest_case.get("purpose")
        or license_name != manifest_case.get("license")
        or source_url != manifest_case.get("source_url")
        or raw.get("filename") != manifest_case.get("filename")
        or start_seconds != manifest_case.get("start_seconds")
        or evaluation_duration != manifest_case.get("duration_seconds")
        or transform != manifest_case.get("transform")
    ):
        return None
    expectations = manifest_case.get("expectations")
    media_sha256 = manifest_case.get("media_sha256")
    if not isinstance(expectations, dict) or not isinstance(media_sha256, str):
        return None
    measurements = raw.get("measurements")
    if not isinstance(measurements, dict):
        return None
    if not isinstance(raw.get("passed"), bool):
        return None
    repeatability = raw.get("repeatability")
    if not isinstance(repeatability, dict):
        return None
    repeatability_runs = _bounded_count(repeatability.get("runs"), maximum=20)
    if repeatability_runs != expected_repeatability_runs:
        return None
    mismatches = repeatability.get("mismatches")
    if not isinstance(mismatches, list) or len(mismatches) > 1_000:
        return None
    mismatch_count = len(mismatches)
    if not isinstance(repeatability.get("passed"), bool):
        return None
    repeatability_passed = repeatability["passed"] is True and mismatch_count == 0
    checks = raw.get("checks")
    if not isinstance(checks, list) or not checks or len(checks) > 100:
        return None
    check_states: dict[str, bool] = {}
    declared_checks: list[dict[str, object]] = []
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("passed"), bool):
            return None
        check_id = check.get("id")
        if (
            not isinstance(check_id, str)
            or TRACKING_CHECK_ID_PATTERN.fullmatch(check_id) is None
            or check_id in check_states
        ):
            return None
        check_states[check_id] = check["passed"] is True
        if check_id not in REQUIRED_TRACKING_CHECKS:
            label = _bounded_text(check.get("label"), maximum=120)
            expected = _bounded_text(check.get("expected"), maximum=120)
            if label is None or expected is None:
                return None
            declared_checks.append({
                "id": check_id,
                "label": label,
                "passed": check["passed"] is True,
                "expected": expected,
            })
    expected_check_ids = REQUIRED_TRACKING_CHECKS | set(expectations)
    if set(check_states) != expected_check_ids:
        return None
    by_id = {
        check["id"]: check
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("id"), str)
    }
    media_check = by_id["exact_media_sha256"]
    model_check = by_id["exact_model_sha256"]
    repeatability_check = by_id["exact_repeatability"]
    if (
        media_check.get("passed") is not True
        or media_check.get("actual") != media_sha256
        or media_check.get("expected") != media_sha256
        or model_check.get("passed") is not True
        or model_check.get("actual") != model_sha256
        or model_check.get("expected") != model_sha256
        or repeatability_check.get("passed") is not repeatability_passed
        or repeatability_check.get("actual") != mismatches
        or repeatability_check.get("expected")
        != "no deterministic metric differences"
    ):
        return None
    declared_by_id = {check["id"]: check for check in declared_checks}
    for check_id, threshold in expectations.items():
        definition = TRACKING_CHECK_METRICS.get(check_id)
        declared = declared_by_id.get(check_id)
        if definition is None or declared is None:
            return None
        metric_name, comparison, expected_label, metric_maximum = definition
        raw_metric = measurements.get(metric_name)
        metric_number = (
            None
            if raw_metric is None
            else _bounded_number(raw_metric, maximum=metric_maximum)
        )
        raw_actual = by_id[check_id].get("actual")
        actual_number = (
            None
            if raw_actual is None
            else _bounded_number(raw_actual, maximum=metric_maximum)
        )
        operator = ">=" if comparison == "min" else "<="
        computed_pass = bool(
            metric_number is not None
            and (
                metric_number >= float(threshold)
                if comparison == "min"
                else metric_number <= float(threshold)
            )
        )
        if (
            not isinstance(threshold, (int, float))
            or metric_name not in measurements
            or raw_metric is not None and metric_number is None
            or raw_actual is not None and actual_number is None
            or actual_number != metric_number
            or by_id[check_id].get("passed") is not computed_pass
            or declared["label"] != expected_label
            or declared["expected"] != f"{operator} {float(threshold):g}"
        ):
            return None
    valid_ratio = _bounded_number(
        measurements.get("valid_face_frame_ratio"), maximum=1.0
    )
    multi_face_ratio = _bounded_number(
        measurements.get("multi_face_frame_ratio"), maximum=1.0
    )
    dropout_count = _bounded_count(measurements.get("dropout_run_count"))
    analyzed_fps = _bounded_number(measurements.get("analyzed_fps"), maximum=120.0)
    raw_jitter = measurements.get("landmark_jitter_p95")
    landmark_jitter = _bounded_number(raw_jitter, maximum=100.0)
    interval = _invalid_track_interval(measurements)
    if (
        valid_ratio is None
        or multi_face_ratio is None
        or dropout_count is None
        or analyzed_fps is None
        or (raw_jitter is not None and landmark_jitter is None)
        or (dropout_count > 0 and interval is None)
    ):
        return None
    case_passed = repeatability_passed and all(check_states.values())
    if raw["passed"] is not case_passed:
        return None
    return {
        "id": media_id,
        "title": media_id.replace("-", " ").title()[:100],
        "dataset_role": role,
        "purpose": purpose,
        "license": license_name,
        "source_url": source_url,
        "passed": case_passed,
        "valid_face_frame_ratio": valid_ratio,
        "multi_face_frame_ratio": multi_face_ratio,
        "dropout_run_count": dropout_count,
        "observed_invalid_track_interval": interval,
        "landmark_jitter_p95": landmark_jitter,
        "analyzed_fps": analyzed_fps,
        "repeatability": {
            "runs": repeatability_runs,
            "passed": repeatability_passed,
            "mismatch_count": mismatch_count,
        },
        "declared_checks": declared_checks[:12],
    }


def _tracking_report_payload(
    report: object, *, manifest_path: Path
) -> dict[str, object]:
    manifest = _tracking_manifest_contract(manifest_path)
    manifest_digest = manifest.get("sha256") if manifest is not None else None
    manifest_cases = manifest.get("cases", {}) if manifest is not None else {}
    if not isinstance(manifest_cases, dict):
        manifest_cases = {}
    manifest_model_sha256 = (
        manifest.get("model_sha256") if manifest is not None else None
    )
    raw_cases = report.get("cases", []) if isinstance(report, dict) else []
    if not isinstance(raw_cases, list):
        raw_cases = []
    repeatability_runs = (
        _bounded_count(report.get("repeatability_runs_per_case"), maximum=20)
        if isinstance(report, dict)
        else None
    )
    cases: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_cases[:100]:
        raw_media_id = raw.get("media_id") if isinstance(raw, dict) else None
        manifest_case = manifest_cases.get(raw_media_id)
        case = (
            _tracking_case_payload(
                raw,
                expected_repeatability_runs=repeatability_runs,
                manifest_case=manifest_case,
                model_sha256=manifest_model_sha256,
            )
            if (
                repeatability_runs is not None
                and isinstance(manifest_case, dict)
                and isinstance(manifest_model_sha256, str)
            )
            else None
        )
        if case is None or str(case["id"]) in seen:
            continue
        seen.add(str(case["id"]))
        cases.append(case)
        if len(cases) == TRACKING_CASE_LIMIT:
            break
    passed = sum(case["passed"] is True for case in cases)
    roles = {
        role: sum(case["dataset_role"] == role for case in cases)
        for role in sorted(TRACKING_ROLES)
    }
    raw_roles = report.get("dataset_roles", {}) if isinstance(report, dict) else {}
    reported_roles = (
        {
            role: _bounded_count(raw_roles.get(role), maximum=TRACKING_CASE_LIMIT)
            for role in sorted(TRACKING_ROLES)
        }
        if isinstance(raw_roles, dict) and set(raw_roles) == set(TRACKING_ROLES)
        else None
    )
    reported_count = (
        _bounded_count(report.get("case_count"), maximum=10_000)
        if isinstance(report, dict)
        else None
    )
    generated_at = (
        _bounded_text(report.get("generated_at"), maximum=64)
        if isinstance(report, dict)
        else None
    )
    model = report.get("model", {}) if isinstance(report, dict) else {}
    if not isinstance(model, dict):
        model = {}
    reported_manifest_digest = (
        report.get("manifest_sha256") if isinstance(report, dict) else None
    )
    reported_passed = (
        _bounded_count(report.get("passed"), maximum=TRACKING_CASE_LIMIT)
        if isinstance(report, dict)
        else None
    )
    reported_pass_rate = (
        _bounded_number(report.get("pass_rate_percent"), maximum=100.0)
        if isinstance(report, dict)
        else None
    )
    computed_pass_rate = round(100.0 * passed / len(cases), 1) if cases else 0.0
    privacy_contract = bool(
        isinstance(report, dict)
        and report.get("schema_version") == 1
        and report.get("evaluation_only") is True
        and report.get("used_for_training") is False
        and report.get("identity_recognition") is False
        and report.get("landmarks_persisted") is False
        and report.get("partial_run") is False
    )
    trusted = bool(
        privacy_contract
        and repeatability_runs is not None
        and repeatability_runs >= 2
        and manifest is not None
        and model.get("matches_manifest") is True
        and model.get("filename") == manifest.get("model_filename")
        and model.get("sha256") == manifest_model_sha256
        and isinstance(reported_manifest_digest, str)
        and manifest_digest is not None
        and compare_digest(reported_manifest_digest, manifest_digest)
        and report.get("target_fps") == manifest.get("target_fps")
        and reported_count == TRACKING_CASE_LIMIT
        and len(raw_cases) == TRACKING_CASE_LIMIT
        and len(cases) == TRACKING_CASE_LIMIT
        and seen == set(manifest_cases)
        and all(case["repeatability"]["passed"] is True for case in cases)
        and reported_roles == roles
        and reported_passed == passed
        and reported_pass_rate == computed_pass_rate
    )
    if not trusted:
        cases = []
        passed = 0
        roles = {role: 0 for role in sorted(TRACKING_ROLES)}
        computed_pass_rate = 0.0
    else:
        for case in cases:
            media_id = str(case["id"])
            manifest_case = manifest_cases.get(media_id)
            if isinstance(manifest_case, dict):
                case["media"] = _tracking_media_payload(
                    media_dir=manifest_path.parent,
                    media_id=media_id,
                    manifest_case=manifest_case,
                )
    return {
        "available": trusted,
        "trusted": trusted,
        "trust_message": (
            "Manifest, privacy, model, repeatability, and case contracts verified."
            if trusted
            else "Tracking results are hidden because the local report did not pass every verification contract."
        ),
        "case_count": len(cases),
        "passed": passed,
        "pass_rate_percent": computed_pass_rate,
        "generated_at": generated_at,
        "partial_run": report.get("partial_run") is True if isinstance(report, dict) else False,
        "repeatability_runs_per_case": repeatability_runs,
        "target_fps": (
            _bounded_number(report.get("target_fps"), maximum=120.0)
            if isinstance(report, dict)
            else None
        ),
        "model_verified": bool(
            manifest is not None
            and model.get("matches_manifest") is True
            and model.get("filename") == manifest.get("model_filename")
            and model.get("sha256") == manifest_model_sha256
        ),
        "manifest_verified": bool(
            isinstance(reported_manifest_digest, str)
            and manifest_digest is not None
            and compare_digest(reported_manifest_digest, manifest_digest)
        ),
        "evaluation_only": True,
        "used_for_training": False,
        "identity_recognition": False,
        "face_embeddings": False,
        "landmarks_persisted": False,
        "dataset_roles": roles,
        "cases": cases,
    }


def _stream_file_sha256(
    path: Path, *, maximum_bytes: int = 1024 * 1024 * 1024
) -> str | None:
    """Hash one ordinary bounded file without following a manifest path escape."""

    try:
        if not path.is_file() or path.is_symlink():
            return None
        size = path.stat().st_size
        if size <= 0 or size > maximum_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _dynamic_canonical_sha256(value: object) -> str | None:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _dynamic_power_policy_document(policy: object) -> dict[str, object]:
    return {
        "minimum_trackable_frames": policy.minimum_trackable_frames,
        "minimum_nontrackable_frames": policy.minimum_nontrackable_frames,
        "minimum_multi_face_frames": policy.minimum_multi_face_frames,
        "minimum_trackable_seconds": float(policy.minimum_trackable_seconds),
        "minimum_reacquisition_events": policy.minimum_reacquisition_events,
        "minimum_sampling_coverage_ratio": float(
            policy.minimum_sampling_coverage_ratio
        ),
    }


def _read_dynamic_report(path: Path) -> tuple[bool, dict[str, object] | None]:
    """Return report presence separately from validity so absence is not a failure."""

    if not path.exists():
        return False, None
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or not 0 < path.stat().st_size <= 4 * 1024 * 1024
        ):
            return True, None

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate report field")
                value[key] = item
            return value

        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return True, None
    return True, document if isinstance(document, dict) else None


def _dynamic_source_hashes(project_root: Path) -> dict[str, str] | None:
    package = project_root / "src" / "stroke_screening"
    paths = {
        "presentation_dynamic_eval.py": package / "presentation_dynamic_eval.py",
        "presentation_tracking_eval.py": package / "presentation_tracking_eval.py",
        "presentation_tracking_ground_truth.py": package
        / "presentation_tracking_ground_truth.py",
        "landmarks.py": package / "landmarks.py",
        "presentation_video_timing.py": package / "presentation_video_timing.py",
        "scripts/evaluate_dynamic_presentations.py": project_root
        / "scripts"
        / "evaluate_dynamic_presentations.py",
    }
    result: dict[str, str] = {}
    for name in DYNAMIC_SOURCE_FILES:
        digest = _stream_file_sha256(paths[name], maximum_bytes=2 * 1024 * 1024)
        if digest is None:
            return None
        result[name] = digest
    return result


def _dynamic_expected_configuration(target_fps: float) -> dict[str, object]:
    return {
        "target_fps": target_fps,
        "start_seconds": 0.0,
        "transform": {"type": "identity"},
        "sampler": "presentation_tracking_eval.sample_tracking_clip",
        "development_repeatability_runs": 2,
        "holdout_repeatability_runs": 1,
        "repeatability_projection": (
            "all aggregate model outputs excluding runtime-only metrics"
        ),
        "face_landmarker": {
            "max_face_count": MAX_FACE_COUNT,
            "min_face_detection_confidence": FACE_DETECTION_CONFIDENCE,
            "min_face_presence_confidence": FACE_PRESENCE_CONFIDENCE,
            "min_tracking_confidence": FACE_TRACKING_CONFIDENCE,
            "landmark_smoothing_time_constant_ms": (
                LANDMARK_SMOOTHING_TIME_CONSTANT_MS
            ),
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


def _dynamic_manifest_items(manifest: object) -> dict[str, Mapping[str, object]]:
    clips = manifest.document.get("clips")
    if not isinstance(clips, list):
        raise ValueError("dynamic manifest clips are unavailable")
    result: dict[str, Mapping[str, object]] = {}
    for item in clips:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("dynamic manifest clip metadata is malformed")
        result[str(item["id"])] = item
    return result


def resolve_dynamic_test_media(
    *, media_dir: Path, media_id: str
) -> dict[str, object] | None:
    """Resolve only a manifest-listed canonical MP4; never return it to JSON APIs."""

    if TRACKING_ID_PATTERN.fullmatch(media_id) is None:
        return None
    try:
        manifest = load_dynamic_manifest(
            media_dir / "dynamic_presentation_manifest.json"
        )
    except DynamicEvaluationError:
        return None
    spec = next((item for item in manifest.clips if item.clip_id == media_id), None)
    if spec is None:
        return None
    try:
        directory = media_dir.resolve(strict=True)
        path = (directory / spec.filename).resolve(strict=True)
        path.relative_to(directory)
    except (OSError, ValueError):
        return None
    if not path.is_file() or path.is_symlink() or path.suffix.lower() != ".mp4":
        return None
    return {
        "path": path,
        "sha256": spec.media_sha256,
        "mimetype": "video/mp4",
    }


def _dynamic_media_payload(*, media_dir: Path, media_id: str) -> dict[str, object]:
    contract = resolve_dynamic_test_media(media_dir=media_dir, media_id=media_id)
    installed = False
    if contract is not None:
        digest = _stream_file_sha256(
            contract["path"], maximum_bytes=1024 * 1024 * 1024
        )
        installed = bool(
            digest is not None
            and compare_digest(digest, str(contract["sha256"]))
        )
    return {
        "installed": installed,
        "video_url": f"/api/test-media/{media_id}/video" if installed else None,
        "playback_window": {
            "start_seconds": 0.0,
            "duration_seconds": 60.0,
            "end_seconds": 60.0,
        },
        "represents_exact_evaluation_input": True,
        "representation": "canonical_manifest_locked_mp4",
        "integrity": "sha256_verified_on_request",
    }


def _dynamic_case_metadata(
    *, spec: object, raw: Mapping[str, object], media_dir: Path
) -> dict[str, object]:
    source = raw.get("source")
    attribution = raw.get("attribution")
    if not isinstance(source, dict) or not isinstance(attribution, dict):
        raise ValueError("dynamic source metadata is malformed")
    source_url = _safe_source_url(source.get("page_url"))
    license_url = _safe_source_url(attribution.get("license_url"))
    title = _bounded_text(source.get("title"), maximum=240)
    creator = _bounded_text(attribution.get("creator"), maximum=180)
    license_name = _bounded_text(attribution.get("license_name"), maximum=180)
    credit_line = _bounded_text(attribution.get("credit_line"), maximum=500)
    change_notice = _bounded_text(attribution.get("change_notice"), maximum=700)
    if None in {
        source_url, license_url, title, creator, license_name, credit_line,
        change_notice,
    }:
        raise ValueError("dynamic source metadata is unsafe")
    return {
        "id": spec.clip_id,
        "title": title,
        "dataset_role": spec.dataset_role,
        "source_url": source_url,
        "creator": creator,
        "license": license_name,
        "license_spdx": spec.license_spdx,
        "license_url": license_url,
        "credit_line": credit_line,
        "change_notice": change_notice,
        "purpose": (
            "Canonical 60-second moving-speaker presentation excerpt for "
            "anonymous observed-state tracking checks."
        ),
        "media": _dynamic_media_payload(
            media_dir=media_dir, media_id=spec.clip_id
        ),
        "result": None,
    }


def _dynamic_validate_tracking_metrics(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != DYNAMIC_TRACKING_METRIC_KEYS:
        raise ValueError("dynamic tracking metrics do not match the schema")
    count_names = {
        "analyzed_frame_count", "valid_face_frame_count", "no_face_frame_count",
        "multi_face_frame_count", "inconsistent_detection_count",
        "multi_face_abstention_violations", "dropout_run_count",
        "reacquisition_event_count", "stable_segment_count",
        "stable_transition_count",
    }
    counts = {
        name: _bounded_count(value.get(name), maximum=108_000)
        for name in count_names
    }
    if any(number is None for number in counts.values()) or not counts["analyzed_frame_count"]:
        raise ValueError("dynamic tracking counts are malformed")
    total = int(counts["analyzed_frame_count"] or 0)
    ratio_pairs = {
        "valid_face_frame_ratio": "valid_face_frame_count",
        "no_face_frame_ratio": "no_face_frame_count",
        "multi_face_frame_ratio": "multi_face_frame_count",
    }
    for ratio_name, count_name in ratio_pairs.items():
        ratio = _bounded_number(value.get(ratio_name), maximum=1.0)
        expected = round(float(counts[count_name]) / total, 6)
        if ratio is None or ratio != expected:
            raise ValueError("dynamic tracking ratio is inconsistent")
    runs = value.get("dropout_runs")
    if not isinstance(runs, list) or len(runs) != counts["dropout_run_count"]:
        raise ValueError("dynamic dropout runs are inconsistent")
    reacquisition_values: list[float] = []
    longest = 0.0
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "kind", "start_seconds", "end_seconds", "duration_seconds",
            "frame_count", "bracketed_by_valid_track", "reacquisition_seconds",
        }:
            raise ValueError("dynamic dropout run is malformed")
        if run.get("kind") not in TRACKING_INTERVAL_KINDS:
            raise ValueError("dynamic dropout kind is unsupported")
        start = _bounded_number(run.get("start_seconds"), maximum=1_800.0)
        end = _bounded_number(run.get("end_seconds"), maximum=1_800.0)
        duration = _bounded_number(run.get("duration_seconds"), maximum=1_800.0)
        frames = _bounded_count(run.get("frame_count"), maximum=108_000)
        if (
            start is None or end is None or duration is None or frames is None
            or end < start or not isinstance(run.get("bracketed_by_valid_track"), bool)
        ):
            raise ValueError("dynamic dropout interval is malformed")
        longest = max(longest, duration)
        raw_reacquisition = run.get("reacquisition_seconds")
        if raw_reacquisition is not None:
            reacquisition = _bounded_number(raw_reacquisition, maximum=1_800.0)
            if reacquisition is None or run.get("bracketed_by_valid_track") is not True:
                raise ValueError("dynamic reacquisition interval is malformed")
            reacquisition_values.append(reacquisition)
    if (
        value.get("longest_dropout_seconds") != round(longest, 4)
        or counts["reacquisition_event_count"] != len(reacquisition_values)
    ):
        raise ValueError("dynamic dropout summary is inconsistent")
    expected_mean = (
        None if not reacquisition_values
        else round(sum(reacquisition_values) / len(reacquisition_values), 4)
    )
    expected_max = (
        None if not reacquisition_values else round(max(reacquisition_values), 4)
    )
    if (
        value.get("mean_reacquisition_seconds") != expected_mean
        or value.get("max_reacquisition_seconds") != expected_max
    ):
        raise ValueError("dynamic reacquisition summary is inconsistent")
    for name, maximum in {
        "analyzed_fps": 120.0,
        "longest_dropout_seconds": 1_800.0,
        "wall_clock_seconds": 100_000.0,
        "processing_throughput_fps": 10_000.0,
    }.items():
        if _bounded_number(value.get(name), maximum=maximum) is None:
            raise ValueError(f"dynamic metric {name} is malformed")
    for name, maximum in {
        "landmark_jitter_median": 100.0,
        "landmark_jitter_p95": 100.0,
        "mean_inference_ms": 100_000.0,
        "p95_inference_ms": 100_000.0,
    }.items():
        if value.get(name) is not None and _bounded_number(
            value.get(name), maximum=maximum
        ) is None:
            raise ValueError(f"dynamic metric {name} is malformed")
    return {
        "analyzed_frame_count": total,
        "analyzed_fps": float(value["analyzed_fps"]),
        "processing_throughput_fps": float(value["processing_throughput_fps"]),
        "valid_face_frame_ratio": float(value["valid_face_frame_ratio"]),
        "dropout_run_count": int(counts["dropout_run_count"] or 0),
    }


def _dynamic_validate_ground_truth(
    value: object,
    *,
    spec: object,
    annotation: object,
    manifest: object,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != DYNAMIC_GROUND_TRUTH_KEYS:
        raise ValueError("dynamic ground-truth metrics do not match the schema")
    if (
        value.get("schema_version") != 1
        or value.get("clip_id") != spec.clip_id
        or value.get("split_group_id") != spec.split_group_id
        or value.get("target_fps") != manifest.target_fps
        or value.get("explicit_gaze_labels") is not False
        or value.get("gaze_accuracy") is not None
        or value.get("power_policy")
        != _dynamic_power_policy_document(manifest.power_policy)
    ):
        raise ValueError("dynamic ground-truth identity or policy is inconsistent")
    count_names = {
        "observation_count", "labeled_observation_count",
        "unlabeled_observation_count", "uncertain_observation_count",
        "trackable_observation_count", "trackable_valid_observation_count",
        "nontrackable_observation_count", "false_valid_observation_count",
        "multi_face_observation_count", "multi_face_abstained_observation_count",
        "dropout_run_count", "reacquisition_event_count",
        "reacquisition_success_count",
    }
    counts = {
        name: _bounded_count(value.get(name), maximum=108_000)
        for name in count_names
    }
    if any(number is None for number in counts.values()):
        raise ValueError("dynamic ground-truth counts are malformed")
    if (
        counts["labeled_observation_count"]
        + counts["unlabeled_observation_count"]
        != counts["observation_count"]
        or counts["trackable_valid_observation_count"]
        > counts["trackable_observation_count"]
        or counts["false_valid_observation_count"]
        > counts["nontrackable_observation_count"]
        or counts["multi_face_abstained_observation_count"]
        > counts["multi_face_observation_count"]
        or counts["reacquisition_success_count"]
        > counts["reacquisition_event_count"]
    ):
        raise ValueError("dynamic ground-truth counts are inconsistent")
    motion_counts = value.get("motion_observation_counts")
    if not isinstance(motion_counts, dict) or set(motion_counts) != {
        "head_turn", "mixed", "scale", "static", "translate",
    } or any(_bounded_count(item, maximum=108_000) is None for item in motion_counts.values()):
        raise ValueError("dynamic motion counts are malformed")
    events = value.get("reacquisition_events")
    if not isinstance(events, list) or len(events) != counts["reacquisition_event_count"]:
        raise ValueError("dynamic reacquisition events are malformed")
    success_count = 0
    for event in events:
        if not isinstance(event, dict) or set(event) != {
            "onset_seconds", "reacquired", "latency_seconds",
        }:
            raise ValueError("dynamic reacquisition event is malformed")
        if _bounded_number(event.get("onset_seconds"), maximum=1_800.0) is None:
            raise ValueError("dynamic reacquisition onset is malformed")
        reacquired = event.get("reacquired")
        latency = event.get("latency_seconds")
        if not isinstance(reacquired, bool) or (
            reacquired and _bounded_number(latency, maximum=1_800.0) is None
        ) or (not reacquired and latency is not None):
            raise ValueError("dynamic reacquisition result is malformed")
        success_count += reacquired
    if success_count != counts["reacquisition_success_count"]:
        raise ValueError("dynamic reacquisition success count is inconsistent")
    reasons = value.get("underpowered_reasons")
    if (
        not isinstance(reasons, dict)
        or any(
            not isinstance(key, str)
            or _bounded_text(reason, maximum=600) != reason
            for key, reason in reasons.items()
        )
    ):
        raise ValueError("dynamic underpowered reasons are malformed")
    single = aggregate_ground_truth_metrics(
        [value],
        annotations=[annotation],
        target_fps=manifest.target_fps,
        power_policy=manifest.power_policy,
    )
    direct_map = {
        "observation_count": "observation_count",
        "labeled_observation_count": "labeled_observation_count",
        "trackable_valid_observation_count": "trackable_valid_observation_count",
        "trackable_observation_count": "trackable_observation_count",
        "false_valid_observation_count": "nontrackable_false_valid_observation_count",
        "nontrackable_observation_count": "nontrackable_observation_count",
        "multi_face_abstained_observation_count": "multi_face_abstained_observation_count",
        "multi_face_observation_count": "multi_face_observation_count",
        "dropout_run_count": "dropout_run_count",
        "annotated_trackable_seconds": "annotated_trackable_seconds",
        "trackable_sampling_coverage": "trackable_sampling_coverage",
        "nontrackable_sampling_coverage": "nontrackable_sampling_coverage",
        "multi_face_sampling_coverage": "multi_face_sampling_coverage",
        "trackable_coverage": "trackable_coverage",
        "false_valid_rate": "false_valid_rate",
        "multi_face_abstention": "multi_face_abstention",
        "dropouts_per_annotated_minute": "dropouts_per_annotated_minute",
        "reacquisition_success_count": "reacquisition_success_count",
        "reacquisition_event_count": "reacquisition_event_count",
        "reacquisition_success_rate": "reacquisition_success_rate",
        "mean_reacquisition_latency_seconds": "mean_reacquisition_latency_seconds",
        "max_reacquisition_latency_seconds": "max_reacquisition_latency_seconds",
        "gaze_accuracy": "gaze_accuracy",
    }
    if any(value.get(raw) != single.get(computed) for raw, computed in direct_map.items()):
        raise ValueError("dynamic per-case ground-truth summary is inconsistent")
    return {
        "observation_count": counts["observation_count"],
        "trackable_coverage": value.get("trackable_coverage"),
        "false_valid_rate": value.get("false_valid_rate"),
        "multi_face_abstention": value.get("multi_face_abstention"),
        "dropout_run_count": counts["dropout_run_count"],
        "reacquisition_event_count": counts["reacquisition_event_count"],
        "reacquisition_success_rate": value.get("reacquisition_success_rate"),
    }


def _dynamic_report_payload(
    report: object,
    *,
    report_present: bool,
    split: str,
    manifest_path: Path,
    project_root: Path,
) -> dict[str, object]:
    """Verify one isolated split report and expose only bounded aggregate evidence."""

    if split not in DYNAMIC_SPLITS:
        raise ValueError("unsupported dynamic split")
    media_dir = manifest_path.parent
    manifest = None
    metadata_cases: list[dict[str, object]] = []
    manifest_digest: str | None = None
    model_verified = False
    annotations_verified = False
    try:
        manifest = load_dynamic_manifest(manifest_path)
        manifest_digest = _stream_file_sha256(
            manifest_path, maximum_bytes=1024 * 1024
        )
        raw_items = _dynamic_manifest_items(manifest)
        selected_specs = tuple(
            item for item in manifest.clips if item.dataset_role == split
        )
        metadata_cases = [
            _dynamic_case_metadata(
                spec=spec, raw=raw_items[spec.clip_id], media_dir=media_dir
            )
            for spec in selected_specs
        ]
        model_digest = _stream_file_sha256(
            project_root / "models" / manifest.model_filename,
            maximum_bytes=128 * 1024 * 1024,
        )
        proposal_model_digest = _stream_file_sha256(
            project_root / "models" / YUNET_MODEL_FILENAME,
            maximum_bytes=128 * 1024 * 1024,
        )
        model_verified = bool(
            model_digest is not None
            and compare_digest(model_digest, manifest.model_sha256)
            and proposal_model_digest is not None
            and compare_digest(proposal_model_digest, YUNET_MODEL_SHA256)
        )
        annotation_digests = [
            _stream_file_sha256(
                media_dir / spec.annotation_file,
                maximum_bytes=2 * 1024 * 1024,
            )
            for spec in selected_specs
        ]
        annotations_verified = all(
            digest is not None and compare_digest(digest, spec.annotation_sha256)
            for digest, spec in zip(annotation_digests, selected_specs)
        )
    except (DynamicEvaluationError, ValueError):
        selected_specs = ()
        raw_items = {}

    absent_state = "untouched" if split == "model_output_holdout" else "not_run"
    if not report_present:
        return {
            "state": absent_state if manifest is not None else "manifest_unavailable",
            "available": False,
            "trusted": False,
            "trust_message": (
                "Untouched model-output holdout—no report has been run or read."
                if split == "model_output_holdout" and manifest is not None
                else "Development benchmark has not been run."
                if manifest is not None
                else "The moving-presenter manifest is unavailable or invalid."
            ),
            "split": split,
            "expected_case_count": len(selected_specs),
            "case_count": 0,
            "manifest_verified": manifest is not None and manifest_digest is not None,
            "model_verified": model_verified,
            "annotations_verified": annotations_verified,
            "release_gate_status": "not_run",
            "underpowered_required_metrics": [],
            "failed_required_metrics": [],
            "aggregate": {},
            "checks": [],
            "cases": metadata_cases,
        }

    trusted = False
    parsed_results: dict[str, dict[str, object]] = {}
    aggregate_ui: dict[str, object] = {}
    checks_ui: list[dict[str, object]] = []
    release_status = "untrusted"
    underpowered: list[str] = []
    failed: list[str] = []
    generated_at: str | None = None
    try:
        if manifest is None or not isinstance(report, dict):
            raise ValueError("dynamic report or manifest is invalid")
        if set(report) != DYNAMIC_REPORT_KEYS or report.get("schema_version") != 1:
            raise ValueError("dynamic report schema is invalid")
        if (
            report.get("split") != split
            or report.get("split_isolation")
            != "one report contains exactly one dataset_role"
            or report.get("evaluation_only") is not True
            or report.get("used_for_training") is not False
            or report.get("identity_recognition") is not False
            or report.get("face_embeddings_created") is not False
            or report.get("landmarks_persisted") is not False
            or report.get("gaze_accuracy_evaluated") is not False
            or report.get("annotation_evidence")
            != "single-curator coarse observed-state intervals; not independently adjudicated accuracy"
        ):
            raise ValueError("dynamic report split or privacy contract is invalid")
        generated_at = _bounded_text(report.get("generated_at"), maximum=64)
        if generated_at is None:
            raise ValueError("dynamic report timestamp is invalid")
        timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("dynamic report timestamp lacks a timezone")
        expected_count = len(selected_specs)
        if _bounded_count(report.get("case_count"), maximum=50) != expected_count:
            raise ValueError("dynamic report case count is invalid")
        if (
            not isinstance(report.get("manifest_sha256"), str)
            or manifest_digest is None
            or not compare_digest(str(report["manifest_sha256"]), manifest_digest)
        ):
            raise ValueError("dynamic report manifest digest is invalid")
        model = report.get("model")
        if not isinstance(model, dict) or set(model) != {
            "filename", "sha256", "matches_manifest",
        } or model != {
            "filename": manifest.model_filename,
            "sha256": manifest.model_sha256,
            "matches_manifest": True,
        } or not model_verified:
            raise ValueError("dynamic report model digest is invalid")
        expected_configuration = _dynamic_expected_configuration(manifest.target_fps)
        if (
            report.get("configuration") != expected_configuration
            or report.get("configuration_sha256")
            != _dynamic_canonical_sha256(expected_configuration)
        ):
            raise ValueError("dynamic report configuration digest is invalid")
        expected_policy = {
            "power_policy": _dynamic_power_policy_document(manifest.power_policy),
            "scoring_policy": dict(manifest.scoring_policy),
        }
        if (
            report.get("policy") != expected_policy
            or report.get("policy_sha256")
            != _dynamic_canonical_sha256(expected_policy)
        ):
            raise ValueError("dynamic report policy digest is invalid")
        expected_sources = _dynamic_source_hashes(project_root)
        if expected_sources is None or report.get("source_sha256") != expected_sources:
            raise ValueError("dynamic report source digests are stale")
        git = report.get("git")
        if not isinstance(git, dict) or set(git) != {
            "commit", "dirty", "status_sha256",
        }:
            raise ValueError("dynamic report git evidence is malformed")
        if git.get("dirty") is not None and not isinstance(git.get("dirty"), bool):
            raise ValueError("dynamic report dirty state is malformed")
        for key, length in (("commit", 40), ("status_sha256", 64)):
            value = git.get(key)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != length
                or re.fullmatch(r"[0-9a-f]+", value) is None
            ):
                raise ValueError("dynamic report git digest is malformed")
        raw_cases = report.get("cases")
        if not isinstance(raw_cases, list) or len(raw_cases) != expected_count:
            raise ValueError("dynamic report case set is invalid")
        annotation_objects = []
        ground_truth_objects = []
        for raw_case, spec in zip(raw_cases, selected_specs):
            if not isinstance(raw_case, dict) or set(raw_case) != DYNAMIC_CASE_KEYS:
                raise ValueError("dynamic report case schema is invalid")
            raw_item = raw_items[spec.clip_id]
            if (
                raw_case.get("clip_id") != spec.clip_id
                or raw_case.get("dataset_role") != split
                or raw_case.get("split_group_id") != spec.split_group_id
                or raw_case.get("filename") != spec.filename
                or raw_case.get("media_sha256") != spec.media_sha256
                or raw_case.get("annotation_file") != spec.annotation_file
                or raw_case.get("annotation_sha256") != spec.annotation_sha256
                or raw_case.get("source_title") != spec.source_title
                or raw_case.get("source_page_url") != spec.source_page_url
                or raw_case.get("license_spdx") != spec.license_spdx
            ):
                raise ValueError("dynamic report case does not match the manifest")
            candidate_media = media_dir / spec.filename
            if candidate_media.exists():
                media_contract = resolve_dynamic_test_media(
                    media_dir=media_dir, media_id=spec.clip_id
                )
                installed_digest = (
                    _stream_file_sha256(
                        media_contract["path"],
                        maximum_bytes=1024 * 1024 * 1024,
                    )
                    if media_contract is not None
                    else None
                )
                if installed_digest is None or not compare_digest(
                    installed_digest, spec.media_sha256
                ):
                    raise ValueError("installed dynamic media digest is invalid")
            duration = _bounded_number(
                raw_case.get("evaluation_duration_seconds"),
                minimum=max(0.1, spec.duration_seconds - 1.0),
                maximum=spec.duration_seconds,
            )
            if duration is None:
                raise ValueError("dynamic report duration is invalid")
            annotation_path = media_dir / spec.annotation_file
            annotation_digest = _stream_file_sha256(
                annotation_path, maximum_bytes=2 * 1024 * 1024
            )
            if annotation_digest is None or not compare_digest(
                annotation_digest, spec.annotation_sha256
            ):
                raise ValueError("dynamic annotation digest is invalid")
            annotation = load_ground_truth_annotations(annotation_path)
            annotation_objects.append(annotation)
            tracking_ui = _dynamic_validate_tracking_metrics(
                raw_case.get("tracking_metrics")
            )
            ground_truth = raw_case.get("ground_truth_metrics")
            ground_truth_ui = _dynamic_validate_ground_truth(
                ground_truth,
                spec=spec,
                annotation=annotation,
                manifest=manifest,
            )
            ground_truth_objects.append(ground_truth)
            repeatability = raw_case.get("repeatability")
            expected_repeatability = (
                {"runs": 2, "status": "pass", "exact_non_runtime_metrics": True}
                if split == "development"
                else {
                    "runs": 1,
                    "status": "not_run_for_holdout",
                    "exact_non_runtime_metrics": None,
                }
            )
            if repeatability != expected_repeatability:
                raise ValueError("dynamic repeatability contract is invalid")
            parsed_results[spec.clip_id] = {
                "evaluation_duration_seconds": duration,
                "tracking": tracking_ui,
                "observed_state": ground_truth_ui,
                "repeatability": expected_repeatability,
            }
        recomputed_aggregate = aggregate_ground_truth_metrics(
            ground_truth_objects,
            annotations=annotation_objects,
            target_fps=manifest.target_fps,
            power_policy=manifest.power_policy,
        )
        if report.get("aggregate") != recomputed_aggregate:
            raise ValueError("dynamic aggregate does not match pooled case evidence")
        recomputed_gate = apply_acceptance_policy(
            recomputed_aggregate, manifest.acceptance_policy
        )
        if report.get("release_gate") != recomputed_gate:
            raise ValueError("dynamic release gate does not match the frozen policy")
        underpowered = list(recomputed_gate["underpowered_required_metrics"])
        failed = list(recomputed_gate["failed_required_metrics"])
        # A required underpowered check fails the frozen release gate closed.
        # Keep that overall failure visible while the individual check remains
        # explicitly "not_evaluated" in the UI.
        release_status = str(recomputed_gate["status"])
        aggregate_ui = {
            key: recomputed_aggregate.get(key)
            for key in DYNAMIC_AGGREGATE_UI_METRICS
        }
        aggregate_ui.update({
            "observation_count": recomputed_aggregate.get("observation_count"),
            "labeled_observation_count": recomputed_aggregate.get(
                "labeled_observation_count"
            ),
            "reacquisition_event_count": recomputed_aggregate.get(
                "reacquisition_event_count"
            ),
        })
        checks_ui = [
            {
                "metric": item["metric"],
                "required": item["required"],
                "status": item["status"],
                "actual": item["actual"],
                "comparison": item["comparison"],
                "threshold": item["threshold"],
            }
            for item in recomputed_gate["checks"]
        ]
        trusted = True
    except (
        DynamicEvaluationError,
        GroundTruthValidationError,
        OSError,
        TypeError,
        ValueError,
    ):
        trusted = False

    if trusted:
        for item in metadata_cases:
            item["result"] = parsed_results.get(str(item["id"]))
    else:
        parsed_results = {}
        aggregate_ui = {}
        checks_ui = []
        generated_at = None
        release_status = "untrusted"
        underpowered = []
        failed = []
    return {
        "state": release_status if trusted else "untrusted",
        "available": trusted,
        "trusted": trusted,
        "trust_message": (
            "Manifest, model, media contracts, annotations, source, configuration, policy, pooled metrics, and release gate verified."
            if trusted
            else "Results are hidden because this report failed at least one integrity or recomputation contract."
        ),
        "split": split,
        "expected_case_count": len(selected_specs),
        "case_count": len(parsed_results),
        "manifest_verified": manifest is not None and manifest_digest is not None,
        "model_verified": model_verified,
        "annotations_verified": annotations_verified,
        "release_gate_status": release_status,
        "underpowered_required_metrics": underpowered,
        "failed_required_metrics": failed,
        "aggregate": aggregate_ui,
        "checks": checks_ui,
        "generated_at": generated_at,
        "cases": metadata_cases,
    }


def test_lab_payload(
    *,
    media_dir: Path,
    reports_dir: Path,
    local_model_status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return bounded UI evidence without exposing arbitrary local paths."""
    video_report = _read_report(reports_dir / "presentcoach_video_eval.json") or {}
    llm_report = _read_report(reports_dir / "presentcoach_llm_eval.json") or {}
    tracking_report = _read_report(
        reports_dir / "presentcoach_tracking_eval.json"
    ) or {}
    dynamic_reports = {
        split: _read_dynamic_report(reports_dir / filename)
        for split, filename in DYNAMIC_REPORT_FILENAMES.items()
    }
    resolved_model_status: Mapping[str, object] = local_model_status or {}
    if (
        local_model_status is None
        and llm_report.get("schema_version") == LLM_REPORT_SCHEMA_VERSION
    ):
        resolved_model_status = OllamaPresentationLLM().status()
    results = {
        str(item.get("media_id")): item
        for item in video_report.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("media_id"), str)
    }
    clips = []
    for source in TEST_MEDIA:
        path = media_dir / source["filename"]
        result = results.get(source["id"])
        clips.append({
            **source,
            "available": path.is_file(),
            "video_url": f"/api/test-media/{source['id']}/video" if path.is_file() else None,
            "result": result,
        })
    return {
        "evaluation_only": True,
        "used_for_training": False,
        "video_eval": {
            "case_count": int(video_report.get("case_count", 0) or 0),
            "passed": int(video_report.get("passed", 0) or 0),
            "pass_rate_percent": float(video_report.get("pass_rate_percent", 0) or 0),
            "generated_at": video_report.get("generated_at"),
        },
        "llm_eval": _llm_report_payload(
            llm_report,
            project_root=Path(__file__).resolve().parents[2],
            local_model_status=resolved_model_status,
        ),
        "tracking_eval": _tracking_report_payload(
            tracking_report,
            manifest_path=media_dir / "face_tracking_manifest.json",
        ),
        "dynamic_eval": {
            "evaluation_only": True,
            "used_for_training": False,
            "identity_recognition": False,
            "face_embeddings": False,
            "landmarks_persisted": False,
            "gaze_accuracy_evaluated": False,
            "claim_boundary": (
                "Single-curator observed-state engineering gates for this fixed "
                "corpus—not population accuracy, gaze accuracy, identity recognition, "
                "or training evidence."
            ),
            "development": _dynamic_report_payload(
                dynamic_reports["development"][1],
                report_present=dynamic_reports["development"][0],
                split="development",
                manifest_path=media_dir / "dynamic_presentation_manifest.json",
                project_root=Path(__file__).resolve().parents[2],
            ),
            "holdout": _dynamic_report_payload(
                dynamic_reports["model_output_holdout"][1],
                report_present=dynamic_reports["model_output_holdout"][0],
                split="model_output_holdout",
                manifest_path=media_dir / "dynamic_presentation_manifest.json",
                project_root=Path(__file__).resolve().parents[2],
            ),
        },
        "clips": clips,
    }
