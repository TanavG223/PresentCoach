"""Public, reproducible evaluation evidence for the local PresentCoach UI."""

from __future__ import annotations

import json
import hashlib
from hmac import compare_digest
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


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


def test_lab_payload(*, media_dir: Path, reports_dir: Path) -> dict[str, object]:
    """Return bounded UI evidence without exposing arbitrary local paths."""
    video_report = _read_report(reports_dir / "presentcoach_video_eval.json") or {}
    llm_report = _read_report(reports_dir / "presentcoach_llm_eval.json") or {}
    tracking_report = _read_report(
        reports_dir / "presentcoach_tracking_eval.json"
    ) or {}
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
        "llm_eval": {
            "model": llm_report.get("model"),
            "case_count": int(llm_report.get("case_count", 0) or 0),
            "passed": int(llm_report.get("passed", 0) or 0),
            "pass_rate_percent": float(llm_report.get("pass_rate_percent", 0) or 0),
        },
        "tracking_eval": _tracking_report_payload(
            tracking_report,
            manifest_path=media_dir / "face_tracking_manifest.json",
        ),
        "clips": clips,
    }
