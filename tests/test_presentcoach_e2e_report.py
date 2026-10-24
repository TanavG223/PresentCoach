import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_presentcoach_e2e.py"
SPEC = importlib.util.spec_from_file_location("presentcoach_e2e_evaluator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
E2E = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E2E
SPEC.loader.exec_module(E2E)

E2EContractError = E2E.E2EContractError
build_report = E2E.build_report
default_output_for_role = E2E.default_output_for_role
feedback_is_metric_grounded = E2E.feedback_is_metric_grounded
load_dynamic_manifest = E2E.load_dynamic_manifest
selected_clips = E2E.selected_clips
validate_api_document = E2E.validate_api_document
write_report = E2E.write_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotation(clip_id: str, split_group_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "clip_id": clip_id,
        "split_group_id": split_group_id,
        "duration_seconds": 60,
        "intervals": [
            {
                "start_seconds": 0,
                "end_seconds": 60,
                "face_state": "single_trackable",
                "motion": "mixed",
                "gaze_truth": "unknown",
            }
        ],
    }


def _clip(
    media_dir: Path,
    *,
    clip_id: str,
    role: str,
    split_group_id: str,
) -> dict[str, object]:
    annotation_relative = f"annotations/{clip_id}.json"
    annotation_path = media_dir / annotation_relative
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(_annotation(clip_id, split_group_id)), encoding="utf-8"
    )
    holdout = role == "model_output_holdout"
    return {
        "id": clip_id,
        "filename": f"{clip_id}-60s.mp4",
        "sha256": "1" * 64 if not holdout else "2" * 64,
        "dataset_role": role,
        "split_group_id": split_group_id,
        "excerpt_start_seconds": 60,
        "excerpt_duration_seconds": 60,
        "annotation_file": annotation_relative,
        "annotation_sha256": _sha256(annotation_path),
        "manual_visual_inspection_only": holdout,
        "model_output_examined_during_scouting": not holdout,
        "source": {
            "title": f"Licensed presentation {clip_id}",
            "page_url": f"https://commons.wikimedia.org/wiki/File:{clip_id}.webm",
            "derivative_url": f"https://upload.wikimedia.org/{clip_id}.webm",
            "derivative_filename": f"source-{clip_id}.webm",
            "derivative_sha256": "3" * 64,
            "original_sha1": "4" * 40,
            "original_duration_seconds": 600,
            "original_resolution": [1280, 720],
            "derivative_resolution": [854, 480],
        },
        "attribution": {
            "creator": "Fixture creator",
            "license_name": "Creative Commons Attribution 3.0",
            "license_spdx": "CC-BY-3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "credit_line": "Fixture creator, CC BY 3.0.",
            "change_notice": "Converted into a 60-second fixture excerpt.",
            "redistribution_constraints": ["Retain attribution"],
        },
    }


def _manifest_document(media_dir: Path) -> dict[str, object]:
    clips = [
        _clip(
            media_dir,
            clip_id="dynamic-development",
            role="development",
            split_group_id="source-development",
        ),
        _clip(
            media_dir,
            clip_id="dynamic-holdout",
            role="model_output_holdout",
            split_group_id="source-holdout",
        ),
    ]
    return {
        "schema_version": 1,
        "description": "Frozen identity-free dynamic presentation fixture.",
        "target_fps": 15,
        "model": {"filename": "face_landmarker.task", "sha256": "5" * 64},
        "evaluation_only": True,
        "never_training_data": True,
        "identity_recognition": False,
        "face_embeddings_allowed": False,
        "landmarks_persisted": False,
        "gaze_ground_truth_available": False,
        "gaze_annotation_policy": "unknown_for_every_manual_interval",
        "binary_distribution": "Fixture binaries are local and excluded from Git.",
        "canonical_excerpt": {
            "container": "MP4",
            "video_codec": "H.264 via libx264",
            "width": 854,
            "height": 480,
            "frame_rate": 30,
            "pixel_format": "yuv420p",
            "audio_codec": "AAC",
            "audio_sample_rate_hz": 16000,
            "audio_channels": 1,
            "duration_seconds": 60,
            "metadata_removed": True,
            "reference_encoder": "Fixture FFmpeg",
            "reproducibility_note": "Fixture contract only.",
        },
        "split_policy": {
            "development": "May be inspected.",
            "model_output_holdout": "Output remains sealed.",
            "grouping": "Group by source.",
        },
        "annotation_contract": {"method": "fixture", "gaze_truth": "unknown"},
        "power_policy": {
            "minimum_trackable_frames": 15,
            "minimum_nontrackable_frames": 15,
            "minimum_multi_face_frames": 15,
            "minimum_trackable_seconds": 10.0,
            "minimum_reacquisition_events": 3,
            "minimum_sampling_coverage_ratio": 0.8,
        },
        "scoring_policy": {
            "thresholds_frozen_before_model_output_holdout": True,
            "scope": "one global fixture policy",
        },
        "clips": clips,
    }


def _write_manifest(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "dynamic_presentation_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_strict_manifest_loader_keeps_development_and_holdout_separate(tmp_path: Path):
    path = _write_manifest(tmp_path, _manifest_document(tmp_path))

    manifest = load_dynamic_manifest(path)

    assert len(manifest.clips) == 2
    assert [item.clip_id for item in selected_clips(manifest, "development")] == [
        "dynamic-development"
    ]
    assert [
        item.clip_id for item in selected_clips(manifest, "model_output_holdout")
    ] == ["dynamic-holdout"]
    assert manifest.sha256 == _sha256(path)


def test_manifest_loader_rejects_duplicates_unknown_fields_and_holdout_leakage(
    tmp_path: Path,
):
    document = _manifest_document(tmp_path)
    path = _write_manifest(tmp_path, document)
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(E2EContractError, match="duplicate field"):
        load_dynamic_manifest(path)

    document = _manifest_document(tmp_path)
    document["undeclared_threshold"] = 0.9
    path = _write_manifest(tmp_path, document)
    with pytest.raises(E2EContractError, match="frozen schema"):
        load_dynamic_manifest(path)

    document = _manifest_document(tmp_path)
    holdout = document["clips"][1]
    holdout["model_output_examined_during_scouting"] = True
    path = _write_manifest(tmp_path, document)
    with pytest.raises(E2EContractError, match="sealed model-output holdout"):
        load_dynamic_manifest(path)

    document = _manifest_document(tmp_path)
    holdout = document["clips"][1]
    holdout["split_group_id"] = "source-development"
    annotation_path = tmp_path / holdout["annotation_file"]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["split_group_id"] = "source-development"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    holdout["annotation_sha256"] = _sha256(annotation_path)
    path = _write_manifest(tmp_path, document)
    with pytest.raises(E2EContractError, match="leak"):
        load_dynamic_manifest(path)


def _api_document() -> dict[str, object]:
    quality = {
        "face_detected": "good",
        "eye_contact": "good",
        "head_stability": "good",
        "expression_variety": "good",
        "audio_clear": "good",
    }
    metrics = {
        "session_id": "a" * 32,
        "duration_seconds": 60.0,
        "aggregate": {
            "duration_seconds": 60.0,
            "analyzed_vision_fps": 15.0,
            "eye_contact_percent": 80.0,
            "head_rotation_std_degrees": 2.0,
            "head_position_std_percent": 1.5,
            "expression_variety_index": 3.0,
            "face_presence_percent": 95.0,
            "overall_words_per_minute": 120.0,
            "filler_count": 0,
            "filler_rate_per_minute": 0.0,
            "strict_filler_count": 0,
            "strict_filler_rate_per_minute": 0.0,
            "pauses_over_2_seconds": 0,
            "pauses_over_3_seconds": 0,
            "long_pause_rate_per_minute": 0.0,
            "longest_pause_seconds": 0.0,
        },
        "quality_flags": quality,
        "insufficient_metrics": [],
        "timeline": {
            "pace_windows": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 15.0,
                    "words": 30,
                    "words_per_minute": 120.0,
                }
            ],
            "filler_clusters": [],
            "longest_gaze_break": None,
            "longest_pause": None,
        },
    }
    return {
        "session": {
            "session_id": "a" * 32,
            "session_kind": "imported",
            "duration_seconds": 60.0,
            "quality_flags": quality,
            "transcript": [
                {
                    "text": "hello",
                    "start_seconds": 0.1,
                    "end_seconds": 0.4,
                    "probability": 0.9,
                },
                {
                    "text": "world",
                    "start_seconds": 0.5,
                    "end_seconds": 0.9,
                    "probability": 0.9,
                },
            ],
            "vision_metrics": [
                {"timestamp_seconds": 0.0, "face_detected": True},
                {"timestamp_seconds": 1.0, "face_detected": True},
            ],
        },
        "metrics": metrics,
        "feedback": {
            "status": "ready",
            "source": "local_llm_verified_general_practice",
            "strengths": [
                {
                    "text": "Across the session ending at 60 seconds, camera contact was 80 %.",
                    "metric": "eye_contact_percent",
                    "value": 80.0,
                    "unit": "%",
                    "timestamp_seconds": 60.0,
                }
            ],
            "improvements": [],
            "insufficient_data": [],
        },
        "coaching_cues": {
            "schema_version": "presentcoach.review-cues.v1",
            "session_id": "a" * 32,
            "duration_seconds": 60.0,
            "status": "ready",
            "cues": [
                {
                    "cue_id": "cue-0001",
                    "start_seconds": 5.0,
                    "end_seconds": 6.0,
                    "seek_seconds": 5.0,
                }
            ],
            "counts": {},
            "limitations": {
                "descriptive_measured_behavior_only": True,
                "infers_confidence": False,
                "infers_reading": False,
            },
        },
        "media": {
            "available": True,
            "mime_type": "video/mp4",
            "source": "upload",
            "playback_url": "/api/profiles/p/sessions/s/video?token=signed",
        },
    }


def test_api_document_validator_accepts_only_grounded_bounded_evidence():
    document = _api_document()
    observed_call = {
        "system_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "schema_sha256": "3" * 64,
        "response_sha256": "4" * 64,
        "completed": True,
    }

    checks = validate_api_document(
        document,
        expected_duration=60.0,
        observed_llm_calls=[observed_call],
    )

    assert checks
    assert all(item["passed"] for item in checks), checks
    grounded, detail = feedback_is_metric_grounded(
        document["feedback"], document["metrics"]
    )
    assert grounded is True
    assert "exact stored facts" in detail

    document["feedback"]["strengths"][0]["value"] = 81.0
    grounded, detail = feedback_is_metric_grounded(
        document["feedback"], document["metrics"]
    )
    assert grounded is False
    assert "does not match stored metrics" in detail


def test_api_document_accepts_explicit_zero_face_abstention_when_quality_fails_closed():
    document = _api_document()
    document["session"]["vision_metrics"] = [
        {"timestamp_seconds": 0.0, "face_detected": False},
        {"timestamp_seconds": 1.0, "face_detected": False},
    ]
    bad_visual = [
        "face_detected",
        "eye_contact",
        "head_stability",
        "expression_variety",
    ]
    for name in bad_visual:
        document["session"]["quality_flags"][name] = "bad"
        document["metrics"]["quality_flags"][name] = "bad"
    document["feedback"] = {
        "status": "ready",
        "source": "local_llm_verified_general_practice",
        "strengths": [
            {
                "text": "Across the session ending at 60 seconds, speaking pace was 120 WPM.",
                "metric": "overall_words_per_minute",
                "value": 120.0,
                "unit": "WPM",
                "timestamp_seconds": 60.0,
            }
        ],
        "improvements": [],
        "insufficient_data": bad_visual,
    }

    checks = validate_api_document(
        document,
        expected_duration=60.0,
        observed_llm_calls=[
            {
                "system_sha256": "1" * 64,
                "prompt_sha256": "2" * 64,
                "schema_sha256": "3" * 64,
                "response_sha256": "4" * 64,
                "completed": True,
            }
        ],
    )

    assert all(item["passed"] for item in checks), checks
    tracking = next(
        item for item in checks
        if item["label"] == "Anonymous face tracking path executed"
    )
    assert tracking["actual"] == {"bucket_decisions": 2, "detected_buckets": 0}


def test_api_document_validator_rejects_nonmonotonic_timestamps_and_bad_cue_bounds():
    document = _api_document()
    document["session"]["vision_metrics"] = [
        {"timestamp_seconds": 1.0, "face_detected": True},
        {"timestamp_seconds": 0.0, "face_detected": True},
    ]
    document["coaching_cues"]["cues"][0]["end_seconds"] = 61.0

    checks = validate_api_document(
        document,
        expected_duration=60.0,
        observed_llm_calls=[
            {
                "system_sha256": "1" * 64,
                "prompt_sha256": "2" * 64,
                "schema_sha256": "3" * 64,
                "response_sha256": "4" * 64,
                "completed": True,
            }
        ],
    )

    failures = {item["label"] for item in checks if not item["passed"]}
    assert "Vision timestamps monotonic and bounded" in failures
    assert "Coaching cue bounds" in failures


def test_report_explicitly_marks_unrun_split_and_is_written_user_only(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, _manifest_document(tmp_path))
    manifest = load_dynamic_manifest(manifest_path)
    case = {
        "case_id": "dynamic-development",
        "passed": True,
        "checks": [{"label": "fixture", "passed": True}],
    }
    report = build_report(
        selected_role="development",
        manifest=manifest,
        model_status={
            "model": "presentcoach-local",
            "digest": "6" * 64,
            "available": True,
        },
        artifacts={
            "manifest_sha256": manifest.sha256,
            "evaluator_sha256": "7" * 64,
            "mediapipe_model_sha256": "8" * 64,
            "whisper_model_sha256": "9" * 64,
            "feedback_schema_sha256": "a" * 64,
        },
        cases=[case],
    )
    output = tmp_path / "reports" / "e2e.json"
    write_report(output, report)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["results"]["development"]["status"] == "passed"
    assert loaded["results"]["model_output_holdout"] == {
        "dataset_role": "model_output_holdout",
        "status": "not_run",
        "case_count": 0,
        "passed": 0,
        "pass_rate_percent": None,
        "cases": [],
    }
    assert loaded["holdout_default"] == "sealed_not_run"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert default_output_for_role("development").name == (
        "presentcoach_e2e_development.json"
    )
    assert default_output_for_role("model_output_holdout").name == (
        "presentcoach_e2e_holdout.json"
    )
    assert default_output_for_role("development") != default_output_for_role(
        "model_output_holdout"
    )
