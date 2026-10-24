import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from presentcoach.presentation_dynamic_eval import (
    DynamicEvaluationError,
    aggregate_ground_truth_metrics,
    apply_acceptance_policy,
    atomic_write_json,
    build_dynamic_evaluation_report,
    load_dynamic_manifest,
)
from presentcoach.presentation_tracking_eval import TrackingObservation
from presentcoach.presentation_tracking_ground_truth import (
    GroundTruthPowerPolicy,
    parse_ground_truth_annotations,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _face_points() -> np.ndarray:
    points = np.zeros((478, 3), dtype=np.float64)
    points[:, :2] = 0.5
    points[33, :2] = (0.4, 0.45)
    points[263, :2] = (0.6, 0.45)
    for index, landmark in enumerate((1, 6, 10, 109, 152, 168, 234, 338, 454)):
        points[landmark, :2] = (0.42 + index * 0.018, 0.35 + index * 0.025)
    return points


def _annotations(clip_id: str, split_group_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "clip_id": clip_id,
        "split_group_id": split_group_id,
        "duration_seconds": 2,
        "intervals": [
            {
                "start_seconds": 0,
                "end_seconds": 0.5,
                "face_state": "no_face",
                "motion": "unknown",
                "gaze_truth": "unknown",
            },
            {
                "start_seconds": 0.5,
                "end_seconds": 1.5,
                "face_state": "single_trackable",
                "motion": "mixed",
                "gaze_truth": "unknown",
            },
            {
                "start_seconds": 1.5,
                "end_seconds": 2,
                "face_state": "multiple_faces",
                "motion": "unknown",
                "gaze_truth": "unknown",
            },
        ],
    }


def _clip(
    *,
    clip_id: str,
    role: str,
    media_path: Path,
    annotation_path: Path,
) -> dict[str, object]:
    group = f"source-{clip_id}"
    return {
        "id": clip_id,
        "filename": media_path.name,
        "sha256": _sha256(media_path),
        "dataset_role": role,
        "split_group_id": group,
        "excerpt_start_seconds": 10,
        "excerpt_duration_seconds": 2,
        "annotation_file": f"annotations/{annotation_path.name}",
        "annotation_sha256": _sha256(annotation_path),
        "manual_visual_inspection_only": role == "model_output_holdout",
        "model_output_examined_during_scouting": role == "development",
        "source": {
            "title": f"Source {clip_id}",
            "page_url": "https://commons.wikimedia.org/wiki/File:Example.webm",
            "derivative_url": "https://upload.wikimedia.org/example.webm",
            "derivative_filename": f"source-{clip_id}.webm",
            "derivative_sha256": "b" * 64,
            "original_sha1": "a" * 40,
            "original_duration_seconds": 100,
            "original_resolution": [1920, 1080],
            "derivative_resolution": [854, 480],
        },
        "attribution": {
            "creator": "Example creator",
            "license_name": "Creative Commons Attribution 3.0",
            "license_spdx": "CC-BY-3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "credit_line": "Example creator, CC BY 3.0.",
            "change_notice": "Trimmed and transcoded.",
            "redistribution_constraints": ["Retain attribution"],
        },
    }


def _manifest(clips: list[dict[str, object]], model_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "description": "Locked moving-speaker corpus.",
        "evaluation_only": True,
        "never_training_data": True,
        "identity_recognition": False,
        "face_embeddings_allowed": False,
        "landmarks_persisted": False,
        "gaze_ground_truth_available": False,
        "gaze_annotation_policy": "Unknown for natural multi-camera footage.",
        "binary_distribution": "Downloaded locally.",
        "target_fps": 15,
        "model": {"filename": model_path.name, "sha256": _sha256(model_path)},
        "power_policy": {
            "minimum_trackable_frames": 1,
            "minimum_nontrackable_frames": 1,
            "minimum_multi_face_frames": 1,
            "minimum_trackable_seconds": 0.5,
            "minimum_reacquisition_events": 1,
            "minimum_sampling_coverage_ratio": 0.8,
        },
        "scoring_policy": {
            "scope": "one_global_policy_for_development_and_model_output_holdout",
            "thresholds_frozen_before_model_output_holdout": True,
            "aggregation": "Pool observations within each split.",
            "uncertain_and_unlabeled_intervals": "excluded",
            "underpowered_metric_behavior": "Return null and fail closed.",
            "gaze_metric_behavior": "Do not score gaze.",
            "required_metrics": [
                "trackable_coverage",
                "false_valid_rate",
                "multi_face_abstention",
                "dropouts_per_annotated_minute",
                "reacquisition_success_rate",
                "max_reacquisition_latency_seconds",
            ],
            "acceptance_thresholds": {
                "trackable_coverage_min": 0.9,
                "false_valid_rate_max": 0.1,
                "multi_face_abstention_min": 0.9,
                "dropouts_per_annotated_minute_max": 1,
                "reacquisition_success_rate_min": 0.9,
                "max_reacquisition_latency_seconds_max": 0.1,
            },
            "claim_boundary": "Engineering gates, not population accuracy.",
        },
        "annotation_contract": {
            "method": "single_curator_manual_coarse_intervals",
            "curator_count": 1,
            "annotation_date": "2026-08-20",
            "manual_sampling_interval_seconds": 5,
            "boundary_refinement_interval_seconds": 1,
            "boundary_uncertainty_seconds": 2.5,
            "frame_exhaustive": False,
            "independently_adjudicated": False,
            "accuracy_claim_allowed": False,
            "single_trackable": "One visible trackable face.",
            "no_face": "No visible face.",
            "multiple_faces": "More than one visible face.",
            "occluded_or_tiny": "Not sufficiently visible.",
            "uncertain": "Cannot defensibly label.",
            "motion": "Coarse visible motion.",
            "gaze_truth": "Always unknown.",
            "limitations": "Coarse single-curator evidence.",
            "pre_release_curator_qa": {
                "completed_date": "2026-08-20",
                "status": "visual_only_pass_after_corrections",
                "canonical_excerpts_reviewed": len(clips),
                "method": "Visual-only fixture review.",
                "model_or_evaluator_output_viewed": False,
                "results_generated_from_draft_annotations": "invalidated",
                "corrections": ["Fixture labels reviewed."],
            },
        },
        "canonical_excerpt": {
            "container": "MP4",
            "video_codec": "H.264",
            "width": 854,
            "height": 480,
            "frame_rate": 30,
            "pixel_format": "yuv420p",
            "audio_codec": "AAC",
            "audio_sample_rate_hz": 16000,
            "audio_channels": 1,
            "duration_seconds": 2,
            "metadata_removed": True,
            "reference_encoder": "test encoder",
            "reproducibility_note": "Pinned bytes.",
        },
        "split_policy": {
            "development": "May be inspected.",
            "model_output_holdout": "Must remain untouched until freeze.",
            "grouping": "Same source stays in one split.",
        },
        "clips": clips,
    }


def _project(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "test_media" / "annotations").mkdir(parents=True)
    model = tmp_path / "models" / "face.task"
    model.write_bytes(b"locked model")
    shutil.copy2(
        Path(__file__).resolve().parents[1]
        / "models"
        / "face_detection_yunet_2023mar.onnx",
        tmp_path / "models" / "face_detection_yunet_2023mar.onnx",
    )
    clips = []
    for clip_id, role in (("dev", "development"), ("secret", "model_output_holdout")):
        media = tmp_path / "test_media" / f"{clip_id}.mp4"
        media.write_bytes(f"media-{clip_id}".encode())
        annotation = tmp_path / "test_media" / "annotations" / f"{clip_id}.json"
        annotation.write_text(
            json.dumps(_annotations(clip_id, f"source-{clip_id}")), encoding="utf-8"
        )
        clips.append(
            _clip(
                clip_id=clip_id,
                role=role,
                media_path=media,
                annotation_path=annotation,
            )
        )
    manifest = tmp_path / "test_media" / "dynamic_presentation_manifest.json"
    manifest.write_text(json.dumps(_manifest(clips, model)), encoding="utf-8")
    return tmp_path, manifest


def _sampler_calls(calls: list[str]):
    points = _face_points()

    def sample(path: Path, **_kwargs):
        calls.append(path.name)
        observations = []
        for index in range(30):
            timestamp = index / 15
            if timestamp < 0.5:
                observations.append(TrackingObservation(timestamp, 0, 4.0, None))
            elif timestamp < 1.5:
                observations.append(TrackingObservation(timestamp, 1, 4.0, points.copy()))
            else:
                observations.append(TrackingObservation(timestamp, 2, 4.0, None))
        run = len(calls)
        return tuple(observations), 2.0, {
            "wall_clock_seconds": float(run),
            "processing_throughput_fps": 30.0 / run,
        }

    return sample


def test_default_development_report_never_reads_or_runs_holdout(tmp_path: Path):
    root, manifest = _project(tmp_path)
    calls: list[str] = []
    holdout_media = root / "test_media" / "secret.mp4"
    holdout_media.unlink()

    report = build_dynamic_evaluation_report(
        manifest_path=manifest,
        project_root=root,
        sampler=_sampler_calls(calls),
        generated_at="2026-08-20T00:00:00Z",
    )

    assert report["split"] == "development"
    assert report["case_count"] == 1
    assert calls == ["dev.mp4", "dev.mp4"]
    assert report["cases"][0]["repeatability"] == {
        "runs": 2,
        "status": "pass",
        "exact_non_runtime_metrics": True,
    }
    assert report["release_gate"]["status"] == "pass"
    assert report["landmarks_persisted"] is False
    assert "pass_rate" not in json.dumps(report)
    assert "observations" not in report["cases"][0]
    assert len(report["configuration_sha256"]) == 64
    assert len(report["policy_sha256"]) == 64
    assert set(report["source_sha256"]) == {
        "presentation_dynamic_eval.py",
        "presentation_tracking_eval.py",
        "presentation_tracking_ground_truth.py",
        "landmarks.py",
        "presentation_video_timing.py",
        "scripts/evaluate_dynamic_presentations.py",
    }
    assert report["configuration"]["face_landmarker"] == {
        "max_face_count": 2,
        "min_face_detection_confidence": 0.55,
        "min_face_presence_confidence": 0.55,
        "min_tracking_confidence": 0.55,
        "landmark_smoothing_time_constant_ms": 80.0,
        "reacquisition_gap_ms": 250,
        "temporal_hold_ms": 80,
        "multiscale_fallback_enabled": True,
        "multiscale_tile_size": 0.5,
        "multiscale_tile_offsets": [0.0, 0.25, 0.5],
        "multiscale_tile_count": 9,
        "proposal_model_filename": "face_detection_yunet_2023mar.onnx",
        "proposal_model_sha256": (
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
        ),
        "proposal_score_threshold": 0.85,
        "proposal_nms_threshold": 0.3,
        "proposal_top_k": 50,
        "proposal_roi_scales": [1.25],
        "proposal_liveness_window_ms": 800,
        "proposal_liveness_hold_ms": 2000,
        "proposal_liveness_min_iou": 0.01,
        "proposal_liveness_range_threshold": 0.11,
        "proposal_provisional_track_ms": 500,
        "proposal_provisional_rearm_gap_ms": 1000,
        "proposal_provisional_luminance_change_ratio": 0.35,
        "small_face_liveness_area_threshold": 0.015,
        "minimum_eye_to_lower_face_luminance_ratio": 0.50,
        "proposal_appearance_size": 48,
        "proposal_appearance_lag_min_ms": 150,
        "proposal_appearance_lag_max_ms": 300,
        "proposal_appearance_motion_threshold": 0.18,
        "multiscale_deduplication_iou": 0.25,
    }
    assert report["model"]["matches_manifest"] is True


def test_holdout_runs_once_only_when_explicitly_selected(tmp_path: Path):
    root, manifest = _project(tmp_path)
    calls: list[str] = []

    report = build_dynamic_evaluation_report(
        manifest_path=manifest,
        project_root=root,
        split="model_output_holdout",
        sampler=_sampler_calls(calls),
        generated_at="2026-08-20T00:00:00Z",
    )

    assert report["split"] == "model_output_holdout"
    assert calls == ["secret.mp4"]
    assert report["cases"][0]["repeatability"] == {
        "runs": 1,
        "status": "not_run_for_holdout",
        "exact_non_runtime_metrics": None,
    }


def test_development_repeatability_fails_on_changed_model_output(tmp_path: Path):
    root, manifest = _project(tmp_path)
    calls = 0

    def nondeterministic(path: Path, **kwargs):
        nonlocal calls
        calls += 1
        observations, duration, runtime = _sampler_calls([])(path, **kwargs)
        if calls == 2:
            changed = list(observations)
            changed[10] = TrackingObservation(changed[10].timestamp_seconds, 0, 4.0, None)
            observations = tuple(changed)
        return observations, duration, runtime

    with pytest.raises(DynamicEvaluationError, match="deterministic metrics changed"):
        build_dynamic_evaluation_report(
            manifest_path=manifest,
            project_root=root,
            sampler=nondeterministic,
        )


def test_manifest_rejects_duplicate_keys_unknown_fields_and_traversal(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(DynamicEvaluationError, match="duplicate field"):
        load_dynamic_manifest(duplicate)

    root, manifest_path = _project(tmp_path / "project")
    document = json.loads(manifest_path.read_text())
    document["surprise"] = True
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DynamicEvaluationError, match=r"unknown=\['surprise'\]"):
        load_dynamic_manifest(manifest_path)

    document.pop("surprise")
    document["clips"][0]["annotation_file"] = "../private.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DynamicEvaluationError, match="safe relative path"):
        load_dynamic_manifest(manifest_path)


def test_integrity_hash_mismatch_fails_before_sampling(tmp_path: Path):
    root, manifest = _project(tmp_path)
    (root / "test_media" / "dev.mp4").write_bytes(b"tampered")
    calls: list[str] = []

    with pytest.raises(DynamicEvaluationError, match="Media dev SHA-256 mismatch"):
        build_dynamic_evaluation_report(
            manifest_path=manifest,
            project_root=root,
            sampler=_sampler_calls(calls),
        )
    assert calls == []


def test_proposal_model_tampering_fails_before_sampling(tmp_path: Path):
    root, manifest = _project(tmp_path)
    (root / "models" / "face_detection_yunet_2023mar.onnx").write_bytes(
        b"tampered proposal model"
    )
    calls: list[str] = []

    with pytest.raises(
        DynamicEvaluationError, match="Anonymous proposal model SHA-256 mismatch"
    ):
        build_dynamic_evaluation_report(
            manifest_path=manifest,
            project_root=root,
            sampler=_sampler_calls(calls),
        )

    assert calls == []


def test_aggregate_reports_underpowered_as_not_evaluated():
    annotations = parse_ground_truth_annotations({
        "schema_version": 1,
        "clip_id": "tiny",
        "split_group_id": "tiny-source",
        "duration_seconds": 1,
        "intervals": [{
            "start_seconds": 0,
            "end_seconds": 1,
            "face_state": "single_trackable",
            "motion": "static",
        }],
    })
    metrics = {
        "observation_count": 1,
        "labeled_observation_count": 1,
        "trackable_observation_count": 1,
        "trackable_valid_observation_count": 1,
        "nontrackable_observation_count": 0,
        "false_valid_observation_count": 0,
        "multi_face_observation_count": 0,
        "multi_face_abstained_observation_count": 0,
        "dropout_run_count": 0,
        "reacquisition_event_count": 0,
        "reacquisition_success_count": 0,
        "reacquisition_events": [],
    }
    aggregate = aggregate_ground_truth_metrics(
        [metrics],
        annotations=[annotations],
        target_fps=15,
        power_policy=GroundTruthPowerPolicy(),
    )
    assert aggregate["trackable_coverage"] is None
    assert aggregate["false_valid_rate"] is None
    assert aggregate["multi_face_abstention"] is None
    assert aggregate["reacquisition_success_rate"] is None
    policy = {
        "required_metrics": ["trackable_coverage"],
        "trackable_coverage_min": 0.9,
        "false_valid_rate_max": 0.1,
        "multi_face_abstention_min": 0.9,
        "dropouts_per_annotated_minute_max": 1,
        "reacquisition_success_rate_min": 0.9,
        "max_reacquisition_latency_seconds_max": 1,
    }
    gate = apply_acceptance_policy(aggregate, policy)
    assert gate["status"] == "fail"
    assert gate["underpowered_required_metrics"] == ["trackable_coverage"]
    assert gate["checks"][0]["passed"] is None


def test_atomic_write_replaces_complete_json_and_cleans_temporary_files(tmp_path: Path):
    destination = tmp_path / "report.json"
    destination.write_text('{"old":true}\n', encoding="utf-8")

    atomic_write_json(destination, {"new": True})

    assert json.loads(destination.read_text()) == {"new": True}
    assert not list(tmp_path.glob(".report.json.*.tmp"))
