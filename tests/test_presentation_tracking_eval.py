import json
from pathlib import Path

import numpy as np
import pytest

from stroke_screening.presentation_tracking_eval import (
    TrackingEvaluationError,
    TrackingObservation,
    build_tracking_checks,
    compute_tracking_metrics,
    deterministic_metric_projection,
    load_tracking_manifest,
)


def _face_points(*, shift_x: float = 0.0, rigid_delta: float = 0.0) -> np.ndarray:
    points = np.zeros((478, 3), dtype=np.float64)
    points[:, 0] = 0.5 + shift_x
    points[:, 1] = 0.5
    points[33, :2] = (0.4 + shift_x, 0.45)
    points[263, :2] = (0.6 + shift_x, 0.45)
    rigid = (1, 6, 10, 109, 152, 168, 234, 338, 454)
    for index, landmark_index in enumerate(rigid):
        points[landmark_index, :2] = (
            0.44 + shift_x + index * 0.015 + rigid_delta,
            0.35 + index * 0.025,
        )
    return points


def _observation(
    timestamp: float,
    *,
    face_count: int = 1,
    points: np.ndarray | None = None,
) -> TrackingObservation:
    if points is None and face_count == 1:
        points = _face_points()
    return TrackingObservation(timestamp, face_count, 5.0, points)


def test_tracking_metrics_measure_dropouts_reacquisition_and_multi_face_abstention():
    observations = (
        _observation(0.0),
        _observation(0.1),
        _observation(0.2, face_count=0),
        _observation(0.3, face_count=0),
        _observation(0.4),
        _observation(0.5, face_count=2),
        _observation(0.6),
    )

    metrics = compute_tracking_metrics(
        observations,
        evaluation_duration_seconds=0.7,
        target_fps=10,
        stable_min_frames=2,
    )

    assert metrics["valid_face_frame_count"] == 4
    assert metrics["valid_face_frame_ratio"] == pytest.approx(4 / 7, abs=1e-6)
    assert metrics["dropout_run_count"] == 2
    assert metrics["reacquisition_event_count"] == 2
    assert metrics["max_reacquisition_seconds"] == pytest.approx(0.2)
    assert metrics["multi_face_frame_count"] == 1
    assert metrics["multi_face_abstention_violations"] == 0


def test_tracking_metrics_only_report_jitter_for_long_stable_runs():
    stable = tuple(
        _observation(
            index / 10,
            points=_face_points(shift_x=index * 0.001, rigid_delta=(index % 2) * 0.0002),
        )
        for index in range(10)
    )
    metrics = compute_tracking_metrics(
        stable,
        evaluation_duration_seconds=1,
        target_fps=10,
        stable_min_frames=8,
    )
    assert metrics["stable_segment_count"] == 1
    assert metrics["stable_transition_count"] == 9
    assert metrics["landmark_jitter_p95"] is not None
    assert metrics["landmark_jitter_p95"] < 0.01

    insufficient = compute_tracking_metrics(
        stable[:4],
        evaluation_duration_seconds=0.4,
        target_fps=10,
        stable_min_frames=8,
    )
    assert insufficient["stable_transition_count"] == 0
    assert insufficient["landmark_jitter_p95"] is None


def test_multi_face_landmarks_are_counted_as_abstention_violations():
    metrics = compute_tracking_metrics(
        (_observation(0.0, face_count=2, points=_face_points()),),
        evaluation_duration_seconds=0.1,
        target_fps=10,
        stable_min_frames=2,
    )
    assert metrics["multi_face_frame_count"] == 1
    assert metrics["multi_face_abstention_violations"] == 1
    assert metrics["valid_face_frame_ratio"] == 0


def test_malformed_single_face_landmarks_fail_closed():
    metrics = compute_tracking_metrics(
        (_observation(0.0, points=np.zeros((4, 3), dtype=np.float64)),),
        evaluation_duration_seconds=0.1,
        target_fps=10,
        stable_min_frames=2,
    )
    assert metrics["valid_face_frame_ratio"] == 0
    assert metrics["inconsistent_detection_count"] == 1


def test_checks_fail_closed_when_a_required_metric_is_unavailable():
    checks = build_tracking_checks(
        {"landmark_jitter_p95": None},
        {"max_landmark_jitter_p95": 0.03},
    )
    assert checks == [{
        "id": "max_landmark_jitter_p95",
        "label": "Stable-segment landmark jitter p95",
        "passed": False,
        "actual": None,
        "expected": "<= 0.03",
    }]


def test_repeatability_projection_excludes_runtime_but_keeps_model_metrics():
    projection = deterministic_metric_projection({
        "valid_face_frame_ratio": 0.9,
        "dropout_runs": [{"duration_seconds": 0.2}],
        "mean_inference_ms": 4.1,
        "p95_inference_ms": 5.2,
        "wall_clock_seconds": 2.0,
        "processing_throughput_fps": 200.0,
    })
    assert projection == {
        "valid_face_frame_ratio": 0.9,
        "dropout_runs": [{"duration_seconds": 0.2}],
    }


def test_manifest_rejects_parent_paths_and_unknown_expectations(tmp_path: Path):
    manifest = {
        "schema_version": 1,
        "target_fps": 15,
        "model": {"filename": "face.task", "sha256": "a" * 64},
        "clips": [{
            "id": "bad",
            "filename": "../private.mov",
            "sha256": "b" * 64,
            "source_url": "https://example.invalid/source",
            "license": "test",
            "purpose": "test",
            "duration_seconds": 1,
            "expectations": {"invented_check": 1},
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrackingEvaluationError, match="base filename"):
        load_tracking_manifest(path)


def test_manifest_rejects_invalid_dataset_role_and_source_digest(tmp_path: Path):
    manifest = {
        "schema_version": 1,
        "target_fps": 15,
        "model": {"filename": "face.task", "sha256": "a" * 64},
        "clips": [{
            "id": "bad-role",
            "filename": "clip.webm",
            "sha256": "b" * 64,
            "source_url": "https://example.invalid/source",
            "source_sha1": "not-a-source-digest",
            "license": "test",
            "purpose": "test",
            "dataset_role": "training",
            "duration_seconds": 1,
            "expectations": {"min_analyzed_fps": 1},
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrackingEvaluationError, match="source_sha1"):
        load_tracking_manifest(path)

    manifest["clips"][0]["source_sha1"] = "c" * 40
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrackingEvaluationError, match="dataset_role"):
        load_tracking_manifest(path)


def test_checked_in_tracking_manifest_is_valid():
    root = Path(__file__).resolve().parents[1]
    manifest, clips = load_tracking_manifest(
        root / "test_media" / "face_tracking_manifest.json"
    )
    assert manifest["target_fps"] == 15
    assert len(clips) == 12
    assert {
        "stable-address-tracking",
        "mixed-shot-reacquisition",
        "distant-face-abstention",
        "derived-two-face-abstention",
    } <= {clip.clip_id for clip in clips}
    assert sum(clip.dataset_role == "holdout" for clip in clips) == 4
    assert sum(clip.dataset_role == "development" for clip in clips) == 4
