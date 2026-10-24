import json
from pathlib import Path

import numpy as np
import pytest

from presentcoach.presentation_tracking_eval import TrackingObservation
from presentcoach.presentation_tracking_ground_truth import (
    GroundTruthPowerPolicy,
    GroundTruthValidationError,
    load_ground_truth_annotations,
    parse_ground_truth_annotations,
    score_tracking_ground_truth,
    validate_split_group_isolation,
)


def _points() -> np.ndarray:
    return np.zeros((478, 3), dtype=np.float64)


def _observation(timestamp: float, *, valid: bool = True, face_count: int | None = None):
    count = (1 if valid else 0) if face_count is None else face_count
    return TrackingObservation(
        timestamp_seconds=timestamp,
        face_count=count,
        inference_ms=1.0,
        landmarks=_points() if valid else None,
    )


def _document(intervals, *, duration: float = 10.0):
    return {
        "schema_version": 1,
        "clip_id": "dynamic-talk-01",
        "split_group_id": "source-group-01",
        "duration_seconds": duration,
        "intervals": intervals,
    }


def _interval(start, end, state, *, motion="unknown", gaze=None):
    value = {
        "start_seconds": start,
        "end_seconds": end,
        "face_state": state,
        "motion": motion,
    }
    if gaze is not None:
        value["gaze_truth"] = gaze
    return value


def _small_policy() -> GroundTruthPowerPolicy:
    return GroundTruthPowerPolicy(
        minimum_trackable_frames=1,
        minimum_nontrackable_frames=1,
        minimum_multi_face_frames=1,
        minimum_trackable_seconds=0.1,
        minimum_reacquisition_events=1,
    )


def test_annotation_parser_accepts_only_explicit_bounded_labels(tmp_path: Path):
    document = _document([
        _interval(0, 2, "no_face"),
        _interval(2, 6, "single_trackable", motion="translate", gaze="contact"),
        _interval(6, 8, "single_trackable", motion="head_turn", gaze="away"),
        _interval(8, 10, "uncertain"),
    ])
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    annotations = load_ground_truth_annotations(path)

    assert annotations.clip_id == "dynamic-talk-01"
    assert annotations.split_group_id == "source-group-01"
    assert annotations.has_explicit_gaze_labels is True
    assert [item.motion for item in annotations.intervals] == [
        "unknown", "translate", "head_turn", "unknown",
    ]


def test_annotation_loader_rejects_duplicate_json_fields(tmp_path: Path):
    path = tmp_path / "duplicates.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )
    with pytest.raises(GroundTruthValidationError, match="duplicate field"):
        load_ground_truth_annotations(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("face_state", "presenter_maybe", "face_state"),
        ("motion", "walking_confidently", "motion"),
        ("gaze_truth", "probably_contact", "gaze_truth"),
    ),
)
def test_annotation_parser_rejects_unknown_labels(field, value, match):
    interval = _interval(0, 1, "single_trackable", motion="static")
    interval[field] = value
    with pytest.raises(GroundTruthValidationError, match=match):
        parse_ground_truth_annotations(_document([interval], duration=1))


def test_annotation_parser_rejects_overlap_out_of_range_and_unknown_fields():
    overlapping = _document([
        _interval(0, 2, "no_face"),
        _interval(1.9, 3, "single_trackable", motion="static"),
    ], duration=3)
    with pytest.raises(GroundTruthValidationError, match="non-overlapping"):
        parse_ground_truth_annotations(overlapping)

    out_of_range = _document([
        _interval(0, 3.1, "no_face"),
    ], duration=3)
    with pytest.raises(GroundTruthValidationError, match="end_seconds"):
        parse_ground_truth_annotations(out_of_range)

    unknown = _document([_interval(0, 1, "no_face")], duration=1)
    unknown["model_guess"] = "one face"
    with pytest.raises(GroundTruthValidationError, match="unsupported fields"):
        parse_ground_truth_annotations(unknown)


def test_nontrackable_intervals_cannot_claim_motion_or_camera_contact():
    with pytest.raises(GroundTruthValidationError, match="motion must be unknown"):
        parse_ground_truth_annotations(_document([
            _interval(0, 1, "multiple_faces", motion="translate"),
        ], duration=1))
    with pytest.raises(GroundTruthValidationError, match="cannot claim contact"):
        parse_ground_truth_annotations(_document([
            _interval(0, 1, "occluded_or_tiny", gaze="contact"),
        ], duration=1))


def test_split_groups_cannot_cross_into_model_output_holdout():
    validate_split_group_isolation([
        {"clip_id": "dev-original", "dataset_role": "development", "split_group_id": "group-a"},
        {
            "clip_id": "dev-derived",
            "dataset_role": "derived_regression",
            "split_group_id": "group-a",
        },
        {
            "clip_id": "held-out",
            "dataset_role": "model_output_holdout",
            "split_group_id": "group-b",
        },
    ])

    with pytest.raises(GroundTruthValidationError, match="leaks"):
        validate_split_group_isolation([
            {"clip_id": "dev", "dataset_role": "development", "split_group_id": "shared"},
            {
                "clip_id": "held-out",
                "dataset_role": "model_output_holdout",
                "split_group_id": "shared",
            },
        ])

    with pytest.raises(GroundTruthValidationError, match="leaks"):
        validate_split_group_isolation([
            {
                "clip_id": "held-out",
                "dataset_role": "model_output_holdout",
                "split_group_id": "shared",
            },
            {
                "clip_id": "derived",
                "dataset_role": "derived_regression",
                "split_group_id": "shared",
            },
        ])


def test_ground_truth_metrics_measure_coverage_abstention_dropouts_and_reacquisition():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 1, "no_face"),
        _interval(1, 3, "single_trackable", motion="translate"),
        _interval(3, 4, "multiple_faces"),
        _interval(4, 6, "single_trackable", motion="scale"),
        _interval(6, 7, "occluded_or_tiny"),
        _interval(7, 10, "single_trackable", motion="head_turn"),
    ]))
    observations = []
    for index in range(40):
        timestamp = index / 4
        valid = 1 <= timestamp < 3 or 4 <= timestamp < 6 or 7 <= timestamp < 10
        if timestamp in {1.0, 2.0, 2.25}:
            valid = False
        if timestamp == 3.0:
            valid = True  # Two labeled faces, but the detector emitted one valid face.
        face_count = 1 if valid else (2 if 3 <= timestamp < 4 else 0)
        observations.append(
            _observation(timestamp, valid=valid, face_count=face_count)
        )

    metrics = score_tracking_ground_truth(
        observations,
        annotations,
        target_fps=4,
        power_policy=_small_policy(),
    )

    assert metrics["trackable_observation_count"] == 28
    assert metrics["trackable_valid_observation_count"] == 25
    assert metrics["trackable_coverage"] == pytest.approx(25 / 28, abs=1e-6)
    assert metrics["nontrackable_observation_count"] == 12
    assert metrics["false_valid_observation_count"] == 1
    assert metrics["false_valid_rate"] == pytest.approx(1 / 12, abs=1e-6)
    assert metrics["multi_face_observation_count"] == 4
    assert metrics["multi_face_abstained_observation_count"] == 3
    assert metrics["multi_face_abstention"] == pytest.approx(3 / 4)
    assert metrics["motion_observation_counts"] == {
        "head_turn": 12,
        "mixed": 0,
        "scale": 8,
        "static": 0,
        "translate": 8,
    }
    assert metrics["dropout_run_count"] == 1
    assert metrics["dropouts_per_annotated_minute"] == pytest.approx(60 / 7, abs=1e-6)
    assert metrics["reacquisition_event_count"] == 3
    assert metrics["reacquisition_success_count"] == 3
    assert metrics["reacquisition_success_rate"] == 1.0
    assert metrics["mean_reacquisition_latency_seconds"] == pytest.approx(1 / 12, abs=1e-6)
    assert metrics["max_reacquisition_latency_seconds"] == 0.25
    assert metrics["reacquisition_events"][0] == {
        "onset_seconds": 1.0,
        "reacquired": True,
        "latency_seconds": 0.25,
    }


def test_failed_three_frame_reacquisition_is_not_hidden_by_successful_events():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 1, "no_face"),
        _interval(1, 2, "single_trackable", motion="static"),
        _interval(2, 3, "no_face"),
        _interval(3, 4, "single_trackable", motion="static"),
        _interval(4, 5, "no_face"),
        _interval(5, 6, "single_trackable", motion="static"),
    ], duration=6))
    observations = []
    for index in range(24):
        timestamp = index / 4
        valid = 1 <= timestamp < 2 or 5 <= timestamp < 6
        observations.append(_observation(timestamp, valid=valid))

    metrics = score_tracking_ground_truth(
        observations,
        annotations,
        target_fps=4,
        power_policy=_small_policy(),
    )

    assert metrics["reacquisition_event_count"] == 3
    assert metrics["reacquisition_success_count"] == 2
    assert metrics["reacquisition_success_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["mean_reacquisition_latency_seconds"] is None
    assert metrics["max_reacquisition_latency_seconds"] is None
    assert "never produced three consecutive" in metrics["underpowered_reasons"][
        "mean_reacquisition_latency_seconds"
    ]


def test_underpowered_metrics_are_none_with_specific_reasons():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 0.5, "single_trackable", motion="static"),
        _interval(0.5, 1, "uncertain"),
    ], duration=1))
    metrics = score_tracking_ground_truth(
        [_observation(0), _observation(0.5)], annotations, target_fps=2
    )

    for metric in (
        "trackable_coverage",
        "false_valid_rate",
        "multi_face_abstention",
        "dropouts_per_annotated_minute",
        "reacquisition_success_rate",
        "mean_reacquisition_latency_seconds",
        "max_reacquisition_latency_seconds",
    ):
        assert metrics[metric] is None
        assert metric in metrics["underpowered_reasons"]
    assert metrics["uncertain_observation_count"] == 1


def test_sparse_samples_cannot_create_flattering_zero_dropout_rate():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 10, "single_trackable", motion="translate"),
    ]))
    observations = [_observation(index / 15) for index in range(15)]
    metrics = score_tracking_ground_truth(
        observations,
        annotations,
        target_fps=15,
        power_policy=GroundTruthPowerPolicy(
            minimum_trackable_frames=15,
            minimum_nontrackable_frames=1,
            minimum_multi_face_frames=1,
            minimum_trackable_seconds=10,
            minimum_reacquisition_events=1,
        ),
    )

    assert metrics["trackable_sampling_coverage"] == 0.1
    assert metrics["trackable_coverage"] is None
    assert metrics["dropouts_per_annotated_minute"] is None
    assert "10.0% sampling coverage" in metrics["underpowered_reasons"][
        "dropouts_per_annotated_minute"
    ]


def test_gaze_accuracy_is_never_claimed_from_tracking_observations():
    unlabelled_gaze = parse_ground_truth_annotations(_document([
        _interval(0, 1, "single_trackable", motion="static"),
    ], duration=1))
    no_truth = score_tracking_ground_truth(
        [_observation(0)], unlabelled_gaze, target_fps=1,
        power_policy=_small_policy(),
    )
    assert no_truth["explicit_gaze_labels"] is False
    assert no_truth["gaze_accuracy"] is None
    assert "no explicit" in no_truth["underpowered_reasons"]["gaze_accuracy"]

    explicit_gaze = parse_ground_truth_annotations(_document([
        _interval(0, 1, "single_trackable", motion="static", gaze="away"),
    ], duration=1))
    labelled = score_tracking_ground_truth(
        [_observation(0)], explicit_gaze, target_fps=1,
        power_policy=_small_policy(),
    )
    assert labelled["explicit_gaze_labels"] is True
    assert labelled["gaze_accuracy"] is None
    assert "no gaze prediction" in labelled["underpowered_reasons"]["gaze_accuracy"]


def test_observation_validation_rejects_duplicate_out_of_range_and_bad_landmarks():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 1, "single_trackable", motion="static"),
    ], duration=1))
    with pytest.raises(GroundTruthValidationError, match="strictly increasing"):
        score_tracking_ground_truth(
            [_observation(0), _observation(0)], annotations, target_fps=15
        )
    with pytest.raises(GroundTruthValidationError, match="out of range"):
        score_tracking_ground_truth(
            [_observation(1)], annotations, target_fps=15
        )

    malformed = [
        TrackingObservation(index / 15, 1, 1.0, np.zeros((4, 3)))
        for index in range(15)
    ]
    metrics = score_tracking_ground_truth(
        malformed, annotations, target_fps=15,
        power_policy=_small_policy(),
    )
    assert metrics["trackable_coverage"] == 0.0


def test_timestamp_gap_breaks_dropout_and_reacquisition_runs():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 1, "no_face"),
        _interval(1, 3, "single_trackable", motion="translate"),
    ], duration=3))
    observations = [
        _observation(0, valid=False),
        _observation(1.0, valid=False),
        _observation(1.1, valid=False),
        _observation(2.0, valid=False),
        _observation(2.1),
        _observation(2.2),
        _observation(2.8),
    ]

    metrics = score_tracking_ground_truth(
        observations,
        annotations,
        target_fps=10,
        power_policy=_small_policy(),
    )

    assert metrics["dropout_run_count"] == 1
    assert metrics["reacquisition_success_count"] == 0


def test_invalid_power_policy_fails_before_scoring():
    annotations = parse_ground_truth_annotations(_document([
        _interval(0, 1, "no_face"),
    ], duration=1))
    with pytest.raises(GroundTruthValidationError, match="minimum_trackable_frames"):
        score_tracking_ground_truth(
            [],
            annotations,
            target_fps=15,
            power_policy=GroundTruthPowerPolicy(minimum_trackable_frames=0),
        )
