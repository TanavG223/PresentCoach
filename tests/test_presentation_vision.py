from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import stroke_screening.presentation_vision as presentation_vision


class TimestampEngine:
    def __init__(self, _path: Path) -> None:
        self.timestamps: list[int | None] = []

    def detect(self, _frame, timestamp_ms=None):
        self.timestamps.append(timestamp_ms)
        return SimpleNamespace(
            landmarks=None,
            blendshapes={},
            face_count=0,
            inference_ms=1.0,
            transformation_matrix=None,
        )

    def close(self) -> None:
        pass


def test_vision_analyzer_uses_media_timestamps_for_deterministic_smoothing(monkeypatch):
    monkeypatch.setattr(presentation_vision, "FaceLandmarkEngine", TimestampEngine)
    analyzer = presentation_vision.PresentationVisionAnalyzer(Path("unused.task"))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    analyzer.process(frame, 0.0)
    analyzer.process(frame, 1 / 15)
    analyzer.process(frame, 2 / 15)

    assert analyzer._engine.timestamps == [0, 67, 133]


def _face_points() -> np.ndarray:
    points = np.full((478, 3), 0.5, dtype=np.float64)
    points[33, :2], points[133, :2], points[468, :2] = (
        (0.40, 0.50), (0.50, 0.50), (0.45, 0.50)
    )
    points[362, :2], points[263, :2], points[473, :2] = (
        (0.50, 0.50), (0.60, 0.50), (0.55, 0.50)
    )
    points[159, 1], points[145, 1] = 0.45, 0.55
    points[386, 1], points[374, 1] = 0.45, 0.55
    return points


def _detection(*, matrix=True, blendshapes=None, landmarks=True):
    return SimpleNamespace(
        landmarks=_face_points() if landmarks else None,
        blendshapes=blendshapes or {},
        face_count=1 if landmarks else 0,
        inference_ms=1.0,
        transformation_matrix=np.eye(4) if matrix else None,
    )


class SequenceEngine:
    def __init__(self, detections) -> None:
        self.detections = iter(detections)

    def detect(self, _frame, timestamp_ms=None):
        del timestamp_ms
        return next(self.detections)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("detection", "gaze_available"),
    (
        (_detection(matrix=False), True),
        (_detection(blendshapes={"eyeBlinkLeft": 0.9}), False),
    ),
)
def test_contact_excludes_missing_pose_and_blinked_frames(
    monkeypatch, detection, gaze_available
):
    engine = SequenceEngine((detection,))
    monkeypatch.setattr(
        presentation_vision, "FaceLandmarkEngine", lambda _path: engine
    )
    analyzer = presentation_vision.PresentationVisionAnalyzer(Path("unused.task"))
    analyzer.process(np.zeros((4, 4, 3), dtype=np.uint8), 0.0)

    sample = analyzer.finish(1.0)[0]
    assert sample.eye_contact is None
    assert sample.contact_eligible_frame_count == 0
    assert (sample.gaze_horizontal is not None) is gaze_available
    assert (sample.gaze_vertical is not None) is gaze_available


def test_degenerate_eye_geometry_is_invalid_instead_of_center_contact():
    points = _face_points()
    points[133, :2] = points[33, :2]
    assert presentation_vision._eye_ratio(points, 33, 133, 468) is None

    points[145, 1] = points[159, 1]
    assert presentation_vision._eye_vertical(points, 468, (159, 145)) is None


@pytest.mark.parametrize(
    "rotation",
    (
        np.asarray(((np.nan, 0, 0), (0, 1, 0), (0, 0, 1))),
        np.asarray(((np.inf, 0, 0), (0, 1, 0), (0, 0, 1))),
        np.zeros((3, 3)),
        np.eye(3) * 1.25,
    ),
)
def test_head_angles_reject_corrupt_or_nonrigid_transforms(rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    assert presentation_vision._head_angles(matrix) == (None, None, None)


def test_expression_delta_resets_after_tracking_gap(monkeypatch):
    engine = SequenceEngine((
        _detection(blendshapes={"jawOpen": 0.1}),
        _detection(landmarks=False),
        _detection(blendshapes={"jawOpen": 0.9}),
    ))
    monkeypatch.setattr(
        presentation_vision, "FaceLandmarkEngine", lambda _path: engine
    )
    analyzer = presentation_vision.PresentationVisionAnalyzer(Path("unused.task"))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    analyzer.process(frame, 0.1)
    analyzer.process(frame, 1.1)
    analyzer.process(frame, 2.1)

    samples = analyzer.finish(3.0)
    assert samples[0].expression_change == 0.0
    assert samples[2].expression_change == 0.0


def test_sparse_gaze_eligibility_does_not_create_a_bucket_contact_claim(monkeypatch):
    engine = SequenceEngine((
        _detection(),
        *(
            _detection(blendshapes={"eyeBlinkLeft": 0.9})
            for _ in range(14)
        ),
    ))
    monkeypatch.setattr(
        presentation_vision, "FaceLandmarkEngine", lambda _path: engine
    )
    analyzer = presentation_vision.PresentationVisionAnalyzer(Path("unused.task"))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for index in range(15):
        analyzer.process(frame, index / 15)

    sample = analyzer.finish(1.0)[0]
    assert sample.face_detected is True
    assert sample.contact_eligible_frame_count == 1
    assert sample.contact_frame_count == 1
    assert sample.eye_contact is None


def test_sparse_face_presence_does_not_create_a_bucket_contact_claim(monkeypatch):
    engine = SequenceEngine((
        _detection(),
        *(_detection(landmarks=False) for _ in range(14)),
    ))
    monkeypatch.setattr(
        presentation_vision, "FaceLandmarkEngine", lambda _path: engine
    )
    analyzer = presentation_vision.PresentationVisionAnalyzer(Path("unused.task"))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for index in range(15):
        analyzer.process(frame, index / 15)

    sample = analyzer.finish(1.0)[0]
    assert sample.face_detected is False
    assert sample.detected_frame_count == 1
    assert sample.contact_eligible_frame_count == 1
    assert sample.eye_contact is None
