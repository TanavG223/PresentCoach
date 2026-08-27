from types import SimpleNamespace

import numpy as np
import pytest

from stroke_screening.landmarks import (
    FaceLandmarkEngine,
    _FaceCandidate,
    _deduplicate_candidates,
    _intersection_over_union,
    _eye_visibility_ratio,
    _multiscale_crops,
    _proposal_crop,
)


def _candidate(bounds):
    left, top, right, bottom = bounds
    points = np.zeros((478, 3), dtype=np.float64)
    points[:, 0] = (left + right) / 2
    points[:, 1] = (top + bottom) / 2
    return _FaceCandidate(
        landmarks=points,
        blendshapes={},
        transformation_matrix=None,
        bounds=bounds,
        area=(right - left) * (bottom - top),
    )


def test_multiscale_candidate_deduplication_keeps_distinct_people():
    first = _candidate((0.10, 0.10, 0.30, 0.40))
    duplicate = _candidate((0.11, 0.11, 0.31, 0.41))
    second_person = _candidate((0.60, 0.10, 0.80, 0.40))

    unique = _deduplicate_candidates([duplicate, second_person, first])

    assert len(unique) == 2
    assert any(item.bounds == second_person.bounds for item in unique)
    assert _intersection_over_union(first.bounds, duplicate.bounds) > 0.25
    assert _intersection_over_union(first.bounds, second_person.bounds) == 0


def test_crop_landmarks_map_back_to_full_frame_with_blendshape_and_pose():
    landmarks = [
        SimpleNamespace(
            x=0.25 + (index % 2) * 0.10,
            y=0.50 + (index % 3) * 0.05,
            z=0.10,
        )
        for index in range(478)
    ]
    result = SimpleNamespace(
        face_landmarks=[landmarks],
        face_blendshapes=[[
            SimpleNamespace(category_name="jawOpen", score=0.75)
        ]],
        facial_transformation_matrixes=[np.eye(4)],
    )

    candidate = FaceLandmarkEngine._candidate_from_result(
        result,
        0,
        crop=(100, 50, 300, 250),
        frame_size=(400, 300),
    )

    assert candidate is not None
    assert candidate.landmarks.shape == (478, 3)
    assert candidate.landmarks[0] == pytest.approx((0.375, 0.5, 0.05))
    assert candidate.blendshapes == {"jawOpen": 0.75}
    assert np.array_equal(candidate.transformation_matrix, np.eye(4))


def test_invalid_crop_candidate_fails_closed():
    result = SimpleNamespace(
        face_landmarks=[[
            SimpleNamespace(x=float("nan"), y=0.5, z=0.0)
            for _ in range(478)
        ]],
        face_blendshapes=[],
        facial_transformation_matrixes=[],
    )
    assert FaceLandmarkEngine._candidate_from_result(
        result,
        0,
        crop=(0, 0, 100, 100),
        frame_size=(100, 100),
    ) is None


def test_legacy_crop_grid_is_bounded_and_covers_the_frame():
    crops = _multiscale_crops((854, 480))

    assert len(crops) == 9
    assert len(set(crops)) == 9
    assert all(0 <= x0 < x1 <= 854 for x0, _y0, x1, _y1 in crops)
    assert all(0 <= y0 < y1 <= 480 for _x0, y0, _x1, y1 in crops)

    covered = np.zeros((480, 854), dtype=bool)
    for x0, y0, x1, y1 in crops:
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_yunet_proposal_crop_is_bounded_and_rejects_bad_boxes():
    detection = np.asarray([250.2, 130.4, 32.0, 40.0, *([0.0] * 10)])

    assert _proposal_crop(detection, (854, 480), 1.5) == (236, 120, 297, 181)
    assert _proposal_crop(np.asarray([0, 0, -1, 10]), (854, 480), 1.5) is None
    assert _proposal_crop(np.asarray([np.nan, 0, 10, 10]), (854, 480), 1.5) is None


def test_eye_visibility_ratio_rejects_shadowed_eyes():
    frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    detection = np.asarray(
        [20, 20, 60, 60, 35, 40, 65, 40, 50, 55, 38, 68, 62, 68, 0.95],
        dtype=np.float32,
    )
    frame[33:48, 28:43] = 10
    frame[33:48, 58:73] = 10

    ratio = _eye_visibility_ratio(frame, detection)

    assert ratio is not None
    assert ratio < 0.5


def test_temporal_hold_is_single_frame_bounded_and_requires_one_face():
    engine = FaceLandmarkEngine.__new__(FaceLandmarkEngine)
    face = _candidate((0.2, 0.2, 0.4, 0.5))
    engine._last_fresh_candidate = face
    engine._last_fresh_candidate_timestamp_ms = 1_000
    engine._proposal_confirms_single_face = True

    assert engine._temporal_hold_candidate(1_067) is face
    assert engine._temporal_hold_candidate(1_081) is None
    engine._proposal_confirms_single_face = False
    assert engine._temporal_hold_candidate(1_067) is None
