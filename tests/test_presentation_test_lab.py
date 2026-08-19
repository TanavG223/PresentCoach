import json
from pathlib import Path

from stroke_screening.presentation_test_lab import (
    test_lab_payload as build_test_lab_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    media = tmp_path / "media"
    reports = tmp_path / "reports"
    media.mkdir(parents=True)
    reports.mkdir()
    manifest = (ROOT / "test_media" / "face_tracking_manifest.json").read_bytes()
    (media / "face_tracking_manifest.json").write_bytes(manifest)
    report = json.loads(
        (ROOT / "reports" / "presentcoach_tracking_eval.json").read_text(
            encoding="utf-8"
        )
    )
    return media, reports, report


def _payload(media: Path, reports: Path, report: dict[str, object]):
    (reports / "presentcoach_tracking_eval.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return build_test_lab_payload(media_dir=media, reports_dir=reports)["tracking_eval"]


def test_tracking_report_is_manifest_locked_bounded_and_redacted(tmp_path: Path):
    media, reports, report = _fixture(tmp_path)
    manifest = json.loads((media / "face_tracking_manifest.json").read_text())
    first_manifest_case = manifest["clips"][0]
    (media / first_manifest_case["filename"]).write_bytes(b"installed-test-video")
    tracking = _payload(media, reports, report)

    assert tracking["trusted"] is True
    assert tracking["manifest_verified"] is True
    assert tracking["model_verified"] is True
    assert tracking["case_count"] == tracking["passed"] == 12
    assert tracking["dataset_roles"] == {
        "derived": 1, "development": 4, "holdout": 4, "regression": 3,
    }
    case = tracking["cases"][0]
    assert {
        "filename", "start_seconds", "evaluation_duration_seconds", "transform",
        "checks", "dropout_runs", "landmarks",
    }.isdisjoint(case)
    assert case["repeatability"] == {
        "runs": 2, "passed": True, "mismatch_count": 0,
    }
    assert case["declared_checks"]
    assert case["media"] == {
        "installed": True,
        "video_url": f"/api/test-media/{case['id']}/video",
        "playback_window": {
            "start_seconds": float(first_manifest_case["start_seconds"]),
            "duration_seconds": float(first_manifest_case["duration_seconds"]),
            "end_seconds": round(
                float(first_manifest_case["start_seconds"])
                + float(first_manifest_case["duration_seconds"]),
                3,
            ),
        },
        "represents_exact_evaluation_input": True,
        "representation": "exact_source_excerpt",
        "integrity": "sha256_verified_on_request",
    }
    assert str(media) not in json.dumps(tracking)

    derived = next(item for item in tracking["cases"] if item["dataset_role"] == "derived")
    assert derived["media"]["represents_exact_evaluation_input"] is False
    assert derived["media"]["representation"] == (
        "source_excerpt_before_in_memory_transform"
    )


def test_tracking_report_fails_closed_on_partial_or_unverified_evidence(tmp_path: Path):
    media, reports, report = _fixture(tmp_path)
    report["partial_run"] = True
    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []
    assert tracking["passed"] == 0

    media, reports, report = _fixture(tmp_path / "manifest-mismatch")
    with (media / "face_tracking_manifest.json").open("ab") as manifest:
        manifest.write(b"\n")
    tracking = _payload(media, reports, report)
    assert tracking["manifest_verified"] is False
    assert tracking["trusted"] is False
    assert tracking["cases"] == []


def test_tracking_pass_is_derived_from_typed_checks_not_raw_case_status(tmp_path: Path):
    media, reports, report = _fixture(tmp_path)
    first = report["cases"][0]
    assert isinstance(first, dict)
    first["passed"] = True
    checks = first["checks"]
    assert isinstance(checks, list)
    checks[-1]["passed"] = False

    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []


def test_tracking_report_requires_exact_manifest_case_and_expectation_sets(tmp_path: Path):
    media, reports, report = _fixture(tmp_path)
    first = report["cases"][0]
    assert isinstance(first, dict)
    checks = first["checks"]
    assert isinstance(checks, list)
    checks.pop()

    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []

    media, reports, report = _fixture(tmp_path / "metadata-mismatch")
    first = report["cases"][0]
    assert isinstance(first, dict)
    first["dataset_role"] = "holdout"
    report["dataset_roles"] = {
        "regression": 2, "development": 4, "holdout": 5, "derived": 1,
    }
    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []


def test_tracking_report_keeps_a_coherent_scenario_failure_visible(tmp_path: Path):
    media, reports, report = _fixture(tmp_path)
    first = report["cases"][0]
    assert isinstance(first, dict)
    checks = first["checks"]
    assert isinstance(checks, list)
    scenario_check = next(
        check for check in checks if check["id"] == "min_analyzed_fps"
    )
    measurements = first["measurements"]
    assert isinstance(measurements, dict)
    measurements["analyzed_fps"] = 14.0
    scenario_check["actual"] = 14.0
    scenario_check["passed"] = False
    first["passed"] = False
    report["passed"] = 11
    report["pass_rate_percent"] = 91.7

    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is True
    assert tracking["case_count"] == 12
    assert tracking["passed"] == 11
    assert tracking["pass_rate_percent"] == 91.7
    assert tracking["cases"][0]["passed"] is False


def test_tracking_report_recomputes_checks_from_measurements(tmp_path: Path):
    media, reports, report = _fixture(tmp_path)
    first = report["cases"][0]
    assert isinstance(first, dict)
    measurements = first["measurements"]
    assert isinstance(measurements, dict)
    measurements["valid_face_frame_ratio"] = 0.0

    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []


def test_tracking_report_binds_exact_excerpt_and_transform(tmp_path: Path):
    media, reports, report = _fixture(tmp_path / "start")
    first = report["cases"][0]
    assert isinstance(first, dict)
    first["start_seconds"] = float(first["start_seconds"]) + 1.0
    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []

    media, reports, report = _fixture(tmp_path / "duration")
    first = report["cases"][0]
    assert isinstance(first, dict)
    first["evaluation_duration_seconds"] = (
        float(first["evaluation_duration_seconds"]) - 1.0
    )
    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []

    media, reports, report = _fixture(tmp_path / "transform")
    derived = next(
        case
        for case in report["cases"]
        if case["media_id"] == "derived-two-face-abstention"
    )
    assert isinstance(derived, dict)
    derived["transform"] = {"type": "identity"}
    tracking = _payload(media, reports, report)
    assert tracking["trusted"] is False
    assert tracking["cases"] == []
