import json
from pathlib import Path
import shutil

import pytest

from presentcoach.presentation_test_lab import (
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
    return build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={
            "available": False,
            "installed": False,
            "model": "presentcoach-local",
            "digest": "",
        },
    )["tracking_eval"]


def _dynamic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "media"
    reports = tmp_path / "reports"
    (media / "annotations").mkdir(parents=True)
    reports.mkdir()
    shutil.copy2(
        ROOT / "test_media" / "dynamic_presentation_manifest.json",
        media / "dynamic_presentation_manifest.json",
    )
    for annotation in (ROOT / "test_media" / "annotations").glob("*.json"):
        shutil.copy2(annotation, media / "annotations" / annotation.name)
    return media, reports


def test_dynamic_split_absence_preserves_untouched_holdout_state(tmp_path: Path):
    media, reports = _dynamic_fixture(tmp_path)

    dynamic = build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={
            "available": False,
            "installed": False,
            "model": "presentcoach-local",
            "digest": "",
        },
    )["dynamic_eval"]

    assert dynamic["development"]["state"] == "not_run"
    assert dynamic["development"]["trusted"] is False
    assert dynamic["development"]["expected_case_count"] == 10
    assert dynamic["holdout"]["state"] == "untouched"
    assert dynamic["holdout"]["trusted"] is False
    assert dynamic["holdout"]["case_count"] == 0
    assert dynamic["holdout"]["expected_case_count"] == 5
    assert "no report has been run or read" in dynamic["holdout"]["trust_message"]
    assert all(item["result"] is None for item in dynamic["holdout"]["cases"])
    assert dynamic["claim_boundary"].startswith("Single-curator observed-state")
    assert "population accuracy" in dynamic["claim_boundary"]
    assert "gaze accuracy" in dynamic["claim_boundary"]
    assert "training evidence" in dynamic["claim_boundary"]
    assert str(media) not in json.dumps(dynamic)


def test_dynamic_report_presence_fails_closed_without_changing_holdout(tmp_path: Path):
    media, reports = _dynamic_fixture(tmp_path)
    (reports / "presentcoach_dynamic_development.json").write_text(
        '{"schema_version":1,"case_count":5}', encoding="utf-8"
    )

    dynamic = build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={},
    )["dynamic_eval"]

    assert dynamic["development"]["state"] == "untrusted"
    assert dynamic["development"]["case_count"] == 0
    assert dynamic["development"]["aggregate"] == {}
    assert all(item["result"] is None for item in dynamic["development"]["cases"])
    assert dynamic["holdout"]["state"] == "untouched"


def test_dynamic_development_report_recomputes_and_exposes_bounded_evidence(
    tmp_path: Path,
):
    media, reports = _dynamic_fixture(tmp_path)
    shutil.copy2(
        ROOT / "reports" / "presentcoach_dynamic_development.json",
        reports / "presentcoach_dynamic_development.json",
    )

    dynamic = build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={},
    )["dynamic_eval"]
    development = dynamic["development"]

    assert development["trusted"] is True
    assert development["manifest_verified"] is True
    assert development["model_verified"] is True
    assert development["annotations_verified"] is True
    assert development["case_count"] == development["expected_case_count"] == 10
    assert development["state"] in {"pass", "fail", "not_evaluated"}
    assert len(development["checks"]) == 6
    assert set(development["aggregate"]) == {
        "observation_count",
        "labeled_observation_count",
        "reacquisition_event_count",
        "trackable_coverage",
        "false_valid_rate",
        "multi_face_abstention",
        "dropouts_per_annotated_minute",
        "reacquisition_success_rate",
        "max_reacquisition_latency_seconds",
    }
    first = development["cases"][0]
    assert first["result"] is not None
    assert set(first["result"]) == {
        "evaluation_duration_seconds", "tracking", "observed_state",
        "repeatability",
    }
    serialized = json.dumps(development)
    assert str(media) not in serialized
    assert "annotation_file" not in serialized
    assert "filename" not in serialized
    assert "dropout_runs" not in serialized
    assert "landmarks" not in serialized
    assert dynamic["holdout"]["state"] == "untouched"
    assert dynamic["holdout"]["case_count"] == 0


def test_dynamic_one_shot_holdout_failure_remains_trusted_and_visible(tmp_path: Path):
    media, reports = _dynamic_fixture(tmp_path)
    shutil.copy2(
        ROOT / "reports" / "presentcoach_dynamic_holdout.json",
        reports / "presentcoach_dynamic_holdout.json",
    )

    holdout = build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={},
    )["dynamic_eval"]["holdout"]

    assert holdout["trusted"] is True
    assert holdout["state"] == holdout["release_gate_status"] == "fail"
    assert holdout["case_count"] == holdout["expected_case_count"] == 5
    assert holdout["failed_required_metrics"] == [
        "false_valid_rate",
        "dropouts_per_annotated_minute",
    ]
    assert holdout["underpowered_required_metrics"] == [
        "multi_face_abstention"
    ]
    assert all(item["result"] is not None for item in holdout["cases"])


@pytest.mark.parametrize("tamper", ["aggregate", "gate", "configuration", "source"])
def test_dynamic_development_report_fails_closed_on_tampering(
    tmp_path: Path, tamper: str,
):
    media, reports = _dynamic_fixture(tmp_path)
    report = json.loads(
        (ROOT / "reports" / "presentcoach_dynamic_development.json").read_text(
            encoding="utf-8"
        )
    )
    if tamper == "aggregate":
        report["aggregate"]["observation_count"] += 1
    elif tamper == "gate":
        report["release_gate"]["checks"][0]["passed"] = not bool(
            report["release_gate"]["checks"][0]["passed"]
        )
    elif tamper == "configuration":
        report["configuration"]["face_landmarker"][
            "min_tracking_confidence"
        ] = 0.01
    else:
        report["source_sha256"]["landmarks.py"] = "0" * 64
    (reports / "presentcoach_dynamic_development.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    development = build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={},
    )["dynamic_eval"]["development"]

    assert development["trusted"] is False
    assert development["state"] == "untrusted"
    assert development["case_count"] == 0
    assert development["aggregate"] == {}
    assert development["checks"] == []
    assert all(item["result"] is None for item in development["cases"])


def test_dynamic_report_rejects_an_installed_media_hash_mismatch(tmp_path: Path):
    media, reports = _dynamic_fixture(tmp_path)
    shutil.copy2(
        ROOT / "reports" / "presentcoach_dynamic_development.json",
        reports / "presentcoach_dynamic_development.json",
    )
    manifest = json.loads(
        (media / "dynamic_presentation_manifest.json").read_text(encoding="utf-8")
    )
    development_clip = next(
        item for item in manifest["clips"] if item["dataset_role"] == "development"
    )
    (media / development_clip["filename"]).write_bytes(b"tampered canonical media")

    development = build_test_lab_payload(
        media_dir=media,
        reports_dir=reports,
        local_model_status={},
    )["dynamic_eval"]["development"]

    assert development["trusted"] is False
    assert development["state"] == "untrusted"
    matching = next(
        item for item in development["cases"] if item["id"] == development_clip["id"]
    )
    assert matching["media"]["installed"] is False
    assert matching["media"]["video_url"] is None


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
