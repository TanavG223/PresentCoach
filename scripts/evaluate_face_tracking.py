#!/usr/bin/env python3
"""Run the manifest-locked, identity-free face-tracking evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
from pathlib import Path

from stroke_screening.presentation_tracking_eval import (
    TrackingEvaluationError,
    analyze_tracking_clip,
    build_tracking_checks,
    deterministic_metric_projection,
    load_tracking_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "test_media" / "face_tracking_manifest.json"
DEFAULT_OUTPUT = ROOT / "reports" / "presentcoach_tracking_eval.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path; partial runs default to a separate .partial.json file.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the named case (repeatable); the report is then marked partial.",
    )
    parser.add_argument(
        "--repeatability-runs",
        type=int,
        default=2,
        help="Analyze each case this many times and require identical deterministic metrics (2-5).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if not 2 <= arguments.repeatability_runs <= 5:
        raise SystemExit("--repeatability-runs must be between 2 and 5")
    manifest_path = arguments.manifest.resolve()
    manifest, specs = load_tracking_manifest(manifest_path)
    selected_ids = set(arguments.only)
    if selected_ids:
        unknown = selected_ids - {spec.clip_id for spec in specs}
        if unknown:
            raise SystemExit(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        specs = tuple(spec for spec in specs if spec.clip_id in selected_ids)

    model_document = manifest["model"]
    if not isinstance(model_document, dict):
        raise TrackingEvaluationError("Manifest model entry is invalid")
    model_path = ROOT / "models" / str(model_document["filename"])
    expected_model_digest = str(model_document["sha256"]).lower()
    actual_model_digest = sha256_file(model_path) if model_path.is_file() else None
    model_verified = actual_model_digest == expected_model_digest
    target_fps = float(manifest["target_fps"])

    results: list[dict[str, object]] = []
    for spec in specs:
        print(f"ANALYZE {spec.clip_id}", flush=True)
        media_path = ROOT / "test_media" / spec.filename
        result: dict[str, object] = {
            "media_id": spec.clip_id,
            "filename": spec.filename,
            "source_url": spec.source_url,
            "source_sha1": spec.source_sha1,
            "license": spec.license_name,
            "purpose": spec.purpose,
            "dataset_role": spec.dataset_role,
            "transform": dict(spec.transform),
        }
        try:
            media_digest = sha256_file(media_path)
            digest_check = {
                "id": "exact_media_sha256",
                "label": "Exact licensed input",
                "passed": media_digest == spec.sha256,
                "actual": media_digest,
                "expected": spec.sha256,
            }
            model_check = {
                "id": "exact_model_sha256",
                "label": "Exact MediaPipe model artifact",
                "passed": model_verified,
                "actual": actual_model_digest,
                "expected": expected_model_digest,
            }
            if not digest_check["passed"] or not model_check["passed"]:
                result.update({"passed": False, "checks": [digest_check, model_check]})
            else:
                metrics, evaluated_duration = analyze_tracking_clip(
                    media_path,
                    model_path=model_path,
                    start_seconds=spec.start_seconds,
                    duration_seconds=spec.duration_seconds,
                    target_fps=target_fps,
                    transform=spec.transform,
                )
                reference_projection = deterministic_metric_projection(metrics)
                repeat_mismatches: list[dict[str, object]] = []
                for repeat_index in range(1, arguments.repeatability_runs):
                    repeated_metrics, repeated_duration = analyze_tracking_clip(
                        media_path,
                        model_path=model_path,
                        start_seconds=spec.start_seconds,
                        duration_seconds=spec.duration_seconds,
                        target_fps=target_fps,
                        transform=spec.transform,
                    )
                    repeated_projection = deterministic_metric_projection(repeated_metrics)
                    differing_keys = sorted(
                        key
                        for key in set(reference_projection) | set(repeated_projection)
                        if reference_projection.get(key) != repeated_projection.get(key)
                    )
                    if repeated_duration != evaluated_duration or differing_keys:
                        repeat_mismatches.append({
                            "run": repeat_index + 1,
                            "duration_changed": repeated_duration != evaluated_duration,
                            "differing_metrics": differing_keys,
                        })
                repeatability_check = {
                    "id": "exact_repeatability",
                    "label": "Exact repeated tracking metrics",
                    "passed": not repeat_mismatches,
                    "actual": repeat_mismatches,
                    "expected": "no deterministic metric differences",
                }
                checks = [
                    digest_check,
                    model_check,
                    repeatability_check,
                    *build_tracking_checks(metrics, spec.expectations),
                ]
                result.update({
                    "passed": all(bool(check["passed"]) for check in checks),
                    "start_seconds": spec.start_seconds,
                    "evaluation_duration_seconds": round(evaluated_duration, 4),
                    "measurements": metrics,
                    "repeatability": {
                        "runs": arguments.repeatability_runs,
                        "passed": not repeat_mismatches,
                        "mismatches": repeat_mismatches,
                    },
                    "checks": checks,
                })
        except (OSError, TrackingEvaluationError) as error:
            result.update({"passed": False, "error": str(error), "checks": []})
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {spec.clip_id}", flush=True)

    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_only": True,
        "used_for_training": False,
        "identity_recognition": False,
        "landmarks_persisted": False,
        "partial_run": bool(selected_ids),
        "repeatability_runs_per_case": arguments.repeatability_runs,
        "target_fps": target_fps,
        "mediapipe_version": package_version("mediapipe"),
        "opencv_version": package_version("opencv-contrib-python"),
        "model": {
            "filename": model_document["filename"],
            "sha256": actual_model_digest,
            "matches_manifest": model_verified,
        },
        "dataset_roles": {
            role: sum(result.get("dataset_role") == role for result in results)
            for role in ("regression", "development", "holdout", "derived")
        },
        "metric_definitions": {
            "analyzed_fps": "sampled frames divided by evaluated media duration; not wall-clock render FPS",
            "valid_face_frame_ratio": "frames containing exactly one face and valid landmarks divided by analyzed frames",
            "dropout_run": "contiguous analyzed frames without a valid single-face landmark track",
            "reacquisition_seconds": "time from the first invalid frame in a bracketed dropout to the next valid frame",
            "landmark_jitter": "frame-to-frame RMS movement of rigid landmarks after eye-line alignment and inter-eye normalization, only in stable runs",
            "multi_face_abstention": "frames with more than one detected face must expose no landmarks to downstream scoring",
        },
        "case_count": len(results),
        "passed": passed,
        "pass_rate_percent": round(100.0 * passed / len(results), 1) if results else 0.0,
        "cases": results,
    }
    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else (
            ROOT / "reports" / "presentcoach_tracking_eval.partial.json"
            if selected_ids
            else DEFAULT_OUTPUT
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_count": report["case_count"],
        "passed": report["passed"],
        "pass_rate_percent": report["pass_rate_percent"],
        "partial_run": report["partial_run"],
    }), flush=True)
    return 0 if passed == len(results) and model_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
