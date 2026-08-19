#!/usr/bin/env python3
"""Run the actual PresentCoach pipeline on the licensed local test clips."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from stroke_screening.presentation_audio import WhisperCppTranscriber
from stroke_screening.presentation_core import compute_metrics
from stroke_screening.presentation_test_lab import TEST_MEDIA
from stroke_screening.presentation_video import LocalVideoAnalyzer


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "tarun-short-distance": "338f79f313972a57d064c835ca2b7c12421a034bd6e8529c3834ce20e7c13923",
    "hawking-mixed-shot": "f342090520d248594a6819d7bee45960eeab9efac9c3551a0c73c76511cb5091",
    "weekly-address": "17127f6dcfbc7e0913d42d75a01f4bba37a36c205047ffccd661bad5674e5ba2",
}


def _check(label: str, passed: bool, actual: object, expected: str) -> dict[str, object]:
    return {"label": label, "passed": bool(passed), "actual": actual, "expected": expected}


def _checks(media_id: str, session, metrics: dict[str, object], digest: str) -> list[dict[str, object]]:
    aggregate = metrics["aggregate"]
    quality = session.quality_flags
    common = [
        _check("Exact licensed input", digest == EXPECTED_SHA256[media_id], digest[:12], "SHA-256 matches manifest"),
        _check("Video decoded", session.duration_seconds > 0, round(session.duration_seconds, 2), "duration > 0 seconds"),
        _check("Vision throughput", aggregate["analyzed_vision_fps"] >= 14.5, aggregate["analyzed_vision_fps"], ">= 14.5 analyzed FPS"),
    ]
    if media_id == "tarun-short-distance":
        return common + [
            _check("Short-session guardrail", session.duration_seconds < 30, round(session.duration_seconds, 2), "under 30 seconds"),
            _check("Face data abstains", quality["face_detected"] == "bad", quality["face_detected"], "bad / insufficient"),
        ]
    if media_id == "hawking-mixed-shot":
        return common + [
            _check("Feedback-duration input", session.duration_seconds >= 30, round(session.duration_seconds, 2), ">= 30 seconds"),
            _check("Speech timestamped", len(session.transcript) >= 10, len(session.transcript), ">= 10 words"),
            _check("Quality flags produced", len(quality) == 5, len(quality), "5 per-metric flags"),
        ]
    return common + [
        _check("Sustained-duration input", session.duration_seconds >= 240, round(session.duration_seconds, 2), ">= 240 seconds"),
        _check("Face presence", aggregate["face_presence_percent"] >= 90, aggregate["face_presence_percent"], ">= 90%"),
        _check("Speech timestamped", len(session.transcript) >= 400, len(session.transcript), ">= 400 words"),
        _check("All metric inputs usable", all(value == "good" for value in quality.values()), dict(quality), "all 5 flags good"),
    ]


def main() -> int:
    analyzer = LocalVideoAnalyzer(
        model_path=ROOT / "models" / "face_landmarker.task",
        transcriber=WhisperCppTranscriber(
            model_path=ROOT / "models" / "whisper" / "ggml-base.en-q5_1.bin"
        ),
    )
    results = []
    for media in TEST_MEDIA:
        path = ROOT / "test_media" / media["filename"]
        if not path.is_file():
            results.append({
                "media_id": media["id"], "title": media["title"], "passed": False,
                "error": "clip missing; run zsh scripts/download_test_videos.sh",
            })
            continue
        print(f"ANALYZE {media['title']}", flush=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        session = analyzer.analyze(path, note="Reproducible evaluation only; never training data")
        metrics = compute_metrics(session)
        checks = _checks(media["id"], session, metrics, digest)
        result = {
            "media_id": media["id"],
            "title": media["title"],
            "passed": all(item["passed"] for item in checks),
            "measurements": {
                "duration_seconds": round(session.duration_seconds, 2),
                "analyzed_vision_fps": metrics["aggregate"]["analyzed_vision_fps"],
                "face_presence_percent": metrics["aggregate"]["face_presence_percent"],
                "transcript_word_count": len(session.transcript),
                "quality_flags": dict(session.quality_flags),
            },
            "checks": checks,
        }
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {media['id']}", flush=True)
    passed = sum(bool(item["passed"]) for item in results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_only": True,
        "used_for_training": False,
        "case_count": len(results),
        "passed": passed,
        "pass_rate_percent": round(100.0 * passed / len(results), 1),
        "cases": results,
    }
    output = ROOT / "reports" / "presentcoach_video_eval.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "passed", "pass_rate_percent")}), flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
