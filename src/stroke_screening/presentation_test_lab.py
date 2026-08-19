"""Public, reproducible evaluation evidence for the local PresentCoach UI."""

from __future__ import annotations

import json
from pathlib import Path


TEST_MEDIA = (
    {
        "id": "tarun-short-distance",
        "filename": "tarun-speaking-cc0.webm",
        "title": "Short, distant-face clip",
        "license": "CC0 1.0",
        "source_url": "https://commons.wikimedia.org/wiki/File:Tarun_speaking_01.webm",
        "purpose": "Verifies short-session refusal and face-quality abstention.",
    },
    {
        "id": "hawking-mixed-shot",
        "filename": "stephen-hawking-nasa-public-domain.webm",
        "title": "Mixed-shot NASA clip",
        "license": "U.S. public domain (NASA)",
        "source_url": "https://commons.wikimedia.org/wiki/File:StephenHawking-videoselection-2018.webm",
        "purpose": "Checks decoding, transcription, and honest face-presence quality flags across changing shots.",
    },
    {
        "id": "weekly-address",
        "filename": "weekly-address-public-domain-full.webm",
        "title": "Stable camera-facing address",
        "license": "U.S. federal public domain",
        "source_url": "https://commons.wikimedia.org/wiki/File:2015-11-21_President_Obama%27s_Weekly_Address.webm",
        "purpose": "Checks sustained 15 FPS vision analysis, clear local transcription, and stable face presence.",
    },
)
TEST_MEDIA_BY_ID = {item["id"]: item for item in TEST_MEDIA}


def _read_report(path: Path) -> dict[str, object] | None:
    try:
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def test_lab_payload(*, media_dir: Path, reports_dir: Path) -> dict[str, object]:
    """Return bounded UI evidence without exposing arbitrary local paths."""
    video_report = _read_report(reports_dir / "presentcoach_video_eval.json") or {}
    llm_report = _read_report(reports_dir / "presentcoach_llm_eval.json") or {}
    results = {
        str(item.get("media_id")): item
        for item in video_report.get("cases", [])
        if isinstance(item, dict) and isinstance(item.get("media_id"), str)
    }
    clips = []
    for source in TEST_MEDIA:
        path = media_dir / source["filename"]
        result = results.get(source["id"])
        clips.append({
            **source,
            "available": path.is_file(),
            "video_url": f"/api/test-media/{source['id']}/video" if path.is_file() else None,
            "result": result,
        })
    return {
        "evaluation_only": True,
        "used_for_training": False,
        "video_eval": {
            "case_count": int(video_report.get("case_count", 0) or 0),
            "passed": int(video_report.get("passed", 0) or 0),
            "pass_rate_percent": float(video_report.get("pass_rate_percent", 0) or 0),
            "generated_at": video_report.get("generated_at"),
        },
        "llm_eval": {
            "model": llm_report.get("model"),
            "case_count": int(llm_report.get("case_count", 0) or 0),
            "passed": int(llm_report.get("passed", 0) or 0),
            "pass_rate_percent": float(llm_report.get("pass_rate_percent", 0) or 0),
        },
        "clips": clips,
    }
