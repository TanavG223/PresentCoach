"""Personal baseline and repeatability gates for presentation measurements."""

from __future__ import annotations

from typing import Mapping, Sequence

from .presentation_coaching import general_coaching_hints
from .presentation_core import compute_metrics, feedback_metric_has_good_quality
from .presentation_store import PresentationArchive, StoredPresentation


REPEATABILITY_TOLERANCES: dict[str, float] = {
    "eye_contact_percent": 8.0,
    "head_rotation_std_degrees": 2.0,
    "head_position_std_percent": 1.5,
    "expression_variety_index": 3.0,
    "face_presence_percent": 5.0,
    "overall_words_per_minute": 12.0,
    "filler_count": 2.0,
    "pauses_over_2_seconds": 2.0,
}


def _quality_approved(stored: StoredPresentation) -> bool:
    flags = compute_metrics(stored.session)["quality_flags"]
    return (
        stored.session.duration_seconds >= 30.0
        and all(flags.get(metric) == "good" for metric in (
            "face_detected", "eye_contact", "head_stability",
            "expression_variety", "audio_clear",
        ))
    )


def calibration_status(archive: PresentationArchive) -> dict[str, object]:
    baseline_id = archive.calibration.get("baseline_session_id")
    baseline = next(
        (item for item in archive.sessions if item.session.session_id == baseline_id),
        None,
    )
    if baseline is None:
        candidate = next(
            (item for item in archive.sessions if item.session.session_kind == "baseline"),
            None,
        )
        candidate_approved = candidate is not None and _quality_approved(candidate)
        return {
            "stage": "review_baseline" if candidate_approved else "record_baseline",
            "ready": False,
            "baseline": candidate.metrics.get("aggregate") if candidate else None,
            "baseline_session_id": candidate.session.session_id if candidate else None,
            "message": (
                "Record at least 30 seconds to inspect your raw baseline numbers."
                if candidate is None
                else "That baseline was too short or had insufficient face/audio data. Record it again."
                if not candidate_approved
                else "Review the raw baseline numbers, then confirm this reference."
            ),
        }
    if not bool(archive.calibration.get("baseline_confirmed")):
        if not _quality_approved(baseline):
            return {
                "stage": "record_baseline",
                "ready": False,
                "baseline": baseline.metrics.get("aggregate"),
                "baseline_session_id": baseline.session.session_id,
                "message": "That baseline was too short or had insufficient face/audio data. Record it again.",
            }
        return {
            "stage": "review_baseline",
            "ready": False,
            "baseline": baseline.metrics.get("aggregate"),
            "baseline_session_id": baseline.session.session_id,
            "message": "Review the raw baseline numbers, then confirm this reference.",
        }
    repeats = [
        item for item in archive.sessions
        if item.session.session_kind == "repeat" and _quality_approved(item)
    ]
    if len(repeats) < 2:
        return {
            "stage": "record_repeats",
            "ready": False,
            "baseline": baseline.metrics.get("aggregate"),
            "baseline_session_id": baseline.session.session_id,
            "repeat_count": len(repeats),
            "message": f"Record two nearly identical repeats; {len(repeats)}/2 quality-approved.",
        }
    left, right = repeats[-2:]
    left_aggregate = left.metrics.get("aggregate", {})
    right_aggregate = right.metrics.get("aggregate", {})
    deltas: dict[str, dict[str, object]] = {}
    passed = True
    for metric, tolerance in REPEATABILITY_TOLERANCES.items():
        first = float(left_aggregate.get(metric, 0))
        second = float(right_aggregate.get(metric, 0))
        delta = abs(second - first)
        item_passed = delta <= tolerance
        passed = passed and item_passed
        deltas[metric] = {
            "first": round(first, 2),
            "second": round(second, 2),
            "delta": round(delta, 2),
            "tolerance": tolerance,
            "passed": item_passed,
        }
    return {
        "stage": "ready" if passed else "repeatability_failed",
        "ready": passed,
        "baseline": baseline.metrics.get("aggregate"),
        "baseline_session_id": baseline.session.session_id,
        "repeat_count": len(repeats),
        "repeatability": deltas,
        "message": (
            "Personal baseline and repeatability check are ready."
            if passed
            else "The last two repeats were not close enough; record another matched repeat."
        ),
    }


def prepare_feedback_metrics(
    metrics: Mapping[str, object], calibration: Mapping[str, object]
) -> dict[str, object]:
    prepared = dict(metrics)
    prepared["calibration_ready"] = calibration.get("ready") is True
    if calibration.get("ready") is not True:
        prepared["feedback_mode"] = "general_practice"
        prepared["role_hints"] = general_coaching_hints(metrics)
        return prepared
    prepared["feedback_mode"] = "personal_reference"
    reference = calibration.get("baseline", {})
    aggregate = metrics.get("aggregate", {})
    quality = metrics.get("quality_flags", {})
    if (
        not isinstance(reference, Mapping)
        or not isinstance(aggregate, Mapping)
        or not isinstance(quality, Mapping)
    ):
        return prepared
    positive_high = {"eye_contact_percent", "face_presence_percent"}
    negative_low = {
        "head_rotation_std_degrees", "head_position_std_percent",
        "filler_count", "pauses_over_2_seconds",
    }
    role_hints: list[dict[str, object]] = []
    for metric, tolerance in REPEATABILITY_TOLERANCES.items():
        if (
            metric not in reference
            or metric not in aggregate
            or not feedback_metric_has_good_quality(metric, quality)
        ):
            continue
        current = float(aggregate[metric])
        baseline = float(reference[metric])
        delta = current - baseline
        if metric in positive_high:
            role = "strength" if delta >= -tolerance else "improvement"
        elif metric in negative_low:
            role = "strength" if delta <= tolerance else "improvement"
        else:
            role = "strength" if abs(delta) <= tolerance else "improvement"
        role_hints.append({
            "metric": metric,
            "role": role,
            "current": round(current, 2),
            "personal_reference": round(baseline, 2),
            "repeatability_tolerance": tolerance,
            "delta": round(delta, 2),
        })
    prepared["personal_reference"] = dict(reference)
    prepared["role_hints"] = role_hints
    return prepared
