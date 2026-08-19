"""Deterministic, evidence-backed cues for post-session video review.

The browser may seek a recording to ``seek_seconds`` and display ``text`` beside
the video.  This module does not classify personality, confidence, or whether a
speaker is reading.  It only turns stored measurements into timestamped cues.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from numbers import Real
from statistics import median
from typing import Mapping, Sequence

from .presentation_core import PROHIBITED_FEEDBACK_TERMS, StructuredFeedback


REVIEW_CUE_SCHEMA = "presentcoach.review-cues.v1"
LONG_TRANSCRIPT_GAP_SECONDS = 2.0
# This is a display floor for normalized blendshape movement, not a speaking
# quality threshold. It only suppresses zero-valued buckets and numeric noise.
MIN_REVIEW_EXPRESSION_CHANGE = 0.001


@dataclass(frozen=True)
class ReviewCue:
    """One measured review note, optionally tied to a seekable time range."""

    cue_id: str
    kind: str
    role: str
    start_seconds: float
    end_seconds: float
    seek_seconds: float
    title: str
    text: str
    metric: str
    value: float | None
    unit: str | None
    source: str
    evidence: dict[str, object]
    seekable: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_KIND_ORDER = {
    "quality": 0,
    "face_tracking_gap": 1,
    "strict_filler": 2,
    "camera_contact_break": 3,
    "expression_movement": 4,
    "pace_spike": 5,
    "transcript_gap": 6,
    "verified_coaching": 7,
}

_QUALITY_BY_FEEDBACK_METRIC = {
    "eye_contact_percent": "eye_contact",
    "longest_gaze_break_seconds": "eye_contact",
    "face_presence_percent": "face_detected",
    "head_rotation_std_degrees": "head_stability",
    "head_position_std_percent": "head_stability",
    "expression_variety_index": "expression_variety",
    "overall_words_per_minute": "audio_clear",
    "window_words_per_minute": "audio_clear",
    "filler_count": "audio_clear",
    "filler_rate_per_minute": "audio_clear",
    "strict_filler_count": "audio_clear",
    "strict_filler_rate_per_minute": "audio_clear",
    "strict_filler_cluster_count": "audio_clear",
    "pauses_over_2_seconds": "audio_clear",
    "pauses_over_3_seconds": "audio_clear",
    "long_pause_rate_per_minute": "audio_clear",
    "longest_pause_seconds": "audio_clear",
}

_AGGREGATE_FEEDBACK_UNITS = {
    "eye_contact_percent": "%",
    "head_rotation_std_degrees": "degrees",
    "head_position_std_percent": "% frame",
    "expression_variety_index": "index",
    "face_presence_percent": "%",
    "overall_words_per_minute": "WPM",
    "filler_count": "fillers",
    "filler_rate_per_minute": "fillers/min",
    "strict_filler_count": "um/uh fillers",
    "strict_filler_rate_per_minute": "um/uh per min",
    "pauses_over_2_seconds": "pauses",
    "pauses_over_3_seconds": "pauses",
    "long_pause_rate_per_minute": "pauses/min",
    "longest_pause_seconds": "seconds",
}

_FEEDBACK_LABELS = {
    "eye_contact_percent": "Camera contact",
    "longest_gaze_break_seconds": "Camera-contact break",
    "face_presence_percent": "Face presence",
    "head_rotation_std_degrees": "Head rotation variation",
    "head_position_std_percent": "Head position variation",
    "expression_variety_index": "Expression movement",
    "overall_words_per_minute": "Speaking pace",
    "window_words_per_minute": "Pace window",
    "strict_filler_count": "Um / uh",
    "strict_filler_rate_per_minute": "Um / uh rate",
    "strict_filler_cluster_count": "Um / uh cluster",
    "pauses_over_2_seconds": "Transcript gaps",
    "pauses_over_3_seconds": "Long transcript gaps",
    "long_pause_rate_per_minute": "Transcript-gap rate",
    "longest_pause_seconds": "Longest transcript gap",
}

_SEEKABLE_FEEDBACK_METRICS = {
    "longest_gaze_break_seconds",
    "window_words_per_minute",
    "strict_filler_cluster_count",
    "longest_pause_seconds",
}

_PROHIBITED_INFERENCE_TERMS = (
    *PROHIBITED_FEEDBACK_TERMS,
    "confidence",
    "confident",
    "reading",
    "reading off",
    "your notes",
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _count(value: object) -> int | None:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _bounded_time(value: object, duration: float) -> float | None:
    number = _number(value)
    if number is None or number < 0 or number > duration + 0.11:
        return None
    return round(min(number, duration), 2)


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _time_text(seconds: float) -> str:
    hundredths = round(max(0.0, seconds) * 100.0)
    minutes, within_minute = divmod(hundredths, 6000)
    whole_seconds, fraction = divmod(within_minute, 100)
    if fraction == 0:
        return f"{minutes:02d}:{whole_seconds:02d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _draft(
    *,
    kind: str,
    role: str,
    start: float,
    end: float,
    title: str,
    text: str,
    metric: str,
    value: float | None,
    unit: str | None,
    source: str,
    evidence: Mapping[str, object],
    seekable: bool = True,
) -> dict[str, object]:
    return {
        "kind": kind,
        "role": role,
        "start_seconds": round(start, 2),
        "end_seconds": round(max(start, end), 2),
        "seek_seconds": round(start, 2),
        "title": title,
        "text": text,
        "metric": metric,
        "value": None if value is None else round(value, 4),
        "unit": unit,
        "source": source,
        "evidence": dict(evidence),
        "seekable": seekable,
    }


def _quality_cues(
    quality: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    descriptions = {
        "face_detected": (
            "Face-tracking quality insufficient",
            "Face tracking did not pass the session quality gate, so face-presence coaching is unavailable.",
        ),
        "eye_contact": (
            "Camera-contact data insufficient",
            "Camera-contact measurements did not pass the eligibility gate, so camera-contact coaching is unavailable.",
        ),
        "head_stability": (
            "Head-movement data insufficient",
            "Head-pose measurements did not pass the coverage gate, so head-movement coaching is unavailable.",
        ),
        "expression_variety": (
            "Expression data insufficient",
            "Mouth and brow measurements did not pass the coverage gate, so expression-movement coaching is unavailable.",
        ),
        "audio_clear": (
            "Audio data insufficient",
            "Audio did not pass the quality gate, so filler, pace, and transcript-gap coaching is unavailable.",
        ),
    }
    cues: list[dict[str, object]] = []
    for metric, (title, text) in descriptions.items():
        if quality.get(metric) == "good":
            continue
        cues.append(_draft(
            kind="quality",
            role="insufficient",
            start=0.0,
            end=duration,
            title=title,
            text=text,
            metric=metric,
            value=None,
            unit=None,
            source="quality_gate",
            evidence={"quality_flag": "bad", "session_duration_seconds": duration},
            seekable=False,
        ))
    return cues


def _strict_filler_cues(
    timeline: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    cues: list[dict[str, object]] = []
    for raw in _sequence(timeline.get("fillers")):
        if not isinstance(raw, Mapping):
            continue
        phrase = str(raw.get("phrase", "")).casefold().strip()
        if phrase not in {"um", "uh"}:
            continue
        start = _bounded_time(raw.get("start_seconds"), duration)
        end = _bounded_time(raw.get("end_seconds"), duration)
        if start is None or end is None or end < start:
            continue
        span = round(end - start, 2)
        cues.append(_draft(
            kind="strict_filler",
            role="review",
            start=start,
            end=end,
            title=f"“{phrase.capitalize()}” detected",
            text=(
                f"Whisper detected “{phrase}” at {_time_text(start)}; "
                f"the timestamped word spans {span:g} seconds."
            ),
            metric="strict_filler_occurrence",
            value=1.0,
            unit="occurrence",
            source="whisper_word_timestamp",
            evidence={"phrase": phrase, "duration_seconds": span},
        ))
    return cues


def _contiguous_runs(
    rows: Sequence[tuple[float, Mapping[str, object]]],
) -> list[list[tuple[float, Mapping[str, object]]]]:
    runs: list[list[tuple[float, Mapping[str, object]]]] = []
    for timestamp, row in rows:
        if not runs or timestamp - runs[-1][-1][0] > 1.5:
            runs.append([(timestamp, row)])
        else:
            runs[-1].append((timestamp, row))
    return runs


def _vision_rows(
    timeline: Mapping[str, object], duration: float
) -> list[tuple[float, Mapping[str, object]]]:
    rows: list[tuple[float, Mapping[str, object]]] = []
    for raw in _sequence(timeline.get("vision")):
        if not isinstance(raw, Mapping):
            continue
        timestamp = _bounded_time(raw.get("timestamp_seconds"), duration)
        if timestamp is not None:
            rows.append((timestamp, raw))
    return sorted(rows, key=lambda item: item[0])


def _camera_contact_cues(
    timeline: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    eligible: list[tuple[float, Mapping[str, object]]] = []
    for timestamp, row in _vision_rows(timeline, duration):
        eligible_frames = _count(row.get("contact_eligible_frame_count"))
        contact_frames = _count(row.get("contact_frame_count"))
        if (
            row.get("face_detected") is True
            and row.get("eye_contact") is False
            and eligible_frames is not None
            and contact_frames is not None
            and 0 <= contact_frames < eligible_frames / 2
        ):
            eligible.append((timestamp, row))
    cues: list[dict[str, object]] = []
    for run in _contiguous_runs(eligible):
        start = run[0][0]
        end = min(duration, run[-1][0] + 1.0)
        eligible_frames = sum(
            _count(row.get("contact_eligible_frame_count")) or 0
            for _, row in run
        )
        contact_frames = sum(
            _count(row.get("contact_frame_count")) or 0
            for _, row in run
        )
        if eligible_frames <= 0:
            continue
        percent = round(100.0 * contact_frames / eligible_frames, 1)
        span = round(end - start, 2)
        cues.append(_draft(
            kind="camera_contact_break",
            role="review",
            start=start,
            end=end,
            title="Camera-contact break",
            text=(
                f"From {_time_text(start)} to {_time_text(end)}, camera contact measured "
                f"{contact_frames} of {eligible_frames} eligible frames ({percent:g}%) "
                f"across {span:g} seconds."
            ),
            metric="camera_contact_break_seconds",
            value=span,
            unit="seconds",
            source="mediapipe_contact_buckets",
            evidence={
                "contact_frames": contact_frames,
                "eligible_frames": eligible_frames,
                "contact_percent": percent,
                "bucket_count": len(run),
            },
        ))
    return cues


def _face_tracking_cues(
    timeline: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    rows = _vision_rows(timeline, duration)
    failed: list[tuple[float, Mapping[str, object]]] = []
    empty: list[tuple[float, Mapping[str, object]]] = []
    for timestamp, row in rows:
        total = _count(row.get("frame_count"))
        detected = _count(row.get("detected_frame_count"))
        if total is None:
            continue
        if total == 0:
            empty.append((timestamp, row))
        elif (
            row.get("face_detected") is False
            and detected is not None
            and detected <= total
        ):
            failed.append((timestamp, row))

    cues: list[dict[str, object]] = []
    for run in _contiguous_runs(failed):
        start = run[0][0]
        end = min(duration, run[-1][0] + 1.0)
        total_frames = sum(_count(row.get("frame_count")) or 0 for _, row in run)
        detected_frames = sum(
            _count(row.get("detected_frame_count")) or 0 for _, row in run
        )
        usable_percent = round(100.0 * detected_frames / total_frames, 1) if total_frames else 0.0
        cues.append(_draft(
            kind="face_tracking_gap",
            role="quality",
            start=start,
            end=end,
            title="Face tracking unavailable",
            text=(
                f"Across {len(run)} one-second bucket{'s' if len(run) != 1 else ''} from "
                f"{_time_text(start)} to {_time_text(end)}, a usable single face was found in "
                f"{detected_frames} of {total_frames} analyzed frames ({usable_percent:g}%); "
                "the per-bucket face gate did not pass."
            ),
            metric="usable_single_face_percent",
            value=usable_percent,
            unit="% analyzed frames",
            source="mediapipe_face_buckets",
            evidence={
                "detected_frames": detected_frames,
                "analyzed_frames": total_frames,
                "bucket_count": len(run),
            },
        ))
    for run in _contiguous_runs(empty):
        start = run[0][0]
        end = min(duration, run[-1][0] + 1.0)
        cues.append(_draft(
            kind="face_tracking_gap",
            role="quality",
            start=start,
            end=end,
            title="Vision frames unavailable",
            text=(
                f"No video frames were analyzed in {len(run)} one-second bucket"
                f"{'s' if len(run) != 1 else ''} from {_time_text(start)} to {_time_text(end)}."
            ),
            metric="analyzed_frame_count",
            value=0.0,
            unit="frames",
            source="vision_sampling",
            evidence={"bucket_count": len(run)},
        ))
    return cues


def _expression_movement_cue(
    timeline: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    """Return the earliest bucket with the largest finite expression change."""

    candidates: list[tuple[float, float, float, float]] = []
    for timestamp, row in _vision_rows(timeline, duration):
        mouth = _number(row.get("mouth_activity"))
        brow = _number(row.get("brow_activity"))
        change = _number(row.get("expression_change"))
        if (
            row.get("face_detected") is not True
            or mouth is None or brow is None or change is None
            or not 0.0 <= mouth <= 1.0
            or not 0.0 <= brow <= 1.0
            or not MIN_REVIEW_EXPRESSION_CHANGE <= change <= math.sqrt(2.0)
        ):
            continue
        candidates.append((change, timestamp, mouth, brow))
    if not candidates:
        return []

    # A tied maximum resolves to the earliest timestamp, making replay stable.
    change, start, mouth, brow = min(
        candidates, key=lambda item: (-item[0], item[1])
    )
    end = min(duration, start + 1.0)
    mouth_display = round(mouth, 4)
    brow_display = round(brow, 4)
    change_display = round(change, 4)
    return [_draft(
        kind="expression_movement",
        role="observation",
        start=start,
        end=end,
        title="Largest measured expression change",
        text=(
            f"In the analyzed bucket from {_time_text(start)} to {_time_text(end)}, "
            f"measured mouth activity was {mouth_display:.4f}, brow activity was "
            f"{brow_display:.4f}, and normalized expression change was "
            f"{change_display:.4f}."
        ),
        metric="expression_change",
        value=change_display,
        unit="normalized change",
        source="mediapipe_expression_bucket",
        evidence={
            "bucket_start_seconds": start,
            "bucket_end_seconds": end,
            "mouth_activity": mouth_display,
            "brow_activity": brow_display,
            "expression_change": change_display,
            "display_floor": MIN_REVIEW_EXPRESSION_CHANGE,
        },
    )]


def _pace_spike_cues(
    timeline: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    pace_values = [
        value
        for raw in _sequence(timeline.get("pace_windows"))
        if isinstance(raw, Mapping)
        if (value := _number(raw.get("words_per_minute"))) is not None
    ]
    pace_median = median(pace_values) if len(pace_values) >= 2 else None
    cues: list[dict[str, object]] = []
    for raw in _sequence(timeline.get("pace_spikes")):
        if not isinstance(raw, Mapping):
            continue
        start = _bounded_time(raw.get("start_seconds"), duration)
        end = _bounded_time(raw.get("end_seconds"), duration)
        wpm = _number(raw.get("words_per_minute"))
        words = _number(raw.get("words"))
        if (
            start is None or end is None or end < start or wpm is None
            or pace_median is None or wpm < pace_median + 25.0
        ):
            continue
        comparison = (
            f", {wpm - pace_median:g} WPM above the {pace_median:g} WPM session-window median"
        )
        cues.append(_draft(
            kind="pace_spike",
            role="review",
            start=start,
            end=end,
            title="Pace spike",
            text=(
                f"From {_time_text(start)} to {_time_text(end)}, transcript timing measured "
                f"{wpm:g} WPM{comparison}."
            ),
            metric="window_words_per_minute",
            value=wpm,
            unit="WPM",
            source="whisper_pace_window",
            evidence={
                "words": None if words is None else int(words),
                "window_median_wpm": round(pace_median, 1),
            },
        ))
    return cues


def _transcript_gap_cues(
    timeline: Mapping[str, object], duration: float
) -> list[dict[str, object]]:
    cues: list[dict[str, object]] = []
    for raw in _sequence(timeline.get("pause_events")):
        if not isinstance(raw, Mapping):
            continue
        start = _bounded_time(raw.get("start_seconds"), duration)
        end = _bounded_time(raw.get("end_seconds"), duration)
        gap = _number(raw.get("duration_seconds"))
        if (
            start is None or end is None or end < start or gap is None
            or gap <= LONG_TRANSCRIPT_GAP_SECONDS
            or abs((end - start) - gap) > 0.11
        ):
            continue
        cues.append(_draft(
            kind="transcript_gap",
            role="review",
            start=start,
            end=end,
            title="Long transcript gap",
            text=(
                f"Whisper timestamps contain a {gap:g}-second gap between words from "
                f"{_time_text(start)} to {_time_text(end)}; review whether it was intentional."
            ),
            metric="transcript_gap_seconds",
            value=gap,
            unit="seconds",
            source="whisper_word_timestamps",
            evidence={"threshold_seconds": LONG_TRANSCRIPT_GAP_SECONDS},
        ))
    return cues


def _feedback_document(
    feedback: StructuredFeedback | Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if isinstance(feedback, StructuredFeedback):
        return feedback.to_dict()
    return feedback if isinstance(feedback, Mapping) else None


def _feedback_facts(
    metrics: Mapping[str, object], duration: float
) -> list[tuple[str, float, str, float]]:
    aggregate = metrics.get("aggregate")
    timeline = metrics.get("timeline")
    facts: list[tuple[str, float, str, float]] = []
    if isinstance(aggregate, Mapping):
        for metric, unit in _AGGREGATE_FEEDBACK_UNITS.items():
            value = _number(aggregate.get(metric))
            if value is not None:
                facts.append((metric, value, unit, duration))
    if not isinstance(timeline, Mapping):
        return facts
    gaze = timeline.get("longest_gaze_break")
    if isinstance(gaze, Mapping):
        value = _number(gaze.get("duration_seconds"))
        timestamp = _bounded_time(gaze.get("start_seconds"), duration)
        if value is not None and timestamp is not None:
            facts.append(("longest_gaze_break_seconds", value, "seconds", timestamp))
    for raw in _sequence(timeline.get("pace_windows")):
        if isinstance(raw, Mapping):
            value = _number(raw.get("words_per_minute"))
            timestamp = _bounded_time(raw.get("start_seconds"), duration)
            if value is not None and timestamp is not None:
                facts.append(("window_words_per_minute", value, "WPM", timestamp))
    for raw in _sequence(timeline.get("filler_clusters")):
        if isinstance(raw, Mapping):
            value = _number(raw.get("count"))
            timestamp = _bounded_time(raw.get("start_seconds"), duration)
            if value is not None and timestamp is not None:
                facts.append(("strict_filler_cluster_count", value, "um/uh fillers", timestamp))
    longest_pause = timeline.get("longest_pause")
    if isinstance(longest_pause, Mapping):
        value = _number(longest_pause.get("duration_seconds"))
        timestamp = _bounded_time(longest_pause.get("start_seconds"), duration)
        if value is not None and timestamp is not None:
            facts.append(("longest_pause_seconds", value, "seconds", timestamp))
    return facts


def _verified_feedback_cues(
    metrics: Mapping[str, object],
    feedback: StructuredFeedback | Mapping[str, object] | None,
    quality: Mapping[str, object],
    duration: float,
) -> list[dict[str, object]]:
    document = _feedback_document(feedback)
    if not document or document.get("status") != "ready":
        return []
    source = str(document.get("source", ""))
    if source != "local_llm_verified" and not source.startswith("local_llm_verified_"):
        return []
    facts = _feedback_facts(metrics, duration)
    cues: list[dict[str, object]] = []
    for role, limit in (("strength", 2), ("improvement", 3)):
        group = document.get(f"{role}s")
        for raw in _sequence(group)[:limit]:
            if not isinstance(raw, Mapping):
                continue
            text = str(raw.get("text", "")).strip()[:2000]
            metric = str(raw.get("metric", "")).strip()
            value = _number(raw.get("value"))
            unit = str(raw.get("unit", "")).strip()
            timestamp = _bounded_time(raw.get("timestamp_seconds"), duration)
            lowered = text.casefold()
            required_quality = _QUALITY_BY_FEEDBACK_METRIC.get(metric)
            if (
                not text or value is None or timestamp is None
                or required_quality is None or quality.get(required_quality) != "good"
                or any(term in lowered for term in _PROHIBITED_INFERENCE_TERMS)
            ):
                continue
            if not any(
                metric == fact_metric
                and unit == fact_unit
                and abs(value - fact_value) <= 0.11
                and abs(timestamp - fact_timestamp) <= 0.11
                for fact_metric, fact_value, fact_unit, fact_timestamp in facts
            ):
                continue
            cues.append(_draft(
                kind="verified_coaching",
                role=role,
                start=timestamp,
                end=timestamp,
                title=_FEEDBACK_LABELS.get(metric, metric.replace("_", " ").title()),
                text=text,
                metric=metric,
                value=value,
                unit=unit,
                source=source,
                evidence={"feedback_status": "ready", "quality_flag": required_quality},
                seekable=metric in _SEEKABLE_FEEDBACK_METRICS,
            ))
    return cues


def build_review_cues(
    metrics: Mapping[str, object],
    feedback: StructuredFeedback | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a fail-closed payload for the post-session review UI.

    Event cues are seekable; aggregate and quality notes are session-wide. Raw
    cues are deterministic. Existing AI claims are copied only when they
    retain a verified source marker, match current metric evidence exactly, and
    pass the current per-metric quality gate.
    """
    duration = _number(metrics.get("duration_seconds"))
    duration = round(max(0.0, duration or 0.0), 2)
    quality_raw = metrics.get("quality_flags")
    quality = quality_raw if isinstance(quality_raw, Mapping) else {}
    timeline_raw = metrics.get("timeline")
    timeline = timeline_raw if isinstance(timeline_raw, Mapping) else {}

    drafts = _quality_cues(quality, duration)
    drafts.extend(_face_tracking_cues(timeline, duration))
    if quality.get("audio_clear") == "good":
        drafts.extend(_strict_filler_cues(timeline, duration))
        drafts.extend(_pace_spike_cues(timeline, duration))
        drafts.extend(_transcript_gap_cues(timeline, duration))
    if quality.get("face_detected") == "good" and quality.get("eye_contact") == "good":
        drafts.extend(_camera_contact_cues(timeline, duration))
    if (
        quality.get("face_detected") == "good"
        and quality.get("expression_variety") == "good"
    ):
        drafts.extend(_expression_movement_cue(timeline, duration))
    drafts.extend(_verified_feedback_cues(metrics, feedback, quality, duration))

    drafts.sort(key=lambda item: (
        float(item["start_seconds"]),
        _KIND_ORDER.get(str(item["kind"]), 99),
        float(item["end_seconds"]),
        str(item["metric"]),
        str(item["text"]),
    ))
    cues = [
        ReviewCue(cue_id=f"cue-{index:04d}", **draft).to_dict()
        for index, draft in enumerate(drafts, start=1)
    ]
    counts = Counter(str(cue["kind"]) for cue in cues)
    return {
        "schema_version": REVIEW_CUE_SCHEMA,
        "session_id": str(metrics.get("session_id", "")),
        "duration_seconds": duration,
        "status": "ready" if cues else "no_cues",
        "cues": cues,
        "counts": {kind: counts.get(kind, 0) for kind in _KIND_ORDER},
        "limitations": {
            "descriptive_measured_behavior_only": True,
            "infers_confidence": False,
            "infers_reading": False,
        },
    }
