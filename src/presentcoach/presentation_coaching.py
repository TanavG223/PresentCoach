"""Deterministic, transparent presentation-coaching bands and wording.

The local language model may select from these hints, but it never decides
whether a measurement is a strength or an improvement.  That comparison is
owned by Python so the same numbers always produce the same assessment.
"""

from __future__ import annotations

from typing import Mapping, Sequence


# Pace and the low-filler target come from published presentation-coaching
# guidance.  Camera orientation and event-duration cutoffs are deliberately
# named product heuristics; they are not universal measures of good speaking.
PACE_MIN_WPM = 100.0
PACE_MAX_WPM = 165.0
FILLER_TARGET_PER_MINUTE = 2.0
FILLER_FREQUENT_PER_MINUTE = 4.0
CAMERA_MOSTLY_ORIENTED_PERCENT = 70.0
CAMERA_MIXED_PERCENT = 50.0
FACE_USABLE_PERCENT = 80.0
EXTENDED_GAZE_BREAK_SECONDS = 5.0
LONG_PAUSE_SECONDS = 3.0
FREQUENT_LONG_PAUSES_PER_MINUTE = 2.0


def _number(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _hint(
    metric: str,
    role: str,
    *,
    value: float,
    unit: str,
    timestamp_seconds: float,
    assessment: str,
    practice_band: str,
    recommendation: str,
    priority: int,
) -> dict[str, object]:
    return {
        "metric": metric,
        "role": role,
        "value": round(value, 2),
        "unit": unit,
        "timestamp_seconds": round(timestamp_seconds, 2),
        "assessment": assessment,
        "comparison_mode": "general_practice_band",
        "practice_band": practice_band,
        "recommendation": recommendation,
        "priority": priority,
    }


def general_coaching_hints(metrics: Mapping[str, object]) -> list[dict[str, object]]:
    """Classify only quality-approved measurements into transparent bands."""
    aggregate = metrics.get("aggregate", {})
    quality = metrics.get("quality_flags", {})
    timeline = metrics.get("timeline", {})
    if not isinstance(aggregate, Mapping) or not isinstance(quality, Mapping):
        return []
    if not isinstance(timeline, Mapping):
        timeline = {}
    duration = float(metrics.get("duration_seconds", 0) or 0)
    strengths: list[dict[str, object]] = []
    improvements: list[dict[str, object]] = []

    if quality.get("eye_contact") == "good" and quality.get("face_detected") == "good":
        contact = _number(aggregate, "eye_contact_percent")
        if contact is not None:
            if contact >= CAMERA_MOSTLY_ORIENTED_PERCENT:
                strengths.append(_hint(
                    "eye_contact_percent", "strength", value=contact, unit="%",
                    timestamp_seconds=duration, assessment="mostly camera-oriented",
                    practice_band="PresentCoach heuristic: 70% or more",
                    recommendation="Keep finishing complete thoughts toward the camera.",
                    priority=10,
                ))
            else:
                assessment = "mixed camera orientation" if contact >= CAMERA_MIXED_PERCENT else "frequently away from camera"
                improvements.append(_hint(
                    "eye_contact_percent", "improvement", value=contact, unit="%",
                    timestamp_seconds=duration, assessment=assessment,
                    practice_band="PresentCoach heuristic: 70% or more",
                    recommendation="Deliver one complete sentence toward the camera before checking notes.",
                    priority=10,
                ))
        gaze = timeline.get("longest_gaze_break")
        if isinstance(gaze, Mapping):
            gaze_duration = _number(gaze, "duration_seconds")
            gaze_start = _number(gaze, "start_seconds")
            if gaze_duration is not None and gaze_start is not None and gaze_duration >= EXTENDED_GAZE_BREAK_SECONDS:
                improvements.append(_hint(
                    "longest_gaze_break_seconds", "improvement",
                    value=gaze_duration, unit="seconds", timestamp_seconds=gaze_start,
                    assessment="extended camera-contact break",
                    practice_band="PresentCoach review marker: 5 seconds or longer",
                    recommendation="Rehearse that exact section and shorten the next look away.",
                    priority=30,
                ))

    if quality.get("face_detected") == "good":
        presence = _number(aggregate, "face_presence_percent")
        if presence is not None:
            if presence >= FACE_USABLE_PERCENT:
                strengths.append(_hint(
                    "face_presence_percent", "strength", value=presence, unit="%",
                    timestamp_seconds=duration, assessment="consistently in frame",
                    practice_band="PresentCoach quality target: 80% or more",
                    recommendation="Keep the same camera framing and lighting.",
                    priority=40,
                ))
            else:
                improvements.append(_hint(
                    "face_presence_percent", "improvement", value=presence, unit="%",
                    timestamp_seconds=duration, assessment="face frequently out of frame",
                    practice_band="PresentCoach quality target: 80% or more",
                    recommendation="Recenter the camera and keep your face visible before the next run.",
                    priority=20,
                ))

    if quality.get("audio_clear") == "good":
        pace = _number(aggregate, "overall_words_per_minute")
        if pace is not None:
            if PACE_MIN_WPM <= pace <= PACE_MAX_WPM:
                strengths.append(_hint(
                    "overall_words_per_minute", "strength", value=pace, unit="WPM",
                    timestamp_seconds=duration, assessment="within the broad pace range",
                    practice_band="100 to 165 WPM",
                    recommendation="Keep this overall pace while varying emphasis naturally.",
                    priority=20,
                ))
            else:
                assessment = "slower than the broad pace range" if pace < PACE_MIN_WPM else "faster than the broad pace range"
                recommendation = (
                    "Tighten one transition and rehearse it without restarting."
                    if pace < PACE_MIN_WPM
                    else "Mark a short pause after each key point and rehearse the fastest section."
                )
                improvements.append(_hint(
                    "overall_words_per_minute", "improvement", value=pace, unit="WPM",
                    timestamp_seconds=duration, assessment=assessment,
                    practice_band="100 to 165 WPM", recommendation=recommendation,
                    priority=20,
                ))

        filler_rate = _number(aggregate, "strict_filler_rate_per_minute")
        if filler_rate is not None:
            if filler_rate <= FILLER_TARGET_PER_MINUTE:
                strengths.append(_hint(
                    "strict_filler_rate_per_minute", "strength", value=filler_rate,
                    unit="um/uh per min", timestamp_seconds=duration,
                    assessment="within the low-filler practice target",
                    practice_band="0 to 2 fillers per minute",
                    recommendation="Keep replacing filler sounds with a silent breath.",
                    priority=30,
                ))
            elif filler_rate > FILLER_FREQUENT_PER_MINUTE:
                improvements.append(_hint(
                    "strict_filler_rate_per_minute", "improvement", value=filler_rate,
                    unit="um/uh per min", timestamp_seconds=duration,
                    assessment="frequent filler use",
                    practice_band="Practice target: 0 to 2 fillers per minute; over 4 is an app review marker",
                    recommendation="Replace the next filler with a silent breath, especially at transitions.",
                    priority=10,
                ))

        clusters = timeline.get("filler_clusters", ())
        if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)) and clusters:
            first = clusters[0]
            if isinstance(first, Mapping):
                count = _number(first, "count")
                start = _number(first, "start_seconds")
                if count is not None and start is not None:
                    improvements.append(_hint(
                        "strict_filler_cluster_count", "improvement", value=count,
                        unit="um/uh fillers", timestamp_seconds=start,
                        assessment="clustered filler use",
                        practice_band="PresentCoach review marker: 3 or more close together",
                        recommendation="Rehearse this transition once using a silent pause instead.",
                        priority=40,
                    ))

        pause_rate = _number(aggregate, "long_pause_rate_per_minute")
        if pause_rate is not None and pause_rate > FREQUENT_LONG_PAUSES_PER_MINUTE:
            improvements.append(_hint(
                "long_pause_rate_per_minute", "improvement", value=pause_rate,
                unit="pauses/min", timestamp_seconds=duration,
                assessment="frequent transcript gaps longer than 3 seconds",
                practice_band="PresentCoach heuristic: no more than 2 long transcript gaps per minute",
                recommendation="Check whether each gap was intentional; rehearse the choppy transition.",
                priority=30,
            ))
        longest_pause = timeline.get("longest_pause")
        if isinstance(longest_pause, Mapping):
            pause_duration = _number(longest_pause, "duration_seconds")
            pause_start = _number(longest_pause, "start_seconds")
            if pause_duration is not None and pause_start is not None and pause_duration > LONG_PAUSE_SECONDS:
                improvements.append(_hint(
                    "longest_pause_seconds", "improvement", value=pause_duration,
                    unit="seconds", timestamp_seconds=pause_start,
                    assessment="long gap between transcript words",
                    practice_band="Review marker: longer than 3 seconds",
                    recommendation="Check whether this gap was deliberate; if not, rehearse the transition into the next sentence.",
                    priority=25,
                ))

    strengths.sort(key=lambda item: int(item["priority"]))
    improvements.sort(key=lambda item: int(item["priority"]))
    return strengths[:2] + improvements[:3]


def render_coaching_text(
    *,
    metric: str,
    role: str,
    value: float,
    unit: str,
    timestamp_seconds: float,
    hint: Mapping[str, object] | None,
) -> str:
    """Render actionable text after the LLM output passes evidence checks."""
    assessment = str((hint or {}).get("assessment", "measured behavior"))
    band = str((hint or {}).get("practice_band", "the selected practice reference"))
    recommendation = str((hint or {}).get("recommendation", "Review this moment in the recording."))
    value_text = f"{value:g} {unit}"
    time_text = f"{timestamp_seconds:g} seconds"
    if metric == "eye_contact_percent":
        lead = f"Across the session ending at {time_text}, camera contact was {value_text}"
    elif metric == "face_presence_percent":
        lead = f"Across the session ending at {time_text}, your face was visible for {value_text}"
    elif metric == "overall_words_per_minute":
        lead = f"Across the session ending at {time_text}, speaking pace was {value_text}"
    elif metric == "strict_filler_rate_per_minute":
        lead = f"Across the session ending at {time_text}, measured um/uh use was {value_text}"
    elif metric == "long_pause_rate_per_minute":
        lead = f"Across the session ending at {time_text}, long transcript-gap frequency was {value_text}"
    elif metric == "longest_gaze_break_seconds":
        lead = f"At {time_text}, the longest camera-contact break lasted {value_text}"
    elif metric == "longest_pause_seconds":
        lead = f"At {time_text}, the longest gap between transcript words lasted {value_text}"
    elif metric == "strict_filler_cluster_count":
        lead = f"At {time_text}, an um/uh cluster contained {value_text}"
    else:
        lead = f"At {time_text}, {metric.replace('_', ' ')} measured {value_text}"
    relation = "meeting" if role == "strength" else "compared with"
    return f"{lead}—{assessment}, {relation} {band}. {recommendation}"
