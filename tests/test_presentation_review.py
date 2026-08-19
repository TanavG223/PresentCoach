from copy import deepcopy

from stroke_screening.presentation_core import FeedbackClaim, StructuredFeedback
from stroke_screening.presentation_review import build_review_cues


def measured_metrics():
    return {
        "session_id": "session-1",
        "duration_seconds": 60.0,
        "aggregate": {
            "eye_contact_percent": 75.0,
            "face_presence_percent": 95.0,
            "overall_words_per_minute": 130.0,
            "strict_filler_count": 2,
            "strict_filler_rate_per_minute": 2.0,
            "longest_pause_seconds": 4.0,
        },
        "quality_flags": {
            "face_detected": "good",
            "eye_contact": "good",
            "head_stability": "good",
            "expression_variety": "good",
            "audio_clear": "good",
        },
        "timeline": {
            "fillers": [
                {"phrase": "like", "start_seconds": 3.0, "end_seconds": 3.2},
                {"phrase": "um", "start_seconds": 5.0, "end_seconds": 5.25},
                {"phrase": "uh", "start_seconds": 22.0, "end_seconds": 22.4},
            ],
            "vision": [
                {
                    "timestamp_seconds": 0.0, "frame_count": 15,
                    "face_detected": True, "eye_contact": True,
                    "detected_frame_count": 15, "contact_frame_count": 15,
                    "contact_eligible_frame_count": 15,
                    "mouth_activity": 0.1, "brow_activity": 0.05,
                    "expression_change": 0.0,
                },
                {
                    "timestamp_seconds": 1.0, "frame_count": 15,
                    "face_detected": True, "eye_contact": False,
                    "detected_frame_count": 15, "contact_frame_count": 3,
                    "contact_eligible_frame_count": 15,
                    "mouth_activity": 0.12, "brow_activity": 0.06,
                    "expression_change": 0.01,
                },
                {
                    "timestamp_seconds": 2.0, "frame_count": 15,
                    "face_detected": True, "eye_contact": False,
                    "detected_frame_count": 15, "contact_frame_count": 6,
                    "contact_eligible_frame_count": 15,
                    "mouth_activity": 0.18, "brow_activity": 0.08,
                    "expression_change": 0.07,
                },
                {
                    "timestamp_seconds": 3.0, "frame_count": 15,
                    "face_detected": True, "eye_contact": True,
                    "detected_frame_count": 15, "contact_frame_count": 15,
                    "contact_eligible_frame_count": 15,
                    "mouth_activity": 0.11, "brow_activity": 0.06,
                    "expression_change": 0.01,
                },
                {
                    "timestamp_seconds": 8.0, "frame_count": 15,
                    "face_detected": False, "eye_contact": None,
                    "detected_frame_count": 0, "contact_frame_count": 0,
                    "contact_eligible_frame_count": 0,
                    "mouth_activity": 0.9, "brow_activity": 0.9,
                    "expression_change": 0.9,
                },
                {
                    "timestamp_seconds": 9.0, "frame_count": 15,
                    "face_detected": False, "eye_contact": None,
                    "detected_frame_count": 4, "contact_frame_count": 0,
                    "contact_eligible_frame_count": 0,
                    "mouth_activity": 0.8, "brow_activity": 0.8,
                    "expression_change": 0.8,
                },
            ],
            "pace_windows": [
                {"start_seconds": 0.0, "end_seconds": 15.0, "words": 25, "words_per_minute": 100.0},
                {"start_seconds": 15.0, "end_seconds": 30.0, "words": 25, "words_per_minute": 100.0},
                {"start_seconds": 30.0, "end_seconds": 45.0, "words": 40, "words_per_minute": 160.0},
                {"start_seconds": 45.0, "end_seconds": 60.0, "words": 25, "words_per_minute": 100.0},
            ],
            "pace_spikes": [
                {"start_seconds": 30.0, "end_seconds": 45.0, "words": 40, "words_per_minute": 160.0},
            ],
            "pause_events": [
                {"start_seconds": 10.0, "end_seconds": 12.0, "duration_seconds": 2.0},
                {"start_seconds": 18.0, "end_seconds": 22.0, "duration_seconds": 4.0},
            ],
            "longest_pause": {
                "start_seconds": 18.0, "end_seconds": 22.0,
                "duration_seconds": 4.0,
            },
            "longest_gaze_break": {
                "start_seconds": 1.0, "end_seconds": 3.0,
                "duration_seconds": 2.0,
            },
            "filler_clusters": [],
        },
    }


def verified_feedback():
    return {
        "status": "ready",
        "source": "local_llm_verified_general_practice",
        "strengths": [{
            "text": "Across the session ending at 60 seconds, camera contact was 75 %.",
            "metric": "eye_contact_percent",
            "value": 75.0,
            "unit": "%",
            "timestamp_seconds": 60.0,
        }],
        "improvements": [{
            "text": "At 18 seconds, the longest gap between transcript words lasted 4 seconds.",
            "metric": "longest_pause_seconds",
            "value": 4.0,
            "unit": "seconds",
            "timestamp_seconds": 18.0,
        }],
        "insufficient_data": [],
    }


def test_build_review_cues_covers_measured_events_and_verified_claims():
    payload = build_review_cues(measured_metrics(), verified_feedback())

    assert payload["schema_version"] == "presentcoach.review-cues.v1"
    assert payload["session_id"] == "session-1"
    assert payload["status"] == "ready"
    assert payload["counts"] == {
        "quality": 0,
        "face_tracking_gap": 1,
        "strict_filler": 2,
        "camera_contact_break": 1,
        "expression_movement": 1,
        "pace_spike": 1,
        "transcript_gap": 1,
        "verified_coaching": 2,
    }
    cues = payload["cues"]
    assert [cue["cue_id"] for cue in cues] == [
        f"cue-{index:04d}" for index in range(1, len(cues) + 1)
    ]
    assert [cue["start_seconds"] for cue in cues] == sorted(
        cue["start_seconds"] for cue in cues
    )

    fillers = [cue for cue in cues if cue["kind"] == "strict_filler"]
    assert [cue["evidence"]["phrase"] for cue in fillers] == ["um", "uh"]
    assert "like" not in " ".join(cue["text"].casefold() for cue in fillers)

    contact = next(cue for cue in cues if cue["kind"] == "camera_contact_break")
    assert contact["start_seconds"] == 1.0
    assert contact["end_seconds"] == 3.0
    assert contact["value"] == 2.0
    assert contact["evidence"]["contact_frames"] == 9
    assert contact["evidence"]["eligible_frames"] == 30

    expression = next(cue for cue in cues if cue["kind"] == "expression_movement")
    assert expression["role"] == "observation"
    assert expression["start_seconds"] == 2.0
    assert expression["end_seconds"] == 3.0
    assert expression["metric"] == "expression_change"
    assert expression["value"] == 0.07
    assert expression["evidence"] == {
        "bucket_start_seconds": 2.0,
        "bucket_end_seconds": 3.0,
        "mouth_activity": 0.18,
        "brow_activity": 0.08,
        "expression_change": 0.07,
        "display_floor": 0.001,
    }
    assert "mouth activity was 0.1800" in expression["text"]
    assert "brow activity was 0.0800" in expression["text"]
    assert "normalized expression change was 0.0700" in expression["text"]
    assert not any(
        term in expression["text"].casefold()
        for term in ("emotion", "flat affect", "confidence", "intent")
    )

    face = next(cue for cue in cues if cue["kind"] == "face_tracking_gap")
    assert face["start_seconds"] == 8.0
    assert face["end_seconds"] == 10.0
    assert face["evidence"] == {
        "detected_frames": 4,
        "analyzed_frames": 30,
        "bucket_count": 2,
    }
    assert face["value"] == 13.3

    gap = next(cue for cue in cues if cue["kind"] == "transcript_gap")
    assert gap["value"] == 4.0
    assert gap["start_seconds"] == 18.0

    spike = next(cue for cue in cues if cue["kind"] == "pace_spike")
    assert spike["value"] == 160.0
    assert spike["evidence"]["window_median_wpm"] == 100.0

    coaching = [cue for cue in cues if cue["kind"] == "verified_coaching"]
    assert [cue["role"] for cue in coaching] == ["improvement", "strength"]
    assert next(
        cue for cue in coaching if cue["metric"] == "longest_pause_seconds"
    )["seekable"] is True
    assert next(
        cue for cue in coaching if cue["metric"] == "eye_contact_percent"
    )["seekable"] is False
    assert payload["limitations"] == {
        "descriptive_measured_behavior_only": True,
        "infers_confidence": False,
        "infers_reading": False,
    }


def test_bad_audio_suppresses_audio_coaching_and_emits_quality_cue():
    metrics = measured_metrics()
    metrics["quality_flags"]["audio_clear"] = "bad"

    payload = build_review_cues(metrics, verified_feedback())
    kinds = [cue["kind"] for cue in payload["cues"]]

    assert "strict_filler" not in kinds
    assert "pace_spike" not in kinds
    assert "transcript_gap" not in kinds
    assert not any(
        cue["kind"] == "verified_coaching"
        and cue["metric"] == "longest_pause_seconds"
        for cue in payload["cues"]
    )
    quality = [
        cue for cue in payload["cues"]
        if cue["kind"] == "quality" and cue["metric"] == "audio_clear"
    ]
    assert len(quality) == 1
    assert quality[0]["role"] == "insufficient"
    assert quality[0]["seekable"] is False
    assert "unavailable" in quality[0]["text"]


def test_bad_contact_quality_suppresses_contact_break_and_contact_claim():
    metrics = measured_metrics()
    metrics["quality_flags"]["eye_contact"] = "bad"

    payload = build_review_cues(metrics, verified_feedback())

    assert not any(cue["kind"] == "camera_contact_break" for cue in payload["cues"])
    assert not any(
        cue["kind"] == "verified_coaching"
        and cue["metric"] == "eye_contact_percent"
        for cue in payload["cues"]
    )
    assert any(
        cue["kind"] == "quality" and cue["metric"] == "eye_contact"
        for cue in payload["cues"]
    )


def test_bad_expression_quality_suppresses_expression_movement_cue():
    metrics = measured_metrics()
    metrics["quality_flags"]["expression_variety"] = "bad"

    payload = build_review_cues(metrics, verified_feedback())

    assert payload["counts"]["expression_movement"] == 0
    assert not any(
        cue["kind"] == "expression_movement" for cue in payload["cues"]
    )
    assert any(
        cue["kind"] == "quality" and cue["metric"] == "expression_variety"
        for cue in payload["cues"]
    )


def test_expression_cue_requires_finite_nontrivial_complete_bucket_values():
    for invalid_change in (0.0, 0.0009, float("nan"), float("inf")):
        metrics = measured_metrics()
        for bucket in metrics["timeline"]["vision"]:
            bucket["expression_change"] = invalid_change
        assert build_review_cues(metrics)["counts"]["expression_movement"] == 0

    incomplete = measured_metrics()
    for bucket in incomplete["timeline"]["vision"]:
        bucket["mouth_activity"] = None
    assert build_review_cues(incomplete)["counts"]["expression_movement"] == 0


def test_unverified_or_inference_based_feedback_is_not_copied():
    metrics = measured_metrics()
    unverified = verified_feedback()
    unverified["source"] = "unverified_remote_model"
    assert build_review_cues(metrics, unverified)["counts"]["verified_coaching"] == 0

    prohibited = verified_feedback()
    prohibited["strengths"][0]["text"] = (
        "At 60 seconds, 75% camera contact proves you were confident."
    )
    prohibited["improvements"][0]["text"] = (
        "At 18 seconds, a 4-second gap proves you were reading from notes."
    )
    payload = build_review_cues(metrics, prohibited)
    assert payload["counts"]["verified_coaching"] == 0
    all_text = " ".join(cue["text"].casefold() for cue in payload["cues"])
    assert "confident" not in all_text
    assert "reading from" not in all_text


def test_feedback_must_match_current_number_unit_and_timestamp():
    metrics = measured_metrics()
    feedback = verified_feedback()
    feedback["strengths"][0]["value"] = 99.0
    feedback["improvements"][0]["timestamp_seconds"] = 19.0

    payload = build_review_cues(metrics, feedback)

    assert payload["counts"]["verified_coaching"] == 0


def test_structured_feedback_object_uses_the_same_verified_contract():
    feedback = StructuredFeedback(
        status="ready",
        strengths=(FeedbackClaim(
            text="Across the session ending at 60 seconds, camera contact was 75 %.",
            metric="eye_contact_percent",
            value=75.0,
            unit="%",
            timestamp_seconds=60.0,
        ),),
        source="local_llm_verified_general_practice",
    )

    payload = build_review_cues(measured_metrics(), feedback)

    verified = [
        cue for cue in payload["cues"] if cue["kind"] == "verified_coaching"
    ]
    assert len(verified) == 1
    assert verified[0]["metric"] == "eye_contact_percent"


def test_review_cue_payload_is_deterministic_and_does_not_mutate_inputs():
    metrics = measured_metrics()
    feedback = verified_feedback()
    original_metrics = deepcopy(metrics)
    original_feedback = deepcopy(feedback)

    first = build_review_cues(metrics, feedback)
    second = build_review_cues(metrics, feedback)

    assert first == second
    assert metrics == original_metrics
    assert feedback == original_feedback


def test_malformed_timeline_values_fail_closed_without_crashing():
    metrics = measured_metrics()
    metrics["timeline"] = {
        "fillers": [{"phrase": "um", "start_seconds": float("nan"), "end_seconds": 2}],
        "vision": [{"timestamp_seconds": "not-a-time", "face_detected": False}],
        "pace_windows": [{"words_per_minute": float("inf")}],
        "pace_spikes": [{"start_seconds": 0, "end_seconds": 15, "words_per_minute": float("nan")}],
        "pause_events": [{"start_seconds": 1, "end_seconds": 5, "duration_seconds": 99}],
    }

    payload = build_review_cues(metrics)

    assert payload["status"] == "no_cues"
    assert payload["cues"] == []
