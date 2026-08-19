from dataclasses import replace

from stroke_screening.presentation_core import (
    TranscriptWord,
    VisionSample,
    analyze_session,
    compute_metrics,
    generate_feedback,
)
from stroke_screening.presentation_calibration import prepare_feedback_metrics


def vision_samples(seconds=40):
    return [
        VisionSample(
            timestamp_seconds=float(second), frame_count=15,
            face_detected=True, eye_contact=second % 5 != 0,
            gaze_horizontal=0.5, gaze_vertical=0.5,
            yaw_degrees=1 + second * 0.02, pitch_degrees=2,
            roll_degrees=0.5, face_center_x=0.5,
            face_center_y=0.5, mouth_activity=0.1 + second * 0.002,
            brow_activity=0.05 + second * 0.001,
            expression_change=0.01, inference_ms=10,
            detected_frame_count=15,
            contact_frame_count=15 if second % 5 != 0 else 0,
            contact_eligible_frame_count=15,
        )
        for second in range(seconds)
    ]


def test_analyze_session_builds_requested_model_and_filler_timestamps():
    words = [
        TranscriptWord("Hello", 0, 0.4, 0.9),
        TranscriptWord("um", 1, 1.2, 0.9),
        TranscriptWord("you", 2, 2.2, 0.9),
        TranscriptWord("know", 2.3, 2.6, 0.9),
    ]
    session = analyze_session(
        vision_samples(),
        {"words": words, "duration_seconds": 40, "waveform_rms": 0.04},
    )
    assert session.duration_seconds == 40
    assert session.audio_metrics.filler_counts["um"] == 1
    assert session.audio_metrics.filler_counts["you know"] == 1
    assert session.quality_flags["audio_clear"] == "good"
    assert session.quality_flags["face_detected"] == "good"


def test_compute_metrics_reports_timeline_and_15_fps():
    session = analyze_session(
        vision_samples(),
        {"words": [], "duration_seconds": 40, "waveform_rms": 0},
    )
    metrics = compute_metrics(session)
    assert metrics["aggregate"]["eye_contact_percent"] == 80.0
    assert metrics["aggregate"]["analyzed_vision_fps"] == 15.0
    assert metrics["timeline"]["longest_gaze_break"]["duration_seconds"] == 1.0


def test_camera_contact_and_presence_are_weighted_by_analyzed_frames():
    samples = [
        replace(
            sample,
            face_detected=True,
            eye_contact=True,
            detected_frame_count=15,
            contact_eligible_frame_count=15,
            contact_frame_count=8,
        )
        for sample in vision_samples()
    ]
    session = analyze_session(
        samples,
        {"words": [], "duration_seconds": 40, "waveform_rms": 0},
    )
    metrics = compute_metrics(session)
    assert metrics["aggregate"]["eye_contact_percent"] == 53.3
    assert metrics["aggregate"]["face_presence_percent"] == 100.0


class NeverCalledLLM:
    def complete_json(self, **_kwargs):
        raise AssertionError("LLM must not be called")


def test_feedback_refuses_short_recording_before_llm():
    result = generate_feedback(
        {"duration_seconds": 29.9, "calibration_ready": True}, NeverCalledLLM()
    )
    assert result.status == "refused_short_session"


class GeneralCoachingLLM:
    def complete_json(self, **_kwargs):
        return {
            "strengths": [
                {
                    "text": "Camera contact was 80% at 40 seconds.",
                    "metric": "eye_contact_percent", "value": 80.0,
                    "unit": "%", "timestamp_seconds": 40.0,
                },
                {
                    "text": "Face presence was 100% at 40 seconds.",
                    "metric": "face_presence_percent", "value": 100.0,
                    "unit": "%", "timestamp_seconds": 40.0,
                },
            ],
            "improvements": [{
                "text": "The contact break was 1 second at 0 seconds.",
                "metric": "longest_gaze_break_seconds", "value": 1.0,
                "unit": "seconds", "timestamp_seconds": 0.0,
            }],
            "insufficient_data": ["audio_clear"],
        }


def test_feedback_without_calibration_is_actionable_and_evidence_locked():
    session = analyze_session(
        vision_samples(),
        {"words": [], "duration_seconds": 40, "waveform_rms": 0},
    )
    metrics = prepare_feedback_metrics(compute_metrics(session), {"ready": False})
    result = generate_feedback(metrics, GeneralCoachingLLM())
    assert result.status == "ready"
    assert result.source == "local_llm_verified_general_practice"
    assert "camera contact was 80 %" in result.strengths[0].text
    assert "PresentCoach heuristic: 70% or more" in result.strengths[0].text
    assert "Keep finishing complete thoughts" in result.strengths[0].text
    assert result.improvements == ()
    assert result.insufficient_data == ("audio_clear",)


def test_compute_metrics_normalizes_strict_um_uh_and_timestamps_transcript_gaps():
    words = [
        TranscriptWord("so", 0.0, 0.2, .9),
        TranscriptWord("like", .3, .5, .9),
        TranscriptWord("um", 1.0, 1.2, .9),
        TranscriptWord("uh", 5.2, 5.4, .9),
    ]
    session = analyze_session(
        vision_samples(60),
        {"words": words, "duration_seconds": 60, "waveform_rms": .04},
    )
    metrics = compute_metrics(session)
    assert metrics["aggregate"]["filler_count"] == 4
    assert metrics["aggregate"]["strict_filler_count"] == 2
    assert metrics["aggregate"]["strict_filler_rate_per_minute"] == 2.0
    assert metrics["timeline"]["longest_pause"] == {
        "start_seconds": 1.2,
        "end_seconds": 5.2,
        "duration_seconds": 4.0,
    }


def test_missing_face_does_not_extend_a_camera_contact_break():
    samples = vision_samples(40)
    samples[0] = replace(samples[0], eye_contact=False)
    samples[0] = replace(samples[0], contact_frame_count=0)
    samples[1] = replace(
        samples[1], face_detected=False, eye_contact=None,
        detected_frame_count=0, contact_frame_count=0,
        contact_eligible_frame_count=0,
    )
    samples[2] = replace(samples[2], eye_contact=False, contact_frame_count=0)
    samples[3] = replace(samples[3], eye_contact=True)
    session = analyze_session(
        samples,
        {"words": [], "duration_seconds": 40, "waveform_rms": 0},
    )
    assert compute_metrics(session)["timeline"]["longest_gaze_break"]["duration_seconds"] == 1.0


def test_vision_quality_requires_fourteen_analyzed_frames_per_second():
    slow = [replace(sample, frame_count=1) for sample in vision_samples()]
    session = analyze_session(
        slow,
        {"words": [], "duration_seconds": 40, "waveform_rms": 0},
    )
    assert session.quality_flags["face_detected"] == "bad"
    assert session.quality_flags["eye_contact"] == "bad"
    forged_legacy = replace(
        session,
        quality_flags={name: "good" for name in session.quality_flags},
    )
    recomputed = compute_metrics(forged_legacy)
    assert recomputed["quality_flags"]["face_detected"] == "bad"
    assert recomputed["quality_flags"]["eye_contact"] == "bad"


def test_legacy_bucket_only_vision_abstains_from_percentage_coaching():
    legacy = [
        replace(
            sample,
            detected_frame_count=None,
            contact_frame_count=None,
            contact_eligible_frame_count=None,
        )
        for sample in vision_samples()
    ]
    session = analyze_session(
        legacy,
        {"words": [], "duration_seconds": 40, "waveform_rms": 0},
    )
    metrics = compute_metrics(session)
    assert metrics["quality_flags"]["face_detected"] == "bad"
    assert metrics["quality_flags"]["eye_contact"] == "bad"


def test_python_corrects_llm_section_for_a_deterministic_strength():
    legacy = [
        replace(
            sample,
            detected_frame_count=None,
            contact_frame_count=None,
            contact_eligible_frame_count=None,
        )
        for sample in vision_samples()
    ]
    words = [TranscriptWord("word", index * .5, index * .5 + .2, .9) for index in range(80)]
    session = analyze_session(
        legacy,
        {"words": words, "duration_seconds": 40, "waveform_rms": .04},
    )
    metrics = prepare_feedback_metrics(compute_metrics(session), {"ready": False})

    class MisplacedLLM:
        def complete_json(self, **_kwargs):
            return {
                "strengths": [{
                    "text": "120 WPM at 40 seconds",
                    "metric": "overall_words_per_minute", "value": 120,
                    "unit": "WPM", "timestamp_seconds": 40,
                }],
                "improvements": [{
                    "text": "0 um/uh per min at 40 seconds",
                    "metric": "strict_filler_rate_per_minute", "value": 0,
                    "unit": "um/uh per min", "timestamp_seconds": 40,
                }],
                "insufficient_data": [
                    "face_detected", "eye_contact", "head_stability",
                    "expression_variety",
                ],
            }

    feedback = generate_feedback(metrics, MisplacedLLM())
    assert [claim.metric for claim in feedback.strengths] == [
        "overall_words_per_minute", "strict_filler_rate_per_minute",
    ]
    assert feedback.improvements == ()
