from dataclasses import replace

from stroke_screening.presentation_core import (
    TranscriptWord,
    VisionSample,
    analyze_session,
    compute_metrics,
    generate_feedback,
)


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


class NeverCalledLLM:
    def complete_json(self, **_kwargs):
        raise AssertionError("LLM must not be called")


def test_feedback_refuses_short_recording_before_llm():
    result = generate_feedback(
        {"duration_seconds": 29.9, "calibration_ready": True}, NeverCalledLLM()
    )
    assert result.status == "refused_short_session"


def test_feedback_refuses_uncalibrated_metrics_before_llm():
    result = generate_feedback(
        {"duration_seconds": 60, "calibration_ready": False}, NeverCalledLLM()
    )
    assert result.status == "calibration_required"
