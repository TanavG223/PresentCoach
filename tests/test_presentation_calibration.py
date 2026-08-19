from stroke_screening.presentation_calibration import calibration_status, prepare_feedback_metrics
from stroke_screening.presentation_core import TranscriptWord, VisionSample, analyze_session, compute_metrics
from stroke_screening.presentation_coaching import general_coaching_hints
from stroke_screening.presentation_store import PresentationArchive, StoredPresentation


def stored(kind, *, eye=True, wpm_words=80):
    vision = [VisionSample(
        float(i), 15, True, eye, .5, .5, 1, 1, 1, .5, .5, .1, .1, .01, 10,
        detected_frame_count=15,
        contact_frame_count=15 if eye else 0,
        contact_eligible_frame_count=15,
    ) for i in range(40)]
    words = [TranscriptWord("word", i * .45, i * .45 + .2, .9) for i in range(wpm_words)]
    session = analyze_session(vision, {"words": words, "duration_seconds": 40, "waveform_rms": .05}, session_kind=kind)
    return StoredPresentation(session, compute_metrics(session))


def test_calibration_enforces_review_and_two_repeats():
    baseline = stored("baseline")
    archive = PresentationArchive("a" * 32, "Test", (baseline,), {})
    assert calibration_status(archive)["stage"] == "review_baseline"
    archive = PresentationArchive("a" * 32, "Test", (baseline,), {"baseline_session_id": baseline.session.session_id, "baseline_confirmed": True})
    assert calibration_status(archive)["stage"] == "record_repeats"
    first, second = stored("repeat"), stored("repeat")
    archive = PresentationArchive(archive.profile_id, archive.name, (baseline, first, second), archive.calibration)
    assert calibration_status(archive)["ready"] is True


def test_short_baseline_can_be_re_recorded():
    baseline = stored("baseline")
    short_session = baseline.session.__class__(
        **{**baseline.session.__dict__, "duration_seconds": 12.0}
    )
    short = StoredPresentation(short_session, baseline.metrics)
    archive = PresentationArchive(
        "a" * 32,
        "Test",
        (short,),
        {"baseline_session_id": short.session.session_id, "baseline_confirmed": False},
    )
    assert calibration_status(archive)["stage"] == "record_baseline"


def test_uncalibrated_feedback_uses_ranked_general_practice_hints():
    item = stored("practice")
    prepared = prepare_feedback_metrics(item.metrics, {"ready": False})
    assert prepared["feedback_mode"] == "general_practice"
    roles = {(hint["metric"], hint["role"]) for hint in prepared["role_hints"]}
    assert ("eye_contact_percent", "strength") in roles
    assert ("overall_words_per_minute", "strength") in roles
    assert all(hint["comparison_mode"] == "general_practice_band" for hint in prepared["role_hints"])
    assert "personal_reference" not in prepared


def test_general_policy_calls_low_contact_fast_pace_and_frequent_um_uh_actionable():
    metrics = {
        "duration_seconds": 60.0,
        "aggregate": {
            "eye_contact_percent": 42.0,
            "face_presence_percent": 98.0,
            "overall_words_per_minute": 182.0,
            "strict_filler_rate_per_minute": 6.0,
            "long_pause_rate_per_minute": 0.0,
        },
        "quality_flags": {
            "eye_contact": "good", "face_detected": "good", "audio_clear": "good",
        },
        "timeline": {
            "longest_gaze_break": {
                "start_seconds": 18.0, "duration_seconds": 7.0,
            },
            "filler_clusters": [{"start_seconds": 20.0, "count": 3}],
            "longest_pause": None,
        },
    }
    hints = general_coaching_hints(metrics)
    improvements = [hint for hint in hints if hint["role"] == "improvement"]
    assert [hint["metric"] for hint in improvements] == [
        "eye_contact_percent",
        "strict_filler_rate_per_minute",
        "overall_words_per_minute",
    ]
    assert improvements[0]["assessment"] == "frequently away from camera"
    assert "silent breath" in improvements[1]["recommendation"]


def test_two_second_gap_is_measured_but_not_automatically_criticized():
    metrics = {
        "duration_seconds": 60.0,
        "aggregate": {
            "overall_words_per_minute": 130.0,
            "strict_filler_rate_per_minute": 1.0,
            "long_pause_rate_per_minute": 0.0,
        },
        "quality_flags": {
            "eye_contact": "bad", "face_detected": "bad", "audio_clear": "good",
        },
        "timeline": {
            "filler_clusters": [],
            "longest_pause": {"start_seconds": 12.0, "duration_seconds": 2.4},
        },
    }
    hints = general_coaching_hints(metrics)
    assert not [hint for hint in hints if hint["role"] == "improvement"]
