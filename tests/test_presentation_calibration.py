from stroke_screening.presentation_calibration import calibration_status, prepare_feedback_metrics
from stroke_screening.presentation_core import TranscriptWord, VisionSample, analyze_session, compute_metrics
from stroke_screening.presentation_store import PresentationArchive, StoredPresentation


def stored(kind, *, eye=True, wpm_words=80):
    vision = [VisionSample(float(i), 15, True, eye, .5, .5, 1, 1, 1, .5, .5, .1, .1, .01, 10) for i in range(40)]
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


def test_uncalibrated_feedback_uses_neutral_observation_hints():
    item = stored("practice")
    prepared = prepare_feedback_metrics(item.metrics, {"ready": False})
    assert prepared["feedback_mode"] == "descriptive"
    roles = {(hint["metric"], hint["role"]) for hint in prepared["role_hints"]}
    assert ("face_presence_percent", "strength") in roles
    assert ("overall_words_per_minute", "strength") in roles
    assert "personal_reference" not in prepared
