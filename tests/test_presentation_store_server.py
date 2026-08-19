import hashlib
from io import BytesIO
import json
from pathlib import Path
from dataclasses import replace

from stroke_screening.presentation_core import TranscriptWord, VisionSample, analyze_session, compute_metrics
from stroke_screening.presentation_server import _serialize_archive, create_app
from stroke_screening.presentation_store import PresentationArchive, PresentationStore, StoredPresentation


ROOT = Path(__file__).resolve().parents[1]


class MemoryKeys:
    def __init__(self): self.values = {}
    def save(self, profile, key): self.values[profile] = key
    def load(self, profile): return self.values[profile]
    def delete(self, profile): self.values.pop(profile, None)


class FakeLLM:
    def status(self): return {"available": True, "installed": True, "model": "fake"}
    def complete_json(self, **_kwargs):
        return {
            "strengths": [
                {"text": "100 % at 31 seconds", "metric": "eye_contact_percent", "value": 100, "unit": "%", "timestamp_seconds": 31},
                {"text": "0 um/uh per min at 31 seconds", "metric": "strict_filler_rate_per_minute", "value": 0, "unit": "um/uh per min", "timestamp_seconds": 31},
            ],
            "improvements": [
                {"text": "60 WPM at 31 seconds", "metric": "overall_words_per_minute", "value": 60, "unit": "WPM", "timestamp_seconds": 31},
            ],
            "insufficient_data": [],
        }


class FakeRecorder:
    def is_active(self): return False


class FakeVideoAnalyzer:
    def __init__(self): self.calls = 0
    def is_active(self): return False
    def analyze(self, path, *, note=None):
        self.calls += 1
        assert path.read_bytes() == b"local-video"
        samples = tuple(
            VisionSample(
                timestamp_seconds=float(second), frame_count=15,
                face_detected=True, eye_contact=True, gaze_horizontal=0.5,
                gaze_vertical=0.5, yaw_degrees=0.0, pitch_degrees=0.0,
                roll_degrees=0.0, face_center_x=0.5, face_center_y=0.5,
                mouth_activity=0.1, brow_activity=0.1,
                expression_change=0.01, inference_ms=1.0,
                detected_frame_count=15, contact_frame_count=15,
                contact_eligible_frame_count=15,
            )
            for second in range(31)
        )
        words = tuple(
            TranscriptWord("word", float(second), float(second) + 0.4, 0.9)
            for second in range(31)
        )
        return analyze_session(
            samples,
            {"words": words, "duration_seconds": 31, "waveform_rms": 0.1},
            session_kind="imported",
            note=note,
        )


def test_encrypted_profile_round_trip_and_csrf(tmp_path: Path):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    app = create_app(store=store, llm=FakeLLM(), recorder=FakeRecorder(), testing=True)
    with app.test_client() as client:
        bootstrap = client.get("/api/bootstrap")
        csrf = bootstrap.get_json()["csrf_token"]
        denied = client.post("/api/profiles", json={"name": "Tanav"})
        assert denied.status_code == 403
        created = client.post(
            "/api/profiles", json={"name": "Tanav"}, headers={"X-CSRF-Token": csrf}
        )
        assert created.status_code == 201
        profile_id = created.get_json()["profile"]["id"]
        loaded = client.get(f"/api/bootstrap?profile={profile_id}").get_json()
        assert loaded["profile"] == {"id": profile_id, "name": "Tanav"}
        assert loaded["calibration"]["stage"] == "record_baseline"
        encrypted = next((tmp_path / "data").glob("*.presentcoach")).read_bytes()
        assert b"Tanav" not in encrypted


def test_uploaded_video_is_analyzed_locally_without_becoming_a_baseline(tmp_path: Path):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    analyzer = FakeVideoAnalyzer()
    app = create_app(
        store=store, llm=FakeLLM(), recorder=FakeRecorder(),
        video_analyzer=analyzer, testing=True,
    )
    with app.test_client() as client:
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles", json={"name": "Tanav"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        response = client.post(
            "/api/videos/analyze",
            data={
                "profile_id": profile["id"],
                "note": "Public-domain test",
                "video": (BytesIO(b"local-video"), "practice.webm"),
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 201
        document = response.get_json()
        assert document["session"]["session_kind"] == "imported"
        assert document["session"]["note"] == "Public-domain test"
        assert document["metrics"]["aggregate"]["analyzed_vision_fps"] == 15.0
        assert document["calibration"]["stage"] == "record_baseline"
        assert document["feedback"]["status"] == "ready"
        assert analyzer.calls == 1
        refreshed = client.post(
            f"/api/profiles/{profile['id']}/sessions/{document['session']['session_id']}/feedback",
            json={}, headers={"X-CSRF-Token": csrf},
        )
        assert refreshed.status_code == 200
        assert refreshed.get_json()["feedback"]["status"] == "ready"
        invalid = client.post(
            "/api/videos/analyze",
            data={
                "profile_id": profile["id"],
                "video": (BytesIO(b"local-video"), "practice.txt"),
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf},
        )
        assert invalid.status_code == 415
        assert analyzer.calls == 1


def test_legacy_bucket_feedback_is_invalidated_until_regenerated(tmp_path: Path):
    source = tmp_path / "legacy.webm"
    source.write_bytes(b"local-video")
    session = FakeVideoAnalyzer().analyze(source)
    legacy = replace(
        session,
        vision_metrics=tuple(
            replace(
                sample,
                detected_frame_count=None,
                contact_frame_count=None,
                contact_eligible_frame_count=None,
            )
            for sample in session.vision_metrics
        ),
    )
    feedback = {
        "status": "ready", "source": "local_llm_verified_descriptive",
        "strengths": [{"metric": "eye_contact_percent"}],
        "improvements": [], "insufficient_data": [],
    }
    archive = PresentationArchive(
        "a" * 32, "Tanav",
        (StoredPresentation(legacy, compute_metrics(session), feedback),),
        {},
    )
    document = _serialize_archive(archive)[0]
    assert document["session"]["quality_flags"]["eye_contact"] == "bad"
    assert document["feedback"]["status"] == "calibration_required"


def test_test_lab_exposes_only_allowlisted_local_clips_and_reports(tmp_path: Path):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    media = tmp_path / "media"
    reports = tmp_path / "reports"
    media.mkdir()
    reports.mkdir()
    (media / "tarun-speaking-cc0.webm").write_bytes(b"webm-evidence")
    (media / "private.webm").write_bytes(b"must-not-leak")
    (reports / "presentcoach_video_eval.json").write_text(
        '{"case_count":3,"passed":3,"pass_rate_percent":100,"cases":[]}',
        encoding="utf-8",
    )
    (reports / "presentcoach_llm_eval.json").write_text(
        '{"model":"fake","case_count":30,"passed":30,"pass_rate_percent":100}',
        encoding="utf-8",
    )
    app = create_app(
        store=store, llm=FakeLLM(), recorder=FakeRecorder(),
        test_media_dir=media, reports_dir=reports, testing=True,
    )
    with app.test_client() as client:
        lab = client.get("/api/bootstrap").get_json()["test_lab"]
        assert lab["evaluation_only"] is True
        assert lab["used_for_training"] is False
        assert lab["video_eval"] == {
            "case_count": 3, "generated_at": None, "pass_rate_percent": 100.0, "passed": 3,
        }
        assert lab["llm_eval"]["passed"] == 30
        assert lab["clips"][0]["available"] is True
        assert lab["clips"][1]["available"] is False
        clip = client.get("/api/test-media/tarun-short-distance/video")
        assert clip.status_code == 200
        assert clip.data == b"webm-evidence"
        assert client.get("/api/test-media/private/video").status_code == 404
        assert client.get("/api/test-media/../private.webm/video").status_code == 404
        denied = client.get(
            "/api/test-media/tarun-short-distance/video",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert denied.status_code == 403


def test_tracking_test_media_is_manifest_allowlisted_integrity_checked_and_range_capable(
    tmp_path: Path,
):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    media = tmp_path / "media"
    reports = tmp_path / "reports"
    media.mkdir()
    reports.mkdir()
    manifest = json.loads(
        (ROOT / "test_media" / "face_tracking_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    good_bytes = b"0123456789"
    good = manifest["clips"][0]
    good["filename"] = "range-test.webm"
    good["sha256"] = hashlib.sha256(good_bytes).hexdigest()
    bad = manifest["clips"][1]
    bad["filename"] = "tampered-test.webm"
    bad["sha256"] = hashlib.sha256(b"expected-content").hexdigest()
    (media / good["filename"]).write_bytes(good_bytes)
    (media / bad["filename"]).write_bytes(b"tampered-content")
    (media / "private.webm").write_bytes(b"must-not-leak")
    (media / "face_tracking_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        test_media_dir=media,
        reports_dir=reports,
        testing=True,
    )

    with app.test_client() as client:
        ranged = client.get(
            f"/api/test-media/{good['id']}/video",
            headers={"Range": "bytes=2-5"},
        )
        assert ranged.status_code == 206
        assert ranged.data == b"2345"
        assert ranged.headers["Content-Range"] == "bytes 2-5/10"
        assert ranged.mimetype == "video/webm"

        rejected = client.get(f"/api/test-media/{bad['id']}/video")
        assert rejected.status_code == 404
        assert bad["filename"] not in rejected.get_data(as_text=True)
        assert client.get("/api/test-media/private/video").status_code == 404
        denied = client.get(
            f"/api/test-media/{good['id']}/video",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert denied.status_code == 403
