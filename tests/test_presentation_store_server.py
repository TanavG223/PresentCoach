import hashlib
from io import BytesIO
import json
from pathlib import Path
from dataclasses import replace
import tempfile
import threading
import time

import itsdangerous.timed
import pytest

from stroke_screening.presentation_audio import AudioCaptureError
from stroke_screening.presentation_calibration import prepare_feedback_metrics
from stroke_screening.presentation_camera import CameraSessionError
from stroke_screening.presentation_core import (
    TranscriptWord,
    VisionSample,
    analyze_session,
    compute_metrics,
    generate_feedback,
)
from stroke_screening.presentation_recording import (
    PresentationRecordingError,
    RecordedPresentation,
    RecordingMedia,
)
from stroke_screening.presentation_server import (
    MEDIA_PLAYBACK_RANGE_BYTES,
    _remove_access_url,
    _serialize_archive,
    _write_access_url,
    create_app,
)
from stroke_screening.presentation_store import (
    MEDIA_CHUNK_BYTES,
    MEDIA_TAG_BYTES,
    PresentationArchive,
    PresentationStorageError,
    PresentationStore,
    StoredPresentation,
)
from stroke_screening.presentation_test_lab import TEST_MEDIA_BY_ID
from stroke_screening.presentation_video import ImportedVideoResult, PlaybackVideo


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


class InterruptingLLM(FakeLLM):
    """Simulate a process interruption only after durable session persistence."""

    def __init__(self, store):
        self.store = store
        self.profile_id = None

    def complete_json(self, **_kwargs):
        assert isinstance(self.profile_id, str)
        archive = self.store.load_profile(self.profile_id)
        assert len(archive.sessions) == 1
        stored = archive.sessions[0]
        assert stored.feedback["status"] == "feedback_pending"
        assert self.store.session_media(
            self.profile_id, stored.session.session_id
        ).source == "upload"
        raise KeyboardInterrupt("simulated process interruption during local AI")


class FakeRecorder:
    def is_active(self): return False


class ExclusiveFakeRecorder:
    def __init__(self):
        self.active = False
        self.owner = "recording-owner"
        self.cancel_calls = 0

    def is_active(self):
        return self.active

    def start(self, **_kwargs):
        assert not self.active
        self.active = True
        return self.owner

    def cancel(self, owner):
        assert owner == self.owner
        self.cancel_calls += 1
        self.active = False

    def status(self, owner):
        assert owner == self.owner
        return {"recording": self.active, "elapsed_seconds": 1.0}


class FailFirstStartRecorder(ExclusiveFakeRecorder):
    def __init__(self):
        super().__init__()
        self.start_attempts = 0

    def start(self, **kwargs):
        self.start_attempts += 1
        if self.start_attempts == 1:
            raise PresentationRecordingError("camera start failed safely")
        return super().start(**kwargs)


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

    def analyze_with_playback(self, path, destination, *, note=None):
        presentation = self.analyze(path, note=note)
        destination.write_bytes(b"browser-video")
        return ImportedVideoResult(
            session=presentation,
            playback=PlaybackVideo(
                path=destination,
                mime_type="video/mp4",
                duration_seconds=31.0,
                width=960,
                height=540,
            ),
            media_error=None,
        )


class BlockingPlaybackAnalyzer(FakeVideoAnalyzer):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def analyze_with_playback(self, path, destination, *, note=None):
        self.entered.set()
        assert self.release.wait(timeout=3.0), "test did not release video analysis"
        return super().analyze_with_playback(path, destination, note=note)


class FailingPlaybackAnalyzer(FakeVideoAnalyzer):
    def analyze_with_playback(self, path, destination, *, note=None):
        del destination
        return ImportedVideoResult(
            session=self.analyze(path, note=note),
            playback=None,
            media_error=(
                "The measurements were saved, but a browser-safe local video "
                "could not be created."
            ),
        )


class LargePlaybackAnalyzer(FakeVideoAnalyzer):
    def analyze_with_playback(self, path, destination, *, note=None):
        presentation = self.analyze(path, note=note)
        destination.write_bytes(
            b"a" * MEDIA_CHUNK_BYTES
            + b"b" * MEDIA_CHUNK_BYTES
            + b"c" * MEDIA_CHUNK_BYTES
        )
        return ImportedVideoResult(
            session=presentation,
            playback=PlaybackVideo(
                path=destination,
                mime_type="video/mp4",
                duration_seconds=31.0,
                width=960,
                height=540,
            ),
            media_error=None,
        )


class RangeCapPlaybackAnalyzer(FakeVideoAnalyzer):
    def analyze_with_playback(self, path, destination, *, note=None):
        presentation = self.analyze(path, note=note)
        destination.write_bytes(b"".join(
            bytes([ord("a") + index]) * MEDIA_CHUNK_BYTES
            for index in range(6)
        ))
        return ImportedVideoResult(
            session=presentation,
            playback=PlaybackVideo(
                path=destination,
                mime_type="video/mp4",
                duration_seconds=31.0,
                width=960,
                height=540,
            ),
            media_error=None,
        )


class CompletedMediaRecorder:
    def __init__(self, source_session):
        self.source_session = source_session
        self.temporary_path = None

    def is_active(self):
        return False

    def stop(self, owner):
        assert owner == "recording-owner"
        temporary = tempfile.TemporaryDirectory(prefix="presentcoach-test-live-")
        self.temporary_path = Path(temporary.name)
        path = self.temporary_path / "session.mp4"
        path.write_bytes(b"recorded-local-video")
        return RecordedPresentation(
            self.source_session,
            RecordingMedia(path=path, _temporary=temporary),
        )


class TransitionRecorder(CompletedMediaRecorder):
    def __init__(self, source_session):
        super().__init__(source_session)
        self.active = True

    def is_active(self):
        return self.active

    def stop(self, owner):
        self.active = False
        return super().stop(owner)

    def start(self, **_kwargs):
        assert not self.active
        self.active = True
        return "next-recording-owner"


class FailingTerminalRecorder(ExclusiveFakeRecorder):
    def __init__(self, terminal_action: str, error: Exception):
        super().__init__()
        self.terminal_action = terminal_action
        self.error = error

    def stop(self, owner):
        assert self.terminal_action == "stop"
        assert owner == self.owner
        self.active = False
        raise self.error

    def cancel(self, owner):
        assert self.terminal_action == "cancel"
        assert owner == self.owner
        self.active = False
        raise self.error


class RetryableTerminalRecorder(ExclusiveFakeRecorder):
    def __init__(self, failing_action: str):
        super().__init__()
        self.failing_action = failing_action
        self.failed_once = False

    def stop(self, owner):
        assert owner == self.owner
        if self.failing_action == "stop" and not self.failed_once:
            self.failed_once = True
            raise PresentationRecordingError("retryable stop failure")
        raise AssertionError("the retryable test does not finalize a stop")

    def cancel(self, owner):
        assert owner == self.owner
        if self.failing_action == "cancel" and not self.failed_once:
            self.failed_once = True
            raise PresentationRecordingError("retryable cancel failure")
        return super().cancel(owner)


def _authorize(client, app) -> None:
    capability = app.extensions["presentation_access_capability"]
    response = client.get("/", query_string={"access": capability})
    assert response.status_code == 303
    assert response.headers["Location"] == "/"
    assert "access=" not in response.headers["Location"]
    response.close()


def test_private_launch_capability_unlocks_one_browser_and_strips_query(
    tmp_path: Path,
):
    capability = "private-local-capability-" + "a" * 48
    app = create_app(
        store=PresentationStore(
            data_dir=tmp_path / "data", key_store=MemoryKeys()
        ),
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        access_capability=capability,
        testing=True,
    )
    assert app.permanent_session_lifetime.total_seconds() >= 24 * 60 * 60
    client = app.test_client()

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/test-media/not-allowlisted/video").status_code == 404
    locked = client.get("/")
    assert locked.status_code == 401
    assert capability not in locked.get_data(as_text=True)
    blocked = client.get("/api/bootstrap")
    assert blocked.status_code == 401
    assert blocked.get_json()["code"] == "local_access_required"
    with client.session_transaction() as browser_session:
        browser_session["csrf_token"] = "attacker-fixed-token"
        browser_session["presentation_owner"] = "attacker-fixed-owner"
    for path, method in (
        ("/api/recordings/status", "GET"),
        ("/api/recordings/stream.mjpg", "GET"),
        ("/api/profiles/unknown/sessions/unknown/video?token=stolen", "GET"),
        ("/api/profiles", "POST"),
        ("/api/profiles/unknown/calibration/confirm", "POST"),
        ("/api/profiles/unknown/sessions/unknown/feedback", "POST"),
        ("/api/profiles/unknown/sessions/unknown/video", "DELETE"),
        ("/api/recordings/start", "POST"),
        ("/api/recordings/stop", "POST"),
        ("/api/recordings/cancel", "POST"),
        ("/api/videos/analyze", "POST"),
    ):
        response = client.open(path, method=method)
        assert response.status_code == 401
        assert response.get_json()["code"] == "local_access_required"

    invalid = client.get("/", query_string={"access": "b" * 64})
    assert invalid.status_code == 401
    assert capability not in invalid.get_data(as_text=True)
    exchanged = client.get("/", query_string={"access": capability})
    assert exchanged.status_code == 303
    assert exchanged.headers["Location"] == "/"
    assert "access=" not in exchanged.headers["Location"]
    assert capability not in exchanged.get_data(as_text=True)
    assert capability not in "\n".join(exchanged.headers.values())
    assert exchanged.headers["Referrer-Policy"] == "no-referrer"
    content_security_policy = exchanged.headers["Content-Security-Policy"]
    assert "script-src 'self';" in content_security_policy
    assert "script-src 'self' 'unsafe-inline'" not in content_security_policy
    assert "style-src 'self';" in content_security_policy
    assert "style-src-attr 'unsafe-inline';" in content_security_policy
    cookie = exchanged.headers["Set-Cookie"]
    assert "presentcoach_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Expires=" not in cookie
    assert "Max-Age=" not in cookie
    with client.session_transaction() as browser_session:
        assert browser_session.permanent is False
        assert browser_session.get("csrf_token") is None
        assert browser_session.get("presentation_owner") is None
        authorization_nonce = browser_session.get(
            "presentation_authorization_nonce"
        )
        assert isinstance(authorization_nonce, str)
        assert len(authorization_nonce) >= 32
    assert client.get("/").status_code == 200
    assert client.get("/api/bootstrap").status_code == 200


def test_private_launch_url_is_written_user_only_and_removed_safely(
    tmp_path: Path,
):
    access_file = tmp_path / "presentcoach-data" / "access-url"
    access_url = "http://127.0.0.1:8765/?access=" + "d" * 64

    _write_access_url(access_file, access_url)

    assert access_file.read_text(encoding="utf-8") == access_url + "\n"
    assert access_file.stat().st_mode & 0o777 == 0o600
    assert access_file.parent.stat().st_mode & 0o777 == 0o700
    _remove_access_url(access_file, access_url + "-different-process")
    assert access_file.exists()
    _remove_access_url(access_file, access_url)
    assert not access_file.exists()


def test_playback_grant_is_bound_to_authorized_browser_session(tmp_path: Path):
    capability = "private-local-capability-" + "c" * 48
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        video_analyzer=FakeVideoAnalyzer(),
        access_capability=capability,
        testing=True,
    )
    owner_client = app.test_client()
    _authorize(owner_client, app)
    csrf = owner_client.get("/api/bootstrap").get_json()["csrf_token"]
    profile = owner_client.post(
        "/api/profiles",
        json={"name": "Session-bound replay"},
        headers={"X-CSRF-Token": csrf},
    ).get_json()["profile"]
    uploaded = owner_client.post(
        "/api/videos/analyze",
        data={
            "profile_id": profile["id"],
            "video": (BytesIO(b"local-video"), "practice.webm"),
        },
        content_type="multipart/form-data",
        headers={"X-CSRF-Token": csrf},
    ).get_json()
    owner_playback_url = uploaded["media"]["playback_url"]
    assert owner_client.get(owner_playback_url).data == b"browser-video"

    fresh_client = app.test_client()
    assert fresh_client.get("/api/bootstrap").status_code == 401
    copied_while_locked = fresh_client.get(owner_playback_url)
    assert copied_while_locked.status_code == 401
    assert copied_while_locked.get_json()["code"] == "local_access_required"

    _authorize(fresh_client, app)
    copied_after_unlock = fresh_client.get(owner_playback_url)
    assert copied_after_unlock.status_code == 403
    assert copied_after_unlock.get_json()["code"] == "media_access_denied"
    fresh_bootstrap = fresh_client.get(
        "/api/bootstrap", query_string={"profile": profile["id"]}
    ).get_json()
    fresh_playback_url = fresh_bootstrap["sessions"][0]["media"]["playback_url"]
    assert fresh_playback_url != owner_playback_url
    fresh_playback = fresh_client.get(fresh_playback_url)
    assert fresh_playback.status_code == 200
    assert fresh_playback.data == b"browser-video"


def test_encrypted_profile_round_trip_and_csrf(tmp_path: Path):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    app = create_app(store=store, llm=FakeLLM(), recorder=FakeRecorder(), testing=True)
    with app.test_client() as client:
        _authorize(client, app)
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


def test_second_tab_cannot_replace_or_cancel_an_active_recording(tmp_path: Path):
    recorder = ExclusiveFakeRecorder()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Two tabs"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        first_client = "recording-client-tab-a"
        second_client = "recording-client-tab-b"

        started = client.post(
            "/api/recordings/start",
            json={"profile_id": profile["id"], "client_id": first_client},
            headers={"X-CSRF-Token": csrf},
        )
        assert started.status_code == 200
        assert started.get_json()["preview_url"].endswith(
            f"client_id={first_client}"
        )

        reopened = client.get(
            "/",
            query_string={
                "access": app.extensions["presentation_access_capability"]
            },
        )
        assert reopened.status_code == 303
        resumed_status = client.get(
            "/api/recordings/status", query_string={"client_id": first_client}
        )
        assert resumed_status.status_code == 200
        assert resumed_status.get_json()["recording"] is True

        conflicting_start = client.post(
            "/api/recordings/start",
            json={"profile_id": profile["id"], "client_id": second_client},
            headers={"X-CSRF-Token": csrf},
        )
        assert conflicting_start.status_code == 409
        assert recorder.active is True
        assert recorder.cancel_calls == 0

        foreign_cancel = client.post(
            "/api/recordings/cancel",
            json={"client_id": second_client},
            headers={"X-CSRF-Token": csrf},
        )
        assert foreign_cancel.status_code == 409
        assert recorder.active is True
        assert recorder.cancel_calls == 0
        foreign_cancel.close()

        owned_cancel = client.post(
            "/api/recordings/cancel",
            json={"client_id": first_client},
            headers={"X-CSRF-Token": csrf},
        )
        assert owned_cancel.status_code == 200
        assert recorder.active is False
        assert recorder.cancel_calls == 1
        owned_cancel.close()


def test_stale_pre_recording_cookie_cannot_overwrite_winning_owner(tmp_path: Path):
    recorder = ExclusiveFakeRecorder()
    app = create_app(
        store=PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys()),
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    assert app.config["SESSION_REFRESH_EACH_REQUEST"] is False
    winner = app.test_client()
    _authorize(winner, app)
    csrf = winner.get("/api/bootstrap").get_json()["csrf_token"]
    profile = winner.post(
        "/api/profiles",
        json={"name": "Cookie transition"},
        headers={"X-CSRF-Token": csrf},
    ).get_json()["profile"]
    cookie_name = app.config["SESSION_COOKIE_NAME"]
    pre_recording_cookie = winner.get_cookie(cookie_name)
    assert pre_recording_cookie is not None

    stale_start = app.test_client()
    stale_status = app.test_client()
    stale_start.set_cookie(cookie_name, pre_recording_cookie.value)
    stale_status.set_cookie(cookie_name, pre_recording_cookie.value)
    winning_client_id = "recording-client-cookie-winner"
    started = winner.post(
        "/api/recordings/start",
        json={
            "profile_id": profile["id"],
            "client_id": winning_client_id,
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert started.status_code == 200
    winning_cookie = winner.get_cookie(cookie_name)
    assert winning_cookie is not None
    assert winning_cookie.value != pre_recording_cookie.value

    conflicting = stale_start.post(
        "/api/recordings/start",
        json={
            "profile_id": profile["id"],
            "client_id": "recording-client-cookie-stale",
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert conflicting.status_code == 409
    assert conflicting.headers.getlist("Set-Cookie") == []
    stale_poll = stale_status.get(
        "/api/recordings/status",
        query_string={"client_id": winning_client_id},
    )
    assert stale_poll.status_code == 409
    assert stale_poll.headers.getlist("Set-Cookie") == []

    winning_poll = winner.get(
        "/api/recordings/status",
        query_string={"client_id": winning_client_id},
    )
    assert winning_poll.status_code == 200
    assert winning_poll.get_json()["recording"] is True
    assert recorder.active is True


def test_failed_start_cookie_cannot_overwrite_subsequent_owner(tmp_path: Path):
    recorder = FailFirstStartRecorder()
    app = create_app(
        store=PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys()),
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    setup = app.test_client()
    _authorize(setup, app)
    csrf = setup.get("/api/bootstrap").get_json()["csrf_token"]
    profile = setup.post(
        "/api/profiles",
        json={"name": "Failed start cookie"},
        headers={"X-CSRF-Token": csrf},
    ).get_json()["profile"]
    cookie_name = app.config["SESSION_COOKIE_NAME"]
    pre_start_cookie = setup.get_cookie(cookie_name)
    assert pre_start_cookie is not None
    failing_client = app.test_client()
    winning_client = app.test_client()
    failing_client.set_cookie(cookie_name, pre_start_cookie.value)
    winning_client.set_cookie(cookie_name, pre_start_cookie.value)

    failed = failing_client.post(
        "/api/recordings/start",
        json={
            "profile_id": profile["id"],
            "client_id": "recording-client-failed-start",
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert failed.status_code == 409
    assert failed.headers.getlist("Set-Cookie") == []

    winning_client_id = "recording-client-after-failed-start"
    started = winning_client.post(
        "/api/recordings/start",
        json={"profile_id": profile["id"], "client_id": winning_client_id},
        headers={"X-CSRF-Token": csrf},
    )
    assert started.status_code == 200
    failed.close()
    status = winning_client.get(
        "/api/recordings/status",
        query_string={"client_id": winning_client_id},
    )
    assert status.status_code == 200
    assert status.get_json()["recording"] is True
    assert recorder.active is True


def test_uploaded_video_is_analyzed_locally_without_becoming_a_baseline(
    tmp_path: Path, monkeypatch,
):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    analyzer = FakeVideoAnalyzer()
    app = create_app(
        store=store, llm=FakeLLM(), recorder=FakeRecorder(),
        video_analyzer=analyzer, testing=True,
    )
    # Keep browser authorization valid long enough to isolate the one-hour
    # playback-grant expiry in this test.
    app.config["PERMANENT_SESSION_LIFETIME"] = 7200
    with app.test_client() as client:
        _authorize(client, app)
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
        response.close()
        assert document["session"]["session_kind"] == "imported"
        assert document["session"]["note"] == "Public-domain test"
        assert document["metrics"]["aggregate"]["analyzed_vision_fps"] == 15.0
        assert document["calibration"]["stage"] == "record_baseline"
        assert document["feedback"]["status"] == "ready"
        assert document["coaching_cues"]["schema_version"] == "presentcoach.review-cues.v1"
        assert document["coaching_cues"]["session_id"] == document["session"]["session_id"]
        assert document["coaching_cues"]["limitations"]["infers_confidence"] is False
        assert document["media"]["available"] is True
        assert document["media"]["mime_type"] == "video/mp4"
        assert document["media"]["source"] == "upload"
        assert document["media"]["token_expires_in_seconds"] == 3600
        assert analyzer.calls == 1
        encrypted_media = next((tmp_path / "data").glob("*.presentcoach-media"))
        ciphertext = encrypted_media.read_bytes()
        assert b"local-video" not in ciphertext
        assert b"browser-video" not in ciphertext
        ranged = client.get(
            document["media"]["playback_url"], headers={"Range": "bytes=2-5"}
        )
        assert ranged.status_code == 206
        assert ranged.data == b"owse"
        assert ranged.headers["Content-Range"] == "bytes 2-5/13"
        assert ranged.headers["Accept-Ranges"] == "bytes"
        assert ranged.headers["Cache-Control"] == "no-store"
        ranged.close()
        suffix = client.get(
            document["media"]["playback_url"], headers={"Range": "bytes=-5"}
        )
        assert suffix.status_code == 206
        assert suffix.data == b"video"
        suffix.close()
        invalid_range = client.get(
            document["media"]["playback_url"], headers={"Range": "bytes=50-"}
        )
        assert invalid_range.status_code == 416
        assert invalid_range.headers["Content-Range"] == "bytes */13"
        assert invalid_range.get_json()["code"] == "media_range_invalid"
        denied = client.get(
            document["media"]["playback_url"],
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert denied.status_code == 403
        assert denied.get_json()["code"] == "media_access_denied"
        missing_token = client.get(
            f"/api/profiles/{profile['id']}/sessions/{document['session']['session_id']}/video"
        )
        assert missing_token.status_code == 403
        loaded = client.get(f"/api/bootstrap?profile={profile['id']}").get_json()
        assert loaded["sessions"][0]["media"]["available"] is True
        assert loaded["sessions"][0]["coaching_cues"] == document["coaching_cues"]
        refreshed = client.post(
            f"/api/profiles/{profile['id']}/sessions/{document['session']['session_id']}/feedback",
            json={}, headers={"X-CSRF-Token": csrf},
        )
        assert refreshed.status_code == 200
        assert refreshed.get_json()["feedback"]["status"] == "ready"
        assert refreshed.get_json()["coaching_cues"]["schema_version"] == "presentcoach.review-cues.v1"
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
        issued_at = time.time()
        with monkeypatch.context() as scoped:
            scoped.setattr(
                itsdangerous.timed.time, "time", lambda: issued_at + 3602
            )
            expired = client.get(document["media"]["playback_url"])
        assert expired.status_code == 403
        assert expired.get_json()["code"] == "media_access_denied"
        assert expired.headers.getlist("Set-Cookie") == []
        # The one-hour playback grant expires independently; the process-local
        # browser-session authorization remains valid for the active app run.
        resumed = client.get(
            f"/api/bootstrap?profile={profile['id']}"
        ).get_json()
        csrf = resumed["csrf_token"]
        current_playback_url = resumed["sessions"][0]["media"]["playback_url"]
        removed = client.delete(
            (
                f"/api/profiles/{profile['id']}/sessions/"
                f"{document['session']['session_id']}/video"
            ),
            headers={"X-CSRF-Token": csrf},
        )
        assert removed.status_code == 200
        assert removed.get_json()["media"] == {"available": False}
        assert not list((tmp_path / "data").glob("*.presentcoach-media"))
        unavailable = client.get(current_playback_url)
        assert unavailable.status_code == 404


def test_completed_upload_survives_interruption_during_optional_local_feedback(
    tmp_path: Path,
):
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    interrupted_llm = InterruptingLLM(store)
    app = create_app(
        store=store,
        llm=interrupted_llm,
        recorder=FakeRecorder(),
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Crash-safe feedback"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        interrupted_llm.profile_id = profile["id"]

        with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
            client.post(
                "/api/videos/analyze",
                data={
                    "profile_id": profile["id"],
                    "video": (BytesIO(b"local-video"), "practice.webm"),
                },
                content_type="multipart/form-data",
                headers={"X-CSRF-Token": csrf},
            )

    archive = store.load_profile(profile["id"])
    assert len(archive.sessions) == 1
    stored = archive.sessions[0]
    assert stored.feedback["status"] == "feedback_pending"
    assert store.session_media(profile["id"], stored.session.session_id).source == "upload"

    restarted = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    with restarted.test_client() as client:
        _authorize(client, restarted)
        loaded = client.get(
            "/api/bootstrap", query_string={"profile": profile["id"]}
        ).get_json()
        pending = loaded["sessions"][0]
        assert pending["feedback"]["status"] == "feedback_pending"
        assert pending["media"]["available"] is True
        regenerated = client.post(
            f"/api/profiles/{profile['id']}/sessions/{stored.session.session_id}/feedback",
            json={},
            headers={"X-CSRF-Token": loaded["csrf_token"]},
        )
        assert regenerated.status_code == 200
        assert regenerated.get_json()["feedback"]["status"] == "ready"


def test_uploaded_analysis_survives_browser_normalization_failure(tmp_path: Path):
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        video_analyzer=FailingPlaybackAnalyzer(),
        testing=True,
    )

    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Normalization fallback"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        response = client.post(
            "/api/videos/analyze",
            data={
                "profile_id": profile["id"],
                "video": (BytesIO(b"local-video"), "legacy.mov"),
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf},
        )

        assert response.status_code == 201
        document = response.get_json()
        assert document["session"]["session_kind"] == "imported"
        assert document["metrics"]["aggregate"]["analyzed_vision_fps"] == 15.0
        assert document["feedback"]["status"] == "ready"
        assert document["coaching_cues"]["schema_version"] == "presentcoach.review-cues.v1"
        assert document["media"] == {
            "available": False,
            "message": "The measurements were saved, but the local video is unavailable.",
        }
        assert not list((tmp_path / "data").glob("*.presentcoach-media"))

        loaded = client.get(
            f"/api/bootstrap?profile={profile['id']}"
        ).get_json()["sessions"][0]
        assert loaded["session"]["session_id"] == document["session"]["session_id"]
        assert loaded["coaching_cues"] == document["coaching_cues"]
        assert loaded["media"] == {"available": False}


def test_video_route_rejects_corrupt_second_requested_chunk_before_streaming(
    tmp_path: Path,
):
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        video_analyzer=LargePlaybackAnalyzer(),
        testing=True,
    )

    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Corruption test"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        uploaded = client.post(
            "/api/videos/analyze",
            data={
                "profile_id": profile["id"],
                "video": (BytesIO(b"local-video"), "source.mov"),
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf},
        ).get_json()
        playback_url = uploaded["media"]["playback_url"]
        encrypted = next((tmp_path / "data").glob("*.presentcoach-media"))
        payload = bytearray(encrypted.read_bytes())
        second_record = len(payload) - 2 * (MEDIA_CHUNK_BYTES + MEDIA_TAG_BYTES)
        payload[second_record + 17] ^= 0x01
        encrypted.write_bytes(payload)

        first_only = client.get(
            playback_url,
            headers={"Range": "bytes=0-1023"},
        )
        assert first_only.status_code == 206
        assert first_only.data == b"a" * 1024
        first_only.close()

        crossing = client.get(
            playback_url,
            headers={"Range": f"bytes=0-{MEDIA_CHUNK_BYTES}"},
            buffered=False,
        )

        assert crossing.status_code == 404
        assert crossing.mimetype == "application/json"
        assert crossing.get_json()["code"] == "media_unavailable"
        assert "Content-Range" not in crossing.headers
        assert crossing.headers.get("Content-Disposition") != "inline"
        crossing.close()


def test_explicit_video_ranges_are_capped_to_four_mebibytes(tmp_path: Path):
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=FakeRecorder(),
        video_analyzer=RangeCapPlaybackAnalyzer(),
        testing=True,
    )
    total_bytes = 6 * MEDIA_CHUNK_BYTES

    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Range cap"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        uploaded = client.post(
            "/api/videos/analyze",
            data={
                "profile_id": profile["id"],
                "video": (BytesIO(b"local-video"), "source.mov"),
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": csrf},
        ).get_json()
        playback_url = uploaded["media"]["playback_url"]

        full = client.get(playback_url)
        assert full.status_code == 200
        assert len(full.data) == total_bytes
        assert "Content-Range" not in full.headers
        full.close()

        open_ended = client.get(
            playback_url, headers={"Range": "bytes=0-"}
        )
        assert open_ended.status_code == 206
        assert len(open_ended.data) == MEDIA_PLAYBACK_RANGE_BYTES
        assert open_ended.data[:1] == b"a"
        assert open_ended.data[-1:] == b"d"
        assert open_ended.headers["Content-Range"] == (
            f"bytes 0-{MEDIA_PLAYBACK_RANGE_BYTES - 1}/{total_bytes}"
        )
        open_ended.close()

        start = MEDIA_CHUNK_BYTES
        oversized = client.get(
            playback_url,
            headers={"Range": f"bytes={start}-{total_bytes - 1}"},
        )
        assert oversized.status_code == 206
        assert len(oversized.data) == MEDIA_PLAYBACK_RANGE_BYTES
        assert oversized.data[:1] == b"b"
        assert oversized.data[-1:] == b"e"
        assert oversized.headers["Content-Range"] == (
            f"bytes {start}-{start + MEDIA_PLAYBACK_RANGE_BYTES - 1}/{total_bytes}"
        )
        oversized.close()

        suffix = client.get(
            playback_url,
            headers={"Range": f"bytes=-{5 * MEDIA_CHUNK_BYTES}"},
        )
        suffix_start = total_bytes - MEDIA_PLAYBACK_RANGE_BYTES
        assert suffix.status_code == 206
        assert len(suffix.data) == MEDIA_PLAYBACK_RANGE_BYTES
        assert suffix.data[:1] == b"c"
        assert suffix.data[-1:] == b"f"
        assert suffix.headers["Content-Range"] == (
            f"bytes {suffix_start}-{total_bytes - 1}/{total_bytes}"
        )
        suffix.close()


def test_completed_live_recording_is_encrypted_and_playable_then_temp_is_removed(
    tmp_path: Path,
):
    source = tmp_path / "source.webm"
    source.write_bytes(b"local-video")
    presentation = replace(
        FakeVideoAnalyzer().analyze(source), session_kind="practice"
    )
    recorder = CompletedMediaRecorder(presentation)
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )

    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Live presenter"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        with client.session_transaction() as browser_session:
            browser_session["presentation_owner"] = "recording-owner"
            browser_session["presentation_profile"] = profile["id"]
            browser_session["presentation_client_id"] = "recording-client-live"

        response = client.post(
            "/api/recordings/stop",
            json={"client_id": "recording-client-live"},
            headers={"X-CSRF-Token": csrf},
        )

        assert response.status_code == 201
        document = response.get_json()
        assert document["media"]["available"] is True
        assert document["media"]["source"] == "recording"
        assert recorder.temporary_path is not None
        assert not recorder.temporary_path.exists()
        playback = client.get(document["media"]["playback_url"])
        assert playback.status_code == 200
        assert playback.data == b"recorded-local-video"
        playback.close()


def test_new_recording_waits_until_previous_stop_response_is_closed(tmp_path: Path):
    source = tmp_path / "source.webm"
    source.write_bytes(b"local-video")
    presentation = replace(
        FakeVideoAnalyzer().analyze(source), session_kind="practice"
    )
    recorder = TransitionRecorder(presentation)
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )

    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Transition lock"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        with client.session_transaction() as browser_session:
            browser_session["presentation_owner"] = "recording-owner"
            browser_session["presentation_profile"] = profile["id"]
            browser_session["presentation_client_id"] = "recording-client-first"

        stopped = client.post(
            "/api/recordings/stop",
            json={"client_id": "recording-client-first"},
            headers={"X-CSRF-Token": csrf},
        )
        assert stopped.status_code == 201
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        assert transition_lock.locked()

        blocked = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": "recording-client-second",
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert blocked.status_code == 409
        assert recorder.active is False

        stopped.close()
        assert not transition_lock.locked()
        restarted = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": "recording-client-second",
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert restarted.status_code == 200
        assert recorder.active is True


def test_new_recording_waits_until_cancel_response_is_closed(tmp_path: Path):
    recorder = ExclusiveFakeRecorder()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )

    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Cancel transition lock"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        started = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": "recording-client-first",
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert started.status_code == 200

        cancelled = client.post(
            "/api/recordings/cancel",
            json={"client_id": "recording-client-first"},
            headers={"X-CSRF-Token": csrf},
        )
        assert cancelled.status_code == 200
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        assert transition_lock.locked()
        assert recorder.active is False

        blocked = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": "recording-client-second",
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert blocked.status_code == 409
        assert recorder.active is False

        cancelled.close()
        assert not transition_lock.locked()
        restarted = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": "recording-client-second",
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert restarted.status_code == 200
        assert recorder.active is True


@pytest.mark.parametrize("terminal_action", ("stop", "cancel"))
def test_terminal_failure_retains_owner_when_recorder_remains_active(
    tmp_path: Path, terminal_action: str,
):
    recorder = RetryableTerminalRecorder(terminal_action)
    app = create_app(
        store=PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys()),
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Retryable terminal transition"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        client_id = "recording-client-retryable-terminal"
        started = client.post(
            "/api/recordings/start",
            json={"profile_id": profile["id"], "client_id": client_id},
            headers={"X-CSRF-Token": csrf},
        )
        assert started.status_code == 200

        failed = client.post(
            f"/api/recordings/{terminal_action}",
            json={"client_id": client_id},
            headers={"X-CSRF-Token": csrf},
        )
        assert failed.status_code == 409
        assert failed.headers.getlist("Set-Cookie") == []
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        assert transition_lock.locked()
        assert recorder.active is True
        failed.close()

        status = client.get(
            "/api/recordings/status", query_string={"client_id": client_id}
        )
        assert status.status_code == 200
        assert status.get_json()["recording"] is True
        retry_cancel = client.post(
            "/api/recordings/cancel",
            json={"client_id": client_id},
            headers={"X-CSRF-Token": csrf},
        )
        assert retry_cancel.status_code == 200
        assert recorder.active is False
        retry_cancel.close()
        assert not transition_lock.locked()


@pytest.mark.parametrize(
    ("terminal_action", "error_factory", "expected_status", "expected_error"),
    (
        (
            "stop",
            lambda: PresentationRecordingError("stop failed safely"),
            409,
            "stop failed safely",
        ),
        (
            "cancel",
            lambda: CameraSessionError("cancel failed safely"),
            409,
            "cancel failed safely",
        ),
        (
            "stop",
            lambda: AudioCaptureError("audio failed safely"),
            503,
            "audio failed safely",
        ),
        (
            "stop",
            lambda: PresentationStorageError("storage failed safely"),
            500,
            "could not be opened safely",
        ),
        (
            "cancel",
            lambda: ValueError("validation failed safely"),
            422,
            "validation failed safely",
        ),
        (
            "stop",
            lambda: RuntimeError("unexpected terminal failure"),
            500,
            "transition failed safely",
        ),
    ),
    ids=(
        "stop-recording-error",
        "cancel-camera-error",
        "stop-audio-error",
        "stop-storage-error",
        "cancel-value-error",
        "stop-runtime-error",
    ),
)
def test_terminal_error_cookie_is_emitted_before_replacement_start(
    tmp_path: Path,
    terminal_action: str,
    error_factory,
    expected_status: int,
    expected_error: str,
):
    recorder = FailingTerminalRecorder(terminal_action, error_factory())
    app = create_app(
        store=PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys()),
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=FakeVideoAnalyzer(),
        testing=True,
    )
    with app.test_client() as client:
        _authorize(client, app)
        csrf = client.get("/api/bootstrap").get_json()["csrf_token"]
        profile = client.post(
            "/api/profiles",
            json={"name": "Terminal error transition"},
            headers={"X-CSRF-Token": csrf},
        ).get_json()["profile"]
        first_client_id = "recording-client-terminal-first"
        started = client.post(
            "/api/recordings/start",
            json={"profile_id": profile["id"], "client_id": first_client_id},
            headers={"X-CSRF-Token": csrf},
        )
        assert started.status_code == 200

        failed = client.post(
            f"/api/recordings/{terminal_action}",
            json={"client_id": first_client_id},
            headers={"X-CSRF-Token": csrf},
        )
        assert failed.status_code == expected_status
        assert expected_error in failed.get_json()["error"]
        assert failed.headers.getlist("Set-Cookie")
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        assert transition_lock.locked()
        assert recorder.active is False

        replacement_client_id = "recording-client-terminal-replacement"
        blocked = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": replacement_client_id,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert blocked.status_code == 409
        assert recorder.active is False

        failed.close()
        assert not transition_lock.locked()
        restarted = client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": replacement_client_id,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert restarted.status_code == 200
        replacement_status = client.get(
            "/api/recordings/status",
            query_string={"client_id": replacement_client_id},
        )
        assert replacement_status.status_code == 200
        assert replacement_status.get_json()["recording"] is True


def test_new_recording_cannot_overlap_blocking_uploaded_video_analysis(
    tmp_path: Path,
):
    recorder = ExclusiveFakeRecorder()
    analyzer = BlockingPlaybackAnalyzer()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    app = create_app(
        store=store,
        llm=FakeLLM(),
        recorder=recorder,
        video_analyzer=analyzer,
        testing=True,
    )
    upload_client = app.test_client()
    start_client = app.test_client()
    _authorize(upload_client, app)
    _authorize(start_client, app)
    upload_csrf = upload_client.get("/api/bootstrap").get_json()["csrf_token"]
    start_csrf = start_client.get("/api/bootstrap").get_json()["csrf_token"]
    profile = upload_client.post(
        "/api/profiles",
        json={"name": "Upload transition lock"},
        headers={"X-CSRF-Token": upload_csrf},
    ).get_json()["profile"]
    upload_result: dict[str, object] = {}

    def run_upload():
        upload_result["response"] = upload_client.post(
            "/api/videos/analyze",
            data={
                "profile_id": profile["id"],
                "video": (BytesIO(b"local-video"), "practice.webm"),
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": upload_csrf},
        )

    upload_thread = threading.Thread(target=run_upload)
    upload_thread.start()
    try:
        assert analyzer.entered.wait(timeout=2.0)
        transition_lock = app.extensions["presentation_recording_transition_lock"]
        assert transition_lock.locked()
        blocked = start_client.post(
            "/api/recordings/start",
            json={
                "profile_id": profile["id"],
                "client_id": "recording-client-after-upload",
            },
            headers={"X-CSRF-Token": start_csrf},
        )
        assert blocked.status_code == 409
        assert recorder.active is False
    finally:
        analyzer.release.set()
        upload_thread.join(timeout=4.0)

    assert not upload_thread.is_alive()
    uploaded = upload_result["response"]
    assert uploaded.status_code == 201
    assert transition_lock.locked()
    still_blocked = start_client.post(
        "/api/recordings/start",
        json={
            "profile_id": profile["id"],
            "client_id": "recording-client-after-upload",
        },
        headers={"X-CSRF-Token": start_csrf},
    )
    assert still_blocked.status_code == 409

    uploaded.close()
    assert not transition_lock.locked()
    started = start_client.post(
        "/api/recordings/start",
        json={
            "profile_id": profile["id"],
            "client_id": "recording-client-after-upload",
        },
        headers={"X-CSRF-Token": start_csrf},
    )
    assert started.status_code == 200
    assert recorder.active is True


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
    assert document["coaching_cues"]["limitations"]["infers_reading"] is False


def test_stored_feedback_is_invalidated_when_claim_no_longer_matches_metrics(
    tmp_path: Path,
):
    source = tmp_path / "practice.webm"
    source.write_bytes(b"local-video")
    presentation = FakeVideoAnalyzer().analyze(source)
    forged_feedback = {
        "status": "ready",
        "source": "local_llm_verified_general_practice",
        "strengths": [{
            "text": "Across this session, camera contact was 12 %.",
            "metric": "eye_contact_percent",
            "value": 12,
            "unit": "%",
            "timestamp_seconds": 31,
        }],
        "improvements": [],
        "insufficient_data": [],
    }
    archive = PresentationArchive(
        "a" * 32,
        "Tanav",
        (StoredPresentation(presentation, compute_metrics(presentation), forged_feedback),),
        {},
    )

    document = _serialize_archive(archive)[0]

    assert document["metrics"]["aggregate"]["eye_contact_percent"] == 100.0
    assert document["feedback"]["status"] == "calibration_required"
    assert document["feedback"]["strengths"] == []
    assert document["coaching_cues"]["counts"]["verified_coaching"] == 0


def test_stored_feedback_is_invalidated_when_claim_role_changes(tmp_path: Path):
    source = tmp_path / "practice.webm"
    source.write_bytes(b"local-video")
    presentation = FakeVideoAnalyzer().analyze(source)
    metrics = compute_metrics(presentation)
    feedback = generate_feedback(
        prepare_feedback_metrics(metrics, {"ready": False}), FakeLLM()
    ).to_dict()
    moved = feedback["strengths"].pop(0)
    feedback["improvements"].append(moved)
    archive = PresentationArchive(
        "a" * 32,
        "Tanav",
        (StoredPresentation(presentation, metrics, feedback),),
        {},
    )

    document = _serialize_archive(archive)[0]

    assert document["feedback"]["status"] == "calibration_required"
    assert document["coaching_cues"]["counts"]["verified_coaching"] == 0


def test_test_lab_exposes_only_allowlisted_integrity_checked_local_clips_and_reports(
    tmp_path: Path, monkeypatch,
):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    media = tmp_path / "media"
    reports = tmp_path / "reports"
    media.mkdir()
    reports.mkdir()
    clip_bytes = b"webm-evidence"
    (media / "tarun-speaking-cc0.webm").write_bytes(clip_bytes)
    monkeypatch.setitem(
        TEST_MEDIA_BY_ID["tarun-short-distance"],
        "sha256",
        hashlib.sha256(clip_bytes).hexdigest(),
    )
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
        _authorize(client, app)
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
        assert clip.data == clip_bytes
        (media / "tarun-speaking-cc0.webm").write_bytes(b"tampered-clip")
        assert client.get("/api/test-media/tarun-short-distance/video").status_code == 404
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
