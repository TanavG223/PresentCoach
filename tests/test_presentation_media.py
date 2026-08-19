import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import time

import numpy as np
import pytest

import stroke_screening.presentation_recording as presentation_recording_module
from stroke_screening.presentation_core import (
    TranscriptWord,
    VisionSample,
    analyze_session,
    compute_metrics,
)
from stroke_screening.presentation_recording import (
    LocalRecordingMediaCapture,
    PresentationRecordingError,
    PresentationRecordingService,
    RecordingMedia,
)
from stroke_screening.presentation_audio import AudioCaptureError, TranscriptionError
from stroke_screening.presentation_camera import CameraSessionError
from stroke_screening.presentation_store import (
    MEDIA_CHUNK_BYTES,
    MEDIA_TAG_BYTES,
    MacOSKeychainKeyStore,
    PresentationStorageError,
    PresentationStore,
    SessionMediaNotFoundError,
    StoredPresentation,
)


class MemoryKeys:
    def __init__(self):
        self.values = {}

    def save(self, profile, key):
        self.values[profile] = key

    def load(self, profile):
        return self.values[profile]

    def delete(self, profile):
        self.values.pop(profile, None)


def test_macos_keychain_key_is_cached_for_range_playback(monkeypatch):
    profile_id = "a" * 32
    key = b"k" * 32
    calls = []

    def fake_run(self, arguments, *, secret_input=None):
        calls.append(tuple(arguments))
        if arguments[0] == "find-generic-password":
            import base64
            return subprocess.CompletedProcess(
                arguments, 0, stdout=base64.b64encode(key).decode("ascii") + "\n", stderr=""
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(MacOSKeychainKeyStore, "_run", fake_run)
    key_store = MacOSKeychainKeyStore()

    assert key_store.load(profile_id) == key
    assert key_store.load(profile_id) == key
    assert sum(call[0] == "find-generic-password" for call in calls) == 1

    key_store.delete(profile_id)
    assert key_store.load(profile_id) == key
    assert sum(call[0] == "find-generic-password" for call in calls) == 2


def test_live_audio_starts_only_after_vision_and_video_are_ready(tmp_path: Path):
    events = []

    class Camera:
        def start(self):
            events.append("camera.start")
            return "camera-owner"

        def next_frame(self, owner, generation, timeout):
            assert owner == "camera-owner"
            time.sleep(0.005)
            return np.zeros((16, 16, 3), dtype=np.uint8), generation + 1

        def stop(self, owner):
            events.append("camera.stop")

    class Audio:
        def start(self):
            events.append("audio.start")

        def cancel(self):
            events.append("audio.cancel")

    class Vision:
        def __init__(self, _path):
            events.append("vision.ready")

        def process(self, _frame, _timestamp):
            events.append("vision.frame")

        def close(self):
            events.append("vision.close")

        def frame_count(self):
            return events.count("vision.frame")

        def overlay_points(self):
            return ()

    class Media:
        error = None

        def __init__(self):
            events.append("media.ready")

        def write(self, _frame, _timestamp):
            events.append("media.frame")

        def cancel(self):
            events.append("media.cancel")

    service = PresentationRecordingService(
        camera=Camera(),
        model_path=tmp_path / "face.task",
        transcriber=object(),
        audio_factory=Audio,
        vision_factory=Vision,
        media_factory=Media,
    )

    owner = service.start()
    deadline = time.monotonic() + 1.0
    while "vision.frame" not in events and time.monotonic() < deadline:
        time.sleep(0.005)
    service.cancel(owner)

    assert events.index("vision.ready") < events.index("audio.start")
    assert events.index("media.ready") < events.index("audio.start")
    assert events.index("audio.start") < events.index("vision.frame")


def test_audio_factory_failure_rolls_back_started_camera(tmp_path: Path):
    events: list[str] = []

    class Camera:
        def start(self):
            events.append("camera.start")
            return "camera-owner"

        def stop(self, owner):
            assert owner == "camera-owner"
            events.append("camera.stop")

    def fail_audio_factory():
        raise RuntimeError("audio construction failed")

    service = PresentationRecordingService(
        camera=Camera(),
        model_path=tmp_path / "face.task",
        transcriber=object(),
        audio_factory=fail_audio_factory,
    )

    with pytest.raises(RuntimeError, match="audio construction failed"):
        service.start()

    assert events == ["camera.start", "camera.stop"]
    assert service.is_active() is False


def test_start_failure_attempts_every_resource_rollback(tmp_path: Path):
    events: list[str] = []

    class Camera:
        def start(self):
            events.append("camera.start")
            return "camera-owner"

        def stop(self, owner):
            assert owner == "camera-owner"
            events.append("camera.stop")

    class Audio:
        def start(self):
            events.append("audio.start")
            raise AudioCaptureError("microphone failed")

        def cancel(self):
            events.append("audio.cancel")
            raise RuntimeError("audio cancel also failed")

    class Vision:
        def close(self):
            events.append("vision.close")
            raise RuntimeError("vision close also failed")

    class Media:
        def cancel(self):
            events.append("media.cancel")

    service = PresentationRecordingService(
        camera=Camera(),
        model_path=tmp_path / "face.task",
        transcriber=object(),
        audio_factory=Audio,
        vision_factory=lambda _path: Vision(),
        media_factory=Media,
    )

    with pytest.raises(AudioCaptureError, match="microphone failed"):
        service.start()

    assert events == [
        "camera.start",
        "audio.start",
        "audio.cancel",
        "camera.stop",
        "vision.close",
        "media.cancel",
    ]
    assert service.is_active() is False


def _lifecycle_recording_service(
    tmp_path: Path,
    *,
    lease_timeout_seconds: float,
    max_duration_seconds: float,
    max_duration_grace_seconds: float = 30.0,
    analysis_join_timeout_seconds: float = 3.0,
    camera_stop_delay_seconds: float = 0.0,
    audio_hard_cap_seconds: float | None = None,
):
    events: list[str] = []
    cleanup_reached = threading.Event()
    capture_workspace = tmp_path / "live-capture"
    capture_workspace.mkdir()
    capture_path = capture_workspace / "video-only.mp4"
    capture_path.write_bytes(b"private-partial-video")

    class Camera:
        def __init__(self):
            self.stopped = threading.Event()

        def start(self):
            events.append("camera.start")
            return "camera-owner"

        def next_frame(self, owner, generation, timeout):
            assert owner == "camera-owner"
            if self.stopped.wait(min(timeout, 0.005)):
                raise CameraSessionError("camera stopped")
            return np.zeros((16, 16, 3), dtype=np.uint8), generation + 1

        def stop(self, owner):
            assert owner == "camera-owner"
            if not self.stopped.is_set():
                events.append("camera.stop")
                self.stopped.set()
                if camera_stop_delay_seconds:
                    time.sleep(camera_stop_delay_seconds)

        def preview_jpeg(self, owner, overlay_points):
            assert owner == "camera-owner"
            assert overlay_points == ()
            assert not self.stopped.is_set()
            events.append("camera.preview")
            return b"jpeg"

    class Audio:
        def start(self):
            events.append("audio.start")
            self.started = time.monotonic()

        def stop(self):
            if (
                audio_hard_cap_seconds is not None
                and time.monotonic() - self.started >= audio_hard_cap_seconds
            ):
                raise RuntimeError("fake audio hard cap crossed")
            events.append("audio.stop")
            return np.zeros(160, dtype=np.float32)

        def cancel(self):
            events.append("audio.cancel")

    class Vision:
        def process(self, _frame, _timestamp):
            events.append("vision.frame")

        def close(self):
            events.append("vision.close")

        def finish(self, _duration):
            events.append("vision.finish")
            return ()

        def frame_count(self):
            return events.count("vision.frame")

        def overlay_points(self):
            return ()

    class Media:
        error = None

        def write(self, _frame, _timestamp):
            events.append("media.frame")

        def stats(self):
            return {
                "encoded_frames": events.count("media.frame"),
                "buffered_frames": 0,
                "buffer_capacity": 1,
            }

        def finish(self, _samples, _duration):
            events.append("media.finish")
            return None

        def cancel(self):
            events.append("media.cancel")
            shutil.rmtree(capture_workspace, ignore_errors=True)
            cleanup_reached.set()

    camera = Camera()
    service = PresentationRecordingService(
        camera=camera,
        model_path=tmp_path / "face.task",
        transcriber=object(),
        audio_factory=Audio,
        vision_factory=lambda _path: Vision(),
        media_factory=Media,
        lease_timeout_seconds=lease_timeout_seconds,
        max_duration_seconds=max_duration_seconds,
        max_duration_grace_seconds=max_duration_grace_seconds,
        analysis_join_timeout_seconds=analysis_join_timeout_seconds,
    )
    return service, events, cleanup_reached, capture_workspace


def _assert_deadline_cancelled_entire_recording(
    service: PresentationRecordingService,
    events: list[str],
    cleanup_reached: threading.Event,
    capture_workspace: Path,
) -> None:
    assert cleanup_reached.wait(timeout=2.0), "recording teardown did not finish"
    deadline = time.monotonic() + 1.0
    while service.is_active() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert service.is_active() is False
    assert events.count("camera.stop") == 1
    assert events.count("audio.cancel") == 1
    assert events.count("vision.close") == 1
    assert events.count("media.cancel") == 1
    assert "audio.stop" not in events
    assert "media.finish" not in events
    assert not capture_workspace.exists()


def test_abandoned_recording_lease_cancels_all_resources_after_heartbeats_end(
    tmp_path: Path,
):
    service, events, cleanup_reached, capture_workspace = (
        _lifecycle_recording_service(
            tmp_path,
            lease_timeout_seconds=0.50,
            max_duration_seconds=2.0,
        )
    )
    owner = service.start()

    # Both the JSON status poll and authenticated MJPEG frame activity renew
    # the same lease.  Survive beyond each earlier deadline, then abandon both.
    assert not cleanup_reached.wait(timeout=0.15)
    assert service.status(owner)["recording"] is True
    assert not cleanup_reached.wait(timeout=0.15)
    assert service.preview_jpeg(owner) == b"jpeg"
    assert not cleanup_reached.wait(timeout=0.40)

    _assert_deadline_cancelled_entire_recording(
        service, events, cleanup_reached, capture_workspace
    )
    with pytest.raises(PresentationRecordingError, match="expired"):
        service.status(owner)


def test_soft_maximum_duration_remains_retrievable_and_finalizable(
    tmp_path: Path,
):
    service, events, cleanup_reached, capture_workspace = (
        _lifecycle_recording_service(
            tmp_path,
            lease_timeout_seconds=2.0,
            max_duration_seconds=0.08,
            max_duration_grace_seconds=0.50,
            audio_hard_cap_seconds=0.30,
        )
    )
    owner = service.start()
    deadline = time.monotonic() + 1.0
    status = service.status(owner)
    while not status["maximum_duration_reached"] and time.monotonic() < deadline:
        time.sleep(0.01)
        status = service.status(owner)

    assert status["recording"] is True
    assert status["maximum_duration_seconds"] == 0.1
    assert status["maximum_duration_reached"] is True
    assert service.is_active() is True

    completed = service.stop(owner)
    completed.close()
    assert service.is_active() is False
    assert events.count("audio.stop") == 1
    assert events.count("audio.cancel") == 0
    assert events.count("vision.finish") == 1
    assert events.count("media.finish") == 1
    assert events.count("vision.close") == 1
    assert events.count("media.cancel") == 1
    assert not capture_workspace.exists()


def test_default_soft_limit_leaves_polling_margin_before_audio_hard_cap():
    assert (
        presentation_recording_module.DEFAULT_MAX_RECORDING_DURATION_SECONDS
        == 28 * 60
    )
    assert (
        presentation_recording_module.DEFAULT_RECORDING_LIMIT_GRACE_SECONDS
        == 2 * 60
    )
    assert (
        presentation_recording_module.DEFAULT_MAX_RECORDING_DURATION_SECONDS
        + presentation_recording_module.DEFAULT_RECORDING_LIMIT_GRACE_SECONDS
        == presentation_recording_module.MAX_RECORDING_SECONDS
    )
    with pytest.raises(ValueError, match="must not exceed the audio cap"):
        PresentationRecordingService(
            camera=object(),
            model_path=Path("unused.task"),
            transcriber=object(),
            max_duration_seconds=1795.0,
            max_duration_grace_seconds=10.0,
        )


def test_duration_grace_expiry_cancels_entire_recording_with_live_lease(
    tmp_path: Path,
):
    service, events, cleanup_reached, capture_workspace = (
        _lifecycle_recording_service(
            tmp_path,
            lease_timeout_seconds=2.0,
            max_duration_seconds=0.05,
            max_duration_grace_seconds=0.08,
        )
    )
    owner = service.start()
    assert service.status(owner)["recording"] is True

    _assert_deadline_cancelled_entire_recording(
        service, events, cleanup_reached, capture_workspace
    )


def test_stop_freezes_duration_and_microphone_before_slow_camera_teardown(
    tmp_path: Path,
):
    service, events, _cleanup_reached, _capture_workspace = (
        _lifecycle_recording_service(
            tmp_path,
            lease_timeout_seconds=2.0,
            max_duration_seconds=2.0,
            camera_stop_delay_seconds=0.20,
        )
    )
    owner = service.start()
    time.sleep(0.03)

    stop_started = time.monotonic()
    completed = service.stop(owner)
    stop_wall_seconds = time.monotonic() - stop_started

    assert events.index("audio.stop") < events.index("camera.stop")
    assert stop_wall_seconds >= 0.18
    assert completed.session.duration_seconds < 0.15
    completed.close()


def test_audio_stop_failure_preserves_video_only_replay_and_vision_session(
    tmp_path: Path,
):
    events: list[str] = []
    finish_samples: list[np.ndarray] = []

    class Camera:
        def __init__(self):
            self.stopped = threading.Event()

        def start(self):
            return "camera-owner"

        def next_frame(self, owner, generation, timeout):
            assert owner == "camera-owner"
            if self.stopped.wait(min(timeout, 0.005)):
                raise CameraSessionError("camera stopped")
            return np.zeros((8, 8, 3), dtype=np.uint8), generation + 1

        def stop(self, owner):
            assert owner == "camera-owner"
            events.append("camera.stop")
            self.stopped.set()

    class Audio:
        def start(self):
            events.append("audio.start")

        def stop(self):
            events.append("audio.stop")
            raise AudioCaptureError("fake microphone stop failure")

        def cancel(self):
            events.append("audio.cancel")

    class Vision:
        def process(self, _frame, _timestamp):
            return None

        def finish(self, _duration):
            events.append("vision.finish")
            return ()

        def close(self):
            events.append("vision.close")

        def frame_count(self):
            return 1

        def overlay_points(self):
            return ()

    class Media:
        error = None

        def write(self, _frame, _timestamp):
            return None

        def stop_writing(self):
            events.append("media.stop_writing")

        def finish(self, samples, _duration):
            events.append("media.finish")
            finish_samples.append(np.asarray(samples).copy())
            temporary = tempfile.TemporaryDirectory(
                prefix="presentcoach-audio-stop-failure-", dir=tmp_path
            )
            path = Path(temporary.name) / "session.mp4"
            path.write_bytes(b"video-only-replay")
            return RecordingMedia(path=path, _temporary=temporary)

        def cancel(self):
            events.append("media.cancel")

    class UnexpectedTranscriber:
        def transcribe(self, _samples):
            raise AssertionError("empty failed audio must not be transcribed")

    service = PresentationRecordingService(
        camera=Camera(),
        model_path=tmp_path / "face.task",
        transcriber=UnexpectedTranscriber(),
        audio_factory=Audio,
        vision_factory=lambda _path: Vision(),
        media_factory=Media,
    )
    owner = service.start()
    time.sleep(0.02)

    completed = service.stop(owner)

    assert events.count("audio.stop") == 1
    assert events.count("media.finish") == 1
    assert len(finish_samples) == 1
    assert finish_samples[0].size == 0
    assert completed.session.transcript == ()
    assert completed.session.quality_flags["audio_clear"] == "bad"
    assert completed.media is not None
    assert completed.media.path.read_bytes() == b"video-only-replay"
    assert completed.media_error is not None
    assert "microphone audio could not be finalized" in completed.media_error
    assert service.is_active() is False
    replay_directory = completed.media.path.parent
    completed.close()
    assert not replay_directory.exists()


def test_transcription_failure_preserves_vision_session_and_replay(
    tmp_path: Path,
):
    transcribe_called = threading.Event()

    class Camera:
        def __init__(self):
            self.stopped = threading.Event()

        def start(self):
            return "camera-owner"

        def next_frame(self, owner, generation, timeout):
            assert owner == "camera-owner"
            if self.stopped.wait(min(timeout, 0.005)):
                raise CameraSessionError("camera stopped")
            return np.zeros((8, 8, 3), dtype=np.uint8), generation + 1

        def stop(self, owner):
            assert owner == "camera-owner"
            self.stopped.set()

    class Audio:
        def start(self):
            return None

        def stop(self):
            return np.full(1600, 0.1, dtype=np.float32)

        def cancel(self):
            return None

    class Vision:
        def process(self, _frame, _timestamp):
            return None

        def finish(self, _duration):
            return ()

        def close(self):
            return None

        def frame_count(self):
            return 1

        def overlay_points(self):
            return ()

    class Media:
        error = None

        def write(self, _frame, _timestamp):
            return None

        def stop_writing(self):
            return None

        def finish(self, _samples, _duration):
            temporary = tempfile.TemporaryDirectory(
                prefix="presentcoach-transcription-failure-", dir=tmp_path
            )
            path = Path(temporary.name) / "session.mp4"
            path.write_bytes(b"retained-replay")
            return RecordingMedia(path=path, _temporary=temporary)

        def cancel(self):
            return None

    class FailingTranscriber:
        def transcribe(self, _samples):
            transcribe_called.set()
            raise TranscriptionError("whisper failed")

    service = PresentationRecordingService(
        camera=Camera(),
        model_path=tmp_path / "face.task",
        transcriber=FailingTranscriber(),
        audio_factory=Audio,
        vision_factory=lambda _path: Vision(),
        media_factory=Media,
    )
    owner = service.start()
    time.sleep(0.02)

    completed = service.stop(owner)

    assert transcribe_called.is_set()
    assert completed.session.transcript == ()
    assert completed.session.quality_flags["audio_clear"] == "bad"
    assert completed.media is not None
    assert completed.media.path.read_bytes() == b"retained-replay"
    assert completed.media_error is not None
    assert "transcription failed" in completed.media_error
    replay_directory = completed.media.path.parent
    completed.close()
    assert not replay_directory.exists()


@pytest.mark.parametrize("terminal_action", ("cancel", "stop"))
def test_terminal_action_defers_close_until_blocked_vision_worker_exits(
    tmp_path: Path, terminal_action: str,
):
    events: list[str] = []
    process_entered = threading.Event()
    release_process = threading.Event()
    processors_closed = threading.Event()

    class Camera:
        def start(self):
            return "camera-owner"

        def next_frame(self, owner, generation, timeout):
            assert owner == "camera-owner"
            return np.zeros((8, 8, 3), dtype=np.uint8), generation + 1

        def stop(self, owner):
            assert owner == "camera-owner"
            events.append("camera.stop")

    class Audio:
        def start(self):
            return None

        def cancel(self):
            events.append("audio.cancel")

        def stop(self):
            events.append("audio.stop")
            return np.zeros(160, dtype=np.float32)

    class Vision:
        def __init__(self):
            self.processing = False
            self.closed_during_process = False
            self.process_count = 0

        def process(self, _frame, _timestamp):
            self.processing = True
            self.process_count += 1
            process_entered.set()
            assert release_process.wait(timeout=2.0)
            self.processing = False

        def close(self):
            self.closed_during_process = self.processing
            events.append("vision.close")

        def frame_count(self):
            return self.process_count

        def overlay_points(self):
            return ()

    class Media:
        error = None

        def write(self, _frame, _timestamp):
            events.append("media.write")

        def cancel(self):
            events.append("media.cancel")
            processors_closed.set()

    vision = Vision()
    service = PresentationRecordingService(
        camera=Camera(),
        model_path=tmp_path / "face.task",
        transcriber=object(),
        audio_factory=Audio,
        vision_factory=lambda _path: vision,
        media_factory=Media,
        lease_timeout_seconds=2.0,
        max_duration_seconds=2.0,
        analysis_join_timeout_seconds=0.03,
    )
    owner = service.start()
    assert process_entered.wait(timeout=1.0)

    if terminal_action == "cancel":
        service.cancel(owner)
        audio_event = "audio.cancel"
    else:
        with pytest.raises(PresentationRecordingError, match="partial recording"):
            service.stop(owner)
        audio_event = "audio.stop"

    assert events.index(audio_event) < events.index("camera.stop")
    assert service.is_active() is True
    assert "vision.close" not in events
    assert "media.cancel" not in events
    with pytest.raises(PresentationRecordingError, match="already active"):
        service.start()

    release_process.set()
    assert processors_closed.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while service.is_active() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert service.is_active() is False
    assert vision.closed_during_process is False
    assert vision.process_count == 1
    assert events.count("vision.close") == 1
    assert events.count("media.cancel") == 1


def presentation():
    samples = tuple(
        VisionSample(
            timestamp_seconds=float(second), frame_count=15,
            face_detected=True, eye_contact=True, gaze_horizontal=0.5,
            gaze_vertical=0.5, yaw_degrees=0.0, pitch_degrees=0.0,
            roll_degrees=0.0, face_center_x=0.5, face_center_y=0.5,
            mouth_activity=0.1, brow_activity=0.1, expression_change=0.01,
            inference_ms=1.0, detected_frame_count=15, contact_frame_count=15,
            contact_eligible_frame_count=15,
        )
        for second in range(31)
    )
    words = tuple(
        TranscriptWord("word", float(second), float(second) + 0.3, 0.9)
        for second in range(31)
    )
    return analyze_session(
        samples,
        {"words": words, "duration_seconds": 31, "waveform_rms": 0.1},
        session_kind="imported",
    )


def stored_profile(tmp_path: Path):
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    profile = store.create_profile("Media owner")
    session = presentation()
    store.append(
        profile["id"], StoredPresentation(session, compute_metrics(session), None)
    )
    return store, profile["id"], session.session_id


def test_profile_mutations_are_serialized_without_lost_sessions(
    tmp_path: Path, monkeypatch,
):
    store = PresentationStore(data_dir=tmp_path / "data", key_store=MemoryKeys())
    profile_id = store.create_profile("Concurrent presenter")["id"]
    first = presentation()
    second = presentation()
    store.append(
        profile_id, StoredPresentation(first, compute_metrics(first), {"status": "pending"})
    )

    original_write = PresentationStore._write
    append_entered = threading.Event()
    replace_entered = threading.Event()
    release_append = threading.Event()
    errors = []

    def coordinated_write(self, archive, key):
        if threading.current_thread().name == "append-second":
            append_entered.set()
            if not release_append.wait(2):
                raise AssertionError("Timed out releasing the append write")
        elif threading.current_thread().name == "replace-first":
            replace_entered.set()
        return original_write(self, archive, key)

    monkeypatch.setattr(PresentationStore, "_write", coordinated_write)

    def append_second():
        try:
            store.append(
                profile_id,
                StoredPresentation(second, compute_metrics(second), {"status": "pending"}),
            )
        except BaseException as error:
            errors.append(error)

    def replace_first():
        try:
            store.replace_feedback(
                profile_id, first.session_id, {"status": "ready", "marker": "updated"}
            )
        except BaseException as error:
            errors.append(error)

    append_thread = threading.Thread(target=append_second, name="append-second")
    replace_thread = threading.Thread(target=replace_first, name="replace-first")
    append_thread.start()
    assert append_entered.wait(1)
    replace_thread.start()
    try:
        assert not replace_entered.wait(0.15)
    finally:
        release_append.set()
    append_thread.join(2)
    replace_thread.join(2)

    assert not append_thread.is_alive()
    assert not replace_thread.is_alive()
    assert errors == []
    archive = store.load_profile(profile_id)
    assert [item.session.session_id for item in archive.sessions] == [
        first.session_id, second.session_id,
    ]
    assert archive.sessions[0].feedback == {"status": "ready", "marker": "updated"}


def test_session_media_is_encrypted_bound_and_short_lived_when_decrypted(tmp_path: Path):
    store, profile_id, session_id = stored_profile(tmp_path)
    plaintext = b"private-session-video-0123456789"
    source = tmp_path / "session.mp4"
    source.write_bytes(plaintext)

    metadata = store.put_session_media(
        profile_id, session_id, source, mime_type="video/mp4", source="upload"
    )
    encrypted = next((tmp_path / "data").glob("*.presentcoach-media"))

    assert metadata.plaintext_bytes == len(plaintext)
    assert plaintext not in encrypted.read_bytes()
    assert stat.S_IMODE(encrypted.stat().st_mode) == 0o600
    assert store.session_media(profile_id, session_id).source == "upload"

    lease = store.decrypt_session_media(profile_id, session_id)
    decrypted_path = lease.path
    decrypted_directory = lease.path.parent
    assert decrypted_path.read_bytes() == plaintext
    assert stat.S_IMODE(decrypted_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(decrypted_directory.stat().st_mode) == 0o700
    lease.close()
    assert not decrypted_directory.exists()

    with pytest.raises(SessionMediaNotFoundError):
        store.session_media(profile_id, "f" * 32)


def test_session_media_tampering_fails_authentication_without_plaintext_output(
    tmp_path: Path,
):
    store, profile_id, session_id = stored_profile(tmp_path)
    source = tmp_path / "session.webm"
    source.write_bytes(b"authenticated-video")
    store.put_session_media(
        profile_id, session_id, source, mime_type="video/webm", source="upload"
    )
    encrypted = next((tmp_path / "data").glob("*.presentcoach-media"))
    payload = bytearray(encrypted.read_bytes())
    payload[-1] ^= 0x01
    encrypted.write_bytes(payload)

    with pytest.raises(PresentationStorageError, match="failed authentication"):
        store.decrypt_session_media(profile_id, session_id)


def test_session_media_cannot_be_rebound_to_a_different_profile_or_session(
    tmp_path: Path,
):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    first_profile = store.create_profile("First")["id"]
    first_session = presentation()
    store.append(
        first_profile,
        StoredPresentation(first_session, compute_metrics(first_session), None),
    )
    source = tmp_path / "session.mp4"
    source.write_bytes(b"profile-bound-video")
    store.put_session_media(
        first_profile,
        first_session.session_id,
        source,
        mime_type="video/mp4",
        source="upload",
    )

    second_profile = store.create_profile("Second")["id"]
    second_session = presentation()
    store.append(
        second_profile,
        StoredPresentation(second_session, compute_metrics(second_session), None),
    )
    encrypted = (
        tmp_path / "data"
        / f"{first_profile}-{first_session.session_id}.presentcoach-media"
    )
    rebound = (
        tmp_path / "data"
        / f"{second_profile}-{second_session.session_id}.presentcoach-media"
    )
    shutil.copyfile(encrypted, rebound)

    with pytest.raises(PresentationStorageError, match="identity is invalid"):
        store.open_session_media(second_profile, second_session.session_id)


def test_session_media_rejects_a_symlink_source(tmp_path: Path):
    store, profile_id, session_id = stored_profile(tmp_path)
    real_source = tmp_path / "real.mp4"
    real_source.write_bytes(b"private-video")
    linked_source = tmp_path / "linked.mp4"
    linked_source.symlink_to(real_source)

    with pytest.raises(PresentationStorageError, match="could not be opened"):
        store.put_session_media(
            profile_id,
            session_id,
            linked_source,
            mime_type="video/mp4",
            source="upload",
        )


def test_session_media_range_reader_authenticates_only_touched_chunks(
    tmp_path: Path,
):
    store, profile_id, session_id = stored_profile(tmp_path)
    plaintext = (
        b"a" * MEDIA_CHUNK_BYTES
        + b"b" * MEDIA_CHUNK_BYTES
        + b"c" * 4096
    )
    source = tmp_path / "large-session.mp4"
    source.write_bytes(plaintext)
    store.put_session_media(
        profile_id, session_id, source, mime_type="video/mp4", source="upload"
    )

    reader = store.open_session_media(profile_id, session_id)
    start = MEDIA_CHUNK_BYTES + 123
    end = start + 4096
    selected = b"".join(reader.iter_range(start, end))
    stats = reader.stats()
    reader.close()

    assert selected == plaintext[start:end]
    assert stats["authenticated_chunks"] == 1
    assert stats["encrypted_bytes_read"] == MEDIA_CHUNK_BYTES + MEDIA_TAG_BYTES

    crossing = store.open_session_media(profile_id, session_id)
    start = MEDIA_CHUNK_BYTES - 8
    end = MEDIA_CHUNK_BYTES + 8
    selected = b"".join(crossing.iter_range(start, end))
    crossing_stats = crossing.stats()
    crossing.close()
    assert selected == plaintext[start:end]
    assert crossing_stats["authenticated_chunks"] == 2

    encrypted = next((tmp_path / "data").glob("*.presentcoach-media"))
    payload = bytearray(encrypted.read_bytes())
    payload[-1] ^= 0x01
    encrypted.write_bytes(payload)
    unaffected = store.open_session_media(profile_id, session_id)
    assert b"".join(unaffected.iter_range(0, 32)) == plaintext[:32]
    unaffected.close()
    corrupted_tail = store.open_session_media(profile_id, session_id)
    with pytest.raises(PresentationStorageError, match="failed authentication"):
        b"".join(
            corrupted_tail.iter_range(len(plaintext) - 32, len(plaintext))
        )
    corrupted_tail.close()


def test_live_capture_returns_a_cleanup_owned_mp4_lease(
    tmp_path: Path, monkeypatch,
):
    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "sys.stdin.buffer.read()\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'local-mp4')\n",
        encoding="utf-8",
    )
    os.chmod(fake_ffmpeg, 0o700)
    fake_ffprobe = tmp_path / "fake-ffprobe"
    fake_ffprobe.write_text(
        "#!/usr/bin/env python3\nprint('0.2')\n", encoding="utf-8"
    )
    os.chmod(fake_ffprobe, 0o700)
    commands: list[list[str]] = []
    real_popen = subprocess.Popen

    def capturing_popen(command, *args, **kwargs):
        commands.append(list(command))
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(
        presentation_recording_module.subprocess,
        "Popen",
        capturing_popen,
    )
    capture = LocalRecordingMediaCapture(
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=fake_ffprobe,
        fps=10.0,
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    capture.write(frame, 0.0)
    capture.write(frame + 1, 0.1)

    media = capture.finish(np.zeros(1600, dtype=np.float32), 0.2)

    assert media is not None, capture.error
    assert media.path.read_bytes() == b"local-mp4"
    command = next(item for item in commands if "rawvideo" in item)
    assert command[command.index("-filter_threads") + 1] == "2"
    assert command[command.index("-threads:v") + 1] == "2"
    directory = media.path.parent
    media.close()
    assert not directory.exists()


def test_live_capture_stretches_controlled_frame_lag_to_session_timeline(
    tmp_path: Path,
):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("real FFmpeg and ffprobe runtimes are required")

    capture = LocalRecordingMediaCapture(
        ffmpeg_executable=Path(ffmpeg),
        ffprobe_executable=Path(ffprobe),
        fps=10.0,
    )
    encoded_eight = threading.Event()
    original_write = capture._write_frame

    def stop_after_eight(frame):
        original_write(frame)
        if capture.stats()["encoded_frames"] >= 8:
            capture._writer_stop.set()
            encoded_eight.set()

    capture._write_frame = stop_after_eight
    capture.write(np.zeros((90, 160, 3), dtype=np.uint8), 0.0)
    assert encoded_eight.wait(timeout=2.0)
    assert capture.stats()["encoded_frames"] == 8

    media = capture.finish(np.zeros(16_000, dtype=np.float32), 1.0)

    assert media is not None, capture.error
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(media.path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    playback_duration = float(result.stdout.strip())
    assert abs(playback_duration - 1.0) <= 0.15
    assert playback_duration - (8 / 10.0) >= 0.05
    media.close()


def test_live_capture_rejects_replay_when_verified_timing_drifts(
    tmp_path: Path,
):
    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "sys.stdin.buffer.read()\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'local-mp4')\n",
        encoding="utf-8",
    )
    fake_ffprobe = tmp_path / "fake-ffprobe"
    fake_ffprobe.write_text(
        "#!/usr/bin/env python3\nprint('9.0')\n", encoding="utf-8"
    )
    os.chmod(fake_ffmpeg, 0o700)
    os.chmod(fake_ffprobe, 0o700)
    capture = LocalRecordingMediaCapture(
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=fake_ffprobe,
        fps=10.0,
    )
    capture.write(np.zeros((16, 16, 3), dtype=np.uint8), 0.0)

    media = capture.finish(np.zeros(1600, dtype=np.float32), 0.1)

    assert media is None
    assert capture.error == "The local recorded video timing did not match the session"
    assert capture._temporary is None


def test_live_capture_large_timestamp_jump_never_creates_a_catch_up_burst(
    tmp_path: Path,
):
    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "sys.stdin.buffer.read()\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'local-mp4')\n",
        encoding="utf-8",
    )
    os.chmod(fake_ffmpeg, 0o700)
    capture = LocalRecordingMediaCapture(ffmpeg_executable=fake_ffmpeg, fps=15.0)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    started = time.perf_counter()
    capture.write(frame, 0.0)
    capture.write(frame, 10_000_000.0)
    producer_elapsed = time.perf_counter() - started
    time.sleep(0.08)
    stats = capture.stats()

    assert producer_elapsed < 0.25
    assert stats["submitted_frames"] == 2
    assert stats["buffer_capacity"] == 1
    assert stats["buffered_frames"] in {0, 1}
    assert int(stats["encoded_frames"]) <= 3
    capture.cancel()


def test_stalled_encoder_cannot_block_or_grow_the_analysis_producer(
    tmp_path: Path,
):
    stalled_ffmpeg = tmp_path / "stalled-ffmpeg"
    stalled_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    os.chmod(stalled_ffmpeg, 0o700)
    capture = LocalRecordingMediaCapture(
        ffmpeg_executable=stalled_ffmpeg, fps=15.0
    )
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    capture.write(frame, 0.0)
    time.sleep(0.1)

    started = time.perf_counter()
    for index in range(20):
        capture.write(frame, 1_000_000.0 + index)
    producer_elapsed = time.perf_counter() - started
    stats = capture.stats()

    assert producer_elapsed < 0.5
    assert stats["submitted_frames"] == 21
    assert stats["buffer_capacity"] == 1
    assert stats["buffered_frames"] == 1
    cancel_started = time.perf_counter()
    capture.cancel()
    assert time.perf_counter() - cancel_started < 4.0
