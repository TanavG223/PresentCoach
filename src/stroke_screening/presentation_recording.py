"""Synchronized local camera, microphone, vision, and Whisper orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets
import threading
import time
from typing import Callable

from .presentation_audio import (
    AudioCaptureError,
    LocalAudioRecorder,
    WhisperCppTranscriber,
    audio_signal_metrics,
)
from .presentation_core import PresentationSession, analyze_session
from .presentation_vision import PresentationVisionAnalyzer
from .presentation_camera import CameraSessionError, PresentationCameraService


TARGET_ANALYSIS_FPS = 15.0


class PresentationRecordingError(RuntimeError):
    """Raised when a recording owner or state is invalid."""


class PresentationRecordingService:
    """Own exactly one bounded local presentation recording at a time."""

    def __init__(
        self,
        *,
        camera: PresentationCameraService,
        model_path: Path,
        transcriber: WhisperCppTranscriber,
        audio_factory: Callable[[], LocalAudioRecorder] = LocalAudioRecorder,
        vision_factory: Callable[[Path], PresentationVisionAnalyzer] = PresentationVisionAnalyzer,
    ) -> None:
        self.camera = camera
        self.model_path = model_path
        self.transcriber = transcriber
        self.audio_factory = audio_factory
        self.vision_factory = vision_factory
        self._lock = threading.Lock()
        self._owner: str | None = None
        self._camera_owner: str | None = None
        self._audio: LocalAudioRecorder | None = None
        self._vision: PresentationVisionAnalyzer | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_monotonic = 0.0
        self._started_time = ""
        self._session_kind = "practice"
        self._note: str | None = None
        self._error: str | None = None

    def _assert_owner(self, owner: str) -> None:
        if not owner or not secrets.compare_digest(owner, self._owner or ""):
            raise PresentationRecordingError("The presentation recording expired")

    def start(self, *, session_kind: str = "practice", note: str | None = None) -> str:
        if session_kind not in {"baseline", "repeat", "practice"}:
            raise ValueError("session_kind must be baseline, repeat, or practice")
        with self._lock:
            if self._owner is not None:
                raise PresentationRecordingError("A presentation recording is already active")
        camera_owner = self.camera.start()
        audio = self.audio_factory()
        vision: PresentationVisionAnalyzer | None = None
        try:
            audio.start()
            vision = self.vision_factory(self.model_path)
        except Exception:
            audio.cancel()
            self.camera.stop(camera_owner)
            if vision is not None:
                vision.close()
            raise
        owner = secrets.token_urlsafe(24)
        started = time.monotonic()
        with self._lock:
            self._owner = owner
            self._camera_owner = camera_owner
            self._audio = audio
            self._vision = vision
            self._started_monotonic = started
            self._started_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self._session_kind = session_kind
            self._note = (note or "").strip()[:1000] or None
            self._error = None
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._analyze_loop,
                args=(camera_owner, vision, started),
                daemon=True,
                name="presentcoach-vision",
            )
            self._thread.start()
        return owner

    def _analyze_loop(
        self,
        camera_owner: str,
        vision: PresentationVisionAnalyzer,
        started: float,
    ) -> None:
        generation = -1
        interval = 1.0 / TARGET_ANALYSIS_FPS
        next_tick = started
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_tick:
                    self._stop.wait(next_tick - now)
                    continue
                frame, generation = self.camera.next_frame(
                    camera_owner, generation, timeout=3.0
                )
                vision.process(frame, time.monotonic() - started)
                next_tick = max(next_tick + interval, time.monotonic())
        except CameraSessionError as error:
            if not self._stop.is_set():
                with self._lock:
                    self._error = str(error)
        except Exception as error:
            with self._lock:
                self._error = f"Vision analysis stopped: {type(error).__name__}"

    def status(self, owner: str) -> dict[str, object]:
        with self._lock:
            self._assert_owner(owner)
            elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            vision = self._vision
            error = self._error
        frames = vision.frame_count() if vision is not None else 0
        return {
            "recording": True,
            "elapsed_seconds": round(elapsed, 1),
            "analyzed_frames": frames,
            "analysis_fps": round(frames / max(elapsed, 0.1), 1),
            "face_detected": bool(vision and vision.overlay_points()),
            "error": error,
        }

    def preview_jpeg(self, owner: str) -> bytes:
        with self._lock:
            self._assert_owner(owner)
            camera_owner = self._camera_owner
            vision = self._vision
        if camera_owner is None or vision is None:
            raise PresentationRecordingError("The local preview is not ready")
        return self.camera.preview_jpeg(camera_owner, vision.overlay_points())

    def stop(self, owner: str) -> PresentationSession:
        with self._lock:
            self._assert_owner(owner)
            self._stop.set()
            thread = self._thread
            audio = self._audio
            vision = self._vision
            camera_owner = self._camera_owner
            started = self._started_monotonic
            started_time = self._started_time
            session_kind = self._session_kind
            note = self._note
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        try:
            duration = max(0.0, time.monotonic() - started)
            samples = audio.stop() if audio is not None else None
            if camera_owner is not None:
                try:
                    self.camera.stop(camera_owner)
                except CameraSessionError:
                    pass
            if samples is None:
                raise AudioCaptureError("No microphone samples were recorded")
            signal = audio_signal_metrics(samples)
            trustworthy_signal = (
                signal["waveform_rms"] >= 0.003
                and signal["clipping_fraction"] < 0.01
            )
            words = self.transcriber.transcribe(samples) if trustworthy_signal else ()
            vision_samples = vision.finish(duration) if vision is not None else ()
            return analyze_session(
                vision_samples,
                {
                    "words": words,
                    "duration_seconds": duration,
                    "waveform_rms": signal["waveform_rms"] if trustworthy_signal else 0.0,
                },
                start_time=started_time,
                session_kind=session_kind,
                note=note,
            )
        finally:
            if camera_owner is not None:
                try:
                    self.camera.stop(camera_owner)
                except CameraSessionError:
                    pass
            if vision is not None:
                vision.close()
            self._clear()

    def cancel(self, owner: str | None = None) -> None:
        with self._lock:
            if owner is not None:
                self._assert_owner(owner)
            if self._owner is None:
                return
            self._stop.set()
            thread = self._thread
            audio = self._audio
            vision = self._vision
            camera_owner = self._camera_owner
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        if audio is not None:
            audio.cancel()
        if camera_owner is not None:
            try:
                self.camera.stop(camera_owner)
            except CameraSessionError:
                pass
        if vision is not None:
            vision.close()
        self._clear()

    def _clear(self) -> None:
        with self._lock:
            self._owner = None
            self._camera_owner = None
            self._audio = None
            self._vision = None
            self._thread = None
            self._started_monotonic = 0.0
            self._started_time = ""
            self._note = None
            self._error = None
            self._stop.clear()
