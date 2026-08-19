"""Synchronized local camera, microphone, vision, and Whisper orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable
import wave

import numpy as np

from .presentation_audio import (
    AudioCaptureError,
    LocalAudioRecorder,
    MAX_RECORDING_SECONDS,
    TranscriptionError,
    WhisperCppTranscriber,
    audio_signal_metrics,
)
from .presentation_core import PresentationSession, analyze_session
from .presentation_vision import PresentationVisionAnalyzer
from .presentation_camera import CameraSessionError, PresentationCameraService


TARGET_ANALYSIS_FPS = 15.0
MAX_LIVE_VIDEO_ONLY_BYTES = 480 * 1024 * 1024
MAX_LIVE_MEDIA_DURATION_DRIFT_SECONDS = 0.5
DEFAULT_RECORDING_LEASE_SECONDS = 10.0
DEFAULT_MAX_RECORDING_DURATION_SECONDS = 28 * 60.0
DEFAULT_RECORDING_LIMIT_GRACE_SECONDS = 2 * 60.0


class PresentationRecordingError(RuntimeError):
    """Raised when a recording owner or state is invalid."""


@dataclass
class RecordingMedia:
    """Private, short-lived browser-compatible media awaiting encryption."""

    path: Path
    mime_type: str = "video/mp4"
    source: str = "recording"
    _temporary: tempfile.TemporaryDirectory[str] | None = field(
        repr=False, default=None
    )

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


@dataclass
class RecordedPresentation:
    """Completed measurements plus an optional local media lease."""

    session: PresentationSession
    media: RecordingMedia | None = None
    media_error: str | None = None

    def close(self) -> None:
        if self.media is not None:
            self.media.close()


class LocalRecordingMediaCapture:
    """Encode camera/audio with a one-frame, nonblocking producer buffer.

    Vision only replaces the latest frame. A dedicated encoder thread samples
    that slot at a fixed cadence, so a slow encoder can neither block landmark
    analysis nor accumulate a catch-up queue.
    """

    def __init__(
        self,
        *,
        ffmpeg_executable: Path | None = None,
        ffprobe_executable: Path | None = None,
        fps: float = TARGET_ANALYSIS_FPS,
    ) -> None:
        executable = ffmpeg_executable or (
            Path(found) if (found := shutil.which("ffmpeg")) else None
        )
        if executable is None or not executable.is_file():
            raise PresentationRecordingError(
                "The local ffmpeg runtime required for video recording is missing"
            )
        if not math.isfinite(fps) or not 1.0 <= fps <= 60.0:
            raise ValueError("Recording FPS must be between 1 and 60")
        self.ffmpeg_executable = executable.resolve()
        probe = ffprobe_executable or (
            Path(found) if (found := shutil.which("ffprobe")) else None
        )
        self.ffprobe_executable = probe.resolve() if probe is not None else None
        self.fps = fps
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._video_path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._shape: tuple[int, int] | None = None
        self._written_frames = 0
        self._last_frame: np.ndarray | None = None
        self._error: str | None = None
        self._state_lock = threading.Lock()
        self._writer_stop = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._submitted_frames = 0
        self._latest_sequence = 0
        self._consumed_sequence = 0
        self._overwritten_frames = 0
        self._accepting_frames = True

    @property
    def error(self) -> str | None:
        with self._state_lock:
            return self._error

    def stats(self) -> dict[str, int | float | str | None]:
        with self._state_lock:
            return {
                "submitted_frames": self._submitted_frames,
                "encoded_frames": self._written_frames,
                "overwritten_frames": self._overwritten_frames,
                "buffered_frames": int(
                    self._latest_sequence > self._consumed_sequence
                ),
                "buffer_capacity": 1,
                "latest_timestamp_seconds": getattr(
                    self, "_latest_timestamp_seconds", None
                ),
                "error": self._error,
            }

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            if self._error is None:
                self._error = message

    def _start_encoder(self, height: int, width: int) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="presentcoach-live-")
        os.chmod(temporary.name, 0o700)
        video_path = Path(temporary.name) / "video-only.mp4"
        command = [
            str(self.ffmpeg_executable), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-filter_threads", "2",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(self.fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-pix_fmt", "yuv420p", "-threads:v", "2",
            "-movflags", "+faststart",
            "-fs", str(MAX_LIVE_VIDEO_ONLY_BYTES), "-y", str(video_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            temporary.cleanup()
            raise PresentationRecordingError(
                "The local video encoder could not start"
            ) from error
        self._temporary = temporary
        self._video_path = video_path
        self._process = process
        self._shape = (height, width)

    def _write_frame(self, frame: np.ndarray) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise PresentationRecordingError("The local video encoder stopped unexpectedly")
        try:
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as error:
            raise PresentationRecordingError(
                "The local video encoder stopped unexpectedly"
            ) from error
        with self._state_lock:
            self._written_frames += 1

    def _writer_loop(self) -> None:
        try:
            with self._state_lock:
                shape = self._shape
            if shape is None:
                raise PresentationRecordingError("The camera preview is not ready")
            if self._writer_stop.is_set():
                return
            self._start_encoder(*shape)
            if self._writer_stop.is_set():
                self._terminate()
                return
            interval = 1.0 / self.fps
            next_tick = time.monotonic()
            while not self._writer_stop.is_set():
                with self._state_lock:
                    frame = self._last_frame
                    sequence = self._latest_sequence
                if frame is None:
                    self._writer_stop.wait(min(interval, 0.01))
                    continue
                self._write_frame(frame)
                with self._state_lock:
                    self._consumed_sequence = max(
                        self._consumed_sequence, sequence
                    )
                next_tick = max(next_tick + interval, time.monotonic())
                self._writer_stop.wait(max(0.0, next_tick - time.monotonic()))
        except PresentationRecordingError as error:
            self._set_error(str(error))
            self._writer_stop.set()
            self._terminate()
        except Exception as error:
            self._set_error(f"Local video encoder failed: {type(error).__name__}")
            self._writer_stop.set()
            self._terminate()

    def write(self, frame: np.ndarray, timestamp_seconds: float) -> None:
        """Replace the bounded latest-frame slot and return without encoder I/O."""

        if self.error is not None:
            return
        try:
            candidate = np.asarray(frame)
            if (
                candidate.ndim != 3
                or candidate.shape[2] != 3
                or min(candidate.shape[:2]) < 1
                or not math.isfinite(timestamp_seconds)
            ):
                raise PresentationRecordingError(
                    "The camera returned an invalid video frame"
                )
            owned = np.ascontiguousarray(candidate, dtype=np.uint8).copy()
            thread_to_start: threading.Thread | None = None
            with self._state_lock:
                if not self._accepting_frames:
                    return
                shape = tuple(owned.shape[:2])
                if self._shape is not None and self._shape != shape:
                    raise PresentationRecordingError(
                        "The camera resolution changed during the recording"
                    )
                if self._latest_sequence > self._consumed_sequence:
                    self._overwritten_frames += 1
                self._shape = shape
                self._last_frame = owned
                self._submitted_frames += 1
                self._latest_sequence += 1
                self._latest_timestamp_seconds = max(0.0, timestamp_seconds)
                if self._writer_thread is None:
                    thread_to_start = threading.Thread(
                        target=self._writer_loop,
                        daemon=True,
                        name="presentcoach-media-encoder",
                    )
                    self._writer_thread = thread_to_start
            if thread_to_start is not None:
                try:
                    thread_to_start.start()
                except RuntimeError as error:
                    with self._state_lock:
                        if self._writer_thread is thread_to_start:
                            self._writer_thread = None
                    raise PresentationRecordingError(
                        "The local video encoder thread could not start"
                    ) from error
        except PresentationRecordingError as error:
            self._set_error(str(error))
            self._writer_stop.set()

    def stop_writing(self) -> None:
        """Reject new frames and stop the encoder cadence at the UI stop time."""

        with self._state_lock:
            self._accepting_frames = False
        self._writer_stop.set()

    def _probe_duration(self, path: Path) -> float:
        if self.ffprobe_executable is None or not self.ffprobe_executable.is_file():
            raise PresentationRecordingError(
                "The local ffprobe runtime required to verify video timing is missing"
            )
        try:
            result = subprocess.run(
                [
                    str(self.ffprobe_executable), "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
                start_new_session=True,
            )
            duration = float(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            raise PresentationRecordingError(
                "The local recorded video duration could not be verified"
            ) from error
        if result.returncode != 0 or not math.isfinite(duration) or duration <= 0:
            raise PresentationRecordingError(
                "The local recorded video duration could not be verified"
            )
        return duration

    @staticmethod
    def _write_audio(path: Path, samples: np.ndarray) -> None:
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        pcm = np.rint(clipped * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(pcm.tobytes())

    def finish(
        self, samples: np.ndarray, duration_seconds: float
    ) -> RecordingMedia | None:
        self.stop_writing()
        thread = self._writer_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        if thread is not None and thread.is_alive():
            self._set_error("The local video encoder did not keep up with the recording")
            self._terminate()
            thread.join(timeout=2)
        if self.error is None and self._process is None:
            with self._state_lock:
                shape = self._shape
                last_frame = self._last_frame
            if shape is not None and last_frame is not None:
                try:
                    self._start_encoder(*shape)
                    self._write_frame(last_frame)
                except PresentationRecordingError as error:
                    self._set_error(str(error))
        if self.error is not None or self._process is None or self._temporary is None:
            self.cancel()
            return None
        process = self._process
        try:
            expected_frames = max(1, round(max(0.0, duration_seconds) * self.fps))
            with self._state_lock:
                written_frames = self._written_frames
                last_frame = self._last_frame
            if written_frames == 0 and last_frame is not None:
                # A sub-frame recording may stop before the writer's first
                # scheduled tick. Finalization permits exactly one frame; it
                # never performs timestamp-based catch-up work.
                self._write_frame(last_frame)
                written_frames = 1
            if written_frames < max(1, math.floor(expected_frames * 0.8)):
                raise PresentationRecordingError(
                    "The local video encoder could not maintain a usable frame rate"
                )
            if process.stdin is not None:
                process.stdin.close()
            try:
                return_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait(timeout=5)
                raise PresentationRecordingError(
                    "The local video encoder did not finish"
                ) from error
            if return_code != 0 or self._video_path is None or not self._video_path.is_file():
                raise PresentationRecordingError("The local video encoder rejected the recording")
            final_path = Path(self._temporary.name) / "session.mp4"
            encoded_duration = written_frames / self.fps
            timestamp_scale = max(0.000001, duration_seconds) / encoded_duration
            audio = np.asarray(samples, dtype=np.float32).reshape(-1)
            if audio.size:
                audio_path = Path(self._temporary.name) / "audio.wav"
                self._write_audio(audio_path, audio)
                command = [
                    str(self.ffmpeg_executable), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-itsscale", f"{timestamp_scale:.9f}",
                    "-i", str(self._video_path), "-i", str(audio_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "96k", "-shortest",
                    "-t", f"{duration_seconds:.6f}",
                    "-movflags", "+faststart", "-y", str(final_path),
                ]
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=120,
                    start_new_session=True,
                )
                if result.returncode != 0 or not final_path.is_file():
                    raise PresentationRecordingError(
                        "The local recorder could not attach microphone audio"
                    )
            else:
                command = [
                    str(self.ffmpeg_executable), "-nostdin", "-hide_banner",
                    "-loglevel", "error",
                    "-itsscale", f"{timestamp_scale:.9f}",
                    "-i", str(self._video_path), "-map", "0:v:0",
                    "-c:v", "copy", "-an", "-t", f"{duration_seconds:.6f}",
                    "-movflags", "+faststart", "-y", str(final_path),
                ]
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=120,
                    start_new_session=True,
                )
                if result.returncode != 0 or not final_path.is_file():
                    raise PresentationRecordingError(
                        "The local recorder could not preserve video timing"
                    )
            verified_duration = self._probe_duration(final_path)
            duration_tolerance = max(
                0.15,
                min(
                    MAX_LIVE_MEDIA_DURATION_DRIFT_SECONDS,
                    max(0.0, duration_seconds) * 0.005,
                ),
            )
            if abs(verified_duration - duration_seconds) > duration_tolerance:
                raise PresentationRecordingError(
                    "The local recorded video timing did not match the session"
                )
            temporary = self._temporary
            self._temporary = None
            self._process = None
            self._video_path = None
            self._last_frame = None
            return RecordingMedia(path=final_path, _temporary=temporary)
        except (OSError, subprocess.TimeoutExpired, PresentationRecordingError) as error:
            self._set_error(str(error))
            self.cancel()
            return None

    def _terminate(self) -> None:
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._process = None

    def cancel(self) -> None:
        self.stop_writing()
        thread = self._writer_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        if thread is not None and thread.is_alive():
            self._terminate()
            thread.join(timeout=2)
        self._terminate()
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = None
        self._video_path = None
        self._last_frame = None
        self._writer_thread = None


class PresentationRecordingService:
    """Own exactly one bounded, browser-leased recording at a time.

    The browser renews the lease through ``status`` or authenticated preview
    frames. A dedicated lifecycle thread cancels every owned resource when
    those heartbeats disappear or the hard recording-duration limit is
    reached. Both paths deliberately cancel rather than finalize: an
    unattended recording must not run Whisper or persist a partial session.
    """

    def __init__(
        self,
        *,
        camera: PresentationCameraService,
        model_path: Path,
        transcriber: WhisperCppTranscriber,
        audio_factory: Callable[[], LocalAudioRecorder] = LocalAudioRecorder,
        vision_factory: Callable[[Path], PresentationVisionAnalyzer] = PresentationVisionAnalyzer,
        media_factory: Callable[[], LocalRecordingMediaCapture] = LocalRecordingMediaCapture,
        lease_timeout_seconds: float = DEFAULT_RECORDING_LEASE_SECONDS,
        max_duration_seconds: float = DEFAULT_MAX_RECORDING_DURATION_SECONDS,
        max_duration_grace_seconds: float = DEFAULT_RECORDING_LIMIT_GRACE_SECONDS,
        analysis_join_timeout_seconds: float = 3.0,
    ) -> None:
        if (
            not math.isfinite(lease_timeout_seconds)
            or lease_timeout_seconds <= 0
        ):
            raise ValueError("Recording lease timeout must be positive")
        if (
            not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0
        ):
            raise ValueError("Maximum recording duration must be positive")
        if (
            not math.isfinite(analysis_join_timeout_seconds)
            or analysis_join_timeout_seconds <= 0
        ):
            raise ValueError("Analysis join timeout must be positive")
        if (
            not math.isfinite(max_duration_grace_seconds)
            or max_duration_grace_seconds <= 0
        ):
            raise ValueError("Recording duration grace must be positive")
        if max_duration_seconds + max_duration_grace_seconds > MAX_RECORDING_SECONDS:
            raise ValueError(
                "The recording soft limit and grace must not exceed the audio cap"
            )
        self.camera = camera
        self.model_path = model_path
        self.transcriber = transcriber
        self.audio_factory = audio_factory
        self.vision_factory = vision_factory
        self.media_factory = media_factory
        self.lease_timeout_seconds = float(lease_timeout_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self.max_duration_grace_seconds = float(max_duration_grace_seconds)
        self.analysis_join_timeout_seconds = float(analysis_join_timeout_seconds)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._terminal_lock = threading.Lock()
        self._owner: str | None = None
        self._camera_owner: str | None = None
        self._audio: LocalAudioRecorder | None = None
        self._vision: PresentationVisionAnalyzer | None = None
        self._media_capture: LocalRecordingMediaCapture | None = None
        self._thread: threading.Thread | None = None
        self._lifecycle_thread: threading.Thread | None = None
        self._run_stop = threading.Event()
        self._terminal_action: str | None = None
        self._analysis_exited_owner: str | None = None
        self._started_monotonic = 0.0
        self._lease_deadline = 0.0
        self._started_time = ""
        self._session_kind = "practice"
        self._note: str | None = None
        self._error: str | None = None

    def _assert_owner(self, owner: str) -> None:
        if not owner or not secrets.compare_digest(owner, self._owner or ""):
            raise PresentationRecordingError("The presentation recording expired")

    def is_active(self) -> bool:
        with self._lock:
            return self._owner is not None

    def start(self, *, session_kind: str = "practice", note: str | None = None) -> str:
        if session_kind not in {"baseline", "repeat", "practice"}:
            raise ValueError("session_kind must be baseline, repeat, or practice")
        with self._lock:
            if self._owner is not None:
                raise PresentationRecordingError("A presentation recording is already active")
        camera_owner = self.camera.start()
        audio: LocalAudioRecorder | None = None
        vision: PresentationVisionAnalyzer | None = None
        media_capture: LocalRecordingMediaCapture | None = None
        try:
            audio = self.audio_factory()
            vision = self.vision_factory(self.model_path)
            media_capture = self.media_factory()
            # Model and encoder construction can be comparatively slow. Start
            # the microphone only after both are ready so audio, video, vision,
            # and Whisper timestamps share the same session epoch.
            audio.start()
        except Exception:
            if audio is not None:
                try:
                    audio.cancel()
                except Exception:
                    pass
            try:
                self.camera.stop(camera_owner)
            except Exception:
                pass
            if vision is not None:
                try:
                    vision.close()
                except Exception:
                    pass
            if media_capture is not None:
                try:
                    media_capture.cancel()
                except Exception:
                    pass
            raise
        owner = secrets.token_urlsafe(24)
        started = time.monotonic()
        run_stop = threading.Event()
        with self._condition:
            self._owner = owner
            self._camera_owner = camera_owner
            self._audio = audio
            self._vision = vision
            self._media_capture = media_capture
            self._started_monotonic = started
            self._lease_deadline = min(
                started
                + self.max_duration_seconds
                + self.max_duration_grace_seconds,
                started + self.lease_timeout_seconds,
            )
            self._started_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self._session_kind = session_kind
            self._note = (note or "").strip()[:1000] or None
            self._error = None
            self._run_stop = run_stop
            self._terminal_action = None
            self._analysis_exited_owner = None
            self._thread = threading.Thread(
                target=self._analyze_loop,
                args=(owner, run_stop, camera_owner, vision, media_capture, started),
                daemon=True,
                name="presentcoach-vision",
            )
            self._lifecycle_thread = threading.Thread(
                target=self._lifecycle_loop,
                args=(owner, run_stop),
                daemon=True,
                name="presentcoach-recording-lifecycle",
            )
            analysis_thread = self._thread
            lifecycle_thread = self._lifecycle_thread
        try:
            analysis_thread.start()
            lifecycle_thread.start()
        except RuntimeError as error:
            self.cancel(owner)
            raise PresentationRecordingError(
                "The local recording threads could not start"
            ) from error
        return owner

    def _lifecycle_loop(
        self, owner: str, run_stop: threading.Event
    ) -> None:
        """Expire one immutable owner without allowing a later session match."""

        while True:
            with self._condition:
                if self._owner != owner or run_stop.is_set():
                    return
                now = time.monotonic()
                hard_deadline = (
                    self._started_monotonic
                    + self.max_duration_seconds
                    + self.max_duration_grace_seconds
                )
                deadline = min(self._lease_deadline, hard_deadline)
                remaining = deadline - now
                if remaining > 0:
                    # A status heartbeat notifies the condition so this wait
                    # immediately adopts the extended lease deadline.
                    self._condition.wait(timeout=remaining)
                    continue
            try:
                self.cancel(owner)
            except PresentationRecordingError:
                # A user stop may win the terminal-operation race after the
                # deadline check.  In that case its cleanup owns the resources.
                pass
            return

    def _analyze_loop(
        self,
        owner: str,
        run_stop: threading.Event,
        camera_owner: str,
        vision: PresentationVisionAnalyzer,
        media_capture: LocalRecordingMediaCapture,
        started: float,
    ) -> None:
        generation = -1
        interval = 1.0 / TARGET_ANALYSIS_FPS
        next_tick = started
        try:
            while not run_stop.is_set():
                now = time.monotonic()
                if now < next_tick:
                    run_stop.wait(next_tick - now)
                    continue
                frame, generation = self.camera.next_frame(
                    camera_owner, generation, timeout=3.0
                )
                if run_stop.is_set():
                    break
                timestamp = time.monotonic() - started
                media_capture.write(frame, timestamp)
                if run_stop.is_set():
                    break
                vision.process(frame, timestamp)
                next_tick = max(next_tick + interval, time.monotonic())
        except CameraSessionError as error:
            if not run_stop.is_set():
                with self._condition:
                    if self._owner != owner:
                        return
                    self._error = str(error)
        except Exception as error:
            with self._condition:
                if self._owner == owner and not run_stop.is_set():
                    self._error = f"Vision analysis stopped: {type(error).__name__}"
        finally:
            self._analysis_worker_finished(owner, vision, media_capture)

    def _analysis_worker_finished(
        self,
        owner: str,
        vision: PresentationVisionAnalyzer,
        media_capture: LocalRecordingMediaCapture,
    ) -> None:
        with self._condition:
            if self._owner != owner:
                return
            self._analysis_exited_owner = owner
            should_cleanup = self._terminal_action == "cancel"
            self._condition.notify_all()
        if should_cleanup:
            self._cleanup_processors(vision, media_capture)
            self._clear(owner)

    def status(self, owner: str) -> dict[str, object]:
        with self._condition:
            now = time.monotonic()
            elapsed, expired = self._renew_lease_locked(owner, now)
            vision = self._vision
            media_capture = self._media_capture
            error = self._error
        if expired:
            try:
                self.cancel(owner)
            except PresentationRecordingError:
                pass
            raise PresentationRecordingError("The presentation recording expired")
        frames = vision.frame_count() if vision is not None else 0
        media_stats = media_capture.stats() if media_capture is not None else {}
        return {
            "recording": True,
            "elapsed_seconds": round(elapsed, 1),
            "maximum_duration_seconds": round(self.max_duration_seconds, 1),
            "maximum_duration_reached": elapsed >= self.max_duration_seconds,
            "analyzed_frames": frames,
            "analysis_fps": round(frames / max(elapsed, 0.1), 1),
            "face_detected": bool(vision and vision.overlay_points()),
            "media_recording": bool(media_capture and media_capture.error is None),
            "media_encoded_frames": media_stats.get("encoded_frames", 0),
            "media_buffered_frames": media_stats.get("buffered_frames", 0),
            "media_buffer_capacity": media_stats.get("buffer_capacity", 1),
            "error": error or (media_capture.error if media_capture else None),
        }

    def preview_jpeg(self, owner: str) -> bytes:
        with self._condition:
            _elapsed, expired = self._renew_lease_locked(
                owner, time.monotonic()
            )
            camera_owner = self._camera_owner
            vision = self._vision
        if expired:
            try:
                self.cancel(owner)
            except PresentationRecordingError:
                pass
            raise PresentationRecordingError("The presentation recording expired")
        if camera_owner is None or vision is None:
            raise PresentationRecordingError("The local preview is not ready")
        return self.camera.preview_jpeg(camera_owner, vision.overlay_points())

    def _renew_lease_locked(
        self, owner: str, now: float
    ) -> tuple[float, bool]:
        """Renew one authenticated browser heartbeat; caller owns the lock."""

        self._assert_owner(owner)
        elapsed = max(0.0, now - self._started_monotonic)
        hard_deadline = (
            self._started_monotonic
            + self.max_duration_seconds
            + self.max_duration_grace_seconds
        )
        expired = (
            self._run_stop.is_set()
            or now >= self._lease_deadline
            or now >= hard_deadline
        )
        if not expired:
            self._lease_deadline = min(
                hard_deadline, now + self.lease_timeout_seconds
            )
            self._condition.notify_all()
        return elapsed, expired

    def stop(self, owner: str) -> RecordedPresentation:
        with self._terminal_lock:
            with self._condition:
                self._assert_owner(owner)
                self._terminal_action = "stop"
                run_stop = self._run_stop
                run_stop.set()
                self._condition.notify_all()
                thread = self._thread
                audio = self._audio
                vision = self._vision
                media_capture = self._media_capture
                camera_owner = self._camera_owner
                started = self._started_monotonic
                stopped = time.monotonic()
                started_time = self._started_time
                session_kind = self._session_kind
                note = self._note
            duration = max(0.0, stopped - started)
            samples: np.ndarray | None = None
            audio_error: AudioCaptureError | None = None
            if audio is not None:
                try:
                    samples = audio.stop()
                except AudioCaptureError as error:
                    audio_error = error
                except Exception:
                    # A third-party/native adapter must not strand the camera
                    # or erase an otherwise valid video-only recording.
                    audio_error = AudioCaptureError(
                        "The microphone did not stop cleanly"
                    )
            self._stop_media_writer(media_capture)
            if camera_owner is not None:
                try:
                    self.camera.stop(camera_owner)
                except CameraSessionError:
                    pass
            self._join_thread(
                thread, timeout=self.analysis_join_timeout_seconds
            )
            if thread is not None and thread.is_alive():
                direct_cleanup = self._arm_deferred_cancel(owner, thread)
                if direct_cleanup:
                    self._cleanup_processors(vision, media_capture)
                    self._clear(owner)
                raise PresentationRecordingError(
                    "The local vision analysis did not stop cleanly; "
                    "the partial recording was cancelled"
                )
            recorded_media: RecordingMedia | None = None
            try:
                if samples is None:
                    samples = np.zeros(0, dtype=np.float32)
                recorded_media = (
                    media_capture.finish(samples, duration)
                    if media_capture is not None else None
                )
                signal = audio_signal_metrics(samples)
                trustworthy_signal = (
                    signal["waveform_rms"] >= 0.003
                    and signal["clipping_fraction"] < 0.01
                )
                words = ()
                completion_warning: str | None = None
                if audio_error is not None:
                    completion_warning = (
                        "The replay and vision measurements were saved, but "
                        "microphone audio could not be finalized."
                        if recorded_media is not None
                        else
                        "The vision measurements were saved, but microphone "
                        "audio could not be finalized."
                    )
                elif trustworthy_signal:
                    try:
                        words = self.transcriber.transcribe(samples)
                    except TranscriptionError:
                        completion_warning = (
                            "The replay and vision measurements were saved, but "
                            "local speech transcription failed."
                        )
                vision_samples = vision.finish(duration) if vision is not None else ()
                presentation = analyze_session(
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
                return RecordedPresentation(
                    session=presentation,
                    media=recorded_media,
                    media_error=(
                        completion_warning
                        or (
                            media_capture.error
                            if recorded_media is None and media_capture is not None
                            else None
                        )
                    ),
                )
            except Exception:
                if recorded_media is not None:
                    recorded_media.close()
                raise
            finally:
                self._cleanup_processors(vision, media_capture)
                self._clear(owner)

    def cancel(self, owner: str | None = None) -> None:
        with self._terminal_lock:
            with self._condition:
                if owner is not None:
                    self._assert_owner(owner)
                active_owner = self._owner
                if active_owner is None:
                    return
                self._terminal_action = "cancel_pending"
                run_stop = self._run_stop
                run_stop.set()
                self._condition.notify_all()
                thread = self._thread
                audio = self._audio
                vision = self._vision
                media_capture = self._media_capture
                camera_owner = self._camera_owner
            if audio is not None:
                try:
                    audio.cancel()
                except Exception:
                    pass
            self._stop_media_writer(media_capture)
            if camera_owner is not None:
                try:
                    self.camera.stop(camera_owner)
                except CameraSessionError:
                    pass
            self._join_thread(
                thread, timeout=self.analysis_join_timeout_seconds
            )
            direct_cleanup = self._arm_deferred_cancel(active_owner, thread)
            if direct_cleanup:
                self._cleanup_processors(vision, media_capture)
                self._clear(active_owner)

    def _arm_deferred_cancel(
        self, owner: str, thread: threading.Thread | None
    ) -> bool:
        """Transfer processor cleanup to a live worker, or claim it directly."""

        with self._condition:
            if self._owner != owner:
                return False
            self._terminal_action = "cancel"
            direct_cleanup = (
                self._analysis_exited_owner == owner
                or thread is None
                or thread.ident is None
            )
            self._condition.notify_all()
            return direct_cleanup

    @staticmethod
    def _stop_media_writer(
        media_capture: LocalRecordingMediaCapture | None,
    ) -> None:
        stop_writing = getattr(media_capture, "stop_writing", None)
        if callable(stop_writing):
            try:
                stop_writing()
            except Exception:
                pass

    @staticmethod
    def _cleanup_processors(
        vision: PresentationVisionAnalyzer | None,
        media_capture: LocalRecordingMediaCapture | None,
    ) -> None:
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        if media_capture is not None:
            try:
                media_capture.cancel()
            except Exception:
                pass

    @staticmethod
    def _join_thread(thread: threading.Thread | None, *, timeout: float) -> None:
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.ident is not None
        ):
            thread.join(timeout=timeout)

    def _clear(self, owner: str) -> None:
        with self._condition:
            if self._owner != owner:
                return
            self._owner = None
            self._camera_owner = None
            self._audio = None
            self._vision = None
            self._media_capture = None
            self._thread = None
            self._lifecycle_thread = None
            self._run_stop = threading.Event()
            self._terminal_action = None
            self._analysis_exited_owner = None
            self._started_monotonic = 0.0
            self._lease_deadline = 0.0
            self._started_time = ""
            self._note = None
            self._error = None
            self._condition.notify_all()
