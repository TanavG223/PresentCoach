"""Bounded, offline analysis for user-supplied presentation videos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import threading
import time
from typing import Callable

import numpy as np

from .presentation_audio import SAMPLE_RATE, WhisperCppTranscriber, audio_signal_metrics
from .presentation_core import PresentationSession, TranscriptWord, analyze_session
from .presentation_vision import PresentationVisionAnalyzer


MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_UPLOAD_SECONDS = 30 * 60
TARGET_IMPORT_FPS = 15.0
MAX_IMPORT_SOURCE_PIXELS = 3840 * 2160
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".webm"})
PLAYBACK_MIME_TYPE = "video/mp4"
MAX_PLAYBACK_WIDTH = 1920
MAX_PLAYBACK_HEIGHT = 1080
MAX_PLAYBACK_FPS = 30.0
MAX_NORMALIZATION_TIMEOUT_SECONDS = 15 * 60.0
AUDIO_SAMPLE_BYTES = 4
AUDIO_DECODE_SLACK_SECONDS = 2.0
AUDIO_READ_CHUNK_BYTES = 1024 * 1024


class VideoImportError(ValueError):
    """Raised when an imported video is unsafe, unsupported, or unreadable."""


class VideoNormalizationError(RuntimeError):
    """Raised when an analyzed video cannot get a safe browser playback copy."""


@dataclass(frozen=True)
class VideoProbe:
    duration_seconds: float
    width: int
    height: int
    frame_rate: float
    rotation_degrees: int
    has_audio: bool
    video_codec: str
    pixel_format: str
    audio_codec: str | None
    format_names: tuple[str, ...]


@dataclass(frozen=True)
class PlaybackVideo:
    path: Path
    mime_type: str
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class ImportedVideoResult:
    session: PresentationSession
    playback: PlaybackVideo | None
    media_error: str | None


def _runtime(name: str) -> Path:
    executable = shutil.which(name)
    if not executable:
        raise VideoImportError(f"The local {name} runtime is missing")
    return Path(executable).resolve()


def _parse_frame_rate(value: object) -> float:
    raw = str(value or "0").strip()
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(raw)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return rate if math.isfinite(rate) and rate > 0 else 0.0


def _parse_rotation(stream: dict[str, object]) -> int:
    candidates: list[object] = []
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        candidates.extend(
            item.get("rotation")
            for item in side_data
            if isinstance(item, dict) and "rotation" in item
        )
    tags = stream.get("tags")
    if isinstance(tags, dict):
        candidates.append(tags.get("rotate"))
    for candidate in candidates:
        try:
            rotation = float(candidate)
        except (TypeError, ValueError):
            continue
        nearest = round(rotation / 90.0) * 90
        if math.isfinite(rotation) and abs(rotation - nearest) <= 1.0:
            return int(nearest) % 360
    return 0


class LocalVideoAnalyzer:
    """Decode a local video, sample vision at 15 FPS, and transcribe its audio."""

    def __init__(
        self,
        *,
        model_path: Path,
        transcriber: WhisperCppTranscriber,
        ffmpeg_executable: Path | None = None,
        ffprobe_executable: Path | None = None,
        vision_factory: Callable[[Path], PresentationVisionAnalyzer] = PresentationVisionAnalyzer,
        normalization_timeout_seconds: float | None = None,
        audio_timeout_seconds: float | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.transcriber = transcriber
        self.ffmpeg_executable = (ffmpeg_executable or _runtime("ffmpeg")).resolve()
        self.ffprobe_executable = (ffprobe_executable or _runtime("ffprobe")).resolve()
        self.vision_factory = vision_factory
        if (
            normalization_timeout_seconds is not None
            and (
                not math.isfinite(normalization_timeout_seconds)
                or normalization_timeout_seconds <= 0
            )
        ):
            raise ValueError("Normalization timeout must be positive")
        self.normalization_timeout_seconds = normalization_timeout_seconds
        if (
            audio_timeout_seconds is not None
            and (
                not math.isfinite(audio_timeout_seconds)
                or audio_timeout_seconds <= 0
            )
        ):
            raise ValueError("Audio timeout must be positive")
        self.audio_timeout_seconds = audio_timeout_seconds
        self._analysis_lock = threading.Lock()

    def is_active(self) -> bool:
        return self._analysis_lock.locked()

    def _probe(self, path: Path) -> VideoProbe:
        try:
            size = path.stat().st_size
        except OSError as error:
            raise VideoImportError("The selected video could not be opened") from error
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            raise VideoImportError("Video files must be between 1 byte and 512 MB")
        command = [
            str(self.ffprobe_executable), "-v", "error", "-show_entries",
            (
                "format=duration,format_name:"
                "stream=codec_type,codec_name,pix_fmt,width,height,"
                "avg_frame_rate,r_frame_rate:stream_tags=rotate:"
                "stream_side_data=rotation"
            ),
            "-of", "json", str(path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                start_new_session=True,
            )
            document = json.loads(result.stdout) if result.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            raise VideoImportError("The selected file is not a readable local video") from error
        streams = document.get("streams", ()) if isinstance(document, dict) else ()
        if not isinstance(streams, list):
            streams = []
        video_stream = next(
            (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (
                item for item in streams
                if isinstance(item, dict) and item.get("codec_type") == "audio"
            ),
            None,
        )
        if video_stream is None:
            raise VideoImportError("The selected file does not contain a video stream")
        try:
            duration = float(document.get("format", {}).get("duration", 0))
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
        except (TypeError, ValueError, AttributeError) as error:
            raise VideoImportError("The selected video metadata is invalid") from error
        if not math.isfinite(duration) or not 0.1 <= duration <= MAX_UPLOAD_SECONDS:
            raise VideoImportError("Imported videos must be between 0.1 seconds and 30 minutes")
        if (
            not 1 <= width <= 7680
            or not 1 <= height <= 7680
            or width * height > MAX_IMPORT_SOURCE_PIXELS
        ):
            raise VideoImportError("The selected video resolution is unsupported")
        return VideoProbe(
            duration_seconds=duration,
            width=width,
            height=height,
            frame_rate=(
                _parse_frame_rate(video_stream.get("avg_frame_rate"))
                or _parse_frame_rate(video_stream.get("r_frame_rate"))
            ),
            rotation_degrees=_parse_rotation(video_stream),
            has_audio=audio_stream is not None,
            video_codec=str(video_stream.get("codec_name", "")).casefold()[:64],
            pixel_format=str(video_stream.get("pix_fmt", "")).casefold()[:64],
            audio_codec=(
                str(audio_stream.get("codec_name", "")).casefold()[:64]
                if audio_stream is not None else None
            ),
            format_names=tuple(sorted(
                name.strip().casefold()
                for name in str(
                    document.get("format", {}).get("format_name", "")
                ).split(",")
                if name.strip()
            )),
        )

    def _audio(self, path: Path, probe: VideoProbe) -> np.ndarray:
        if not probe.has_audio:
            return np.zeros(0, dtype=np.float32)
        maximum_bytes = (
            math.ceil(
                (probe.duration_seconds + AUDIO_DECODE_SLACK_SECONDS)
                * SAMPLE_RATE
            )
            * AUDIO_SAMPLE_BYTES
        )
        command = [
            str(self.ffmpeg_executable), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-threads", "2", "-i", str(path),
            "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-t", f"{probe.duration_seconds:.6f}",
            "-f", "f32le", "pipe:1",
        ]
        process: subprocess.Popen | None = None
        decoded = bytearray()
        timeout = (
            self.audio_timeout_seconds
            if self.audio_timeout_seconds is not None
            else max(120.0, probe.duration_seconds * 2.0)
        )
        deadline = time.monotonic() + timeout
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            if process.stdout is None:
                raise VideoImportError(
                    "The local audio decoder did not start"
                )
            decoded = self._read_bounded_audio(
                process.stdout,
                maximum_bytes=maximum_bytes,
                deadline=deadline,
            )
            remaining = max(0.01, deadline - time.monotonic())
            return_code = process.wait(timeout=remaining)
        except (TimeoutError, subprocess.TimeoutExpired) as error:
            raise VideoImportError("The local audio decoder did not complete") from error
        except OSError as error:
            raise VideoImportError("The local audio decoder did not complete") from error
        finally:
            if process is not None:
                self._terminate_process_group(process)
                if process.stdout is not None:
                    process.stdout.close()
        if return_code != 0:
            raise VideoImportError("The selected video's audio could not be decoded")
        if len(decoded) % AUDIO_SAMPLE_BYTES:
            raise VideoImportError("The decoded audio exceeded its expected size")
        return np.frombuffer(decoded, dtype="<f4").astype(np.float32, copy=True)

    @staticmethod
    def _read_bounded_audio(
        stream, *, maximum_bytes: int, deadline: float
    ) -> bytearray:
        import select

        descriptor = stream.fileno()
        decoded = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The bounded audio decoder timed out")
            readable, _, _ = select.select(
                [descriptor], [], [], min(1.0, remaining)
            )
            if not readable:
                continue
            read_size = min(
                AUDIO_READ_CHUNK_BYTES,
                maximum_bytes - len(decoded) + 1,
            )
            try:
                block = os.read(descriptor, read_size)
            except BlockingIOError:
                continue
            if not block:
                return decoded
            decoded.extend(block)
            if len(decoded) > maximum_bytes:
                raise VideoImportError(
                    "The decoded audio exceeded its expected size"
                )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    @staticmethod
    def _read_raw_frame(
        stream, frame_bytes: int, *, deadline: float
    ) -> bytes:
        import select
        import time

        descriptor = stream.fileno()
        frame = bytearray()
        while len(frame) < frame_bytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The bounded video decoder timed out")
            readable, _, _ = select.select(
                [descriptor], [], [], min(1.0, remaining)
            )
            if not readable:
                continue
            try:
                block = os.read(descriptor, min(1024 * 1024, frame_bytes - len(frame)))
            except BlockingIOError:
                continue
            if not block:
                break
            frame.extend(block)
        return bytes(frame)

    def _vision(self, path: Path, probe: VideoProbe):
        import time

        display_width, display_height = probe.width, probe.height
        if probe.rotation_degrees in {90, 270}:
            display_width, display_height = display_height, display_width
        scale = min(1.0, 960.0 / display_width, 960.0 / display_height)
        output_width = max(1, round(display_width * scale))
        output_height = max(1, round(display_height * scale))
        frame_bytes = output_width * output_height * 3
        filters = (
            f"fps=fps={TARGET_IMPORT_FPS:.6f},"
            f"scale={output_width}:{output_height}:flags=fast_bilinear"
        )
        command = [
            str(self.ffmpeg_executable),
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-threads", "2", "-filter_threads", "2",
            "-autorotate", "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", filters,
            "-pix_fmt", "bgr24", "-threads:v", "2",
            "-t", f"{probe.duration_seconds:.6f}",
            "-f", "rawvideo", "pipe:1",
        ]
        analyzer = self.vision_factory(self.model_path)
        process: subprocess.Popen | None = None
        processed = 0
        deadline = time.monotonic() + max(120.0, probe.duration_seconds * 3.0)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            if process.stdout is None:
                raise VideoImportError(
                    "The bounded local video decoder did not start"
                )
            while True:
                raw = self._read_raw_frame(
                    process.stdout, frame_bytes, deadline=deadline
                )
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise VideoImportError(
                        "The selected video's frame stream ended unexpectedly"
                    )
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (output_height, output_width, 3)
                )
                timestamp = processed / TARGET_IMPORT_FPS
                analyzer.process(frame, min(timestamp, probe.duration_seconds))
                processed += 1
            remaining = max(0.01, deadline - time.monotonic())
            if process.wait(timeout=remaining) != 0:
                raise VideoImportError(
                    "The selected video's frames could not be decoded"
                )
            if processed == 0:
                raise VideoImportError("No video frames could be decoded")
            return analyzer.finish(probe.duration_seconds)
        except TimeoutError as error:
            raise VideoImportError(
                "The bounded local video decoder timed out"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise VideoImportError(
                "The bounded local video decoder timed out"
            ) from error
        except OSError as error:
            raise VideoImportError(
                "The bounded local video decoder could not run"
            ) from error
        finally:
            if process is not None:
                self._terminate_process_group(process)
                if process.stdout is not None:
                    process.stdout.close()
            analyzer.close()

    def _source_and_probe(self, path: Path) -> tuple[Path, VideoProbe]:
        try:
            source = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise VideoImportError("The selected video could not be opened") from error
        if source.suffix.casefold() not in SUPPORTED_VIDEO_SUFFIXES:
            raise VideoImportError("Use an MP4, MOV, M4V, or WebM video")
        return source, self._probe(source)

    def _analyze_probed(
        self, source: Path, probe: VideoProbe, *, note: str | None
    ) -> PresentationSession:
        vision_samples = self._vision(source, probe)
        audio = self._audio(source, probe)
        signal = audio_signal_metrics(audio)
        trustworthy = (
            signal["waveform_rms"] >= 0.003
            and signal["clipping_fraction"] < 0.01
        )
        raw_words = self.transcriber.transcribe(audio) if trustworthy else ()
        words = tuple(
            TranscriptWord(
                text=word.text,
                start_seconds=min(
                    max(0.0, word.start_seconds), probe.duration_seconds
                ),
                end_seconds=min(
                    max(0.0, word.end_seconds), probe.duration_seconds
                ),
                probability=word.probability,
            )
            for word in raw_words
            if word.start_seconds < probe.duration_seconds
        )
        return analyze_session(
            vision_samples,
            {
                "words": words,
                "duration_seconds": probe.duration_seconds,
                "waveform_rms": signal["waveform_rms"] if trustworthy else 0.0,
            },
            start_time=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            session_kind="imported",
            note=note,
        )

    def _normalization_timeout(self, probe: VideoProbe) -> float:
        if self.normalization_timeout_seconds is not None:
            return self.normalization_timeout_seconds
        return min(
            MAX_NORMALIZATION_TIMEOUT_SECONDS,
            max(120.0, probe.duration_seconds * 1.25),
        )

    @staticmethod
    def _run_normalization_process(command: list[str], timeout: float) -> int:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # FFmpeg is normally a single process, but killing its isolated
            # process group also prevents an unexpected helper from surviving
            # a bounded conversion timeout.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            raise

    def _normalize_probed(
        self, source: Path, destination: Path, probe: VideoProbe
    ) -> PlaybackVideo:
        try:
            parent_metadata = destination.parent.lstat()
            if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
                parent_metadata.st_mode
            ):
                raise VideoNormalizationError(
                    "The private playback workspace is invalid"
                )
            parent = destination.parent.resolve(strict=True)
        except VideoNormalizationError:
            raise
        except OSError as error:
            raise VideoNormalizationError(
                "The private playback workspace is unavailable"
            ) from error
        target = parent / destination.name
        if target.suffix.casefold() != ".mp4" or target == source:
            raise VideoNormalizationError(
                "The browser playback destination must be a new MP4 file"
            )
        try:
            target.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise VideoNormalizationError(
                "The browser playback destination is unavailable"
            ) from error
        else:
            raise VideoNormalizationError(
                "The browser playback destination already exists"
            )

        temporary = parent / (
            f".{target.stem}-{secrets.token_hex(8)}.normalizing.mp4"
        )
        target_fps = min(
            MAX_PLAYBACK_FPS,
            probe.frame_rate if probe.frame_rate > 0 else MAX_PLAYBACK_FPS,
        )
        filters = (
            f"fps=fps={target_fps:.6f},"
            "scale=w='min(1920,iw)':h='min(1080,ih)':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        )
        command = [
            str(self.ffmpeg_executable),
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-threads", "2", "-filter_threads", "2",
            "-autorotate", "-i", str(source),
            "-map", "0:v:0",
        ]
        if probe.has_audio:
            command.extend(["-map", "0:a:0"])
        command.extend([
            "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
            "-vf", filters,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-threads:v", "2",
        ])
        if probe.has_audio:
            command.extend([
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
            ])
        else:
            command.append("-an")
        command.extend([
            "-t", f"{probe.duration_seconds:.6f}",
            "-max_muxing_queue_size", "1024",
            "-movflags", "+faststart",
            "-fs", str(MAX_UPLOAD_BYTES),
            "-f", "mp4", "-y", str(temporary),
        ])

        try:
            return_code = self._run_normalization_process(
                command, self._normalization_timeout(probe)
            )
            if return_code != 0:
                raise VideoNormalizationError(
                    "The browser-safe local video conversion failed"
                )
            metadata = temporary.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or not 1 <= metadata.st_size <= MAX_UPLOAD_BYTES
            ):
                raise VideoNormalizationError(
                    "The browser-safe local video output is invalid"
                )
            os.chmod(temporary, 0o600)
            try:
                normalized = self._probe(temporary)
            except VideoImportError as error:
                raise VideoNormalizationError(
                    "The browser-safe local video failed verification"
                ) from error
            compatible_container = bool(
                set(normalized.format_names)
                & {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
            )
            duration_tolerance = max(1.0, probe.duration_seconds * 0.02)
            if (
                normalized.video_codec != "h264"
                or normalized.pixel_format != "yuv420p"
                or not 0 < normalized.frame_rate <= MAX_PLAYBACK_FPS + 0.01
                or normalized.rotation_degrees != 0
                or not compatible_container
                or normalized.width > MAX_PLAYBACK_WIDTH
                or normalized.height > MAX_PLAYBACK_HEIGHT
                or abs(normalized.duration_seconds - probe.duration_seconds)
                > duration_tolerance
                or (probe.has_audio and normalized.audio_codec != "aac")
                or (not probe.has_audio and normalized.has_audio)
            ):
                raise VideoNormalizationError(
                    "The browser-safe local video failed its codec contract"
                )
            os.replace(temporary, target)
            os.chmod(target, 0o600)
            return PlaybackVideo(
                path=target,
                mime_type=PLAYBACK_MIME_TYPE,
                duration_seconds=normalized.duration_seconds,
                width=normalized.width,
                height=normalized.height,
            )
        except subprocess.TimeoutExpired as error:
            raise VideoNormalizationError(
                "The browser-safe local video conversion timed out"
            ) from error
        except OSError as error:
            raise VideoNormalizationError(
                "The browser-safe local video could not be created"
            ) from error
        finally:
            try:
                temporary.unlink()
            except OSError:
                # Best-effort cleanup must not turn a verified analysis into a
                # failed upload. The enclosing TemporaryDirectory provides a
                # second deterministic cleanup boundary for server imports.
                pass

    def normalize_for_playback(
        self, path: Path, destination: Path
    ) -> PlaybackVideo:
        if not self._analysis_lock.acquire(blocking=False):
            raise VideoNormalizationError(
                "Another imported video is already being processed"
            )
        try:
            source, probe = self._source_and_probe(path)
            return self._normalize_probed(source, destination, probe)
        finally:
            self._analysis_lock.release()

    def analyze_with_playback(
        self,
        path: Path,
        destination: Path,
        *,
        note: str | None = None,
    ) -> ImportedVideoResult:
        if not self._analysis_lock.acquire(blocking=False):
            raise VideoImportError("Another imported video is already being analyzed")
        try:
            source, probe = self._source_and_probe(path)
            presentation = self._analyze_probed(source, probe, note=note)
            try:
                playback = self._normalize_probed(source, destination, probe)
            except VideoNormalizationError:
                return ImportedVideoResult(
                    session=presentation,
                    playback=None,
                    media_error=(
                        "The measurements were saved, but a browser-safe local "
                        "video could not be created."
                    ),
                )
            return ImportedVideoResult(
                session=presentation,
                playback=playback,
                media_error=None,
            )
        finally:
            self._analysis_lock.release()

    def analyze(self, path: Path, *, note: str | None = None) -> PresentationSession:
        if not self._analysis_lock.acquire(blocking=False):
            raise VideoImportError("Another imported video is already being analyzed")
        try:
            source, probe = self._source_and_probe(path)
            return self._analyze_probed(source, probe, note=note)
        finally:
            self._analysis_lock.release()
