"""Bounded, offline analysis for user-supplied presentation videos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Callable

import cv2
import numpy as np

from .presentation_audio import SAMPLE_RATE, WhisperCppTranscriber, audio_signal_metrics
from .presentation_core import PresentationSession, TranscriptWord, analyze_session
from .presentation_vision import PresentationVisionAnalyzer


MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_UPLOAD_SECONDS = 30 * 60
TARGET_IMPORT_FPS = 15.0
SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".webm"})


class VideoImportError(ValueError):
    """Raised when an imported video is unsafe, unsupported, or unreadable."""


@dataclass(frozen=True)
class VideoProbe:
    duration_seconds: float
    width: int
    height: int
    has_audio: bool


def _runtime(name: str) -> Path:
    executable = shutil.which(name)
    if not executable:
        raise VideoImportError(f"The local {name} runtime is missing")
    return Path(executable).resolve()


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
    ) -> None:
        self.model_path = model_path.resolve()
        self.transcriber = transcriber
        self.ffmpeg_executable = (ffmpeg_executable or _runtime("ffmpeg")).resolve()
        self.ffprobe_executable = (ffprobe_executable or _runtime("ffprobe")).resolve()
        self.vision_factory = vision_factory
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
            "format=duration:stream=codec_type,width,height", "-of", "json", str(path),
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
        if not 1 <= width <= 7680 or not 1 <= height <= 4320:
            raise VideoImportError("The selected video resolution is unsupported")
        return VideoProbe(
            duration_seconds=duration,
            width=width,
            height=height,
            has_audio=any(
                isinstance(item, dict) and item.get("codec_type") == "audio"
                for item in streams
            ),
        )

    def _audio(self, path: Path, probe: VideoProbe) -> np.ndarray:
        if not probe.has_audio:
            return np.zeros(0, dtype=np.float32)
        command = [
            str(self.ffmpeg_executable), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=max(120.0, probe.duration_seconds * 2.0),
                start_new_session=True,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VideoImportError("The local audio decoder did not complete") from error
        if result.returncode != 0:
            raise VideoImportError("The selected video's audio could not be decoded")
        maximum_bytes = int((probe.duration_seconds + 2.0) * SAMPLE_RATE * 4)
        if len(result.stdout) > maximum_bytes or len(result.stdout) % 4:
            raise VideoImportError("The decoded audio exceeded its expected size")
        return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)

    def _vision(self, path: Path, probe: VideoProbe):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoImportError("The selected video's frames could not be decoded")
        analyzer = self.vision_factory(self.model_path)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        next_sample = 0.0
        frame_index = 0
        processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not math.isfinite(timestamp) or timestamp < 0:
                    timestamp = frame_index / source_fps
                frame_index += 1
                if timestamp + (0.5 / source_fps) < next_sample:
                    continue
                if frame.shape[1] > 960:
                    scale = 960.0 / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (960, max(1, round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                analyzer.process(frame, min(timestamp, probe.duration_seconds))
                processed += 1
                next_sample += 1.0 / TARGET_IMPORT_FPS
                if timestamp > next_sample:
                    next_sample = timestamp + 1.0 / TARGET_IMPORT_FPS
            if processed == 0:
                raise VideoImportError("No video frames could be decoded")
            return analyzer.finish(probe.duration_seconds)
        finally:
            capture.release()
            analyzer.close()

    def analyze(self, path: Path, *, note: str | None = None) -> PresentationSession:
        if not self._analysis_lock.acquire(blocking=False):
            raise VideoImportError("Another imported video is already being analyzed")
        try:
            source = path.resolve(strict=True)
            if source.suffix.casefold() not in SUPPORTED_VIDEO_SUFFIXES:
                raise VideoImportError("Use an MP4, MOV, M4V, or WebM video")
            probe = self._probe(source)
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
                    start_seconds=min(max(0.0, word.start_seconds), probe.duration_seconds),
                    end_seconds=min(max(0.0, word.end_seconds), probe.duration_seconds),
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
        except FileNotFoundError as error:
            raise VideoImportError("The selected video could not be opened") from error
        finally:
            self._analysis_lock.release()
