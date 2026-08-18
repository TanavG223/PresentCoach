"""Bounded local microphone capture and native whisper.cpp transcription."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import wave

import numpy as np
import sounddevice as sd

from .presentation_core import TranscriptWord


SAMPLE_RATE = 16_000
MAX_RECORDING_SECONDS = 30 * 60
EXPECTED_MODEL_SHA256 = "4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f"


class AudioCaptureError(RuntimeError):
    """Raised when local microphone capture cannot start or complete."""


class TranscriptionError(RuntimeError):
    """Raised when the pinned local Whisper runtime cannot be trusted or used."""


class LocalAudioRecorder:
    """Capture 16 kHz mono float audio with a fixed memory ceiling."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._error: str | None = None
        self._samples = 0

    def _callback(self, indata, frames, _time_info, status) -> None:
        if status and status.input_overflow:
            self._error = "The microphone buffer overflowed"
        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._lock:
            if self._samples + frames <= self.sample_rate * MAX_RECORDING_SECONDS:
                self._chunks.append(block)
                self._samples += frames
            else:
                self._error = "The 30-minute local recording limit was reached"

    def start(self) -> None:
        if self._stream is not None:
            raise AudioCaptureError("The microphone is already recording")
        with self._lock:
            self._chunks = []
            self._samples = 0
            self._error = None
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1600,
                latency="low",
                callback=self._callback,
            )
            stream.start()
        except Exception as error:
            raise AudioCaptureError(
                "Microphone access is unavailable. Allow PresentCoach in "
                "System Settings > Privacy & Security > Microphone."
            ) from error
        self._stream = stream

    def stop(self) -> np.ndarray:
        stream = self._stream
        if stream is None:
            raise AudioCaptureError("The microphone is not recording")
        self._stream = None
        try:
            stream.stop()
            stream.close()
        except Exception as error:
            raise AudioCaptureError("The microphone did not stop cleanly") from error
        with self._lock:
            chunks = tuple(self._chunks)
            error = self._error
            self._chunks = []
        if error:
            raise AudioCaptureError(error)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def cancel(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._chunks = []
            self._samples = 0


def audio_signal_metrics(samples: np.ndarray) -> dict[str, float]:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {"waveform_rms": 0.0, "clipping_fraction": 0.0}
    finite = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return {
        "waveform_rms": float(np.sqrt(np.mean(np.square(finite), dtype=np.float64))),
        "clipping_fraction": float(np.mean(np.abs(finite) >= 0.99)),
    }


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _group_tokens(document: dict[str, object]) -> tuple[TranscriptWord, ...]:
    raw_segments = document.get("transcription", ())
    if not isinstance(raw_segments, list):
        raise TranscriptionError("Whisper returned an invalid transcript")
    words: list[TranscriptWord] = []
    current_text = ""
    current_start = 0.0
    current_end = 0.0
    probabilities: list[float] = []

    def flush() -> None:
        nonlocal current_text, current_start, current_end, probabilities
        text = current_text.strip()
        if text and probabilities and sum(probabilities) / len(probabilities) >= 0.25:
            words.append(
                TranscriptWord(
                    text=text,
                    start_seconds=round(current_start, 2),
                    end_seconds=round(max(current_start, current_end), 2),
                    probability=round(sum(probabilities) / len(probabilities), 4),
                )
            )
        current_text = ""
        probabilities = []

    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        tokens = segment.get("tokens", ())
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = str(token.get("text", ""))
            if not text or text.startswith("<|"):
                continue
            offsets = token.get("offsets", {})
            if not isinstance(offsets, dict):
                continue
            start = max(0.0, float(offsets.get("from", 0)) / 1000.0)
            end = max(start, float(offsets.get("to", offsets.get("from", 0))) / 1000.0)
            probability = min(1.0, max(0.0, float(token.get("p", 0))))
            starts_word = bool(text[:1].isspace())
            punctuation_only = not any(character.isalnum() for character in text)
            if current_text and starts_word and not punctuation_only:
                flush()
            if not current_text:
                current_start = start
            current_text += text
            current_end = end
            probabilities.append(probability)
    flush()
    return tuple(words)


class WhisperCppTranscriber:
    """Run a pinned whisper.cpp model without a shell or network endpoint."""

    def __init__(
        self,
        *,
        executable: Path = Path("/opt/homebrew/bin/whisper-cli"),
        model_path: Path,
        timeout_seconds: float = 300.0,
        verify_digest: bool = True,
    ) -> None:
        self.executable = executable.resolve()
        self.model_path = model_path.resolve()
        self.timeout_seconds = timeout_seconds
        if not self.executable.is_file() or not self.model_path.is_file():
            raise TranscriptionError("The local Whisper runtime or model is missing")
        if verify_digest:
            hasher = hashlib.sha256()
            with self.model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != EXPECTED_MODEL_SHA256:
                raise TranscriptionError("The local Whisper model failed its integrity check")

    def transcribe(
        self, samples: np.ndarray, *, sample_rate: int = SAMPLE_RATE
    ) -> tuple[TranscriptWord, ...]:
        if sample_rate != SAMPLE_RATE:
            raise ValueError("Whisper audio must be 16 kHz")
        if len(samples) > SAMPLE_RATE * MAX_RECORDING_SECONDS:
            raise ValueError("Audio exceeds the 30-minute local limit")
        with tempfile.TemporaryDirectory(prefix="presentcoach-whisper-") as temporary:
            directory = Path(temporary)
            input_path = directory / "recording.wav"
            output_stem = directory / "transcript"
            _write_wav(input_path, samples, sample_rate)
            command = [
                str(self.executable), "-m", str(self.model_path), "-f", str(input_path),
                "-l", "en", "-ojf", "-of", str(output_stem), "-np", "-nt",
                "--no-fallback", "--suppress-nst",
                "--prompt", "Um, uh, like, you know, so.",
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.timeout_seconds,
                    start_new_session=True,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise TranscriptionError("The local Whisper process did not complete") from error
            if result.returncode != 0:
                raise TranscriptionError("The local Whisper process rejected the recording")
            output_path = output_stem.with_suffix(".json")
            try:
                if output_path.stat().st_size > 16 * 1024 * 1024:
                    raise TranscriptionError("The local transcript exceeded its size limit")
                document = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise TranscriptionError("The local Whisper transcript was unreadable") from error
            if not isinstance(document, dict):
                raise TranscriptionError("The local Whisper transcript was invalid")
            return _group_tokens(document)
