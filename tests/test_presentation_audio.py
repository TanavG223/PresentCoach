import numpy as np
import pytest

import stroke_screening.presentation_audio as presentation_audio_module
from stroke_screening.presentation_audio import (
    AudioCaptureError,
    LocalAudioRecorder,
    audio_signal_metrics,
)


def test_audio_signal_metrics_detects_silence_and_clipping():
    assert audio_signal_metrics(np.zeros(160, dtype=np.float32))["waveform_rms"] == 0
    clipped = audio_signal_metrics(np.ones(160, dtype=np.float32))
    assert clipped["clipping_fraction"] == 1


def test_audio_signal_metrics_reports_finite_rms():
    metrics = audio_signal_metrics(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert 0.4 < metrics["waveform_rms"] < 0.42
    assert metrics["clipping_fraction"] == 0


def test_microphone_start_failure_aborts_and_closes_allocated_stream(monkeypatch):
    class FailingStartStream:
        def __init__(self, **_kwargs):
            self.aborted = 0
            self.closed = 0

        def start(self):
            raise RuntimeError("PortAudio start failed")

        def abort(self):
            self.aborted += 1
            raise RuntimeError("abort also failed")

        def close(self):
            self.closed += 1

    stream = FailingStartStream()
    monkeypatch.setattr(
        presentation_audio_module.sd, "InputStream", lambda **_kwargs: stream
    )
    recorder = LocalAudioRecorder()

    with pytest.raises(AudioCaptureError, match="Microphone access"):
        recorder.start()

    assert stream.aborted == 1
    assert stream.closed == 1
    assert recorder._stream is None


def test_microphone_stop_failure_aborts_closes_and_discards_samples(monkeypatch):
    class FailingStopStream:
        def __init__(self, **_kwargs):
            self.aborted = 0
            self.closed = 0

        def start(self):
            return None

        def stop(self):
            raise RuntimeError("PortAudio stop failed")

        def abort(self):
            self.aborted += 1

        def close(self):
            self.closed += 1

    stream = FailingStopStream()
    monkeypatch.setattr(
        presentation_audio_module.sd, "InputStream", lambda **_kwargs: stream
    )
    recorder = LocalAudioRecorder()
    recorder.start()
    recorder._callback(
        np.ones((4, 1), dtype=np.float32), 4, None, None
    )

    with pytest.raises(AudioCaptureError, match="did not stop cleanly"):
        recorder.stop()

    assert stream.aborted == 1
    assert stream.closed == 1
    assert recorder._stream is None
    assert recorder._chunks == []
    assert recorder._samples == 0
