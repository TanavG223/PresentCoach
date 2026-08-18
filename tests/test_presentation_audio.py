import numpy as np

from stroke_screening.presentation_audio import audio_signal_metrics


def test_audio_signal_metrics_detects_silence_and_clipping():
    assert audio_signal_metrics(np.zeros(160, dtype=np.float32))["waveform_rms"] == 0
    clipped = audio_signal_metrics(np.ones(160, dtype=np.float32))
    assert clipped["clipping_fraction"] == 1


def test_audio_signal_metrics_reports_finite_rms():
    metrics = audio_signal_metrics(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert 0.4 < metrics["waveform_rms"] < 0.42
    assert metrics["clipping_fraction"] == 0
