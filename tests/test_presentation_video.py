from pathlib import Path

import pytest

from stroke_screening.presentation_video import LocalVideoAnalyzer, VideoImportError
from stroke_screening.presentation_video_timing import frame_timestamp_seconds


class NeverTranscribe:
    def transcribe(self, _samples):
        raise AssertionError("unsupported files must fail before transcription")


def test_video_import_rejects_unsupported_suffix_before_decoding(tmp_path: Path):
    source = tmp_path / "not-a-video.txt"
    source.write_bytes(b"not a video")
    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path("/usr/bin/true"),
        ffprobe_executable=Path("/usr/bin/true"),
    )
    with pytest.raises(VideoImportError, match="MP4, MOV, M4V, or WebM"):
        analyzer.analyze(source)


def test_frame_timestamp_falls_back_when_decoder_clock_stalls_or_moves_back():
    first = frame_timestamp_seconds(
        0.0, frame_index=0, source_fps=30.0, previous_seconds=None
    )
    stalled = frame_timestamp_seconds(
        0.0, frame_index=1, source_fps=30.0, previous_seconds=first
    )
    backwards = frame_timestamp_seconds(
        10.0, frame_index=2, source_fps=30.0, previous_seconds=stalled
    )

    assert first == 0.0
    assert stalled == pytest.approx(1 / 30)
    assert backwards == pytest.approx(2 / 30)


def test_frame_timestamp_preserves_valid_seek_origin():
    timestamp = frame_timestamp_seconds(
        40_000.0,
        frame_index=0,
        source_fps=15.0,
        previous_seconds=None,
        origin_seconds=40.0,
    )
    assert timestamp == 40.0


def test_frame_timestamp_rejects_implausible_forward_decoder_jump():
    first = frame_timestamp_seconds(
        0.0, frame_index=0, source_fps=30.0, previous_seconds=None
    )
    jumped = frame_timestamp_seconds(
        10_000.0, frame_index=1, source_fps=30.0, previous_seconds=first
    )
    bogus_first = frame_timestamp_seconds(
        3_600_000.0, frame_index=0, source_fps=30.0, previous_seconds=None
    )

    assert jumped == pytest.approx(1 / 30)
    assert bogus_first == 0.0
