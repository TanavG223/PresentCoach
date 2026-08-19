from pathlib import Path

import pytest

from stroke_screening.presentation_video import LocalVideoAnalyzer, VideoImportError


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
