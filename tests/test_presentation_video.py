from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import pytest

import stroke_screening.presentation_video as presentation_video_module
from stroke_screening.presentation_audio import SAMPLE_RATE
from stroke_screening.presentation_core import analyze_session
from stroke_screening.presentation_video import (
    LocalVideoAnalyzer,
    VideoImportError,
    VideoNormalizationError,
    VideoProbe,
)
from stroke_screening.presentation_video_timing import frame_timestamp_seconds


class NeverTranscribe:
    def transcribe(self, _samples):
        raise AssertionError("unsupported files must fail before transcription")


def _local_video_analyzer(
    tmp_path: Path,
    *,
    ffmpeg_executable: Path | None = None,
    ffprobe_executable: Path | None = None,
    normalization_timeout_seconds: float | None = None,
    audio_timeout_seconds: float | None = None,
) -> LocalVideoAnalyzer:
    ffmpeg = ffmpeg_executable or shutil.which("ffmpeg")
    ffprobe = ffprobe_executable or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("real FFmpeg and ffprobe runtimes are required")
    return LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path(ffmpeg),
        ffprobe_executable=Path(ffprobe),
        normalization_timeout_seconds=normalization_timeout_seconds,
        audio_timeout_seconds=audio_timeout_seconds,
    )


def _run_fixture_ffmpeg(arguments: list[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("real FFmpeg runtime is required")
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"required local FFmpeg encoder is unavailable: {result.stderr}")


def _ffprobe_document(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("real ffprobe runtime is required")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries",
            (
                "format=format_name,duration:"
                "stream=codec_type,codec_name,pix_fmt,avg_frame_rate,width,height:"
                "stream_side_data=rotation"
            ),
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _fixture_probe(*, has_audio: bool) -> VideoProbe:
    return VideoProbe(
        duration_seconds=1.0,
        width=320,
        height=240,
        frame_rate=15.0,
        rotation_degrees=0,
        has_audio=has_audio,
        video_codec="prores",
        pixel_format="yuv422p10le",
        audio_codec="pcm_s16le" if has_audio else None,
        format_names=("mov",),
    )


class SourceOnlyProbeAnalyzer(LocalVideoAnalyzer):
    """Trust only the unit-test source; use real probing for generated output."""

    def __init__(self, *, source: Path, source_probe: VideoProbe, **kwargs):
        super().__init__(**kwargs)
        self.source = source.resolve()
        self.source_probe = source_probe

    def _probe(self, path: Path) -> VideoProbe:
        if path.resolve() == self.source:
            return self.source_probe
        return super()._probe(path)


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


def test_video_probe_rejects_above_4k_pixel_area_before_decoding(tmp_path: Path):
    source = tmp_path / "oversized.mp4"
    source.write_bytes(b"container-placeholder")
    fake_ffprobe = tmp_path / "fake-ffprobe"
    fake_ffprobe.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({"
        "'format': {'duration': '1', 'format_name': 'mov,mp4'}, "
        "'streams': [{'codec_type': 'video', 'codec_name': 'hevc', "
        "'pix_fmt': 'yuv420p', 'width': 7680, 'height': 4320, "
        "'avg_frame_rate': '30/1'}]}))\n",
        encoding="utf-8",
    )
    fake_ffprobe.chmod(0o700)
    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path("/usr/bin/true"),
        ffprobe_executable=fake_ffprobe,
    )

    with pytest.raises(VideoImportError, match="resolution is unsupported"):
        analyzer.analyze(source)


def test_audio_decode_stops_at_hard_pcm_ceiling_and_kills_process_group(
    tmp_path: Path,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"container-placeholder")
    arguments_path = tmp_path / "audio-arguments.json"
    spawned = tmp_path / "child-was-spawned"
    sentinel = tmp_path / "orphaned-child-ran"
    fake_ffmpeg = tmp_path / "unbounded-audio-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, subprocess, sys\n"
        f"pathlib.Path({str(arguments_path)!r}).write_text("
        "json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        f"child_code = \"import pathlib,time; time.sleep(0.5); "
        f"pathlib.Path({str(sentinel)!r}).write_text('orphan')\"\n"
        "subprocess.Popen([sys.executable, '-c', child_code])\n"
        f"pathlib.Path({str(spawned)!r}).write_text('spawned')\n"
        "chunk = b'\\x00' * 65536\n"
        "while True:\n"
        "    os.write(1, chunk)\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=Path("/usr/bin/true"),
        audio_timeout_seconds=5.0,
    )
    probe = replace(_fixture_probe(has_audio=True), duration_seconds=0.1)

    started = time.monotonic()
    with pytest.raises(VideoImportError, match="exceeded its expected size"):
        analyzer._audio(source, probe)

    assert time.monotonic() - started < 5.0
    assert spawned.exists()
    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    assert arguments[arguments.index("-t") + 1] == "0.100000"
    time.sleep(0.6)
    assert not sentinel.exists()


def test_audio_decode_timeout_kills_ffmpeg_process_group(tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"container-placeholder")
    spawned = tmp_path / "child-was-spawned"
    sentinel = tmp_path / "orphaned-child-ran"
    fake_ffmpeg = tmp_path / "stalled-audio-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\n"
        f"(sleep 2; printf orphaned > {shlex.quote(str(sentinel))}) &\n"
        f"printf spawned > {shlex.quote(str(spawned))}\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=Path("/usr/bin/true"),
        audio_timeout_seconds=1.0,
    )

    with pytest.raises(VideoImportError, match="did not complete"):
        analyzer._audio(source, _fixture_probe(has_audio=True))

    assert spawned.exists()
    time.sleep(2.1)
    assert not sentinel.exists()


def test_audio_decode_preserves_nonzero_ffmpeg_error(tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"container-placeholder")
    fake_ffmpeg = tmp_path / "failed-audio-ffmpeg"
    fake_ffmpeg.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    fake_ffmpeg.chmod(0o700)
    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=Path("/usr/bin/true"),
        audio_timeout_seconds=1.0,
    )

    with pytest.raises(
        VideoImportError, match="audio could not be decoded"
    ):
        analyzer._audio(source, _fixture_probe(has_audio=True))


def test_real_audio_decode_obeys_probed_duration_cutoff(tmp_path: Path):
    source = tmp_path / "longer-source.mp4"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "2", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-y", str(source),
    ])
    analyzer = _local_video_analyzer(tmp_path)
    shortened_probe = replace(
        analyzer._probe(source), duration_seconds=0.25
    )

    audio = analyzer._audio(source, shortened_probe)

    assert audio.dtype.name == "float32"
    assert int(0.20 * SAMPLE_RATE) <= audio.size
    assert audio.size <= int(0.27 * SAMPLE_RATE)


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


def test_prores_pcm_mov_normalizes_to_h264_aac_mp4(tmp_path: Path):
    source = tmp_path / "legacy-presentation.mov"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "1", "-shortest",
        "-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le",
        "-c:a", "pcm_s16le", "-y", str(source),
    ])
    destination = tmp_path / "browser.mp4"

    playback = _local_video_analyzer(tmp_path).normalize_for_playback(
        source, destination
    )

    document = _ffprobe_document(playback.path)
    streams = document["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert playback.path == destination
    assert playback.mime_type == "video/mp4"
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert audio["codec_name"] == "aac"
    assert "mp4" in document["format"]["format_name"].split(",")
    assert os.stat(destination).st_mode & 0o777 == 0o600
    assert not any("normalizing" in item.name for item in tmp_path.iterdir())


def test_hevc_aac_mp4_normalizes_to_browser_h264_aac(tmp_path: Path):
    source = tmp_path / "phone-hevc.mp4"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "1", "-shortest",
        "-c:v", "libx265", "-preset", "ultrafast",
        "-x265-params", "pools=2:frame-threads=2:log-level=error",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
        "-y", str(source),
    ])
    analyzer = _local_video_analyzer(tmp_path)
    assert analyzer._probe(source).video_codec == "hevc"

    playback = analyzer.normalize_for_playback(source, tmp_path / "browser.mp4")

    document = _ffprobe_document(playback.path)
    streams = document["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert audio["codec_name"] == "aac"


def test_mislabeled_webm_without_audio_normalizes_to_video_only_mp4(
    tmp_path: Path,
):
    webm = tmp_path / "source.webm"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=2048x64:rate=120",
        "-t", "1", "-an",
        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
        "-y", str(webm),
    ])
    mislabeled = tmp_path / "content-is-webm.mp4"
    webm.replace(mislabeled)

    playback = _local_video_analyzer(tmp_path).normalize_for_playback(
        mislabeled, tmp_path / "browser.mp4"
    )

    document = _ffprobe_document(playback.path)
    streams = document["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert playback.width == 1920
    assert playback.height == 60
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    numerator, denominator = video["avg_frame_rate"].split("/", 1)
    assert float(numerator) / float(denominator) <= 30.0
    assert not any(stream["codec_type"] == "audio" for stream in streams)


def test_high_fps_import_uses_bounded_15_fps_thread_capped_decoder(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "high-fps.webm"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=2048x64:rate=120",
        "-t", "1", "-an",
        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
        "-y", str(source),
    ])
    observed: list[tuple[tuple[int, ...], float]] = []
    commands: list[list[str]] = []

    class CollectingVision:
        def __init__(self, _model_path):
            pass

        def process(self, frame, timestamp_seconds):
            observed.append((frame.shape, timestamp_seconds))

        def finish(self, _duration_seconds):
            return tuple(observed)

        def close(self):
            pass

    real_popen = subprocess.Popen

    def capturing_popen(command, *args, **kwargs):
        commands.append(list(command))
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(presentation_video_module.subprocess, "Popen", capturing_popen)
    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path(shutil.which("ffmpeg") or "/missing/ffmpeg"),
        ffprobe_executable=Path(shutil.which("ffprobe") or "/missing/ffprobe"),
        vision_factory=CollectingVision,
    )
    probe = analyzer._probe(source)

    analyzer._vision(source, probe)

    assert 14 <= len(observed) <= 16
    assert all(shape == (30, 960, 3) for shape, _ in observed)
    assert observed[-1][1] <= 1.0
    command = commands[-1]
    assert command[command.index("-threads") + 1] == "2"
    assert command[command.index("-filter_threads") + 1] == "2"
    assert command[command.index("-threads:v") + 1] == "2"
    assert "fps=fps=15.000000" in command[command.index("-vf") + 1]


def test_rotated_phone_video_preserves_portrait_geometry_in_analysis_and_playback(
    tmp_path: Path,
):
    base = tmp_path / "landscape.mp4"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
        "-t", "1", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", str(base),
    ])
    rotated = tmp_path / "phone-portrait.mov"
    _run_fixture_ffmpeg([
        "-display_rotation", "90", "-i", str(base),
        "-c", "copy", "-y", str(rotated),
    ])
    observed_shapes: list[tuple[int, ...]] = []

    class CollectingVision:
        def __init__(self, _model_path):
            pass

        def process(self, frame, _timestamp_seconds):
            observed_shapes.append(frame.shape)

        def finish(self, _duration_seconds):
            return tuple(observed_shapes)

        def close(self):
            pass

    analyzer = LocalVideoAnalyzer(
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path(shutil.which("ffmpeg") or "/missing/ffmpeg"),
        ffprobe_executable=Path(shutil.which("ffprobe") or "/missing/ffprobe"),
        vision_factory=CollectingVision,
    )
    probe = analyzer._probe(rotated)

    analyzer._vision(rotated, probe)
    playback = analyzer.normalize_for_playback(rotated, tmp_path / "browser.mp4")

    assert probe.rotation_degrees == 90
    assert observed_shapes
    assert all(shape == (320, 180, 3) for shape in observed_shapes)
    assert (playback.width, playback.height) == (180, 320)
    normalized = _ffprobe_document(playback.path)
    video = next(
        stream
        for stream in normalized["streams"]
        if stream["codec_type"] == "video"
    )
    assert (video["width"], video["height"]) == (180, 320)
    assert not video.get("side_data_list")


def test_malformed_normalized_output_is_rejected_and_cleaned(tmp_path: Path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-placeholder")
    fake_ffmpeg = tmp_path / "fake-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'not a video')\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    analyzer = SourceOnlyProbeAnalyzer(
        source=source,
        source_probe=_fixture_probe(has_audio=False),
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=Path("/usr/bin/true"),
    )
    destination = tmp_path / "browser.mp4"

    with pytest.raises(VideoNormalizationError, match="failed verification"):
        analyzer.normalize_for_playback(source, destination)

    assert not destination.exists()
    assert not any("normalizing" in item.name for item in tmp_path.iterdir())


def test_valid_but_non_browser_codec_output_fails_closed(tmp_path: Path):
    source = tmp_path / "source.webm"
    _run_fixture_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15",
        "-t", "1", "-an",
        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
        "-y", str(source),
    ])
    fake_ffmpeg = tmp_path / "copy-source"
    fake_ffmpeg.write_text(
        "#!/bin/sh\n"
        "for last do :; done\n"
        f"cp {shlex.quote(str(source))} \"$last\"\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    analyzer = _local_video_analyzer(
        tmp_path,
        ffmpeg_executable=fake_ffmpeg,
    )
    destination = tmp_path / "browser.mp4"

    with pytest.raises(VideoNormalizationError, match="codec contract"):
        analyzer.normalize_for_playback(source, destination)

    assert not destination.exists()
    assert not any("normalizing" in item.name for item in tmp_path.iterdir())


def test_normalization_timeout_is_bounded_and_partial_output_is_cleaned(
    tmp_path: Path,
):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-placeholder")
    sentinel = tmp_path / "orphaned-child-ran"
    spawned = tmp_path / "child-was-spawned"
    fake_ffmpeg = tmp_path / "slow-ffmpeg"
    fake_ffmpeg.write_text(
        "#!/bin/sh\n"
        f"(sleep 0.4; printf orphaned > {shlex.quote(str(sentinel))}) &\n"
        f"printf spawned > {shlex.quote(str(spawned))}\n"
        "for last do :; done\n"
        "printf partial > \"$last\"\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o700)
    analyzer = SourceOnlyProbeAnalyzer(
        source=source,
        source_probe=_fixture_probe(has_audio=False),
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=fake_ffmpeg,
        ffprobe_executable=Path("/usr/bin/true"),
        normalization_timeout_seconds=0.15,
    )
    destination = tmp_path / "browser.mp4"

    with pytest.raises(VideoNormalizationError, match="timed out"):
        analyzer.normalize_for_playback(source, destination)

    assert spawned.exists()
    time.sleep(0.5)
    assert not sentinel.exists()
    assert not destination.exists()
    assert not any("normalizing" in item.name for item in tmp_path.iterdir())


def test_normalization_never_overwrites_existing_destination(tmp_path: Path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-placeholder")
    destination = tmp_path / "browser.mp4"
    destination.write_bytes(b"keep-me")
    analyzer = SourceOnlyProbeAnalyzer(
        source=source,
        source_probe=_fixture_probe(has_audio=False),
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path("/usr/bin/true"),
        ffprobe_executable=Path("/usr/bin/true"),
    )

    with pytest.raises(VideoNormalizationError, match="already exists"):
        analyzer.normalize_for_playback(source, destination)

    assert destination.read_bytes() == b"keep-me"


def test_normalization_rejects_a_symlink_playback_workspace(tmp_path: Path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-placeholder")
    real_directory = tmp_path / "real-private-workspace"
    real_directory.mkdir(mode=0o700)
    linked_directory = tmp_path / "linked-workspace"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    analyzer = SourceOnlyProbeAnalyzer(
        source=source,
        source_probe=_fixture_probe(has_audio=False),
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path("/usr/bin/true"),
        ffprobe_executable=Path("/usr/bin/true"),
    )

    with pytest.raises(VideoNormalizationError, match="workspace is invalid"):
        analyzer.normalize_for_playback(
            source, linked_directory / "browser.mp4"
        )

    assert not (real_directory / "browser.mp4").exists()


def test_successful_analysis_is_returned_when_playback_normalization_fails(
    tmp_path: Path,
):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-placeholder")
    presentation = analyze_session(
        (),
        {"words": (), "duration_seconds": 31.0, "waveform_rms": 0.0},
        session_kind="imported",
    )

    class FailingNormalizer(SourceOnlyProbeAnalyzer):
        def _analyze_probed(self, _source, _probe, *, note):
            assert note == "keep measurements"
            return presentation

        def _normalize_probed(self, _source, _destination, _probe):
            raise VideoNormalizationError("synthetic conversion failure")

    analyzer = FailingNormalizer(
        source=source,
        source_probe=_fixture_probe(has_audio=False),
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path("/usr/bin/true"),
        ffprobe_executable=Path("/usr/bin/true"),
    )

    imported = analyzer.analyze_with_playback(
        source,
        tmp_path / "browser.mp4",
        note="keep measurements",
    )

    assert imported.session is presentation
    assert imported.playback is None
    assert imported.media_error is not None
    assert "measurements were saved" in imported.media_error


def test_successful_analysis_survives_playback_workspace_resolution_race(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "source.mov"
    source.write_bytes(b"source-placeholder")
    playback_directory = tmp_path / "private-playback"
    playback_directory.mkdir(mode=0o700)
    presentation = analyze_session(
        (),
        {"words": (), "duration_seconds": 31.0, "waveform_rms": 0.0},
        session_kind="imported",
    )

    class AnalysisOnly(SourceOnlyProbeAnalyzer):
        def _analyze_probed(self, _source, _probe, *, note):
            return presentation

    analyzer = AnalysisOnly(
        source=source,
        source_probe=_fixture_probe(has_audio=False),
        model_path=tmp_path / "model.task",
        transcriber=NeverTranscribe(),
        ffmpeg_executable=Path("/usr/bin/true"),
        ffprobe_executable=Path("/usr/bin/true"),
    )
    real_resolve = Path.resolve

    def fail_workspace_resolve(path, *args, **kwargs):
        if path == playback_directory:
            raise OSError("workspace disappeared after lstat")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_workspace_resolve)

    imported = analyzer.analyze_with_playback(
        source, playback_directory / "browser.mp4"
    )

    assert imported.session is presentation
    assert imported.playback is None
    assert imported.media_error is not None
