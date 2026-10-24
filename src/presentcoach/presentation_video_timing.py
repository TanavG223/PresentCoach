"""Deterministic media-timeline helpers for decoded presentation frames."""

from __future__ import annotations

import math


def frame_timestamp_seconds(
    raw_milliseconds: float,
    *,
    frame_index: int,
    source_fps: float,
    previous_seconds: float | None,
    origin_seconds: float = 0.0,
) -> float:
    """Return a finite, strictly increasing timestamp for a decoded frame.

    Some OpenCV/container combinations repeat, move backward, or jump far
    forward in ``CAP_PROP_POS_MSEC``. Falling back to the decoded-frame index
    keeps MediaPipe VIDEO mode and PresentCoach's temporal smoother on a sane
    media clock.
    """

    if frame_index < 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("frame_index and source_fps must be valid")
    if not math.isfinite(origin_seconds) or origin_seconds < 0:
        raise ValueError("origin_seconds must be finite and non-negative")
    fallback = origin_seconds + frame_index / source_fps
    candidate = float(raw_milliseconds) / 1000.0
    if not math.isfinite(candidate) or candidate < origin_seconds:
        candidate = fallback
    maximum_frame_step = max(1.0, 5.0 / source_fps)
    if previous_seconds is None and candidate > fallback + maximum_frame_step:
        candidate = fallback
    elif (
        previous_seconds is not None
        and candidate > previous_seconds + maximum_frame_step
    ):
        candidate = max(fallback, previous_seconds + 1.0 / source_fps)
    if previous_seconds is not None and candidate <= previous_seconds:
        candidate = max(fallback, previous_seconds + 1.0 / source_fps)
    return candidate
