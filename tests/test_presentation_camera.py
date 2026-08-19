import threading

import numpy as np
import pytest

import stroke_screening.presentation_camera as presentation_camera_module
from stroke_screening.presentation_camera import (
    CameraSessionError,
    PresentationCameraService,
)


def test_blocked_reader_retains_and_eventually_closes_concrete_camera(monkeypatch):
    class BlockedCamera:
        instance = None

        def __init__(self, *_args, **_kwargs):
            type(self).instance = self
            self.read_started = threading.Event()
            self.release_read = threading.Event()
            self.closed_after_read = threading.Event()
            self.close_count = 0

        def open(self):
            return None

        def read(self):
            self.read_started.set()
            assert self.release_read.wait(timeout=2.0)
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            self.close_count += 1
            if self.close_count >= 2:
                self.closed_after_read.set()

    monkeypatch.setattr(presentation_camera_module, "LocalCamera", BlockedCamera)
    service = PresentationCameraService(
        timeout_seconds=30.0, reader_join_timeout_seconds=0.03
    )
    owner = service.start()
    camera = BlockedCamera.instance
    assert camera is not None
    assert camera.read_started.wait(timeout=1.0)
    reader = service._thread

    service.stop(owner)

    assert camera.close_count == 1
    assert reader is not None and reader.is_alive()
    with pytest.raises(CameraSessionError, match="expired"):
        service.next_frame(owner, timeout=0.01)

    camera.release_read.set()
    assert camera.closed_after_read.wait(timeout=1.0)
    reader.join(timeout=1.0)
    assert not reader.is_alive()
    assert camera.close_count == 2


@pytest.mark.parametrize("failure_phase", ("reader_start", "timer_start"))
def test_camera_start_failure_rolls_back_owner_and_closes_capture(
    monkeypatch, failure_phase: str
):
    class Camera:
        instance = None

        def __init__(self, *_args, **_kwargs):
            type(self).instance = self
            self.closed = threading.Event()
            self.close_count = 0

        def open(self):
            return None

        def read(self):
            self.closed.wait(timeout=1.0)
            raise RuntimeError("camera closed")

        def close(self):
            self.close_count += 1
            self.closed.set()

    monkeypatch.setattr(presentation_camera_module, "LocalCamera", Camera)

    if failure_phase == "reader_start":
        original_start = threading.Thread.start

        def fail_reader_start(thread):
            if thread.name == "presentcoach-camera":
                raise RuntimeError("reader start failed")
            return original_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_reader_start)
    else:
        class FailingTimer:
            def __init__(self, *_args, **_kwargs):
                self.daemon = False

            def start(self):
                raise RuntimeError("timer start failed")

            def cancel(self):
                return None

        monkeypatch.setattr(presentation_camera_module.threading, "Timer", FailingTimer)

    service = PresentationCameraService(
        timeout_seconds=30.0, reader_join_timeout_seconds=0.1
    )
    with pytest.raises(CameraSessionError, match="worker could not start"):
        service.start()

    camera = Camera.instance
    assert camera is not None
    assert camera.close_count >= 1
    assert service._owner is None
    assert service._camera is None
    assert service._thread is None
    assert service._timer is None
