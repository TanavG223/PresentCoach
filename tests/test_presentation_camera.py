import threading

import numpy as np
import pytest

import presentcoach.presentation_camera as presentation_camera_module
from presentcoach.presentation_camera import (
    CameraSessionError,
    PresentationCameraService,
)


def test_blocked_reader_is_never_closed_concurrently_and_eventually_closes(monkeypatch):
    class BlockedCamera:
        instance = None

        def __init__(self, *_args, **_kwargs):
            type(self).instance = self
            self.read_started = threading.Event()
            self.release_read = threading.Event()
            self.closed_after_read = threading.Event()
            self.read_active = False
            self.close_during_read = False
            self.close_count = 0
            self.read_count = 0

        def open(self):
            return None

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return np.zeros((8, 8, 3), dtype=np.uint8)
            self.read_started.set()
            self.read_active = True
            try:
                assert self.release_read.wait(timeout=2.0)
                return np.zeros((8, 8, 3), dtype=np.uint8)
            finally:
                self.read_active = False

        def close(self):
            if self.read_active:
                self.close_during_read = True
            self.close_count += 1
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

    assert camera.close_count == 0
    assert camera.close_during_read is False
    assert reader is not None and reader.is_alive()
    with pytest.raises(CameraSessionError, match="expired"):
        service.next_frame(owner, timeout=0.01)

    camera.release_read.set()
    assert camera.closed_after_read.wait(timeout=1.0)
    reader.join(timeout=1.0)
    assert not reader.is_alive()
    assert camera.close_count == 1
    assert camera.close_during_read is False
    assert service._owner is None
    assert service._camera is None


def test_camera_start_requires_a_real_frame_before_returning(monkeypatch):
    class NoFrameCamera:
        instance = None

        def __init__(self, *_args, **_kwargs):
            type(self).instance = self
            self.closed = threading.Event()

        def open(self):
            return None

        def read(self):
            raise RuntimeError("native capture returned no frame")

        def close(self):
            self.closed.set()

    monkeypatch.setattr(presentation_camera_module, "LocalCamera", NoFrameCamera)
    service = PresentationCameraService(
        timeout_seconds=30.0,
        reader_join_timeout_seconds=0.1,
        startup_frame_timeout_seconds=0.1,
    )

    with pytest.raises(CameraSessionError, match="opened but did not return a frame"):
        service.start()

    assert NoFrameCamera.instance is not None
    assert NoFrameCamera.instance.closed.wait(timeout=1.0)
    assert service._owner is None
    assert service._camera is None


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
    assert camera.closed.wait(timeout=1.5)
    thread = service._thread
    if thread is not None:
        thread.join(timeout=1.0)
    assert camera.close_count == 1
    assert service._owner is None
    assert service._camera is None
    assert service._thread is None
    assert service._timer is None
