"""The dog's camera from go2_world (CameraLink) and BoardTwinRobot.get_image."""
import io
import socket
import threading
import time
import types

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from cyberdog.sim.robot.go2_board_robot import (CAM_HEAD, CAM_MAGIC, CAM_PORT,  # noqa: E402
                                                BoardTwinRobot, CameraLink, CameraLost)


def jpeg(colour):
    buf = io.BytesIO()
    Image.fromarray(np.full((480, 640, 3), colour, np.uint8)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class FakeWorld:
    """go2_world's camera end: sends the frames it is given."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.conn = None
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        self.conn, _ = self.srv.accept()

    def send(self, seq, colour):
        for _ in range(100):
            if self.conn is not None:
                break
            time.sleep(0.02)
        j = jpeg(colour)
        self.conn.sendall(CAM_HEAD.pack(CAM_MAGIC, seq, 1.0, time.time(), len(j)) + j)

    def close(self):
        if self.conn:
            self.conn.close()
        self.srv.close()


def wait_for(link, seq, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if link._newest is not None and link._newest[0] == seq:
            return
        time.sleep(0.01)
    raise AssertionError(f"frame {seq} never came")


def test_newest_frame_decoded():
    world = FakeWorld()
    link = CameraLink("127.0.0.1", world.port).start()
    try:
        assert link.get() == (None, None)
        world.send(1, (200, 0, 0))
        world.send(2, (0, 0, 200))
        wait_for(link, 2)
        img, age = link.get()
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        assert img[240, 320, 2] > 150 and img[240, 320, 0] < 50     # the newest, blue
        assert 0.0 <= age < 1.0
        assert link.frames == 2
        assert "2 frames" in link.report()
    finally:
        link.close()
        world.close()


def fake_robot(link, clock):
    r = types.SimpleNamespace(camera=link, _camera_since=None,
                              board=types.SimpleNamespace(clock=clock),
                              CAM_STALE_S=BoardTwinRobot.CAM_STALE_S,
                              CAM_FIRST_S=BoardTwinRobot.CAM_FIRST_S)
    return r


class StubLink:
    addr = ("192.168.55.100", CAM_PORT)

    def __init__(self, img=None, age=None):
        self.img, self.age = img, age

    def get(self):
        return self.img, self.age


def test_get_image_hands_over_a_fresh_frame():
    img = np.zeros((480, 640, 3), np.uint8)
    r = fake_robot(StubLink(img, 0.1), time.monotonic)
    assert BoardTwinRobot.get_image(r) is img


def test_a_stale_frame_ends_the_run():
    r = fake_robot(StubLink(np.zeros((480, 640, 3), np.uint8), 2.5), time.monotonic)
    with pytest.raises(CameraLost, match="2.5 s old"):
        BoardTwinRobot.get_image(r)
    assert CameraLost.layer == "camera"


def test_no_frame_at_all_ends_the_run_after_the_first_wait():
    t = [0.0]

    def clock():
        t[0] += 1.0                   # every look at the clock: a second later
        return t[0]
    r = fake_robot(StubLink(), clock)
    with pytest.raises(CameraLost, match="no frame"):
        BoardTwinRobot.get_image(r)


def test_same_wire_format_as_go2_world():
    cam = pytest.importorskip("go2_world.camera", reason="go2_world not installed")
    assert (cam.MAGIC, cam.HEAD.format, cam.CAM_PORT) == (CAM_MAGIC, CAM_HEAD.format, CAM_PORT)


def test_same_lens_as_the_twin():
    cam = pytest.importorskip("go2_world.camera", reason="go2_world not installed")
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot
    from cyberdog.sim.scene.build_scene import camera_params
    fovy, height = camera_params()
    assert abs(cam.FOVY - fovy) < 0.01 and cam.CAM_HEIGHT == height
    assert cam.CAM_FORWARD == MujocoRobot.CAM_FORWARD
    assert (cam.W, cam.H) == (MujocoRobot.CAM_W, MujocoRobot.CAM_H)
