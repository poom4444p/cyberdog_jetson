"""Dreaming time per frame: counted against the brain's control tick.

The tick that delivers VAMOS's answer also dreams it, so a frame whose
dreams take longer than TICK_MS holds that tick up. VamosPolicy counts the
time per frame (stats, dream_ms) so a run can report its worst frame.
"""
import time

from cyberdog.sim.robot.go2_board_robot import Go2BoardRobot
from cyberdog.sim.robot.mujoco_robot import MujocoRobot
from cyberdog.sim.vamos_client import TICK_MS, VamosPolicy

POSE = (0.0, 0.0, 0.0)
PATHS = [[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
         [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0)]]


class SlowDream:
    """Takes `ms` per candidate and passes it."""

    def __init__(self, ms):
        self.ms = ms

    def factor(self, path, pose):
        time.sleep(self.ms / 1000.0)
        return 1.0, {"confident": True}


def policy(dream):
    return VamosPolicy(cam=None, is_free=lambda x, y: True, dream=dream)


def test_tick_is_the_control_tick():
    assert TICK_MS == MujocoRobot.CONTROL_DT * 1000 == Go2BoardRobot.CONTROL_DT * 1000


def test_fast_frame_counted_not_over():
    p = policy(SlowDream(1))
    best, _, _ = p.deliver(PATHS, POSE, (5.0, 0.0))
    assert best is not None
    st = p.stats
    assert st["dream_frames"] == 1 and st["dream_over_tick"] == 0
    assert 2 <= p.dream_ms < TICK_MS
    assert st["dream_ms_max"] == p.dream_ms == st["dream_ms_sum"]


def test_slow_frame_over_tick_and_worst_kept():
    p = policy(SlowDream(1))
    p.deliver(PATHS, POSE, (5.0, 0.0))
    fast = p.dream_ms
    p.dream = SlowDream(30)             # 2 candidates x 30 ms > 50 ms
    p.deliver(PATHS, POSE, (5.0, 0.0))
    st = p.stats
    assert p.dream_ms > TICK_MS
    assert st["dream_frames"] == 2 and st["dream_over_tick"] == 1
    assert st["dream_ms_max"] == p.dream_ms
    assert abs(st["dream_ms_sum"] - (fast + p.dream_ms)) < 1e-9
    line = p.dream_summary()
    assert "2 frames" in line and "1 over the 50 ms tick" in line


def test_no_dream_no_count():
    p = policy(None)
    p.deliver(PATHS, POSE, (5.0, 0.0))
    assert p.dream_ms is None and p.stats["dream_frames"] == 0
    assert p.dream_summary() == "dream time: no frame dreamed"
    # A failed call dreams nothing either, and clears the last frame's time.
    p = policy(SlowDream(1))
    p.deliver(PATHS, POSE, (5.0, 0.0))
    p.deliver(None, POSE, (5.0, 0.0))
    assert p.dream_ms is None and p.stats["dream_frames"] == 1
