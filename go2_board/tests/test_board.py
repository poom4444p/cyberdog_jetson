"""Board, Safety and Pose against a fake bus: no sim, no DDS."""
import math

import numpy as np
import pytest

from go2_board import go2
from go2_board.board import Board
from go2_board.bus import BoardState, MotorBus, MotorCmd
from go2_board.controllers import Pose
from go2_board.safety import STALE_S, TILT_MAX, Safety, tilt

UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


def state(q=go2.STAND, quat=UPRIGHT, stamp=0.0):
    return BoardState(q=np.array(q, float), dq=np.zeros(go2.N), quat=np.array(quat, float),
                      gyro=np.zeros(3), acc=np.zeros(3), stamp=stamp)


def tipped(rad):
    """Rolled about x by `rad`."""
    return [math.cos(rad / 2), math.sin(rad / 2), 0.0, 0.0]


class FakeBus(MotorBus):
    def __init__(self, s=None):
        self.s, self.sent = s, []

    def read(self):
        return self.s

    def write(self, cmd):
        self.sent.append(cmd)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


# -- the legs as numbered ---------------------------------------------------

def test_order_and_limits_are_the_sims():
    assert go2.NAMES[:3] == ["FR_hip", "FR_thigh", "FR_calf"]
    assert go2.NAMES[9:] == ["RL_hip", "RL_thigh", "RL_calf"]
    assert np.all(go2.Q_MIN < go2.Q_MAX)
    for pose in (go2.LIE, go2.STAND):
        assert np.all(pose >= go2.Q_MIN) and np.all(pose <= go2.Q_MAX)


# -- safety -----------------------------------------------------------------

def test_tilt():
    assert tilt(UPRIGHT) == pytest.approx(0.0)
    assert tilt(tipped(math.radians(30))) == pytest.approx(math.radians(30))


def test_passes_a_reasonable_command_unchanged():
    s = state()
    cmd = MotorCmd.hold(go2.STAND, go2.KP_STAND, go2.KD_STAND)
    out = Safety().filter(cmd, s, now=0.0)
    np.testing.assert_allclose(out.q, cmd.q)
    np.testing.assert_allclose(out.kp, cmd.kp)


def test_keeps_targets_off_the_end_stops():
    cmd = MotorCmd.hold(go2.Q_MAX + 1.0, 5.0, 1.0)
    out = Safety().filter(cmd, state(q=go2.Q_MAX - 0.1), now=0.0)
    assert np.all(out.q <= go2.Q_MAX - go2.Q_MARGIN + 1e-9)


def test_lowers_a_target_that_would_exceed_peak_torque():
    s = state()
    far = go2.STAND + 2.0                       # kp 60 * 2 rad = 120 N*m
    out = Safety().filter(MotorCmd.hold(far, 60.0, 0.0), s, now=0.0)
    est = out.kp * (out.q - s.q)
    assert np.all(np.abs(est) <= go2.TAU_MAX + 1e-6)
    assert np.all(out.q > s.q)                  # still pushing the same way


@pytest.mark.parametrize("why,s,cmd", [
    ("stale", state(stamp=0.0), MotorCmd.hold(go2.STAND, 60, 5)),
    ("tilted", state(quat=tipped(TILT_MAX + 0.1), stamp=1.0), MotorCmd.hold(go2.STAND, 60, 5)),
    ("controller failed", state(stamp=1.0), None),
])
def test_goes_limp_and_stays_limp(why, s, cmd):
    safety = Safety()
    out = safety.filter(cmd, s, now=1.0 + (STALE_S + 0.01 if why == "stale" else 0.0))
    assert safety.tripped and why.split()[0] in safety.tripped
    assert np.all(out.kp == 0) and np.all(out.kd == go2.KD_DAMP)
    # Latched: a good command and a fresh, upright state do not undo it.
    again = safety.filter(MotorCmd.hold(go2.STAND, 60, 5), state(stamp=5.0), now=5.0)
    assert np.all(again.kp == 0)
    safety.reset()
    assert safety.filter(MotorCmd.hold(go2.STAND, 60, 5), state(stamp=5.0), now=5.0).kp[0] == 60


def test_sends_nothing_before_the_first_state():
    assert Safety().filter(MotorCmd(), None, now=0.0) is None


# -- pose -------------------------------------------------------------------

def test_pose_moves_from_where_the_legs_are():
    p = Pose(go2.STAND, seconds=1.0)
    np.testing.assert_allclose(p.step(state(q=go2.LIE), 10.0).q, go2.LIE)
    np.testing.assert_allclose(p.step(state(q=go2.LIE), 10.5).q, (go2.LIE + go2.STAND) / 2)
    np.testing.assert_allclose(p.step(state(q=go2.LIE), 12.0).q, go2.STAND)
    assert p.done(11.0)


# -- the loop ---------------------------------------------------------------

class FollowingBus(FakeBus):
    """Legs that reach each target by the next state, and a state that is
    always fresh."""
    def __init__(self, s, clock):
        super().__init__(s)
        self.clock = clock

    def read(self):
        self.s.stamp = self.clock()
        return self.s

    def write(self, cmd):
        super().write(cmd)
        self.s.q = cmd.q.copy()


def test_board_runs_at_its_rate():
    clock = Clock()
    bus = FollowingBus(state(q=go2.LIE), clock)
    board = Board(bus, rate_hz=500, clock=clock, sleep=clock.sleep)
    board.run(Pose(go2.STAND, 1.0), seconds=1.0)
    assert board.stats["sent"] == pytest.approx(500, abs=1)
    np.testing.assert_allclose(bus.sent[-1].q, go2.STAND, atol=0.01)


def test_a_crashing_controller_does_not_stop_the_loop():
    clock = Clock()
    bus = FakeBus(state(stamp=0.0))
    board = Board(bus, clock=clock, sleep=clock.sleep)

    class Boom:
        def step(self, s, t):
            raise RuntimeError("policy broke")

    board.tick(Boom())
    assert "controller failed" in board.safety.tripped
    assert np.all(bus.sent[-1].kp == 0)


def test_go_limp_leaves_damping_on_the_bus():
    clock = Clock()
    bus = FakeBus(state())
    Board(bus, clock=clock, sleep=clock.sleep).go_limp(0.1)
    assert bus.sent and all(np.all(c.kp == 0) for c in bus.sent)


def test_stale_limit_can_be_raised_for_a_simulator():
    hold = MotorCmd.hold(go2.STAND, 60, 5)
    safety = Safety(stale_s=0.2)
    assert safety.filter(hold, state(stamp=1.0), now=1.1) is not None and safety.tripped is None
    safety.filter(hold, state(stamp=1.0), now=1.21)
    assert safety.tripped.startswith("stale state")
    assert Safety().stale_s == STALE_S and Board(FakeBus(), stale_s=0.2).safety.stale_s == 0.2
