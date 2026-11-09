"""Modes, the brain link and the App, with a fake bus and real localhost UDP."""
import math
import time

import numpy as np
import pytest

from go2_board import go2
from go2_board.app import App, yaw
from go2_board.board import Board
from go2_board.bus import BoardState, MotorBus, MotorCmd
from go2_board.link import CMD_TIMEOUT_S, BoardLink, BrainLink
from go2_board.modes import STAND_UP_S, Modes

UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
FALLEN = np.array([math.cos(0.75), math.sin(0.75), 0.0, 0.0])     # rolled 86 deg


def state(q=go2.LIE, quat=UPRIGHT, stamp=0.0):
    return BoardState(q=np.array(q, float), dq=np.zeros(go2.N), quat=np.array(quat, float),
                      gyro=np.zeros(3), acc=np.zeros(3), stamp=stamp, pos=np.array([1.0, 2.0, 0.3]))


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class Legs(MotorBus):
    """Joints that reach each target by the next read; a state always fresh."""
    def __init__(self, clock, quat=UPRIGHT):
        self.clock, self.s, self.sent = clock, state(quat=quat), []

    def read(self):
        self.s.stamp = self.clock()
        return self.s

    def write(self, cmd):
        self.sent.append(cmd)
        if np.any(cmd.kp > 0):
            self.s.q = cmd.q.copy()


class Inbox:
    """BoardLink without sockets: the test sets what the brain last said."""
    def __init__(self, clock):
        from go2_board.link import Command
        self.Command, self.clock = Command, clock
        self.last, self.states, self.seq = Command(), [], 0

    def say(self, mode, vx=0.0, vy=0.0, w=0.0):
        self.seq += 1
        self.last = self.Command(vx, vy, w, mode, self.seq, self.clock())

    def poll(self):
        return self.last

    def velocity(self):
        c = self.last
        return (0.0, 0.0, 0.0) if self.clock() - c.stamp > CMD_TIMEOUT_S else (c.vx, c.vy, c.w)

    def send_state(self, **s):
        self.states.append(s)


def app(walker=None, quat=UPRIGHT):
    clock = Clock()
    bus = Legs(clock, quat)
    board = Board(bus, clock=clock, sleep=clock.sleep)
    inbox = Inbox(clock)
    return App(bus, link=inbox, walker=walker, board=board), inbox, clock


def run(a, seconds):
    for _ in range(int(seconds * 500)):
        a.tick()
        a.board.clock.sleep(a.board.dt)


class Walker:
    def __init__(self):
        self.seen = []

    def step(self, s, t, vx, vy, w):
        self.seen.append((vx, vy, w))
        return MotorCmd.hold(go2.STAND, go2.KP_STAND, go2.KD_STAND)


# -- modes ------------------------------------------------------------------

def test_starts_limp_and_stands_up_then_holds():
    a, inbox, _ = app()
    assert a.modes.mode == "LIMP"
    inbox.say("stand")
    run(a, 0.1)
    assert a.modes.mode == "STAND_UP"
    run(a, STAND_UP_S + 0.1)
    assert a.modes.mode == "STAND"
    np.testing.assert_allclose(a.bus.s.q, go2.STAND, atol=1e-6)


def test_walk_is_refused_without_a_policy():
    a, inbox, _ = app()
    inbox.say("stand")
    run(a, STAND_UP_S + 0.2)
    inbox.say("walk")
    run(a, 0.1)
    assert a.modes.mode == "STAND" and "no walking policy" in a.modes.refused


def test_walk_passes_the_brains_velocity_and_zero_after_silence():
    w = Walker()
    a, inbox, clock = app(walker=w)
    inbox.say("stand")
    run(a, STAND_UP_S + 0.2)
    inbox.say("walk", vx=0.4, w=0.2)
    run(a, 0.1)
    assert a.modes.mode == "WALK" and w.seen[-1] == (0.4, 0.0, 0.2)
    run(a, CMD_TIMEOUT_S + 0.05)                 # the brain went quiet
    assert w.seen[-1] == (0.0, 0.0, 0.0) and a.modes.mode == "WALK"


def test_lie_down_ends_limp():
    a, inbox, _ = app()
    inbox.say("stand")
    run(a, STAND_UP_S + 0.2)
    inbox.say("lie")
    run(a, 2.0)
    assert a.modes.mode == "LIMP"


def test_limp_from_any_mode_at_once():
    a, inbox, _ = app()
    inbox.say("stand")
    run(a, 0.5)
    inbox.say("limp")
    run(a, 0.01)
    assert a.modes.mode == "LIMP" and np.all(a.bus.sent[-1].kp == 0)


def test_a_fall_is_not_undone_by_a_stand_already_on_its_way():
    a, inbox, _ = app()
    inbox.say("stand")
    run(a, STAND_UP_S + 0.2)
    a.bus.s.quat = FALLEN
    inbox.say("stand")                           # the brain keeps saying stand
    run(a, 0.5)
    assert a.modes.mode == "LIMP" and "tilted" in a.board.safety.tripped
    a.bus.s.quat = UPRIGHT                       # set back on its feet
    run(a, 0.5)
    assert a.modes.mode == "LIMP"                # still limp: nothing new was asked
    inbox.say("limp")
    run(a, 0.01)
    inbox.say("stand")
    run(a, 0.1)
    assert a.modes.mode == "STAND_UP" and a.board.safety.tripped is None


def test_still_fallen_trips_again_at_once():
    a, inbox, _ = app(quat=FALLEN)
    inbox.say("stand")
    run(a, 0.1)
    assert a.modes.mode == "LIMP" and a.board.safety.tripped


def test_reports_pose_and_mode_at_50_hz():
    a, inbox, _ = app()
    run(a, 1.0)
    assert len(inbox.states) == 50
    s = inbox.states[-1]
    assert (s["x"], s["y"], s["z"]) == (1.0, 2.0, 0.3) and s["mode"] == "LIMP"
    assert s["yaw"] == pytest.approx(0.0)


def test_yaw():
    half = math.radians(90) / 2
    assert yaw([math.cos(half), 0, 0, math.sin(half)]) == pytest.approx(math.radians(90))


# -- the link, over real localhost UDP --------------------------------------

def test_link_round_trip():
    board, brain = BoardLink(port=19870, brain_port=19871), BrainLink(port=19871, board_port=19870)
    try:
        brain.send(0.3, 0.0, -0.1, "walk")
        brain.send(0.5, 0.1, 0.2, "walk")
        time.sleep(0.05)
        c = board.poll()
        assert (c.vx, c.vy, c.w, c.mode, c.seq) == (0.5, 0.1, 0.2, "walk", 2)
        assert board.velocity() == (0.5, 0.1, 0.2)
        board.send_state(mode="STAND", x=1.0)
        time.sleep(0.05)
        assert brain.poll() == {"mode": "STAND", "x": 1.0}
    finally:
        board.close()
        brain.close()


def test_link_drops_bad_datagrams_and_unknown_modes():
    import json
    import socket
    board = BoardLink(port=19872, brain_port=19873)
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for d in (b"not json", b'{"vx": 1}',
                  json.dumps({"seq": 1, "vx": 0.2, "vy": 0, "w": 0, "mode": "jump"}).encode()):
            raw.sendto(d, ("127.0.0.1", 19872))
        time.sleep(0.05)
        c = board.poll()
        assert c.seq == 1 and c.vx == 0.2 and c.mode is None
    finally:
        raw.close()
        board.close()


def test_restarted_brain_is_heard_once_the_old_one_is_quiet():
    import json
    import socket
    clock = Clock()
    board = BoardLink(port=19876, brain_port=19877, clock=clock)
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(seq, mode):
        raw.sendto(json.dumps({"seq": seq, "vx": 0, "vy": 0, "w": 0, "mode": mode}).encode(),
                   ("127.0.0.1", 19876))
        time.sleep(0.02)
        return board.poll()

    try:
        assert send(300, "walk").seq == 300
        assert send(299, "lie").seq == 300            # late datagram while talking: dropped
        clock.t += CMD_TIMEOUT_S + 0.01
        c = send(1, "limp")                           # new brain, counting from 1
        assert (c.seq, c.mode) == (1, "limp")
        assert send(2, "stand").mode == "stand"
    finally:
        raw.close()
        board.close()


def test_silence_is_zero_velocity():
    clock = Clock()
    board = BoardLink(port=19874, brain_port=19875, clock=clock)
    try:
        from go2_board.link import Command
        board.last = Command(0.5, 0.0, 0.0, "walk", 1, clock())
        assert board.velocity() == (0.5, 0.0, 0.0)
        clock.t += CMD_TIMEOUT_S + 0.01
        assert board.velocity() == (0.0, 0.0, 0.0)
    finally:
        board.close()


def test_link_between_two_addresses():
    # HIL: board and brain each on their own address (two loopback ones here;
    # Linux has all of 127/8, macOS only 127.0.0.1 unless aliased)
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.2", 0))
    except OSError:
        pytest.skip("no 127.0.0.2 on this machine")
    finally:
        probe.close()
    board = BoardLink(port=19890, brain_port=19891, listen="127.0.0.1", brain_host="127.0.0.2")
    brain = BrainLink(port=19891, board_port=19890, listen="127.0.0.2", board_host="127.0.0.1")
    try:
        brain.send(0.2, 0.0, 0.1, "walk")
        time.sleep(0.05)
        assert board.poll().vx == 0.2
        board.send_state(mode="WALK")
        time.sleep(0.05)
        assert brain.poll() == {"mode": "WALK"}
    finally:
        board.close()
        brain.close()
