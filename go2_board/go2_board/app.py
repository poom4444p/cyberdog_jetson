"""The board as a running program: bus, safety, modes and the brain link.

Each 500 Hz tick: read the brain's newest command, act on a changed mode
request, tick the motors through Safety, and every STATE_EVERY ticks tell
the brain where the dog is and what the board is doing.

With a LiDAR (go2_board.lidar.Lidar), the brain's velocity is limited by the
free distance it sees each way before the walker gets it, and the state
message carries that clearance. Without one, the velocity passes as it is.

Mode requests act on change only. The brain sends one with every command;
if the dog falls while "stand" is still arriving 20 times a second, the
board stays limp until the request changes ("limp", then "stand" again).
"""
import math

from go2_board.board import Board
from go2_board.lidar import guard
from go2_board.link import BoardLink
from go2_board.modes import Modes

STATE_EVERY = 10          # ticks between state messages: 50 Hz at 500 Hz


def yaw(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class App:
    def __init__(self, bus, link=None, walker=None, board=None, lidar=None):
        self.bus = bus
        self.lidar = lidar
        self.board = board or Board(bus)
        self.link = link or BoardLink(clock=self.board.clock)
        self.modes = Modes(walker)
        self.asked = None          # the last mode request acted on
        self.n = 0

    def step(self, state, t):
        """The controller Board ticks: Modes, after the brain's latest word."""
        cmd = self.link.poll()
        if cmd.mode != self.asked:
            self.asked = cmd.mode
            self.modes.request(cmd.mode, t)
            if self.modes.refused:
                print(f"[board] refused {cmd.mode!r}: {self.modes.refused}")
        if self.modes.wants_reset:
            self.board.safety.reset()
            self.modes.wants_reset = False
        v = self.link.velocity()
        if self.lidar is not None:
            v = guard(*v, self.lidar.clearance(), t)
            self.lidar.commanded(v[0], v[1], t)     # levelling: is the body accelerating
        self.modes.velocity = v
        return self.modes.step(state, t)

    def tick(self):
        self.board.tick(self)
        # Checked as a state, not an edge: a stand that resets safety and
        # trips it again within the same tick still leaves the board limp.
        if self.board.safety.tripped and self.modes.mode != "LIMP":
            self.modes.tripped()
        self.n += 1
        if self.n % STATE_EVERY == 0:
            self.report()

    def report(self):
        s = self.bus.read()
        if s is not None and hasattr(self.lidar, "truth"):   # the truth a LIO is judged by
            self.lidar.truth(self.board.clock(), s.pos, s.quat)
        pos = s.pos if s is not None and s.pos is not None else (None, None, None)
        self.link.send_state(
            t=self.board.clock(),
            x=None if pos[0] is None else float(pos[0]),
            y=None if pos[1] is None else float(pos[1]),
            z=None if pos[2] is None else float(pos[2]),
            yaw=None if s is None else yaw(s.quat),
            mode=self.modes.mode, tripped=self.board.safety.tripped,
            walker=self.modes.walker is not None,
            lidar=self._lidar_state())

    def _lidar_state(self):
        if self.lidar is None:
            return None
        c = self.lidar.clearance()
        return {"age": None} if c is None else c.as_dict(self.board.clock())

    def run(self, seconds=None):
        """Tick on the board's schedule until `seconds` pass (None: forever)."""
        board = self.board
        t0 = nxt = board.clock()
        while seconds is None or board.clock() - t0 < seconds:
            self.tick()
            nxt += board.dt
            wait = nxt - board.clock()
            if wait > 0:
                board.sleep(wait)
            else:
                nxt = board.clock()
