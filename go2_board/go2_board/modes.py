"""What the board is doing, and what it may do next.

    LIMP --stand--> STAND_UP --(done)--> STAND --walk--> WALK
      ^                                    |  ^            |
      |                                    |  +---stand----+
      +----(done)---- LIE_DOWN <---lie-----+
    any mode --limp--> LIMP                (the person's e-stop, or the brain's)

Safety tripping puts the board in LIMP; only an explicit "stand" from LIMP
resets it, so a dog that fell is never stood up again by a command that was
already on its way. WALK needs a walker (the walking policy); without one a
"walk" request is refused and the dog stays standing.
"""
from go2_board import go2
from go2_board.bus import MotorCmd
from go2_board.controllers import Pose

STAND_UP_S = 1.5
LIE_DOWN_S = 1.5


class Modes:
    def __init__(self, walker=None):
        self.walker = walker          # step(state, t, vx, vy, w) -> MotorCmd
        self.mode = "LIMP"
        self.pose = None
        self.velocity = (0.0, 0.0, 0.0)
        self.refused = None           # why the last request was not followed
        self.wants_reset = False      # the board should reset safety before the next tick

    def request(self, want, t):
        """Act on what the brain (or the person) asks for. Unknown or None: no change."""
        self.refused = None
        m = self.mode
        if want == "limp":
            self._enter("LIMP")
        elif want == "stand":
            if m == "LIMP":
                self.wants_reset = True
                self._enter("STAND_UP", Pose(go2.STAND, STAND_UP_S))
            elif m == "WALK":
                self._enter("STAND", Pose(go2.STAND, 0.0))
        elif want == "walk":
            if m == "STAND":
                if self.walker is None:
                    self.refused = "no walking policy loaded"
                else:
                    self._enter("WALK")
            elif m != "WALK":
                self.refused = f"cannot walk from {m}: stand first"
        elif want == "lie":
            if m in ("STAND", "WALK"):
                self._enter("LIE_DOWN", Pose(go2.LIE, LIE_DOWN_S))

    def tripped(self):
        """Safety went to damping: the board is limp whatever was asked."""
        self._enter("LIMP")

    def _enter(self, mode, pose=None):
        if mode != self.mode or pose is not None:
            self.mode, self.pose = mode, pose
            if mode == "WALK" and hasattr(self.walker, "reset"):
                self.walker.reset()

    def step(self, state, t):
        """The controller the board ticks. Moves on from timed modes itself."""
        if self.mode == "LIMP":
            return MotorCmd.damping(state.q)
        if self.mode == "WALK":
            return self.walker.step(state, t, *self.velocity)
        cmd = self.pose.step(state, t)
        if self.pose.done(t):
            if self.mode == "STAND_UP":
                self._enter("STAND", Pose(go2.STAND, 0.0))
            elif self.mode == "LIE_DOWN":
                self._enter("LIMP")
        return cmd
