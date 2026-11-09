"""What decides the motor targets each tick, below the brain.

A controller is step(state, t) -> MotorCmd, called by the board at 500 Hz.
Pose moves the legs smoothly from wherever they are to a pose and holds it
(stand up, lie down). The walking policy will be one more controller, taking
vx, vy, w from the brain.
"""
import numpy as np

from go2_board import go2
from go2_board.bus import MotorCmd


class Pose:
    """Move from the joints' angles at the first step to `target` over
    `seconds`, then hold it. Linear in joint space, as Unitree's example."""

    def __init__(self, target, seconds, kp=go2.KP_STAND, kd=go2.KD_STAND):
        self.target = np.asarray(target, float)
        self.seconds = float(seconds)
        self.kp, self.kd = kp, kd
        self.start = None
        self.t0 = None

    def step(self, state, t):
        if self.start is None:
            self.start, self.t0 = state.q.copy(), t
        a = 1.0 if self.seconds <= 0 else min(1.0, (t - self.t0) / self.seconds)
        return MotorCmd.hold((1 - a) * self.start + a * self.target, self.kp, self.kd)

    def done(self, t):
        return self.t0 is not None and t - self.t0 >= self.seconds
