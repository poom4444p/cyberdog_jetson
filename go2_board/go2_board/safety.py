"""What the Go2's own main board did and this board now has to: never send a
command the legs cannot take, and go limp (damping) instead of sending one
it cannot trust.

Every command passes through Safety.filter() on its way to the bus. It goes
to damping, and stays there until reset(), when:
  - the state is stale: no LowState for STALE_S, so the board is blind;
  - the body has tipped past TILT_MAX: it has fallen, or is falling;
  - the controller failed (raised, or returned nothing) this tick.
Otherwise it clips each target inside the joint's range and lowers it until
the torque the driver will compute stays inside the motor's peak.
"""
import math

import numpy as np

from go2_board import go2
from go2_board.bus import MotorCmd

STALE_S = 0.05            # 25 missed states at 500 Hz
TILT_MAX = math.radians(60)


def tilt(quat):
    """Angle between the body's up axis and the world's, from w x y z."""
    w, x, y, z = quat
    up_z = 1.0 - 2.0 * (x * x + y * y)      # z component of the body's z axis
    return math.acos(max(-1.0, min(1.0, up_z)))


class Safety:
    def __init__(self):
        self.tripped = None       # why it went to damping, or None

    def reset(self):
        self.tripped = None

    def filter(self, cmd, state, now):
        """The command to send. `cmd` is None when the controller failed."""
        if state is None:
            return None                       # nothing known: send nothing yet
        if self.tripped is None:
            if now - state.stamp > STALE_S:
                self.tripped = f"stale state ({(now - state.stamp) * 1e3:.0f} ms old)"
            elif tilt(state.quat) > TILT_MAX:
                self.tripped = f"tilted {math.degrees(tilt(state.quat)):.0f} deg"
            elif cmd is None:
                self.tripped = "controller failed"
        if self.tripped is not None:
            return MotorCmd.damping(state.q)
        return self.limit(cmd, state)

    @staticmethod
    def limit(cmd, state):
        q = np.clip(cmd.q, go2.Q_MIN + go2.Q_MARGIN, go2.Q_MAX - go2.Q_MARGIN)
        tau = np.clip(cmd.tau, -go2.TAU_MAX, go2.TAU_MAX)
        # The torque the driver will compute from this target, from where the
        # joint is now. Where it is past the peak, move the target in until
        # it is not. Exact only at the instant sent; the driver keeps
        # recomputing as the joint moves, which only ever lowers it.
        est = tau + cmd.kp * (q - state.q) + cmd.kd * (cmd.dq - state.dq)
        over = np.abs(est) > go2.TAU_MAX
        stiff = cmd.kp > 1e-6
        fix = over & stiff
        if fix.any():
            want = np.sign(est[fix]) * go2.TAU_MAX[fix]
            q[fix] = state.q[fix] + (want - tau[fix]
                                     - cmd.kd[fix] * (cmd.dq[fix] - state.dq[fix])) / cmd.kp[fix]
        return MotorCmd(q=q, dq=cmd.dq.copy(), kp=cmd.kp.copy(), kd=cmd.kd.copy(), tau=tau)
