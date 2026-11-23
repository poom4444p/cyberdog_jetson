"""What the Go2's own main board did and this board now has to: never send a
command the legs cannot take, and go limp (damping) instead of sending one
it cannot trust.

Every command passes through Safety.filter() on its way to the bus. It goes
to damping, and stays there until reset(), when:
  - the state is stale: no LowState for STALE_S, so the board is blind;
  - the body has tipped past TILT_MAX: it has fallen, or is falling;
  - the controller failed (raised, or returned nothing) this tick;
  - the state or the command holds a NaN or an inf: np.clip passes a NaN
    through, so without this check it would go to the motors as a target.
Otherwise it clips each target inside the joint's range and lowers it until
the torque the driver will compute stays inside the motor's peak; where the
two disagree (a joint running fast at a stop), the range wins.
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


def finite(*arrays):
    """True when every number in every array is finite (no NaN, no inf)."""
    return all(np.all(np.isfinite(a)) for a in arrays)


class Safety:
    def __init__(self, stale_s=STALE_S):
        self.tripped = None       # why it went to damping, or None
        # Only a simulator on a busy desktop needs more (board_main --stale-ms):
        # the motors' own 500 Hz never pauses like a Mac running four sims.
        self.stale_s = stale_s

    def reset(self):
        self.tripped = None

    def filter(self, cmd, state, now):
        """The command to send. `cmd` is None when the controller failed."""
        if state is None:
            return None                       # nothing known: send nothing yet
        if self.tripped is None:
            if now - state.stamp > self.stale_s:
                self.tripped = f"stale state ({(now - state.stamp) * 1e3:.0f} ms old)"
            elif not finite(state.q, state.dq, state.quat):
                self.tripped = "state not finite"
            elif tilt(state.quat) > TILT_MAX:
                self.tripped = f"tilted {math.degrees(tilt(state.quat)):.0f} deg"
            elif cmd is None:
                self.tripped = "controller failed"
            elif not finite(cmd.q, cmd.dq, cmd.kp, cmd.kd, cmd.tau):
                self.tripped = "command not finite"
        if self.tripped is not None:
            return MotorCmd.damping(state.q)
        return self.limit(cmd, state)

    @staticmethod
    def limit(cmd, state):
        lo, hi = go2.Q_MIN + go2.Q_MARGIN, go2.Q_MAX - go2.Q_MARGIN
        q = np.clip(cmd.q, lo, hi)
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
            # A joint running fast at a stop: its damping alone is past the
            # peak, and lowering that would put the target beyond the stop.
            # The range wins; the motor caps its own torque at the peak.
            q = np.clip(q, lo, hi)
        return MotorCmd(q=q, dq=cmd.dq.copy(), kp=cmd.kp.copy(), kd=cmd.kd.copy(), tau=tau)
