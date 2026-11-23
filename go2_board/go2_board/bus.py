"""The line between the board and the motors.

Everything above it (the board loop, safety, the walking policy, the brain)
reads BoardState and writes MotorCmd, and never knows what is on the other
end: SimBus talks DDS to unitree_mujoco, and a RealBus will talk straight to
the Go2's motor drivers. Each driver closes its own loop on
    tau + kp * (q - q_measured) + kd * (dq - dq_measured)
so a command is a target and two gains per joint, not a torque.
"""
from dataclasses import dataclass, field

import numpy as np

from go2_board import go2


@dataclass
class BoardState:
    q: np.ndarray            # (12,) joint angles, rad
    dq: np.ndarray           # (12,) joint speeds, rad/s
    quat: np.ndarray         # (4,) body orientation, w x y z
    gyro: np.ndarray         # (3,) body rates, rad/s
    acc: np.ndarray          # (3,) m/s^2
    stamp: float             # time.monotonic() when it arrived
    pos: np.ndarray = None   # (3,) body position in the odometry frame, if the bus has one


@dataclass
class MotorCmd:
    q: np.ndarray = field(default_factory=lambda: np.zeros(go2.N))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(go2.N))
    kp: np.ndarray = field(default_factory=lambda: np.zeros(go2.N))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(go2.N))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(go2.N))

    @classmethod
    def hold(cls, q, kp, kd):
        n = go2.N
        return cls(q=np.asarray(q, float).copy(), kp=np.full(n, kp), kd=np.full(n, kd))

    @classmethod
    def damping(cls, q):
        """No stiffness, only damping: the legs fold under the body's weight,
        slowly. What the board sends whenever it cannot trust what it would
        otherwise send. A NaN angle is sent as 0: kp is 0, but 0 * NaN is
        still NaN in the driver's sum."""
        return cls(q=np.nan_to_num(np.asarray(q, float), nan=0.0, posinf=0.0, neginf=0.0),
                   kd=np.full(go2.N, go2.KD_DAMP))


class MotorBus:
    def read(self):
        """The latest BoardState, or None before the first one arrives."""
        raise NotImplementedError

    def write(self, cmd):
        """Send one MotorCmd to all 12 motors."""
        raise NotImplementedError

    def close(self):
        pass
