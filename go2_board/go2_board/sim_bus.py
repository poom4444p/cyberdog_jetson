"""MotorBus over DDS, in Unitree's message format: rt/lowcmd out, rt/lowstate in.

The far end is unitree_mujoco today. The messages are the ones a Go2's motor
side speaks, so the format is not ours to change; only what is behind it is.

The state is taken, newest only, by the board's own loop when it reads --
not handed over by a callback. With unitree_sdk2py's callback (a Python
listener on CycloneDDS's receive thread, then a queue and a second thread)
every LowState needed the GIL twice. On the J4012 the 500 Hz loop takes
longer than its 2 ms and never sleeps, so those threads got the GIL every
5 ms (sys.getswitchinterval): the receive thread fell behind, the socket's
213 KB buffer filled, and the walker steered on states 250-330 ms old --
measured, by matching each state it used against when the sim published
it. It fell within two seconds of every start. With no listener the receive
thread is CycloneDDS's own C and never waits for Python; the reader keeps
only the newest sample, and `read` takes it.
"""
import sys
import threading
import time

import numpy as np

if sys.platform == "darwin":
    from go2_board import _macos_timerfd  # noqa: F401  (before unitree_sdk2py)

from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.internal import InvalidSample  # noqa: E402
from cyclonedds.qos import Policy, Qos  # noqa: E402
from cyclonedds.sub import DataReader  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher  # noqa: E402
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

from go2_board import go2  # noqa: E402
from go2_board.fast_crc import lowcmd_crc  # noqa: E402
from go2_board.bus import BoardState, MotorBus  # noqa: E402

DOMAIN = 0                # the real Go2 and Unitree's examples use 0
NEWEST = Qos(Policy.History.KeepLast(1))   # a reader holds the latest sample, no backlog
TAKE_N = 32               # taken at once, the newest kept, should a reader hold more


def _newest(reader):
    """The newest valid sample a reader holds, taking all of them; or None."""
    got = [s for s in reader.take(N=TAKE_N) if not isinstance(s, InvalidSample)]
    return got[-1] if got else None


def to_state(msg, stamp, pos=None):
    ms, imu = msg.motor_state, msg.imu_state
    return BoardState(
        q=np.array([ms[i].q for i in range(go2.N)]),
        dq=np.array([ms[i].dq for i in range(go2.N)]),
        quat=np.array(imu.quaternion, float),
        gyro=np.array(imu.gyroscope, float),
        acc=np.array(imu.accelerometer, float),
        stamp=stamp,
        pos=pos)


class SimBus(MotorBus):
    def __init__(self, interface="lo0" if sys.platform == "darwin" else "lo"):
        ChannelFactoryInitialize(DOMAIN, interface)   # the domain, on that interface
        self._lock = threading.Lock()
        self._state = None
        self._crc = CRC()
        self._msg = unitree_go_msg_dds__LowCmd_()
        self._msg.head[0], self._msg.head[1] = 0xFE, 0xEF
        self._msg.level_flag = 0xFF        # low level
        for m in self._msg.motor_cmd:
            m.mode = 0x01                  # PMSM, servo
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        dp = DomainParticipant(DOMAIN)
        self._low = DataReader(dp, Topic(dp, "rt/lowstate", LowState_), qos=NEWEST)
        # The sim's body position. It is the simulator's ground truth, not an
        # estimate: on the real dog this comes from FAST-LIO instead.
        self._pos = None
        self._high = DataReader(dp, Topic(dp, "rt/sportmodestate", SportModeState_), qos=NEWEST)

    def read(self):
        """The newest state, taken now; the last one again if none has come
        since (its stamp then says how old it is, for Safety's stale check)."""
        with self._lock:
            high = _newest(self._high)
            if high is not None:
                self._pos = np.array(high.position, float)
            msg = _newest(self._low)
            if msg is not None:
                self._state = to_state(msg, time.monotonic(), self._pos)
            return self._state

    def write(self, cmd):
        for i, m in enumerate(self._msg.motor_cmd[:go2.N]):
            m.q, m.dq = float(cmd.q[i]), float(cmd.dq[i])
            m.kp, m.kd, m.tau = float(cmd.kp[i]), float(cmd.kd[i]), float(cmd.tau[i])
        self._msg.crc = lowcmd_crc(self._crc, self._msg)
        self._pub.Write(self._msg)

    def wait_for_state(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.read() is None:
            if time.monotonic() > end:
                raise TimeoutError("no rt/lowstate: is the sim running, on this interface?")
            time.sleep(0.01)
        return self.read()
