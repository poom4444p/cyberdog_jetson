"""MotorBus over DDS, in Unitree's message format: rt/lowcmd out, rt/lowstate in.

The far end is unitree_mujoco today. The messages are the ones a Go2's motor
side speaks, so the format is not ours to change; only what is behind it is.
"""
import sys
import threading
import time

import numpy as np

if sys.platform == "darwin":
    from go2_board import _macos_timerfd  # noqa: F401  (before unitree_sdk2py)

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,  # noqa: E402
                                         ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

from go2_board import go2  # noqa: E402
from go2_board.bus import BoardState, MotorBus  # noqa: E402

DOMAIN = 0                # the real Go2 and Unitree's examples use 0


class SimBus(MotorBus):
    def __init__(self, interface="lo0" if sys.platform == "darwin" else "lo"):
        ChannelFactoryInitialize(DOMAIN, interface)
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
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        ms, imu = msg.motor_state, msg.imu_state
        state = BoardState(
            q=np.array([ms[i].q for i in range(go2.N)]),
            dq=np.array([ms[i].dq for i in range(go2.N)]),
            quat=np.array(imu.quaternion, float),
            gyro=np.array(imu.gyroscope, float),
            acc=np.array(imu.accelerometer, float),
            stamp=time.monotonic())
        with self._lock:
            self._state = state

    def read(self):
        with self._lock:
            return self._state

    def write(self, cmd):
        for i, m in enumerate(self._msg.motor_cmd[:go2.N]):
            m.q, m.dq = float(cmd.q[i]), float(cmd.dq[i])
            m.kp, m.kd, m.tau = float(cmd.kp[i]), float(cmd.kd[i]), float(cmd.tau[i])
        self._msg.crc = self._crc.Crc(self._msg)
        self._pub.Write(self._msg)

    def wait_for_state(self, timeout=5.0):
        end = time.monotonic() + timeout
        while self.read() is None:
            if time.monotonic() > end:
                raise TimeoutError("no rt/lowstate: is the sim running, on this interface?")
            time.sleep(0.01)
        return self.read()
