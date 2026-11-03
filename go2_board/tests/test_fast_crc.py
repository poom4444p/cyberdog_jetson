"""fast_crc gives the SDK's own CRC, bit for bit, and fast."""
import random
import sys
import time

import pytest

if sys.platform == "darwin":
    from go2_board import _macos_timerfd  # noqa: F401
pytest.importorskip("unitree_sdk2py")
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

from go2_board.fast_crc import crc32_words, lowcmd_crc  # noqa: E402


def test_matches_the_sdk_on_random_words():
    sdk, rng = CRC(), random.Random(0)
    for n in (1, 2, 7, 202):
        words = [rng.getrandbits(32) for _ in range(n)]
        assert crc32_words(words) == sdk._crc_py(words)


def test_matches_the_sdk_on_lowcmds():
    sdk, rng = CRC(), random.Random(1)
    for _ in range(20):
        m = unitree_go_msg_dds__LowCmd_()
        m.head[0], m.head[1], m.level_flag = 0xFE, 0xEF, 0xFF
        for c in m.motor_cmd[:12]:
            c.mode = 1
            c.q, c.dq, c.tau = (rng.uniform(-3, 3) for _ in range(3))
            c.kp, c.kd = rng.uniform(0, 60), rng.uniform(0, 5)
        assert lowcmd_crc(sdk, m) == sdk._crc_py(sdk._CRC__PackLowCmd(m))


def test_fits_the_tick():
    sdk, m = CRC(), unitree_go_msg_dds__LowCmd_()
    t = time.perf_counter()
    for _ in range(200):
        lowcmd_crc(sdk, m)
    assert (time.perf_counter() - t) / 200 < 0.3e-3     # of a 2 ms tick; the SDK's loop took 1.5 ms
