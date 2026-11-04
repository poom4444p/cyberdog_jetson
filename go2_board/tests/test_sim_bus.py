"""SimBus over real DDS on the loopback: the newest state, never a backlog."""
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from go2_board.sim_bus import SimBus
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import (unitree_go_msg_dds__LowState_,
                                        unitree_go_msg_dds__SportModeState_)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_


@pytest.fixture(scope="module")
def bus():
    return SimBus()            # loopback: lo0 on macOS, lo on Linux


@pytest.fixture(scope="module")
def sim(bus):
    pub = ChannelPublisher("rt/lowstate", LowState_)
    pub.Init()
    msg = unitree_go_msg_dds__LowState_()

    def send(k):
        msg.motor_state[0].q = float(k)
        pub.Write(msg)

    # discovery: until the bus sees one
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        send(-1)
        s = bus.read()
        if s is not None and s.q[0] == -1:
            return send
        time.sleep(0.02)
    pytest.skip("no DDS on the loopback here")


def test_one_read_gets_the_newest_not_the_oldest_waiting(bus, sim):
    # 200 states while the board is busy elsewhere and not reading -- what
    # the J4012's loop did to the old callback, which then worked through
    # a queue of them and steered on states 0.3 s old.
    for k in range(200):
        sim(k)
    time.sleep(0.2)
    assert bus.read().q[0] == 199


def test_no_new_state_returns_the_last_with_its_own_stamp(bus, sim):
    sim(500)
    time.sleep(0.1)
    s = bus.read()
    assert s.q[0] == 500
    time.sleep(0.05)
    again = bus.read()                   # nothing sent since
    assert again is s and again.stamp == s.stamp   # so Safety sees it age


def test_the_body_position_rides_with_the_state(bus, sim):
    pub = ChannelPublisher("rt/sportmodestate", SportModeState_)
    pub.Init()
    high = unitree_go_msg_dds__SportModeState_()
    high.position = [1.5, -2.0, 0.3]
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        pub.Write(high)
        sim(7)
        time.sleep(0.02)
        s = bus.read()
        if s.pos is not None:
            break
    assert list(s.pos) == pytest.approx([1.5, -2.0, 0.3])


# A sim in another process, as the real one is: LowState at 500 Hz carrying
# in q[0] the seconds since its own start, which it prints first.
PUBLISHER = textwrap.dedent("""
    import sys, time
    if sys.platform == "darwin":
        from go2_board import _macos_timerfd  # noqa: F401
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    ChannelFactoryInitialize(0, "lo0" if sys.platform == "darwin" else "lo")
    pub = ChannelPublisher("rt/lowstate", LowState_); pub.Init()
    msg = unitree_go_msg_dds__LowState_()
    t0 = time.time(); print(t0, flush=True)
    nxt = time.monotonic()
    while time.time() - t0 < float(sys.argv[1]):
        msg.motor_state[0].q = time.time() - t0
        pub.Write(msg)
        nxt += 0.002
        time.sleep(max(0.0, nxt - time.monotonic()))
""")


def test_a_loop_that_never_sleeps_still_gets_fresh_states(bus):
    # The J4012's board loop: each tick's work longer than its 2 ms, so it
    # never sleeps and holds the GIL but for the 5 ms switch interval. The
    # old callback, starved of the GIL, let the socket fill: states 0.3 s old.
    p = subprocess.Popen([sys.executable, "-c", PUBLISHER, "4"], stdout=subprocess.PIPE, text=True)
    try:
        t0 = float(p.stdout.readline())
        end = time.monotonic() + 3.0
        while (bus.read() is None or bus.read().q[0] < 0.2) and time.monotonic() < end:
            time.sleep(0.01)                          # discovery
        ages = []
        stop = time.monotonic() + 2.0
        while time.monotonic() < stop:
            busy = time.perf_counter() + 0.003         # 3 ms of Python, no sleep
            while time.perf_counter() < busy:
                pass
            s = bus.read()
            ages.append((time.time() - t0) - s.q[0])  # how long ago the sim sent it
        ages = np.array(ages[50:])                     # after the first 0.15 s
        assert np.percentile(ages, 99) < 0.02, f"states {np.median(ages) * 1e3:.0f} ms old"
    finally:
        p.kill()
