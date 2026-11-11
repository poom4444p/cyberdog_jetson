"""Clearance, the velocity guard, clouds to the brain and the App with a LiDAR.
No sensor: frames are built in the sensor frame from a known room."""
import math
import socket
import time

import numpy as np
import pytest

from go2_board import lidar as L
from go2_board.lidar import CloudLink, Lidar, clearance, cloud_datagrams, guard, levelling
from go2_board.livox import Frame, ImuSample

R_TS = L.mount_rotation()


def to_sensor(pts_level, pitch_extra=0.0, height=L.SENSOR_HEIGHT):
    """Levelled trunk-frame points as the sensor would report them, with the
    trunk itself pitched nose-down by pitch_extra degrees, the sensor `height`
    above the floor."""
    p = np.asarray(pts_level, float) - [L.MOUNT_X, 0.0, height]
    R_wt = L.mount_rotation(pitch_extra)          # trunk pitched, as seen from the level frame
    return np.einsum("ij,jk->ik", p, R_wt @ R_TS)  # (R_wt R_ts)^T p, row-wise


def frame(pts_sensor):
    n = len(pts_sensor)
    return Frame(np.asarray(pts_sensor, float), np.zeros(n, np.uint8), np.zeros(n, np.uint8),
                 np.zeros(n, np.int64), 0, 0, 1, 0)


def still_imu(pitch_extra=0.0, n=400):
    up = (L.mount_rotation(pitch_extra) @ R_TS).T @ [0.0, 0.0, 9.81]
    return [ImuSample(0, np.zeros(3), up) for _ in range(n)]


def floor_and_wall(wall_x=2.0):
    xs, ys = np.meshgrid(np.linspace(0.5, 6, 60), np.linspace(-2, 2, 21))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    zs, ys = np.meshgrid(np.linspace(0.0, 1.5, 16), np.linspace(-1, 1, 21))
    wall = np.stack([np.full(zs.size, wall_x), ys.ravel(), zs.ravel()], 1)
    return np.vstack([floor, wall])


class Clock:
    def __init__(self, t=10.0):
        self.t = t

    def __call__(self):
        return self.t


# -- geometry -----------------------------------------------------------------

def test_levelling_takes_up_to_z_and_keeps_heading():
    g = R_TS.T @ [0, 0, 1]
    R = levelling(g)
    assert np.allclose(R @ g, [0, 0, 1])
    fwd = R @ (R_TS.T @ [1, 0, 0])
    assert np.allclose(fwd, [1, 0, 0], atol=1e-9)


@pytest.mark.parametrize("pitch", [0.0, 6.0, -6.0])
def test_wall_ahead_and_no_floor_even_when_the_dog_pitches(pitch):
    li = Lidar(clock=Clock(), brain=None)
    li.process([], still_imu(pitch))
    li.process([frame(to_sensor(floor_and_wall(2.0), pitch))], [])
    c = li.clearance()
    assert c.front == pytest.approx(2.0 - L.BODY_FRONT, abs=0.03)
    assert math.isinf(c.back) and math.isinf(c.left) and math.isinf(c.right)


@pytest.mark.parametrize("height", [0.32, 0.50, 0.60])     # lying, Cyberdog's, tall
def test_floor_found_whatever_the_height(height):
    li = Lidar(clock=Clock(), brain=None)
    li.process([], still_imu())
    li.process([frame(to_sensor(floor_and_wall(2.5), height=height))], [])
    assert li.height == pytest.approx(height, abs=0.01)
    assert li.clearance().front == pytest.approx(2.5 - L.BODY_FRONT, abs=0.03)


def test_wall_filling_the_view_keeps_the_last_floor():
    li = Lidar(clock=Clock(), brain=None)
    li.process([], still_imu())
    li.process([frame(to_sensor(floor_and_wall(2.5), height=0.32))], [])
    zs, ys = np.meshgrid(np.linspace(0.0, 1.5, 16), np.linspace(-1, 1, 21))
    wall = np.stack([np.full(zs.size, 0.6), ys.ravel(), zs.ravel()], 1)   # nose to a wall
    li.process([frame(to_sensor(wall, height=0.32))], [])
    assert li.height == pytest.approx(0.32, abs=0.01)
    assert li.clearance().front == pytest.approx(0.6 - L.BODY_FRONT, abs=0.03)


def test_without_the_imu_a_pitching_dog_sees_the_floor_rise():
    li = Lidar(clock=Clock(), brain=None)                  # trusts the mount alone
    li.process([frame(to_sensor(floor_and_wall(5.0), 6.0))], [])
    assert li.clearance().front < 5.0 - L.BODY_FRONT - 0.5  # why the IMU is used


def test_clearance_each_way():
    pts = np.array([[1.35, 0.0, 0.5], [-0.85, 0.1, 0.3], [0.0, 0.9, 0.5], [0.2, -0.45, 0.5],
                    [0.5, 0.0, 0.05],                       # too low: floor clutter
                    [0.5, 0.0, 1.5],                        # too high: passes over
                    [0.6, 1.5, 0.5]])                       # ahead but off to the side
    c = clearance(pts, 0.0)
    assert (c.front, c.back) == pytest.approx((1.0, 0.5))
    assert (c.left, c.right) == pytest.approx((0.7, 0.25))


# -- the guard ----------------------------------------------------------------

def C(front=9.0, back=9.0, left=9.0, right=9.0, stamp=0.0):
    return L.Clearance(front, back, left, right, stamp, 0)


def test_guard_slows_then_stops_each_way():
    assert guard(0.5, 0.0, 0.3, C(front=5.0), 0.0) == (0.5, 0.0, 0.3)
    vx, _, _ = guard(0.5, 0.0, 0.0, C(front=(L.STOP_M + L.SLOW_M) / 2), 0.0)
    assert vx == pytest.approx(0.25)
    assert guard(0.5, 0.2, 0.3, C(front=0.1, left=0.1), 0.0) == (0.0, 0.0, 0.3)
    assert guard(-0.5, -0.2, 0.0, C(front=0.0, left=0.0), 0.0) == (-L.REVERSE_MAX, -0.2, 0.0)  # backing away
    assert guard(-0.5, -0.2, 0.0, C(back=0.0, right=0.0), 0.0) == (0.0, 0.0, 0.0)


def test_guard_backs_up_slowly_even_with_nothing_seen_behind():
    assert guard(-0.6, 0.0, 0.2, C(back=9.0), 0.0) == (-L.REVERSE_MAX, 0.0, 0.2)
    assert guard(-0.1, 0.0, 0.0, C(back=9.0), 0.0) == (-0.1, 0.0, 0.0)
    vx, _, _ = guard(-0.6, 0.0, 0.0, C(back=(L.STOP_M + L.SLOW_M) / 2), 0.0)
    assert vx == pytest.approx(-L.REVERSE_MAX / 2)


def test_guard_blind_stops_translation_but_not_turning():
    assert guard(0.5, 0.2, 0.3, None, 0.0) == (0.0, 0.0, 0.3)
    assert guard(0.5, 0.2, 0.3, C(stamp=0.0), L.STALE_S + 0.01) == (0.0, 0.0, 0.3)
    assert guard(0.5, 0.2, 0.3, C(stamp=0.0), L.STALE_S - 0.01) == (0.5, 0.2, 0.3)


# -- clouds to the brain --------------------------------------------------------

def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def drain(link, want=lambda c: c is not None, timeout=1.0):
    end = time.monotonic() + timeout
    c = link.poll()
    while not want(c) and time.monotonic() < end:
        time.sleep(0.005)
        c = link.poll()
    return c


def test_datagrams_fit_macos_limit():
    ds = list(cloud_datagrams(1, np.zeros((20160, 3)), 0, 0.0))
    assert len(ds) == math.ceil(20160 / L.CHUNK) and max(map(len, ds)) <= 9216


def test_cloud_to_the_brain_whole_or_not_at_all():
    port = free_port()
    link = CloudLink(port)
    li = Lidar(clock=Clock(42.0), brain=("127.0.0.1", port))
    try:
        li.process([], still_imu())
        pts = to_sensor(floor_and_wall(2.0))
        li.process([frame(pts)], [])
        c = drain(link)
        assert c is not None and c.seq == 1 and c.t == 42.0 and len(c.xyz) == len(pts)
        assert np.allclose(np.sort(c.xyz[:, 2])[[0, -1]], [0.0, 1.5], atol=1e-3)   # levelled
        # A frame missing a chunk is never handed over.
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ds = list(cloud_datagrams(7, np.ones((1500, 3)), 0, 1.0))
        for d in ds[:-1]:
            tx.sendto(d, ("127.0.0.1", port))
        for d in cloud_datagrams(8, np.ones((10, 3)), 0, 2.0):
            tx.sendto(d, ("127.0.0.1", port))
        c = drain(link, lambda c: c is not None and c.seq == 8)
        assert c.seq == 8 and link.dropped == 1
        tx.close()
    finally:
        li.stop()
        link.close()


# -- in the App -----------------------------------------------------------------

class FakeLidar:
    def __init__(self, c=None):
        self.c = c
        self.cmds = []

    def clearance(self):
        return self.c

    def commanded(self, vx, vy, t):
        self.cmds.append((vx, vy))


def test_app_limits_the_walker_and_reports_clearance():
    from test_app import Walker, app, run
    from go2_board.modes import STAND_UP_S

    w = Walker()
    a, inbox, clock = app(walker=w)
    a.lidar = FakeLidar()
    inbox.say("stand")
    run(a, STAND_UP_S + 0.2)
    inbox.say("walk", vx=0.4, w=0.2)
    run(a, 0.1)
    assert a.modes.mode == "WALK" and w.seen[-1] == (0.0, 0.0, 0.2)        # blind: turn only
    a.lidar.c = C(front=5.0, stamp=clock())
    inbox.say("walk", vx=0.4, w=0.2)
    run(a, 0.02)
    assert w.seen[-1] == (0.4, 0.0, 0.2)
    assert a.lidar.cmds[-1] == (0.4, 0.0)       # the guarded command, for levelling
    a.lidar.c = C(front=0.1, stamp=clock())
    run(a, 0.02)
    assert w.seen[-1] == (0.0, 0.0, 0.2)
    run(a, 0.05)
    lid = inbox.states[-1]["lidar"]
    assert lid["front"] == 0.1 and lid["back"] == 9.0 and lid["age"] == pytest.approx(0.05, abs=0.01)


def test_app_without_a_lidar_reports_none():
    from test_app import app, run
    a, inbox, _ = app()
    run(a, 0.05)
    assert inbox.states[-1]["lidar"] is None


def test_floor_height_without_a_usable_floor_keeps_the_last_quietly():
    import warnings
    from go2_board.lidar import floor_height
    pts = np.zeros((40, 3))
    pts[:, 0] = np.linspace(1.0, 2.5, 40)
    pts[:, 2] = np.nan                              # a patch with no usable heights
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert floor_height(pts, 0.42) == 0.42


# -- levelling a trotting dog ---------------------------------------------------

def rocking(t, amp=3.0, hz=2.0):
    """Trunk pitch (deg, nose-down) and its rate (rad/s) at t seconds."""
    w = 2 * math.pi * hz
    return amp * math.sin(w * t), math.radians(amp) * w * math.cos(w * t)


def trot_imu(t0, t1, bump=2.0, surge=1.0, gyro_bias=0.0, brake=None):
    """Mid-360 IMU samples at 200 Hz over [t0, t1) s of a rocking trunk: the gyro
    (about y, which the 20 deg mount shares with the trunk) and the specific
    force, with the body's own vertical bounce (`bump`) and forward surge
    (`surge`), m/s^2, on it. `brake` (from, seconds, m/s^2): slowing down,
    which the accelerometer reads as a push forward."""
    out = []
    for k in range(int(round((t1 - t0) * 200))):
        t = t0 + k / 200
        pitch, rate = rocking(t)
        R_ws = L.mount_rotation(pitch) @ R_TS
        ax = surge * math.sin(2 * math.pi * 2.0 * t + 1.0)
        if brake is not None and brake[0] <= t < brake[0] + brake[1]:
            ax += brake[2]
        acc = R_ws.T @ [ax, 0.0, 9.81 + bump * math.sin(2 * math.pi * 4.0 * t)]
        out.append(ImuSample(int(t * 1e9), np.array([0.0, rate + gyro_bias, 0.0]), acc))
    return out


def trot_frame(t0, slices=10):
    """One 100 ms frame of floor and wall, each slice seen at that moment's pitch."""
    pts = floor_and_wall(5.0)
    groups = np.array_split(np.arange(len(pts)), slices)
    xyz, ts = [], []
    for j, g in enumerate(groups):
        t = t0 + (j + 0.5) * 0.1 / slices
        xyz.append(to_sensor(pts[g], rocking(t)[0]))
        ts.append(np.full(len(g), int(t * 1e9), np.int64))
    xyz, ts = np.vstack(xyz), np.concatenate(ts)
    return Frame(xyz, np.zeros(len(xyz), np.uint8), np.zeros(len(xyz), np.uint8), ts,
                 int(t0 * 1e9), 0, 1, 0)


def worst_floor_tilt(li, t_end=3.0, t_from=1.0, **imu):
    """Run 1 s to settle, then frames to t_end; the worst floor tilt seen from
    t_from on, deg."""
    li.process([], trot_imu(0.0, 1.0, **imu))
    worst, t = 0.0, 1.0
    while t < t_end:
        li.process([], trot_imu(t, t + 0.1, **imu))
        f = trot_frame(t)
        pts = li.level(f.xyz, f.t)
        floor = pts[(pts[:, 0] > 1.0) & (pts[:, 0] < 4.5) & (pts[:, 2] < 0.3)]
        slope = np.polyfit(floor[:, 0], floor[:, 2], 1)[0]
        if t >= t_from:
            worst = max(worst, abs(math.degrees(math.atan(slope))))
        t += 0.1
    return worst


def test_a_trotting_dog_sees_a_level_floor():
    # 3 deg of rocking at 2 Hz, 2 m/s^2 of bounce, 1 m/s^2 of surge
    assert worst_floor_tilt(Lidar(clock=Clock(), brain=None), t_end=8.0, t_from=4.0) < 0.4


def test_gyro_bias_is_learnt():
    # 1 deg/s of bias, more than a Mid-360's: learnt, and the floor stays level
    li = Lidar(clock=Clock(), brain=None)
    assert worst_floor_tilt(li, t_end=30.0, t_from=15.0, gyro_bias=math.radians(1.0)) < 0.4
    assert math.degrees(li.tilt.bias[1]) == pytest.approx(1.0, abs=0.1)


def braking_floor_tilt(tell_the_lidar):
    """Trot at 0.65 m/s, then from 4.0 s brake at 1.6 m/s^2 for 0.4 s (HIL
    11-11 12:38, the server room); the worst floor tilt from 4 s on, deg."""
    clock = Clock(0.0)
    li = Lidar(clock=clock, brain=None)
    li.commanded(0.65, 0.0, 0.0)
    brake = (4.0, 0.4, 1.6)
    li.process([], trot_imu(0.0, 1.0, brake=brake))
    worst, t = 0.0, 1.0
    while t < 6.0:
        clock.t = t
        if tell_the_lidar and t >= 3.95:
            li.commanded(0.0, 0.0, t)
        li.process([], trot_imu(t, t + 0.1, brake=brake))
        f = trot_frame(t)
        pts = li.level(f.xyz, f.t)
        floor = pts[(pts[:, 0] > 1.0) & (pts[:, 0] < 4.5) & (pts[:, 2] < 0.3)]
        if t >= 4.0:
            worst = max(worst, abs(math.degrees(math.atan(np.polyfit(floor[:, 0], floor[:, 2], 1)[0]))))
        t += 0.1
    return worst


def test_a_braking_dog_sees_a_level_floor():
    # told the command dropped: gyro only through the braking
    assert braking_floor_tilt(True) < 0.6
    # not told (the old board): gravity correction follows the braking push
    # and the floor tilts ~1.6 deg, 0.13 m at 4.5 m -- over the brain's 0.08
    assert braking_floor_tilt(False) > 1.2


def test_only_a_real_change_of_command_holds_the_accelerometer():
    li = Lidar(clock=Clock(0.0), brain=None)
    li.commanded(0.60, 0.0, 0.0)
    li.commanded(0.65, 0.05, 1.0)               # steering noise: < CMD_DV
    assert li.hold_until == -math.inf
    li.commanded(0.40, 0.0, 2.0)
    assert li.hold_until == pytest.approx(2.0 + L.ACC_HOLD_S)
    li.commanded(0.40, 0.0, 2.5)                # steady again: the hold runs out
    assert li.hold_until == pytest.approx(2.0 + L.ACC_HOLD_S)


def test_gravity_alone_could_not_follow_the_trot():
    # the old way, for comparison: up from low-passed gravity, no gyro
    li = Lidar(clock=Clock(), brain=None)
    li.process([], trot_imu(0.0, 1.0))
    up, worst, t = li.up, 0.0, 1.0
    while t < 3.0:
        for s in trot_imu(t, t + 0.1):
            up = 0.98 * up + 0.02 * s.acc / np.linalg.norm(s.acc)
        pts = np.einsum("ij,kj->ik", trot_frame(t).xyz, levelling(up))
        floor = pts[(pts[:, 0] > 1.0) & (pts[:, 0] < 4.5) & (pts[:, 2] < -0.2)]
        slope = np.polyfit(floor[:, 0], floor[:, 2], 1)[0]
        worst = max(worst, abs(math.degrees(math.atan(slope))))
        t += 0.1
    assert worst > 1.5
