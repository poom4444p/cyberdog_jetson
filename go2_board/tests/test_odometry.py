"""go2_board.odometry: the LiDAR's own pose, and where it sits in the bus frame."""
import math

import numpy as np
import pytest

from go2_board.odometry import LioOdometry, PoseAnchor


def test_the_first_pose_is_anchored_to_the_bus_and_moves_with_the_odometry():
    a = PoseAnchor()
    assert a.place((0.0, 0.0, 0.0), None) is None                  # no bus pose yet: nothing to say
    assert a.place((0.0, 0.0, 0.0), (2.0, 1.0, 0.3, math.pi / 2)) == pytest.approx((2.0, 1.0, math.pi / 2))
    # One metre forward in the odometry's frame is one metre along the bus heading (+y).
    x, y, yaw = a.place((1.0, 0.0, 0.0), (2.0, 2.0, 0.3, math.pi / 2))
    assert (x, y) == pytest.approx((2.0, 2.0)) and yaw == pytest.approx(math.pi / 2)
    assert a.anchors == 1 and a.err[-1][0] == pytest.approx(0.0, abs=1e-9)


def test_the_odometry_drifts_and_the_error_says_so():
    a = PoseAnchor()
    a.place((0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0))
    a.place((10.2, 0.0, 0.05), (10.0, 0.0, 0.1, 0.0))
    xy, dyaw = a.err[-1]
    assert xy == pytest.approx(0.2) and dyaw == pytest.approx(0.05)
    assert "xy end 0.20 m" in a.summary()


def test_a_floor_change_anchors_again_once_the_height_settles():
    # The lift: the car rises with the dog in it, the LiDAR sees no motion. Not
    # anchored on the way up (it was, every metre: 4 times in one ride), but
    # once out of the car, the height steady for SETTLE_S.
    from go2_board.odometry import SETTLE_S
    a = PoseAnchor()
    a.place((0.0, 0.0, 0.0), (0.0, 0.0, 0.1, 0.0), t=0.0)
    a.place((0.5, 0.0, 0.0), (0.5, 0.0, 0.1, 0.0), t=1.0)          # same floor: no new anchor
    for k in range(40):                                              # 4.4 m up in 4 s
        a.place((0.6, 0.1, 0.2), (1.0, 0.0, 0.1 + 0.11 * (k + 1), 0.0), t=2.0 + 0.1 * k)
    assert a.anchors == 1                                            # never in the car
    t = 6.0
    while t < 6.0 + SETTLE_S + 0.2:                                  # out, height steady
        out = a.place((0.6, 0.1, 0.2), (4.0, -5.0, 4.5, -math.pi / 2), t=t)
        t += 0.1
    assert a.anchors == 2 and out == pytest.approx((4.0, -5.0, -math.pi / 2))


def test_app_reports_the_odometry_pose_and_keeps_the_bus_height():
    from test_app import app
    a, inbox, clock = app()

    class Odo:
        def odometry(self):
            return (0.5, 0.0, 0.0, clock(), 3.0)

        def clearance(self):
            return None

    a.lidar, a.anchor = Odo(), PoseAnchor()
    a.report()                                       # bus at (1.0, 2.0, 0.3), yaw 0: the anchor
    s = inbox.states[-1]
    assert (s["x"], s["y"], s["yaw"], s["z"]) == pytest.approx((1.0, 2.0, 0.0, 0.3))


def test_no_odometry_yet_means_no_pose():
    from test_app import app
    a, inbox, clock = app()

    class Blind:
        def odometry(self):
            return None

        def clearance(self):
            return None

    a.lidar, a.anchor = Blind(), PoseAnchor()
    a.report()
    assert inbox.states[-1]["x"] is None                    # the brain waits; it is not handed the bus's


def test_lio_odometry_follows_a_room_it_walks_through():
    """Only where small_gicp is installed (the board's go2env; not unitree_sim)."""
    odo = LioOdometry()
    if not odo.available:
        pytest.skip("small_gicp not installed")
    rng = np.random.default_rng(0)
    # A room 8 x 6 m, walls and floor, seen from a body walking +x 0.1 m a sweep.
    walls = np.concatenate([
        np.column_stack([rng.uniform(-4, 4, 3000), np.full(3000, 3.0), rng.uniform(-0.5, 1.5, 3000)]),
        np.column_stack([rng.uniform(-4, 4, 3000), np.full(3000, -3.0), rng.uniform(-0.5, 1.5, 3000)]),
        np.column_stack([np.full(2000, 4.0), rng.uniform(-3, 3, 2000), rng.uniform(-0.5, 1.5, 2000)]),
        np.column_stack([np.full(2000, -4.0), rng.uniform(-3, 3, 2000), rng.uniform(-0.5, 1.5, 2000)]),
        np.column_stack([rng.uniform(-4, 4, 3000), rng.uniform(-3, 3, 3000), np.full(3000, -0.5)])])
    T_sb = odo.T_sb
    for k in range(10):
        body = np.eye(4)
        body[0, 3] = 0.1 * k - 1.0
        T_ws = body @ np.linalg.inv(T_sb)                    # sensor in the room
        pts = (np.linalg.inv(T_ws) @ np.column_stack([walls, np.ones(len(walls))]).T).T[:, :3]
        P = odo.add(pts, np.arange(len(pts)) * 1000 + k * 100_000_000)
    assert P[0, 3] == pytest.approx(0.9, abs=0.05) and abs(P[1, 3]) < 0.05
