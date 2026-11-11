"""BoardModelRobot (--robot twin-board): the twin turning late and short, as
the board walks the dog, and stopped by go2_board's own guard."""
import math
import os

import pytest

from cyberdog import paths
from cyberdog.sim.robot import board_model_robot as bm

SCENE = str(paths.BUILDING_SCENE)
needs_scene = pytest.mark.skipif(not os.path.exists(SCENE),
                                 reason="no building scene: build_scene --building")
TROLLEY_WEST = 23.40         # floor 1 trolley's west face (x 23.40-24.60, y 9.05-9.75)


def test_turn_gain_follows_the_fit_and_holds_at_its_ends():
    assert bm.turn_gain(0.0) == bm.TURN_GAIN[0]
    assert bm.turn_gain(-0.75) == bm.turn_gain(0.75) == bm.TURN_GAIN[-1]
    assert bm.turn_gain(2.0) == bm.TURN_GAIN[-1]
    gains = [bm.turn_gain(v / 10) for v in range(9)]
    assert gains == sorted(gains)                          # faster walking, more of the turn


def robot(xy=(15.0, 9.5), yaw=0.0, clock=1.0):
    from cyberdog.sim.scene import levels
    return bm.BoardModelRobot(SCENE, start_xy=xy, start_yaw=yaw,
                              start_z=levels.floor_z(1), clock=clock)


def drive(r, cmd, seconds):
    for _ in range(int(round(seconds / r.CONTROL_DT))):
        r.set_velocity(*cmd)
        r.step()
    return r.get_pose()


@needs_scene
class TestBoardModelRobot:
    def test_it_meets_the_guard_like_the_board(self):
        r = robot()
        assert r.GUARDED and r.MAX_W == 1.0
        assert set(r.board_clearance()) == {"front", "back", "left", "right"}

    def test_a_turn_starts_late(self):
        r = robot()
        delay = int(round(bm.TURN_DELAY / r.CONTROL_DT))
        yaws = [drive(r, (0.0, 0.0, 1.0), r.CONTROL_DT)[2] for _ in range(delay)]
        assert all(abs(y) < 1e-9 for y in yaws)

    def test_a_pivot_gets_about_a_fifth_of_the_turn(self):
        # HIL: standing, the board's dog turned at ~0.20 of the rate asked
        # (0.24 of it on the world's own clock).
        _, _, yaw = drive(robot(), (0.0, 0.0, 1.0), 2.0)
        assert 0.8 * bm.TURN_GAIN[0] * 1.8 < yaw < bm.TURN_GAIN[0] * 2.0

    def test_walking_it_turns_at_most_of_the_rate(self):
        _, _, yaw = drive(robot(), (0.75, 0.0, 0.3), 1.0)
        settled = 0.3 * bm.TURN_GAIN[-1]
        assert 0.75 * settled * 1.0 < yaw < settled * 1.0

    def test_the_guard_stops_it_short_of_the_trolley(self):
        r = robot(xy=(21.5, 9.4))                          # facing its west face, 1.9 m off
        x, _, _ = drive(r, (0.5, 0.0, 0.0), 8.0)
        from go2_board import lidar
        front = TROLLEY_WEST - x - lidar.BODY_FRONT
        assert 0.8 * lidar.STOP_M < front < lidar.SLOW_M  # stopped in the guard's band
        assert r.guarded_ticks > 0

    def test_the_hil_clock_slows_it_as_the_rig_does(self):
        x1, _, _ = drive(robot(), (0.5, 0.0, 0.0), 2.0)
        x83, _, _ = drive(robot(clock=bm.HIL_CLOCK), (0.5, 0.0, 0.0), 2.0)
        assert abs((x83 - 15.0) / (x1 - 15.0) - bm.HIL_CLOCK) < 0.01
