"""The floor-1 trolley with somebody standing in its 1.10 m north gap: wait a
while for them, then through the 0.90 m south gap lined up, so the board's
guard (0.30 m either side of the heading) lets the dog through."""
import math
from types import SimpleNamespace

from cyberdog.sim import run_building as rb
from cyberdog.sim.run_building import axis_clear, narrow_aim, narrow_lined_up, small_thing_beside

# Trolley x 23.4-24.6, y 9.05-9.75; walls y 8.15 and 10.85. The crossing
# free_destination chose: 0.45 m south of the pinch beside the trolley.
TROLLEY = (23.4, 24.6, 9.05, 9.75)
INFO = dict(pinch=(24.0, 9.0), normal=(0.0, 1.0), along=(1.0, 0.0), side=-1.0,
            want=0.40, room=0.45)


def detected_at(x, y):
    """Room from what the LiDAR found and the map did not explain: the trolley.
    The walls are the map's (perception drops returns it explains)."""
    x0, x1, y0, y1 = TROLLEY
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def person(x, y, r=0.17):
    return SimpleNamespace(x=x, y=y, r=r, moving=False, walked=False)


def open_floor(x, y):
    return 1.0


def test_somebody_standing_in_the_north_gap_is_found():
    assert small_thing_beside(INFO, [person(23.88, 10.15)], open_floor) is not None   # HIL 12:48


def test_not_a_crate_not_the_trolley_not_on_the_chosen_side():
    assert small_thing_beside(INFO, [person(23.9, 10.15, r=0.6)], open_floor) is None   # too big
    assert small_thing_beside(INFO, [person(23.9, 8.4)], open_floor) is None            # the narrow side
    assert small_thing_beside(INFO, [person(27.0, 10.15)], open_floor) is None          # well past it
    assert small_thing_beside(INFO, [person(23.9, 10.15)], lambda x, y: 0.0) is None    # in a wall


def test_from_the_start_it_lines_up_before_the_gap():
    aim = narrow_aim(INFO, (20.9, 9.5), detected_at)
    assert abs(aim[1] - 8.6) < 0.01                                  # on the axis
    assert aim[0] < TROLLEY[0]                                       # before the entrance


def test_lined_up_it_goes_straight_through():
    aim = narrow_aim(INFO, (22.6, 8.62), detected_at)
    assert abs(aim[1] - 8.6) < 0.01 and abs(aim[0] - (22.6 + rb.ALIGN_LOOK)) < 0.01


def test_the_line_to_the_aim_is_gentle_when_there_is_room():
    for xy in [(20.9, 9.5), (21.0, 7.6), (19.0, 10.3)]:            # 2+ m before the entrance
        aim = narrow_aim(INFO, xy, detected_at)
        assert abs(aim[1] - 8.6) < 0.01                              # on the axis
        assert aim[0] > xy[0]                                        # ahead of the dog
        slope = math.degrees(math.atan2(abs(aim[1] - xy[1]), aim[0] - xy[0]))
        assert slope <= rb.ALIGN_MAX_DEG + 1e-6


def test_close_to_the_entrance_the_axis_is_reached_by_it():
    # 35 deg from (22.4, 9.6) would land on the axis at x 23.83, past the
    # trolley's corner (23.4): the aim is the entrance instead, steeper.
    _, _, _, to_entrance = rb.narrow_axis(INFO, (22.4, 9.6), detected_at)
    aim = narrow_aim(INFO, (22.4, 9.6), detected_at)
    assert abs(aim[1] - 8.6) < 0.01
    assert abs(aim[0] - (22.4 + max(to_entrance, rb.ALIGN_LOOK))) < 1e-9
    assert aim[0] < 22.4 + 1.0 / math.tan(math.radians(rb.ALIGN_MAX_DEG)) - 0.3


def test_the_wait_is_long_enough_for_somebody_to_move_and_not_much_more():
    assert rb.PATIENCE < rb.NARROW_WAIT_S <= 10.0
    assert 0.45 < rb.NARROW_ROOM <= 0.50          # the south gap narrow, the north not


def with_person(px, py, r=0.17):
    def at(x, y):
        return min(detected_at(x, y), max(0.0, math.hypot(x - px, y - py) - r))
    return at


def test_the_south_gap_axis_is_clear():
    assert axis_clear(INFO, detected_at)


def test_a_crossing_beside_somebody_further_along_is_not_a_way():
    # 12:57: the north crossing (23.38, 10.32) read 0.46 m; they stood at (23.88, 10.15)
    north = dict(INFO, pinch=(23.38, 9.32), side=1.0, want=1.0, room=0.46)
    assert axis_clear(north, detected_at)                      # nobody there: fine
    assert not axis_clear(north, with_person(23.88, 10.15))    # them: shut


# The dog as the board walks it, closely enough to show the 13:14 hold: turns
# at TURN_GAIN of what it is told, TURN_LAG s late (go2_flat_v5 via the board:
# 44-75 %, 0.5-1 s); the board's guard looks BODY_FRONT + STOP_M..SLOW_M ahead
# in a corridor BODY_HALF_W + MARGIN either side, and slows, then stops, the
# forward command. Walls y 8.15 and 10.85 (the map's), and the trolley.
TURN_GAIN, TURN_LAG = 0.6, 0.5
BODY_FRONT, HALF_W, STOP_M, SLOW_M = 0.35, 0.30, 0.30, 1.00
SIM_DT = 0.05
OBSTACLES = ([(x / 50, 8.15) for x in range(950, 1500)]
             + [(x / 50, 10.85) for x in range(950, 1500)]
             + [(TROLLEY[0] + k * 0.02, TROLLEY[2]) for k in range(61)]
             + [(TROLLEY[0], TROLLEY[2] + k * 0.02) for k in range(36)])


def board_front(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    free = rb.INF
    for ox, oy in OBSTACLES:
        fwd = (ox - x) * c + (oy - y) * s
        lat = -(ox - x) * s + (oy - y) * c
        if fwd > BODY_FRONT and abs(lat) < HALF_W:
            free = min(free, fwd - BODY_FRONT)
    return free


def walk_through(x, y, yaw, seconds=20.0):
    """Steer by narrow_aim / narrow_lined_up as run_building does (PASS_V,
    K_W, TURN_ONLY), through the board, until past the trolley. Returns the
    path [(t, x, y, yaw, forward command, board front)]."""
    from cyberdog.sim.control import K_W, TURN_ONLY
    w, path, t = 0.0, [], 0.0
    while t < seconds and x < TROLLEY[1] + 0.5:
        aim = narrow_aim(INFO, (x, y), detected_at)
        speed = rb.PASS_V if narrow_lined_up(INFO, (x, y), yaw, detected_at) else rb.ALIGN_SLOW_V
        err = rb.wrap(math.atan2(aim[1] - y, aim[0] - x) - yaw)
        v = 0.0 if abs(err) > TURN_ONLY else speed * math.cos(err)
        free = board_front(x, y, yaw)
        v *= min(1.0, max(0.0, (free - STOP_M) / (SLOW_M - STOP_M)))
        w += min(1.0, SIM_DT / TURN_LAG) * (TURN_GAIN * K_W * err - w)
        path.append((t, x, y, yaw, v, free))
        x += v * math.cos(yaw) * SIM_DT
        y += v * math.sin(yaw) * SIM_DT
        yaw += w * SIM_DT
        t += SIM_DT
    return path


def at_entrance(path, x_entrance=TROLLEY[0]):
    return next(p for p in path if p[1] >= x_entrance)


def test_from_the_1314_start_it_curves_onto_the_axis_without_a_hold():
    # HIL 11-10 13:14: done waiting at (21.0, 9.5), heading -0.28.
    path = walk_through(21.0, 9.5, -0.28)
    assert path[-1][1] > TROLLEY[1]                                  # through, in time
    _, x, y, yaw, _, _ = at_entrance(path)
    assert abs(y - 8.6) <= rb.ALIGN_TOL                              # on the axis...
    assert abs(math.degrees(yaw)) <= rb.ALIGN_YAW_DEG                # ...along it
    assert min(p[5] for p in path) > STOP_M                          # the guard never stops it
    assert all(p[4] > 0.0 for p in path[5:])                         # and it never pivots


def test_from_wherever_it_waits_no_sharp_cut():
    # The 13:23 case (a lining-up point far back cut down at -72 deg), the
    # lane, and further north. Not from much closer: 1 m off the axis and
    # under 1 m before the entrance there is no curve the board's turning
    # makes, and the old lining-up point did no better (stuck from every
    # start in this model); the dog waits where it first sees the gap is
    # narrow, 2-3 m back.
    for start in [(20.9, 9.5, 0.0), (20.9, 9.5, 0.3), (19.5, 8.87, 0.0),
                  (20.5, 9.9, 0.0), (20.0, 10.3, 0.0)]:
        path = walk_through(*start)
        assert path[-1][1] > TROLLEY[1], start
        assert min(p[5] for p in path) > STOP_M, start
        assert max(abs(p[3]) for p in path) < math.radians(45), start   # no -72 deg cut
        _, x, y, yaw, _, _ = at_entrance(path)
        assert abs(y - 8.6) <= rb.ALIGN_TOL, (start, y)
        assert abs(math.degrees(yaw)) <= rb.ALIGN_YAW_DEG, (start, yaw)


def test_not_lined_up_near_the_entrance_it_slows():
    assert narrow_lined_up(INFO, (20.0, 9.5), -0.5, detected_at)            # far off: no matter
    assert narrow_lined_up(INFO, (22.8, 8.62), 0.02, detected_at)           # lined up
    assert not narrow_lined_up(INFO, (22.8, 8.85), 0.0, detected_at)        # off the axis
    assert not narrow_lined_up(INFO, (22.8, 8.62), -0.4, detected_at)       # at an angle


def test_vamos_waits_until_the_dog_is_well_out_of_a_narrow_way():
    last = (23.4, 8.6)                                  # last steered through the south gap here
    assert rb.leaving_narrow(last, (24.6, 8.6))         # at the trolley's end: still lane point
    assert rb.leaving_narrow(last, (26.3, 8.8))
    assert not rb.leaving_narrow(last, (26.5, 8.85))    # NARROW_EXIT_M on: VAMOS again
    assert not rb.leaving_narrow(None, (24.0, 8.6))     # no narrow way: nothing held


def test_on_the_lane_point_it_is_back_in_its_lane_before_vamos_steers():
    # HIL 11-11 10:36: out of the south gap at y 8.6 (0.45 m off the wall),
    # VAMOS kept it there at 0.28-0.40 m/s. Steered by the lane point for
    # NARROW_EXIT_M, the same dog (board turning, guard) ends up near its
    # lane (8.87), where VAMOS's imagined runs keep their room.
    from cyberdog.sim.control import K_W, TURN_ONLY
    route = [(20.0, 8.87), (43.6, 8.87)]
    x, y, yaw, w = 23.4, 8.6, 0.0, 0.0
    last = (x, y)
    while rb.leaving_narrow(last, (x, y)):
        tx, ty = rb.lane_point((x, y), route, 1)
        err = rb.wrap(math.atan2(ty - y, tx - x) - yaw)
        v = 0.0 if abs(err) > TURN_ONLY else rb.PASS_V * math.cos(err)
        assert board_front(x, y, yaw) > STOP_M
        w += min(1.0, SIM_DT / TURN_LAG) * (TURN_GAIN * K_W * err - w)
        x += v * math.cos(yaw) * SIM_DT
        y += v * math.sin(yaw) * SIM_DT
        yaw += w * SIM_DT
    assert y - 8.15 > 0.6                               # room from the south wall
    assert abs(y - 8.87) < 0.1
