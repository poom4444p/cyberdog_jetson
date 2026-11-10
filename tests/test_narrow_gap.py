"""The floor-1 trolley with somebody standing in its 1.10 m north gap: wait a
while for them, then through the 0.90 m south gap lined up, so the board's
guard (0.30 m either side of the heading) lets the dog through."""
import math
from types import SimpleNamespace

from cyberdog.sim import run_building as rb
from cyberdog.sim.run_building import axis_clear, narrow_aim, small_thing_beside

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
    assert abs(aim[1] - 8.6) < 0.01 and aim[0] > 24.0



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
