"""Keeping to the right-hand lane while walking it, and passing people in the
other lane rather than stopping for them (HIL 11-09: floor 1 walked in the
middle, and stood still 41-45 s for passers-by)."""
import math
import types

from cyberdog.sim import run_building as rb

ROUTE = [(20.0, 8.87), (43.6, 8.87), (45.0, 9.5)]     # floor 1, east, right-hand lane


def test_lane_point_is_ahead_on_the_route_not_at_the_far_waypoint():
    x, y = rb.lane_point((30.0, 9.4), ROUTE, 1)          # 0.53 m left of the lane
    assert abs(y - 8.87) < 1e-9 and abs(x - (30.0 + rb.LANE_LOOK)) < 1e-9


def test_lane_point_carries_round_a_corner_and_stops_at_the_end():
    x, y = rb.lane_point((43.0, 8.87), ROUTE, 1)         # 0.6 m before the corner
    assert x > 43.6 and y > 8.87
    assert rb.lane_point((44.9, 9.45), ROUTE, 2) == (45.0, 9.5)


def test_in_lane_holds_a_vamos_target_within_tol():
    a, b = ROUTE[0], ROUTE[1]
    assert rb.in_lane((30.0, 9.0), a, b) == (30.0, 9.0)               # 0.13 m: as is
    x, y = rb.in_lane((30.0, 9.6), a, b)                              # 0.73 m left
    assert x == 30.0 and abs(y - (8.87 + rb.LANE_TOL)) < 1e-9


def person(x, y, vx, vy=0.0):
    return types.SimpleNamespace(x=x, y=y, vx=vx, vy=vy, moving=True)


def test_someone_coming_the_other_way_in_the_other_lane_passes_by():
    # Dog in its right-hand lane going east; they keep right going west.
    assert rb.passes_by((30.0, 8.87), 0.0, person(33.0, 10.1, -1.0))


def test_someone_coming_in_the_dogs_lane_or_crossing_does_not():
    assert not rb.passes_by((30.0, 8.87), 0.0, person(33.0, 9.0, -1.0))        # head-on
    assert not rb.passes_by((30.0, 8.87), 0.0, person(32.0, 10.1, -0.3, -0.6))  # crossing over
    assert not rb.passes_by((30.0, 8.87), 0.0, person(33.0, 9.5, -1.0))        # 0.63 m: too near
