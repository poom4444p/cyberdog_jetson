"""Blocked is blocked: go round anything, ask when there is no way round,
then find another way (run_building ASK_S, find_another_way, close_spot)."""
import math
from types import SimpleNamespace

import numpy as np

from cyberdog.planning.building_router import BuildingRouter
from cyberdog.sim import run_building as rb


def track(x, y, moving, r=0.25):
    return SimpleNamespace(x=x, y=y, r=r, moving=moving, walked=moving)


def test_only_somebody_walking_holds_the_dog_off():
    pose, ahead = (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)
    assert rb.closing_on([track(0.5, 0.0, moving=True)], pose, ahead)
    assert not rb.closing_on([track(0.5, 0.0, moving=False)], pose, ahead)    # gone round instead
    assert not rb.closing_on([track(-0.5, 0.0, moving=True)], pose, ahead)    # behind it


def test_the_thing_in_the_way_is_the_nearest_return_ahead():
    pts = np.array([[1.8, 0.1], [0.9, -0.2], [-0.5, 0.0], [3.5, 0.0]])
    assert rb.blocked_point(pts, (0.0, 0.0, 0.0)) == (0.9, -0.2)
    # Nothing seen ahead (inside the LiDAR's minimum range): half a metre on.
    x, y = rb.blocked_point(np.empty((0, 2)), (1.0, 2.0, math.pi / 2))
    assert (round(x, 6), round(y, 6)) == (1.0, 2.5)


def test_a_closed_corridor_sends_the_trip_another_way():
    router = BuildingRouter()
    stops = [("server room", None)]
    _, legs, _ = rb.plan_stops(router, stops, start=(1, (30.0, 9.5)))
    assert legs[0].via == "lift"
    rb.close_spot(router, 1, (35.0, 9.4), 0.0)       # blocked between the dog and the lift
    _, legs, _ = rb.plan_stops(router, stops, start=(1, (30.0, 9.5)))
    assert legs[0].via == "stairs"                   # back the other way, to the stairs
    assert max(p[0] for p in legs[0].points) <= 30.0 + 0.1
    assert legs[-1].floor == 3


def test_closing_leaves_the_dog_where_it_stands_open():
    router = BuildingRouter()
    g = router.plan_grids[1]
    before = g.grid.copy()
    rb.close_spot(router, 1, (35.0, 9.4), 0.0)
    r, c = g.world_to_grid(35.0 - rb.BLOCK_HALF - 0.3, 9.4)   # where the dog stood, short of it
    assert g.grid[r, c] == before[r, c]
    r, c = g.world_to_grid(35.0, 9.4)
    assert g.grid[r, c] == 100


def test_how_far_ahead_the_thing_in_the_way_is():
    pts = np.array([[7.7, 0.3], [3.0, 1.5], [-1.0, 0.0]])    # a row of people, one off to the side
    assert rb.nearest_ahead(pts, (0.0, 0.0, 0.0)) == 7.7     # still a room away: walk on up
    assert rb.nearest_ahead(np.empty((0, 2)), (0.0, 0.0, 0.0)) == rb.INF
