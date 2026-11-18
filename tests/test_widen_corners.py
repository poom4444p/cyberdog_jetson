"""lane.widen_corners: a right turn out of the right-hand lane swings wide.

HIL 11-18, mechanical engineering lab: turning from the main corridor into
the wing (floor 2) the rounded line passed 0.32 m from the inflated wall
corner at (40.2, 10.65); the dog cut it, the board's guard closed the front
and its turn on the spot pinned the dog against the corner for 30 s.
"""
import math

import pytest

import cyberdog.planning.building_router as br
from cyberdog.mapping import lane

START = (45.0, 4.0)
APEX = (40.2, 10.65)                      # inflated corner, main corridor -> wing


@pytest.fixture(scope="module")
def router():
    return br.BuildingRouter()


def last_line(router, name, widen=True):
    tgt = router.resolve(name, 1, START)
    floor, route = router.plan(1, START, tgt)[-1]
    real = br.widen_corners
    if not widen:
        br.widen_corners = lambda pts, says, *a, **k: (pts, says)
    try:
        return floor, router.walk_line(floor, route)[0]
    finally:
        br.widen_corners = real


@pytest.mark.parametrize("name", ["mechanical engineering lab", "cafeteria"])
def test_into_the_wing_the_line_swings_wide(router, name):
    _, before = last_line(router, name, widen=False)
    _, after = last_line(router, name)
    near = lambda line: min(math.dist(p, APEX) for p in line)
    assert near(after) > near(before) + 0.08
    # up the corridor's far half first: the turn starts south of the lane
    assert min(p[1] for p in after if 39.5 < p[0] < 41.0 and p[1] < 11) < 9.7


def test_heading_into_the_wing_before_the_apex(router):
    _, line = last_line(router, "mechanical engineering lab")
    seg = [p for p in line if 39.0 < p[0] < 41.0 and 10.2 <= p[1] <= 11.2]
    assert len(seg) >= 2
    for a, b in zip(seg, seg[1:]):
        heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        assert 90 <= heading <= 120                 # north, a little west


@pytest.mark.parametrize("name", ["room 201", "server room", "office", "chemistry lab",
                                  "biology lab", "room 101"])
def test_main_block_routes_unchanged(router, name):
    assert last_line(router, name) == last_line(router, name, widen=False)


def test_straight_line_untouched(router):
    g = router.plan_grids[1]
    pts = [(10.0, 9.5), (20.0, 9.5), (30.0, 9.5)]
    out, _ = lane.widen_corners(pts, [[], [], []], g, router.doorways(1))
    assert out == pts
