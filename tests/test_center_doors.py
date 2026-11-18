"""lane.center_doors: walk lines go through doorways square and in the middle.

HIL 11-17: into the mechanical engineering lab (floor 2) the line crossed the
1.2 m door at x 39.27, 0.33 m from the east jamb, on a slant; the dog rubbed
the jamb for 0.9 s. Since the L-shaped building (#47) the lab is in the wing,
its door in the corridor's west wall (x 37.5); the cafeteria is the open hall
at the wing's end, no door. The doors of the west-end rooms (x01) sit beside the
stairwell's no-go zone, which tilted the first version's axis 17 degrees.
"""
import math

import numpy as np
import pytest

from cyberdog.mapping import lane
from cyberdog.planning.building_router import BuildingRouter

START = (45.0, 4.0)                                   # main entrance, floor 1
# name -> (floor, door centre along the wall, wall line, wall axis):
# axis "x" = the wall runs along x (main corridor), "y" = along y (the wing)
DOORS = {
    "mechanical engineering lab": (2, 26.5, 37.5, "y"),
    "room 201": (2, 3.0, 8.0, "x"),
    "room 101": (1, 3.0, 8.0, "x"),
    "room 205": (2, 27.0, 8.0, "x"),
    "office": (1, 39.0, 8.0, "x"),
    "server room": (3, 45.0, 8.0, "x"),
}


def wall_frame(p, axis):
    """(along the wall, across it) for a point, whichever way the wall runs."""
    return (p[0], p[1]) if axis == "x" else (p[1], p[0])


@pytest.fixture(scope="module")
def router():
    return BuildingRouter()


def last_leg(router, name):
    tgt = router.resolve(name, 1, START)
    floor, route = router.plan(1, START, tgt)[-1]
    return floor, route, router.walk_line(floor, route)


@pytest.mark.parametrize("name", sorted(DOORS))
def test_through_the_middle_and_square(router, name):
    floor, along, wall, axis = DOORS[name]
    f, _, (pts, _) = last_leg(router, name)
    assert f == floor
    gates = [g for g in router.gates(floor, pts)
             if abs(wall_frame(g[0], axis)[0] - along) < 1.0
             and abs(wall_frame(g[0], axis)[1] - wall) < 1.0]
    assert gates, f"no gate at the {name} door"
    for entry, exit_, u in gates:
        assert abs(wall_frame(entry, axis)[0] - along) < 0.05
        assert abs(wall_frame(exit_, axis)[0] - along) < 0.05
        assert abs(wall_frame(u, axis)[0]) < math.sin(math.radians(5))   # square


@pytest.mark.parametrize("name", ["mechanical engineering lab", "room 201"])
def test_lined_up_before_the_frame(router, name):
    # Within 0.6 m of the wall line, either side, the line is on the door's
    # axis: round_corners' arcs end before the doorway.
    _, along, wall, axis = DOORS[name]
    _, _, (pts, _) = last_leg(router, name)
    near = [q for q in (wall_frame(p, axis) for p in pts)
            if abs(q[1] - wall) < 0.6 and abs(q[0] - along) < 1.5]
    assert len(near) >= 2
    assert max(abs(q[0] - along) for q in near) < 0.05


def test_the_mechanical_engineering_lab_no_longer_crosses_by_the_jamb(router):
    _, along, wall, axis = DOORS["mechanical engineering lab"]
    _, _, (pts, _) = last_leg(router, "mechanical engineering lab")
    crossed = 0
    for a, b in zip(pts, pts[1:]):
        a, b = wall_frame(a, axis), wall_frame(b, axis)
        if (a[1] - wall) * (b[1] - wall) <= 0 and a[1] != b[1]:
            t = a[0] + (b[0] - a[0]) * (wall - a[1]) / (b[1] - a[1])
            assert abs(t - along) < 0.05                    # 11-17: 0.27 off, by the jamb
            crossed += 1
    assert crossed


def test_announcements_are_kept(router):
    for name in DOORS:
        floor, route, _ = last_leg(router, name)
        says = [[] for _ in route.polyline]
        for c in route.checkpoints:
            says[c.index] = list(c.announcements)
        pts = [(float(x), float(y)) for x, y in route.polyline]
        _, out = lane.center_doors(pts, says, router.plan_grids[floor],
                                   router.doorways(floor), router.lane_walls(floor))
        assert sorted(w for s in out for w in s) == sorted(w for s in says for w in s)


def test_no_doorway_no_change(router):
    floor = 1
    pts = [(10.0, 9.5), (20.0, 9.5), (30.0, 9.5)]           # along the corridor
    says = [["a"], [], ["b"]]
    out, out_says = lane.center_doors(pts, says, router.plan_grids[floor],
                                      router.doorways(floor), router.lane_walls(floor))
    assert out == pts and out_says == says


def test_axis_is_square_beside_the_stairwell_zone(router):
    from scipy import ndimage
    floor = 2
    g = router.plan_grids[floor]
    labels, _ = ndimage.label(router.doorways(floor))
    r, c = g.world_to_grid(3.0, 8.0)
    label = int(labels[r - 10:r + 10, c - 10:c + 10].max())
    centre, n = lane._door_axis(g, labels, router.lane_walls(floor), label)
    assert centre == pytest.approx((3.0, 8.0), abs=0.06)
    assert abs(n[0]) < math.sin(math.radians(5)) and abs(abs(n[1]) - 1) < 0.01


DOORS_OWN_SIDE = {"biology lab": (3, 33.0), "electrical engineering lab": (2, 33.0),
                  "room 106": (1, 3.0)}               # name -> (floor, door x), wall y 11


@pytest.mark.parametrize("name", sorted(DOORS_OWN_SIDE))
def test_door_on_the_lanes_own_side_keeps_the_old_line(router, name):
    # The lane runs 0.9 m off the north wall, so the approach point would
    # sit on it: a 90 deg corner at the door, no room to round it. The
    # selftest's handler swung into the wall / the floor-2 crate there.
    # Such doors keep the line the lane gave.
    import cyberdog.planning.building_router as br
    tgt = router.resolve(name, 1, START)
    floor, route = router.plan(1, START, tgt)[-1]
    with_fix = router.walk_line(floor, route)
    real = br.center_doors
    br.center_doors = lambda pts, says, *a, **k: (pts, says)
    try:
        without = router.walk_line(floor, route)
    finally:
        br.center_doors = real
    door = (DOORS_OWN_SIDE[name][1], 11.0)
    near = lambda line: [p for p in line[0] if math.dist(p, door) < 2.0]
    assert near(with_fix) == near(without)


def test_sharp_turn_onto_the_axis_is_refused(router):
    floor = 1
    g = router.plan_grids[floor]
    # along the lane by the north wall, then up through room 106's door (3, 11)
    pts = [(8.0, 10.1), (3.0, 10.1), (3.0, 13.0)]
    says = [[], [], []]
    out, _ = lane.center_doors(pts, says, g, router.doorways(floor), router.lane_walls(floor))
    assert out == pts
