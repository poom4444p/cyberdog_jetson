"""floor_guard: what the simulated guard sees, when it says so, and what that becomes."""
import math
import random

import numpy as np

from cyberdog.planning.checkpoint_projector import load_camera_config
from cyberdog.sim.sensing import floor_guard as fg
from cyberdog.sim.sensing import perception

CAM = load_camera_config()
DT = 0.05


def open_floor(xs, ys):
    return np.full(len(xs), 5.0)


def wall_at_x(x0):
    """A wall across the floor at x = x0 (and everything past it reads open)."""
    def static(xs, ys):
        xs = np.asarray(xs)
        return np.where(np.abs(xs - x0) < 0.05, 0.0, 5.0)
    return static


def guard(**kw):
    return fg.FloorGuard(CAM, DT, **kw)


def puddle(x, y, r=0.4, floor=1, id=0):
    return fg.Puddle(id, floor, x, y, r)


def test_cap_is_the_vlm_rule_with_the_guards_own_latency():
    g = guard(see_d=2.0, latency=0.25)
    assert math.isclose(g.cap(), (2.0 - fg.STOP_D) / 0.5)
    assert g.lat_ticks == 5


def test_seen_ahead_in_range_not_behind_or_far():
    g = guard()
    pose = (0.0, 0.0, 0.0)                      # lens at the origin, facing +x
    assert g.visible(puddle(1.5, 0.0), pose, open_floor)
    assert not g.visible(puddle(-1.5, 0.0), pose, open_floor)
    assert not g.visible(puddle(4.0, 0.0), pose, open_floor)
    # Under the lens: the camera does not see the floor right in front of it.
    assert not g.visible(puddle(0.1, 0.0, r=0.05), pose, open_floor)


def test_not_seen_through_a_wall():
    g = guard()
    assert not g.visible(puddle(1.5, 0.0, r=0.2), (0.0, 0.0, 0.0), wall_at_x(1.0))


def test_answer_lands_one_latency_after_the_frame_and_frames_run_back_to_back():
    g = guard(latency=0.25)                      # 5 ticks
    g.add([puddle(1.5, 0.0)])
    pose = (0.0, 0.0, 0.0)
    new = [g.tick(n, 1, pose, (0.0, 0.0), open_floor) for n in range(6)]
    assert all(not x for x in new[:5])           # frame at tick 0, answer at tick 5
    assert [p.id for p in new[5]] == [0]
    assert g.frames == 2                          # the next frame was taken at tick 5
    assert math.isclose(g.known_at[0], 1.5 - 0.4)


def test_a_puddle_coming_into_view_after_a_frame_waits_for_the_next():
    g = guard(latency=0.25)
    g.add([puddle(1.5, 0.0)])
    g.tick(0, 1, (0.0, 0.0, math.pi), (0.0, 0.0), open_floor)   # facing away at tick 0
    for n in range(1, 10):
        assert not g.tick(n, 1, (0.0, 0.0, 0.0), (0.0, 0.0), open_floor)
    assert [p.id for p in g.tick(10, 1, (0.0, 0.0, 0.0), (0.0, 0.0), open_floor)] == [0]


def test_other_floors_and_known_puddles_are_not_reported_again():
    g = guard(latency=0.05)
    g.add([puddle(1.5, 0.0, floor=2)])
    for n in range(5):
        assert not g.tick(n, 1, (0.0, 0.0, 0.0), (0.0, 0.0), open_floor)
    g = guard(latency=0.05)
    g.add([puddle(1.5, 0.0)])
    got = [p for n in range(6) for p in g.tick(n, 1, (0.0, 0.0, 0.0), (0.0, 0.0), open_floor)]
    assert len(got) == 1


def test_always_missed_is_never_known():
    g = guard(latency=0.05, p_miss=1.0)
    g.add([puddle(1.5, 0.0)])
    for n in range(20):
        assert not g.tick(n, 1, (0.0, 0.0, 0.0), (0.0, 0.0), open_floor)


def test_returns_fill_the_puddle_at_a_height_perception_keeps():
    g = guard()
    g.known[0] = puddle(3.0, 1.0, r=0.3)
    pts = g.returns(1, 0.0, (0.0, 0.0), reach=8.0)
    assert len(pts) > 50
    assert np.all(np.hypot(pts[:, 0] - 3.0, pts[:, 1] - 1.0) <= 0.3 + 1e-9)
    assert perception.MIN_H < fg.RETURN_H < perception.MAX_H
    assert len(g.returns(2, 0.0, (0.0, 0.0), reach=8.0)) == 0     # another floor
    assert len(g.returns(1, 0.0, (20.0, 0.0), reach=8.0)) == 0    # out of reach


def test_stepped_in_counts_ticks_inside_radius_plus_feet():
    g = guard()
    g.add([puddle(1.0, 0.0, r=0.3)])
    g.score(1, (1.0 + 0.3 + fg.STEP_R - 0.01, 0.0))
    g.score(1, (1.0 + 0.3 + fg.STEP_R + 0.01, 0.0))
    assert g.stepped == {0: 1}


def test_placed_on_open_floor_off_the_ends_apart_and_reproducible():
    route = [(0.0, 0.0), (20.0, 0.0)]
    a = fg.place_puddles(3, 1, route, open_floor, random.Random(4))
    b = fg.place_puddles(3, 1, route, open_floor, random.Random(4))
    assert a == b and len(a) == 3
    for p in a:
        assert fg.PUDDLE_ENDS <= p.x <= 20.0 - fg.PUDDLE_ENDS
        assert abs(p.y) <= fg.PUDDLE_LAT
    for p, q in [(p, q) for p in a for q in a if p.id < q.id]:
        assert math.hypot(p.x - q.x, p.y - q.y) >= p.r + q.r + 1.0


def test_no_puddle_on_a_short_leg_against_walls_or_by_a_door():
    assert fg.place_puddles(2, 1, [(0.0, 0.0), (5.0, 0.0)], open_floor, random.Random(0)) == []
    tight = lambda xs, ys: np.full(len(xs), 0.3)
    assert fg.place_puddles(2, 1, [(0.0, 0.0), (20.0, 0.0)], tight, random.Random(0)) == []
    door = fg.place_puddles(2, 1, [(0.0, 0.0), (20.0, 0.0)], open_floor, random.Random(0),
                            near_door=lambda xy: True)
    assert door == []


def test_a_puddle_that_leaves_no_way_round_is_not_placed():
    # A 2.3 m corridor along x, walls at y = -1.15 and +1.15.
    def corridor(xs, ys):
        return np.clip(1.15 - np.abs(np.asarray(ys)), 0.0, None)
    assert math.isclose(fg.side_gap(0.0, 0.0, 0.5, 0.0, 1.0, corridor), 0.65, abs_tol=0.06)
    got = fg.place_puddles(3, 1, [(0.0, 0.0), (30.0, 0.0)], corridor, random.Random(0))
    for p in got:
        assert max(fg.side_gap(p.x, p.y, p.r, 0.0, 1.0, corridor),
                   fg.side_gap(p.x, p.y, p.r, 0.0, -1.0, corridor)) >= fg.PUDDLE_GAP


def test_a_known_puddle_keeps_the_dog_off_without_a_shadow_behind_it():
    # Floor 1 corridor east of the trolley; the dog at (30.4, 8.9), a puddle 2.5 m on.
    live = perception.LiveClearance(1)
    g = guard()
    g.known[0] = puddle(33.0, 9.3, r=0.3)
    flat = g.returns(1, 0.0, (30.4, 8.9), reach=8.0)[:, :2]
    live.update(np.empty((0, 3)), 0.0, (30.4, 8.9), origin=(30.4, 8.9, 0.5), flat=flat)
    assert live.detected_at(33.0, 9.3) < 0.05            # on it
    assert live.detected_at(33.0 + 0.3 + 0.5, 9.3 + 0.12) > 0.4   # 0.5 m beyond it: open
    assert len(live.points) == 0                         # not something the tracker sees
