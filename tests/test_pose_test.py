"""LiveClearance.pose_test gives CE-RRT* the same answer, pose for pose, as
go_round's rules asked through static_at / detected_at / wall_at /
Run.handler_room -- it is only faster."""
import math
import types

import numpy as np
import pytest

from cyberdog.sim import run_building as rb
from cyberdog.sim.sensing import perception as pc

DOG = (13.175, 9.325)          # floor 1, open floor


def reference(live, s0, c0, n0, h0):
    run = types.SimpleNamespace()
    run.handler_room = rb.Run.handler_room.__get__(run)

    def ok(x, y, yaw):
        fx = x + rb.SWING_NOSE * math.cos(yaw)
        fy = y + rb.SWING_NOSE * math.sin(yaw)
        return (live.static_at(x, y) >= s0 and live.detected_at(x, y) >= c0
                and live.wall_at(fx, fy) >= n0
                and run.handler_room(live, x, y, yaw) >= h0)
    return ok


def scene(fields):
    live = pc.LiveClearance(1)
    if fields == "none":
        return live
    rng = np.random.default_rng(3)
    x, y = DOG
    crate = np.column_stack([x + 1.5 + rng.uniform(0, 0.4, 300),
                             y - 0.8 + rng.uniform(0, 0.4, 300), np.full(300, 0.4)])
    walker = np.column_stack([x + 3.0 + rng.uniform(0, 0.3, 100),
                              y + 0.5 + rng.uniform(0, 0.3, 100), np.full(100, 0.4)])
    live.update(np.vstack([crate, walker]), 0.0, DOG, DOG)
    live.exclude([(x + 3.15, y + 0.65, 0.3)])
    if fields == "still+recent":
        live.remember([], 10.0, DOG)
    return live


def poses(live, k=20000):
    rng = np.random.default_rng(7)
    x, y = DOG
    # Mostly round the dog, where the window fields are; some far off and off
    # the map; some on cell edges, where int() truncation decides.
    xs = np.concatenate([x + rng.uniform(-9, 9, k), rng.uniform(-5, 60, k // 4),
                         live.ox + np.round(rng.uniform(0, 300, k // 4)) * live.res])
    ys = np.concatenate([y + rng.uniform(-9, 9, k), rng.uniform(-5, 30, k // 4),
                         live.oy + np.round(rng.uniform(0, 300, k // 4)) * live.res])
    yaws = rng.uniform(-math.pi, math.pi, len(xs))
    return zip(xs.tolist(), ys.tolist(), yaws.tolist())


@pytest.mark.parametrize("fields", ["none", "still", "still+recent"])
@pytest.mark.parametrize("floor", [
    (rb.ROBOT_R, rb.PLAN_CRATE, rb.PLAN_NOSE, rb.HANDLER_ROOM),   # the thresholds
    (0.0, rb.PLAN_CRATE, rb.PLAN_NOSE, rb.HANDLER_ROOM),         # s0 0: off the map passes
    (0.1, 0.05, 0.15, 0.2),                                      # squeezed start
])
def test_same_answer_as_the_methods(fields, floor):
    live = scene(fields)
    assert (live.still is not None) == (fields != "none")
    assert (live.recent is not None) == (fields == "still+recent")
    fast = live.pose_test(*floor, rb.SWING_NOSE, rb.HANDLER_BEHIND)
    ref = reference(live, *floor)
    n = mismatched = passed = 0
    for x, y, yaw in poses(live):
        a, b = fast(x, y, yaw), ref(x, y, yaw)
        n += 1
        passed += b
        mismatched += a != b
    assert mismatched == 0
    assert 0 < passed < n          # both answers actually occur


def test_thresholds_on_a_cell_value_exactly():
    # >= on the very value a cell holds: a pose there passes, both ways.
    live = scene("still+recent")
    x, y, yaw = DOG[0] + 1.0, DOG[1], 0.0
    s0, c0 = live.static_at(x, y), live.detected_at(x, y)
    fast = live.pose_test(s0, c0, 0.0, 0.0, rb.SWING_NOSE, rb.HANDLER_BEHIND)
    ref = reference(live, s0, c0, 0.0, 0.0)
    assert fast(x, y, yaw) == ref(x, y, yaw) is True
