"""Run.spare_handler's hold: a turn that would put the person on the handle into
a wall waits HOLD_TURN_S, then goes -- and keeps going, not one tick in three s."""
import math
import types

from cyberdog.sim import run_building as rb

DT = 0.05


def stub(pinned=True):
    """A run whose every roll-forward puts the person into the wall (pinned),
    or leaves them plenty of room."""
    run = types.SimpleNamespace(robot=types.SimpleNamespace(CONTROL_DT=DT), turn_held=0, side_held=0.0, obs={"eased": 0})
    run.room = 0.05 if pinned else 0.6
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.10 if pinned else 0.6
    run.swing = lambda live, pose, cmd, people=(), dt=0.1, need=rb.ROBOT_R, ahead=0.0: (
        run.room, True, run.room, 0.5)
    run.spare_handler = rb.Run.spare_handler.__get__(run)
    return run


def ticks(run, n, cmd=(0.0, 0.0, 0.8)):
    return [run.spare_handler(None, (0.0, 0.0, 0.0), cmd, limit=0.5) for _ in range(n)]


HOLD = int(round(rb.HOLD_TURN_S / DT))


def test_a_turn_into_the_wall_is_held_for_hold_turn_s():
    run = stub()
    assert all(c == (0.0, 0.0, 0.0) for c in ticks(run, HOLD))


def test_after_the_hold_the_turn_goes_on_not_for_one_tick():
    # Out of the floor-3 lift, the user against the car and the wall: the old
    # count reset after the one tick it let through, so the dog turned 0.04
    # rad every 3 s and stood there for 215 s.
    run = stub()
    ticks(run, HOLD)
    after = ticks(run, 40)
    assert all(c[2] != 0.0 for c in after), [c for c in after if c[2] == 0.0][:3]


def test_free_again_the_next_pinned_turn_waits_again():
    run = stub()
    ticks(run, HOLD + 5)
    run.room = 0.6                                   # the turn no longer reaches the wall
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.6
    assert ticks(run, 1) == [(0.0, 0.0, 0.8)] and run.turn_held == 0
    run.room = 0.05
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.10
    assert ticks(run, 1) == [(0.0, 0.0, 0.0)]       # pinned anew: held anew


def test_a_turn_with_room_is_not_held():
    run = stub(pinned=False)
    assert ticks(run, 3) == [(0.0, 0.0, 0.8)] * 3


class Corridor:
    """Distance to the nearest wall of an east-west corridor whose south wall
    is at y = SOUTH, ending (optionally) in a wall at x = END south of y = 9.0:
    the floor-1 lift's shaft corner, as the brain's map has it."""
    SOUTH, END = 8.32, 45.3

    def __init__(self, end=True):
        self.end = end

    def __call__(self, x, y):
        d = y - self.SOUTH
        if self.end:                                 # the corner block: x >= END, y <= 9.0
            d = min(d, math.hypot(max(self.END - x, 0.0), max(y - 9.0, 0.0)))
        return max(d, 0.0)

    detected_at = remembered_at = lambda self, x, y: rb.INF
    wall_at = __call__


def real(end=True):
    run = types.SimpleNamespace(robot=types.SimpleNamespace(CONTROL_DT=DT), turn_held=0, side_held=0.0, obs={"eased": 0})
    for name in ("handler_room", "swing", "spare_handler"):
        setattr(run, name, getattr(rb.Run, name).__get__(run))
    return run, Corridor(end)


PIVOT = (0.0, 0.0, 2.0)                  # 1 rad left of a heading east: the lift's wait point
AT_THE_CORNER = (44.59, 8.77, 0.06)      # HIL 11-09 13:02, t 78 s


def test_easing_does_not_walk_on_into_the_shaft_corner():
    run, live = real()
    vx, vy, w = run.spare_handler(live, AT_THE_CORNER, PIVOT, limit=0.3)
    assert vx == 0.0 and vy > 0.0, (vx, vy, w)      # a step left, towards the turn


def test_with_room_ahead_easing_still_walks_on():
    run, live = real(end=False)
    vx, vy, w = run.spare_handler(live, AT_THE_CORNER, PIVOT, limit=0.3)
    assert vx == rb.SWING_V and w < PIVOT[2], (vx, vy, w)


def sidestep_only(cmd):
    """A run where `cmd` would put the person into something, walking forward
    in any form is not clear, and a step either way is: the dog's nose in the
    cafeteria door's east jamb (selftest, t 22-300 s)."""
    run = types.SimpleNamespace(robot=types.SimpleNamespace(CONTROL_DT=DT), turn_held=0,
                                side_held=0.0, obs={"eased": 0}, cmd=cmd)
    run.handler_room = lambda live, x, y, yaw, walls_only=False: 0.6

    def swing(live, pose, c, people=(), dt=0.1, need=rb.ROBOT_R, ahead=0.0):
        if c[1] != 0.0:
            return 0.6, True, 0.6, 0.5                  # a step sideways: fine
        if c == run.cmd:
            return 0.0, True, 0.6, 0.5                  # as commanded: into it
        return 0.6, False, 0.6, 0.1                     # walking on: nose in the jamb
    run.swing = swing
    run.spare_handler = rb.Run.spare_handler.__get__(run)
    return run


def step(run, w):
    run.cmd = (0.8, 0.0, w)
    return run.spare_handler(None, (33.84, 11.17, 2.14), run.cmd, limit=0.8)


def test_a_turn_picks_the_side_of_the_sidestep():
    assert step(sidestep_only((0.8, 0.0, -0.25)), -0.25)[1] == -rb.SIDE_V


def test_a_turn_rate_near_zero_keeps_the_side_already_stepped_to():
    # Selftest cafeteria: w +-0.01 flipping sign every tick stepped the dog
    # -0.2, +0.2, -0.2 m/s on the spot for 278 s.
    run = sidestep_only((0.8, 0.0, -0.25))
    step(run, -0.25)
    sides = [step(run, w)[1] for w in (0.01, -0.01, 0.02, -0.005, 0.01)]
    assert sides == [-rb.SIDE_V] * 5, sides


def test_a_turn_rate_near_zero_with_no_side_held_does_not_sidestep():
    run = sidestep_only((0.8, 0.0, 0.01))
    assert step(run, 0.01)[1] == 0.0 and run.side_held == 0.0


def test_the_held_side_is_let_go_once_the_command_is_fine():
    run = sidestep_only((0.8, 0.0, -0.25))
    step(run, -0.25)
    run.swing = lambda live, pose, c, people=(), dt=0.1, need=rb.ROBOT_R, ahead=0.0: (0.6, True, 0.6, 0.5)
    assert step(run, 0.01) == (0.8, 0.0, 0.01) and run.side_held == 0.0
