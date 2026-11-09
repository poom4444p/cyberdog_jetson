"""Run.funnel: steering into a doorway on its axis. The floor-3 server room's
door: a gap in the corridor's south wall at x = 45.0, entered heading south."""
import math

from cyberdog.sim import run_building as rb

GATE = ((45.0, 8.375), (45.0, 7.675), (0.0, -1.0))     # entry, exit, unit direction
AXIS_X = 45.0


def aim(xy):
    return rb.Run.funnel([GATE], xy, (45.0, 4.0))


def test_from_well_to_the_side_it_lines_up_before_the_frame():
    # HIL 11-09 13:25: back from 1.6 m west of the door, the point it steered
    # at was in the doorway and the line to it crossed the west jamb.
    (tx, ty), _ = aim((43.38, 8.91))
    assert tx == AXIS_X and abs(ty - (GATE[0][1] + rb.FUNNEL_PRE)) < 1e-9


def test_lined_up_on_the_axis_it_steers_into_the_doorway():
    (tx, ty), at_door = aim((45.0, 8.9))
    assert tx == AXIS_X and ty < GATE[0][1] and at_door


def test_a_little_off_the_axis_the_funnel_is_unchanged():
    x, y = 44.6, 9.4                                  # 0.4 m off, 1.0 m out
    s = GATE[0][1] - y
    (tx, ty), _ = aim((x, y))
    assert abs(ty - (GATE[0][1] - (s + rb.FUNNEL_LEAD))) < 1e-9
    assert math.atan2(abs(x - AXIS_X), y - ty) < rb.FUNNEL_ANGLE


def test_past_the_doorway_or_far_from_it_the_target_stands():
    assert aim((45.0, 6.0)) == ((45.0, 4.0), False)  # through it already
    assert aim((40.0, 9.0)) == ((45.0, 4.0), False)  # 5 m along the corridor


def test_door_through_is_past_the_far_side_while_lined_up_for_it():
    # What the LiDAR finds beyond it is in the next room (HIL 11-09 14:55: the
    # server room's racks, 3-5 m in, turned into "stepping around it" at the door).
    x, y = rb.Run.door_through([GATE], (45.0, 9.05))
    assert x == AXIS_X and abs(y - (GATE[1][1] - rb.FUNNEL_PAST)) < 1e-9


def test_no_door_through_far_from_a_doorway_or_once_past_it():
    assert rb.Run.door_through([GATE], (40.0, 9.0)) is None   # 5 m along the corridor
    assert rb.Run.door_through([GATE], (45.0, 6.0)) is None   # in the room already
