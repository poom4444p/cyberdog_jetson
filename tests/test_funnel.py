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


# The cafeteria leg on floor 1 (selftest): north through the lift lobby's
# doorway, whose last gates have slanted axes, then 11 m west and north into
# the cafeteria door. Gates as Router.gates gives them for that walk line.
CAFETERIA_GATES = [
    ((44.75, 7.66), (44.75, 7.71), (-0.07, 1.00)),
    ((44.75, 7.71), (44.76, 8.02), (0.03, 1.00)),
    ((44.76, 8.02), (44.83, 8.33), (0.24, 0.97)),
    ((44.83, 8.33), (44.86, 8.39), (0.34, 0.94)),
    ((33.38, 10.62), (33.34, 10.68), (-0.55, 0.83)),
    ((33.34, 10.68), (33.24, 10.97), (-0.31, 0.95)),
    ((33.24, 10.97), (33.22, 11.38), (-0.05, 1.00)),
]


def test_a_slanted_gate_behind_the_dog_does_not_switch_the_funnel_off():
    # From (34.36, 10.53) the (0.24, 0.97) gate 11 m behind no longer reads as
    # walked through, and the funnel stopped there: steered at the lane point,
    # the dog cut the corner into the cafeteria door's east jamb.
    xy = (34.36, 10.53)
    (tx, ty), _ = rb.Run.funnel(CAFETERIA_GATES, xy, (33.21, 11.49))
    entry, _, (ux, uy) = CAFETERIA_GATES[4]
    assert abs((tx - entry[0]) * -uy + (ty - entry[1]) * ux) < 1e-9      # on the door's axis
    assert rb.Run.door_through(CAFETERIA_GATES, xy) is not None
    assert rb.Run.door_near(CAFETERIA_GATES, xy, rb.DOOR_MAP_D)


def test_mid_corridor_on_the_cafeteria_leg_no_door_is_near():
    xy = (39.1, 10.2)
    assert rb.Run.funnel(CAFETERIA_GATES, xy, (37.1, 10.2)) == ((37.1, 10.2), False)
    assert rb.Run.door_through(CAFETERIA_GATES, xy) is None
    assert not rb.Run.door_near(CAFETERIA_GATES, xy, rb.DOOR_MAP_D)


def test_on_its_route_the_dog_is_not_lined_up_before_the_frame():
    # The electrical engineering lab: walking its route's rounded corner, the
    # line-up held it in the corridor to 0.5 m before the door, then it pivoted
    # there and swung its user onto the floor-2 crate (selftest). On its route
    # the funnel steers FUNNEL_LEAD along the axis, as before the line-up.
    x, y = 43.38, 8.91
    (tx, ty), _ = rb.Run.funnel([GATE], (x, y), (45.0, 4.0), off_route=False)
    assert tx == AXIS_X and abs(ty - (GATE[0][1] - ((GATE[0][1] - y) + rb.FUNNEL_LEAD))) < 1e-9


def test_route_offset_is_the_distance_to_the_rest_of_the_route():
    route = [(45.0, 9.5), (45.0, 7.0), (45.0, 4.0)]
    assert abs(rb.route_offset((43.38, 8.91), route, 1) - 1.62) < 1e-9   # HIL 11-09: off it
    assert rb.route_offset((45.1, 8.0), route, 1) < rb.FUNNEL_OFF         # on it
    assert abs(rb.route_offset((44.0, 8.0), route, 2) - math.hypot(1.0, 1.0)) < 1e-9  # walked part not counted
