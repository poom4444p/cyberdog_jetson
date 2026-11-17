"""Run.exit_spot: out of the lift, where to stand and which way to turn before
the leg on takes over. A corner is turned standing, where the person on the
handle has room to come round; a way straight on is walked as before."""
import math
import types

from cyberdog.planning.building_router import BuildingRouter
from cyberdog.sim import run_building as rb
from cyberdog.sim.scene import lift


def run_and_leg(place, start=(1, (20.92, 9.5))):
    router = BuildingRouter(allow_stairs=False, lift_in_service=True)
    run = types.SimpleNamespace(router=router, live={}, funnel=rb.Run.funnel)
    for name in ("handler_room", "exit_spot", "clearance"):
        setattr(run, name, getattr(rb.Run, name).__get__(run))
    stops = rb.resolve_stops(router, place, use_nlu=False, guessed=[])
    legs = [o for o in rb.plan_stops(router, stops, start=start) if isinstance(o, list)][0]
    return run, legs[-1]


def test_server_room_turns_left_standing_in_front_of_its_door():
    # Its door is 1.1 m south of the waiting point: out of the car facing west,
    # the dog used to walk up to 1.5 m west before it could turn (HIL 11-09).
    run, leg = run_and_leg("server room")
    (x, y), heading = run.exit_spot(leg.floor, leg.points)
    assert abs(heading - (-math.pi / 2)) < 0.05                 # south, into the door
    assert abs(x - lift.WAIT_XY[0]) < 0.05 and y < lift.WAIT_XY[1]   # on its axis, nearer
    live = run.clearance(leg.floor)
    turn = rb.wrap(heading - math.pi)
    room = min(run.handler_room(live, x, y, math.pi + turn * k / 8) for k in range(9))
    assert room >= rb.HANDLER_ROOM


def test_straight_on_is_walked_from_the_waiting_point():
    run, leg = run_and_leg("room 301")
    assert run.exit_spot(leg.floor, leg.points) == (lift.WAIT_XY, math.pi)


def test_nothing_onward_is_the_old_exit():
    run, _ = run_and_leg("room 301")
    assert run.exit_spot(3, None) == (lift.WAIT_XY, math.pi)


class Walker:
    """A track as the tracker gives it: where, how fast, how wide."""

    def __init__(self, x, y, vx, vy, r=0.25, moving=True):
        self.x, self.y, self.vx, self.vy, self.r, self.moving = x, y, vx, vy, r, moving

    def predict(self, t):
        return (self.x + self.vx * t, self.y + self.vy * t)


# HIL 11-17 12:29, floor 3: the dog out of the car, turned south for the
# server room, its user in the doorway behind it; somebody walking east 1.9 m
# off on its right. It stopped there for them.
DOG, SOUTH = (45.01, 9.19), -math.pi / 2
CROSSER = Walker(43.14, 8.51, 0.73, 0.0)


def test_somebody_walking_past_off_to_the_side_is_not_right_in_front():
    assert not rb.right_in_front(DOG, SOUTH, CROSSER)


def test_somebody_in_the_lane_ahead_is_right_in_front():
    assert rb.right_in_front(DOG, SOUTH, Walker(45.1, 8.2, 0.0, 0.0))
    # ...or walking into it within EXIT_SEE_T
    assert rb.right_in_front(DOG, SOUTH, Walker(44.0, 8.4, 0.9, 0.0))


def test_behind_or_beside_is_not_right_in_front():
    assert not rb.right_in_front(DOG, SOUTH, Walker(45.0, 10.0, 0.0, 0.0))   # behind
    assert not rb.right_in_front(DOG, SOUTH, Walker(44.0, 9.2, 0.0, 0.0))    # beside


def way_out():
    x0, y0 = lift.WAIT_XY
    return [lift.WAIT_XY] + [(x0, y0 - d) for d in (0.6, 1.2, rb.EXIT_CLEAR_M)]


def test_the_car_waits_for_somebody_about_to_cross_the_way_out():
    assert rb.crossing_exit(CROSSER, way_out())


def test_the_car_does_not_wait_for_somebody_walking_away_or_standing():
    x0, y0 = lift.WAIT_XY
    assert not rb.crossing_exit(Walker(x0 - 3.0, y0 - 1.0, -0.8, 0.0), way_out())
    assert not rb.crossing_exit(Walker(x0 - 1.0, y0, 0.0, 0.0, moving=False), way_out())
