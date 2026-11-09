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
