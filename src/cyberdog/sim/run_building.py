"""Drive the dog through the whole three-floor building, not one floor.

run_demo.py and record_demo.py each load floor1.xml and stop there,
so a route that crosses floors had nowhere to go. This runs the same stack
against output/scene_cache/building.xml -- all three floors and the lift between
them in one model -- so "take me to the server room" is one continuous run:
corridor, lift, corridor, lift, corridor.

The lift, never the stairs. The person holding the handle cannot see the steps
and has no free hand for a rail, so the stairwells are stop zones the planner
refuses to route through and this driver halts in if the dog ever drifts into
one. At the lift the dog stops and says so, because it cannot press a call
button -- a human does that, and presses continue.

    python -m cyberdog.sim.scene.build_scene --building          # once, to make the scene
    python -m cyberdog.sim.run_building "take me to the chemistry lab"
    python -m cyberdog.sim.run_building "go to the library on the 2nd floor, then the server room"
    python -m cyberdog.sim.run_building "restroom on floor 3" --no-video   # fast, headless
    python -m cyberdog.sim.run_building "server room" --nlu                # through the Gemma layer
    python -m cyberdog.sim.run_building "server room" --no-vamos           # map route only, no VLM server
    python -m cyberdog.sim.run_building "server room" --shadow             # VLM + dream, map still drives
    python -m cyberdog.sim.run_building "server room" --auto-confirm       # don't wait at the lift

Without --nlu the destination is matched against locations.json by name, which
keeps torch out of the process; --nlu runs the real Input Treating Layer.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from typing import NamedTuple

import numpy as np


from cyberdog import paths
from cyberdog.language.place_match import (alias_place, movement, near_place,
                                           need_place, numbers_to_digits)
from cyberdog.planning import ce_rrt
from cyberdog.planning.building_router import BuildingRouter, NoAccessibleRoute, Transit
from cyberdog.planning.checkpoint_projector import (load_camera_config,
                                                    pick_destination, project_route,
                                                    project_to_pixel,
                                                    vamos_prompt)
from cyberdog.sim.control import (GOAL_R, K_W, TURN_ONLY, advance, path_target,
                                  wrap)
from cyberdog.sim.handle import HandleScript, SafetyMux
from cyberdog.sim.overlay import (ChaseCam, draw_marker, draw_path,
                                  draw_points, label_places, safety_bar)
from cyberdog.sim.robot.mujoco_robot import MujocoRobot
from cyberdog.sim.robot.go2_board_robot import BoardStopped
from cyberdog.sim.scene import levels, lift, obstacles, pedestrians, stairs
from cyberdog.sim.sensing import perception
from cyberdog.sim.sensing import floor_guard as fguard
from cyberdog.sim.sensing.dreaming import ROBOT_R
from cyberdog.sim.sensing.lidar import MOUNT_H
from cyberdog.sim.sensing.perception import (INF, LiveClearance, free_destination,
                                             line_clear)
from cyberdog.sim.sensing.tracking import (CLEAR_R, MAX_RADIUS, MIN_RADIUS, MISS_R,
                                           NEAR_R, Tracker, closest_approach, conflict)
from cyberdog.sim.vamos_client import ahead

START_LOCATION = "main entrance"
# Where go2_board's odometry starts in the map frame, with --robot go2_board:
# go2_world's building is this map shifted so the Go2 starts at its origin
# (see go2_world/assets/building.xml). On the dog, FAST-LIO's start instead.
GO2_WORLD_ORIGIN = (21.0, 9.5, 0.0)
START_FLOOR = 1
FPS = 20                  # equals the control rate, so the video is real time
SAY_AGAIN_S = 2.0         # seconds before the same step aside is said again
GONE_S = 1.0              # seconds with nothing in sight before what was there is
                          # gone. One empty scan is not it: each one re-announced
                          # the detour, often on the other side -- 6 times "stepping
                          # around it" left and right in the floor-3 server room,
                          # round something standing by the destination (HIL 11-09).
WAIT_S = 1.5              # held at the lift doors, and while auto-confirming
ANNOUNCE_R = 2.5          # metres out that a checkpoint's line is spoken
STEP_V = 0.4              # m/s in and out of the car -- slower than corridor pace
# Out of the lift with the way on round a corner. The dog leaves the car facing
# out, its user behind it in the car; turning the corner while walking, the turn
# would swing them into the car, so it was eased into walking on: out of the
# floor-3 lift for the server room, 1.1 m south of the doors, the dog went up to
# 1.5 m west, turned 200-260 deg in all and said 4-6 things over 9-22 s before
# it was through the door (HIL 11-09). A guide dog stops, says which way, and
# turns slowly where there is room for the person to come round with it.
EXIT_TURN_MIN = math.radians(35)  # less than this off straight out: the leg turns as it walks
EXIT_TURN_W = 0.5         # rad/s: the turn, scripted, slow enough to follow on a handle
EXIT_SHIFT = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)   # m from WAIT_XY towards the way on
# Out of the lift, the dog and its user leave together and clear the door
# before anything stops them. Out of the floor-3 lift for the server room the
# dog turned in front of the door and then, on its first tick in the corridor,
# stopped for somebody walking past 1.9 m off -- its user, 1.1 m behind, still
# standing in the doorway (HIL 11-17 12:29). Somebody about to walk across the
# way out is waited for in the car, doors open, where the two of them stand
# together (exit_wait); out of it, until EXIT_CLEAR_M from the waiting point,
# only somebody right in front of the dog stops it (right_in_front).
EXIT_CLEAR_M = 1.8        # m from lift.WAIT_XY that is clear of the door
EXIT_FRONT_M = 1.5        # m ahead, and
EXIT_LANE_M = 0.6         # m either side of the dog's heading: "right in front"
EXIT_SEE_T = 1.0          # s on that a walker is looked at, as well as now
EXIT_ZONE_R = 1.2         # m round the way out a walker must keep, from the car
EXIT_WAIT_T = 3.0         # s ahead the car looks at walkers' paths
EXIT_WAIT_MAX_S = 20.0    # s, then out anyway: a corridor that never clears
STAIR_V = 0.25            # m/s along a flight: a tread at a time, a third of corridor pace
# CE-RRT* searches on the control thread, ~160 ms each on the J4012 (25 W).
# A search that found nothing used to be repeated on every tick, at both of
# ce_rrt.SPEEDS in turn: ~320 ms ticks, the loop at ~3 Hz beside people
# (HIL 11-12 13:10: 118 searches, 55 found nothing, 28 s of the run). Now one
# speed a search, the next speed on the next, and after one that found
# nothing the dog waits as it would with no plan (the fallback the chance
# constraint rests on) and searches again this much later. In ticks, not on
# the clock: the plans stay repeatable (ce_rrt's docstring, Gate D).
PLAN_RETRY_S = 0.5
# While a plan is being walked, a new one used to be searched for every
# ce_rrt.REPLAN_S (0.1 s) whether or not the old one was still good: ~400 ms
# a search on the J4012 (HIL 11-12 13:43: 16 searches in one going-round,
# ~7 s, worst 741 ms). Now, inside this long of the last search, the plan is
# only checked (ce_rrt.still_good, cheap) and kept if it passes; it is
# searched for again when the check fails or this has passed. Safety is still
# checked every REPLAN_S; only looking for a better plan slows to 2 Hz. In
# ticks, like PLAN_RETRY_S, so plans stay repeatable.
PLAN_HOLD_S = 0.5
# VAMOS is asked back to back: as soon as an answer lands, from the newest
# frame. It used to be asked on 20-tick boundaries (10 while blocked), and an
# answer that landed between two of them waited for the next, ~0.5 s a cycle
# on the J4012 (2.6 s a call). VAMOS watches all the time; there are no
# event-only calls.
#
# How fast the dog may walk on VAMOS's word: an answer is planned from a
# frame taken L before it lands, and it is followed until the next answer
# lands, another L later: 2 L from frame to fresh plan. In that time the dog
# must not walk off the end of what VAMOS planned, less the room to stop.
# VLM_REACH_D is how far ahead VAMOS plans: HIL steer 11-19 (merged_L, 403
# answers), dog to chosen path end median 3.3 m, p75 3.5 m; the short ones
# end at the destination, where the dog stops anyway. People and walls are
# not on this cap: the LiDAR and the planner check safety every REPLAN_S.
# At the median the cap is ~MAX_V at L = 1.8 s (merged_L, 0.81 m/s): it only
# bites when VAMOS is slower (2.8 s: 0.52 m/s). 3.0 m (0.72 m/s) failed
# Gate D D6 on its own (seed 1, 11-19: 1.20x the map-only time).
VLM_REACH_D = 3.3         # m ahead VAMOS plans (chosen path end, median)
VLM_STOP_D = 0.4          # m to stop from walking pace
VLM_CAP_N = 5             # the cap goes on the slowest of the last this many answers
# Shadow mode: a proposal whose heading is further than this off the map's
# steering target is logged as the dream wanting to go somewhere else.
SHADOW_DISAGREE = math.radians(15)
# Flatter and closer than the single-floor demo: every storey now has a slab
# over it, and the old -28 degree chase camera sat inside the ceiling, which
# rendered as solid grey.
CHASE_D, CHASE_EL = 3.2, -11
# How far a door sign sits off the corridor centreline, towards its own room.
# The corridor is 2.7 m wide, so this puts it essentially on that room's wall --
# and, more to the point, on the correct side of a corridor whose two facing
# rooms share a single door_xy in locations.json.
SIGN_OFF = 1.25
# Azimuth, distance, elevation for the lift -- see frame(). Azimuth is the
# direction the camera looks along, so 0 puts it west of the shaft looking in
# through the open face.
LIFT_CAM = (0, 3.4, -6)
MAX_TICKS = 6000          # per leg

# A leg is done anywhere within GOAL_R of its last waypoint, which is half a
# metre of slop -- fine outside a room, too much in front of a lift, where it
# is the difference between stopping at the doors and stopping in them. Lift
# legs are held to a tighter radius, and the standoff has to beat it.
LIFT_GOAL_R = 0.25
assert lift.STANDOFF > LIFT_GOAL_R, \
    f"lift.STANDOFF ({lift.STANDOFF}) must exceed LIFT_GOAL_R ({LIFT_GOAL_R})"

# What to do when the LiDAR has found something and VAMOS has nothing that
# clears it. Not one rule but three, because "stop" and "carry on" are both
# wrong: crawl while asking again, and only stop once it is close and the
# answer has not changed. The spec wants a pedestrian avoided *without* full
# stops (L6 acceptance); a full stop is what is left when avoidance has failed.
CRAWL = 0.30              # fraction of top speed while looking for a way round
DETOUR = 0.70             # ...and while actually stepping around something
# There is deliberately no "clearance ahead" stop threshold. free_destination is
# the single authority on whether a way past exists -- it validates the line to
# the goal itself -- and a second test of the same line against a different
# number is how the dog ended up halting in gaps it was successfully using.
# What is left is the honest emergency: ROBOT_R, the distance at which the dog
# is touching something rather than approaching it.
BLOCKED_S = 8.0           # seconds stopped with no way past before giving up
# Going round something and getting nowhere, which is not the same as being
# stopped: the detour had an aim, the dog was told to walk it, and something
# downstream held it still. Beside a puddle that left 0.5 m one side and 0.8 m
# the other it stood 280 s without a word (twin, floor guard, seed 1, 11-16).
# A person waiting on the handle cannot see that; after BLOCKED_S of it the
# dog says so and stops, as when no way past was found at all.
PROGRESS_M = 0.20         # m the dog must move in BLOCKED_S while going round something
EXCUSE = "Excuse me, may we pass?"
EXCUSE_S = 6.0            # s between asking a person in the way to let us by
# Blocked is blocked, whatever is there: a crate and somebody standing still
# are both gone round. Only when there is no way round at all does the dog
# stand and ask, every EXCUSE_S, for ASK_S -- whatever it is may be somebody
# who can hear, and move. Still blocked after that, it is not somebody, or
# not somebody who will move: the dog says so, the spot is closed on the
# map (BLOCK_HALF, BLOCK_SPAN) and the rest of the trip is planned again
# from where it stands -- another corridor, or the stairs if the person says
# yes. Only with no other way at all does it stop. (Telling crates from
# people by width and height, and waiting on people only, stood the dog in
# front of anything person-sized asking it to move: HIL 11-17.)
ASK_S = 20.0              # s of asking before finding another way
BLOCK_AHEAD = 2.5         # m ahead of the dog the thing in the way is looked for
BLOCK_HALF = 0.30         # m either side of it, along the way, closed on the map
BLOCK_SPAN = 2.5          # m either side of the way, across it, closed with it
MAX_REROUTES = 3          # other ways tried on one trip before stopping
ASK_NEAR = 1.5            # m: no way round, and the thing in the way this near: stop and ask
APPROACH_LAT = 0.9        # m either side of the dog's line that counts as in the way
# Something inside the LiDAR's minimum range ahead. The rays start at the
# dog's centre and MIN_R is 0.35 m (sensing/lidar.py, go2_world mid360), so
# from 5 cm past its nose nothing is seen at all: somebody stood 0.38 m ahead
# at the start was not one return, and the dog walked 0.17 m into them at
# 0.75 m/s (--pedestrians 4 --seed 3 --puddles 2, 11-16). What does show is
# the hole: a forward sector that should have its share of the returns (the
# floor, the ceiling, the walls: 0.96-1.12 of it, every tick, lift included,
# 0 and 4 people) and has none. Under BLIND_FRAC of that share, nothing moves.
BLIND_AZ = math.radians(25)   # half-width of the forward sector
BLIND_FRAC = 0.30         # of its share of the scan, below which it is blind
BLIND_MIN = 200           # returns in the scan before the share means anything
# Behind, the rays point up (MOUNT_PITCH) and under a high ceiling find
# nothing: 0.79 of the share in the corridors, down to 0.20 on floor 3. So
# only below this is it something there -- and it only ever stops the dog
# backing away (BACK_V), into somebody it cannot see: 0.32 m behind it at the
# start, --pedestrians 4 --seed 11 --puddles 2, 11-16.
BLIND_REAR_FRAC = 0.10
BACK_CLEAR = 0.60         # m behind the dog that must hold no return at all to back off
# The last word on the command, under the handle and the board: never
# towards somebody walking nearer than KEEP_OFF. Only translation is cut:
# turning on the spot does not close on anybody. Things standing still are
# gone round like any other (see ASK_S), at the detours' own margins.
KEEP_OFF = 0.70
# Run.respect_board: a robot with its own LiDAR guard (go2_board). Free
# distance ahead, as the board measures it, under which pushing on is pointless
# (the guard stops at 0.30 m and crawls below 1 m), the free distance that ends
# the turn, how fast to turn with a person on the handle, and how long to try.
BOARD_NEAR_M = 0.45
BOARD_CLEAR_M = 0.80
BOARD_TURN_W = 0.4         # rad/s
# How far to look is how far the dog has turned, not how long it was asked:
# 8 s at BOARD_TURN_W was meant as about half a turn, but go2_flat_v5 turns at
# 0.175 rad/s when asked 0.4 (44 %; 75 % at 0.8), so 8 s was 80 deg, and the
# lift lobby on floor 3 gave up facing a wall it had turned only part way
# from (J4012, 2026-11-05). So a sweep: BOARD_SWEEP_RAD one way, back through
# the start to as far the other way, then help -- or BOARD_GIVE_UP_S for a dog
# that is not turning at all.
BOARD_SWEEP_RAD = math.pi / 2
BOARD_GIVE_UP_S = 30.0
# The plan's turn the other way is a new way round only when it wants to face
# somewhere new, by this much, not the same heading the turn carried past.
BOARD_PLAN_MOVED = 0.3     # rad
BOARD_STALL_S, BOARD_STALL_M = 2.0, 0.05   # asked forward this long, moved less: stuck
# Before turning, a sidestep: the guard closes the front for anything within
# BODY_HALF_W + MARGIN (0.30 m) of the dog's centre line, so a dog 0.25 m from
# a trolley's side is stopped by its corner however clear the way past it is;
# turning on the spot swings the nose into the trolley or the wall, and in the
# 0.90 m gap between them both ways stayed closed (every time, at (23.2, 8.6)).
# A few centimetres sideways, towards the side the board sees more room on,
# puts the corner outside the band. Never backwards; slow, with a person on
# the handle; only towards a side with more than the guard's 0.30 m stop.
BOARD_SIDE_V = 0.12        # m/s sideways
BOARD_SIDE_M = 0.25        # m sideways at most, then turn
BOARD_SIDE_S = 3.0         # s sideways at most, then turn
BOARD_SIDE_ROOM = 0.35     # m free on that side to step into it at all
# ...unless the plan is turning: then the turn is the plan's, not the roomier
# side's. Facing a trolley at 30 deg in that gap, the plan asked -1.0 rad/s
# (right, round it) and the roomier side was left: stepping left took the nose
# closer to the corner (front 0.15 -> 0.08 m) until it turned right after all.
BOARD_PLAN_W = 0.3         # rad/s the plan asks, above which its way is the way
# ...and while turning, too: the turn started right (the plan barely turning,
# the roomier side right), then the plan asked left at 0.3-1.2 rad/s for 4 s
# while the dog turned on into the wall (-0.5 -> -1.16 rad), and the 8 s ran out
# on the way back, 1.4 s short of facing the way on. So the turn follows the
# plan once it has asked the other way this many ticks in a row (20 Hz).
BOARD_PLAN_TICKS = 3
LOOKAHEAD = 3.0           # metres of the route ahead tested for obstructions
BLOCKED_TICKS = 5         # consecutive ticks with no clear goal before believing it
# ...and with the committed side gone before changing sides. One stray Mid-360
# return beside the floor-1 trolley (board L 9.15 -> 3.60 -> 9.15 in one tick)
# emptied the left for a tick, free_destination fell back to both sides and
# said right, and since `lean` followed it the dog was committed to the 0.90 m
# south gap with the 1.10 m north one open again a tick later -- told "left",
# led right, and stuck on the trolley's corner (HIL 15:24 and 15:29).
SIDE_FLIP_TICKS = BLOCKED_TICKS


def lane_point(xy, waypoints, i, look=None):
    """`look` metres along the route from the dog's foot on its current segment
    (waypoints[i-1] -> waypoints[i]), or the route's end."""
    a = waypoints[i - 1] if i > 0 else xy
    b = waypoints[i]
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg = dx * dx + dy * dy
    t = 0.0 if seg < 1e-12 else min(max(((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / seg, 0.0), 1.0)
    p = (a[0] + t * dx, a[1] + t * dy)
    left = LANE_LOOK if look is None else look
    for q in waypoints[i:]:
        d = math.dist(p, q)
        if d >= left:
            return (p[0] + (q[0] - p[0]) * left / d, p[1] + (q[1] - p[1]) * left / d)
        left -= d
        p = (q[0], q[1])
    return tuple(waypoints[-1])


def route_offset(xy, waypoints, i):
    """How far the dog is from the rest of its route: the least distance to
    any segment from waypoints[i-1] on."""
    best = math.dist(xy, waypoints[i])
    for a, b in zip(waypoints[max(i - 1, 0):], waypoints[max(i - 1, 0) + 1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = dx * dx + dy * dy
        t = 0.0 if seg < 1e-12 else min(max(((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / seg, 0.0), 1.0)
        best = min(best, math.hypot(a[0] + t * dx - xy[0], a[1] + t * dy - xy[1]))
    return best


def in_lane(pt, a, b, tol=None):
    """`pt` moved across the line a -> b to within `tol` of it."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return pt
    nx, ny = -dy / n, dx / n                     # left-hand normal
    lat = (pt[0] - a[0]) * nx + (pt[1] - a[1]) * ny
    tol = LANE_TOL if tol is None else tol
    if abs(lat) <= tol:
        return pt
    shift = math.copysign(tol, lat) - lat
    return (pt[0] + nx * shift, pt[1] + ny * shift)


def passes_by(xy, yaw, t, horizon=None):
    """Is track `t` walking along past the dog, PASS_LAT to one side of it the
    whole way? Its sideways offset from the dog's heading, now and `horizon`
    on, on the same side and at least PASS_LAT; moving across slower than
    PASS_LAT_V."""
    c, s = math.cos(yaw), math.sin(yaw)
    lat = -(t.x - xy[0]) * s + (t.y - xy[1]) * c
    lat_v = -t.vx * s + t.vy * c
    if abs(lat_v) > PASS_LAT_V:
        return False
    end = lat + lat_v * (PASS_HORIZON if horizon is None else horizon)
    return lat * end > 0 and min(abs(lat), abs(end)) >= PASS_LAT


def right_in_front(xy, yaw, t):
    """Is track `t` right in front of the dog: EXIT_FRONT_M ahead at most,
    within EXIT_LANE_M of its heading -- now, or EXIT_SEE_T on? The one person
    still waited for while clearing the lift door (see EXIT_CLEAR_M)."""
    c, s = math.cos(yaw), math.sin(yaw)
    for px, py in ((t.x, t.y), t.predict(EXIT_SEE_T)):
        dx, dy = px - xy[0], py - xy[1]
        ahead, lat = dx * c + dy * s, -dx * s + dy * c
        if 0.0 < ahead <= EXIT_FRONT_M and abs(lat) <= EXIT_LANE_M:
            return True
    return False


def crossing_exit(t, way_out):
    """Will walker `t` come within EXIT_ZONE_R of the way out of the lift
    (`way_out`, points from the door to EXIT_CLEAR_M out) in the next
    EXIT_WAIT_T? Asked from inside the car, where the dog is not moving."""
    if not t.moving:
        return False
    return any(closest_approach(p, (0.0, 0.0), (t.x, t.y), (t.vx, t.vy), EXIT_WAIT_T)
               - min(t.r, 0.4) < EXIT_ZONE_R for p in way_out)


def blind_ahead(scan, pose, behind=False):
    """Is the forward sector missing its returns -- something inside MIN_R?
    See BLIND_FRAC. Only from the points, so the board's cloud works too.
    `behind`: the sector behind instead, against BLIND_REAR_FRAC."""
    if len(scan) < BLIND_MIN:
        return False
    x, y, yaw = pose
    frac = BLIND_REAR_FRAC if behind else BLIND_FRAC
    az = np.arctan2(scan[:, 1] - y, scan[:, 0] - x) - yaw - (np.pi if behind else 0.0)
    az = (az + np.pi) % (2 * np.pi) - np.pi
    share = len(scan) * BLIND_AZ / np.pi
    return int((np.abs(az) < BLIND_AZ).sum()) < frac * share


def closing_on(tracks, pose, cmd):
    """Would body-frame `cmd` close on somebody walking, inside KEEP_OFF?"""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy = cmd[0] * c - cmd[1] * s, cmd[0] * s + cmd[1] * c
    for t in tracks:
        if not t.moving:
            continue
        d = math.hypot(t.x - x, t.y - y)
        if 1e-6 < d < KEEP_OFF and (vx * (t.x - x) + vy * (t.y - y)) / d > 0.0:
            return True
    return False


def something_behind(points, pose):
    """Any unexplained return in the half behind the dog within BACK_CLEAR?
    Backing off (BACK_V) only into nothing at all: somebody stood 0.32 m off
    its back quarter, mostly inside MIN_R, showed only as a few returns past
    it, and the dog backed into them (--pedestrians 4 --seed 11, 11-16)."""
    if not len(points):
        return False
    x, y, yaw = pose
    dx, dy = points[:, 0] - x, points[:, 1] - y
    back = dx * math.cos(yaw) + dy * math.sin(yaw) < 0.0
    return bool((back & (np.hypot(dx, dy) < BACK_CLEAR)).any())


def nearest_ahead(points, pose):
    """Metres along the dog's line to the nearest unexplained return in front
    of it within APPROACH_LAT of that line; INF for none. See ASK_NEAR."""
    if not len(points):
        return INF
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    along = dx * c + dy * s
    m = (along > 0.0) & (np.abs(-dx * s + dy * c) < APPROACH_LAT)
    return float(along[m].min()) if m.any() else INF


def blocked_point(points, pose):
    """Where the thing in the way is: the nearest unexplained return in front
    of the dog within BLOCK_AHEAD, or, with none (inside the LiDAR's MIN_R:
    see BLIND_FRAC), half a metre ahead of it. See ASK_S."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    if len(points):
        dx, dy = points[:, 0] - x, points[:, 1] - y
        along = dx * c + dy * s
        d = np.hypot(dx, dy)
        m = (along > 0.0) & (d < BLOCK_AHEAD)
        if m.any():
            i = np.flatnonzero(m)[np.argmin(d[m])]
            return float(points[i, 0]), float(points[i, 1])
    return x + 0.5 * c, y + 0.5 * s


def overtaking(xy, yaw, t, speed=0.0):
    """Is track `t` somebody walking the dog's way who is coming up behind it
    -- behind its centre -- or is level with or ahead of it and pulling away
    (faster than it by PULL_DV), and not cutting across? Nothing the dog does
    helps them by stopping: they pass it, or are gone. Waiting for one stood
    the dog still, stepping left and right, for 5 s with 9 announcements (HIL
    11-10 11:24); and somebody 1 m ahead walking off at 1.3 m/s was "Someone
    ahead. Waiting." (11:45), closer than MISS_R and leaving."""
    if not (t.moving or t.walked):
        return False
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = (t.x - xy[0]) * c + (t.y - xy[1]) * s
    v_fwd = t.vx * c + t.vy * s
    v_lat = -t.vx * s + t.vy * c
    if v_fwd < CROSS_V or abs(v_lat) > max(PASS_LAT_V, 0.5 * v_fwd):
        return False
    return fwd < 0.0 or v_fwd > speed + PULL_DV


def lane_dir(xy, waypoints, i, yaw):
    """Unit vector along the lane the dog is walking: its current route
    segment, or its heading when there is none yet."""
    a = waypoints[i - 1] if i > 0 else xy
    b = waypoints[i]
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return (math.cos(yaw), math.sin(yaw))
    return (dx / n, dy / n)


def leader(xy, u, tracks, speed, lane_off=0.0, keep=None):
    """The nearest person ahead in the dog's lane walking its way slower than
    it, within FOLLOW_LOOK; or None. Ahead means past OVERTAKE_AHEAD: nearer
    than that they are beside it, and somebody overtaking at 0.85 m/s, level
    with the dog 0.68 m to its left, stood it still as a leader 0.09 m
    ahead (HIL 11-10 15:42, t 35.1). `u` is the lane direction; `lane_off`
    how far left of its lane the dog is walking (an overtake), so that the
    lane is the lane and not the dog. `keep`: the one already being
    followed, who stays the leader down to LEAD_KEEP_V (see there)."""
    best, bd = None, FOLLOW_LOOK
    for t in tracks:
        if not (t.moving or t.walked):
            continue
        fwd = (t.x - xy[0]) * u[0] + (t.y - xy[1]) * u[1]
        lat = -(t.x - xy[0]) * u[1] + (t.y - xy[1]) * u[0] + lane_off
        v_fwd = t.vx * u[0] + t.vy * u[1]
        v_lat = -t.vx * u[1] + t.vy * u[0]
        if not OVERTAKE_AHEAD < fwd < bd or abs(lat) >= PASS_LAT:
            continue
        v_min = LEAD_KEEP_V if t is keep else CROSS_V
        if v_fwd < v_min or abs(v_lat) > max(PASS_LAT_V, 0.5 * v_fwd):
            continue                       # not walking the dog's way
        if v_fwd >= speed:
            continue                       # not closing on them: nothing to follow
        best, bd = t, fwd
    return best


def follow_speed(gap, v_lead):
    """Forward speed that keeps FOLLOW_GAP behind somebody walking at
    `v_lead`: their pace, plus FOLLOW_K per metre too far back; nothing at
    FOLLOW_MIN or closer. Never backwards (see BACK_V)."""
    if gap <= FOLLOW_MIN:
        return 0.0
    return max(0.0, v_lead + FOLLOW_K * (gap - FOLLOW_GAP))


def straight_ahead(xy, route):
    """Metres of route ahead of `xy` before it turns by more than PLAN_CORNER
    (or ends)."""
    pts = [tuple(xy)] + [tuple(p) for p in route]
    walked, heading = 0.0, None
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if d < 1e-6:
            continue
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        if heading is not None and abs(wrap(h - heading)) > PLAN_CORNER:
            return walked
        heading = h
        walked += d
    return walked


def overtake_plan(xy, u, lead, others, cruise, shift=None, started=False, v_lead=None):
    """(seconds, metres, their pace) an overtake of `lead` takes at `cruise` -- from where
    the dog is to OVERTAKE_BACK past them, on the line `shift` left of where
    the dog is (OVERTAKE_LAT left of its lane) -- or None when it is not worth one (less than OVERTAKE_DV
    faster than them) or somebody else would come within OVERTAKE_CLEAR of
    that line meanwhile: oncoming people in the other lane, mostly.
    `started`: a pass already under way, which only somebody else stops.
    `v_lead`: their pace, if known better than this tick's velocity."""
    if v_lead is None:
        v_lead = lead.vx * u[0] + lead.vy * u[1]
    dv = cruise - v_lead
    if dv < OVERTAKE_DV:
        if not started:
            return None
        # Once under way, their measured pace is noise round what it was
        # (0.3-0.6 m/s for somebody walking 0.42, HIL 11-10 16:11): it does
        # not call the pass off, and every tick it did was a new one.
        dv = OVERTAKE_DV
    gap = (lead.x - xy[0]) * u[0] + (lead.y - xy[1]) * u[1]
    secs = (gap + OVERTAKE_BACK) / dv
    shift = OVERTAKE_LAT if shift is None else shift
    start = (xy[0] - u[1] * shift, xy[1] + u[0] * shift)
    vel = (u[0] * cruise, u[1] * cruise)
    for t in others:
        if t is lead:
            continue
        d = closest_approach(start, vel, (t.x, t.y), (t.vx, t.vy), horizon=secs)
        if d - min(t.r, 0.4) < OVERTAKE_CLEAR:
            return None
    return secs, cruise * secs, v_lead


class Pace:
    """How much of a straight forward command the dog actually walks, by its
    own clock: metres along its heading per second against the command that
    moved it, each smoothed over PACE_TAU, counted once PACE_TAU of straight
    walking has been seen. Overtaking is judged on what the dog walks, not
    what it is told: in HIL the physics runs ~0.83x the board's clock, and
    somebody walking 0.65 m/s read 0.54 against a commanded 0.8 -- 0.26
    slower, so overtaken, when the dog was really doing 0.67 (HIL 11-10
    16:13). The real dog's policy need not walk 0.8 m/s for 0.8 either."""

    def __init__(self):
        self.v = self.c = self.seen = 0.0
        self.prev = None

    def update(self, t, x, y, yaw, cmd):
        """The pose now (t: the brain's clock) and the command sent now."""
        if self.prev is not None:
            t0, x0, y0, yaw0, c0 = self.prev
            dt = t - t0
            if (0.0 < dt < 0.5 and c0[0] >= PACE_MIN_CMD and abs(c0[1]) < 1e-6
                    and abs(c0[2]) < PACE_MAX_W):
                v = ((x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)) / dt
                a = min(1.0, dt / PACE_TAU)
                self.v += a * (v - self.v)
                self.c += a * (c0[0] - self.c)
                self.seen += dt
        self.prev = (t, x, y, yaw, tuple(cmd))

    @property
    def known(self):
        """Seen enough straight walking to believe: until then nobody is
        overtaken -- the first one at 16:21 (HIL 11-10) was judged on 0.8 s
        of it, the command taken for the pace, and somebody to follow was
        passed instead."""
        return self.seen >= PACE_TAU and self.c > 0.0

    @property
    def ratio(self):
        if not self.known:
            return 1.0
        return min(max(self.v / self.c, PACE_LO), 1.0)


def small_thing_beside(info, tracks, static_at):
    """Something person-sized standing on the other side of the pinch from the
    crossing free_destination chose, on floor the map calls open: the reason
    the wider way is shut, maybe for only a moment. Moving or not -- a person
    standing still is a cluster of returns like any other. None if not."""
    (px, py), (nx, ny), (ux, uy) = info["pinch"], info["normal"], info["along"]
    for t in tracks:
        if not MIN_RADIUS <= t.r <= SMALL_R:
            continue
        lat = (t.x - px) * nx + (t.y - py) * ny
        along = (t.x - px) * ux + (t.y - py) * uy
        if (lat * info["side"] < 0 and 0.2 < abs(lat) < 1.5 and abs(along) < 1.2
                and static_at(t.x, t.y) >= ROBOT_R):
            return t
    return None


def axis_clear(info, detected_at, need=None):
    """The line along a narrow gap, through the middle of the crossing, from
    AXIS_BACK before it to AXIS_ON past it, keeps `need` (ROBOT_R + 5 cm)."""
    (px, py), (nx, ny), (ux, uy) = info["pinch"], info["normal"], info["along"]
    off = info["want"] * info["side"]
    cx, cy = px + nx * off, py + ny * off
    need = ROBOT_R + 0.05 if need is None else need
    s = -AXIS_BACK
    while s <= AXIS_ON + 1e-9:
        if detected_at(cx + ux * s, cy + uy * s) < need:
            return False
        s += 0.1
    return True


def narrow_axis(info, xy, detected_at):
    """A narrow gap's axis as the dog sees it: (centre of the crossing, axis
    direction, the dog's metres left of the axis, metres along it from the
    dog to the entrance). The axis runs along the route through the middle
    of the crossing; the entrance is where, coming back from the crossing
    towards the dog, the room opens up."""
    (px, py), (nx, ny), (ux, uy) = info["pinch"], info["normal"], info["along"]
    off = info["want"] * info["side"]
    cx, cy = px + nx * off, py + ny * off
    entrance = 0.0
    while entrance < 3.0 and detected_at(cx - ux * entrance, cy - uy * entrance) < info["room"] + 0.2:
        entrance += 0.1
    lat = (xy[0] - cx) * nx + (xy[1] - cy) * ny
    to_go = (cx - xy[0]) * ux + (cy - xy[1]) * uy
    return (cx, cy), (ux, uy), lat, to_go - entrance


def narrow_aim(info, xy, detected_at):
    """Where to steer through a narrow gap: a point on its axis ALIGN_LOOK
    ahead of the dog (further when the dog is far off it, so that the line
    to it is no steeper than ALIGN_MAX_DEG -- unless that would put the point
    beyond the entrance: the axis is to be reached by then, and a gentle line
    from close by runs into what makes the gap narrow). Followed, the dog
    curves onto the axis and reaches the entrance along it, not at the angle
    it set off at -- see ALIGN_LOOK."""
    (cx, cy), (ux, uy), lat, to_entrance = narrow_axis(info, xy, detected_at)
    s = (xy[0] - cx) * ux + (xy[1] - cy) * uy          # the dog, along the axis
    look = abs(lat) / math.tan(math.radians(ALIGN_MAX_DEG))
    if to_entrance > 0.0:
        look = min(look, to_entrance)
    look = max(look, ALIGN_LOOK)
    return (cx + ux * (s + look), cy + uy * (s + look))


def leaving_narrow(last, xy):
    """Still within NARROW_EXIT_M of where the dog last steered through a
    narrow way (`last`, or None)?"""
    return last is not None and math.dist(last, xy) < NARROW_EXIT_M


def narrow_lined_up(info, xy, yaw, detected_at):
    """False while the dog is within ALIGN_GATE of a narrow gap's entrance
    (or in it) and not yet on its axis: more than ALIGN_TOL off it, or
    heading more than ALIGN_YAW_DEG across it. It then walks at ALIGN_SLOW_V,
    so the turn it is being given has the time to happen."""
    _, (ux, uy), lat, to_entrance = narrow_axis(info, xy, detected_at)
    if to_entrance > ALIGN_GATE:
        return True
    across = abs(wrap(yaw - math.atan2(uy, ux)))
    return abs(lat) <= ALIGN_TOL and across <= math.radians(ALIGN_YAW_DEG)


def hold_side(lean, aim, offset, held, against, ticks=SIDE_FLIP_TICKS):
    """The detour this tick, given the side already committed to.

    `held` is the last (aim, offset) on the committed side and `against` how
    many ticks free_destination has offered the other side since it last
    offered this one. Under `ticks` the old aim stands; at `ticks` the new
    side is believed. A tick with no offset (straight on, or no way past)
    changes nothing here: that is BLOCKED_TICKS' business.
    Returns (aim, offset, lean, held, against).
    """
    side = math.copysign(1.0, offset) if offset else 0.0
    if not side:
        return aim, offset, lean, held, against
    if lean and side != lean:
        against += 1
        if against < ticks and held is not None:
            return held[0], held[1], lean, held, against
    return aim, offset, side, (aim, offset), 0
CLOUD_EVERY = 4           # --cloud-log keeps every 4th scan: 5 a second
# A gap the board's guard will only let the dog through lined up: its front
# corridor is BODY_HALF_W + MARGIN = 0.30 m either side of the heading, so a
# 0.90 m gap (0.45 m from the middle to either side) passes straight and not
# at an angle. With somebody standing in the trolley's 1.10 m north gap the dog
# took the 0.90 m south one on a slant: 94-102 guard interventions, "too close
# ... turning left" two or three times, 16-17 s past a trolley that takes 9.6
# (HIL 11-10 12:48, 12:51), and pinned for 30 s on 11-09 13:28.
NARROW_ROOM = 0.50        # m, middle of the crossing to the nearest thing: under
                          # this a pass is narrow (south gap 0.45; north 0.50-0.60)
NARROW_WAIT_S = 6.0       # s stood before a narrow way, while something person-
                          # sized stands in the wider one: it may well move
SMALL_R = 0.30            # m, the most a person standing still measures
                          # (tracking.MIN_RADIUS is the least)
ALIGN_TOL = 0.10          # m off a narrow gap's axis that still counts as lined up
ALIGN_LOOK = 1.0          # m ahead along a narrow gap's axis the dog steers at.
                          # It used to steer at a fixed point on the axis 0.5 m
                          # before the entrance, and arrived there at the angle
                          # it set off at: -32 deg, nose 0.31 m from the south
                          # wall, held by the board's guard 3 s and pivoting 2 s
                          # (HIL 11-10 13:14); 1.2 m before it made the cut -72
                          # deg and the route failed (13:23). A point that moves
                          # on with the dog brings it onto the axis on a curve.
ALIGN_MAX_DEG = 35.0      # the steepest the line to that point may be: far off
                          # the axis, the point is further ahead
ALIGN_GATE = 0.5          # m before the entrance from which it must be lined up...
ALIGN_YAW_DEG = 10.0      # ...within ALIGN_TOL and this of the axis' heading...
ALIGN_SLOW_V = 0.2        # ...or it walks this slowly (m/s) until it is
NARROW_EXIT_M = 3.0       # m walked after a narrow way on the lane point, not on
                          # VAMOS: out of the trolley's south gap the dog is on its
                          # axis, 0.49 m from the wall; VAMOS's paths start where
                          # the dog is and spread +-0.25 m, all along the wall, and
                          # their imagined runs (3.5 deg of heading error) came
                          # within ~0.2 m of it -- safety 0.35-0.50, so 0.28-0.40
                          # m/s, for 14 m and 45 s, never back in its lane 0.23 m
                          # away (LANE_TOL 0.3 does not pull it in; HIL 11-11 10:36)
AXIS_BACK, AXIS_ON = 1.0, 1.5   # m of a narrow gap's axis, before and past the
                          # crossing, that must be clear for it to count: room is
                          # judged at the pinch only, and with somebody standing
                          # 0.5 m further along the trolley's north gap a crossing
                          # beside them read 0.46 m and the dog walked at them
                          # (HIL 11-10 12:57, failed)
PROBE_D = 6.0             # metres ahead the blockage is judged at, not the destination's 2-4

# People, who are not crates. A crate is gone round; a person is waited for --
# a guide dog that threads a moving gap is towing someone through it. The
# decision is tracking.conflict(), which asks whether the two paths meet
# rather than whether anyone is near; this is only what happens afterwards.
PATIENCE = 3.0            # seconds a person may stand still before the dog
                          # stops waiting for them and treats them as the
                          # obstacle they have become. Long enough to cover
                          # somebody breaking stride to let the dog past --
                          # that pause is about a second -- and short enough
                          # that a person who has genuinely stopped does not
                          # end the run.
PASS_NEED = 0.60          # going round someone who has stopped: room kept from
                          # what the LiDAR sees, instead of the crates' 0.35.
                          # About 0.8 m centre to centre -- the 0.9 m people
                          # give each other in passing. A crate's 0.35 took
                          # the dog past a standing person at 0.50 m, at full
                          # speed, 9 cm from their shoulder.
PASS_LAT = 0.75           # m, centre to centre across the dog's heading: someone
                          # walking along the corridor who stays this far to the
                          # side, all the way past, is passed at PASS_V, not
                          # waited for. MISS_R (0.95 m surface to centre) made
                          # every passer-by in the next lane of a 2.8 m corridor
                          # a conflict: 41-45 s of 86-97 s on floor 1 stood
                          # still with nothing ahead (HIL 11-09 15:24, 15:29).
                          # A person's 0.22 m, the dog's 0.2 m and 0.3 m between.
PASS_LAT_V = 0.25         # m/s across: slower than this is walking along, not over
OVERTAKE_AHEAD = 0.5      # m in front of the dog's centre someone passing it is
                          # still beside it: "Someone passing" is said up to here
OVERTAKE_SAY_S = 8.0      # s before "Someone passing" is said again on the same side:
                          # a person's track breaks and comes back as a new one,
                          # and once a track was 6 times in 8 s (HIL 11-10 12:25)
PULL_DV = 0.1             # m/s faster than the dog: somebody ahead is pulling away
PASS_HORIZON = 2.5        # s of their walk it has to hold for (tracking.HORIZON)
LANE_LOOK = 2.0           # m along the route ahead of the dog's foot on it that
                          # the map steers at, not the next waypoint: that can be
                          # 20 m off, and a dog 0.5 m out of its lane after a
                          # step round somebody took the length of the corridor
                          # to get back -- on floor 1 it walked the middle (y
                          # 9.25-9.44 for a lane at 8.87, HIL 11-09).
LANE_TOL = 0.30           # m either side of the lane a VAMOS path may take it
PASS_V = 0.40             # m/s while within PASS_R of them: walking past
PASS_R = 2.0              # somebody, not overtaking them
PASS_SIDE = 0.90          # metres either side of the dog's heading a standing
                          # person must be to be passed slowly: PASS_NEED plus
                          # the dog's half-width
# Somebody ahead in the dog's lane walking its way, slower than it: followed,
# not waited for. conflict() saw them as a meeting (the dog closes on them) and
# every one was "Someone ahead. Waiting." or a CE-RRT* plan round them each
# replan. Followed at their pace, and overtaken on the left when that is worth
# it and the left is clear for as long as it takes.
FOLLOW_LOOK = 6.0         # m ahead along the lane a leader is looked for
FOLLOW_GAP = 1.5          # m, dog centre to theirs, kept behind them
FOLLOW_K = 0.5            # m/s per metre of gap over FOLLOW_GAP: closes it gently
FOLLOW_MIN = 1.0          # m: at or under this the dog stands
OVERTAKE_DV = 0.25        # m/s slower than the dog's pace before overtaking: less
                          # and the pass takes 12 s and 10 m of corridor
OVERTAKE_LAT = 1.2        # m left of the lane the dog passes on: FOLLOW_MIN
                          # beside them, and a margin -- steering at a lane point
                          # LANE_LOOK ahead only creeps up on the shifted line
                          # (1.0 reached 0.95 by the time it drew level). Floor
                          # 1: lane 8.87, line 10.07, north wall 10.85.
                          # (PASS_LAT's 0.75 would be closer beside them than
                          # behind them.)
OVERTAKE_BACK = 1.5       # m past them before it moves back into its lane
OVERTAKE_CLEAR = 1.0      # m anybody else must stay from the passing line
PACE_MIN_CMD = 0.4        # m/s forward commanded before a tick counts for Pace
PACE_MAX_W = 0.2          # rad/s of turn under which it counts: walking straight
PACE_TAU = 2.0            # s smoothed over, and seen before it is believed
PACE_LO = 0.5             # the least fraction believed (a stall is not a pace)
LEAD_TAU = 2.0            # s the pace of somebody ahead is smoothed over, and
LEAD_SEEN = 1.0           # s they are watched before an overtake is judged on it:
                          # one tick read 0.41 m/s of somebody walking 0.54, the
                          # next 0.80 (HIL 11-10 16:26), and the 0.41 was passed
LEAD_KEEP_V = 0.1         # m/s along the lane the one being followed must keep up
                          # to stay the leader (a new one needs CROSS_V): somebody
                          # walking 0.33 (brain clock) read 0.24-0.41 tick to tick,
                          # dropped out under CROSS_V every other tick, and
                          # LEAD_SEEN never came round (HIL 11-11 09:37, --walker 0.4)
LEAD_GRACE = 0.3          # s the one being followed may go unseen as the leader
                          # (a missed tick) before their pace and LEAD_SEEN restart
REACQUIRE_R = 0.6         # m from where the one being overtaken was heading that
                          # a new track is taken for them (tracks break)
# No stepping back from them first. It was tried: reversing pushes the rigid
# handle into the person holding it, and nothing the dog senses knows they
# are there -- seed 8 of room 201 backed them into a wall for 9 s. A standoff
# with a pedestrian is broken on their side instead (pedestrians.GIVE_WAY).
YIELD_MAX = 30.0          # the backstop under all of it: still waiting after
                          # this long, creep and keep asking. Reachable only
                          # by a queue of people arriving one after another.
# --planner cerrt: going round a person instead of waiting for them, on a
# path planned where they will be (planning/ce_rrt.py). The wait above stays,
# for when no plan is safe -- a person square in a narrow corridor, a doorway.
PLAN_D = 4.0              # metres along the route the plan aims at: past a
                          # person met head-on, with room to rejoin the lane
PLAN_DOOR = 2.0           # metres from a doorway inside which nobody is gone
                          # round: lining up for the door is the funnel's job,
                          # and a sidestep there is the frame
PLAN_CORNER = math.radians(30)  # a route turning more than this is a corner,
PLAN_SHORT = 1.0          # ...and a plan's goal stays this far short of it.
                          # Aimed PLAN_D along the route whatever the route did
                          # in between, the goal before the cafeteria door was
                          # round the corner: the plan cut it, left the dog
                          # 0.6 m off the door's axis nose-first in the frame,
                          # and it stood there until the run timed out (seeds
                          # 1 and 2, 17 s against the wall). Corners stay with
                          # the turn and the funnel.
PLAN_MIN = 1.5            # metres of straight route needed to plan at all
PLAN_NOSE = 0.25          # metres from the dog's front (SWING_NOSE ahead) to a
                          # real wall face: NOSE_HALF_W and a hand's width. The
                          # centre alone was checked, and a plan bending in
                          # towards the cafeteria door held the dog 40 deg to
                          # the corridor wall with its centre 0.35 m off and
                          # its front legs on it (Gate D, --vamos, seed 2).
PLAN_CRATE = 0.30         # metres a plan keeps from anything the LiDAR found
# ...and while it waits. A dog standing still for somebody is still in the way
# of whoever comes next: on room 201 seed 10 (--vamos) a walker cut diagonally
# across the corridor and passed the waiting dog at 0.549 m, inside PED_NEAR,
# with no way round them to plan. So a waiting dog steps sideways, out of the
# line somebody is predicted on. Sideways, never back and never a turn: a
# step along the axis keeps the handle where it is, so the person on it moves
# over with the dog instead of being swung (see BACK_V for why not back).
EVADE_R = 0.80            # metres: predicted closer than this (PED_NEAR + 0.25)
EVADE_T = 1.5             # ...within this many seconds starts a step aside
EVADE_STEP_T = 0.5        # seconds of SIDE_V each step is judged over
EVADE_GAIN = 0.05         # metres a step must win over standing still
EVADE_KEEP = EVADE_R      # metres: no step takes the dog nearer than this to
                          # anybody standing, unless it is already nearer and
                          # the step does not close it further
EVADE_V = 0.30            # m/s: slower tracks are not about to arrive anywhere
                          # that is not walking: ROBOT_R and a hand's width
CROSS_V = 0.3             # m/s: slower than this, across or along, is not a
                          # direction worth telling anybody (describe_person)
SIDE_DEAD = 0.3           # metres off the dog's line still called "ahead"
AFF_EVERY = 5             # ticks between affordance checks in --affordance-shadow (4 Hz)
AFF_HALF_W = 0.3          # metres either side of the judged walk that ground truth looks at
PED_NEAR = 0.55           # ground truth: nearer than this to a person counts
                          # as a contact. Scoring only, never shown the robot.
DOG_CLOSING = 0.05        # ground truth: m/s the dog itself closes on a person,
                          # inside PED_NEAR, before that tick is the dog's doing
                          # and not theirs. Scoring only, as PED_NEAR.

# The person being guided. They walk behind the dog holding a rigid handle
# (RDog's), so they are always straight behind it along its heading: the
# handle does not swivel. Not in the scene -- nothing the robot senses knows
# they are there -- and scored the way the dog's body is: touching a wall, a
# crate, the lift or the stairs counts. A route the dog clears can still walk
# them into something: on a pivot the handle swings them sideways through
# whatever is beside the dog.
HANDLER_BEHIND = 1.1      # metres, dog centre to theirs: the Go2's 0.41 m of
                          # body behind its centre, ~0.4 m of handle, an arm
HANDLER_R = 0.25          # metres, half a person's shoulder width
HANDLER_H = (0.4, 1.0)    # heights tested: crate height, and hip/hand height
HANDLER_RAYS = 12

# Easing a turn that would swing them into something. The dog knows the
# handle is rigid and how long it is, so it knows where the person will be
# after any turn -- and every handler contact measured was a turn: a pivot
# beside the crate it had just gone round (`electrical engineering lab`, the
# cart on `room 201` with people), a left turn past the lift (`cafeteria`),
# and a sharp left into the floor-2 restroom from the north lane after going
# round somebody, which put the person 1.1 m behind into the north wall. So
# each command is rolled forward SWING_T and, where it would take the person
# closer than HANDLER_ROOM to a wall or anything the LiDAR has found, the turn
# is eased and the dog keeps walking: a wider arc drags them round instead of
# swinging them sideways.
HANDLER_ROOM = 0.35       # metres, their centre to a surface: HANDLER_R + 0.10
SWING_T = 0.8             # seconds a command is rolled forward over
SWING_V = 0.30            # m/s walked while easing a turn a pivot would have made
SWING_EASE = (0.5, 0.25, 0.0)   # fractions of the turn rate tried, in order
SIDE_MIN_W = 0.05         # rad/s: a turn rate below this does not pick the side
                          # of a sidestep (see spare_handler)
SIDE_V = 0.20             # m/s sideways, towards the turn, when no eased turn
                          # walking forward is clear. Setting off round somebody
                          # who stopped in its lane, the dog had them in front and
                          # its user 0.5 m from the wall beside it: walking on
                          # meets the one, pivoting swings the other into the
                          # wall (Gate D on main, --vamos, room 101 seeds 2, 3,
                          # 8, 9, room 201 seed 10). A step sideways takes both
                          # away from the wall; the turn follows once there is
                          # room for it.
SWING_NOSE = 0.30         # metres ahead of the dog's centre its front legs reach.
                          # An eased command walks on while the turn waits, so
                          # the front of the body has to stay clear too, not
                          # only the centre: seed 8 of room 201 --pedestrians 3
                          # kept 0.18 m at the centre and put its front legs on
                          # the floor-2 cart's corner.
EASE_AHEAD = 0.50         # metres beyond the front, at the end of the roll-out,
                          # that an eased command walking on where the plan
                          # pivoted must find clear. It is asked again every
                          # tick, so it walks on for as long as the turn is
                          # eased, not for SWING_T: pushed to the south wall by
                          # a sidestep, 1.2 m short of the floor-1 lift and 55
                          # deg right of it, the dog walked straight at the
                          # shaft's corner at 0.3 m/s until the board's guard
                          # closed the front (BOARD_NEAR_M), then crept left at
                          # 0.03 m/s for 10 s and overshot into the lift's mouth
                          # (HIL 11-09 13:02). More than BOARD_NEAR_M, so an
                          # easing the guard will stop is never chosen.
NOSE_MIN = 0.15           # metres from the dog's front (SWING_NOSE ahead) to
                          # anything the LiDAR found, below which it turns
                          # before it walks on. The proximity stop reads the
                          # centre, and the Go2 is twice as long as it is wide:
                          # seed 8 again, centre 0.25 m clear of the cart, front
                          # 0.10 m, walking on at 0.6 m/s while turning.
NOSE_HALF_W = 0.15        # metres either side of the centreline, the front legs
UNPIN_DT = 0.3            # s of a turn on the spot judged ahead (as unpin)
UNPIN_W = 0.4             # rad/s: the only turn allowed while waiting for somebody,
                          # and only to take the person on the handle off a wall.
                          # Frozen mid-pivot to wait, the dog held its user 8 cm
                          # into the floor-3 north wall for 7 s (Gate D, chemistry
                          # lab seed 9, --vamos): 7.5 of the arm's 11.2 s of D2.
SWING_PEOPLE = 0.70       # metres, dog centre to a person's predicted centre,
                          # kept by any step easing adds. PED_NEAR plus a margin.
GLIDE_TOL_M, GLIDE_TOL_YAW = 0.15, 0.20   # m, rad: near enough the end of a glide()
GLIDE_SETTLE_S = 6.0      # s a board may take, after a glide's line, to get there
GLIDE_STUCK_M, GLIDE_STUCK_S = 0.03, 1.0  # m in s: less headway is a board that will get no nearer
HOLD_TURN_S = 3.0         # seconds a turn that would put the person on the handle
                          # into a wall (closer than HANDLER_R) is held, when no
                          # eased or sideways version avoids it, before the
                          # best of those goes ahead anyway. What made it worth
                          # waiting for is usually somebody at their elbow, and
                          # people move on; a wall does not, so never for ever.

# Doorways. The route goes through each one's middle, but the dog is not always
# on the route when it gets there -- stepped round a person, or on a VAMOS
# path, which cuts corners. From
# FUNNEL_D out it steers at a point FUNNEL_LEAD ahead of itself on the door's
# axis, so it closes on the line before the frame instead of arriving at the
# frame and pivoting to find it. That pivot was the floor-2 restroom: 0.5 m
# east of the axis at the threshold, turning to line up, with the crate beside
# the door on the person's side.
FUNNEL_D = 2.0            # metres before a doorway the funnel takes over
FUNNEL_LEAD = 0.8         # metres ahead on the axis it steers at
FUNNEL_PAST = 0.4         # metres beyond the doorway's far side it lets go
FUNNEL_ANGLE = math.radians(45)   # steepest approach to the axis point it steers at;
FUNNEL_PRE = 0.5          # beyond it, the point is this far before the doorway.
                          # From well to the side the point FUNNEL_LEAD ahead is
                          # in the doorway, and the line to it crosses the frame:
                          # out of the floor-3 lift the dog walked 1.6 m west
                          # (easing the turn for the person on the handle), came
                          # back for the server room at 27 deg to the wall, and
                          # put its nose into the door's west jamb -- the board's
                          # guard held it there 30 s (HIL 11-09 13:25, 13:42).
                          # Lined up in the corridor first, it walks in square.
FUNNEL_OFF = 0.5          # ...but only when the dog is more than this off its
                          # route, as it was there. On it, the route's own rounded
                          # corner (lane.round_corners) is the way in: lined up
                          # instead, the dog walked the corridor to 0.5 m before
                          # the electrical engineering lab's door, then pivoted
                          # there and swung its user onto the floor-2 crate
                          # (selftest, 0.7 s, from #4).
DOOR_MAP_D = 4.0          # metres from a doorway on the route inside which the
                          # map's route steers, not a VAMOS path. The route
                          # starts crossing the corridor for a door about this
                          # far out; a VAMOS path kept the dog in its lane past
                          # that point, left a sharp turn at the door, and the
                          # turn-easing for the person on the handle made it
                          # overshoot: chemistry lab, +14 s over map-only.
DOOR_NEAR = 1.0           # metres before a doorway where it lines up before walking:
DOOR_ERR = 0.25           # rad of heading error it will walk with there, against
                          # TURN_ONLY's 0.8 anywhere else. Walking while turning
                          # at the threshold is the body's corner meeting the frame.

# Backing out of a graze. The proximity stop below sets the speed to zero, and
# that used to be all it did -- which is a deadlock, because clearance cannot
# improve while the dog does not move, so every graze ran the full BLOCKED_S
# and ended the leg. Squeezing past the floor-2 cart leaves 0.18 m against a
# 0.16 m threshold, so which side of that a run lands on is decided by two
# centimetres: the empty building cleared it, the same route with people in it
# did not, and the people were nowhere near -- they had simply perturbed the
# dog's line. A two-centimetre difference should cost a step backwards, not a
# failed run.
BACK_V = 0.15             # m/s, backwards -- slow enough to feel deliberate on
                          # the handle rather than like a flinch
BACK_STEP = 0.45          # how far behind is checked before reversing into it


def resolve_stops(router, text, use_nlu=False, guessed=None):
    """Command -> [(destination name, floor or None)], one entry per stop.

    Splitting and floor phrases are rules either way (that is where they live
    in the pipeline). The destination is found cheapest and surest first:

      1. a no-go place named out loud              -> as named (rules)
      2. a map name in the text, as said           -> as named
         ...or once number words are digits        -> a guess
      3. a nickname ("toilet"), or a near miss     -> a guess (place_match)
         on spelling or on sound
      4. a need ("I need to pee", "I'm starving")  -> a guess (place_match)
      5. --nlu only: the Gemma parser, for         -> a guess, and only if
         commands no rule reads ("I could do          its answer is a place
         with a bite")                                on this map

    So with --nlu on, a command that names its place never waits for the
    model, and the model's answer is held to the map like everything else.
    Every guess is appended to `guessed` as (stop index, name, what was
    heard), for main to confirm before the dog moves.
    """
    from cyberdog.language.command_splitter import split_destinations
    from cyberdog.language.floor_parser import extract_floor
    from cyberdog.language.generate_dataset import UNKNOWN_LOCATION

    parse = None
    if use_nlu:
        # Mac only. The J4012's brain venv (go2env) has no torch, and Gemma
        # does not fit next to VAMOS on it anyway (see place_match: nicknames):
        # what the model was for is in the rules.
        try:
            from cyberdog.language.infer import load_model, parse_command  # torch first, then the maps
        except ImportError as e:
            raise SystemExit(f"--nlu needs torch, transformers and peft, and this machine "
                             f"has not got them ({e.name}). The Gemma parser runs on the "
                             f"Mac only; without --nlu the rules read places, nicknames "
                             f"and needs.")
        load_model()
        parse = parse_command
    # No-go places are never matched by sound: they are taken only from their
    # name said out loud (step 1).
    hazards = {n for n in router.locations if router.hazard_place(n)}

    stops, floor = [], START_FLOOR
    for text_i in split_destinations(text):
        requested, command = extract_floor(text_i, current_floor=floor)

        # Rules before the model: a no-go place named out loud is taken from
        # the raw text, because the parser was fine-tuned on destinations and
        # maps an unknown word onto the nearest one it knows -- "the stairs"
        # comes back as "hallway", and the stairs are never mentioned again.
        # plan_stops turns it into an approach point, not a refusal.
        named = router.hazard_named(text_i)
        if named is not None:
            stops.append((named[0], requested))
            if requested is not None:
                floor = requested
            continue

        # Longest name first, so the most specific match wins when one
        # location's name contains another's.
        heard_as = numbers_to_digits(command).lower()
        name = next((k for k in sorted(router.locations, key=len, reverse=True)
                     if k in heard_as), "")
        guess = None
        if name and name not in command.lower():
            guess = command.strip()            # "room two oh one": said as words
        if not name:
            near = (alias_place(router.locations, heard_as)
                    or near_place(router.locations, heard_as, hazards)
                    or need_place(router.locations, heard_as))
            if near is not None:
                name, guess = near
        if not name and movement(command):
            # Not a trip, so not the model's to guess at (it answers "turn
            # left" with the restroom). The dog only goes to places.
            # Not quoted: command_splitter has made "turn left" "Go to turn left".
            raise SystemExit(f'"{movement(command)}" is a movement, not a place. I can '
                             f"take you to a place in this building: where would you "
                             f"like to go?")
        if not name and parse is not None:
            try:
                said = (parse(command).get("target_location") or "").strip().lower()
            except ValueError:                 # the model's output was not JSON
                said = ""
            if said == UNKNOWN_LOCATION:
                raise SystemExit(f"I don't know where that is: {text_i!r} names "
                                 f"no place in this building. Staying here.")
            if said in router.locations:
                name = said
            elif said:
                # The model said something near a place: held to the map
                # like anything heard. Anything else is not on this map.
                near = near_place(router.locations, said, hazards)
                name = near[0] if near is not None else ""
            guess = command.strip()
        if not name:
            raise SystemExit(f"no destination found in {text_i!r}"
                             + ("" if parse is not None else
                                " (try --nlu, or name a place from locations.json)"))
        if guess is not None and guessed is not None:
            guessed.append((len(stops), name, guess))
        stops.append((name, requested))
        if requested is not None:
            floor = requested
    return stops


def describe_person(dog, yaw, xy, vel):
    """What to tell the person on the handle about somebody the dog is waiting
    for: which way they are moving, and on which side of the dog they are.

    The person on the handle can hear this and cannot see it, so it has to be
    true. It used to be "crossing from the <side>" for everybody, with the side
    read from the sign of their sideways velocity: someone overtaking past the
    room 101 trolley, and someone coming the other way in the far lane, were
    both "crossing from the right" -- they were on the dog's left, and neither
    was crossing. A sideways speed of 0.1 m/s says nothing, and its sign less.

    Side comes from where they are; "crossing" only when they really are going
    across, and then from the side they started on.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = xy[0] - dog[0], xy[1] - dog[1]
    ahead, left = dx * c + dy * s, -dx * s + dy * c
    v_fwd, v_left = vel[0] * c + vel[1] * s, -vel[0] * s + vel[1] * c
    side = ("on your left" if left > SIDE_DEAD else "on your right" if left < -SIDE_DEAD
            else None)
    if abs(v_left) >= CROSS_V and abs(v_left) >= abs(v_fwd):
        return f"Someone crossing from the {'right' if v_left > 0 else 'left'}"
    if v_fwd <= -CROSS_V:
        what = "Someone coming towards us"
    elif v_fwd >= CROSS_V and ahead < 0:
        what = "Someone overtaking"
    else:
        return f"Someone {side or ('ahead' if ahead >= 0 else 'behind us')}"
    return f"{what}, {side}" if side else what


def lift_approach(pts, says):
    """A leg planned to the lift, cut back to the waiting point outside it.

    The router hands over at lift.LIFT_XY, which is inside the car, so the
    velocity controller used to drive the dog through the open face and park it
    in the shaft -- before the lift had been called, and with a scripted "walk
    in" afterwards that had nowhere left to walk. Drop the checkpoints in the
    doorway and end at the standoff; the handover happens there and lift_ride
    walks the last metre in. The final announcement ("take the lift") comes
    with it, because it is still the last thing said on this leg.
    """
    keep = [k for k, p in enumerate(pts) if not lift.in_doorway(p)]
    if keep and keep[-1] == len(pts) - 1:      # handed over outside already
        return pts, says
    return ([pts[k] for k in keep] + [lift.WAIT_XY],
            [says[k] for k in keep] + [says[-1]])


def lift_departure(pts, says):
    """The leg out of the lift, picked up where the dog actually stands.

    lift_ride steps back out to the same waiting point, so starting this leg at
    LIFT_XY meant the first waypoint was behind the dog and the first thing the
    controller did was turn round towards the shaft.
    """
    keep = [k for k, p in enumerate(pts) if not lift.in_doorway(p)]
    if keep and keep[0] == 0:                  # picked up outside already
        return pts, says
    return ([lift.WAIT_XY] + [pts[k] for k in keep],
            [says[0]] + [says[k] for k in keep])


class Leg(NamedTuple):
    """One floor's walk -- plan_stops' unit, and what Run.follow drives."""
    floor: int
    name: str                   # the stop it is part of
    points: list                # [(x, y), ...]
    says: list                  # [[announcement, ...], ...], one per point
    via: str = None             # how it is left: "lift", "stairs", or None (the stop)
    stop: int = 0               # index of its stop in the command
    transit: Transit = None     # choose_transit's answer, when `via` is set


def close_spot(router, floor, xy, yaw):
    """Close the way at `xy` on `floor`'s planning grid, for the rest of the
    trip: a strip BLOCK_HALF either side of it along the dog's heading `yaw`
    and BLOCK_SPAN either side across, wall to wall in any corridor, so the
    next plan cannot squeeze back past it. See ASK_S."""
    g = router.plan_grids[floor]
    c, s = math.cos(yaw), math.sin(yaw)
    res = g.resolution / 2.0
    for a in np.arange(-BLOCK_HALF, BLOCK_HALF + 1e-9, res):
        for b in np.arange(-BLOCK_SPAN, BLOCK_SPAN + 1e-9, res):
            r, col = g.world_to_grid(xy[0] + a * c - b * s, xy[1] + a * s + b * c)
            if 0 <= r < g.grid.shape[0] and 0 <= col < g.grid.shape[1]:
                g.grid[r, col] = 100


def plan_stops(router, stops, consent=None, start=None):
    """Stops -> [Leg], following the planner's own floor handoffs.

    `via` is how a leg is left: "lift" or "stairs" for a leg that ends at a
    floor change, None for one that ends at the stop. Stairs are planned as
    chosen, which is a yes assumed: main asks the person (Run.offer_stairs)
    when the dog is about to set off for them, and plans the rest again with
    `consent` answering no if they say no. `consent(transit, from_floor,
    to_floor)` -> False turns a stairs change into the lift -- or, with the
    lift out of use, NoAccessibleRoute.

    `start` is (floor, xy) to plan from; the main entrance by default.

    Each leg is one floor: a cross-floor stop comes back as a leg to the lift
    and a leg out of it, with the ride in between. The announcements come
    along one list per waypoint -- they are the whole point of the checkpoint
    on a robot whose user cannot see where it is about to turn.

    A stop inside a no-go zone becomes a leg to the corridor outside it, with
    the arrival announcement saying exactly that. Those legs also come back in
    {leg index: point to face on arrival}.
    """
    legs, faces = [], {}
    if start is None:
        floor, xy = START_FLOOR, tuple(router.resolve(START_LOCATION, START_FLOOR, (0, 0))["xy"])
    else:
        floor, xy = start[0], tuple(start[1])
    start_xy = xy
    for i, (name, requested) in enumerate(stops):
        if requested is not None and requested not in router.floors:
            raise SystemExit(f"there is no floor {requested}; the building has "
                             f"floors {router.floors[0]}-{router.floors[-1]}")
        if not router.floors_of(name):
            raise SystemExit(f"unknown destination: {name}")

        # Somewhere we will not go: walk to the corridor outside it instead,
        # and say so. Asking for the stairs is a fair thing to ask -- it is
        # going up them that is not on offer.
        hazard = router.hazard_place(name)
        if hazard is not None:
            approached = router.approach(name, floor, xy, floor=requested)
            if approached is None:
                raise SystemExit(f"there is no {name} on floor {requested}")
            target, line = approached
        else:
            target = router.resolve(name, floor, xy, floor=requested)
        if target is None:
            raise SystemExit(f"there is no {name} on floor {requested} (it is on "
                             f"floor(s) {', '.join(map(str, router.floors_of(name)))})")
        transit = None
        if target["floor"] != floor:
            transit = router.choose_transit(floor, xy, target, lift_stop=lift.WAIT_XY)
            if transit.kind == "stairs" and consent is not None \
                    and not consent(transit, floor, target["floor"]):
                if transit.lift_m is None:
                    raise NoAccessibleRoute(
                        f"{transit.reason} Without the stairs I cannot get you "
                        f"to floor {target['floor']} from here.")
                transit = Transit("lift", tuple(lift.WAIT_XY), tuple(lift.WAIT_XY), "",
                                  transit.lift_m, transit.stairs_m)
        planned = router.plan(floor, xy, target, lift_stop=lift.WAIT_XY, transit=transit)
        if planned is None:
            raise SystemExit(f"no route to {name}")
        via = transit.kind if transit is not None else None
        for j, (f, route) in enumerate(planned):
            pts, says = router.walk_line(f, route)
            # A cross-floor stop comes back as a leg to the lift and a leg out
            # of it; neither should be driven into the car itself. The stairs
            # legs already end and start outside the stairwell zone.
            if len(planned) > 1 and j == 0 and via == "lift":
                pts, says = lift_approach(pts, says)
            elif len(planned) > 1 and j == 1 and via == "lift":
                pts, says = lift_departure(pts, says)
            first = len(planned) > 1 and j == 0
            legs.append(Leg(f, name, pts, says, via if first else None, i,
                            transit if first else None))
        if hazard is not None:
            # Replaces "destination reached" on the last leg of this stop:
            # the dog has not reached what was asked for, and saying so is the
            # announcement. Anything else would be a small lie to someone who
            # cannot check.
            legs[-1][3][-1] = [line]
            # And turn to face the thing at the end of it. The person asked to
            # be taken to the stairs; standing beside them facing down the
            # corridor answers half the question.
            faces[len(legs) - 1] = tuple(router.resolve(
                name, floor, xy, floor=requested, allow_hazard=True)["xy"])
        floor, xy = target["floor"], tuple(target["xy"])
    return start_xy, legs, faces


class HazardStop(Exception):
    """The dog entered a zone it must not move in. Nothing resumes after this.

    The planner will not route through a stairwell, so this firing means
    something upstream is wrong -- which is exactly when a guide robot should
    stop moving and say so rather than carry on.

    `layer` is the part of the stack that ended the run, for the FAILED BY
    line (spec DoD: every layer's failure logged, which layer rejected what).
    """
    layer = "behavior layer"

    def __init__(self, message, layer=None):
        super().__init__(message)
        if layer is not None:
            self.layer = layer


class HandleStop(HazardStop):
    """The dog is waiting for continue on the handle and none is coming:
    after a tug, or at the lift doors.

    Only a scripted handle can know that no continue is coming; the run ends
    here, standing where it waited, rather than at MAX_TICKS.
    """
    layer = "handle"


class Run:
    """One drive through the building, with the video panels attached."""

    def __init__(self, scene, start_xy, start_yaw, router, out=None, vamos=False,
                 auto_confirm=False, speed=1, crowd=0, seed=0, shadow=None,
                 vamos_url=None, vlm_latency=None, vamos_sample=False, handle=None,
                 vamos_samples=5, vamos_affordance=False, vamos_steer=None,
                 affordance_shadow=None, autolabel=None, aware=False, planner="map",
                 track_log=None, robot=None, cloud_log=None, puddles=0, puddle_at=(),
                 floor_guard=None, fg_latency=fguard.LATENCY, fg_see=fguard.SEE_D,
                 fg_miss=0.0):
        self.robot = robot or MujocoRobot(scene, start_xy=start_xy, start_yaw=start_yaw,
                                          start_z=levels.floor_z(START_FLOOR))
        self.cam = load_camera_config()
        self.router = router
        self._doors = {}          # floor -> [(name, door_xy)], for the captions
        self.floor = START_FLOOR
        self.in_lift = False
        self.exit_from = None     # floor just stepped out of the lift onto: see EXIT_CLEAR_M
        self.on_stairs = False
        self.stair_flights = self.stair_steps = 0
        self.behind = False
        self.auto_confirm = auto_confirm
        # Write one frame in `speed`. The building is 48 m end to end and the
        # stairwell and the lift are at opposite ends of it, so a run that
        # visits both spends a minute walking down a corridor that looks the
        # same the whole way. Dropping frames plays that back faster without
        # touching the control loop, which still runs at 20 Hz.
        self.speed = max(int(speed), 1)
        self.tick = 0
        self.writer = self.chase = self.outside = None
        self.policies, self.vamos = {}, vamos
        # Shadow mode (Gate C in docs/dreaming_safety_kpis.md): VAMOS is asked
        # and the dream scores its answer on the same schedule as --vamos, but
        # the map route alone drives. `shadow` is the log file, one JSON line
        # per call; None is off. --vamos and --shadow are exclusive.
        if shadow:
            os.makedirs(os.path.dirname(os.path.abspath(shadow)), exist_ok=True)
        self.shadow = open(shadow, "w") if shadow else None
        self.shadow_n = {"calls": 0, "steer": 0, "none": 0}
        self.vamos_url = vamos_url
        self.vamos_samples = vamos_samples
        self.vamos_sample = vamos_sample
        # How long a VAMOS answer takes to arrive, in seconds of sim time; None
        # is the call's own wall-clock time. See ask_vamos.
        self.vlm_latency = vlm_latency
        # The request in flight, if any: (tick it lands on, its paths, tick it
        # was asked on, body xy then). One at a time, as a GPU serves them.
        self.pending = None
        self.vlm = {"asked": 0, "answers": 0, "dropped": 0, "age": 0.0, "moved": 0.0,
                    "cap_min": None}
        # Robot time each call was asked at, and the last VLM_CAP_N latencies
        # (asked to landed, s): what vlm_speed_cap goes on.
        self.vlm_asked_t = None
        self.vlm_lat = []
        # A robot on the wall clock (go2_board) cannot have the loop paused for
        # the call: the request runs on a worker thread and lands when it is
        # done. The twin keeps the repeatable sim-time release above.
        self.real_time = getattr(self.robot, "REAL_TIME", False)
        self.board_turning, self.board_since, self.board_gave_up = None, 0.0, False
        self.board_push = None    # (t, x, y) since asking forward, for respect_board's stall
        self.board_flipped = False
        self.board_side = None    # (sign, t, x, y) while sidestepping, before any turn
        self.board_against = 0    # ticks in a row the plan has asked the other way round
        self.board_swept, self.board_yaw = 0.0, None   # rad turned since the turn began, last yaw
        self.board_leg = 0.0      # rad turned since the turn last changed direction
        self.board_plan_to = None # the heading the plan wanted when the turn took its way
        self.vlm_worker = None    # the call still running, dropped answer or not
        self.vlm_pool = None
        # Every velocity command sent, fingerprinted. Two runs of the same
        # route and seed with equal fingerprints commanded the dog identically,
        # tick for tick -- which is how a shadow run proves it touched nothing.
        self.commands = hashlib.sha256()
        self.n_commands = 0
        # The person's override, last before the motors (handle.py). With no
        # handle the mux reads zero force every tick and passes commands on
        # unchanged. With one, its continue is also how the person answers
        # the lift (confirm).
        self.has_handle = handle is not None
        self.handle = handle or HandleScript()
        self.mux = SafetyMux()
        self.live = {}            # floor -> LiveClearance, one per storey
        self.seen = np.empty((0, 2))   # this tick's unexplained returns
        # Ground truth, for scoring only -- never shown to the robot. The dog
        # finds these with the LiDAR or not at all; this is how we check.
        self.hits = {"ticks": 0, "boxes": set()}
        self.handler_hits = {"ticks": 0, "boxes": set(), "since": None}
        self.dog_geoms, self.touchable = self.body_contacts()
        self.obs = {"crawl": 0, "stopped": 0, "seen": 0, "min_clear": 99.0, "eased": 0}
        self.turn_held = 0        # ticks spare_handler has held a turn (HOLD_TURN_S)
        self.side_held = 0.0      # side spare_handler is stepping to (SIDE_MIN_W), +1 left
        # The affordance network watching, not steering (--affordance-shadow):
        # what it would say about the walk ahead, against what was really
        # there. `affordance_shadow` is the log file; None is off.
        self.aff = self.elev = self.aff_log = None
        self.aff_n = collections.Counter()
        if affordance_shadow:
            from cyberdog.affordance.runtime import Affordance, ElevationMemory
            self.aff, self.elev = Affordance(), ElevationMemory()
            os.makedirs(os.path.dirname(os.path.abspath(affordance_shadow)), exist_ok=True)
            self.aff_log = open(affordance_shadow, "w")
        # The same network judging VAMOS's candidates (--vamos-affordance):
        # one that calls the ground a path walks onto not walkable rejects it
        # before the dream (vamos_client.deliver). Shares the network and the
        # elevation memory with the shadow when both are on.
        self.vamos_affordance = vamos_affordance
        # A preference VAMOS is asked for with every goal (--vamos-steer).
        self.vamos_steer = vamos_steer
        if vamos_affordance and self.aff is None:
            from cyberdog.affordance.runtime import Affordance, ElevationMemory
            self.aff, self.elev = Affordance(), ElevationMemory()
        # Training pairs for the VAMOS LoRA (--autolabel): the frame, and the
        # path the dog went on to walk from it. Map-only runs only.
        self.autolabel = None
        if autolabel:
            from cyberdog.sim.autolabel import AutoLabel
            self.autolabel = AutoLabel(autolabel, self.cam)
        # People. Not in any grid either, and unlike the crates they move, so
        # one scan cannot describe them -- `tracker` is what two scans give.
        self.crowds = {n: pedestrians.Crowd(n, crowd, seed, aware=aware) for n in (1, 2, 3)}
        if crowd and not self.robot.has_body("ped_f1_0"):
            # A mocap body cannot be added to a loaded model, so people only
            # exist if the scene was built with them. Silently walking an
            # empty building instead is the worst of both: the run looks fine
            # and proves nothing.
            raise SystemExit("this scene has no pedestrian bodies -- rebuild it: "
                             "python -m cyberdog.sim.scene.build_scene --building")
        self.tracker = Tracker(self.robot.CONTROL_DT)
        self.waiting_for = None        # the track being yielded to, if any
        self.reroute = None            # (floor, (x, y), yaw) of the spot to close: see ASK_S
        self.passing = None            # where the person being gone round stands
        self.passer_near = False       # someone passing in another lane within PASS_R
        self.overtakers_said = []      # tracks "Someone passing" has been said for
        self.narrow_waited = 0         # ticks stood before a narrow way (NARROW_WAIT_S)
        self.narrow_at = -INF          # when the dog last had a narrow way ahead
        self.narrow_side = 0.0         # ...and which side it was on
        self.narrow_last = None        # where it last steered through one (NARROW_EXIT_M)
        self.overtake_said_at = {}     # side -> when it was last said
        self.passing_lead = None       # the track being overtaken (see FOLLOW_GAP)
        self.follow_said_at = -INF     # when "Following someone ahead" was last true
        self.pass_said_at = -INF       # ...and "Passing someone on your left"
        self.pace = Pace()             # what the dog walks for what it is told
        self.lead_track, self.lead_t = None, 0.0   # the one ahead being watched...
        self.lead_v, self.lead_seen = 0.0, 0.0     # ...their smoothed pace, and for how long
        self.waited = 0                # ticks spent yielding to it
        self.held_still = 0            # ...of which it has not moved at all
        self.movers = np.empty((0, 2))  # this tick's moving things, for the video
        self.ped = {"yields": 0, "waited": 0, "near": 0, "min_d": 99.0, "since": None}
        self.planner = planner         # "map": wait for people; "cerrt": go round
        self.plan = None               # the ce_rrt.Plan being walked, if any
        self.plan_at = -1              # tick it was made, or last checked and kept
        self.searched_at = None        # tick of the last search: see PLAN_HOLD_S
        self.plan_failed = None        # tick the last search found nothing: see PLAN_RETRY_S
        self.plan_speed = 0            # index into ce_rrt.SPEEDS the next search tries
        self.plan_floor = None         # (static, crate, nose, handle) room this
                                       # going-round may not go below: fixed
                                       # when it starts, like the goal
        self.plan_goal = None          # where this going-round is going: fixed
                                       # until it is reached, so each replan
                                       # aims where the last one did
        self.plans = {"rounds": 0, "made": 0, "none": 0, "ms": 0.0, "ticks": 0,
                      "worst_ms": 0.0, "held": 0}
        self.why_wait = collections.Counter()  # why a wait was not a going-round
        self.gave_up = None       # (layer, why) when follow() ends a leg short
        self.evading = 0.0             # sideways m/s while stepping aside, or 0
        self.evade_said = (-INF, 0.0)  # (when, side) the last step aside was said
        self.evade = {"steps": 0, "ticks": 0}
        self.track_log = open(track_log, "w") if track_log else None
        # --cloud-log: every CLOUD_EVERY-th scan as the brain got it (map
        # frame), with the pose it was placed by; saved as one .npz on close.
        self.cloud_log, self.clouds = cloud_log, []
        self.no_plan = None                    # ...this tick's reason, for yield_to
        self.place_crowd()
        # Puddles (--puddles N a leg): flat, so the LiDAR does not see them;
        # only the floor guard does (--floor-guard sim). With puddles and no
        # guard they are still placed and scored: that is the baseline.
        self.puddle_n = puddles
        self.fg_on = floor_guard == "sim"
        self.fg = (fguard.FloorGuard(self.cam, self.robot.CONTROL_DT, see_d=fg_see,
                                     latency=fg_latency, p_miss=fg_miss, seed=seed)
                   if (puddles or puddle_at or self.fg_on) else None)
        for spec in puddle_at:
            v = [float(t) for t in spec.split(",")]
            self.fg.add([fguard.Puddle(self.fg.next_id(), int(v[3]) if len(v) > 3 else 1,
                                       v[0], v[1], v[2])])
        self.puddle_rng = random.Random(1000 + seed)

        if out is not None:
            import imageio
            import mujoco
            self.writer = imageio.get_writer(out, fps=FPS, macro_block_size=1)
            self.chase = mujoco.MjvCamera()
            self.chase.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.chase.distance, self.chase.elevation = CHASE_D, CHASE_EL
            self.outside = mujoco.Renderer(self.robot.model, 480, 640)

    # -- video ----------------------------------------------------------
    def frame(self, state=None, chosen=None, candidates=(), safety=None):
        if self.writer is None:
            return
        self.tick += 1
        if self.tick % self.speed:
            return

        dog = self.robot.get_image().copy()
        pose = self.robot.camera_pose()
        # Under the paths: what it found that the map had no record of. A dog
        # swerving for nothing visible reads as a bug, not as avoidance.
        if len(self.seen):
            draw_points(dog, self.seen, pose, self.cam)
        # And which of them are walking, in a colour of their own. A dog that
        # stops dead in an empty-looking corridor reads as a fault; these are
        # the reason, and they are the only thing on screen that tells the
        # difference between the crate it went round and the person it did not.
        if len(self.movers):
            draw_points(dog, self.movers, pose, self.cam, colour=(70, 160, 255), r=3)
        # Puddles: a ring at the edge, dim until the floor guard knows of it.
        for p in self.puddles_here():
            ring = [(p.x + p.r * math.cos(a), p.y + p.r * math.sin(a))
                    for a in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)]
            known = p.id in self.fg.known
            draw_points(dog, ring, pose, self.cam,
                        colour=(60, 220, 255) if known else (40, 80, 150), r=2 if known else 1)
        for c in candidates:
            draw_path(dog, c, pose, self.cam, (120, 120, 130))
        if chosen:
            draw_path(dog, chosen, pose, self.cam, (90, 220, 120), thick=2)
        if state and state["state"] == "TRACK":
            draw_marker(dog, *state["pixel"])
        if safety is not None:
            safety_bar(dog, safety)

        x, y, yaw = self.robot.get_pose()
        self.chase.lookat[:] = [x, y, self.robot.z + 0.3]
        if self.in_lift:
            # Chasing from behind puts the camera inside the shaft wall. The
            # car is only open to the west, so watch from out in the corridor
            # and let it rise past the lens.
            self.chase.azimuth, self.chase.distance, self.chase.elevation = LIFT_CAM
        else:
            # Azimuth is the direction the camera looks along, so the usual
            # +180 sits the camera ahead of the dog, walking towards the lens.
            # For the closing hold, drop the 180 and watch from behind: the
            # dog ends a run facing whatever it was sent to -- a wall, or a
            # stairwell -- and a camera in front of it is inside that.
            self.chase.azimuth, self.chase.distance, self.chase.elevation = \
                math.degrees(yaw) + (0 if self.behind else 180), CHASE_D, CHASE_EL
        self.outside.update_scene(self.robot.data, self.chase)
        self.draw_puddles(self.outside.scene)
        left = np.ascontiguousarray(self.outside.render())
        self.floor_tag(left)
        self.name_tag(left, x, y)
        self.writer.append_data(np.hstack([left, dog]))

    def puddles_here(self):
        """The placed puddles on the floor the dog is on (for drawing)."""
        if self.fg is None:
            return []
        return [p for p in self.fg.puddles if p.floor == self.floor]

    def draw_puddles(self, scn):
        """Puddles into the chase camera's scene: they are on no model (the
        LiDAR must not see them), so they are drawn as thin discs, blue until
        the floor guard knows of one, then bright."""
        import mujoco
        z = levels.floor_z(self.floor) + 0.004
        for p in self.puddles_here():
            if scn.ngeom >= scn.maxgeom:
                break
            known = p.id in self.fg.known
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                np.array([p.r, 0.003, 0.0]), np.array([p.x, p.y, z]),
                                np.eye(3).flatten(),
                                np.array([0.2, 0.85, 1.0, 0.9] if known else [0.15, 0.3, 0.7, 0.6],
                                         dtype=np.float32))
            scn.ngeom += 1

    def doors(self, floor):
        """(name, sign_xy) for every named place on one floor, cached.

        Doors rather than room centres: the door is what the dog walks past,
        and a room centre is behind a wall. `hallway` is skipped -- it is the
        corridor itself, so its label would sit on top of every other one.

        Two collisions have to be undone first, and both come from
        `locations.json` being a routing table rather than a signboard:

        `door_xy` is the corridor centreline, so the room on the north side
        and the room on the south share one to the centimetre -- `room 101`
        and `room 106` are both [3.0, 9.5]. Signed as-is, one of every pair is
        dropped as an overlap and half the building is unlabelled. So the sign
        slides SIGN_OFF off the centreline towards its own room, which puts it
        on that room's wall, which is where signage lives anyway.

        And a place can have several names for the parser's benefit -- stairs,
        staircase, stairway and stairwell are one stairwell at [1.0, 9.5].
        A door has one sign, so the shortest name wins and the synonyms go.
        """
        if floor not in self._doors:
            seen, out = {}, []
            for name, entries in self.router.locations.items():
                if name == "hallway":
                    continue
                for e in entries:
                    if e["floor"] != floor:
                        continue
                    door = tuple(e.get("door_xy") or e["xy"])
                    room = tuple(e["xy"])
                    key = (round(door[0], 2), round(door[1], 2),
                           round(room[0], 2), round(room[1], 2))
                    if key in seen and len(seen[key]) <= len(name):
                        continue        # a synonym of something already signed
                    seen[key] = name
            for (dx, dy, rx, ry), name in seen.items():
                vx, vy = rx - dx, ry - dy
                n = math.hypot(vx, vy)
                if n:
                    dx, dy = dx + vx / n * SIGN_OFF, dy + vy / n * SIGN_OFF
                out.append((name, (dx, dy)))
            self._doors[floor] = out
        return self._doors[floor]

    def name_tag(self, img, x, y):
        """Room names on the chase panel, for the rooms the dog is passing.

        The same complaint `floor_tag` answers, one level finer: the pips say
        which storey, and nothing said where along it. Every doorway in this
        building renders identically, so a run to `room 201` and a run to
        `room 206` are the same video.

        Caption only. These are drawn onto the rendered chase frame, not built
        into the scene, so the dog's camera -- the one VAMOS is handed -- has
        no text in it. That boundary is the same one `obstacles.py` and
        `pedestrians.py` keep: the robot is not told anything here.
        """
        cam = ChaseCam(self.outside.scene, img.shape[1], img.shape[0])
        label_places(img, cam, self.doors(self.floor), (x, y),
                     levels.floor_z(self.floor))

    def floor_tag(self, img, n=3):
        """Three stacked pips down the left edge, the current floor lit.

        The chase view of a corridor looks the same on every storey, so
        without this you cannot tell from the video which floor you are on.
        """
        for k in range(n):
            top = 14 + (n - 1 - k) * 22          # floor 1 at the bottom
            lit = (k + 1) == self.floor
            img[top:top + 16, 14:30] = (250, 205, 90) if lit else (70, 72, 78)

    def close(self):
        if self.vlm_pool is not None:
            self.vlm_pool.shutdown(wait=False, cancel_futures=True)
        if self.writer is not None:
            self.writer.close()
        if self.track_log is not None:
            self.track_log.close()
        if self.cloud_log and self.clouds:
            meta = np.array([c[0] for c in self.clouds])
            pts = [c[1] for c in self.clouds]
            np.savez_compressed(self.cloud_log, meta=meta,
                                offsets=np.cumsum([0] + [len(p) for p in pts]),
                                points=np.concatenate(pts).astype(np.float32))

    # -- driving --------------------------------------------------------
    def clearance(self, floor):
        """The floor's clearance field: the static map, plus whatever the LiDAR
        found that the map cannot account for.

        One object per storey, updated in place. The cached VamosPolicy's gate
        and its Dream both close over it, so a scan reaches both of them
        without rebuilding either.
        """
        if floor not in self.live:
            self.live[floor] = LiveClearance(floor)
        return self.live[floor]

    def policy(self, floor):
        """VAMOS policy for a floor -- its gate and its dreams read that floor.

        Both come off the same clearance field, so the cheap geometric test
        and the imagined rollouts cannot disagree about where the walls, the
        stairwells, or the crate that is not on any map are.
        """
        if floor not in self.policies:
            from cyberdog.sim.sensing.dreaming import Dream
            from cyberdog.sim.vamos_client import VamosPolicy
            from cyberdog.sim.vamos_client import DEFAULT_URL
            live = self.clearance(floor)
            url = self.vamos_url or DEFAULT_URL
            p = VamosPolicy(self.cam, lambda x, y: live(x, y) >= ROBOT_R,
                            dream=Dream(live, self.robot.MAX_V,
                                        person=lambda x, y, yaw: self.handler_room(
                                            live, x, y, yaw)), url=url,
                            sample=self.vamos_sample, num_samples=self.vamos_samples,
                            affordance=(lambda ps, pose: self.judge_candidates(live, ps, pose))
                            if self.vamos_affordance else None)
            if not p.available():
                raise SystemExit(f"FAILED BY vamos: server is not answering on {url} -- "
                                 "start vendor/VAMOS/server/vlm_server.py first, "
                                 "or pass --no-vamos to drive the map route alone")
            self.policies[floor] = p
        return self.policies[floor]

    def judge_candidates(self, live, paths, pose):
        """The affordance network on VAMOS's candidates (--vamos-affordance):
        one class name per path, None where it cannot say. One elevation
        patch at the dog's pose now, the map's walls made floor (the network
        judges ground; walls are the clearance field's)."""
        from cyberdog.affordance.runtime import judge_paths
        return judge_paths(self.aff, self.elev.patch(pose, live.wall_at), pose, paths)

    def log_shadow(self, floor, n, pose, target, proposed, safety):
        """One shadow-mode call: what VAMOS offered, what the dream made of
        each candidate, and what it would have steered at against what the map
        actually did.

        "steer" is a proposal pointing more than SHADOW_DISAGREE off the map's
        target; "none" is nothing passing the gate, which under --vamos means
        crawling and asking again. Both are what Gate D is about to measure.
        """
        x, y, yaw = pose
        verdicts = self.policies[floor].verdicts
        if proposed is None:
            kind, angle = "none", None
        else:
            px, py = path_target(proposed, (x, y))
            angle = abs(wrap(math.atan2(py - y, px - x)
                             - math.atan2(target[1] - y, target[0] - x)))
            kind = "steer" if angle > SHADOW_DISAGREE else "agree"
        self.shadow_n["calls"] += 1
        if kind != "agree":
            self.shadow_n[kind] += 1
        rnd = lambda pts: [[round(a, 3), round(b, 3)] for a, b in pts]
        self.shadow.write(json.dumps({
            "floor": floor, "tick": n, "t": round(self.robot.now(), 2),
            "pose": [round(x, 3), round(y, 3), round(yaw, 4)],
            "map_target": [round(target[0], 3), round(target[1], 3)],
            "candidates": [{"path": rnd(p), "factor": None if f is None else round(f, 4),
                            "verdict": v} for p, f, v in verdicts],
            "would_choose": next((k for k, (p, _, _) in enumerate(verdicts)
                                  if p is proposed), None),
            "safety": round(safety, 4),
            "disagree": kind,
            "angle_deg": None if angle is None else round(math.degrees(angle), 1),
            "steer": self.vamos_steer,
        }) + "\n")

    def ask_vamos(self, floor, n, state):
        """Send VAMOS this tick's frame; the answer lands some ticks later.

        The VLM takes about 1.8 s and the control loop has 50 ms, so the loop
        cannot wait for it: plan() used to be called here and block, and the
        twin paused physics for the duration, which hid that a real dog would
        walk on for 1.8 s on its last command with no LiDAR, no stop and no
        yield. Now the loop never waits. The request is made at tick n, from
        this tick's image, and its answer is released at tick n + latency --
        in sim time, so a run with --vlm-latency fixed is repeatable where a
        real thread would land its answer on a different tick every time. On
        the robot this is a worker thread; the timing is the same.

        `vlm_latency` None is the wall-clock time the HTTP call really took:
        the sim is paused while it runs, then charged for it afterwards.
        run_building's own default is VLM_LATENCY, the dog's.
        """
        if self.real_time:
            # Frame and lens pose now, on this thread (the renderer's); only
            # the HTTP call and the projection go to the worker.
            if self.vlm_pool is None:
                self.vlm_pool = concurrent.futures.ThreadPoolExecutor(1, "vamos")
            self.vlm_worker = self.vlm_pool.submit(
                self.policy(floor).request, self.robot.get_image().copy(),
                vamos_prompt(state, self.vamos_steer), self.robot.camera_pose())
            self.pending = (self.vlm_worker, None, n, self.robot.get_pose()[:2])
            self.vlm["asked"] += 1
            self.vlm_asked_t = self.robot.now()
            return
        self.vlm_asked_t = self.robot.now()
        t0 = time.perf_counter()
        cam_pose = self.robot.camera_pose()
        paths = self.policy(floor).request(self.robot.get_image(),
                                           vamos_prompt(state, self.vamos_steer), cam_pose)
        wait = (time.perf_counter() - t0 if self.vlm_latency is None
                else self.vlm_latency)
        self.pending = (n + int(round(wait / self.robot.CONTROL_DT)), paths,
                        n, self.robot.get_pose()[:2])
        self.vlm["asked"] += 1

    def vlm_speed_cap(self):
        """Fastest the dog may walk on VAMOS's word, m/s: see VLM_REACH_D.

        (VLM_REACH_D - VLM_STOP_D) / 2 L, L the slowest of the last VLM_CAP_N
        answers, or the expected latency (--vlm-latency, else VLM_LATENCY)
        before any has landed. A slower VAMOS slows the dog with it.
        """
        lat = max(self.vlm_lat) if self.vlm_lat else (self.vlm_latency or VLM_LATENCY)
        cap = (VLM_REACH_D - VLM_STOP_D) / (2.0 * max(lat, self.robot.CONTROL_DT))
        if self.vlm["cap_min"] is None or cap < self.vlm["cap_min"]:
            self.vlm["cap_min"] = cap
        return cap

    def vlm_busy(self):
        """A call is still running on the worker, even one whose answer was
        dropped at a leg's end: one at a time, as the GPU serves them."""
        return self.vlm_worker is not None and not self.vlm_worker.done()

    def take_answer(self, floor, n, pose, destination):
        """The pending answer, if it has landed: judged from where the dog is
        now, not from where it was asked. Returns plan()'s triple, or None
        while the answer is still on its way.
        """
        if self.pending is None:
            return None
        lands, paths, asked_at, xy0 = self.pending
        if isinstance(lands, concurrent.futures.Future):
            if not lands.done():
                return None
            paths = lands.result()
        elif n < lands:
            return None
        self.pending = None
        self.vlm["answers"] += 1
        self.vlm["age"] += (n - asked_at) * self.robot.CONTROL_DT
        if self.vlm_asked_t is not None:
            # Robot time, not ticks: on the board a slow tick would hide it.
            self.vlm_lat = (self.vlm_lat + [self.robot.now() - self.vlm_asked_t])[-VLM_CAP_N:]
        self.vlm["moved"] += math.dist(xy0, pose[:2])
        return self.policy(floor).deliver(paths, pose, destination)

    def say(self, line):
        """One line of the Preemptive Voice Engine. Printed here, spoken later."""
        print(f"    [voice] {line}")

    def handle_tick(self):
        """Read the handle for this tick, and say what a tug or continue did.

        Before the command goes out, so a tug stops the dog on the tick it is
        felt: one control tick, 50 ms at 20 Hz, inside the spec's 100 ms.
        """
        t, dt = self.robot.now(), self.robot.CONTROL_DT
        event = self.mux.update(self.handle.force(t), self.handle.pressed(t, dt), dt)
        x, y, _ = self.robot.get_pose()
        if event == "tug":
            print(f"    t={t:6.1f}s  HANDLE tug at ({x:.2f}, {y:.2f}): stopped until continue")
            self.say("Stopped. Press continue when you are ready.")
        elif event == "continue":
            print(f"    t={t:6.1f}s  HANDLE continue at ({x:.2f}, {y:.2f})")
            self.say("Carrying on.")
        if self.mux.stopped and not self.handle.will_continue(t):
            self.robot.set_velocity(0.0, 0.0, 0.0)
            raise HandleStop(f"by a tug on the handle at t={t:.1f}s on floor "
                             f"{self.floor}; nobody pressed continue")

    def handle_wait(self):
        """The scripted moves into and out of the lift stop for a tug too.

        They set the pose directly and never reach the mux, so this stands the
        dog where it is, still filming, until continue.
        """
        self.handle_tick()
        while self.mux.stopped:
            x, y, yaw = self.robot.get_pose()
            self.robot.place((x, y), yaw)
            self.mux.stats["stopped_ticks"] += 1
            self.step_crowd()
            self.frame()
            self.handle_tick()

    @staticmethod
    def clear_ahead(live, xy, target, d=LOOKAHEAD, skip=perception.SKIP):
        """Least clearance along the next `d` metres towards `target`.

        Clearance under the dog is not the question -- by the time it is low
        the dog is already in the thing. This is what is coming, and only from
        things the map had no record of.

        The first `skip` metres are left out for the same reason line_clear
        leaves them out: a dog squeezing through a gap beside a crate is
        legitimately 0.16 m from it, and a test that starts at its feet reads
        that as "blocked" and halts it in the middle of the gap it is using.
        """
        dx, dy = target[0] - xy[0], target[1] - xy[1]
        span = math.hypot(dx, dy)
        if span < 1e-6:
            return live.detected_at(*xy)
        d = min(d, span)
        steps = max(int(d / 0.1), 1)
        first = int(skip / 0.1) if d > skip else 0
        # Detected things only. The static field is tight in every doorway on
        # the map and A* already approved those; stopping for one would mean
        # never leaving a room.
        return min(live.detected_at(xy[0] + dx * (d * k / steps) / span,
                                    xy[1] + dy * (d * k / steps) / span)
                   for k in range(first, steps + 1))

    def body_contacts(self):
        """(the dog's geoms, {scene geom: what it is}) for collision scoring.

        The dog's geoms are everything under the body with the free joint:
        trunk, hips, legs, feet. The scene's are everything the body must not
        touch -- walls, crates, the lift shaft and roof, stair steps. Left out:
        what it stands on (`ground`, the `slab`s, the lift car's `lift_plate`),
        where contact is walking, and the people, whose geoms do not collide at
        all and are scored by distance instead (PED_NEAR).
        """
        import mujoco
        m = self.robot.model
        root = next(m.jnt_bodyid[j] for j in range(m.njnt)
                    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)

        def ours(b):
            while b:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False

        dog, scene = set(), {}
        for g in range(m.ngeom):
            if ours(m.geom_bodyid[g]):
                dog.add(g)
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if (name == "ground" or name.startswith("slab") or name == "lift_plate"
                    or not m.geom_contype[g] and not m.geom_conaffinity[g]):
                continue
            scene[g] = (name.split("_", 1)[1] if name.startswith(("obs", "furn"))
                        else "stairs" if name.startswith("step")
                        else "lift" if name.startswith("lift")
                        else "wall")
        return dog, scene

    def score_collision(self, floor, x, y):
        """Ground truth: is any part of the dog touching a wall or an obstacle?

        MuJoCo's own contacts between the Go2's geoms and the scene's, so the
        whole body counts -- trunk, hips, swinging legs -- not just the point
        at its centre. The point test this replaced scored a trunk scraping a
        crate as clean as long as the centre stayed outside the box, and never
        looked at walls at all.

        Scoring only, and never fed back into the robot -- the dog finds these
        with the LiDAR or it does not find them. A run that reports arrival
        while this counter climbed is a run that walked into something.
        """
        d = self.robot.data
        touched = set()
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 in self.dog_geoms else g1 if g2 in self.dog_geoms else None
            if other in self.touchable:
                touched.add(self.touchable[other])
        if touched:
            self.hits["ticks"] += 1
            self.hits["boxes"].update(f"floor {floor} {what}" for what in touched)

    def respect_board(self, cmd, pose):
        """Work with the board's LiDAR guard instead of pushing into it.

        The guard slows the dog from SLOW_M and stops it at STOP_M whatever the
        brain asks, and the brain's own margins (from its plan and its view)
        can disagree with it by a few centimetres: walking past a trolley it
        judged 0.33 m clear, the guard saw a corner inside its 0.30 m band and
        the dog stood there pushing for five minutes. So:
          - never backwards: the person on the handle is right behind;
          - pushed against the guard (front under BOARD_NEAR_M, asking to go
            forward) while the plan is turning (over BOARD_PLAN_W): stop, and
            turn slowly the plan's way, saying so, until the front opens
            (BOARD_CLEAR_M);
          - pushed against it while the plan is going straight: step slowly
            sideways towards the side the board sees more room on -- at most
            BOARD_SIDE_M or BOARD_SIDE_S, and only while that side has more
            than BOARD_SIDE_ROOM;
          - still closed: turn slowly the plan's way (or the roomier side's),
            and round the other way when the plan asks it for
            BOARD_PLAN_TICKS towards a heading it did not want before;
            BOARD_SWEEP_RAD one way and still closed: back through the start
            to as far the other side;
          - the sweep done (or BOARD_GIVE_UP_S, turning or not) without the
            front opening: stop and ask the person for help, as the brain
            does when its own plan has no way past.
        """
        vx, vy, w = cmd
        vx = max(vx, 0.0)                                    # never back into the person
        free = self.robot.board_clearance()
        if free is None:
            return (vx, vy, w)                               # blind: the board turns only
        # Asked forward and going nowhere: the guard crawls the dog from SLOW_M
        # down to a speed the gait does not step at (0.52 m from a wall, asked
        # 0.3 m/s, given 0.09), and that is as stuck as a closed front.
        now, (x, y, yaw) = self.robot.now(), pose
        stalled = False
        if vx > 0.1:
            if self.board_push is None:
                self.board_push = (now, x, y)
            elif (now - self.board_push[0] > BOARD_STALL_S
                  and math.dist((x, y), self.board_push[1:]) < BOARD_STALL_M):
                stalled = True
                self.board_push = None
            else:
                stalled = False
                if now - self.board_push[0] > BOARD_STALL_S:
                    self.board_push = (now, x, y)            # moving: start the window again
        else:
            self.board_push, stalled = None, False
        def roomier():
            return 1.0 if free["left"] >= free["right"] else -1.0

        def planned():
            """The way the plan is turning, when it is turning; else None."""
            return math.copysign(1.0, w) if abs(w) > BOARD_PLAN_W else None

        def room(sign):
            return free["left"] if sign > 0 else free["right"]

        def turn(sign, prefix=""):
            self.board_turning, self.board_flipped, self.board_since = sign, False, now
            self.board_against = 0
            self.board_swept, self.board_leg, self.board_yaw = 0.0, 0.0, yaw
            self.board_plan_to = yaw + w / K_W
            self.say(prefix + "Turning " + ("left" if sign > 0 else "right") + " to find the way.")

        side = self.board_side
        if (self.board_turning is None and side is None and vx > 0.05
                and (free["front"] < BOARD_NEAR_M or stalled)):
            self.obs["board_turns"] = self.obs.get("board_turns", 0) + 1
            sign = roomier()
            if planned() is not None:
                turn(planned(), "Too close to something ahead. ")   # the plan's way round
            elif room(sign) > BOARD_SIDE_ROOM:
                side = self.board_side = (sign, now, x, y)
                self.say("Too close to something ahead. Stepping "
                         + ("left" if sign > 0 else "right") + " to find the way.")
                self.obs["board_sidesteps"] = self.obs.get("board_sidesteps", 0) + 1
            else:
                turn(sign, "Too close to something ahead. ")
        if side is not None:
            sign, t0, x0, y0 = side
            if free["front"] >= BOARD_CLEAR_M:
                self.board_side = None                       # open again: the plan takes over
                return (vx, vy, w)
            if (now - t0 < BOARD_SIDE_S and math.dist((x, y), (x0, y0)) < BOARD_SIDE_M
                    and room(sign) > BOARD_SIDE_ROOM):
                return (0.0, BOARD_SIDE_V * sign, 0.0)
            self.board_side = None                           # stepped as far as it may: turn
            turn(planned() or roomier())
        if self.board_turning is not None:
            if free["front"] >= BOARD_CLEAR_M:
                self.board_turning = None                    # open again: the plan takes over
                return (vx, vy, w)
            want = planned()
            step = abs(wrap(yaw - self.board_yaw))
            self.board_swept += step
            self.board_leg += step
            self.board_yaw = yaw
            heading = yaw + w / K_W                          # where the plan wants to face
            if (want is not None and want != self.board_turning
                    and abs(wrap(heading - self.board_plan_to)) > BOARD_PLAN_MOVED):
                # The plan wants round the other way, and somewhere new: after
                # BOARD_PLAN_TICKS of it in a row (not one tick of a plan
                # changing its mind), go. Only somewhere new: wanting the same
                # heading, the plan's w changes sign each time the turn carries
                # the nose past it, and followed it shook the dog's head +-30 deg
                # at the trolley gap for 10 s (floor 1, J4012, 2026-11-05).
                self.board_against += 1
                if self.board_against >= BOARD_PLAN_TICKS:
                    self.board_turning, self.board_against = want, 0
                    self.board_plan_to, self.board_leg = heading, 0.0
                    self.say("Turning " + ("left" if want > 0 else "right") + ", the way on.")
            else:
                self.board_against = 0
            took = now - self.board_since
            if not self.board_flipped and (self.board_leg > BOARD_SWEEP_RAD
                                           or (took > BOARD_GIVE_UP_S / 2 and want is None)):
                # A quarter turn one way, still closed: sweep back through the
                # start to as far the other side. In the gap between the trolley
                # and the wall, turning towards the wall never opened, and the
                # way on was the other side.
                self.board_turning, self.board_flipped = -self.board_turning, True
                self.board_leg = 0.0
                self.say("Trying the other way.")
            if self.board_swept > 3 * BOARD_SWEEP_RAD or took > BOARD_GIVE_UP_S:
                self.say("I cannot get past this. Please help me find the way.")
                self.gave_up = ("go2_board", f"the board's guard held the front closed for "
                                f"{took:.0f} s at ({x:.1f}, {y:.1f}), turning "
                                f"{math.degrees(self.board_swept):.0f} deg did not open it")
                self.board_gave_up = True
                return (0.0, 0.0, 0.0)
            return (0.0, 0.0, BOARD_TURN_W * self.board_turning)
        return (vx, vy, w)

    def handler_at(self):
        """Where the person on the handle is: straight behind, along the heading."""
        x, y, yaw = self.robot.get_pose()
        return x - HANDLER_BEHIND * math.cos(yaw), y - HANDLER_BEHIND * math.sin(yaw)

    @staticmethod
    def door_near(gates, xy, reach):
        """Is a doorway on the route, not yet walked through, within `reach`?

        Any of them, not only the first one not yet walked through: see
        door_ahead for why the first is not always the one ahead."""
        for entry, exit_, (ux, uy) in gates:
            depth = (exit_[0] - entry[0]) * ux + (exit_[1] - entry[1]) * uy
            if (xy[0] - entry[0]) * ux + (xy[1] - entry[1]) * uy > depth + FUNNEL_PAST:
                continue                          # through this one already
            if math.dist(xy, entry) < reach:
                return True
        return False

    @staticmethod
    def door_ahead(gates, xy):
        """(entry, exit, axis, s, depth, lateral) of the doorway the dog is
        within a funnel's reach of -- FUNNEL_D before it and FUNNEL_D either
        side of its axis -- and not yet through; None when there is none.

        Every gate is looked at, not only the first one not yet walked
        through. "Through" is measured along a gate's own axis, and a gate
        with a slanted axis that is behind the dog stops reading as walked
        through once the dog is far enough to its side: the floor-1 lift
        doorway's (0.24, 0.97) from 11 m west, on the way to the cafeteria.
        Stopping at that one turned the funnel off for the rest of the leg;
        the dog cut the corner into the cafeteria door's east jamb at
        (33.8, 11.2) and stood there pressed into it until the leg timed out
        (selftest, from #6's lane point)."""
        for entry, exit_, (ux, uy) in gates:
            s = (xy[0] - entry[0]) * ux + (xy[1] - entry[1]) * uy
            depth = (exit_[0] - entry[0]) * ux + (exit_[1] - entry[1]) * uy
            if s > depth + FUNNEL_PAST:
                continue                          # through this one already
            # Not a door the dog is walking past on its way somewhere else:
            # only when it is within a funnel's width of the axis.
            lateral = (xy[0] - entry[0]) * -uy + (xy[1] - entry[1]) * ux
            if s < -FUNNEL_D or abs(lateral) > FUNNEL_D:
                continue                          # not there yet, or not this one
            return entry, exit_, (ux, uy), s, depth, lateral
        return None

    @staticmethod
    def door_through(gates, xy):
        """The point FUNNEL_PAST beyond the far side of the doorway the funnel
        is steering the dog into (see funnel), or None when it is not."""
        door = Run.door_ahead(gates, xy)
        if door is None:
            return None
        _, exit_, (ux, uy), _, _, _ = door
        return (exit_[0] + ux * FUNNEL_PAST, exit_[1] + uy * FUNNEL_PAST)

    @staticmethod
    def funnel(gates, xy, target, off_route=True):
        """(`target`, or a point on the axis of the doorway just ahead; whether
        the dog is within DOOR_NEAR of that doorway). See FUNNEL_D. Lines up
        before the frame only `off_route` (see FUNNEL_OFF)."""
        door = Run.door_ahead(gates, xy)
        if door is None:
            return target, False
        entry, _, (ux, uy), s, depth, lateral = door
        ahead = min(s + FUNNEL_LEAD, depth + FUNNEL_PAST)
        if off_route and abs(lateral) > (ahead - s) * math.tan(FUNNEL_ANGLE):
            ahead = min(ahead, -FUNNEL_PRE)              # line up before the frame
        return (entry[0] + ux * ahead, entry[1] + uy * ahead), -DOOR_NEAR < s < 0.0

    def handler_room(self, live, x, y, yaw, walls_only=False):
        """What the dog can know of the room around the person, for a dog pose:
        walls from the map, measured to their real faces rather than the
        planner's inflated ones (clearance.wall_face_field -- the person's
        shoulder lives in the inflation band), and crates from the LiDAR's
        standing field plus what it saw standing in the last few seconds: a
        crate the dog has just passed is behind it, under its rays, and is
        exactly what a pivot swings the person into (perception.MEMORY_S)."""
        px, py = x - HANDLER_BEHIND * math.cos(yaw), y - HANDLER_BEHIND * math.sin(yaw)
        if walls_only:
            return live.wall_at(px, py)
        return min(live.wall_at(px, py), live.detected_at(px, py),
                   live.remembered_at(px, py))

    def unpin(self, live, pose, dt=0.3):
        """Yaw rate while standing still for somebody: 0, unless the person on
        the handle is within HANDLER_ROOM of something and turning slowly one
        way gives them more room. Waiting is not turning (see the person
        branch), except to stop pressing someone into a wall."""
        x, y, yaw = pose
        now = self.handler_room(live, x, y, yaw)
        if now >= HANDLER_ROOM:
            return 0.0
        # The walls first, as in spare_handler: with the person waited for at
        # their elbow, room is near zero whichever way the dog turns, and it
        # stood 28 deg into a sidestep holding its user on the wall for 3.5 s
        # (room 101 seed 6, --vamos).
        def rank(x_, y_, yaw_):
            walls = self.handler_room(live, x_, y_, yaw_, walls_only=True)
            return (min(walls, HANDLER_ROOM), self.handler_room(live, x_, y_, yaw_))
        here = rank(x, y, yaw)
        best, w_best = here, 0.0
        for w in (UNPIN_W, -UNPIN_W):
            r = rank(x, y, yaw + w * dt)
            if (r[0] > best[0] + 0.02 or (r[0] >= best[0] - 1e-9 and r[1] > best[1] + 0.02)):
                best, w_best = r, w
        return w_best

    @staticmethod
    def nose_closing(live, pose, cmd, dt=0.1):
        """Is a front corner of the dog within NOSE_MIN of something, and would
        this command bring that corner closer still? Walking away is fine.

        Corners, not the middle of the front: turning away from a crate swings
        the middle clear while the corner on the crate's side is still closing
        -- seed 8's front-left leg, on the cart, with the dog turning right."""
        def corners(x, y, yaw):
            fx, fy = x + SWING_NOSE * math.cos(yaw), y + SWING_NOSE * math.sin(yaw)
            return [live.detected_at(fx + side * -math.sin(yaw), fy + side * math.cos(yaw))
                    for side in (-NOSE_HALF_W, 0.0, NOSE_HALF_W)]
        x, y, yaw = pose
        now = corners(x, y, yaw)
        if min(now) >= NOSE_MIN:
            return False
        yaw += cmd[2] * dt
        x += cmd[0] * math.cos(yaw) * dt
        y += cmd[0] * math.sin(yaw) * dt
        return any(a < NOSE_MIN and b < a for a, b in zip(now, corners(x, y, yaw)))

    def swing(self, live, pose, cmd, people=(), dt=0.1, need=ROBOT_R, ahead=0.0):
        """Roll `cmd` forward SWING_T: (least room around the person, whether
        the dog itself stays clear -- of everything by `need`, and of where
        each of `people` is heading by SWING_PEOPLE, or by as much as it has
        now if that is less: someone stopped 0.55 m in front of it would
        otherwise rule out stepping away from them too -- and least room
        between the person and the walls alone, and least room the dog's
        centre or front had along the way). With `ahead`, the line that far
        beyond the front at the end must be clear by `need` too (EASE_AHEAD)."""
        x, y, yaw = pose
        vx, vy, w = cmd
        keep = [min(SWING_PEOPLE, math.dist((x, y), (p.x, p.y)) - 1e-6) for p in people]
        worst, walls, body, clear = INF, INF, INF, True
        for k in range(1, int(round(SWING_T / dt)) + 1):
            yaw += w * dt
            x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            worst = min(worst, self.handler_room(live, x, y, yaw))
            walls = min(walls, self.handler_room(live, x, y, yaw, walls_only=True))
            body = min(body, live(x, y),
                       live.detected_at(x + SWING_NOSE * math.cos(yaw),
                                        y + SWING_NOSE * math.sin(yaw)))
            clear = (clear and body >= need
                     and all(math.dist((x, y), p.predict(k * dt)) >= d
                             for p, d in zip(people, keep)))
        steps = int(round(ahead / dt))
        for k in range(1, steps + 1):
            d = SWING_NOSE + ahead * k / steps
            clear = clear and live(x + d * math.cos(yaw), y + d * math.sin(yaw)) >= need
        return worst, clear, walls, body

    def spare_handler(self, live, pose, cmd, limit, people=()):
        """`cmd`, with its turn eased where it would swing the person into
        something. See HANDLER_ROOM. `limit` is the speed every other rule
        has already allowed; easing never walks faster than it.

        `people` are the person-sized tracks. Easing walks on where the
        command would have pivoted, and on seeds 4 and 12 of `room 201
        --pedestrians 3` that walked the dog to 0.42 m of somebody standing
        beside it. So an eased command that walks further than the original
        must keep SWING_PEOPLE from where each of them is heading, or the
        turn goes ahead as commanded. Only then: holding every eased command
        to it, whether or not it added a step, turned the dog's easing off
        wherever anyone was near, and put the person on the handle back
        against walls on 5 seeds of 12."""
        # The walls apart from everything else. Room is the least distance to a
        # wall *or* anything the LiDAR has found, and a pedestrian at the
        # person's elbow takes it to zero -- after which every turn looked no
        # worse than now, and a full pivot went through as commanded, sweeping
        # them into the wall (room 101 seed 5, --vamos, 5.3 s). Whatever is
        # standing beside them, a turn may not take them nearer a wall than
        # HANDLER_ROOM, or than they already are.
        now = self.handler_room(live, *pose)
        wall_now = self.handler_room(live, *pose, walls_only=True)

        def ok(room, walls):
            return ((room >= HANDLER_ROOM or room >= now)
                    and (walls >= HANDLER_ROOM or walls >= wall_now))

        def rank(room, walls):
            return (min(walls, HANDLER_ROOM, wall_now), room)

        worst, _, walls, body = self.swing(live, pose, cmd)
        if ok(worst, walls):
            self.turn_held = 0
            self.side_held = 0.0
            return cmd
        vx, vy, w = cmd
        v = max(vx, min(SWING_V, limit))
        # The dog is held to ROBOT_R -- or, where the command would put the
        # person into something, to what the command itself would leave it,
        # if that is less. The command goes ahead whenever nothing else is
        # clear, so a stricter bar for the alternatives then only ever picks
        # the turn that hurts: pivoting into the electrical engineering lab,
        # every eased turn and sidestep came within 0.15 m of the corridor
        # wall's inflated band (its face was 0.4 m off), so all were ruled out
        # -- and the pivot, 0.15 m from the same band, swung its user onto the
        # floor-2 crate (Gate D, --vamos, seed 1). Only then: in a doorway the
        # band reads zero, the bar went with it, and a sidestep the frame did
        # not allow won every tick for 37 s, where walking on would have kept
        # its user 0.30 m off the frame (Gate D, map-only, seed 8).
        need = min(ROBOT_R, body) if worst < HANDLER_R else ROBOT_R
        best = None
        for f in SWING_EASE:
            alt = (v, vy, w * f)
            room, clear, aw, _ = self.swing(live, pose, alt, people if v > vx else (),
                                            need=need, ahead=EASE_AHEAD if v > vx else 0.0)
            if not clear:
                continue
            if ok(room, aw):
                best = (rank(room, aw), alt)
                break
            if best is None or rank(room, aw) > best[0]:
                best = (rank(room, aw), alt)
        # Which way a sidestep goes: towards the turn -- but a turn rate this
        # small says nothing about which way, and it flipped sign tick to tick
        # with the dog's nose in a door's jamb, so the dog stepped left, right,
        # left... on the spot until the leg timed out (cafeteria, 278 s). Below
        # SIDE_MIN_W the side already being stepped to is kept; with none, no
        # sidestep.
        side = (math.copysign(1.0, w) if abs(w) >= SIDE_MIN_W
                else self.side_held)
        if (best is None or best[0] < rank(min(HANDLER_ROOM, now), wall_now)) and side:
            # Nothing walking forward will do: step sideways. See SIDE_V.
            for f in SWING_EASE:
                alt = (0.0, side * SIDE_V, w * f)
                room, clear, aw, _ = self.swing(live, pose, alt, people, need=need)
                if not clear:
                    continue
                if ok(room, aw):
                    best = (rank(room, aw), alt)
                    break
                if best is None or rank(room, aw) > best[0]:
                    best = (rank(room, aw), alt)
        touch = min(HANDLER_R, wall_now)
        # Into anything, not only a wall: rank() puts walls first, so a pivot
        # that swung its user onto the floor-1 trolley, 0.00 m of room left
        # and 0.40 m off the walls, went through untouched (--pedestrians 4
        # --seed 4 --puddles 2, 11-16, 2.97 rad/s on the spot).
        touch_any = min(HANDLER_R, now)
        into_thing = worst < touch_any and (best is None or best[0][1] < touch_any)
        pinned = w != 0.0 and ((walls < touch and (best is None or best[0][0] < touch))
                               or into_thing)
        if pinned and self.turn_held < HOLD_TURN_S / self.robot.CONTROL_DT:
            # Every way of making this turn puts them into a wall: hold it.
            # Standing still keeps them where they are; the turn is asked for
            # again next tick, and goes once something has moved.
            self.turn_held += 1
            self.obs["eased"] += 1
            return (0.0, 0.0, 0.0)
        if not pinned:
            self.turn_held = 0
        # Pinned past HOLD_TURN_S, the count stays where it is and the turn goes
        # on, tick after tick, until it no longer would put them into the wall.
        # Reset here as well, it went for one tick and was held for 3 s again:
        # out of the floor-3 lift with its user against the car and the wall,
        # the dog turned 0.04 rad every 3 s and stood there for 215 s.
        if best is None or best[0] <= rank(worst, walls):
            self.side_held = 0.0
            return cmd
        self.obs["eased"] += 1
        self.side_held = math.copysign(1.0, best[1][1]) if best[1][1] else 0.0
        return best[1]

    def score_handler(self, floor):
        """Ground truth for the person: is their body touching anything?

        They are not in the MuJoCo scene, so this asks the scene with rays: a
        ring of HANDLER_RAYS at each of HANDLER_H, and a hit within HANDLER_R
        of their centre is a touch. A ray starting inside a solid box sees
        nothing of it, so one more straight down from above the crates: landing
        on something before the floor means they are standing in it. Scoring
        only, like score_collision.
        """
        import mujoco
        m, d = self.robot.model, self.robot.data
        px, py = self.handler_at()
        z0 = levels.floor_z(floor)
        gid = np.zeros(1, np.int32)
        touched = set()

        def ray(p, v, reach):
            dist = mujoco.mj_ray(m, d, np.array(p, float), np.array(v, float),
                                 None, 1, -1, gid)
            if 0 <= dist <= reach and int(gid[0]) in self.touchable:
                touched.add(self.touchable[int(gid[0])])

        for h in HANDLER_H:
            for k in range(HANDLER_RAYS):
                a = 2 * math.pi * k / HANDLER_RAYS
                ray((px, py, z0 + h), (math.cos(a), math.sin(a), 0.0), HANDLER_R)
        ray((px, py, z0 + 1.5), (0.0, 0.0, -1.0), 1.45)
        # When and where each touch starts and ends, so a total can be traced
        # back to the moment it happened.
        x, y, yaw = self.robot.get_pose()
        if touched and self.handler_hits["since"] is None:
            self.handler_hits["since"] = self.robot.now()
            print(f"    t={self.robot.now():6.1f}s  HANDLER touch starts: dog "
                  f"({x:.2f}, {y:.2f}) heading {math.degrees(yaw):.0f} deg, person "
                  f"({px:.2f}, {py:.2f}), touching {', '.join(sorted(touched))}")
        elif not touched and self.handler_hits["since"] is not None:
            print(f"    t={self.robot.now():6.1f}s  HANDLER touch ends after "
                  f"{self.robot.now() - self.handler_hits['since']:.1f}s: dog "
                  f"({x:.2f}, {y:.2f}) heading {math.degrees(yaw):.0f} deg, person "
                  f"({px:.2f}, {py:.2f})")
            self.handler_hits["since"] = None
        if touched:
            self.handler_hits["ticks"] += 1
            self.handler_hits["boxes"].update(f"floor {floor} {what}" for what in touched)

    # -- the crowd ------------------------------------------------------
    def place_crowd(self):
        """Put every pedestrian body where its walker is, this tick.

        Every floor's, not just this one's: the dog can see up a stairwell and
        out of the lift, and a person who only exists while the dog is on
        their storey pops into being mid-corridor.
        """
        poses = [p for crowd in self.crowds.values() for p in crowd.poses()]
        self.robot.move_mocaps(poses)

    def step_crowd(self):
        """One control tick of walking, for everyone.

        Called once per tick from every loop that also steps the robot --
        follow, and the scripted lift moves. A crowd that freezes while the
        dog rides the lift would have the whole building hold still for it.
        """
        dog = tuple(self.robot.get_pose())      # an aware crowd needs the heading too
        for floor, crowd in self.crowds.items():
            crowd.step(self.robot.CONTROL_DT, dog if floor == self.floor else None)
        self.place_crowd()

    def ground_truth(self, floor, pose, target, live):
        """What was really on the walk the network judged: "stairs", "person",
        "obstacle", "wall" or "open floor" -- scoring only, never fed back.

        Points every 0.1 m from the dog to `target` (dog frame), out to
        AFF_HALF_W either side: the band the label's own path_cells covers."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        tx, ty = target
        L = math.hypot(tx, ty)
        ux, uy = tx / L, ty / L
        band = []
        for a in np.arange(0.0, L + 1e-9, 0.1):
            for b in np.arange(-AFF_HALF_W, AFF_HALF_W + 1e-9, 0.1):
                fx, fy = a * ux - b * uy, a * uy + b * ux
                band.append((x + c * fx - s * fy, y + s * fx + c * fy))
        if any(self.router.hazard_at(p) is not None for p in band):
            return "stairs"
        people = list(self.crowds[floor].positions())
        if any(math.dist(p, q) < 0.45 for p in band for q in people):
            return "person"
        for bx0, bx1, by0, by1, _h, _name in obstacles.boxes(floor):
            if any(bx0 - 0.1 <= px <= bx1 + 0.1 and by0 - 0.1 <= py <= by1 + 0.1
                   for px, py in band):
                return "obstacle"
        if any(live.wall_at(*p) < 0.05 for p in band):
            return "wall"
        return "open floor"

    def shadow_affordance(self, floor, pose, steer_xy, live):
        """Ask the network about the walk toward the steering target; log it.

        --affordance-shadow only: nothing here changes what the dog does."""
        from cyberdog.affordance import data as adata
        from cyberdog.affordance.runtime import ahead
        target = ahead(pose, steer_xy)
        if target is None:
            self.aff_n["skipped: turning or too near"] += 1
            return
        cls, probs, seen = self.aff.judge(self.elev.patch(pose, live.wall_at), target)
        if cls is None:
            self.aff_n["skipped: too little seen"] += 1
            return
        truth = self.ground_truth(floor, pose, target, live)
        self.aff_n[(adata.CLASS_NAMES[cls], truth)] += 1
        self.aff_log.write(json.dumps({
            "t": round(self.robot.now(), 2), "floor": floor,
            "pose": [round(v, 3) for v in pose], "target": [round(v, 3) for v in target],
            "class": adata.CLASS_NAMES[cls], "probs": [round(float(v), 3) for v in probs],
            "seen": round(seen, 3), "truth": truth}) + "\n")

    def score_people(self, floor, x, y):
        """Ground truth again: how near the dog actually got to a person.

        Same rule as score_collision -- never fed back into the robot. The
        dog stops for people because it tracked them, or it does not stop.

        And who closed the gap. A person walking into a dog that is standing
        still is a contact, and is reported as one, but it is not the dog
        walking at them: each tick inside PED_NEAR the closing speed is split
        into the dog's part (its own motion along the line to them) and
        theirs. Over DOG_CLOSING of the dog's own is the dog's doing.
        """
        now = self.robot.now()
        prev = self.ped.get("prev")
        dt = now - prev[0] if prev is not None and prev[3] == floor else 0.0
        seen = {}
        near = []
        for w in self.crowds[floor].walkers:
            d = math.hypot(w.x - x, w.y - y)
            seen[id(w)] = (w.x, w.y)
            self.ped["min_d"] = min(self.ped["min_d"], d)
            if d < PED_NEAR:
                self.ped["near"] += 1
                near.append((d, w))
                if dt > 0.0 and d > 1e-6:
                    ux, uy = (w.x - x) / d, (w.y - y) / d
                    v = ((x - prev[1]) * ux + (y - prev[2]) * uy) / dt
                    if v > DOG_CLOSING:
                        self.ped["dog_in"] = self.ped.get("dog_in", 0) + 1
                        self.ped["dog_v"] = max(self.ped.get("dog_v", 0.0), v)
                        self.ped["dog_in_now"] = self.ped.get("dog_in_now", 0) + 1
        self.ped["prev"] = (now, x, y, floor, seen)
        # When and where, as for the handle: a total alone cannot be traced
        # back to the moment it happened.
        yaw = self.robot.get_pose()[2]
        if near and self.ped["since"] is None:
            self.ped["since"] = self.robot.now()
            d, w = min(near, key=lambda n: n[0])
            hx, hy = w.heading()
            print(f"    t={self.robot.now():6.1f}s  CONTACT starts: dog ({x:.2f}, {y:.2f}) "
                  f"heading {math.degrees(yaw):.0f} deg, {w.kind} walker ({w.x:.2f}, {w.y:.2f}) "
                  f"heading {math.degrees(math.atan2(hy, hx)):.0f} deg, {d:.2f} m")
        elif not near and self.ped["since"] is not None:
            k = self.ped.pop("dog_in_now", 0)
            print(f"    t={self.robot.now():6.1f}s  CONTACT ends after "
                  f"{self.robot.now() - self.ped['since']:.1f}s: dog ({x:.2f}, {y:.2f}) "
                  f"heading {math.degrees(yaw):.0f} deg; "
                  + (f"the dog closing in for {k / FPS:.1f}s of it" if k
                     else "the dog never closing in"))
            self.ped["since"] = None

    def yield_to(self, tracks, xy, yaw, speed):
        """The person to wait for, or None. Holds the decision between ticks.

        Two thresholds, not one. Starting to wait asks MISS_R; carrying on
        waiting asks CLEAR_R, which is wider. With a single number the dog
        moves off the instant it is a millimetre clear, immediately conflicts
        again, and shuffles its way past somebody in a series of twitches.
        The gap between the two is what makes it one stop and one start.

        And once it is waiting for somebody it keeps waiting for *them*, not
        for whoever happens to look threatening this tick. That fixes a nasty
        little deadlock: a person who has stopped -- because the dog is in
        front of them, which is why they stopped -- has no velocity, so the
        moving test calls them a crate, so the dog drives at them, so they
        stay stopped. Measured without the latch: 43 stop-starts in one run
        and an ending nose-first against the floor-2 cart.

        PATIENCE is the other end of it, and it is not a tidy-up. Waiting is
        only the right answer while they are actually going somewhere. Someone
        who has stopped and stayed stopped -- reading a noticeboard, holding a
        door -- is furniture now, and furniture is `free_destination`'s job: go
        round it. Without the release the dog waits out the whole run for a
        person who has no intention of moving, which it did, for 267 seconds,
        four metres from a lift it never reached.
        """
        vel = (speed * math.cos(yaw), speed * math.sin(yaw))
        # Walking past in another lane is not a conflict: see PASS_LAT.
        passers = [t for t in tracks if t.moving and passes_by(xy, yaw, t)]
        self.passer_near = any(math.dist(xy, (t.x, t.y)) < PASS_R for t in passers)
        tracks = [t for t in tracks if t not in passers]

        held = self.waiting_for if self.waiting_for in tracks else None
        if held is not None:
            d = closest_approach(xy, vel, (held.x, held.y), (held.vx, held.vy))
            self.held_still = 0 if held.moving else self.held_still + 1
            # Somebody who stops in the way is in the way like anything else:
            # PATIENCE for them to walk on, and then round them (ASK_S).
            if d - min(held.r, 0.4) < CLEAR_R and self.held_still < PATIENCE * FPS:
                self.waited += 1
                self.ped["waited"] += 1
                return held
            if self.held_still >= PATIENCE * FPS:
                # Not "they have stopped": the track can be a person who has
                # walked on, merged into the crate beside them (room 101 seed
                # 6, the trolley). Either way the way is not clearing.
                self.say("The way is still blocked. Going around.")
                self.passing = (held.x, held.y)
                self.waiting_for, self.waited, self.held_still = None, 0, 0
                return None

        person = conflict(xy, vel, tracks, radius=MISS_R)

        if person is None:
            if self.waiting_for is not None:
                self.say("Thank you. Carrying on.")
            self.waiting_for, self.waited, self.held_still = None, 0, 0
            return None
        return self.wait_for(person, xy, yaw)

    def ask_in_way(self, over, n):
        """Past BLOCKED_S with no way round (`over` ticks past it): True while
        ASK_S is not up -- the caller stands still -- with EXCUSE said every
        EXCUSE_S, whatever is there. Then False: find_another_way."""
        if over > ASK_S * FPS:
            return False
        if self.excused is None or n - self.excused >= EXCUSE_S * FPS:
            self.say(EXCUSE)
            self.excused = n
        return True

    def find_another_way(self, floor, points, pose, why):
        """Asked, and still blocked: say so, and end the leg for main to close
        the spot on the map and plan the rest again (see ASK_S)."""
        self.robot.set_velocity(0.0, 0.0, 0.0)
        self.say("This way is blocked. Finding another way.")
        self.reroute = (floor, blocked_point(points, pose), pose[2])
        self.gave_up = ("blocked", why + f", then {ASK_S:.0f} s asking")

    def wait_for(self, person, xy, yaw):
        """Stand still for `person`, and hold on to them: see yield_to.

        Every wait goes through here, so every one is latched and said. The
        one go_round hands back (somebody inside NEAR_R, no way round) used
        to stand still for a tick without either: no "Excuse me", and no
        latch, so when the person paused -- because the dog was in front of
        them -- their track fell under MOVING_V, nothing asked about them any
        more, and the dog walked on at them while they stood there: 0.46 m,
        turning towards them (--pedestrians 4 --seed 6 --puddles 2, 11-16)."""
        if self.waiting_for is None:
            self.ped["yields"] += 1
            if self.planner == "cerrt":
                self.why_wait[self.no_plan or "not asked"] += 1
            if os.environ.get("CYBERDOG_TRACE"):
                # Every yield in a building with nobody in it is a crate being
                # mistaken for a person. This is how to see which crate.
                print(f"    [trace] yield to ({person.x:.1f},{person.y:.1f}) "
                      f"v=({person.vx:.2f},{person.vy:.2f}) |v|={person.speed:.2f} "
                      f"r={person.r:.2f} from ({xy[0]:.1f},{xy[1]:.1f})")
            # Who, and where, because the person on the handle can hear that
            # and cannot see it.
            self.say(describe_person(xy, yaw, (person.x, person.y),
                                     (person.vx, person.vy)) + ". Waiting.")
            # Not "Excuse me": they are walking, and will pass. That is for a
            # way with no way round at all (ASK_S).
        self.waiting_for = person
        self.waited += 1
        self.ped["waited"] += 1
        return person

    @staticmethod
    def straight_goal(xy, route):
        """Where a going-round aims: PLAN_D along the route, or PLAN_SHORT before
        its next corner if that comes first. None if there is not PLAN_MIN of
        straight route to go round anybody on."""
        pts = [tuple(xy)] + [tuple(p) for p in route]
        walked, heading = 0.0, None
        for a, b in zip(pts, pts[1:]):
            d = math.dist(a, b)
            if d < 1e-6:
                continue
            h = math.atan2(b[1] - a[1], b[0] - a[0])
            if heading is not None and abs(wrap(h - heading)) > PLAN_CORNER:
                reach = walked - PLAN_SHORT          # a corner at `a`
                break
            heading = h
            if walked + d >= PLAN_D:
                reach = PLAN_D
                break
            walked += d
        else:
            reach = walked                           # the route ends first
        if reach < PLAN_MIN:
            return None
        walked = 0.0
        for a, b in zip(pts, pts[1:]):
            d = math.dist(a, b)
            if walked + d >= reach and d > 1e-6:
                f = (reach - walked) / d
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            walked += d
        return pts[-1]

    def plan_backoff(self, n):
        """Tick n is inside PLAN_RETRY_S of a search that found nothing, with
        no plan to walk: wait rather than search again."""
        return (self.plan is None and self.plan_failed is not None
                and n - self.plan_failed < round(PLAN_RETRY_S * FPS))

    def plan_hold(self, n):
        """Tick n is inside PLAN_HOLD_S of the last search, with a plan being
        walked: check that plan (ce_rrt.still_good) rather than search again."""
        return (self.plan is not None and self.searched_at is not None
                and n - self.searched_at < round(PLAN_HOLD_S * FPS))

    def go_round(self, live, tracks, pose, route, gates, n, speed, ignore=()):
        """(where to steer to go round people, or None to carry on as before;
        somebody to stand still for). --planner cerrt.

        Asked every tick; plans every ce_rrt.REPLAN_S while somebody's path
        meets the dog's or a plan is being walked -- within PLAN_HOLD_S of the
        last search, a plan being walked is only checked (still_good) and
        kept, and searched for again only if the check fails. A plan that
        cannot be made again, and is no longer safe as it stands (ce_rrt.still_good), is
        dropped there and then, and the dog falls back on the wait:
        the chance constraint only covers the next ce_rrt.COMMIT_T, and that
        promise rests on there being a fallback when the next plan fails.

        The static rules are the ones the dog already keeps -- the map, the
        crates, the room round the person on the handle -- each held to its
        threshold, or to what the dog has now if that is less: a plan
        starting 0.2 m from a wall is not refused for being there, only for
        going closer."""
        x, y, yaw = pose
        vel = (speed * math.cos(yaw), speed * math.sin(yaw))
        # `ignore`: people overtaking the dog. Not a reason to go round
        # anybody, but still people every plan has to keep clear of.
        others = [t for t in tracks if t not in ignore]
        held = self.waiting_for if self.waiting_for in others else None
        threat = conflict((x, y), vel, others, radius=MISS_R) or held
        close = None
        if threat is None:
            # conflict() stops asking inside NEAR_R, and yield_to with it. A
            # going-round leaves the dog beside people, though, and room 101
            # seed 8 stood a metre from somebody who then walked across its
            # nose: nothing asked, and it walked on into them (0.50 m). Here
            # anybody that close whose way meets the dog's is asked about.
            for t in others:
                if (t.moving and math.dist((x, y), (t.x, t.y)) < NEAR_R
                        and closest_approach((x, y), vel, (t.x, t.y), (t.vx, t.vy))
                        - min(t.r, 0.4) < MISS_R):
                    threat = close = t
                    break
        self.no_plan = None
        if threat is None and self.plan is None:
            self.plan_failed, self.plan_speed = None, 0     # the next going-round starts afresh
            return None, None
        if self.door_near(gates, (x, y), PLAN_DOOR):
            self.plan, self.no_plan = None, "doorway"
            return None, close
        if self.plan_backoff(n):
            # Nothing found a moment ago: wait, as for no plan, and search
            # again in PLAN_RETRY_S, not on every tick.
            self.no_plan = "no safe path"
            return None, close
        if self.plan is None or n - self.plan_at >= round(ce_rrt.REPLAN_S * FPS):
            # The same goal for the whole of one going-round. Picked afresh
            # PLAN_D ahead at every replan, it slid along with the dog, the
            # last plan no longer ended at it, and every replan was a new
            # search that now and then found nothing: room 201 seed 10 went
            # from 1 stop to 11, stopping and starting between plans.
            if (self.plan is None or self.plan_goal is None
                    or math.dist((x, y), self.plan_goal) < PLAN_D / 2):
                self.plan_goal = self.straight_goal((x, y), route)
            goal = self.plan_goal
            if goal is None:
                self.plan, self.no_plan = None, "corner"   # wait, as before
                return None, close
            people = [ce_rrt.Person(t.x, t.y, t.vx, t.vy, t.r) for t in tracks
                      if t.moving or (t.walked and MIN_RADIUS <= t.r <= MAX_RADIUS)]
            def nose(px, py, pyaw):
                return live.wall_at(px + SWING_NOSE * math.cos(pyaw),
                                    py + SWING_NOSE * math.sin(pyaw))

            # Each threshold, or what the dog has now if that is less -- taken
            # once, when this going-round starts. Taken at every replan, it
            # ratcheted: each plan could end a little nearer the wall than the
            # last began, and 53 replans walked the dog's front onto the
            # corridor wall by the cafeteria door (Gate D, --vamos, seed 2).
            if self.plan is None or self.plan_floor is None:
                self.plan_floor = (min(ROBOT_R, live.static_at(x, y)),
                                   min(PLAN_CRATE, live.detected_at(x, y)),
                                   min(PLAN_NOSE, nose(x, y, yaw)),
                                   min(HANDLER_ROOM, self.handler_room(live, x, y, yaw)))
            s0, c0, n0, h0 = self.plan_floor

            # static_at >= s0, detected_at >= c0, nose() >= n0 and
            # handler_room() >= h0, as one fast closure: a search asks it
            # ~10-30 thousand times (LiveClearance.pose_test).
            ok = live.pose_test(s0, c0, n0, h0, SWING_NOSE, HANDLER_BEHIND)

            near = min((math.dist((x, y), (p.x, p.y)) for p in people), default=INF)
            if near < ce_rrt.need(0.0):
                # Already inside the chance margin: no plan can start here.
                self.plan, self.no_plan = None, "too close"
                return None, close

            warm = ahead(self.plan.path, (x, y)) if self.plan else None
            if self.plan_hold(n) and ce_rrt.still_good(
                    (x, y, yaw), warm, ok, people, handler_behind=HANDLER_BEHIND,
                    v=self.plan.v):
                # Searched a moment ago and the plan is still safe against
                # where everybody is predicted now: walk on, search again
                # PLAN_HOLD_S after the last search (see PLAN_HOLD_S).
                self.plan = self.plan._replace(path=[(x, y)] + list(warm[1:]))
                self.plan_at = n
                self.plans["held"] += 1
                return self.walk_plan((x, y), close)

            t0 = time.perf_counter()
            # One speed a search (see PLAN_RETRY_S); the next search after
            # one that found nothing tries the next speed.
            new = ce_rrt.plan((x, y, yaw), goal, ok, people, warm=warm,
                              handler_behind=HANDLER_BEHIND,
                              v=ce_rrt.SPEEDS[self.plan_speed], seed=n)
            if new is None:
                self.plan_speed = (self.plan_speed + 1) % len(ce_rrt.SPEEDS)
            ms = (time.perf_counter() - t0) * 1000
            self.plans["ms"] += ms
            self.plans["worst_ms"] = max(self.plans["worst_ms"], ms)
            self.plan_at = self.searched_at = n
            if new is None and self.plan is not None and ce_rrt.still_good(
                    (x, y, yaw), warm, ok, people, handler_behind=HANDLER_BEHIND,
                    v=self.plan.v):
                # Nothing new found, and the plan being walked is still safe
                # against where everybody is predicted now: keep walking it.
                new = self.plan._replace(path=[(x, y)] + list(warm[1:]))
            if new is None:
                self.plans["none"] += 1
                self.plan_failed = n
                self.plan, self.no_plan = None, "no safe path"
                # No way round. Further off, yield_to waits for them; this
                # close it would not, so stand still here.
                return None, close
            self.plans["made"] += 1
            self.plan_failed = None
            if self.plan is None and threat is not None:
                # Said before the dog leans, as for a crate: which way the
                # handle is about to go is what the person cannot see coming.
                self.plans["rounds"] += 1
                side = "left" if new.side > 0 else "right"
                self.say(describe_person((x, y), yaw, (threat.x, threat.y),
                                         (threat.vx, threat.vy))
                         + f". Going round, to your {side}.")
                self.waiting_for, self.waited, self.held_still = None, 0, 0
            self.plan = new
        return self.walk_plan((x, y), close)

    def walk_plan(self, xy, close):
        """go_round's answer while self.plan is walked from xy."""
        rest = ahead(self.plan.path, xy)
        if len(rest) < 2 or math.dist(xy, rest[-1]) < ce_rrt.GOAL_TOL:
            self.plan = None             # walked to its end: back on the route
            return None, close
        self.plans["ticks"] += 1
        return path_target(rest, xy), None

    def mention_overtakers(self, behind, xy, yaw, speed=0.0):
        """Tell the person on the handle that somebody is passing close, and on
        which side -- they will feel them go by and cannot see them -- once a
        person: an overtaker a little faster than the dog is beside it for
        6-10 s, and a repeat every few seconds was said 4-5 times a route."""
        self.overtakers_said = [t for t in self.overtakers_said if t in behind]
        c, s = math.cos(yaw), math.sin(yaw)
        for t in behind:
            if t in self.overtakers_said:
                continue
            fwd = (t.x - xy[0]) * c + (t.y - xy[1]) * s
            lat = -(t.x - xy[0]) * s + (t.y - xy[1]) * c
            if t.vx * c + t.vy * s < speed:
                continue                  # slower than the dog: falling back, not passing
                                          # (the one it has just overtaken, HIL 11-10 16:19)
            if not -HANDLER_BEHIND - 1.0 < fwd < OVERTAKE_AHEAD or abs(lat) > PASS_R:
                continue                  # not beside either of them
            if abs(lat) < SIDE_DEAD:
                continue                  # straight behind: which side is not known yet
            self.overtakers_said.append(t)
            side, now = (1 if lat > 0 else -1), self.robot.now()
            if now - self.overtake_said_at.get(side, -INF) > OVERTAKE_SAY_S:
                self.say(f"Someone passing on your {'left' if side > 0 else 'right'}.")
                self.overtake_said_at[side] = now
            return

    def follow_lead(self, tracks, xy, yaw, waypoints, i, gates, cruise, can_pass=True):
        """(forward speed cap or None, metres left of the lane to steer, the
        person being followed or overtaken or None). See FOLLOW_GAP.

        Somebody slower ahead in the dog's lane is followed FOLLOW_GAP behind,
        or, when they are OVERTAKE_DV slower than `cruise`, the route stays
        straight and clear of doors for the whole pass, and nobody else comes
        near the passing line meanwhile (overtake_plan), overtaken on the left
        and left OVERTAKE_BACK behind before the dog moves back in. A pass
        that gets blocked while the dog is still FOLLOW_MIN behind them drops
        back to following; once beside them it is finished (anybody coming is
        then the ordinary yield's business). `can_pass` False: follow only
        (Pace not known yet)."""
        u = lane_dir(xy, waypoints, i, yaw)
        a = waypoints[i - 1] if i > 0 else xy
        dog_lat = -(xy[0] - a[0]) * u[1] + (xy[1] - a[1]) * u[0]
        now = self.robot.now()

        def fwd_of(t):
            return (t.x - xy[0]) * u[0] + (t.y - xy[1]) * u[1]

        t = self.passing_lead
        if t is not None and t not in tracks:
            # A track breaks and comes back as a new one: take whoever is
            # where they were heading, or the pass is over.
            px, py = t.x + t.vx * self.robot.CONTROL_DT, t.y + t.vy * self.robot.CONTROL_DT
            t = min(tracks, key=lambda q: math.dist((q.x, q.y), (px, py)), default=None)
            if t is not None and math.dist((t.x, t.y), (px, py)) > REACQUIRE_R:
                t = None
        if t is not None:
            gap = fwd_of(t)
            if gap < -OVERTAKE_BACK:
                t = None                                  # past them: back into the lane
            elif gap > FOLLOW_MIN and overtake_plan(xy, u, t, tracks, cruise,
                                                    OVERTAKE_LAT - dog_lat, True) is None:
                t = None                                  # blocked: drop back and follow
            else:
                self.passing_lead = t
                self.pass_said_at = now
                # Not drawing level until out to the side by FOLLOW_MIN: no
                # closing in faster than they walk while still less.
                lat = -(t.x - xy[0]) * u[1] + (t.y - xy[1]) * u[0]
                cap = None
                if gap > 0.0 and abs(lat) < FOLLOW_MIN:
                    v_lead = t.vx * u[0] + t.vy * u[1]
                    cap = max(follow_speed(gap, v_lead), v_lead)
                return cap, OVERTAKE_LAT, t
        self.passing_lead = None

        lead = leader(xy, u, tracks, cruise, dog_lat, keep=self.lead_track)
        if lead is None:
            # A tick or two without them is kept (LEAD_GRACE): their pace and
            # how long they were watched carry on if they come back.
            if self.lead_track is not None and now - self.lead_t > LEAD_GRACE:
                self.lead_track = None
            return None, 0.0, None
        gap = fwd_of(lead)
        # Their pace, smoothed over LEAD_TAU, carried over a broken track.
        v_now = lead.vx * u[0] + lead.vy * u[1]
        prev = self.lead_track
        same = prev is not None and (lead is prev or math.dist(
            (lead.x, lead.y), (prev.x + prev.vx * (now - self.lead_t),
                               prev.y + prev.vy * (now - self.lead_t))) < REACQUIRE_R)
        if same:
            dt = max(0.0, now - self.lead_t)
            self.lead_v += min(1.0, dt / LEAD_TAU) * (v_now - self.lead_v)
            self.lead_seen += dt
        else:
            self.lead_v, self.lead_seen = v_now, 0.0
        self.lead_track, self.lead_t = lead, now
        plan = overtake_plan(xy, u, lead, tracks, cruise, OVERTAKE_LAT - dog_lat,
                             v_lead=self.lead_v)
        if (can_pass and self.lead_seen >= LEAD_SEEN and plan is not None and plan[1] <= straight_ahead(xy, waypoints[i:])
                and not Run.door_near(gates, xy, plan[1])):
            self.passing_lead = lead
            if now - self.pass_said_at > OVERTAKE_SAY_S:
                self.say("Passing someone on your left.")
            self.pass_said_at = now
            return max(follow_speed(gap, plan[2]), plan[2]), OVERTAKE_LAT, lead
        cap = follow_speed(gap, lead.vx * u[0] + lead.vy * u[1])
        if cap < cruise:
            # Said once the dog is held back by them, not while they are
            # still far enough ahead to make no difference (HIL 11-10 15:42,
            # t 21.0: 5.6 m ahead, the cap 2.8 m/s).
            if now - self.follow_said_at > OVERTAKE_SAY_S:
                self.say("Following someone ahead.")
            self.follow_said_at = now
        return cap, 0.0, lead

    def step_aside(self, live, tracks, pose):
        """Sideways m/s for a waiting dog to get out of somebody's way, or 0.
        See EVADE_R.

        Left, right or stay, judged by the least predicted distance to anybody
        walking over the next EVADE_T, the dog stepping at SIDE_V for
        EVADE_STEP_T and then standing. A step must win by EVADE_GAIN, and it
        must not take the dog -- centre or front -- or the room round the
        person on the handle below its threshold, or below what it is now if
        that is less, nor bring the person on the handle within
        ce_rrt.HANDLER_SEP of anybody's predicted path.

        On a robot with its own LiDAR guard (go2_board) the board's word on
        the room sideways goes too, as in respect_board: a step needs more
        than BOARD_SIDE_ROOM free that way, and a blind board (no fresh
        frame) gets no step. The brain's own clearance once judged a step
        towards the trolley fine with 0.09 m free on the board."""
        x, y, yaw = pose
        people = [t for t in tracks
                  if (t.moving or t.walked) and t.speed >= EVADE_V]
        if not people:
            self.evading = 0.0
            return 0.0
        board = None
        if getattr(self.robot, "GUARDED", False):
            board = self.robot.board_clearance()
            if board is None:                            # blind: the board will not step it
                self.evading = 0.0
                return 0.0
        lx, ly = -math.sin(yaw), math.cos(yaw)          # the dog's left
        bx, by = -HANDLER_BEHIND * math.cos(yaw), -HANDLER_BEHIND * math.sin(yaw)
        # Everybody else who may be a person: stood still, or not yet seen to
        # walk. `people` above is only who is walking, and a step away from
        # them only had to keep ROBOT_R off anything else -- the dog stepped
        # left, from somebody walking up on its right towards somebody stood
        # 0.7 m off on its left (--pedestrians 4 --seed 1 --puddles 2, 11-16).
        standing = [t for t in tracks if t not in people
                    and (t.walked or MIN_RADIUS <= t.r <= MAX_RADIUS)]

        def least(v):
            dog = handle = INF
            for k in range(1, int(round(EVADE_T / 0.1)) + 1):
                t = k * 0.1
                s_ = v * min(t, EVADE_STEP_T)
                dx, dy = x + lx * s_, y + ly * s_
                for p in people:
                    px, py = p.predict(t)
                    dog = min(dog, math.hypot(dx - px, dy - py))
                    handle = min(handle, math.hypot(dx + bx - px, dy + by - py))
            return dog, handle

        stay, stay_h = least(0.0)
        if stay >= EVADE_R:
            self.evading = 0.0
            return 0.0

        def nose(px, py):
            return live.wall_at(px + SWING_NOSE * math.cos(yaw),
                                py + SWING_NOSE * math.sin(yaw))

        floors = (min(ROBOT_R, live.static_at(x, y)), min(ROBOT_R, live.detected_at(x, y)),
                  min(PLAN_NOSE, nose(x, y)),
                  min(HANDLER_ROOM, self.handler_room(live, x, y, yaw)))
        best, best_d = 0.0, stay + EVADE_GAIN
        for v in (SIDE_V, -SIDE_V):
            if board is not None and board["left" if v > 0 else "right"] <= BOARD_SIDE_ROOM:
                continue                                 # the board sees too little room that way
            s_ = v * EVADE_STEP_T
            ex, ey = x + lx * s_, y + ly * s_
            if lift.in_doorway((ex, ey)):
                continue                                 # back into the lift's mouth: see EXIT_CLEAR_M
            if (live.static_at(ex, ey) < floors[0] or live.detected_at(ex, ey) < floors[1]
                    or nose(ex, ey) < floors[2]
                    or self.handler_room(live, ex, ey, yaw) < floors[3]):
                continue
            if any(math.hypot(ex - s.x, ey - s.y) < min(EVADE_KEEP, math.hypot(x - s.x, y - s.y))
                   for s in standing):
                continue                                 # away from one, into another
            d, h = least(v)
            if h < min(ce_rrt.HANDLER_SEP, stay_h):
                continue
            if d > best_d:
                best, best_d = v, d
        if best and not self.evading:
            self.evade["steps"] += 1
            # Said when the side is new, or it has been quiet a while: a step
            # aside flickers off and on with the track, and each one was said
            # -- "stepping to your left" 3 times and "right" 3 times inside 3 s
            # in the floor-3 server room (HIL 11-09 15:27). A change of side is
            # always said; the person is about to be led the other way.
            now, side = self.robot.now(), math.copysign(1.0, best)
            if side != self.evade_said[1] or now - self.evade_said[0] > SAY_AGAIN_S:
                self.say(f"Someone close. Stepping to your {'left' if best > 0 else 'right'}.")
                self.evade_said = (now, side)
        if best:
            self.evade["ticks"] += 1
        self.evading = best
        return best

    def follow(self, waypoints, floor, announcements=None, verbose=True,
               goal_r=GOAL_R, to_lift=False):
        """Walk one floor's leg. Same controller as run_demo, plus the frames.

        `goal_r` is how near the last waypoint counts as arrived -- tightened
        for the legs that end at the lift, where the remaining half metre is
        walked scripted and has to start outside the car. On those (`to_lift`)
        the doorway counts too: a dog that comes at the waiting point on a
        slant passes it more than goal_r off, and turning back for it the
        turn-easing walked it on into the car instead -- 1 m in, sidestepping
        a wall for 190 s until the leg timed out (HIL 11-09 14:49). In the
        doorway it is still outside the car, facing it: lift_ride's own start.
        """
        self.floor = floor
        self.robot.set_height(levels.floor_z(floor))
        if self.exit_from != floor:
            self.exit_from = None            # not a leg out of the lift: see EXIT_CLEAR_M
        # An answer still on its way from the last leg is about a corridor the
        # dog has left -- another floor, or the far side of a lift ride.
        if self.pending is not None:
            self.pending = None
            self.vlm["dropped"] += 1
        self.passing = None
        if not waypoints:
            # Asked for where it already is. A* has no checkpoints to give for
            # a route of zero length, and every line below indexes into them --
            # `waypoints[-1]` on an empty list ended the run in an IndexError
            # rather than an answer, for "take me to the main entrance" said at
            # the main entrance.
            self.say("We are already there.")
            return True, 0
        if self.autolabel is not None:
            self.autolabel.start_leg(floor)
        i = 1 if len(waypoints) > 1 else 0
        said = set()
        gates = self.router.gates(floor, waypoints)
        if self.fg is not None:
            self.fg.drop_pending()
            if self.puddle_n:
                # Every door on the floor, not only the route's own: a side
                # door also reads as open floor beside a puddle.
                doors = [tuple(e.get("door_xy") or e["xy"])
                         for entries in self.router.locations.values()
                         for e in entries if e["floor"] == floor]
                placed = fguard.place_puddles(
                    self.puddle_n, floor, waypoints, self.clearance(floor)._static,
                    self.puddle_rng, first_id=self.fg.next_id(),
                    near_door=lambda xy: (self.door_near(gates, xy, fguard.PUDDLE_DOOR)
                                          or any(math.hypot(xy[0] - d[0], xy[1] - d[1])
                                                 < fguard.PUDDLE_DOOR for d in doors)))
                self.fg.add(placed)
                if verbose and placed:
                    print("  puddles: " + ", ".join(f"({p.x:.1f}, {p.y:.1f}) r {p.r:.2f}"
                                                   for p in placed))
        # `proposed` is what VAMOS last offered and the gate let through;
        # `chosen` is what steers -- the same thing under --vamos, never
        # anything under --shadow.
        proposed, chosen, candidates, safety = None, None, [], 1.0
        hunting = halted = False
        stalled = no_goal = unseen = 0
        self.excused = None                  # see ask_in_way
        self.reroute = None                  # see find_another_way
        blind_n = 0                          # ticks blind ahead: see BLIND_FRAC
        progress = None           # (x, y, tick) the last time the dog had moved PROGRESS_M
        lean = 0.0                # the side of a detour once it is committed
        held, against = None, 0   # its last aim, and ticks the other side was offered

        for n in range(MAX_TICKS):
            x, y, yaw = self.robot.get_pose()
            if progress is None or math.hypot(x - progress[0], y - progress[1]) > PROGRESS_M:
                progress = (x, y, n)

            # Before anything else: is this a place the dog must not be? The
            # planner routes around the stairwells, so reaching one means a
            # layer below it went wrong, and the safe answer is to stop dead.
            hazard = self.router.hazard_at((x, y))
            if hazard is not None:
                self.robot.set_velocity(0.0, 0.0, 0.0)
                self.say(hazard)
                raise HazardStop(f"stopped on floor {floor} at ({x:.1f}, {y:.1f}): {hazard}")

            i = advance(i, x, y, waypoints)
            # Preemptive, which is the whole point: the line comes while the
            # turn is still ahead, not as the dog is already in it. The last
            # checkpoint is the exception -- "destination reached" is said on
            # arrival, below, not a couple of metres out.
            if announcements and i not in said and i < len(waypoints) - 1 \
                    and math.hypot(waypoints[i][0] - x, waypoints[i][1] - y) < ANNOUNCE_R:
                said.add(i)
                for line in announcements[i]:
                    self.say(line)
            # What the sensor found, folded into the field the gate and the
            # imagined rollouts read. Everything below that reacts to a crate
            # on no map reacts because of this one call.
            self.step_crowd()
            live = self.clearance(floor)
            if getattr(self.robot, "map_fix", None) is not None:
                self.robot.map_fix.use(live)     # the board's pose onto this floor's map
            scan = self.robot.get_scan()
            scan_xyz = np.asarray(scan).reshape(-1, 3)
            blind = blind_ahead(scan_xyz, (x, y, yaw))
            fresh = False             # the floor guard found a new puddle this tick
            if self.fg_on:
                # The floor guard's answers, kept off as flat ground: see
                # floor_guard.returns and LiveClearance.update(flat=).
                for p in self.fg.tick(n, floor, self.robot.camera_pose(), (x, y), live._static):
                    print(f"  floor guard: puddle at ({p.x:.1f}, {p.y:.1f}), "
                          f"{self.fg.known_at[p.id]:.2f} m off its edge")
                    self.say("Wet floor ahead.")
                    fresh = True
            if self.cloud_log and n % CLOUD_EVERY == 0:
                # meta: t, floor, x, y, yaw, floor z -- see Run.close
                self.clouds.append(((self.robot.now(), floor, x, y, yaw, self.robot.z),
                                    np.asarray(scan, np.float32).reshape(-1, 3)))
            flat = (self.fg.returns(floor, self.robot.z, (x, y), perception.WINDOW)[:, :2]
                    if self.fg_on and self.fg.known else None)
            live.update(scan, self.robot.z, (x, y), origin=(x, y, self.robot.z + MOUNT_H),
                        flat=flat)
            if self.elev is not None:
                self.elev.add(scan, (x, y), self.robot.z)
            self.seen = live.points
            self.obs["seen"] = max(self.obs["seen"], len(live.points))
            # Same returns, asked a different question: which of them moved
            # since last tick. A crate answers "none of me".
            tracks = self.tracker.update(live.points)
            walking = self.tracker.movers()
            # Somebody slower ahead in the lane: followed, or overtaken on the
            # left (see FOLLOW_GAP). Asked before anything plans round them:
            # somebody walking 0.4-0.5 m/s is under MOVING_V most ticks, and
            # free_destination stepped round them as round a crate, at DETOUR
            # pace, instead (HIL 11-10 16:11, --walker 0.5).
            # Judged on the pace the dog really walks (Pace); the cap it
            # returns is in those terms, so it is scaled back up to a command.
            cruise = self.robot.MAX_V * self.router.behavior.query_actions(x, y)["speed_modifier"]
            if self.fg_on:
                # The guard watches the floor now, at its own pace: see floor_guard.
                cruise = min(cruise, self.fg.cap())
            elif self.vamos:
                cruise = min(cruise, self.vlm_speed_cap())   # see VLM_REACH_D
            pace = self.pace.ratio
            follow_cap, pass_shift, lead = self.follow_lead(tracks, (x, y), yaw, waypoints, i,
                                                            gates, cruise * pace,
                                                            can_pass=self.pace.known)
            if follow_cap is not None:
                follow_cap /= pace
            if self.track_log is not None:
                # What the tracker believed and where everybody really was,
                # every tick: scripts/ce_rrt_calibrate.py measures the
                # planner's chance margin from it. Ground truth -- for the
                # log only, never read back.
                self.track_log.write(json.dumps({
                    "t": round(self.robot.now(), 3), "floor": floor,
                    "tracks": [(t.x, t.y, t.vx, t.vy) for t in tracks if t.moving],
                    # Every track, the slow ones too: (x, y, vx, vy, r, moving,
                    # walked) -- whether a person slowed below MOVING_V is
                    # still taken out of the planning field (live.exclude).
                    "all": [(round(t.x, 3), round(t.y, 3), round(t.vx, 3), round(t.vy, 3),
                             round(t.r, 3), t.moving, t.walked) for t in tracks],
                    "pose": (round(x, 3), round(y, 3), round(yaw, 3)),
                    "walkers": self.crowds[floor].positions()}) + "\n")
            # Somebody standing still ahead who was seen walking is a person,
            # not a crate, whether or not the dog waited for them first: go
            # round with a person's room (PASS_NEED), at PASS_V. Only the ones
            # it had waited out got that, and room 201 seed 1 --vamos went
            # past a person stopped by the wall at a crate's 0.44 m. Only the
            # ones near its line (PASS_SIDE): slowing for everybody standing
            # in the other lane cost room 201 seeds 1 and 8 25 s each.
            if self.passing is None:
                c, s_ = math.cos(yaw), math.sin(yaw)
                for t in tracks:
                    fwd = (t.x - x) * c + (t.y - y) * s_
                    side = abs(-(t.x - x) * s_ + (t.y - y) * c)
                    if (t.walked and not t.moving and 0 < fwd < PASS_R
                            and side < PASS_SIDE and t is not lead):
                        self.passing = (t.x, t.y)
                        break
            # Out of the planning field, still in the safety one. A person is
            # waited for; only a crate is gone round. See LiveClearance.exclude.
            live.exclude([(t.x, t.y, t.r) for t in walking]
                         + ([(lead.x, lead.y, lead.r)] if lead is not None
                            and lead not in walking else []))
            live.remember([(t.x, t.y, t.r) for t in tracks if t.walked],
                          self.robot.now(), (x, y))
            self.movers = np.array([(t.x, t.y) for t in walking]).reshape(-1, 2)
            self.score_collision(floor, x, y)
            if self.fg is not None:
                self.fg.score(floor, (x, y))
            self.score_handler(floor)
            self.score_people(floor, x, y)

            # Picked from the body, projected from the lens -- see
            # project_route. The probe below already measures from the body.
            state = project_route(self.robot.camera_pose(), waypoints[i:], self.cam,
                                  pick_from=(x, y))

            # "Map decides WHERE": the same destination, moved sideways when it or
            # the line to it is blocked by something the map never had. VAMOS
            # drives at whatever goal pixel it is given, so a destination inside a
            # crate is five candidate paths into the crate -- measured, before
            # this existed. Move the goal and the model has something to solve.
            aim, offset, blocked = None, 0.0, False
            finfo, narrow_hold, narrow_go = {}, False, False   # see NARROW_ROOM
            narrow_slow = False                                # see ALIGN_GATE
            no_way = False            # the map has a route destination and no clear aim
            # Whenever the route has a destination -- not only when the camera
            # can see it. This used to read `if state["state"] == "TRACK"`, and
            # that is a camera test standing in for a safety one: ALIGN means
            # the goal pixel is outside the image, which happens exactly when
            # the dog is turning into a doorway, and it says nothing whatever
            # about what is in front of it. The LiDAR is a 360-degree sensor.
            # With avoidance gated on the projection, the dog steered at the
            # raw A* waypoint for the whole turn and drove straight through
            # anything the map did not know about: `chemistry lab` spent 0.4 s
            # inside the floor-3 cartons, every tick of it in ALIGN, on a route
            # whose own clearance figures looked fine either side of the turn.
            tracking = state["state"] == "TRACK"
            if state.get("destination") is not None:
                # Judge the blockage further out than the destination -- see
                # free_destination. PROBE_D is far enough to start moving across
                # while there is still open corridor to do it in.
                #
                # But not around a corner. Arc length along the route runs on
                # past turns, and 6 m along a leg that turns into a doorway is
                # a point the straight line reaches only by crossing a wall --
                # so the tightest spot on that line is the wall, and the dog
                # concludes the building is in its way. Take the furthest probe
                # still in open floor on a straight line, and settle for a
                # nearer one at a turn.
                probe = None
                for reach in (PROBE_D, 4.5, 3.0):
                    q = pick_destination((x, y), waypoints[i:], reach, reach + 2.0)
                    if q is not None and line_clear((x, y), q, live.static_at, ROBOT_R):
                        probe = q
                        break
                # And not across a corner either. The destination is 2-4 m along
                # the route, so near a turn it is round the corner, and the
                # straight line to it cuts across whatever the route goes past.
                # `chemistry lab` turns into a south door 1.2 m after the
                # cartons: the line from the lane to the far side of the turn
                # crossed them, avoidance sent the dog round their east end,
                # and it pivoted beside them with its body in them. If the
                # route itself -- the dog to the corner, the corner to the
                # destination -- is clear of everything the LiDAR has found,
                # there is nothing to go round: follow it.
                #
                # Round a person who has stopped, the room is a person's, not a
                # crate's. Where there is not that much, the crate's will do --
                # slowly, below -- rather than no way past at all.
                need = PASS_NEED if self.passing else perception.DESTINATION_CLEAR
                dest, corner = state["destination"], waypoints[i]
                around = (math.dist((x, y), corner) + math.dist(corner, dest)
                          > math.dist((x, y), dest) + 0.1)
                # "Clear of everything" means a comfortable pass, not merely an
                # allowed one: at `need` alone the floor-1 trolley released the
                # dog level with its west face, 0.38 m off, to steer at a corner
                # 20 m away on a line back towards it. Tighter than PASS_ROOM,
                # free_destination decides -- it centres the pass when the gap
                # has room and otherwise says straight on, as this did.
                bar = max(need, perception.PASS_ROOM)
                # Nor through a doorway. Lined up for one on the route, with the
                # way through it clear, what the LiDAR found is in the room
                # beyond: there is nothing in this corridor to go round, and a
                # detour would aim through the wall beside the door (the funnel
                # gives way to an offset). Racks 3-5 m into the floor-3 server
                # room had the dog told "stepping around it to your left",
                # "...right", "no clear way past" in turn, 5 times in 6 s at its
                # door (HIL 11-09 14:55). Past the doorway, free_destination
                # judges the room as it would any other.
                through = self.door_through(gates, (x, y))
                if through is not None and line_clear((x, y), through, live.detected_at, bar):
                    aim, offset = dest, 0.0
                elif (around and line_clear((x, y), corner, live.detected_at, bar)
                        and line_clear(corner, dest, live.detected_at, bar, skip=0.0)):
                    aim, offset = dest, 0.0
                else:
                    aim, offset = free_destination(dest, (x, y), live, probe=probe,
                                                   prefer=lean, need=need, info=finfo)
                    if aim is None and self.passing:
                        aim, offset = free_destination(dest, (x, y), live,
                                                       probe=probe, prefer=lean, info=finfo)
                # Which way round it went, kept until the thing is out of
                # sight: a detour that changes its mind halfway is a swerve,
                # and the announcement below has already told the person which
                # way they are about to be led. Nor on one tick's say-so: see
                # SIDE_FLIP_TICKS.
                aim, offset, lean, held, against = hold_side(lean, aim, offset,
                                                             held, against)
                # A narrow way: wait a while if something person-sized is in
                # the wider one, uncommitted, so it is taken if they move;
                # then through, lined up. See NARROW_ROOM.
                if self.robot.now() - self.narrow_at > 3.0:
                    self.narrow_waited = 0            # a new narrow way gets its own wait
                if (aim is not None and offset and finfo.get("room", INF) < NARROW_ROOM
                        and math.copysign(1.0, offset) == finfo["side"]
                        and not axis_clear(finfo, live.detected_at)):
                    # Narrow, and shut further along: not a way. The other side,
                    # if it has one; otherwise nothing, and the usual stopping.
                    other = {}
                    aim, offset = free_destination(dest, (x, y), live, probe=probe,
                                                   prefer=-finfo["side"], need=need, info=other)
                    if aim is not None and offset and other.get("side") != finfo["side"]:
                        finfo = other
                        lean, held, against = math.copysign(1.0, offset), (aim, offset), 0
                    else:
                        aim, offset, finfo = None, 0.0, {}
                if (aim is not None and offset and finfo.get("room", INF) < NARROW_ROOM
                        and math.copysign(1.0, offset) == finfo["side"]):
                    self.narrow_at = self.robot.now()
                    blocker = small_thing_beside(finfo, tracks, live.static_at)
                    if blocker is not None and self.narrow_waited < NARROW_WAIT_S * FPS:
                        if self.narrow_waited == 0:
                            other = "left" if finfo["side"] < 0 else "right"
                            self.say(f"Something standing in the wider way, on your {other}. "
                                     "Waiting a moment.")
                        self.narrow_waited += 1
                        self.narrow_side = finfo["side"]
                        narrow_hold = True
                        lean, held, against = 0, None, 0
                    else:
                        aim = narrow_aim(finfo, (x, y), live.detected_at)
                        narrow_go = True
                        narrow_slow = not narrow_lined_up(finfo, (x, y), yaw, live.detected_at)
                        self.narrow_last = (x, y)
                # One tick with no clear goal is noise -- the scan is rebuilt
                # from scratch every tick and a single ray landing awkwardly
                # should not start the stopping sequence.
                no_goal = no_goal + 1 if aim is None else 0
                no_way = aim is None
                blocked = no_goal >= BLOCKED_TICKS
                # Re-aiming the VLM is still a TRACK-only affair: a goal pixel
                # is what VAMOS consumes, and during an alignment turn there is
                # not one. Steering by the displaced goal, below, needs no
                # pixel at all.
                if tracking and aim is not None and offset:
                    moved = project_to_pixel(aim, self.robot.camera_pose(), self.cam)
                    if moved["state"] == "TRACK":
                        moved["destination"] = aim
                        state = moved

            # Asked back to back (see VLM_REACH_D): as soon as the last answer
            # has landed, from this tick's frame; the loop carries on either
            # way. `asked` is an answer arriving this tick.
            if ((self.vamos or self.shadow) and state["state"] == "TRACK"
                    and self.pending is None and not self.vlm_busy()):
                self.ask_vamos(floor, n, state)
            answer = self.take_answer(floor, n, (x, y, yaw),
                                      state.get("destination") or waypoints[-1])
            asked = answer is not None
            if asked:
                proposed, candidates, safety = answer
            if self.vamos:
                chosen = proposed

            self.frame(state, proposed, candidates, safety if proposed else None)
            if self.autolabel is not None:
                self.autolabel.tick(n, (x, y, yaw), self.robot.camera_pose(), state,
                                    self.robot.get_image)

            if (math.hypot(waypoints[-1][0] - x, waypoints[-1][1] - y) < goal_r
                    or (to_lift and lift.in_doorway((x, y)))):
                for line in (announcements[-1] if announcements else []):
                    self.say(line)
                return True, n

            # VAMOS decides HOW when it has a usable path; the map still
            # decides WHERE, and alignment turns stay with the rules.
            # While a detour is on, the displaced goal steers and VAMOS keeps
            # proposing and being judged (its candidates and the gate's
            # rejections are in the video). Not a demotion -- a measurement.
            # Zero-shot VAMOS spreads its five candidates about +/-0.25 m over a
            # 2 m path, and getting round a crate in this corridor takes about
            # 0.65 m, so its paths are safe over their own length, pass the
            # gate, and still walk the dog into the thing. The gate cannot
            # reject a path that is fine for 2 m and leads nowhere at 4 m.
            # Spec L4 step 5 -- the LoRA fine-tune -- is what would let the
            # model make this turn itself; until then the map makes it.
            #
            # Only what is left of the path in front of the dog. It is held
            # until the next answer lands, which can now be a couple of metres
            # of walking later, and a path walked to its end is spent, not a
            # reason to turn round for its first point.
            rest = ahead(chosen, (x, y)) if chosen else []
            # An overtake (follow_lead, above) steers OVERTAKE_LAT left of the
            # lane point; VAMOS, which would hold the dog in its lane, waits
            # until it is back.
            # Out of a narrow way, back to the lane before VAMOS steers again:
            # see NARROW_EXIT_M.
            if self.narrow_last is not None and not leaving_narrow(self.narrow_last, (x, y)):
                self.narrow_last = None
            on_vamos = (len(rest) >= 2 and state["state"] == "TRACK" and not offset
                        and not pass_shift
                        and not leaving_narrow(self.narrow_last, (x, y))
                        and not self.door_near(gates, (x, y), DOOR_MAP_D))
            if on_vamos:
                tx, ty = path_target(rest, (x, y))
                if i > 0:                                # held near the lane: see LANE_TOL
                    tx, ty = in_lane((tx, ty), waypoints[i - 1], waypoints[i])
            elif offset:
                # Nothing VAMOS offered survived the gate, but the goal has been
                # moved clear of what the sensor found, so steer at that rather
                # than at the A* waypoint, which may be inside a crate. Only
                # when it actually moved: with nothing detected this is the
                # ordinary waypoint-following the rest of the run depends on.
                tx, ty = aim
            else:
                tx, ty = lane_point((x, y), waypoints, i)
                if pass_shift:
                    ux, uy = lane_dir((x, y), waypoints, i, yaw)
                    tx, ty = tx - uy * pass_shift, ty + ux * pass_shift
            at_door = False
            if not offset:
                (tx, ty), at_door = self.funnel(gates, (x, y), (tx, ty),
                                                off_route=route_offset((x, y), waypoints, i) > FUNNEL_OFF)
            if asked and self.shadow:
                self.log_shadow(floor, n, (x, y, yaw), (tx, ty), proposed, safety)
            if self.aff_log is not None and n % AFF_EVERY == 0:
                self.shadow_affordance(floor, (x, y, yaw), (tx, ty), live)

            # Reported, not acted on: free_destination already decided whether
            # there is a way through, and this is how much room it left.
            self.obs["min_clear"] = min(self.obs["min_clear"],
                                        self.clear_ahead(live, (x, y), (tx, ty)))
            # Stepping round something is not being stuck: only count it as
            # searching when there is no detour to follow either -- and no
            # clear way on the route itself. This used to crawl whenever VAMOS
            # had nothing and the LiDAR saw anything at all, a crate 8 m off or
            # somebody at the far end of the corridor, while free_destination
            # had already found the route clear and the map-only dog walked it
            # at full pace. Gate D, 21 pairs: 266.7 s of crawling under
            # --vamos against 0.0 s map-only, and the candidate slower on every
            # pair (chemistry lab +33 s). VAMOS offering nothing is not a
            # reason to slow down; the map having no way past is.
            searching = (self.vamos and chosen is None and not offset
                         and len(live.points) > 0 and no_way)
            touching = live.detected_at(x, y) < ROBOT_R

            err = wrap(math.atan2(ty - y, tx - x) - yaw)
            # Zones slow the dog down as well as stopping it: "grass ahead" is
            # a speed modifier, not just a sentence.
            speed = cruise
            # People first, and before every other speed rule below: none of
            # them can see that the thing ahead is walking. free_destination would
            # happily route the dog through the gap behind somebody, and the
            # gap moves.
            # Somebody coming up behind, the same way, is let past: not
            # waited for, not gone round, not stepped away from by stopping.
            # They stay in every prediction (plans, step aside, the handle).
            behind = [t for t in tracks if overtaking((x, y), yaw, t, speed)]
            self.mention_overtakers(behind, (x, y), yaw, speed * pace)
            # ...and the one being followed or overtaken is not waited for or
            # gone round either: follow() has them.
            led = behind + ([lead] if lead is not None else [])
            # Just out of the lift: on, clear of the door, together. Only
            # somebody right in front stops the dog there (see EXIT_CLEAR_M);
            # the rest are still in every plan and every step aside.
            if self.exit_from == floor and math.dist((x, y), lift.WAIT_XY) >= EXIT_CLEAR_M:
                self.exit_from = None
            if self.exit_from == floor:
                led = led + [t for t in tracks
                             if t not in led and not right_in_front((x, y), yaw, t)]
            round_to, close =(self.go_round(live, tracks, (x, y, yaw), waypoints[i:],
                                             gates, n, speed, ignore=led)
                               if self.planner == "cerrt" else (None, None))
            if round_to is None and close is not None:
                # Somebody within NEAR_R walking into the dog's way, and no
                # way round them: stand still, as for any other wait -- and
                # the same wait, latched and asked for (wait_for).
                person = self.wait_for(close, (x, y), yaw)
            elif round_to is not None:
                # Going round them: not waiting, and at the pace the plan
                # was timed at, since where it passes them depends on when.
                tx, ty = round_to
                err = wrap(math.atan2(ty - y, tx - x) - yaw)
                speed = min(speed, self.plan.v)
                person = None
            else:
                person = self.yield_to([t for t in tracks if t not in led],
                                       (x, y), yaw, speed)
                if person is None and self.passer_near:
                    speed = min(speed, PASS_V)           # walking past someone: see PASS_LAT
            if follow_cap is not None:
                speed = min(speed, follow_cap)           # behind someone: see FOLLOW_GAP
            # And so does the safety factor: a path the dog only just believes
            # it can walk is walked at half pace.
            unseen = 0 if len(live.points) else unseen + 1
            if unseen == int(GONE_S * FPS) or fresh:
                # Nothing in sight any more: next time is a new obstacle and
                # deserves to be announced again, and gets its own side. So
                # does a puddle the guard has just found, though something is
                # in sight: with people about, `unseen` never ran out, and the
                # side kept from the last thing gone round sent the dog down
                # the 0.4 m side of a puddle with 0.65 m on the other
                # (--pedestrians 3 --seed 2 --puddles 2, 11-16).
                hunting = halted = False
                stalled = lean = against = 0
                held = None
            # Not the narrow side while waiting before it: a tick the thing in
            # the wider way drops out of the scan said "stepping around it to
            # your right", and the dog then went left when they moved (12:59).
            waiting_narrow = (0 < self.narrow_waited < NARROW_WAIT_S * FPS
                              and offset and math.copysign(1.0, offset) == self.narrow_side)
            if offset and not hunting and not narrow_hold and not waiting_narrow:
                # Said before the dog leans, not during: the person's arm is
                # attached to the handle, and which way it is about to go is
                # the one thing they cannot see coming. Positive offset is the
                # left-hand normal to the direction of travel.
                hunting = True
                self.say("Something ahead. Stepping around it to your "
                         f"{'left' if offset > 0 else 'right'}.")
            if (self.waiting_for is not None or self.narrow_waited > 0 or not hunting
                    or self.plan is not None or blind):
                progress = (x, y, n)         # waiting for somebody is not getting nowhere
            elif n - progress[2] > BLOCKED_S * FPS:
                if self.ask_in_way(n - progress[2] - BLOCKED_S * FPS, n):
                    speed, err = 0.0, 0.0
                else:
                    self.find_another_way(floor, live.points, (x, y, yaw),
                                          f"going round something at ({x:.1f}, {y:.1f}) and "
                                          f"under {PROGRESS_M} m walked in {BLOCKED_S:.0f} s")
                    return False, n
            if touching:
                # Unconditional, and above everything else. This used to live
                # only in the branch below, so a dog on a detour had no
                # proximity stop at all and would drive through the thing it
                # was going round. Nothing that moves the dog gets to skip it.
                speed = 0.0
                stalled += 1
                if not halted:
                    halted = True
                    self.say("Stopping. That is too close.")
                    if os.environ.get("CYBERDOG_TRACE"):
                        peeps = [(round(math.hypot(px - x, py - y), 2), round(px, 1), round(py, 1))
                                 for px, py in self.crowds[floor].positions()]
                        print(f"    [trace] too close at ({x:.2f},{y:.2f}) "
                              f"detected={live.detected_at(x, y):.2f} "
                              f"all={live(x, y):.2f} offset={offset:.2f} aim={aim} "
                              f"people={sorted(peeps)[:2]}")
                # Then get out of it. Standing still is not a recovery from
                # being too close to something -- see BACK_V -- so back off
                # the way the dog came, which is the one direction it has
                # already been. Only if that is still clear: reversing blind
                # into a corridor is how a guide dog trips somebody up.
                behind = (x - BACK_STEP * math.cos(yaw),
                          y - BACK_STEP * math.sin(yaw))
                if (live.detected_at(*behind) > ROBOT_R
                        and live.static_at(*behind) > perception.STATIC_MIN
                        and not blind_ahead(scan_xyz, (x, y, yaw), behind=True)
                        and not something_behind(live.points, (x, y, yaw))):
                    speed, err = -BACK_V, 0.0
                if stalled > BLOCKED_S * FPS and not blind:      # blind: see BLIND_FRAC
                    if self.ask_in_way(stalled - BLOCKED_S * FPS, n):
                        speed, err = 0.0, 0.0        # not backing into the handle either
                    else:
                        self.find_another_way(floor, live.points, (x, y, yaw),
                                              f"something within {ROBOT_R} m of the dog for "
                                              f"{BLOCKED_S:.0f} s at ({x:.1f}, {y:.1f}), and "
                                              f"nothing clear behind to back away into")
                        return False, n
            elif on_vamos:
                speed *= safety
            elif offset:
                speed *= DETOUR          # deliberate, not a swerve
            elif searching or blocked:
                # Something is there and nothing has been approved past it.
                # Crawl while asking again; stop once it is close and the
                # answer has not changed; give up out loud rather than
                # grinding to MAX_TICKS in silence.
                if not hunting:
                    hunting = True
                    self.say("Something ahead. Finding a way round.")
                # No way round what it sees is not yet blocked while that is
                # still far off: it walks on up to it at DETOUR pace, and only
                # stops and asks inside ASK_NEAR. It stood asking a row of
                # people 7.7 m away, a room short of them (HIL, --stand x3,
                # 11-17). Not at CRAWL: 0.12 m/s is under what the gait steps
                # at, and it stood 80 s 2 m short while the board, asked to go
                # and going nowhere, sidestepped 38 times.
                # A puddle the floor guard holds is in the way too, though not
                # in `points`: walking on up to "nothing seen" would be into it.
                seen = (live.points if live._flat is None
                        else np.vstack([live.points.reshape(-1, 2), live._flat]))
                near = nearest_ahead(seen, (x, y, yaw))
                speed *= DETOUR
                self.obs["crawl"] += 1
                # Nothing ahead at all and still no way: whatever it is, it
                # is not something to walk up to. Stop here, as before.
                if touching or (blocked and (near < ASK_NEAR or near == INF)):
                    speed, stalled = 0.0, stalled + 1
                    self.obs["stopped"] += 1
                    if not halted:
                        halted = True
                        self.say("Stopping. I have no clear way past.")
                    if (stalled > BLOCKED_S * FPS and not blind and not
                            self.ask_in_way(stalled - BLOCKED_S * FPS, n)):
                        self.find_another_way(floor, live.points, (x, y, yaw),
                                              f"free_destination found no gap round what the "
                                              f"LiDAR saw for {BLOCKED_S:.0f} s at "
                                              f"({x:.1f}, {y:.1f})"
                                              + (", and no VAMOS path passed" if self.vamos else ""))
                        return False, n
                else:
                    stalled, halted = 0, False
            if self.passing is not None:
                # Walking past somebody, not striding: the person on the handle
                # passes them at the same distance a moment later.
                px, py = self.passing
                if math.dist((x, y), (px, py)) < PASS_R:
                    speed = min(speed, PASS_V)
                elif (px - x) * math.cos(yaw) + (py - y) * math.sin(yaw) < 0:
                    self.passing = None          # behind the dog, and clear
            if person is not None:
                # Waiting is not being stuck, and the give-up timer must not
                # think it is. Without this the dog announces that it cannot
                # find a way past a corridor it is standing in politely.
                stalled = 0
                # Stopped, and not turning either: swinging round to track
                # somebody walking past is what a dog does and is not what a
                # handle attached to a person's arm should do.
                # Creeping is for a queue of people walking past one after
                # another, never at somebody stood still: that is the dog
                # walking at them (see yield_to, ASK_S).
                if (self.waited < YIELD_MAX * FPS or self.waiting_for is None
                        or not self.waiting_for.moving):
                    speed, err = 0.0, self.unpin(live, (x, y, yaw)) / K_W
                else:
                    # They are not going anywhere. Neither can the dog stand
                    # here for ever, so creep and keep asking.
                    if self.waited == int(YIELD_MAX * FPS):
                        self.say("They are not moving. Going slowly.")
                    speed *= CRAWL

            if narrow_hold:
                speed = 0.0                       # before a narrow way: see NARROW_WAIT_S
                # Waiting, so as for a person: no turning, except to stop
                # pressing the user into a wall. Stood here 4.5 s at -23 deg
                # beside the floor-1 trolley with them on the north wall
                # (--pedestrians 3 --seed 2, 11-16).
                if self.handler_room(live, x, y, yaw) < HANDLER_ROOM:
                    err = self.unpin(live, (x, y, yaw)) / K_W
            elif narrow_go:
                # through it, lined up; slowly while not yet (ALIGN_GATE)
                speed = min(speed, ALIGN_SLOW_V if narrow_slow else PASS_V)
            turn_only = DOOR_ERR if at_door else TURN_ONLY
            cmd = (0.0 if abs(err) > turn_only else speed * math.cos(err), 0.0, K_W * err)
            if cmd[0] > 0.0 and self.nose_closing(live, (x, y, yaw), cmd):
                cmd = (0.0, 0.0, cmd[2])
            if speed > 0.0:
                people = [t for t in tracks if t.moving or MIN_RADIUS <= t.r <= MAX_RADIUS]
                cmd = self.spare_handler(live, (x, y, yaw), cmd, speed, people)
            elif cmd[2]:
                # Stopped on purpose (no way past, waiting) and turning on the
                # spot: not into a wall with the user. spare_handler only
                # judged walking commands, and a dog stopped by the floor-1
                # trolley turned to -33 deg after its aim, holding its user on
                # the north wall for 4.5 s (--pedestrians 3 --seed 2, 11-16).
                room = self.handler_room(live, x, y, yaw)
                after = self.handler_room(live, x, y, yaw + cmd[2] * UNPIN_DT)
                if after < HANDLER_ROOM and after < room:
                    cmd = (cmd[0], cmd[1], 0.0)
            if self.planner == "cerrt" and person is not None:
                side = self.step_aside(live, tracks, (x, y, yaw))
                if side:
                    cmd = (0.0, side, 0.0)
            elif self.evading:
                self.evading = 0.0
            # Something inside the minimum range ahead: stand still, and ask --
            # it may well be somebody (see BLIND_FRAC). Above everything but
            # the board and the handle, which only ever stop it.
            if blind:
                cmd = (0.0, 0.0, 0.0)
                blind_n += 1
                if blind_n == 1:
                    self.say("Stopping. Something is right in front of me.")
                if (blind_n - 1) % int(EXCUSE_S * FPS) == 0:
                    self.say(EXCUSE)
                if blind_n > (BLOCKED_S + ASK_S) * FPS:
                    self.find_another_way(floor, live.points, (x, y, yaw),
                                          f"no returns ahead -- something inside the "
                                          f"LiDAR's minimum range -- for {BLOCKED_S:.0f} s "
                                          f"at ({x:.1f}, {y:.1f})")
                    return False, n
            else:
                blind_n = 0
            if (cmd[0] or cmd[1]) and closing_on(tracks, (x, y, yaw), cmd):
                cmd = (0.0, 0.0, cmd[2])          # see KEEP_OFF
                self.ped["kept_off"] = self.ped.get("kept_off", 0) + 1
            # The board's guard is the real sensor's word: what the plan wants
            # it cannot overrule, so it works with it instead (a robot with a
            # guard: go2_board, or the twin modelling it; the plain twin has
            # none to meet).
            if getattr(self.robot, "GUARDED", False):
                cmd = self.respect_board(cmd, (x, y, yaw))
                if self.board_gave_up:
                    return False, n
            # Last: the person on the handle overrules everything above.
            self.handle_tick()
            cmd = self.mux.apply(cmd, self.robot.MAX_V)
            self.commands.update(struct.pack("3d", *cmd))
            self.n_commands += 1
            self.robot.set_velocity(*cmd)
            self.pace.update(self.robot.now(), x, y, yaw, cmd)
            self.robot.step()

            if verbose and n % 60 == 0:
                print(f"    t={self.robot.now():6.1f}s  floor {floor}  "
                      f"pos=({x:5.1f},{y:5.1f})  wp={i}/{len(waypoints) - 1}  "
                      f"{state['state']}")
        self.gave_up = ("control", f"{MAX_TICKS} ticks on floor {floor} without arriving, "
                        f"last at checkpoint {i}/{len(waypoints) - 1}, ({x:.1f}, {y:.1f})")
        return False, MAX_TICKS

    def confirm(self, prompt, what="lift", enter="once the lift is here and the doors are open"):
        """Stop, say something, and wait for a human to act on it.

        The robot cannot press a call button, and pretending otherwise is the
        kind of shortcut that turns into a person standing in front of a lift
        that was never called. So the handover is explicit: it asks, a human
        does the part that needs hands, and the run continues.

        With a handle, continue on the handle is the answer, as it would be on
        the robot, and the dog stands at the doors until it comes: --auto-
        confirm does not answer for a person who has a handle to press. A
        continue that releases a tug only releases the tug; the lift needs a
        press of its own.

        Without one, --auto-confirm (and any non-interactive stdin, which is
        every batch run) answers for them, after a visible pause, so the video
        still shows the stop.

        No answer is a stop (HandleStop): this is for the handovers that have
        no other way on. `ask` is the question that does.
        """
        if not self.ask(prompt, what=what, enter=f"press Enter {enter}... "):
            raise HandleStop(f"at the {what} on floor {self.floor}: nobody "
                             f"pressed continue")

    def ask(self, prompt, what="lift", enter=None):
        """Say `prompt` and wait for yes or no: True on continue.

        On the handle, continue is yes, and a script with no continue left to
        come is no. --auto-confirm and batch runs say yes, after the same
        visible pause as `confirm`. At a terminal, `enter` given is a press of
        Enter (confirm's handovers); otherwise it is a y/n question.
        """
        self.say(prompt)
        if self.has_handle:
            dt = self.robot.CONTROL_DT
            while True:
                t = self.robot.now()
                was_stopped = self.mux.stopped
                self.handle_tick()
                if (self.handle.pressed(t, dt) and not was_stopped
                        and not self.mux.stopped):
                    print(f"    t={t:6.1f}s  HANDLE continue: {what} confirmed")
                    return True
                if not self.handle.will_continue(t):
                    return False
                x, y, yaw = self.robot.get_pose()
                self.robot.place((x, y), yaw)
                self.step_crowd()
                self.frame()
        if self.auto_confirm or not sys.stdin.isatty():
            for _ in range(int(WAIT_S * FPS)):
                self.step_crowd()
                self.frame()
            print("    (auto-confirmed)")
            return True
        if enter is not None:
            input(f"    {enter}")
            return True
        return input("    [y/n] ").strip().lower().startswith("y")

    def glide(self, to_xy, yaw, z):
        """Walk a straight short line with the pose set directly.

        Used only for the few metres in and out of the car, where the velocity
        controller would turn on the spot inside a shaft it cannot see. The
        real Go2 walks this under its own controller; here it is scripted, the
        way the whole kinematic twin is.
        """
        x, y, yaw0 = self.robot.get_pose()
        # Turned over the walk, not snapped on the first frame: the dog reaches
        # the lift on whatever heading the corridor left it with, and setting
        # the target yaw outright made the board a teleported quarter-turn.
        dyaw = wrap(yaw - yaw0)
        ticks = max(int(math.dist((x, y), to_xy) / (STEP_V * self.robot.CONTROL_DT)), 1)
        for k in range(1, ticks + 1):
            self.handle_wait()
            t = k / ticks
            self.robot.place((x + (to_xy[0] - x) * t, y + (to_xy[1] - y) * t),
                             yaw0 + dyaw * t, z=z)
            self.step_crowd()
            self.frame()
        # Then get there. The twin is put on each point and already is; a board
        # walks after the moving point at its own limits, and the line ran out
        # with it 1.2 m and 1.3 rad short of the floor-3 lift's waiting spot --
        # its user pinned against the car and the wall, and the leg on from
        # there never got going. Walk on to the end of the line, for a while.
        # ...but not standing in the lift door for a step the board will not
        # make: out of the floor-3 car for the server room, the last 0.27 m
        # was sideways, the board never took it, and the dog stood in front
        # of the door 3-4 s, its user in the doorway (HIL 11-17). No headway
        # for GLIDE_STUCK_S: it is where it is going to get.
        moved = (self.robot.get_pose(), 0)
        for k in range(int(GLIDE_SETTLE_S / self.robot.CONTROL_DT)):
            x, y, th = self.robot.get_pose()
            if math.dist((x, y), to_xy) <= GLIDE_TOL_M and abs(wrap(yaw - th)) <= GLIDE_TOL_YAW:
                break
            (mx, my, mth), at = moved
            if math.dist((x, y), (mx, my)) > GLIDE_STUCK_M or abs(wrap(th - mth)) > GLIDE_TOL_YAW / 2:
                moved = ((x, y, th), k)
            elif (k - at) * self.robot.CONTROL_DT >= GLIDE_STUCK_S:
                break
            self.handle_wait()
            self.robot.place(to_xy, yaw, z=z)
            self.step_crowd()
            self.frame()

    def pivot(self, from_yaw, to_yaw, z, seconds=1.0):
        """Turn on the spot, scripted, inside the car."""
        d = wrap(to_yaw - from_yaw)
        xy = self.robot.get_pose()[:2]
        for k in range(1, int(seconds / self.robot.CONTROL_DT) + 1):
            self.handle_wait()
            self.robot.place(xy, from_yaw + d * k / (seconds / self.robot.CONTROL_DT), z=z)
            self.step_crowd()
            self.frame()

    def hold(self, seconds):
        """Stand still, still filming."""
        for _ in range(int(seconds / self.robot.CONTROL_DT)):
            self.handle_tick()
            self.robot.place(self.robot.get_pose()[:2], self.robot.get_pose()[2])
            self.step_crowd()
            self.frame()

    def face(self, point):
        """Turn to look at a point on the map, walking the turn."""
        x, y, yaw = self.robot.get_pose()
        self.pivot(yaw, math.atan2(point[1] - y, point[0] - x), self.robot.z, seconds=1.2)

    def lift_ride(self, from_floor, to_floor, onward=None):
        """Change floor the only way this robot will: in the lift.

        The leg in stops at lift.WAIT_XY, outside the open face (see
        lift_approach), so the dog is standing in front of the doors when it
        gets here rather than already in the shaft. This is the bit in between:
        ask, walk in, ride, announce, step back out.
        """
        if to_floor not in lift.FLOORS_SERVED:
            raise HazardStop(f"the lift does not serve floor {to_floor}", layer="planning")

        self.in_lift = True
        z_from = levels.floor_z(from_floor)
        # Asked from outside, and nothing moves until it is answered: a dog
        # that boards and then asks for the button has already committed the
        # person it is leading to a car that may not be there.
        self.confirm(f"Lift ahead. Press the call button for floor {to_floor}, "
                     f"then press continue.")
        # Turn to face the doors, then walk straight in through the open west
        # face. Turn first, walk second: rolling the two together -- gliding to
        # the car while the yaw caught up -- walked the dog in sideways, which
        # is not a thing a dog does and looked exactly as wrong as it was.
        self.face(lift.LIFT_XY)
        self.glide(lift.LIFT_XY, 0.0, z_from)
        self.pivot(self.robot.get_pose()[2], math.pi, z_from)
        # Facing the doors, in the car, waiting to go: that is the shot.

        z0, z1 = z_from, levels.floor_z(to_floor)
        yaw = math.pi                      # facing the doors for the ride
        self.say(f"Doors closing. Going to floor {to_floor}.")
        ticks = max(int(abs(z1 - z0) / (lift.RIDE_V * self.robot.CONTROL_DT)), 1)
        if getattr(self.robot, "REAL_TIME", False):
            # A robot that walks itself (go2_board) is not carried by script:
            # it stands, the real car lifts it, and it watches its own height.
            def ride_tick(z):
                self.robot.move_mocap("lift_car", lift.car_pos(z))
                self.floor = min(max(round(z / levels.FLOOR_HEIGHT) + 1,
                                     lift.FLOORS_SERVED[0]), lift.FLOORS_SERVED[-1])
                self.frame()
            self.robot.ride_lift(to_floor, z1, ride_tick)
            ticks = 0
        for k in range(1, ticks + 1):
            z = z0 + (z1 - z0) * k / ticks
            self.robot.place(lift.LIFT_XY, yaw, z=z)
            self.robot.move_mocap("lift_car", lift.car_pos(z))
            # Round to the nearest storey so the pip in the video flips as the
            # car passes each slab, not only when it arrives.
            self.floor = min(max(round(z / levels.FLOOR_HEIGHT) + 1,
                                 lift.FLOORS_SERVED[0]), lift.FLOORS_SERVED[-1])
            self.frame()

        self.robot.set_height(z1)
        self.floor = to_floor
        self.say(f"Floor {to_floor}. Doors opening.")
        # Out west into the corridor before the velocity loop takes over, so
        # the first thing it does is not a turn into the shaft wall -- and,
        # when the way on is round a corner, turned to it here: see EXIT_TURN_MIN.
        spot, heading = self.exit_spot(to_floor, onward)
        self.exit_wait(to_floor, spot, heading)
        self.glide(spot, math.pi, z1)
        turn = wrap(heading - math.pi)
        if turn:
            self.say(f"Out of the lift. Turning {'left' if turn > 0 else 'right'}.")
            self.pivot(math.pi, heading, z1, seconds=abs(turn) / EXIT_TURN_W)
            self.glide(spot, heading, z1)          # a board lags the turn: let it finish
        self.in_lift = False
        self.exit_from = to_floor

    def exit_wait(self, floor, spot, heading):
        """Doors open: stand in the car, the person on the handle with the
        dog, WAIT_S at least, and on while somebody is about to walk across
        the way out (crossing_exit) -- EXIT_WAIT_MAX_S at most. Watched with
        the LiDAR the whole time, so the tracker has walkers' pace by the time
        the dog steps out. See EXIT_CLEAR_M."""
        x, y, yaw = self.robot.get_pose()
        ux, uy = math.cos(heading), math.sin(heading)
        way_out = [lift.WAIT_XY, spot] + [(spot[0] + d * ux, spot[1] + d * uy)
                                         for d in (0.6, 1.2, EXIT_CLEAR_M)]
        live = self.clearance(floor)
        said = False
        for n in range(int(EXIT_WAIT_MAX_S * FPS)):
            self.handle_wait()
            self.step_crowd()
            live.update(self.robot.get_scan(), self.robot.z, (x, y),
                        origin=(x, y, self.robot.z + MOUNT_H))
            tracks = self.tracker.update(live.points)
            self.frame()
            who = next((t for t in tracks if crossing_exit(t, way_out)), None)
            if who is None:
                if n >= WAIT_S * FPS:
                    break
                continue
            if not said:
                self.say(describe_person((x, y), yaw, (who.x, who.y),
                                         (who.vx, who.vy)) + ". Waiting.")
                self.ped["yields"] += 1
                said = True
            self.ped["waited"] += 1

    def exit_spot(self, floor, onward):
        """(where to stand outside the car, the heading to turn to there).

        The way on is the leg out's first target, through the funnel of a door
        if it starts at one. Nearly straight on (EXIT_TURN_MIN): lift.WAIT_XY,
        still facing out, and the leg turns as it walks. Round a corner: a spot
        on the line from WAIT_XY towards it (EXIT_SHIFT), the nearest from
        which the person on the handle keeps HANDLER_ROOM through the whole
        turn, or else the roomiest."""
        x0, y0 = lift.WAIT_XY
        if not onward or len(onward) < 2:
            return lift.WAIT_XY, math.pi
        goal = tuple(onward[1])
        tx, ty = self.funnel(self.router.gates(floor, onward), (x0, y0), goal)[0]
        heading = math.atan2(ty - y0, tx - x0)
        turn = wrap(heading - math.pi)
        if abs(turn) < EXIT_TURN_MIN:
            return lift.WAIT_XY, math.pi
        live = self.clearance(floor)
        ux, uy = math.cos(heading), math.sin(heading)
        best = None
        for d in EXIT_SHIFT:
            x, y = x0 + d * ux, y0 + d * uy
            if live(x, y) < ROBOT_R:
                break
            room = min(self.handler_room(live, x, y, math.pi + turn * k / 8) for k in range(9))
            if room >= HANDLER_ROOM:
                return (x, y), heading
            if best is None or room > best[0]:
                best = (room, (x, y))
        return (best[1] if best else lift.WAIT_XY), heading

    @staticmethod
    def flights(from_floor, to_floor):
        """The flights between two floors, each named by the floor at its foot."""
        if to_floor > from_floor:
            return list(range(from_floor, to_floor))
        return list(range(from_floor - 1, to_floor - 1, -1))

    @staticmethod
    def rail_side(flight, up):
        """Which hand the wall rail is on, walking flight `flight` up or down.

        Each flight runs west to east (stairs.steps) in a lane against one of
        the corridor's long walls; the rail is on that wall."""
        wall_south = stairs.lane(flight) == "south"
        # Going up is walking east, where south is on the right.
        return "right" if wall_south == up else "left"

    def offer_stairs(self, transit, from_floor, to_floor):
        """Ask #1: may this floor change use the stairs? True for yes.

        Said before the dog sets off, with what the person needs to decide:
        why not the lift, how many steps, which way, where the rail is. When
        the lift works, saying no is taking it; when it does not, saying no is
        the end of the trip (plan_stops).
        """
        up = to_floor > from_floor
        flights = self.flights(from_floor, to_floor)
        steps = stairs.N_STEPS * len(flights)
        no = ("Wait, and I will take the lift" if transit.lift_m is not None
              else "Wait, and I will stop here")
        return self.ask(
            f"{transit.reason} I can take you to floor {to_floor} by the stairs: "
            f"{steps} steps {'up' if up else 'down'}, handrail on your "
            f"{self.rail_side(flights[0], up)}. Press continue to take the stairs. "
            f"{no}.", what="stairs offer")

    def stairs_ride(self, from_floor, to_floor):
        """Change floor by the stairs -- only ever after offer_stairs said yes.

        Ask #2 comes here, at the stairs and not before: stopped short of the
        first step, the person is told which way, how many steps and which
        hand the rail is on, and nothing moves until continue. Then slowly --
        STAIR_V, a third of corridor pace -- one step at a time, with every
        landing and the last step said as they come.

        Scripted like the lift ride, and a sketch of the motion rather than a
        simulation of it: the twin's dog has no legs (it glides), and the
        stairwell was built as scenery, short and steep with no landings, so
        the dog passes through the edges of the step boxes. What this tests is
        the decision, the questions and what is said; the climb itself
        belongs in Isaac Lab, where the dog has legs.
        """
        if getattr(self.robot, "REAL_TIME", False):
            raise HazardStop("the board does not climb stairs (go2_flat_v5 walks the flat): "
                             "lift only", layer="go2_board")
        up = to_floor > from_floor
        flights = self.flights(from_floor, to_floor)
        steps = stairs.N_STEPS * len(flights)
        self.confirm(
            f"Stairs {'up' if up else 'down'} to floor {to_floor}, {steps} steps, "
            f"handrail on your {self.rail_side(flights[0], up)}. Press continue "
            f"when you are ready.", what="stairs", enter="when you are ready ")
        self.on_stairs = True
        rise = levels.FLOOR_HEIGHT / stairs.N_STEPS
        run = (stairs.SHAFT_X1 - stairs.SHAFT_X0) / stairs.N_STEPS
        west, east = stairs.SHAFT_X0 - 0.15, stairs.SHAFT_X1 + 0.15
        mid = sum(stairs.LANES["south"][1:] + stairs.LANES["north"][:1]) / 2
        for j, f in enumerate(flights):
            y0, y1 = stairs.LANES[stairs.lane(f)]
            ly = (y0 + y1) / 2
            zf = levels.floor_z(f)
            # Each flight's foot is its west end on floor f, its head the east
            # end on floor f+1. Walk to whichever end comes first along the
            # strip between the two lanes, then into the lane.
            start, end = ((west, zf), (east, zf + levels.FLOOR_HEIGHT)) if up \
                else ((east, zf + levels.FLOOR_HEIGHT), (west, zf))
            heading = 0.0 if up else math.pi
            x, _, yaw = self.robot.get_pose()
            along = yaw if abs(start[0] - x) < 1e-3 else (math.pi if start[0] < x else 0.0)
            self.glide((start[0], mid), along, start[1])
            self.glide((start[0], ly), heading, start[1])
            # Step by step: the body rises (or drops) one riser per tread.
            order = range(stairs.N_STEPS) if up else range(stairs.N_STEPS - 1, -1, -1)
            for i in order:
                x0 = stairs.SHAFT_X0 + i * run
                x_from, x_to = (x0, x0 + run) if up else (x0 + run, x0)
                z = zf + (i + 1) * rise if up else zf + i * rise
                ticks = max(int(run / (STAIR_V * self.robot.CONTROL_DT)), 1)
                for k in range(1, ticks + 1):
                    self.handle_wait()
                    self.robot.place((x_from + (x_to - x_from) * k / ticks, ly), heading, z=z)
                    self.step_crowd()
                    self.frame()
                self.stair_steps += 1
            self.glide((end[0], ly), heading, end[1])
            self.floor = f + 1 if up else f
            self.robot.set_height(levels.floor_z(self.floor))
            if j + 1 < len(flights):
                self.say(f"Landing. Floor {self.floor}. "
                         f"{stairs.N_STEPS} more steps {'up' if up else 'down'}.")
                self.hold(WAIT_S)
        self.say(f"Last step. Floor {to_floor}.")
        self.glide((end[0], mid), heading, levels.floor_z(to_floor))
        self.glide(self.router.stair_stop(to_floor), 0.0, levels.floor_z(to_floor))
        self.on_stairs = False
        self.stair_flights += len(flights)


# Seconds a VAMOS answer takes to come back on the dog: asked to answered on
# the J4012, VAMOS on its GPU (HIL 11-11: 2.7, 3.2, 2.9 s mean per route). The
# twin's default, in sim time. Not the call's own wall-clock time: from the Mac
# that goes through the tunnel and the USB link, ~4.7 s, and in that the dog
# walked 3.3 m -- past where every path VAMOS drew (1.2-4 m ahead) began, so
# 19 of 25 were rejected as already behind it (cafeteria, twin + VAMOS).
VLM_LATENCY = 2.8


def vlm_latency_arg(text):
    """--vlm-latency: seconds, or 'measured' (None: the call's wall-clock time)."""
    return None if text == "measured" else float(text)


def main():
    ap = argparse.ArgumentParser(description="CyberDog whole-building simulation")
    ap.add_argument("command", nargs="?", default="take me to the server room",
                    help="natural-language command; multi-stop is split by rules")
    ap.add_argument("--out", default=str(paths.SCENE_CACHE / "building.mp4"))
    ap.add_argument("--no-video", action="store_true", help="drive without rendering")
    ap.add_argument("--nlu", action="store_true",
                    help="also ask the Gemma parser, for commands no rule reads (Mac only: "
                         "needs torch; not on the J4012, where it does not fit next to VAMOS)")
    ap.add_argument("--vamos", action=argparse.BooleanOptionalAction, default=None,
                    help="VLM in the steering loop (the default: the real robot "
                         "steers with VAMOS); --no-vamos drives the map route alone")
    ap.add_argument("--shadow", action="store_true",
                    help="ask VAMOS and score its paths with the dream, but let the map "
                         "route drive; every call is logged to --shadow-log")
    ap.add_argument("--shadow-log", default=str(paths.OUTPUT_DIR / "shadow_log.jsonl"),
                    metavar="PATH", help="where --shadow writes one JSON line per VAMOS call")
    ap.add_argument("--vamos-url", default=None, metavar="URL",
                    help="VAMOS server (default http://127.0.0.1:8009)")
    ap.add_argument("--vlm-latency", type=vlm_latency_arg, default=VLM_LATENCY, metavar="S",
                    help="with --vamos or --shadow: seconds of sim time before a VAMOS "
                         "answer arrives; the control loop runs on meanwhile (default "
                         f"{VLM_LATENCY}: the J4012's, see VLM_LATENCY; 'measured': the "
                         "call's own wall-clock time)")
    ap.add_argument("--vamos-sample", action="store_true",
                    help="with --vamos or --shadow: let VAMOS sample its paths "
                         "(temperature 1.0, different every run) instead of beam "
                         "search, which gives the same paths for the same frame")
    ap.add_argument("--lift-out-of-service", action="store_true",
                    help="the lift cannot be used: floor changes go by the stairs, "
                         "if the person agrees (or not at all with --no-stairs)")
    ap.add_argument("--no-stairs", action="store_true",
                    help="never offer the stairs: the lift or nothing (spec rule 0 "
                         "as it was before the stairs fallback)")
    ap.add_argument("--auto-confirm", action="store_true",
                    help="answer the lift handover prompt instead of waiting for a human")
    ap.add_argument("--pedestrians", type=int, default=0, metavar="N",
                    help=f"people walking the corridors on each floor (0-{pedestrians.POOL}); "
                         "they are on no map, and the dog stops for them")
    ap.add_argument("--seed", type=int, default=0,
                    help="which crowd -- the same seed is the same people every run")
    ap.add_argument("--puddles", type=int, default=0, metavar="N",
                    help="N puddles on each leg's route (on no map; the LiDAR does not "
                         "see them); scored as 'stepped in'. Placed by --seed")
    ap.add_argument("--puddle-at", action="append", default=[], metavar="X,Y,R[,FLOOR]",
                    help="a puddle exactly here (floor 1 unless given); may repeat. "
                         "Not checked for a way round: for the blocked-way cases")
    ap.add_argument("--floor-guard", choices=["off", "sim"], default="off",
                    help="sim: a simulated fast floor check (sensing/floor_guard.py) sees "
                         "puddles in the camera's view; the dog goes round what it knows, "
                         "and the pace is capped by the guard, not by VAMOS")
    ap.add_argument("--fg-latency", type=float, default=fguard.LATENCY, metavar="S",
                    help=f"floor guard: seconds from a frame to its answer ({fguard.LATENCY})")
    ap.add_argument("--fg-see", type=float, default=fguard.SEE_D, metavar="M",
                    help=f"floor guard: metres from the lens a puddle is seen ({fguard.SEE_D})")
    ap.add_argument("--fg-miss", type=float, default=0.0, metavar="P",
                    help="floor guard: chance a puddle in view is missed, per frame (0)")
    # The defaults are the PoC configuration, the one Gate D and the planner's
    # gate passed on (docs/dreaming_safety_kpis.md, 2026-10-14).
    ap.add_argument("--crowd", choices=["oblivious", "aware"], default="aware",
                    help="how the people behave: aware (the default: they make room "
                         "for the dog and its user, as people do) or oblivious (the "
                         "stress test: they only stop for a dog right in front of "
                         "them); scene/pedestrians.py")
    ap.add_argument("--planner", choices=["map", "cerrt"], default="cerrt",
                    help="what to do about a person in the way: cerrt (the default: "
                         "go round them on a path planned where they will be, waiting "
                         "only when no path is safe; planning/ce_rrt.py) or map (wait "
                         "for them)")
    ap.add_argument("--cloud-log", default=None, metavar="PATH",
                    help="save every %d-th scan (map frame) and the dog's pose to "
                         "an .npz when the run ends" % CLOUD_EVERY)
    ap.add_argument("--track-log", default=None, metavar="PATH",
                    help="write the tracker's moving tracks and every walker's true "
                         "position, each tick, as JSON lines "
                         "(scripts/ce_rrt_calibrate.py)")
    ap.add_argument("--handle", default="", metavar="EVENTS",
                    help='scripted Smart Handle, at seconds of sim time: "tug@30,continue@35", '
                         '"pull@20-25", "push@26-28" (see sim/handle.py). Its continue '
                         'also answers the lift prompt, in place of --auto-confirm')
    ap.add_argument("--affordance-shadow", nargs="?", metavar="PATH", default=None,
                    const=str(paths.OUTPUT_DIR / "affordance_shadow.jsonl"),
                    help="run the affordance network watching, not steering: log what it "
                         "would say about the walk ahead, against what was really there")
    ap.add_argument("--vamos-affordance", action="store_true",
                    help="with --vamos or --shadow: the affordance network judges each "
                         "VAMOS candidate from the LiDAR elevation patch and rejects what "
                         "it calls not walkable, before the safety gate (spec L5 s4)")
    ap.add_argument("--vamos-steer", default=None, metavar="TEXT",
                    help="with --vamos or --shadow: a preference in plain language added "
                         "after the goal in every VAMOS prompt, e.g. \"Keep to the left "
                         "wall.\" (spec L4 s3: image + goal pixel + preference)")
    ap.add_argument("--autolabel", default=None, metavar="DIR",
                    help="keep onboard frames labelled with the path the dog then walked, "
                         "for the VAMOS LoRA (sim/autolabel.py); map-only, not with --vamos")
    ap.add_argument("--robot", choices=["twin", "twin-board", "go2_board"], default="twin",
                    help="who walks: the kinematic twin; the twin turning late and short "
                         "and stopped by the board's guard, as go2_board walks the dog "
                         "(robot/board_model_robot.py); or go2_board over its link "
                         "(go2_world's physics or the dog; floor 1 only)")
    ap.add_argument("--start", default=None, metavar="X,Y[,YAW]",
                    help="twin: start here on floor 1 (map frame; yaw rad) instead of the "
                         "main entrance -- e.g. 20.92,9.5,0 where HIL routes start "
                         "(go2_world's spawn)")
    ap.add_argument("--board-clock", type=float, default=1.0, metavar="F",
                    help="twin-board: walk F of real time (0.83: as the HIL rig's world "
                         "does, to compare with HIL logs; default 1.0, the policy alone)")
    ap.add_argument("--board-host", default=None, metavar="IP",
                    help="with --robot go2_board: the board's address when it is another "
                         "machine (HIL over USB-C: 192.168.55.1); default localhost")
    ap.add_argument("--vamos-samples", type=int, default=5, metavar="N",
                    help="candidate paths per VAMOS call (beams). 5 is the PoC's; on the "
                         "J4012 at 25 W a call takes 4.6 s for 5, 3.0 s for 2, 2.3 s for 1 "
                         "(spec: 2-3 s a plan)")
    ap.add_argument("--twin-lidar", action="store_true",
                    help="with --robot go2_board: perceive with the twin's raycast, not the "
                         "board's Mid-360 cloud (sees no real person; for comparison only)")
    ap.add_argument("--no-map-fix", action="store_true",
                    help="with --robot go2_board: use the board's pose as it is, without "
                         "aligning the scans' walls to the map (sensing/map_fix.py)")
    ap.add_argument("--lift-host", default="127.0.0.1", metavar="IP",
                    help="with --robot go2_board: where go2_world's lift takes calls (udp 9880)")
    ap.add_argument("--brain-listen", default=None, metavar="IP",
                    help="this machine's address on the link to the board (192.168.55.100)")
    ap.add_argument("--speed", type=int, default=1, metavar="N",
                    help="play the video back N times faster (control still runs at 20 Hz)")
    args = ap.parse_args()
    if args.vamos and args.shadow:
        ap.error("--vamos steers with VAMOS and --shadow never does; pick one")
    if args.vamos and args.autolabel:
        # The label is the path the map-only stack walked. Under --vamos it
        # would be VAMOS's own answer, and the model would be trained on itself.
        ap.error("--autolabel records map-only runs; drop --vamos")
    # On unless asked otherwise: --shadow never lets VAMOS steer, and
    # --autolabel records the map-only walk.
    if args.vamos is None:
        args.vamos = not (args.shadow or args.autolabel)
    if args.vamos_affordance and not (args.vamos or args.shadow):
        ap.error("--vamos-affordance judges VAMOS's candidates; it needs --vamos or --shadow")
    if args.vamos_steer is not None and not (args.vamos or args.shadow):
        ap.error("--vamos-steer goes into the VAMOS prompt; it needs --vamos or --shadow")
    try:
        handle = HandleScript(args.handle)
    except ValueError as bad:
        ap.error(str(bad))

    scene = str(paths.BUILDING_SCENE)
    if not os.path.exists(scene):
        raise SystemExit("no building scene yet -- run: python -m cyberdog.sim.scene.build_scene --building")

    # The board cannot climb (BoardTwinRobot.place refuses a scripted climb):
    # a way found round a blocked corridor (ASK_S) must not be the stairs.
    router = BuildingRouter(allow_stairs=not args.no_stairs and args.robot != "go2_board",
                            lift_in_service=not args.lift_out_of_service)
    # A refusal -- stairs as a destination, or a floor the lift misses -- is an
    # answer, not a crash. It is the sentence the user would hear.
    layer, guessed = "language", []
    try:
        stops = resolve_stops(router, args.command, use_nlu=args.nlu, guessed=guessed)
        print(f'command: "{args.command}"')
        for k, (name, floor) in enumerate(stops, 1):
            print(f"  stop {k}: {name}" + (f" (floor {floor})" if floor else ""))

        layer = "planning"
        robot = None
        if args.robot == "go2_board":
            from cyberdog.sim.robot.go2_board_robot import BoardTwinRobot
            from cyberdog.sim.robot.go2_board_robot import LiftLink
            robot = BoardTwinRobot(scene, origin=GO2_WORLD_ORIGIN,
                                   board_host=args.board_host, listen=args.brain_listen,
                                   lift=LiftLink(args.lift_host), clouds=not args.twin_lidar)
            if not args.no_map_fix and not args.twin_lidar:
                from cyberdog.sim.sensing.map_fix import MapFix
                robot.map_fix = MapFix()                # see sensing/map_fix.py
            start_xy, legs, faces = plan_stops(router, stops,
                                               start=(START_FLOOR, robot.get_pose()[:2]))
        else:
            start = None
            if args.start:
                start = (START_FLOOR, tuple(float(v) for v in args.start.split(",")[:2]))
            start_xy, legs, faces = plan_stops(router, stops, start=start)
    except NoAccessibleRoute as refusal:
        raise SystemExit(f"[voice] {refusal}\nREFUSED BY planning: no accessible route")
    except SystemExit as refusal:
        raise SystemExit(f"REFUSED BY {layer}: {refusal}")

    print(f"{len(legs)} leg(s) across floors "
          f"{sorted({leg[0] for leg in legs})}, starting at {START_LOCATION}")

    first = legs[0][2]
    yaw0 = math.atan2(first[1][1] - first[0][1], first[1][0] - first[0][0]) \
        if len(first) > 1 else 0.0
    if args.start and len(args.start.split(",")) > 2:
        yaw0 = float(args.start.split(",")[2])
    if args.robot == "twin-board":
        from cyberdog.sim.robot.board_model_robot import BoardModelRobot
        robot = BoardModelRobot(scene, start_xy=start_xy, start_yaw=yaw0,
                                start_z=levels.floor_z(START_FLOOR), clock=args.board_clock)
    run = Run(scene, start_xy, yaw0, router, out=None if args.no_video else args.out,
              vamos=args.vamos, auto_confirm=args.auto_confirm, speed=args.speed,
              crowd=args.pedestrians, seed=args.seed, aware=args.crowd == "aware",
              planner=args.planner, track_log=args.track_log, cloud_log=args.cloud_log,
              shadow=args.shadow_log if args.shadow else None, vamos_url=args.vamos_url,
              vlm_latency=args.vlm_latency, vamos_sample=args.vamos_sample,
              vamos_samples=args.vamos_samples, vamos_affordance=args.vamos_affordance,
              vamos_steer=args.vamos_steer,
              handle=handle if args.handle else None,
              affordance_shadow=args.affordance_shadow, autolabel=args.autolabel,
              robot=robot, puddles=args.puddles, puddle_at=args.puddle_at,
              floor_guard=args.floor_guard,
              fg_latency=args.fg_latency, fg_see=args.fg_see, fg_miss=args.fg_miss)
    if robot is not None and hasattr(robot, "start"):
        robot.start()

    ok, halted = True, None
    try:
        # A name matched fuzzily is what the dog thinks it heard, so it asks
        # before setting off -- on the handle, as at the lift. No is a stop,
        # by the layer that guessed.
        for _, name, heard in guessed:
            place = name if name.startswith("room") else f"the {name}"
            if not run.ask(f'I heard "{heard}". Did you mean {place}? Press continue '
                           f"to go there.", what="destination"):
                raise HandleStop(f'"{heard}" was not confirmed as {place}',
                                 layer="language")
        k = reroutes = 0
        while k < len(legs):
            floor, name, waypoints, announcements, via, stop, transit = legs[k]
            # Ask #1, as the dog is about to set off for the stairs -- not at
            # the start of a trip that may visit three rooms first. A no is
            # the rest of the trip planned again from here, by the lift; with
            # the lift out of use, it is the end of the trip, said out loud.
            if via == "stairs" and not run.offer_stairs(transit, floor, legs[k + 1].floor):
                here = run.robot.get_pose()[:2]
                try:
                    _, rest, rest_faces = plan_stops(router, stops[stop:], start=(floor, here),
                                                     consent=lambda *_: False)
                except NoAccessibleRoute as refusal:
                    run.say(str(refusal))
                    raise HazardStop(f"on floor {floor}: {refusal}", layer="planning")
                legs = legs[:k] + [leg._replace(stop=leg.stop + stop) for leg in rest]
                faces = {i: p for i, p in faces.items() if i < k}
                faces.update({k + i: p for i, p in rest_faces.items()})
                continue
            print(f"  leg {k + 1}/{len(legs)}: floor {floor}, {len(waypoints)} checkpoints "
                  f"-> {'the ' + via if via else name}")
            ok, ticks = run.follow(waypoints, floor, announcements,
                                   goal_r=LIFT_GOAL_R if via == "lift" else GOAL_R,
                                   to_lift=via == "lift")
            if not ok and run.reroute is not None and reroutes < MAX_REROUTES:
                # Blocked, asked, still blocked: that way is closed for this
                # trip, and the rest is planned again from here (see ASK_S).
                rfloor, spot, ryaw = run.reroute
                close_spot(router, rfloor, spot, ryaw)
                reroutes += 1
                here = run.robot.get_pose()[:2]
                try:
                    _, rest, rest_faces = plan_stops(router, stops[stop:], start=(floor, here))
                except NoAccessibleRoute as refusal:
                    run.say("I cannot find another way. Stopping here.")
                    print(f"  leg {k + 1}: blocked at ({spot[0]:.1f}, {spot[1]:.1f}) and no "
                          f"other way: {refusal}")
                    break
                print(f"  leg {k + 1}: blocked at ({spot[0]:.1f}, {spot[1]:.1f}); planned "
                      f"again from ({here[0]:.1f}, {here[1]:.1f}): {len(rest)} leg(s)")
                legs = legs[:k] + [leg._replace(stop=leg.stop + stop) for leg in rest]
                faces = {i: p for i, p in faces.items() if i < k}
                faces.update({k + i: p for i, p in rest_faces.items()})
                continue
            if not ok:
                if run.reroute is not None:
                    run.say("I cannot find another way. Stopping here.")
                print(f"  leg {k + 1} timed out after {ticks} ticks")
                break
            if k in faces:
                run.face(faces[k])
            if via == "lift":
                run.lift_ride(floor, legs[k + 1][0], onward=legs[k + 1].points)
            elif via == "stairs":
                run.stairs_ride(floor, legs[k + 1][0])
            else:
                # Arrived somewhere the person asked for. Stand still long
                # enough for that to register -- a guide dog that announces a
                # destination and immediately walks off has not delivered it.
                run.hold(WAIT_S * 2)
            k += 1
    except (HazardStop, BoardStopped) as stop:
        ok, halted = False, stop
    if ok:
        run.behind = True
        for _ in range(FPS * 2):      # hold the arrival, seen over its shoulder
            run.frame()
    run.close()
    if args.robot == "go2_board":
        run.robot.board.lie()
        run.robot.close()
        if run.robot.map_fix is not None:
            print(run.robot.map_fix.report())

    x, y, _ = run.robot.get_pose()
    if isinstance(halted, HandleStop):
        print(f"STOPPED {halted}")
    elif halted is not None:
        print(f"HALTED {halted}")
    outcome = "ARRIVED" if ok else "STOPPED" if isinstance(halted, HandleStop) else "FAILED"
    print(f"{outcome} on floor {run.floor} at ({x:.1f}, {y:.1f}) "
          f"after {run.robot.sim_time:.0f}s of sim time")
    if not ok:
        # One line naming the layer that ended the run, so a failed log
        # answers "which layer, and why" without reading the trace above.
        layer, why = ((halted.layer, str(halted)) if halted is not None
                      else run.gave_up or ("control", "leg ended without arriving"))
        print(f"{outcome} BY {layer}: {why}")
    if run.fg is not None:
        print(run.fg.summary(run.robot.CONTROL_DT))
    for floor, p in sorted(run.policies.items()):
        st = p.stats
        mean = st["safety_sum"] / st["chosen"] if st["chosen"] else 0.0
        off_map = (st["rejected"] - st["rejected_by_gate"] - st["rejected_behind"]
                   - st["rejected_by_affordance"])
        aff = (f"{st['rejected_by_affordance']} by the affordance network "
               f"({st['affordance_unjudged']} more it could not judge), "
               if p.affordance is not None else "")
        print(f"VAMOS floor {floor}: {st['calls']} calls, {st['failures']} failed, "
              f"{st['rejected']} candidates rejected "
              f"({st['rejected_by_gate']} of them by the safety gate, {aff}{off_map} by "
              f"traversable() through a wall or obstacle, {st['rejected_behind']} "
              f"already behind the dog), "
              f"mean safety of the paths it followed {mean:.2f}")
        if st["failures"] and p.last_error:
            print(f"  last VAMOS failure: {p.last_error}")
    if args.vamos or args.shadow:
        v = run.vlm
        k = max(v["answers"], 1)
        dropped = v["dropped"] + (run.pending is not None)
        print(f"VLM ASYNC: {v['asked']} asked, {v['answers']} answered, {dropped} "
              f"dropped at a leg's end; answers were {v['age'] / k:.1f}s old on "
              f"arrival, the dog {v['moved'] / k:.2f} m on from where it asked; "
              f"the control loop never waited")
        if args.vamos and v["cap_min"] is not None:
            lat = max(run.vlm_lat) if run.vlm_lat else float("nan")
            print(f"VLM SPEED CAP: {run.vlm_speed_cap():.2f} m/s at the end "
                  f"(slowest of the last {len(run.vlm_lat)} answers {lat:.1f}s), "
                  f"lowest {v['cap_min']:.2f} m/s; ({VLM_REACH_D} - {VLM_STOP_D}) m / 2 L")
    if run.shadow is not None:
        run.shadow.close()
        sn = run.shadow_n
        print(f"SHADOW: {sn['calls']} VAMOS calls logged, {sn['steer']} would have "
              f"steered more than {math.degrees(SHADOW_DISAGREE):.0f} deg off the map "
              f"route, {sn['none']} had nothing pass the gate -> {args.shadow_log}")
    # Tick for tick what the dog was told to do; compare across runs.
    print(f"commands: {run.n_commands} ticks, sha256 {run.commands.hexdigest()[:16]}")
    o, h = run.obs, run.hits
    print(f"obstacles: up to {o['seen']} returns the map could not explain, "
          f"{o['crawl'] / FPS:.1f}s crawling, {o['stopped'] / FPS:.1f}s stopped, "
          f"least clearance ahead {o['min_clear']:.2f} m")
    # Ground truth, and the only line here the robot cannot flatter itself on.
    # Except on the board: the twin is put where the board says the dog is,
    # which with its odometry (POSE=lio) is off by that odometry's error. The
    # world scores the truth itself there (go2_world/score.py, run_hil.sh).
    where = " [on the board's pose]" if args.robot == "go2_board" else ""
    if h["ticks"]:
        print(f"COLLISIONS: {h['ticks'] / FPS:.1f}s touching {', '.join(sorted(h['boxes']))}{where}")
    else:
        print(f"collisions: none -- no part of the dog touched a wall or an obstacle{where}")
    hh = run.handler_hits
    if hh["ticks"]:
        print(f"HANDLER: {hh['ticks'] / FPS:.1f}s with the person on the handle touching "
              f"{', '.join(sorted(hh['boxes']))}{where}")
    else:
        print(f"handler: none -- the person on the handle never touched a wall or an obstacle{where}")
    print(f"turns eased to keep the person on the handle clear: {o['eased'] / FPS:.1f}s")
    if hasattr(run.robot, "guarded_ticks"):          # --robot twin-board
        print(f"board guard (modelled) cut the command: {run.robot.guarded_ticks / FPS:.1f}s")
    if run.aff is not None:
        run.aff_log.close()
        an = run.aff_n
        judged = {k: v for k, v in an.items() if isinstance(k, tuple)}
        total = sum(judged.values())
        skipped = ", ".join(f"{v} {k[9:]}" for k, v in an.items() if isinstance(k, str))
        print(f"AFFORDANCE (shadow): {total} judgements ({skipped or 'none skipped'}) "
              f"-> {args.affordance_shadow}")
        for truth in ("open floor", "wall", "obstacle", "person", "stairs"):
            row = {c: judged.get((c, truth), 0) for c in ("walkable", "caution", "not walkable")}
            k = sum(row.values())
            if k:
                print(f"  {truth:10s} {k:5d}:  " + ", ".join(
                    f"{c} {v / k:.0%}" for c, v in row.items()))
    if run.stair_flights:
        print(f"stairs: {run.stair_flights} flight(s), {run.stair_steps} steps, each "
              f"taken after two yeses (offered, then at the first step)")
    if args.handle:
        hs = run.mux.stats
        print(f"handle: {hs['tugs']} tug(s), {hs['continues']} continue(s), "
              f"{hs['stopped_ticks'] / FPS:.1f}s stopped by the person, "
              f"{hs['slowed_ticks'] / FPS:.1f}s slowed by a pull, "
              f"lowest pace {hs['lowest_pace']:.0%} of top speed")
    # Also when there are none: a yield in an empty building is a false
    # positive, and it should be as visible as a collision is.
    if args.pedestrians or run.ped["yields"]:
        pd = run.ped
        print(f"people: {pd['yields']} times it stopped to let someone past, "
              f"{pd['waited'] / FPS:.1f}s waiting in total; held off moving "
              f"towards somebody within {KEEP_OFF} m for {pd.get('kept_off', 0) / FPS:.1f}s")
        if run.planner == "cerrt":
            pl = run.plans
            tried = pl["made"] + pl["none"]
            print(f"planner: CE-RRT*, went round somebody {pl['rounds']} times, "
                  f"{pl['ticks'] / FPS:.1f}s on a plan; {tried} plans, "
                  f"{pl['none']} found nothing, "
                  f"{pl['ms'] / max(tried, 1):.0f} ms each, worst {pl['worst_ms']:.0f} ms; "
                  f"{pl['held']} times kept the plan without a search")
            if run.evade["steps"]:
                print(f"planner: stepped aside for somebody {run.evade['steps']} times, "
                      f"{run.evade['ticks'] / FPS:.1f}s stepping")
            if run.why_wait:
                print("planner: waited instead because -- " + ", ".join(
                    f"{why} {k}" for why, k in run.why_wait.most_common()))
        if pd["near"]:
            k = pd.get("dog_in", 0)
            print(f"CONTACT: {pd['near'] / FPS:.1f}s within {PED_NEAR} m of a person "
                  f"(closest {pd['min_d']:.2f} m); "
                  + (f"DOG CLOSING IN for {k / FPS:.1f}s of it "
                     f"(up to {pd['dog_v']:.2f} m/s)" if k
                     else f"the dog never closing in faster than {DOG_CLOSING} m/s"))
        elif not args.pedestrians:
            # Nobody in this run's own crowd to be near: on the board the people
            # walk in go2_world, which scores them (run_hil.sh's "world truth").
            print("contact: not scored here -- no people of its own"
                  + (" (they are in go2_world: see its world truth)"
                     if args.robot == "go2_board" else ""))
        else:
            print(f"contact: none -- closest it came to anybody was {pd['min_d']:.2f} m")
    if run.autolabel is not None:
        # Clean is the same bar Gate D sets: arrived, and nothing touched --
        # not the dog, not the person on the handle, not anybody walking past.
        clean = (outcome == "ARRIVED" and not h["ticks"] and not hh["ticks"]
                 and not run.ped["near"])
        kept = run.autolabel.finish({
            "command": args.command, "pedestrians": args.pedestrians, "seed": args.seed,
            "outcome": outcome, "clean": clean, "collision_ticks": h["ticks"],
            "handler_ticks": hh["ticks"], "contact_ticks": run.ped["near"]})
        print(f"autolabel: {kept} labelled frames, run {'clean' if clean else 'NOT clean'}"
              f" -> {args.autolabel}")
    if not args.no_video:
        pace = "real time" if args.speed == 1 else f"{args.speed}x real time"
        print(f"video -> {args.out}  ({run.robot.sim_time / args.speed:.0f}s, {pace})")


if __name__ == "__main__":
    main()
