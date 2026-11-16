"""Floor guard: a fast look at the floor ahead for what the LiDAR cannot see.

A flat puddle has no height, so the Mid-360 walks the dog straight into it.
Under --vamos the pace was capped so that VAMOS, the only thing looking at the
floor, could see one in time: (VLM_SEE_D - VLM_STOP_D) / 2 L with L ~2.3 s a
call on the J4012 -- ~0.32 m/s, well under walking pace. The floor guard is a
small, fast check that takes that job over, so the cap becomes
(see_d - STOP_D) / 2 L with its own L (a fraction of a second).

This is the simulated one (`--floor-guard sim`): the twin has no water, so
puddles are placed on the route (`place_puddles`) and the guard "sees" a puddle
when it would be in the camera's view -- in the image (the projector's own
pinhole), within `see_d` of the lens, and not behind a wall -- answering `L`
later, one frame at a time, back to back, like VAMOS. Each puddle in view is
missed with probability `p_miss` per frame. A real guard (a model on the
camera image) replaces `FloorGuard.look` and keeps everything else.

What the dog does with a puddle it knows of: `returns()` turns each into a
patch of points the clearance field keeps the dog off (LiveClearance.update's
`flat`: occupied, but with no LiDAR shadow behind it, since the camera sees the
floor beyond), so everything that goes round a crate on no map goes round it
too, and stops in front of it when there is no way round.
Puddles are remembered once seen (they do not walk away).
"""
import math
import random
from typing import NamedTuple

import numpy as np

from cyberdog.planning.checkpoint_projector import project_to_pixel

SEE_D = 2.0          # m from the lens a puddle is seen (same estimate as VLM_SEE_D)
STOP_D = 0.4         # m to stop from walking pace (run_building.VLM_STOP_D)
LATENCY = 0.25       # s from a frame to its answer (a small model on the J4012: a guess)
RETURN_H = 0.30      # m above the floor the fake returns sit (perception.MIN_H..MAX_H)
RETURN_STEP = 0.05   # m between fake returns across a puddle
WALL = 0.05          # static clearance under this along the line of sight is a wall
STEP_R = 0.15        # m from the body centre to its feet's edge: closer to a puddle is in it
# The points fill the whole puddle. Pulling them inside its edge, so the dog
# could keep less room from it than from a crate, walked it into one: on the
# way past, the dog only keeps perception.LINE_NEED (0.20 m) from a detected
# point, not DESTINATION_CLEAR, and 0.20 m is barely STEP_R (twin seed 3, 11-16).
RIM = 8              # points on a puddle's rim tried for visibility, besides its centre

PUDDLE_R = (0.30, 0.50)   # m, radius range of a placed puddle
PUDDLE_ENDS = 3.0         # m of route kept dry at each end of a leg
PUDDLE_LAT = 0.6          # m either side of the route a puddle centre may sit
PUDDLE_ROOM = 0.20        # m of static clearance beyond its radius: on open floor
PUDDLE_DOOR = 2.0         # m from a door no puddle is placed (doorways stay passable)
PUDDLE_GAP = 0.9          # m of open floor left beside it on at least one side: a puddle
                          # the dog can go round. Mid-corridor in a 2.3 m corridor a
                          # 0.49 m one left 0.5 / 0.8 m, and the dog stood beside it
                          # for the rest of the leg (twin seed 1, 11-16)


class Puddle(NamedTuple):
    id: int
    floor: int
    x: float
    y: float
    r: float


def _along(waypoints, s):
    """The point `s` metres along a polyline, and the unit direction there."""
    for a, b in zip(waypoints, waypoints[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg > 1e-9 and s <= seg:
            ux, uy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
            return (a[0] + ux * s, a[1] + uy * s), (ux, uy)
        s -= seg
    return None, None


def side_gap(x, y, r, nx, ny, static, reach=2.5, step=0.05):
    """Open floor beside a puddle, m, from its edge to the wall along (nx, ny)."""
    t = np.arange(r, r + reach, step)
    hit = np.nonzero(static(x + nx * t, y + ny * t) < WALL)[0]
    return (t[hit[0]] if len(hit) else r + reach) - r


def place_puddles(n, floor, waypoints, static, rng, near_door=None, first_id=0, tries=200):
    """`n` puddles on open floor near this leg's route, none in a doorway, each
    leaving PUDDLE_GAP of floor beside it on one side at least.

    `static(xs, ys)` is the map's clearance (LiveClearance._static);
    `near_door(xy)` says whether a point is within PUDDLE_DOOR of a door.
    Fewer than `n` come back when the leg is too short to hold them.
    """
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(waypoints, waypoints[1:]))
    if length <= 2 * PUDDLE_ENDS:
        return []
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        (px, py), (ux, uy) = _along(waypoints, rng.uniform(PUDDLE_ENDS, length - PUDDLE_ENDS))
        lat = rng.uniform(-PUDDLE_LAT, PUDDLE_LAT)
        x, y = px - uy * lat, py + ux * lat
        r = rng.uniform(*PUDDLE_R)
        if static(np.array([x]), np.array([y]))[0] < r + PUDDLE_ROOM:
            continue
        if near_door is not None and near_door((x, y)):
            continue
        if max(side_gap(x, y, r, -uy, ux, static), side_gap(x, y, r, uy, -ux, static)) < PUDDLE_GAP:
            continue
        if any(math.hypot(x - p.x, y - p.y) < r + p.r + 1.0 for p in out):
            continue
        out.append(Puddle(first_id + len(out), floor, x, y, r))
    return out


class FloorGuard:
    """The simulated guard: the puddles there are, and what it knows of them."""

    def __init__(self, cam, control_dt, see_d=SEE_D, latency=LATENCY, p_miss=0.0, seed=0):
        self.cam = cam
        self.see_d = see_d
        self.latency = latency
        self.lat_ticks = max(1, int(round(latency / control_dt)))
        self.p_miss = p_miss
        self.rng = random.Random(seed)
        self.puddles = []          # every puddle placed, all floors
        self.known = {}            # id -> Puddle, once an answer has said so
        self.pending = None        # (tick the answer lands, [Puddle]) or None
        self.frames = 0
        self.stepped = {}          # id -> ticks spent in it
        self.known_at = {}         # id -> metres from the body to its edge when it became known

    def cap(self):
        """Fastest the dog may walk on the guard's word, m/s: as vlm_speed_cap."""
        return (self.see_d - STOP_D) / (2.0 * self.latency)

    def add(self, puddles):
        self.puddles.extend(puddles)

    def next_id(self):
        return len(self.puddles)

    def visible(self, p, cam_pose, static):
        """Would one frame from `cam_pose` show this puddle?"""
        lx, ly, _ = cam_pose
        pts = [(p.x, p.y)] + [(p.x + p.r * math.cos(a), p.y + p.r * math.sin(a))
                              for a in np.linspace(0, 2 * math.pi, RIM, endpoint=False)]
        for q in pts:
            d = math.hypot(q[0] - lx, q[1] - ly)
            if d > self.see_d or project_to_pixel(q, cam_pose, self.cam)["state"] != "TRACK":
                continue
            k = max(2, int(d / 0.05))
            t = np.linspace(0.0, 1.0, k)
            if static(lx + t * (q[0] - lx), ly + t * (q[1] - ly)).min() >= WALL:
                return True
        return False

    def look(self, floor, cam_pose, static):
        """One frame: the puddles it shows that are not known yet."""
        seen = []
        for p in self.puddles:
            if p.floor != floor or p.id in self.known:
                continue
            if self.visible(p, cam_pose, static) and self.rng.random() >= self.p_miss:
                seen.append(p)
        return seen

    def tick(self, n, floor, cam_pose, body_xy, static):
        """Called every control tick. Returns the puddles that became known now.

        Back to back: a frame is taken when the last answer lands, so a puddle
        coming into view just after a frame is known at worst 2 L later.
        """
        new = []
        if self.pending is not None and n >= self.pending[0]:
            for p in self.pending[1]:
                if p.id not in self.known:
                    self.known[p.id] = p
                    self.known_at[p.id] = math.hypot(p.x - body_xy[0], p.y - body_xy[1]) - p.r
                    new.append(p)
            self.pending = None
        if self.pending is None:
            self.frames += 1
            self.pending = (n + self.lat_ticks, self.look(floor, cam_pose, static))
        return new

    def drop_pending(self):
        """A leg ended: the frame in flight was of a corridor the dog has left."""
        self.pending = None

    def returns(self, floor, z, centre, reach):
        """Known puddles on this floor near `centre`, as fake LiDAR returns [N, 3]."""
        rows = []
        for p in self.known.values():
            if p.floor != floor or math.hypot(p.x - centre[0], p.y - centre[1]) > reach + p.r:
                continue
            g = np.arange(-p.r, p.r + 1e-9, RETURN_STEP)
            gx, gy = np.meshgrid(g, g)
            m = gx ** 2 + gy ** 2 <= p.r ** 2
            rows.append(np.column_stack([p.x + gx[m], p.y + gy[m], np.full(m.sum(), z + RETURN_H)]))
        return np.concatenate(rows) if rows else np.empty((0, 3))

    def score(self, floor, xy):
        """Count this tick against every puddle the body is standing in."""
        for p in self.puddles:
            if p.floor == floor and math.hypot(p.x - xy[0], p.y - xy[1]) < p.r + STEP_R:
                self.stepped[p.id] = self.stepped.get(p.id, 0) + 1

    def summary(self, control_dt):
        n, k = len(self.puddles), len(self.known)
        d = sorted(self.known_at.values())
        near = f", nearest known {d[0]:.2f} m off its edge" if d else ""
        return (f"FLOOR GUARD: {n} puddles, {k} seen ({self.frames} frames, L {self.latency:.2f} s, "
                f"see {self.see_d:.1f} m, miss {self.p_miss:.2f}, cap {self.cap():.2f} m/s){near}; "
                f"stepped in {len(self.stepped)}"
                + (f" for {sum(self.stepped.values()) * control_dt:.1f} s" if self.stepped else ""))
