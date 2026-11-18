"""Keep the board's pose on the map: align each scan's walls to the map's.

On the board the pose is the Mid-360's own odometry (small_gicp + gyro,
anchored to the bus pose), and it drifts: xy off by up to 0.21 m, yaw by up
to 9.5 deg over a HIL route (11-17). Every return is put into the map with
that pose, so the walls land 0.15-0.3 m off their faces -- 1,758 returns a
scan on average that the map cannot explain -- and as the error changes they
move, and the tracker reads a wall that moves as somebody walking along it.
The dog stood in an empty corridor saying "Someone on your right. Waiting."
to the south wall for most of a minute (HIL PEOPLE=6, 11-17).

So the pose is corrected, not the perception loosened: widening what counts
as the wall (FACE_EXPLAINED 0.30) hid people standing by it again (twin seed
2). A correction C, a rigid transform of the map frame, is kept per floor and
refined every scan: the returns near a wall face are pulled onto the faces
(a few Gauss-Newton steps on the face distance field, about the dog), and
the result is applied to the pose and to every point. Replayed on the HIL
clouds of 11-17 10:17: near-wall unexplained returns 1,758 -> 32 a scan,
correction median 0.08 m / 0.34 deg.

Only where the pose is the board's (BoardTwinRobot). The twin's pose is the
physics' own, nothing to correct.

Along a corridor the walls fix y and yaw but not x: only an end wall or the
jambs of a door say where along it the dog is, so the board's x drift builds
up unseen down the 47 m corridor and is corrected all at once at the far
end. HIL 11-17, routes to the west-end rooms: 0.45-0.60 m at the dog. With
the bound at 0.60 m (MAX_M) a needed 0.6+ reset C to nothing every few scans
(587 resets on the way to room 201): the walls there jumped 0.6 m back and
forth, the tracker read them as people crossing ("Someone crossing from the
right. Waiting." 29 times, on a floor with nobody on it) and the dog timed
out at the door. MAX_M is now 1.5 m, still under half the 6 m between doors,
so a fit cannot lock onto the next door along.

And a floor change keeps the correction: the board's odometry is one frame
for the whole building and the floors share the map's x, y, so the error
the dog had stepping into the lift is the one it has stepping out. It used
to start every floor from none and relearn it from the lift lobby.
"""
import math

import numpy as np
from scipy import ndimage

USE_M = 0.45          # m from a wall face: a return that may be that wall
MIN_POINTS = 200      # wall returns needed to trust a refinement at all
MAX_POINTS = 4000     # subsampled to this, for the J4012
ITERS = 4             # Gauss-Newton steps a scan
STEP_M, STEP_RAD = 0.10, math.radians(3.0)   # the most one step moves
MAX_M, MAX_RAD = 1.5, math.radians(12.0)     # the most C may ever be: past it, reset
HUBER = 0.10          # m off a face past which a return counts for less (Huber)


class MapFix:
    """C per floor: p_map = R(th) p_board + (tx, ty)."""

    def __init__(self):
        self.field = None     # the LiveClearance of the floor the dog is on
        self.corr = {}        # id(field) -> (tx, ty, th)
        self.carry = None     # the last C accepted, on any floor: a new floor starts from it
        self._grad = {}
        self.n = self.used = self.resets = 0
        self.size = []        # |t|, |th| after each refinement, for the summary

    def use(self, field):
        """The floor's LiveClearance (faces, res, ox, oy, H), or None (in the
        lift). Call before each scan. Arriving on a floor -- a new one, or
        back from the lift -- starts from the last correction accepted
        anywhere (`carry`): one odometry frame, one x, y for every floor."""
        if field is not self.field and field is not None and self.carry is not None:
            self.corr[id(field)] = self.carry
        self.field = field

    def _c(self):
        return self.corr.get(id(self.field), (0.0, 0.0, 0.0)) if self.field is not None \
            else (0.0, 0.0, 0.0)

    def pose(self, x, y, yaw):
        """The board's pose on the map."""
        tx, ty, th = self._c()
        c, s = math.cos(th), math.sin(th)
        return (c * x - s * y + tx, s * x + c * y + ty,
                math.atan2(math.sin(yaw + th), math.cos(yaw + th)))

    def points(self, p):
        """(N, 2+) board-frame map points, corrected (z untouched)."""
        tx, ty, th = self._c()
        if not (tx or ty or th) or not len(p):
            return p
        c, s = math.cos(th), math.sin(th)
        out = np.array(p, dtype=float, copy=True)
        out[:, 0] = c * p[:, 0] - s * p[:, 1] + tx
        out[:, 1] = s * p[:, 0] + c * p[:, 1] + ty
        return out

    def _fields(self):
        """Signed metres to the nearest wall face (negative inside a wall),
        and its gradient. Signed, not `faces` itself: that is 0 all through a
        wall, so a scan could sink its walls 0.1-0.2 m into them at no cost
        and the fit had a dead band as thick as the walls."""
        f = self.field
        if id(f) not in self._grad:
            F = np.asarray(f.faces, dtype=float)
            occ = F <= 0.5 * f.res + 1e-9
            depth = ndimage.distance_transform_edt(occ) * f.res
            # Cell centres: the face lies half a cell from each side's last cell.
            S = np.where(occ, -(depth - 0.5 * f.res), F - 0.5 * f.res)
            gr, gc = np.gradient(S, f.res)        # rows run -y
            self._grad[id(f)] = (S, gc, -gr)
        return self._grad[id(f)]

    def _at(self, A, x, y):
        """Bilinear, on the grid LiveClearance._face reads by truncation: cell
        (r, c) holds x - ox in [c, c + 1) res and H - 1 - (y - oy) / res in
        [r, r + 1), so its centre is half a cell in from both."""
        f = self.field
        c = (x - f.ox) / f.res - 0.5
        r = f.H - 1.5 - (y - f.oy) / f.res
        return ndimage.map_coordinates(A, [r, c], order=1, mode="nearest")

    def refine(self, p, centre):
        """Fold one corrected scan (N, 2+), map frame, into C. `centre` is the
        dog's corrected (x, y). Returns the points with the new C applied."""
        if self.field is None or len(p) < MIN_POINTS:
            return p
        self.n += 1
        F, GX, GY = self._fields()
        x, y = p[:, 0], p[:, 1]
        near = np.abs(self._at(F, x, y)) < USE_M
        if near.sum() < MIN_POINTS:
            return p
        idx = np.flatnonzero(near)
        if len(idx) > MAX_POINTS:
            idx = idx[:: int(math.ceil(len(idx) / MAX_POINTS))]
        bx, by = x[idx], y[idx]
        cx, cy = centre
        dx = dy = dth = 0.0
        for _ in range(ITERS):
            c, s = math.cos(dth), math.sin(dth)
            qx = cx + c * (bx - cx) - s * (by - cy) + dx
            qy = cy + s * (bx - cx) + c * (by - cy) + dy
            f = self._at(F, qx, qy)
            m = np.abs(f) < USE_M
            if m.sum() < MIN_POINTS:
                return p
            gx, gy = self._at(GX, qx[m], qy[m]), self._at(GY, qx[m], qy[m])
            rx, ry = qx[m] - cx, qy[m] - cy
            J = np.column_stack([gx, gy, -ry * gx + rx * gy])
            w = HUBER / np.maximum(np.abs(f[m]), HUBER)   # Huber: far ones count less
            H = (J * w[:, None]).T @ J + 1e-3 * np.eye(3)
            step = -np.linalg.solve(H, (J * w[:, None]).T @ f[m])
            dx += float(np.clip(step[0], -STEP_M, STEP_M))
            dy += float(np.clip(step[1], -STEP_M, STEP_M))
            dth += float(np.clip(step[2], -STEP_RAD, STEP_RAD))
        # Compose: the step was about the dog, C is about the map's origin.
        tx, ty, th = self._c()
        c, s = math.cos(dth), math.sin(dth)
        ntx = c * tx - s * ty + cx - (c * cx - s * cy) + dx
        nty = s * tx + c * ty + cy - (s * cx + c * cy) + dy
        nth = math.atan2(math.sin(th + dth), math.cos(th + dth))
        # Bounded by how far the dog can be off, measured from where it is:
        # the translation that matters is the one at the dog.
        at_dog = (math.cos(nth) * cx - math.sin(nth) * cy + ntx - cx,
                  math.sin(nth) * cx + math.cos(nth) * cy + nty - cy)
        if math.hypot(*at_dog) > MAX_M or abs(nth) > MAX_RAD:
            self.corr.pop(id(self.field), None)     # lost: start again from the board's
            self.carry = None
            self.resets += 1
            return p
        self.used += 1
        self.corr[id(self.field)] = self.carry = (ntx, nty, nth)
        self.size.append((math.hypot(*at_dog), abs(nth)))
        out = np.array(p, dtype=float, copy=True)
        out[:, 0] = cx + c * (x - cx) - s * (y - cy) + dx
        out[:, 1] = cy + s * (x - cx) + c * (y - cy) + dy
        return out

    def report(self):
        if not self.size:
            return f"map fix: {self.n} scans, none aligned"
        a = np.array(self.size)
        return (f"map fix: {self.used} of {self.n} scans aligned to the map, {self.resets} resets; "
                f"correction at the dog median {np.median(a[:, 0]):.2f} m, max {a[:, 0].max():.2f} m; "
                f"yaw median {math.degrees(np.median(a[:, 1])):.2f} deg, "
                f"max {math.degrees(a[:, 1].max()):.2f} deg")
