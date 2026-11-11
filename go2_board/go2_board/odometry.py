"""Where the dog is, from its own Mid-360: what a real Go2 on our policy needs.

The board reported go2_world's ground truth as the dog's pose (sim_bus.py);
a Go2 walking on our own policy has no odometry at all. This is it, from the
LiDAR and its IMU, chosen on recordings of HIL routes (go2_board.record)
against the true pose:

  small_gicp (VGICP, voxel map of the last scans), gyro-predicted rotation
  dense route (real point density), 26.4 m of floor 1: 0.17 m off at the end
  (0.6 %), 1.6 deg; KISS-ICP broke within 0.5 m on the same data; without the
  gyro a pivot beside the trolley read 0.88 m / 15 deg off, with it 0.02 m.

Two halves, two processes:

  LioOdometry   in the LiDAR process (lidar_proc): every whole sweep in, the
                body's pose in the odometry's own frame out (where the first
                sweep was taken: x forward, z up).
  PoseAnchor    in board_main: that frame placed where the brain's is (the
                bus frame, like the pose it replaces), anchored at the first
                odometry pose with the pose the bus had then; and anchored
                again after a floor change, which the LiDAR cannot see (the
                car rises with the dog in it) -- standing in for the lift's
                known exit point on a real dog. It also keeps the odometry's
                error against the bus pose, while the bus has the truth.
"""
import math

import numpy as np

from go2_board import lidar as L

LEAF = 0.25                  # m, the voxel map's cells (0.125 m sampling of each scan)
MIN_R, MAX_R = 0.1, 40.0     # m, returns used: the Mid-360's own range
MAX_CORR = 1.0               # m, furthest a point may be matched
MOUNT_Z = 0.20               # m, sensor above the trunk origin (go2_world mid360.MOUNT_XYZ)
CONFINED_M = 1.5             # m: half the returns nearer than this is a confined space
                             # (the lift car): the gyro's rotation, not the scans'
REANCHOR_DZ = 1.0            # m of bus height change: another floor, anchor again...
SETTLE_DZ, SETTLE_S = 0.02, 1.0   # ...once the height has stayed within this for this
                                  # long: out of the car on the new floor. Anchored at
                                  # every metre of the ride, it anchored 4 times in the
                                  # car and rode the last 1.4 m on odometry that cannot
                                  # see the car rise (replay of HIL 11-11 16:10)


def _T(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def _rot(v):
    """Rotation matrix of a rotation vector (rad), Rodrigues."""
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


class LioOdometry:
    """Sensor-frame sweeps (and the IMU) in; the body pose, in the odometry's
    frame, out. Needs small_gicp; `available` is False without it."""

    def __init__(self, mount_pitch=L.MOUNT_PITCH, leaf=LEAF, confined_rotation=True):
        try:
            import small_gicp
        except ImportError:
            self.sg, self.available = None, False
        else:
            self.sg, self.available = small_gicp, True
        self.leaf = leaf
        self.confined_rotation = confined_rotation
        self.confined = 0             # sweeps turned by the gyro alone (CONFINED_M)
        self.T_bs = _T(L.mount_rotation(mount_pitch), (L.MOUNT_X, 0.0, MOUNT_Z))
        self.T_sb = np.linalg.inv(self.T_bs)
        self.vmap = None
        self.T = np.eye(4)            # sensor pose, odometry frame (= first sensor pose)
        self.prev = np.eye(4)
        self.T0_body = None           # body pose of the first sweep, in the odometry frame
        self.gyro = []                # (t ns, rad/s) since the last sweep
        self.last_t = None            # ns, the last sweep's newest point
        self.n = 0

    def imu(self, samples):
        for s in samples:
            self.gyro.append((int(s.t), np.asarray(s.gyro, float)))

    def _gyro_rotation(self, t0, t1):
        R, last = np.eye(3), t0
        keep = []
        for t, w in self.gyro:
            if t <= t0:
                continue
            if t > t1:
                keep.append((t, w))
                continue
            R = R @ _rot(w * (t - last) / 1e9)
            last = t
        self.gyro = keep
        return R

    def add(self, xyz, t):
        """One whole sweep: sensor-frame points (N, 3), their times (ns).
        Returns the body pose (4x4) relative to the first sweep's body."""
        p = np.asarray(xyz, float)
        r = np.linalg.norm(p, axis=1)
        p = p[(r > MIN_R) & (r < MAX_R)]
        pcd, _ = self.sg.preprocess_points(p, downsampling_resolution=self.leaf / 2)
        t1 = int(np.max(t)) if len(t) else None
        if self.vmap is None:
            self.vmap = self.sg.GaussianVoxelMap(self.leaf)
            self.vmap.insert(pcd, self.T)
        else:
            delta = np.linalg.inv(self.prev) @ self.T if self.n > 1 else np.eye(4)
            if self.last_t is not None and t1 is not None:
                delta = delta.copy()
                delta[:3, :3] = self._gyro_rotation(self.last_t, t1)
            guess = self.T @ delta
            res = self.sg.align(self.vmap, pcd, guess, max_correspondence_distance=MAX_CORR)
            T = np.array(res.T_target_source)
            if self.confined_rotation and len(r) and np.median(r) < CONFINED_M:
                # In a box this small (the lift car: walls 0.1-1 m off, near
                # symmetric) the scans cannot tell how far it turned; turning
                # on the spot to face the doors read 15 deg off and walked the
                # pose 0.44 m (replay of HIL 11-11 16:10). The gyro's turn.
                T[:3, :3] = guess[:3, :3]
                self.confined += 1
            self.prev, self.T = self.T, T
            self.vmap.insert(pcd, self.T)
        self.last_t = t1 if t1 is not None else self.last_t
        self.n += 1
        body = self.T @ self.T_sb
        if self.T0_body is None:
            self.T0_body = body
        return np.linalg.inv(self.T0_body) @ body


class PoseAnchor:
    """The odometry's (x, y, yaw) placed in the bus frame; its error kept."""

    def __init__(self):
        self.A = None                 # bus <- odometry, planar (x, y, yaw)
        self.z0 = None                # bus height at the last anchor
        self.z_ref, self.z_since = None, None   # height lately, and since when (SETTLE_S)
        self.anchors = 0
        self.err = []                 # (xy m, yaw rad) against the bus, while it has one

    @staticmethod
    def _pose(x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])

    def place(self, odo, bus, t=None):
        """odo: (x, y, yaw) in the odometry frame; bus: (x, y, z, yaw) or None;
        t: the clock (s), for SETTLE_S. Returns (x, y, yaw) in the bus frame, or
        None before the first anchor."""
        O = self._pose(*odo)
        settled = True
        if bus is not None and t is not None:
            if self.z_ref is None or abs(bus[2] - self.z_ref) > SETTLE_DZ:
                self.z_ref, self.z_since = bus[2], t
            settled = t - self.z_since >= SETTLE_S
        if bus is not None and (self.A is None
                                or (abs(bus[2] - self.z0) > REANCHOR_DZ and settled)):
            self.A = self._pose(bus[0], bus[1], bus[3]) @ np.linalg.inv(O)
            self.z0 = bus[2]
            self.anchors += 1
        if self.A is None:
            return None
        P = self.A @ O
        out = (float(P[0, 2]), float(P[1, 2]), math.atan2(P[1, 0], P[0, 0]))
        if bus is not None:
            dyaw = math.atan2(math.sin(out[2] - bus[3]), math.cos(out[2] - bus[3]))
            self.err.append((math.hypot(out[0] - bus[0], out[1] - bus[1]), dyaw))
        return out

    def summary(self):
        if not self.err:
            return "no odometry against a bus pose"
        e = np.array(self.err)
        return (f"odometry vs bus pose: xy end {e[-1, 0]:.2f} m, max {e[:, 0].max():.2f} m; "
                f"yaw end {math.degrees(abs(e[-1, 1])):.1f} deg, max "
                f"{math.degrees(np.abs(e[:, 1]).max()):.1f} deg; {self.anchors} anchor(s)")
