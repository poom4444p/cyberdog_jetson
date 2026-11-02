"""The Mid-360, put to use: clearance for the board loop, clouds for the brain.

    Mid-360 --UDP 56301/56401--> LidarThread --Clearance--> App.step (500 Hz)
                                        \\
                                         +--UDP 9872--> brain: CloudLink

The receiver runs in its own thread: a dense frame is 210 packets, and parsing
them must not cost the motor loop a tick. The loop only ever reads the newest
Clearance, which is replaced whole, never edited, so it needs no lock.

Points are levelled with the Mid-360's own IMU (gravity, low-passed), so a
dog pitching as it walks does not see the floor rise into an obstacle, and
are moved to the trunk: x forward, y left, z up, origin on the floor under
the trunk origin. That is also the frame the brain gets them in.

Clearance is the free distance from the body's edge to the nearest obstacle
in front, behind, left and right: points 0.10 to 1.0 m above the floor, in a
corridor the width (or length) of the body plus a margin. guard() turns it
into a velocity limit, and a LiDAR that has gone quiet stops translation (but
not turning on the spot), the same rule as a brain that has gone quiet.

What the mount cannot see: the 20 deg nose-down mount looks at the floor from
about 1 m out in front, so low things right at the front feet, and most of
what is low behind the dog, are not seen. The guard is a backstop for the
brain, not a replacement for it.
"""
import math
import socket
import struct
import threading
import time

import numpy as np

from go2_board.livox import LivoxReceiver

# Mount, sensor frame -> trunk: must match go2_world/mid360.py until the real
# bracket is measured. SENSOR_HEIGHT is only where the floor search starts:
# each frame finds the floor itself, since lying, standing up and walking each
# put the sensor at a different height.
MOUNT_X = 0.0
MOUNT_PITCH = 20.0           # degrees nose-down
SENSOR_HEIGHT = 0.50

# The floor: the patch the pitched mount sees well, ahead of the front feet.
FLOOR_PATCH = (0.8, 3.0, 1.0)    # x from, x to, |y| under; levelled sensor frame
FLOOR_BAND = 0.03                # m around the lowest tenth that counts as floor
FLOOR_MIN = 20                   # points, or keep the last height
HEIGHT_RANGE = (0.10, 0.90)      # a height outside this is not believed

# The body, from the trunk origin, and what counts as in the way.
BODY_FRONT, BODY_BACK, BODY_HALF_W = 0.35, 0.35, 0.20
MARGIN = 0.10
OBSTACLE_Z = (0.10, 1.00)    # above the floor

STOP_M, SLOW_M = 0.30, 1.00  # free distance: no motion that way / full speed
# Behind, the nose-down mount's lowest ray (-7 deg + 20 deg) points 13 deg up:
# nothing lower than the sensor is ever seen there, so backing up is slow.
REVERSE_MAX = 0.2           # m/s
STALE_S = 0.30               # three frames missed: treat as blind
G_ALPHA = 0.02               # gravity low-pass a 200 Hz sample: ~0.25 s

BRAIN_CLOUD_PORT = 9872
HOST = "127.0.0.1"


def mount_rotation(pitch_deg=MOUNT_PITCH):
    """R_trunk_sensor: positive pitch tips the sensor's x towards the floor."""
    p = math.radians(pitch_deg)
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def levelling(g):
    """The rotation taking the measured up direction g to +z, shortest way."""
    g = np.asarray(g, float) / np.linalg.norm(g)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(g, z), float(g @ z)
    if c < -0.999:                       # upside down: nothing sensible to do
        return np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


class Clearance:
    __slots__ = ("front", "back", "left", "right", "stamp", "points")

    def __init__(self, front, back, left, right, stamp, points):
        self.front, self.back, self.left, self.right = front, back, left, right
        self.stamp = stamp            # board clock, when the frame was done
        self.points = points          # obstacle points it was made from

    def as_dict(self, now):
        d = {k: (None if math.isinf(v) else round(v, 3))
             for k, v in (("front", self.front), ("back", self.back),
                          ("left", self.left), ("right", self.right))}
        d["age"] = round(now - self.stamp, 3)
        return d


def clearance(xyz, stamp):
    """Free distance each way from levelled trunk-frame points."""
    z = xyz[:, 2]
    p = xyz[(z > OBSTACLE_Z[0]) & (z < OBSTACLE_Z[1])]
    x, y = p[:, 0], p[:, 1]
    side = np.abs(y) < BODY_HALF_W + MARGIN
    along = (x > -BODY_BACK - MARGIN) & (x < BODY_FRONT + MARGIN)

    def nearest(d):
        return float(d.min()) if len(d) else math.inf

    return Clearance(
        max(0.0, nearest(x[side & (x > 0)]) - BODY_FRONT),
        max(0.0, nearest(-x[side & (x < 0)]) - BODY_BACK),
        max(0.0, nearest(y[along & (y > 0)]) - BODY_HALF_W),
        max(0.0, nearest(-y[along & (y < 0)]) - BODY_HALF_W),
        stamp, len(p))


def floor_height(pts, last):
    """The sensor's height above the floor, from levelled sensor-frame points:
    the lowest tenth of the patch ahead, then everything within FLOOR_BAND of
    it. Too few points (a wall right in front) or an unbelievable answer:
    keep the last one."""
    x0, x1, half_w = FLOOR_PATCH
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    patch = z[(x > x0) & (x < x1) & (np.abs(y) < half_w) & np.any(pts != 0, axis=1)]
    if len(patch) < FLOOR_MIN:
        return last
    low = np.percentile(patch, 10)
    floor = patch[np.abs(patch - low) < FLOOR_BAND]
    h = -float(np.median(floor))
    if len(floor) < FLOOR_MIN or not HEIGHT_RANGE[0] < h < HEIGHT_RANGE[1]:
        return last
    return h


def _scale(free):
    return min(1.0, max(0.0, (free - STOP_M) / (SLOW_M - STOP_M)))


def guard(vx, vy, w, c, now):
    """The brain's velocity, limited by what the LiDAR sees. c None or stale:
    no translation, turning still allowed. Backwards: never past REVERSE_MAX."""
    if c is None or now - c.stamp > STALE_S:
        return 0.0, 0.0, w
    if vx < 0:
        vx = max(vx, -REVERSE_MAX)
    vx *= _scale(c.front) if vx > 0 else _scale(c.back) if vx < 0 else 1.0
    vy *= _scale(c.left) if vy > 0 else _scale(c.right) if vy < 0 else 1.0
    return vx, vy, w


# -- clouds to the brain ------------------------------------------------------
#
# One frame is cut into datagrams of CHUNK points: macOS refuses UDP datagrams
# over 9216 bytes by default. Each one: magic, frame seq, chunk i of n, the
# sensor's stamp (ns), the board's clock (s, the same clock as the state
# messages), point count, then float32 x, y, z in the levelled trunk frame.

CLOUD_HEAD = struct.Struct("<4sIHHQdI")
MAGIC = b"G2PC"
CHUNK = 600                  # 7232-byte datagrams


def cloud_datagrams(seq, xyz, sensor_ns, t):
    pts = np.ascontiguousarray(xyz, dtype="<f4")
    n = max(1, math.ceil(len(pts) / CHUNK))
    for i in range(n):
        part = pts[i * CHUNK:(i + 1) * CHUNK]
        yield CLOUD_HEAD.pack(MAGIC, seq & 0xFFFFFFFF, i, n, sensor_ns, t, len(part)) + part.tobytes()


class Cloud:
    __slots__ = ("seq", "xyz", "stamp", "t")

    def __init__(self, seq, xyz, stamp, t):
        self.seq, self.xyz, self.stamp, self.t = seq, xyz, stamp, t


class CloudLink:
    """The brain's end (for Cyberdog): the newest whole cloud, or None.
    A frame missing a chunk is dropped when the next one starts, never
    handed over half done."""

    def __init__(self, port=BRAIN_CLOUD_PORT, host=HOST):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.cloud, self.dropped = None, 0
        self._seq, self._parts = None, {}

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(16384)
            except (BlockingIOError, InterruptedError):
                return self.cloud
            if len(data) < CLOUD_HEAD.size:
                continue
            magic, seq, i, n, stamp, t, count = CLOUD_HEAD.unpack_from(data)
            if magic != MAGIC or len(data) != CLOUD_HEAD.size + 12 * count:
                continue
            if seq != self._seq:
                if self._parts:
                    self.dropped += 1
                self._seq, self._parts = seq, {}
            self._parts[i] = np.frombuffer(data, "<f4", offset=CLOUD_HEAD.size).reshape(-1, 3)
            if len(self._parts) == n:
                xyz = np.concatenate([self._parts[k] for k in range(n)])
                self.cloud, self._parts = Cloud(seq, xyz, stamp, t), {}

    def close(self):
        self.sock.close()


# -- the thread ---------------------------------------------------------------

class Lidar:
    """Frames in, Clearance and clouds out. process() is the work; start()
    runs it in a thread against a LivoxReceiver."""

    def __init__(self, receiver=None, clock=time.monotonic, brain=(HOST, BRAIN_CLOUD_PORT),
                 mount_pitch=MOUNT_PITCH):
        self.rx = receiver
        self.clock = clock
        self.brain = brain
        self.R_ts = mount_rotation(mount_pitch)
        self.up = self.R_ts.T @ np.array([0.0, 0.0, 1.0])   # sensor frame, until the IMU says
        self.height = SENSOR_HEIGHT                          # sensor above the floor, per frame
        self.latest = None                                   # Clearance
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if brain else None
        self._stop = threading.Event()
        self._thread = None

    def clearance(self):
        return self.latest

    def level(self, xyz_sensor):
        """Sensor-frame points -> levelled trunk frame, floor at z = 0."""
        R = levelling(self.up)
        pts = np.einsum("ij,kj->ik", xyz_sensor, R)   # not @: Accelerate warns spuriously on macOS
        self.height = floor_height(pts, self.height)
        pts[:, 0] += MOUNT_X
        pts[:, 2] += self.height
        return pts

    def process(self, frames, samples):
        for s in samples:                # specific force at rest points up
            self.up = (1 - G_ALPHA) * self.up + G_ALPHA * s.acc / np.linalg.norm(s.acc)
        for f in frames:
            pts = self.level(f.xyz)
            now = self.clock()
            self.latest = clearance(pts, now)
            if self.sock is not None:
                self.seq += 1
                for d in cloud_datagrams(self.seq, pts, int(f.stamp), now):
                    try:
                        self.sock.sendto(d, self.brain)
                    except OSError:
                        break            # no brain listening: the board does not care

    def _run(self):
        while not self._stop.is_set():
            frames, samples = self.rx.poll()
            if frames or samples:
                self.process(frames, samples)
            else:
                time.sleep(0.002)

    def start(self):
        if self.rx is None:
            self.rx = LivoxReceiver()
        self._thread = threading.Thread(target=self._run, name="lidar", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        if self.rx is not None:
            self.rx.close()
        if self.sock is not None:
            self.sock.close()
