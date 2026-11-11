"""The Mid-360, put to use: clearance for the board loop, clouds for the brain.

    Mid-360 --UDP 56301/56401--> LidarThread --Clearance--> App.step (500 Hz)
                                        \\
                                         +--UDP 9872--> brain: CloudLink

The receiver runs in its own thread: a dense frame is 210 packets, and parsing
them must not cost the motor loop a tick. The loop only ever reads the newest
Clearance, which is replaced whole, never edited, so it needs no lock.

Points are levelled with the Mid-360's own IMU, so a dog pitching as it walks
does not see the floor rise into an obstacle, and are moved to the trunk: x
forward, y left, z up, origin on the floor under the trunk origin. That is
also the frame the brain gets them in.

The level is tracked with the gyro and only corrected by gravity, slowly
(Tilt, TILT_TAU_S), and each point is levelled with the tilt at its own
timestamp (the IMU and the points share the sensor's clock). Gravity alone,
low-passed over 0.25 s, was both too slow for a trot's rocking and pulled
about by the body's accelerations: on HIL 11-10 the floor 1-8 m ahead was
tilted more than 2 deg in 39 % of frames (p5 -4.5, p95 +3.3 deg), its far
end rose above the brain's 0.08 m obstacle height, and the brain saw a
person-wide "something" across the corridor about 3 m ahead.

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
# A frame is used only when it is a whole sweep. A part of one sees only part
# of what is there: the guard reads the missing sector as open floor, and the
# brain's tracker, given a different part of the same trolley each frame, saw
# its centroid jump and called it somebody walking ("Someone crossing from the
# right. Waiting.", 33 s, PEOPLE=0: HIL 11-11 14:51 with the dense pattern,
# where frames came in bursts of three, most a third of a sweep). Dropped, the
# guard goes blind after STALE_S and stops the dog -- the safe way to fail.
FRAME_S = 0.100              # one sweep at the driver's 10 Hz
MIN_SPAN = 0.9               # of FRAME_S its points' times must cover
MAX_LOST = 0.05              # of its packets it may have lost
# Behind, the nose-down mount's lowest ray (-7 deg + 20 deg) points 13 deg up:
# nothing lower than the sensor is ever seen there, so backing up is slow.
REVERSE_MAX = 0.2           # m/s
STALE_S = 0.30               # three frames missed: treat as blind
IMU_DT = 1.0 / 200            # s, the Mid-360's IMU period: used when a sample has no usable time
TILT_TAU_S = 1.0             # s over which gravity corrects the gyro's level: longer
                             # lets less of the trot's acceleration in
BIAS_TAU_S = 2.0             # s over which the gyro's bias is learnt from what is
                             # left (the integral term), so a long TILT_TAU_S does not
                             # let a biased gyro walk the level off
ACC_TRUST = 0.2              # g away from 1 g at which the accelerometer is not
                             # trusted at all (linear in between): it is the body
                             # accelerating, not gravity
CMD_DV = 0.10                # m/s the commanded velocity changes by before...
ACC_HOLD_S = 0.75            # ...the accelerometer is ignored this long after (the
                             # gyro alone carries the level). ACC_TRUST cannot see a
                             # dog braking or starting: 2 m/s^2 forward moves the
                             # reading 2 % off 1 g but tips it 11 deg, and gravity
                             # correction followed it. HIL 11-11 12:38: braking from
                             # 0.65 m/s at the server room, the floor 3-5 m ahead rose
                             # to +0.15 m, a 5 m wide "obstacle", "Stopping. I have
                             # no clear way past". 0.75 s covers the policy's lag.
TILT_SETTLE_S = 1.0         # s after the first sample over which gravity corrects
TILT_SETTLE_TAU_S = 0.1      # ...this fast: the first sample, taken as up, can be
                             # a body accelerating (a dog started while walking)
TILT_KEEP_S = 0.5            # s of tilt history kept to level a frame's points with
TILT_SLICES = 10             # a frame is levelled in this many time slices

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


def rotate(v, axis_angle):
    """v turned by the rotation vector axis_angle (Rodrigues)."""
    a = float(np.linalg.norm(axis_angle))
    if a < 1e-12:
        return v
    k = axis_angle / a
    c, s = math.cos(a), math.sin(a)
    return v * c + np.cross(k, v) * s + k * float(k @ v) * (1.0 - c)


class Tilt:
    """Which way is up, in the sensor frame, from the Mid-360's IMU.

    The gyro carries it from sample to sample (a world-fixed direction seen
    from a sensor turning at w moves at -w x up); gravity pulls it back over
    TILT_TAU_S, less the further the accelerometer reads from 1 g. Kept for
    TILT_KEEP_S so a frame's points can each be levelled with the tilt of
    the moment they were measured. `add(s, held=True)`: the body is known to
    be speeding up or slowing down (ACC_HOLD_S), gyro only."""

    def __init__(self, up):
        self.up = np.asarray(up, float) / np.linalg.norm(up)
        self.t = None                    # ns of the last sample
        self.seen = False                # a sample has set it from gravity
        self.history = []                # [(ns, up)]
        self.age = 0.0                   # s of samples since the first
        self.bias = np.zeros(3)          # rad/s, the gyro's, as learnt

    def add(self, s, held=False):
        g = float(np.linalg.norm(s.acc))
        if g < 1e-6:
            return
        if not self.seen:                # the first word from gravity: believe it
            self.up, self.seen = s.acc / g, True
        else:
            dt = (s.t - self.t) * 1e-9 if self.t is not None else 0.0
            if not 0.0 < dt < 0.1:       # no usable time (or a gap): one IMU period
                dt = IMU_DT
            self.age += dt
            trust = max(0.0, 1.0 - abs(g / 9.81 - 1.0) / ACC_TRUST)
            if held and self.age >= TILT_SETTLE_S:
                trust = 0.0              # no correction, and no bias learnt from it
            tau = TILT_SETTLE_TAU_S if self.age < TILT_SETTLE_S else TILT_TAU_S
            # Mahony: e turns the level towards gravity (proportional), and
            # what it keeps having to turn is the gyro's bias (integral).
            e = np.cross(self.up, s.acc / g) * trust
            w = np.asarray(s.gyro, float) - self.bias - e / tau
            up = rotate(self.up, -w * dt)
            if self.age >= TILT_SETTLE_S:
                self.bias += e * dt / (tau * BIAS_TAU_S)
            self.up = up / np.linalg.norm(up)
        self.t = s.t
        self.history.append((s.t, self.up))
        cut = s.t - int(TILT_KEEP_S * 1e9)
        while len(self.history) > 2 and self.history[0][0] < cut:
            self.history.pop(0)

    def at(self, t):
        """Up at time t (ns): the nearest kept sample, or the newest."""
        if not self.history:
            return self.up
        ts = [h[0] for h in self.history]
        k = int(np.searchsorted(ts, t, side="right"))   # ts[k-1] <= t < ts[k]
        if k <= 0:
            return self.history[0][1]
        if k >= len(ts):
            return self.history[-1][1]
        return self.history[k][1] if ts[k] - t < t - ts[k - 1] else self.history[k - 1][1]


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
    if len(floor) < FLOOR_MIN:
        return last
    h = -float(np.median(floor))
    if not HEIGHT_RANGE[0] < h < HEIGHT_RANGE[1]:
        return last
    return h


def whole(frame):
    """Is `frame` a whole sweep? Its points span at least MIN_SPAN of FRAME_S
    and it lost no more than MAX_LOST of its packets. Points with no spread in
    time at all carry no sweep to judge: only the lost packets count then."""
    t = getattr(frame, "t", None)
    if t is not None and len(t) and t.max() > t.min():
        if (t.max() - t.min()) / 1e9 < MIN_SPAN * FRAME_S:
            return False
    sent = frame.packets + frame.missing
    return sent > 0 and frame.missing <= MAX_LOST * sent


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
        self.tilt = Tilt(self.R_ts.T @ np.array([0.0, 0.0, 1.0]))  # the mount's, until the IMU says
        self.height = SENSOR_HEIGHT                          # sensor above the floor, per frame
        self.latest = None                                   # Clearance
        self.cmd_ref = None                                  # (vx, vy) at the last change
        self.hold_until = -math.inf                          # clock: accelerometer ignored till
        self.seq = 0
        self.recorder = None                                 # record.LioRecorder, or None
        self.used = self.partial = 0                         # frames: whole, and dropped (see whole)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if brain else None
        self._stop = threading.Event()
        self._thread = None

    def clearance(self):
        return self.latest

    def truth(self, t, pos, quat):
        """The bus pose, for a recording (record.LioRecorder); nothing without one."""
        if self.recorder is not None:
            self.recorder.truth(t, pos, quat)

    @property
    def up(self):
        return self.tilt.up

    def level(self, xyz_sensor, t=None):
        """Sensor-frame points -> levelled trunk frame, floor at z = 0.

        `t`: each point's time (ns). Levelled in TILT_SLICES slices of the
        frame's time span, each with the tilt of its middle; without `t`,
        all with the newest tilt."""
        xyz_sensor = np.asarray(xyz_sensor, float)
        if t is None or len(t) == 0 or not self.tilt.history or t.max() == t.min():
            when = None if t is None or len(t) == 0 else int(np.median(t))
            R = levelling(self.tilt.up if when is None else self.tilt.at(when))
            pts = np.einsum("ij,kj->ik", xyz_sensor, R)   # not @: Accelerate warns spuriously on macOS
        else:
            pts = np.empty_like(xyz_sensor)
            edges = np.linspace(t.min(), t.max(), TILT_SLICES + 1)
            k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, TILT_SLICES - 1)
            for j in range(TILT_SLICES):
                m = k == j
                if m.any():
                    R = levelling(self.tilt.at(int((edges[j] + edges[j + 1]) / 2)))
                    pts[m] = np.einsum("ij,kj->ik", xyz_sensor[m], R)
        self.height = floor_height(pts, self.height)
        pts[:, 0] += MOUNT_X
        pts[:, 2] += self.height
        return pts

    def commanded(self, vx, vy, t):
        """The velocity the walker is being given, at clock time t. A change
        of more than CMD_DV is the body about to speed up or slow down: the
        accelerometer is ignored until ACC_HOLD_S after the last one."""
        if self.cmd_ref is None or math.hypot(vx - self.cmd_ref[0], vy - self.cmd_ref[1]) > CMD_DV:
            if self.cmd_ref is not None:
                self.hold_until = t + ACC_HOLD_S
            self.cmd_ref = (vx, vy)

    def process(self, frames, samples):
        if self.recorder is not None:                        # raw, before anything is done to it
            now = self.clock()
            self.recorder.imu(samples, now)
            self.recorder.frames(frames, now)
        held = self.clock() < self.hold_until
        for s in samples:                # specific force at rest points up
            self.tilt.add(s, held)
        for f in frames:
            if not whole(f):
                self.partial += 1
                continue
            self.used += 1
            pts = self.level(f.xyz, getattr(f, "t", None))
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
