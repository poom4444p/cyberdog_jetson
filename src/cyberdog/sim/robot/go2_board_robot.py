"""The brain's side of the Go2 board (go2_board, on the J4012).

Same RobotInterface as the twin, so run_building drives it unchanged:
set_velocity goes to the board as vx, vy, w over the localhost link, and
get_pose comes back from the board's state messages. The board owns the
legs, safety and the walking policy; nothing here touches a motor.

Needs go2_board installed (pip install -e ~/Work/unitree/go2_board). The
import is inside __init__, so the twin runs without it.

get_scan is the board's Mid-360 cloud; get_image, with camera_host, is the
dog's camera as go2_world renders it (CameraLink), people and all. Without
either they come from the physics twin, which has no people.
"""
import collections
import io
import json
import math
import socket
import struct
import threading
import time

import numpy as np

from cyberdog.sim.robot.robot_interface import RobotInterface

STATE_STALE_S = 0.2       # the board reports at 50 Hz: 10 missed reports


class BrainTicks:
    """The brain's ticks against the 20 Hz deadline, and what ran on each.

    HIL 11-23 12:02 (room 201 P6) counted 425 of 3253 ticks late, the worst
    10.2 s: a count that could not say why. The worst was no tick at all --
    the brain stood at the start and rode the lift through request(), never
    step(), and the first step after came seconds "late". Those are gaps
    now, counted apart. For the rest, each tick keeps how late it was, how
    long the brain worked on it (from the last step's return to this step),
    the ms of what the loop marked (mark: "scan", "plan", "frame", ...) and
    its tags: a CE-RRT* search on this tick (tag), or the VAMOS worker
    dreaming at any time during it (begin/end, from the worker's thread).
    """

    BINS = (10, 25, 50, 100, 250)         # ms past the deadline: histogram edges

    def __init__(self, dt):
        self.dt = dt
        self.ticks = self.late = self.gaps = 0
        self.late_ms_max = self.gap_s_max = 0.0
        self.rows = []                    # (late ms, work ms or None, tags, marks)
        self._lock = threading.Lock()
        self._active = collections.Counter()   # begun, not yet ended (any thread)
        self._seen = set()                     # begun at any time this tick
        self._tags, self._marks = set(), {}

    def begin(self, name):
        with self._lock:
            self._active[name] += 1
            self._seen.add(name)

    def end(self, name):
        with self._lock:
            self._active[name] -= 1
            if self._active[name] <= 0:
                del self._active[name]

    def tag(self, name):
        self._tags.add(name)

    def mark(self, name, ms):
        self._marks[name] = self._marks.get(name, 0.0) + ms

    def gap(self, s):
        """A stretch the brain did not tick (standing, the lift): not late."""
        self.gaps += 1
        self.gap_s_max = max(self.gap_s_max, s)
        self._take()

    def tick(self, late_ms, work_ms):
        """One step(): late_ms > 0 past the deadline, else on time."""
        self.ticks += 1
        if late_ms > 0:
            self.late += 1
            self.late_ms_max = max(self.late_ms_max, late_ms)
        tags, marks = self._take()
        self.rows.append((max(late_ms, 0.0), work_ms, tags, marks))

    def _take(self):
        with self._lock:
            tags = frozenset(self._tags | self._seen)
            self._seen = set(self._active)
        marks, self._tags, self._marks = self._marks, set(), {}
        return tags, marks

    def summary(self):
        """Lines for the end of a run: the count, then the breakdown."""
        dl = self.dt * 1000
        out = [f"BRAIN TICK: {self.late} of {self.ticks} ticks late, worst "
               f"{self.late_ms_max:.0f} ms past the {dl:.0f} ms deadline"
               + (f"; {self.gaps} gaps not counted (start, lift rides), longest "
                  f"{self.gap_s_max:.1f} s" if self.gaps else "")]
        late = [r for r in self.rows if r[0] > 0]
        if not late:
            return out
        edges = (0,) + self.BINS + (float("inf"),)
        counts = [sum(a < r[0] <= b for r in late) for a, b in zip(edges, edges[1:])]
        names = [f"{a}-{b}" for a, b in zip(edges, edges[1:-1])] + [f">{self.BINS[-1]}"]
        out.append("  late by (ms): " + ", ".join(f"{k} {c}" for k, c in zip(names, counts)))
        groups = collections.defaultdict(list)
        for r in self.rows:
            groups["+".join(sorted(r[2])) or "nothing"].append(r)
        for g in sorted(groups, key=lambda g: (-len(groups[g]), g)):
            rows = groups[g]
            lt = [r[0] for r in rows if r[0] > 0]
            work = [r[1] for r in rows if r[1] is not None]
            out.append(f"  {g}: {len(lt)} of {len(rows)} late"
                       + (f" (mean {np.mean(lt):.0f}, worst {max(lt):.0f} ms)" if lt else "")
                       + (f", work mean {np.mean(work):.1f} / worst {max(work):.0f} ms"
                          if work else ""))
        on = [r for r in self.rows if r[0] == 0]
        keys = sorted({k for r in self.rows for k in r[3]})
        for name, rows in (("late", late), ("on-time", on)):
            work = [r[1] for r in rows if r[1] is not None]
            if not work:
                continue
            parts = [f"work {np.mean(work):.1f}"] + [
                f"{k} {np.mean([r[3].get(k, 0.0) for r in rows]):.1f}" for k in keys]
            out.append(f"  mean ms on {name} ticks: " + ", ".join(parts))
        return out


class BoardNotReady(RuntimeError):
    pass


class LiftLink:
    """Calls the lift and reads where it is: go2_world's car over UDP 9880,
    standing in for the person at the call button (or a building's lift API).
    The robot never waits on it for its own height: that comes from the board."""

    def __init__(self, host="127.0.0.1", port=9880, timeout=0.5):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def _ask(self, msg):
        self.sock.sendto(json.dumps(msg).encode(), self.addr)
        return json.loads(self.sock.recvfrom(1024)[0])

    def call(self, floor):
        return self._ask({"call": int(floor)})

    def state(self):
        return self._ask({})

    def close(self):
        self.sock.close()


class BoardStopped(RuntimeError):
    """The board left WALK by itself (a safety trip, or a person's limp):
    whatever the brain sends now, the dog is not walking it. The run ends
    here and says so, rather than tracking a waypoint the dog never nears."""
    layer = "go2_board"


class Go2BoardRobot(RobotInterface):
    CONTROL_DT  = 0.05         # 20 Hz, the twin's rate: the brain is tuned at it
    MAX_V       = 0.8          # m/s   -- the twin's limits. The Go2 policy is
    MAX_W       = 1.5          # rad/s    trained to +-1 rad/s: check before WALK
    CAM_FORWARD = 0.34         # m, lens ahead of body centre: measure on the dog

    def __init__(self, origin=(0.0, 0.0, 0.0), port=None, board_port=None,
                 clock=time.monotonic, sleep=time.sleep, board_host=None, listen=None):
        """`origin` is where the board's odometry frame starts, in the map
        frame: (x, y, yaw) of the dog when the board came up. FAST-LIO starts
        wherever it was switched on; the map is the building's.

        `board_host` / `listen`: the board's address and this machine's on the
        link between them, when the brain is not on the board (HIL over
        USB-C: 192.168.55.1 and 192.168.55.100). Localhost otherwise."""
        from go2_board.link import BOARD_PORT, BRAIN_PORT, HOST, BrainLink
        self.link = BrainLink(port=port or BRAIN_PORT, board_port=board_port or BOARD_PORT,
                              listen=listen or HOST, board_host=board_host or HOST)
        self.origin = origin
        self.clock, self.sleep = clock, sleep
        self._t0 = clock()
        self._next = self._t0 + self.CONTROL_DT
        # The brain's own ticks: how many it handed over after the 20 Hz
        # deadline had passed, by how much, and what ran on them (step).
        self.tick_log = BrainTicks(self.CONTROL_DT)
        self._ret = None          # clock when the last step returned
        self._gap = True          # request() since: the brain stood, not ticked
        self._cmd = (0.0, 0.0, 0.0)
        self._state, self._got = None, None
        # (board clock, map x, y, yaw) of recent states: a cloud is placed
        # with the pose the dog had when the board finished it, not the pose
        # now. Placed with the pose now, a turning dog saw the trolley it was
        # passing walk across the corridor, and stopped to let it by.
        self._history = collections.deque(maxlen=60)

    # -- from the board ------------------------------------------------------

    def board_state(self):
        """The newest state message, or None. Never older than STATE_STALE_S."""
        s = self.link.poll()
        if s is not None and s is not self._state:
            self._state, self._got = s, self.clock()
            if s.get("x") is not None and s.get("t") is not None:
                self._history.append((s["t"],) + self._to_map(s))
        if self._got is None or self.clock() - self._got > STATE_STALE_S:
            return None
        return self._state

    def pose_at(self, t):
        """The map pose at board-clock time t, interpolated between the states
        either side of it (the nearest one outside them), or None."""
        h = self._history
        if not h:
            return None
        if t <= h[0][0]:
            return h[0][1:]
        for a, b in zip(h, list(h)[1:]):
            if a[0] <= t <= b[0]:
                f = (t - a[0]) / max(b[0] - a[0], 1e-9)
                dyaw = math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
                return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw)
        return h[-1][1:]

    def _to_map(self, s):
        x0, y0, yaw0 = self.origin
        c, k = math.cos(yaw0), math.sin(yaw0)
        return (x0 + c * s["x"] - k * s["y"], y0 + k * s["x"] + c * s["y"],
                math.atan2(math.sin(yaw0 + s["yaw"]), math.cos(yaw0 + s["yaw"])))

    def get_pose(self):
        s = self.board_state()
        if s is None or s.get("x") is None:
            raise BoardNotReady("no fresh pose from the board")
        x0, y0, yaw0 = self.origin
        c, k = math.cos(yaw0), math.sin(yaw0)
        x, y = s["x"], s["y"]
        return (x0 + c * x - k * y, y0 + k * x + c * y,
                math.atan2(math.sin(yaw0 + s["yaw"]), math.cos(yaw0 + s["yaw"])))

    def camera_pose(self):
        x, y, yaw = self.get_pose()
        return (x + self.CAM_FORWARD * math.cos(yaw),
                y + self.CAM_FORWARD * math.sin(yaw), yaw)

    def get_scan(self):
        raise NotImplementedError("B6: the LiDAR from the physics twin, or the Mid-360")

    def get_image(self):
        raise NotImplementedError("B6: the camera from the physics twin, or the dog's")

    def now(self):
        return self.clock() - self._t0

    # -- to the board --------------------------------------------------------

    def set_velocity(self, vx, vy=0.0, w=0.0):
        self._cmd = (max(-self.MAX_V, min(self.MAX_V, float(vx))),
                     max(-self.MAX_V, min(self.MAX_V, float(vy))),
                     max(-self.MAX_W, min(self.MAX_W, float(w))))

    def step(self):
        """Send this tick's command, then wait for the next 20 Hz tick.

        The first step after request() (the start, a lift ride) is a gap,
        not a late tick: the brain stood, and the schedule starts afresh."""
        now = self.clock()
        self.link.send(*self._cmd, mode="walk")
        if self._gap:
            self._gap = False
            self.tick_log.gap(0.0 if self._ret is None else now - self._ret)
            self._next = self.clock() + self.CONTROL_DT
        else:
            work_ms = None if self._ret is None else (now - self._ret) * 1000.0
            wait = self._next - self.clock()
            self.tick_log.tick(-wait * 1000.0, work_ms)
            if wait > 0:
                self.sleep(wait)
                self._next += self.CONTROL_DT
            else:
                self._next = self.clock() + self.CONTROL_DT
        self._ret = self.clock()

    def tick_summary(self):
        """The brain's ticks against the 20 Hz deadline: see BrainTicks."""
        return "\n".join(self.tick_log.summary())

    def request(self, mode, until=None, timeout=5.0):
        """Ask the board for a mode ("stand", "lie", "limp") at 20 Hz, with
        zero velocity, until its state says `until` (default: no wait)."""
        end = self.clock() + timeout
        self._gap = True                  # the next step() starts the schedule afresh
        while True:
            self.link.send(0.0, 0.0, 0.0, mode=mode)
            s = self.board_state()
            if until is None or (s is not None and s["mode"] == until):
                return s
            if self.clock() > end:
                got = None if self._state is None else self._state["mode"]
                raise BoardNotReady(f"asked {mode!r}, board still {got!r} after {timeout:.0f} s")
            self.sleep(self.CONTROL_DT)

    def stand(self, timeout=5.0):
        return self.request("stand", until="STAND", timeout=timeout)

    def lie(self, timeout=5.0):
        return self.request("lie", until="LIMP", timeout=timeout)

    def close(self):
        self.link.close()


class CameraLost(BoardStopped):
    """No fresh frame from the dog's camera: VAMOS would be shown an old
    view of a corridor people walk in. The run ends and says so."""
    layer = "camera"


# go2_world/camera.py's wire format (its tests and test_camera_link keep the two the same).
CAM_PORT = 9883
CAM_MAGIC = b"GCAM"
CAM_HEAD = struct.Struct("<4sIddI")     # magic, seq, world time s, sender wall time s, jpeg length


class CameraLink:
    """The dog's camera, as frames from go2_world over TCP (a USB camera on
    the dog): pixels only, never where anything is.

    A thread reads every frame as it comes and keeps the newest, still JPEG;
    it is decoded only when asked for (get), once per frame. A dropped link
    is reconnected every RETRY_S."""
    RETRY_S = 1.0

    def __init__(self, host, port=CAM_PORT, clock=time.monotonic):
        self.addr, self.clock = (host, port), clock
        self._lock = threading.Lock()
        self._newest = None            # (seq, arrived on this clock, jpeg)
        self._decoded = (None, None)   # (seq, image)
        self._stop = threading.Event()
        self._thread = None
        self.frames = self.connects = 0
        self.bytes = 0
        self.transit_ms = []           # arrival - sender's stamp: the clocks' offset included
        self.age_ms = []               # how old the frame was when get() handed it over

    def start(self):
        self._thread = threading.Thread(target=self._run, name="cameralink", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                with socket.create_connection(self.addr, timeout=2.0) as s:
                    s.settimeout(1.0)
                    self.connects += 1
                    while not self._stop.is_set():
                        magic, seq, _t, sent, n = CAM_HEAD.unpack(self._read(s, CAM_HEAD.size))
                        if magic != CAM_MAGIC:
                            raise OSError(f"not a camera frame: {magic!r}")
                        jpeg = self._read(s, n)
                        now = self.clock()
                        self.frames += 1
                        self.bytes += n
                        self.transit_ms.append((time.time() - sent) * 1000)
                        with self._lock:
                            self._newest = (seq, now, jpeg)
            except (OSError, struct.error):
                self._stop.wait(self.RETRY_S)

    def _read(self, s, n):
        out = bytearray()
        while len(out) < n:
            try:
                chunk = s.recv(n - len(out))
            except socket.timeout:
                if self._stop.is_set():
                    raise OSError("closing")
                continue
            if not chunk:
                raise OSError("the camera closed the link")
            out += chunk
        return bytes(out)

    def get(self):
        """(image HxWx3 uint8, its age in s), or (None, None) before any frame."""
        with self._lock:
            newest = self._newest
        if newest is None:
            return None, None
        seq, arrived, jpeg = newest
        if self._decoded[0] != seq:
            from PIL import Image
            self._decoded = (seq, np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")))
        age = self.clock() - arrived
        self.age_ms.append(age * 1000)
        return self._decoded[1], age

    def report(self):
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
        return (f"camera {self.addr[0]}:{self.addr[1]}: {self.frames} frames "
                f"({self.bytes / max(self.frames, 1) / 1024:.0f} KB each), {self.connects} connect(s), "
                f"transit median {med(self.transit_ms):.0f} ms (clock offset included), "
                f"age when used median {med(self.age_ms):.0f} ms max "
                f"{max(self.age_ms) if self.age_ms else float('nan'):.0f} ms")

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


# -- the board in run_building -------------------------------------------------

from cyberdog.sim.robot.mujoco_robot import MujocoRobot  # noqa: E402


class BoardTwinRobot(MujocoRobot):
    """go2_board drives; the twin follows it as a shadow.

    The pose comes from the board (go2_world's physics, or the dog). Every
    tick the twin is put there, so the camera, the virtual LiDAR, the people
    and the ground-truth checks run_building reads from model/data see the
    building from where the dog really is. Floor 1 only: the lift ride and
    the stairs are scripted moves (place() somewhere else), which a real
    board cannot do.
    """
    MAX_W = 1.0               # rad/s: go2_flat_v5 is trained to +-1
    REAL_TIME = True          # the board walks on the wall clock: never block the loop
    GUARDED = True            # its LiDAR guard has the last word: run_building works with it
    LIMP_HOLD = 4             # 20 Hz ticks of "limp" between two stands: 0.2 s
    # place() on the board is a pose to walk to, not one to be put at: the
    # scripted moves in and out of the lift car become velocity commands that
    # chase the moving target. Gains and limits for that.
    TRACK_K, TRACK_KW = 1.5, 2.0
    TRACK_V, TRACK_VY, TRACK_W = 0.5, 0.2, 0.8
    # The lift ride: arrived when the board's own height has settled this near
    # the new floor plus the standing height it rode at, for SETTLE_S.
    ARRIVED_M, SETTLE_S, RIDE_TIMEOUT_S = 0.05, 1.0, 60.0

    CLOUD_STALE_S = 0.5       # an older cloud is no cloud: the brain is blind, as the board is
    CAM_STALE_S = 1.0         # an older frame is no frame: the run ends (CameraLost)
    CAM_FIRST_S = 10.0        # the first frame: the world's camera may still be starting
    NO_STATE_S = 1.0          # no board state this long mid-walk ends the run (see _follow)

    def __init__(self, scene_xml, start_z=0.0, board=None, origin=(0.0, 0.0, 0.0),
                 board_host=None, listen=None, lift=None, clouds=True, camera_host=None):
        self.board = board or Go2BoardRobot(origin=origin, board_host=board_host, listen=listen)
        self._no_state_since = None    # board clock since when no state came, in _follow
        # The board's pose put back on the map (sensing/map_fix.py), or None:
        # run_building sets it, and tells it the floor's field each tick.
        self.map_fix = None
        self.lift = lift          # LiftLink, for rides; None: no floor but the first
        # The Mid-360 as the board sees it, forwarded on 9872: what the brain
        # perceives with, as on the dog. clouds=False keeps the twin's own
        # raycast (it sees only what the twin's scene has, never a real person).
        self.clouds = None
        if clouds:
            from go2_board.lidar import BRAIN_CLOUD_PORT, CloudLink
            from go2_board.link import HOST
            # Received in a thread: the brain reads clouds only while it
            # walks, and a lift ride unread overflowed the socket (CloudLink).
            self.clouds = CloudLink(port=BRAIN_CLOUD_PORT, host=listen or HOST).start()
            if self.clouds.rcvbuf_short:
                print(f"[udp] WARNING: cloud socket rcvbuf capped at "
                      f"{self.clouds.rcvbuf >> 10} KB: raise net.core.rmem_max")
        # The dog's camera from go2_world (people in it), or None: the twin's
        # own render, which has none.
        self.camera = CameraLink(camera_host).start() if camera_host else None
        self._camera_since = None
        x, y, yaw = self._wait_for_pose()
        super().__init__(scene_xml, start_xy=(x, y), start_yaw=yaw, start_z=start_z)

    def _wait_for_pose(self, timeout=5.0):
        end = self.board.clock() + timeout
        while True:
            try:
                return self.board.get_pose()
            except BoardNotReady:
                if self.board.clock() > end:
                    raise
                self.board.sleep(self.board.CONTROL_DT)

    def start(self, timeout=5.0, tries=3):
        """Stand, then walk in place: what the board needs before velocities.

        A stand clears the board's safety trip once, when it is asked; if the
        board trips again straight away (a stale state as it wakes), asking
        "stand" again is no new request. So go limp and ask afresh. The limp
        is held for LIMP_HOLD ticks: the board is limp already, and keeps only
        the newest datagram, so a single one followed at once by a stand
        would never be seen."""
        for k in range(tries):
            try:
                self.board.stand(timeout=timeout)
                break
            except BoardNotReady:
                if k == tries - 1:
                    raise
                for _ in range(self.LIMP_HOLD):
                    self.board.request("limp")
                    self.board.sleep(self.board.CONTROL_DT)
        self.board.request("walk", until="WALK", timeout=timeout)

    def set_velocity(self, vx, vy=0.0, w=0.0):
        super().set_velocity(vx, vy, w)
        self.board.set_velocity(self._vx, self._vy, self._w)

    def step(self):
        self.board.step()                    # send, then wait for the 20 Hz tick
        self._follow()

    def place(self, xy, yaw, z=None):
        """One tick walking toward (xy, yaw): the board cannot be put anywhere.
        A height other than the floor's is a scripted climb (stairs), refused;
        the lift goes through ride_lift()."""
        if z is not None and abs(z - self.z) > 1e-6:
            raise NotImplementedError("a scripted climb: the board rides the lift, see ride_lift")
        x, y, th = self.get_pose()
        ex, ey = xy[0] - x, xy[1] - y
        c, s = math.cos(th), math.sin(th)
        fwd, side = c * ex + s * ey, -s * ex + c * ey
        turn = math.atan2(math.sin(yaw - th), math.cos(yaw - th))
        clip = lambda v, m: max(-m, min(m, v))
        self.board.set_velocity(clip(self.TRACK_K * fwd, self.TRACK_V),
                                clip(self.TRACK_K * side, self.TRACK_VY),
                                clip(self.TRACK_KW * turn, self.TRACK_W))
        self.board.step()
        self._follow()

    def ride_lift(self, to_floor, floor_z, on_tick=lambda z: None):
        """Stand in the car while the lift carries the dog to `to_floor`.

        What the dog does on the real lift: stop walking, stand, and wait
        until it is on the new floor, judged by its own height (the board's
        pose; on the dog, FAST-LIO's), then walk again. The call is the
        person's (here the lift link, standing in for them). `on_tick(z)` is
        called every tick with the car floor's height, for the video."""
        if self.lift is None:
            raise NotImplementedError("no lift link: --robot go2_board needs go2_world's lift")
        if self.map_fix is not None:
            self.map_fix.use(None)             # another floor: its own correction, from none
        self.board.set_velocity(0.0, 0.0, 0.0)
        self.board.request("stand", until="STAND", timeout=5.0)
        s = self.board.board_state()
        stand_h = s["z"] - self.z              # trunk above the floor, standing
        self.lift.call(to_floor)
        goal = floor_z + stand_h
        start, settled = self.board.clock(), None
        while True:
            self.board.request("stand")        # one tick: the board hears from us at 20 Hz
            s = self.board.board_state()
            if s is None:
                raise BoardNotReady("no state from the board during the ride")
            if s["mode"] != "STAND":
                raise BoardStopped(f"the board is {s['mode']} in the lift"
                                   + (f": {s['tripped']}" if s.get("tripped") else ""))
            floor_now = s["z"] - stand_h
            x, y, th = self.board.get_pose()
            MujocoRobot.place(self, (x, y), th, z=floor_now)
            on_tick(floor_now)
            now = self.board.clock()
            if abs(s["z"] - goal) < self.ARRIVED_M:
                settled = settled or now
                if now - settled >= self.SETTLE_S:
                    break
            else:
                settled = None
            if now - start > self.RIDE_TIMEOUT_S:
                raise BoardStopped(f"the lift did not reach floor {to_floor} in "
                                   f"{self.RIDE_TIMEOUT_S:.0f} s (board at z {s['z']:.2f})")
            self.board.sleep(self.CONTROL_DT)
        self.z = floor_z
        self.board.request("walk", until="WALK", timeout=5.0)

    def _follow(self):
        """Put the twin where the board says the dog is, from ONE state read.

        It read the state twice (get_pose, then board_state) and the second
        read could find it gone stale in between: the board's loop stalled
        201 ms (HIL 11-11 15:52, dense LiDAR), board_state() returned None and
        the brain died on a TypeError mid-corridor. A gap shorter than
        NO_STATE_S is ridden out (the twin stays where it last was, the next
        tick asks again); a longer one ends the run, saying why."""
        s = self.board.board_state()
        if s is None or s.get("x") is None:
            now = self.board.clock()
            if self._no_state_since is None:
                self._no_state_since = now
            if now - self._no_state_since > self.NO_STATE_S:
                raise BoardStopped(f"no state from the board for more than {self.NO_STATE_S:.1f} s "
                                   "(its loop stalled, or it stopped)")
            return
        self._no_state_since = None
        x, y, yaw = self.board._to_map(s)
        if self.map_fix is not None:
            x, y, yaw = self.map_fix.pose(x, y, yaw)
        MujocoRobot.place(self, (x, y), yaw)
        if s["mode"] != "WALK":
            why = f": {s['tripped']}" if s.get("tripped") else ""
            raise BoardStopped(f"the board is {s['mode']}, not walking{why}")

    def now(self):
        return self.board.now()

    def board_clearance(self):
        """The board's own LiDAR guard, as it reports it: free distance
        (m) front, back, left, right, inf where it sees nothing, or None when
        it has no fresh frame (blind: it walks nowhere, only turns)."""
        s = self.board.board_state()
        lid = None if s is None else s.get("lidar")
        if not lid or lid.get("age") is None or lid["age"] > self.CLOUD_STALE_S:
            return None
        return {k: (math.inf if lid.get(k) is None else lid[k])
                for k in ("front", "back", "left", "right")}

    def get_scan(self):
        """The board's newest Mid-360 cloud in the map frame, (N, 3).

        The board sends it levelled, in the trunk frame, the floor at z = 0;
        turned by the dog's heading and set on the floor it is on. Empty when
        no cloud has come for CLOUD_STALE_S: blind, which perception treats as
        seeing nothing, while the board itself stops translation."""
        if self.clouds is None:
            return MujocoRobot.get_scan(self)
        c = self.clouds.poll()
        if c is None or self.board.clock() - self._cloud_seen(c) > self.CLOUD_STALE_S:
            return np.empty((0, 3))
        # The board's own map pose, never the twin's: that one is corrected.
        x, y, th = self.board.pose_at(c.t) or self.board.get_pose()
        if self.map_fix is not None:
            x, y, th = self.map_fix.pose(x, y, th)
        cs, sn = math.cos(th), math.sin(th)
        p = c.xyz.astype(float)
        out = np.column_stack((x + cs * p[:, 0] - sn * p[:, 1],
                               y + sn * p[:, 0] + cs * p[:, 1],
                               self.z + p[:, 2]))
        if self.map_fix is not None:
            out = self.map_fix.refine(out, (x, y))
        return out

    def get_image(self):
        """The dog's camera: the newest frame from go2_world (camera_host),
        or the twin's render without it. A frame older than CAM_STALE_S
        (or none CAM_FIRST_S after the first ask) ends the run: VAMOS must
        not plan on a stale view."""
        if self.camera is None:
            return MujocoRobot.get_image(self)
        now = self.board.clock()
        if self._camera_since is None:
            self._camera_since = now
        while True:
            img, age = self.camera.get()
            if img is not None and age <= self.CAM_STALE_S:
                return img
            if img is not None:
                raise CameraLost(f"the camera's newest frame is {age:.1f} s old "
                                 f"(more than {self.CAM_STALE_S:.1f} s)")
            if self.board.clock() - self._camera_since > self.CAM_FIRST_S:
                raise CameraLost(f"no frame from the camera at {self.camera.addr[0]}:"
                                 f"{self.camera.addr[1]} in {self.CAM_FIRST_S:.0f} s")
            time.sleep(0.05)

    def _cloud_seen(self, c):
        """When this cloud first arrived here, on this side's clock."""
        if c.seq != getattr(self, "_cloud_seq", None):
            self._cloud_seq, self._cloud_at = c.seq, self.board.clock()
        return self._cloud_at

    def close(self):
        # Which of the brain's sockets lost datagrams to a full buffer: the
        # board's kernel counts them all together ("receive buffer errors").
        from go2_board.livox import udp_drops
        k = udp_drops(self.board.link.sock)
        line = f"[udp] state udp {self.board.link.sock.getsockname()[1]}: kernel dropped " \
               f"{'n/a' if k is None else k} datagrams"
        if self.clouds is not None:
            line += "; " + self.clouds.report()
        print(line)
        if self.camera is not None:
            print(f"[camera] {self.camera.report()}")
            self.camera.close()
        self.board.close()
        if self.lift is not None:
            self.lift.close()
        if self.clouds is not None:
            self.clouds.close()
