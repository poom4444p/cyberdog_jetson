"""The Mid-360 in its own process: what board_main runs by default.

The Lidar (lidar.py) used to run as a thread inside board_main, next to the
500 Hz motor loop, and Python runs one thread at a time. With the sparse test
pattern that cost nothing; with a real unit's stream -- ~2,100 packets a
second to parse, 20,000 points a frame to level and judge -- the motor loop
waited behind it: 201 ms with the recorder, 423 ms without (lowcmd max, HIL
11-11 15:52 / 15:57, dense), the board's own state went stale and the dog
stood at the trolley. A separate process has its own interpreter: the motor
loop never waits on the LiDAR again. It is how a robot is built anyway -- the
sensor driver is not the motor controller.

LidarProcess offers what App asks of a Lidar -- clearance(), commanded(),
truth(), start(), stop() -- over shared memory:

  clearance  the child writes front, back, left, right, stamp, points and a
             sequence number after every whole sweep; the board reads them
             without waiting (a lock held for six doubles)
  commanded  the board writes the velocity it gives the walker; the child
             passes it to Lidar.commanded before each batch (levelling's hold)
  truth      the bus pose for a recording, through a queue that drops, never
             blocks, when full

Stamps are time.monotonic(), one clock for every process on the machine, so
the guard's age check (STALE_S) means the same as it did in one process.
"""
import math
import multiprocessing as mp
import queue
import time

from go2_board import lidar as L

# clearance block: front, back, left, right, stamp, points, seq
_F, _B, _L, _R, _STAMP, _PTS, _SEQ = range(7)
# odometry block: x, y, yaw (odometry frame), stamp, seq, ms a sweep
_OX, _OY, _OYAW, _OSTAMP, _OSEQ, _OMS = range(6)
INF = 1e300                    # inf in a shared double, read back as math.inf


def _run(brain, record, meta, mount_pitch, rx_args, odometry, clear, odo, cmd, truth_q,
         stats_q, stop):
    """The child: a Lidar of its own, its receiver, (with `record`) the recorder
    and (with `odometry`) the LIO (go2_board.odometry)."""
    from go2_board.livox import LivoxReceiver
    li = L.Lidar(receiver=LivoxReceiver(**rx_args), brain=brain, mount_pitch=mount_pitch)
    if record:
        from go2_board.record import LioRecorder
        li.recorder = LioRecorder(record, clock=li.clock, meta=meta)
    lio = None
    if odometry:
        from go2_board.odometry import LioOdometry
        lio = LioOdometry(mount_pitch)
    stats_q.put({"rcvbuf": li.rx.rcvbuf, "rcvbuf_short": li.rx.rcvbuf_short,
                 "odometry": None if lio is None else lio.available})
    if lio is not None and not lio.available:
        lio = None
    last_cmd_t, seq, oseq, ms = None, 0, 0, None
    while not stop.is_set():
        with cmd.get_lock():
            vx, vy, t = cmd[0], cmd[1], cmd[2]
        if t != last_cmd_t:
            last_cmd_t = t
            li.commanded(vx, vy, t)
        if li.recorder is not None:
            try:
                while True:
                    li.recorder.truth(*truth_q.get_nowait())
            except queue.Empty:
                pass
        frames, samples = li.rx.poll()
        if not (frames or samples):
            time.sleep(0.002)
            continue
        if lio is not None:
            lio.imu(samples)
            for f in frames:
                if L.whole(f):
                    t0 = time.perf_counter()
                    P = lio.add(f.xyz, f.t)
                    dt = 1000 * (time.perf_counter() - t0)
                    ms = dt if ms is None else 0.9 * ms + 0.1 * dt
                    oseq += 1
                    with odo.get_lock():
                        odo[_OX], odo[_OY] = P[0, 3], P[1, 3]
                        odo[_OYAW] = math.atan2(P[1, 0], P[0, 0])
                        odo[_OSTAMP], odo[_OSEQ], odo[_OMS] = li.clock(), oseq, ms
        before = li.latest
        li.process(frames, samples)
        c = li.latest
        if c is not None and c is not before:
            seq += 1
            with clear.get_lock():
                clear[_F], clear[_B], clear[_L], clear[_R] = (
                    min(v, INF) for v in (c.front, c.back, c.left, c.right))
                clear[_STAMP], clear[_PTS], clear[_SEQ] = c.stamp, c.points, seq
    if li.recorder is not None:
        li.recorder.close()
    stats_q.put({"used": li.used, "partial": li.partial, "receiver": li.rx.stats(),
                 "odometry_sweeps": oseq, "odometry_ms": ms})
    li.rx.close()
    if li.sock is not None:
        li.sock.close()


class LidarProcess:
    """A Lidar in a child process, offering App what a Lidar does."""

    def __init__(self, brain=(L.HOST, L.BRAIN_CLOUD_PORT), record=None, meta=None,
                 mount_pitch=L.MOUNT_PITCH, rx_args=None, odometry=False):
        """`rx_args`: LivoxReceiver's host / point_port / imu_port, for tests;
        by default every interface on the sensor's ports, as on the dog."""
        ctx = mp.get_context("spawn")          # a fresh interpreter: no DDS threads forked
        self._clear = ctx.Array("d", 7)
        self._odo = ctx.Array("d", 6)
        self._cmd = ctx.Array("d", [0.0, 0.0, -1.0])
        self._truth = ctx.Queue(maxsize=500)
        self._stats = ctx.Queue()
        self._stop = ctx.Event()
        self.proc = ctx.Process(target=_run, name="lidar", daemon=True,
                                args=(brain, record, meta, mount_pitch, dict(rx_args or {}),
                                      odometry, self._clear, self._odo,
                                      self._cmd, self._truth, self._stats, self._stop))
        self.recording = bool(record)
        self.rcvbuf = self.rcvbuf_short = None
        self.final = None                      # the child's counts, after stop()

    def start(self, timeout=10.0):
        self.proc.start()
        first = self._stats.get(timeout=timeout)   # up, its socket open
        self.rcvbuf, self.rcvbuf_short = first["rcvbuf"], first["rcvbuf_short"]
        self.odometry_available = first["odometry"]       # None: not asked for
        return self

    def odometry(self):
        """(x, y, yaw, stamp, ms a sweep) of the body in the odometry's frame, or None."""
        with self._odo.get_lock():
            v = list(self._odo)
        return None if v[_OSEQ] == 0 else (v[_OX], v[_OY], v[_OYAW], v[_OSTAMP], v[_OMS])

    def clearance(self):
        with self._clear.get_lock():
            v = list(self._clear)
        if v[_SEQ] == 0:
            return None
        f, b, l, r = (math.inf if x >= INF else x for x in v[:4])
        return L.Clearance(f, b, l, r, v[_STAMP], int(v[_PTS]))

    def commanded(self, vx, vy, t):
        with self._cmd.get_lock():
            self._cmd[0], self._cmd[1], self._cmd[2] = vx, vy, t

    def truth(self, t, pos, quat):
        if not self.recording or pos is None or quat is None:
            return
        try:
            self._truth.put_nowait((t, tuple(map(float, pos)), tuple(map(float, quat))))
        except queue.Full:
            pass

    def stop(self, timeout=5.0):
        self._stop.set()
        try:
            self.final = self._stats.get(timeout=timeout)
        except queue.Empty:
            self.final = None
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
        return self.final
