"""A LiDAR-inertial dataset from a run: what a LIO (FAST-LIO, KISS-ICP, ...)
needs, and the pose to judge it against.

    board_main.py --record DIR     (HIL: LIO_RECORD=1 run_hil.sh route ...)

Writes DIR/chunk_NNNN.npz every CHUNK_S, so a board stopped by a signal loses
at most the last few seconds, and DIR/meta.json once. Each chunk:

  frame_stamp  (F,)  int64   ns, sensor clock: the frame's first packet
  frame_recv   (F,)  float64 board clock (s) when the frame was assembled
  frame_n      (F,)  int64   points in each frame; they are concatenated below
  xyz          (P,3) float32 m, SENSOR frame (not levelled: the LIO does that)
  t            (P,)  int64   ns, sensor clock, every point
  reflectivity (P,)  uint8
  imu_t        (I,)  int64   ns, sensor clock
  imu_recv     (I,)  float64 board clock (s) when received
  gyro         (I,3) float32 rad/s, sensor frame
  acc          (I,3) float32 m/s^2, sensor frame (specific force: +g up at rest)
  truth_t      (T,)  float64 board clock (s)
  truth_xyz    (T,3) float64 m, the bus's body position: go2_world's ground
                             truth in HIL; the real Go2 has none (that is
                             what the LIO is for), so these are empty there
  truth_quat   (T,4) float64 body orientation, w x y z (the bus's IMU)

The sensor clock (Livox, power-on) and the board clock are related through
frame_recv / imu_recv: received a few ms after the sensor stamped them.
"""
import json
import os
import threading
import time

import numpy as np

CHUNK_S = 5.0


class LioRecorder:
    def __init__(self, folder, clock=time.monotonic, meta=None, chunk_s=CHUNK_S):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.clock = clock
        self.chunk_s = chunk_s
        self._lock = threading.Lock()
        self._writer = None       # the thread saving the last chunk: never the LiDAR thread's
        self._n = 0
        self._started = None
        self._reset()
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(dict(meta or {}, chunk_s=chunk_s, started=time.time()), f, indent=1)

    def _reset(self):
        self._frames, self._imu, self._truth = [], [], []
        self._since = self.clock()

    def frames(self, frames, now):
        with self._lock:
            for f in frames:
                n = len(f.xyz)
                t = getattr(f, "t", None)                    # per point, ns; else the frame's
                r = getattr(f, "reflectivity", None)
                self._frames.append((
                    int(f.stamp), now, np.asarray(f.xyz, np.float32),
                    np.asarray(t if t is not None else np.full(n, f.stamp), np.int64),
                    np.asarray(r if r is not None else np.zeros(n), np.uint8)))
        self._maybe_write()

    def imu(self, samples, now):
        with self._lock:
            for s in samples:
                self._imu.append((int(s.t), now, np.asarray(s.gyro, np.float32),
                                  np.asarray(s.acc, np.float32)))
        self._maybe_write()

    def truth(self, t, pos, quat):
        if pos is None or quat is None:
            return
        with self._lock:
            self._truth.append((float(t), np.asarray(pos, float), np.asarray(quat, float)))

    def _maybe_write(self):
        if self.clock() - self._since >= self.chunk_s:
            self.flush(wait=False)

    def flush(self, wait=True):
        """Hand what is held to a writer thread. Saving a chunk takes a few
        hundred ms with the dense pattern, and on the LiDAR thread that is
        the guard blind (STALE_S 0.3) and the dog stopped."""
        with self._lock:
            fr, im, tr = self._frames, self._imu, self._truth
            self._reset()
        if self._writer is not None:
            self._writer.join()                  # one at a time, in order
        if fr or im or tr:
            path = os.path.join(self.folder, f"chunk_{self._n:04d}.npz")
            self._n += 1
            self._writer = threading.Thread(target=self._save, args=(path, fr, im, tr),
                                            name="lio-record", daemon=True)
            self._writer.start()
        if wait and self._writer is not None:
            self._writer.join()

    @staticmethod
    def _save(path, fr, im, tr):
        np.savez(
            path,
            frame_stamp=np.array([f[0] for f in fr], np.int64),
            frame_recv=np.array([f[1] for f in fr], float),
            frame_n=np.array([len(f[2]) for f in fr], np.int64),
            xyz=np.concatenate([f[2] for f in fr]) if fr else np.empty((0, 3), np.float32),
            t=np.concatenate([f[3] for f in fr]) if fr else np.empty(0, np.int64),
            reflectivity=np.concatenate([f[4] for f in fr]) if fr else np.empty(0, np.uint8),
            imu_t=np.array([s[0] for s in im], np.int64),
            imu_recv=np.array([s[1] for s in im], float),
            gyro=np.array([s[2] for s in im], np.float32).reshape(-1, 3),
            acc=np.array([s[3] for s in im], np.float32).reshape(-1, 3),
            truth_t=np.array([s[0] for s in tr], float),
            truth_xyz=np.array([s[1] for s in tr], float).reshape(-1, 3),
            truth_quat=np.array([s[2] for s in tr], float).reshape(-1, 4))

    close = flush


def load(folder):
    """Every chunk of a recording, concatenated: one dict of arrays (frames'
    points stay concatenated; frame_n splits them), plus "meta"."""
    names = sorted(n for n in os.listdir(folder) if n.startswith("chunk_") and n.endswith(".npz"))
    parts = [np.load(os.path.join(folder, n)) for n in names]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files} if parts else {}
    with open(os.path.join(folder, "meta.json")) as f:
        out["meta"] = json.load(f)
    return out
