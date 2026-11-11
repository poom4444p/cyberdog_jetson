"""go2_board.record: the LIO dataset a run leaves behind (board_main --record)."""
import json

import numpy as np

from go2_board.lidar import Lidar
from go2_board.livox import Frame, ImuSample
from go2_board.record import LioRecorder, load


class Clock:
    def __init__(self, t=10.0):
        self.t = t

    def __call__(self):
        return self.t


def frame(n, stamp, t0=None):
    xyz = np.arange(n * 3, dtype=float).reshape(n, 3)
    t = None if t0 is None else np.arange(n, dtype=np.int64) * 1000 + t0
    return Frame(xyz, np.full(n, 7, np.uint8), np.zeros(n, np.uint8), t, stamp, 0, 1, 0)


def imu(t):
    return ImuSample(t, np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 9.81]))


def test_frames_imu_and_truth_round_trip(tmp_path):
    clock = Clock()
    rec = LioRecorder(str(tmp_path), clock=clock, meta={"mount_pitch_deg": 20.0})
    rec.frames([frame(3, 1_000, t0=1_000), frame(2, 2_000, t0=2_000)], now=10.1)
    rec.imu([imu(1_500), imu(2_500)], now=10.2)
    rec.truth(10.3, np.array([1.0, 2.0, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
    rec.close()
    d = load(str(tmp_path))
    assert list(d["frame_n"]) == [3, 2] and list(d["frame_stamp"]) == [1_000, 2_000]
    assert d["xyz"].shape == (5, 3) and d["xyz"].dtype == np.float32
    assert list(d["t"]) == [1_000, 2_000, 3_000, 2_000, 3_000]          # every point's own time
    assert list(d["reflectivity"]) == [7] * 5
    assert list(d["imu_t"]) == [1_500, 2_500] and d["acc"][0, 2] == np.float32(9.81)
    assert list(d["truth_t"]) == [10.3] and list(d["truth_xyz"][0]) == [1.0, 2.0, 0.3]
    assert d["meta"]["mount_pitch_deg"] == 20.0


def test_a_frame_without_point_times_gets_its_own(tmp_path):
    rec = LioRecorder(str(tmp_path), clock=Clock())
    rec.frames([frame(2, 5_000)], now=10.0)
    rec.close()
    assert list(load(str(tmp_path))["t"]) == [5_000, 5_000]


def test_chunks_every_chunk_s_and_load_joins_them(tmp_path):
    clock = Clock()
    rec = LioRecorder(str(tmp_path), clock=clock, chunk_s=1.0)
    for k in range(5):                                     # 5 frames, 0.5 s apart
        rec.frames([frame(1, k)], now=clock.t)
        clock.t += 0.5
    rec.close()
    chunks = sorted(p.name for p in tmp_path.glob("chunk_*.npz"))
    assert chunks == ["chunk_0000.npz", "chunk_0001.npz"]  # after the frames at 1.0 s and 2.0 s
    assert list(load(str(tmp_path))["frame_stamp"]) == [0, 1, 2, 3, 4]


def test_nothing_held_writes_no_chunk(tmp_path):
    rec = LioRecorder(str(tmp_path), clock=Clock())
    rec.close()
    assert not list(tmp_path.glob("chunk_*.npz"))
    assert json.loads((tmp_path / "meta.json").read_text())["chunk_s"] == 5.0


def test_lidar_hands_the_raw_frames_and_imu_to_the_recorder(tmp_path):
    li = Lidar(clock=Clock(42.0), brain=None)
    li.recorder = LioRecorder(str(tmp_path), clock=li.clock)
    f = frame(4, 9_000, t0=9_000)
    li.process([f], [imu(8_000)])
    li.recorder.close()
    d = load(str(tmp_path))
    assert np.array_equal(d["xyz"], f.xyz.astype(np.float32))        # sensor frame, not levelled
    assert list(d["frame_recv"]) == [42.0] and list(d["imu_recv"]) == [42.0]


def test_app_records_the_bus_pose_with_each_state_report(tmp_path):
    from test_app import app
    a, inbox, clock = app()

    class WithRecorder:
        recorder = LioRecorder(str(tmp_path), clock=clock)

        def clearance(self):
            return None

    a.lidar = WithRecorder()
    a.report()
    a.lidar.recorder.close()
    d = load(str(tmp_path))
    assert list(d["truth_xyz"][0]) == [1.0, 2.0, 0.3] and list(d["truth_t"]) == [clock.t]
