"""LidarProcess: the Mid-360 in its own process, answering App as a Lidar does."""
import socket
import time

import numpy as np

from go2_board import lidar as L
from go2_board.lidar_proc import LidarProcess
from test_livox import MS, cloud, free_port, raw


def sweep(cnt, start_ms, n=100):
    """n packets 1 ms apart: a whole 100 ms sweep (lidar.whole), points 1 m ahead."""
    return [raw(1, cloud(), 96, i, cnt, (start_ms + i) * MS, interval=10_000) for i in range(n)]


def test_a_sweep_in_the_child_reaches_the_board_as_a_clearance(tmp_path):
    pp, ip = free_port(), free_port()
    lp = LidarProcess(brain=None, record=str(tmp_path / "rec"),
                      rx_args={"host": "127.0.0.1", "point_port": pp, "imu_port": ip}).start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert lp.rcvbuf > 0 and lp.clearance() is None             # up; nothing seen yet
        for d in sweep(0, 0) + sweep(1, 100) + sweep(2, 200)[:1]:  # the third's first packet closes the second
            tx.sendto(d, ("127.0.0.1", pp))
            time.sleep(0.0002)
        end = time.monotonic() + 5.0
        while lp.clearance() is None and time.monotonic() < end:
            time.sleep(0.02)
        c = lp.clearance()
        assert c is not None and c.points >= 0
        assert abs(c.stamp - time.monotonic()) < 5.0                 # one clock for both processes
        lp.commanded(0.3, 0.0, time.monotonic())                    # no error, no wait
        lp.truth(time.monotonic(), np.array([1.0, 2.0, 0.3]), np.array([1.0, 0, 0, 0]))
    finally:
        tx.close()
        final = lp.stop()
    assert final["used"] >= 1 and final["receiver"]["packets"] >= 150   # the last few may be unread at stop
    assert not lp.proc.is_alive()
    assert (tmp_path / "rec" / "meta.json").exists()                # the recorder ran in the child


def test_stop_without_a_frame_still_reports():
    lp = LidarProcess(brain=None, rx_args={"host": "127.0.0.1", "point_port": free_port(),
                                           "imu_port": free_port()}).start()
    final = lp.stop()
    assert final["used"] == 0 and final["partial"] == 0 and not lp.proc.is_alive()


def test_board_main_runs_only_as_a_script():
    # LidarProcess's child (spawn) imports the parent's script again: top-level
    # code there ran a second board_main in the child (HIL 11-11 16:02).
    import ast
    import pathlib
    src = (pathlib.Path(__file__).parents[1] / "scripts" / "board_main.py").read_text()
    top = [n for n in ast.parse(src).body
           if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Expr))]
    assert len(top) == 1 and isinstance(top[0], ast.If)       # only `if __name__ == "__main__":`
