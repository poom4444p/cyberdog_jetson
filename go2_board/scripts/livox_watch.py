"""Listen for a Mid-360 (real or go2_world's) and print what arrives, once a second.

  python scripts/livox_watch.py [--seconds 10]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_board.livox import LivoxReceiver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=float("inf"))
    a = ap.parse_args()

    rx = LivoxReceiver()
    end = time.monotonic() + a.seconds
    nxt = time.monotonic() + 1.0
    frames = imu = 0
    last = None
    try:
        while time.monotonic() < end:
            fs, ss = rx.poll()
            frames += len(fs)
            imu += len(ss)
            if fs:
                last = fs[-1]
            if time.monotonic() >= nxt:
                s = rx.stats()
                line = f"{frames:2d} frames/s  {imu:3d} imu/s  bad {s['bad']}  lost {s['lost']}  late {s['late']}"
                if last is not None and len(last.xyz):
                    r = np.linalg.norm(last.xyz, axis=1)
                    line += f"  | {len(last.xyz)} pts, {last.packets} pkts, nearest {r.min():.2f} m"
                if s["last_error"]:
                    line += f"  | last error: {s['last_error']}"
                print(line, flush=True)
                frames = imu = 0
                nxt += 1.0
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    print(rx.stats())


if __name__ == "__main__":
    main()
