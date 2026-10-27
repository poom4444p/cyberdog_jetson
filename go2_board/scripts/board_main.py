"""The board, running: SimBus, safety, modes, and the brain link on localhost.

  python scripts/board_main.py [--iface lo0] [--seconds N] [--no-lidar]

Starts limp. Listens for the Mid-360 (real, or go2_world's run_lidar.py) on
56301/56401, limits walking by what it sees, and forwards clouds to the brain
on 127.0.0.1:9872. Drive it with scripts/brain_cli.py (or the Cyberdog brain).
"""
import argparse

from go2_board.app import App
from go2_board.lidar import Lidar
from go2_board.sim_bus import SimBus

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default=None, help="DDS interface (default lo0 on macOS, lo on Linux)")
ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: run)")
ap.add_argument("--no-lidar", action="store_true", help="no Mid-360: velocity is not limited")
args = ap.parse_args()

bus = SimBus(args.iface) if args.iface else SimBus()
bus.wait_for_state()
lidar = None if args.no_lidar else Lidar().start()
app = App(bus, lidar=lidar)
print("[board] up, LIMP; waiting for the brain on udp 127.0.0.1:9870"
      + ("" if lidar is None else "; Mid-360 on udp 56301, clouds to 9872"))
try:
    app.run(args.seconds)
finally:
    app.board.go_limp()
    if lidar is not None:
        lidar.stop()
    s = app.board.stats
    print(f"[board] stopped: ticks {s['ticks']}, sent {s['sent']}, mode {app.modes.mode}, "
          f"safety {app.board.safety.tripped or 'never tripped'}")
