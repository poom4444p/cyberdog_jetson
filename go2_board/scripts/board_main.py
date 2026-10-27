"""The board, running: SimBus, safety, modes, and the brain link on localhost.

  python scripts/board_main.py [--iface lo0] [--seconds N]

Starts limp. Drive it with scripts/brain_cli.py (or the Cyberdog brain).
"""
import argparse

from go2_board.app import App
from go2_board.sim_bus import SimBus

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default=None, help="DDS interface (default lo0 on macOS, lo on Linux)")
ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: run)")
args = ap.parse_args()

bus = SimBus(args.iface) if args.iface else SimBus()
bus.wait_for_state()
app = App(bus)
print("[board] up, LIMP; waiting for the brain on udp 127.0.0.1:9870")
try:
    app.run(args.seconds)
finally:
    app.board.go_limp()
    s = app.board.stats
    print(f"[board] stopped: ticks {s['ticks']}, sent {s['sent']}, mode {app.modes.mode}, "
          f"safety {app.board.safety.tripped or 'never tripped'}")
