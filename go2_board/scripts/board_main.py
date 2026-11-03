"""The board, running: SimBus, safety, modes, and the brain link on localhost.

  python scripts/board_main.py [--iface lo0] [--seconds N] [--no-lidar] [--policy DIR]

Starts limp. Listens for the Mid-360 (real, or go2_world's run_lidar.py) on
56301/56401, limits walking by what it sees, and forwards clouds to the brain
on 127.0.0.1:9872. Drive it with scripts/brain_cli.py (or the Cyberdog brain).
With --policy (a folder go2_train exported, e.g. policies/go2_flat_v1), WALK
runs that policy; without one, "walk" is refused.
"""
import argparse

from go2_board.app import App
from go2_board.board import Board
from go2_board.safety import STALE_S
from go2_board.lidar import Lidar
from go2_board.sim_bus import SimBus
from go2_board.walker import PolicyWalker

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default=None, help="DDS interface (default lo0 on macOS, lo on Linux)")
ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: run)")
ap.add_argument("--no-lidar", action="store_true", help="no Mid-360: velocity is not limited")
ap.add_argument("--policy", default=None, help="walking policy folder (policy_spec.json, policy.npz)")
ap.add_argument("--stale-ms", type=float, default=None,
                help="SIM ONLY: go limp after this long without a LowState (default %.0f ms). "
                     "A Mac running the sim, VAMOS and the brain pauses for longer now and then; "
                     "the real dog's motors do not" % (STALE_S * 1e3))
args = ap.parse_args()

walker = None
if args.policy:
    walker = PolicyWalker(args.policy)      # refuses a policy that fails its samples
    print(f"[board] walking policy {walker.name}: {walker.decimation}-tick period, "
          f"samples match to {walker.worst:.1e}")

bus = SimBus(args.iface) if args.iface else SimBus()
bus.wait_for_state()
lidar = None if args.no_lidar else Lidar().start()
stale_s = STALE_S if args.stale_ms is None else args.stale_ms / 1e3
if stale_s != STALE_S:
    print(f"[board] WARNING: stale-state limit {stale_s * 1e3:.0f} ms, not {STALE_S * 1e3:.0f}: "
          "for a simulator only, never on the dog")
app = App(bus, lidar=lidar, walker=walker, board=Board(bus, stale_s=stale_s))
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
