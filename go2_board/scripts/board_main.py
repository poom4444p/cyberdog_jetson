"""The board, running: SimBus, safety, modes, and the brain link on localhost.

  python scripts/board_main.py [--iface lo0] [--seconds N] [--no-lidar] [--policy DIR]

Starts limp. Listens for the Mid-360 (real, or go2_world's run_lidar.py) on
56301/56401, limits walking by what it sees, and forwards clouds to the brain
on 127.0.0.1:9872. Drive it with scripts/brain_cli.py (or the Cyberdog brain).
With --policy (a folder go2_train exported, e.g. policies/go2_flat_v1), WALK
runs that policy; without one, "walk" is refused.
"""
import argparse
import signal
import sys

from go2_board.app import App
from go2_board.link import BRAIN_PORT, HOST, BoardLink
from go2_board.lidar import BRAIN_CLOUD_PORT
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
ap.add_argument("--record", default=None, metavar="DIR",
                help="save the raw Mid-360 frames, its IMU and the bus's pose to DIR "
                     "(go2_board.record: a dataset to test a LIO against)")
ap.add_argument("--stale-ms", type=float, default=None,
                help="SIM ONLY: go limp after this long without a LowState (default %.0f ms). "
                     "A Mac running the sim, VAMOS and the brain pauses for longer now and then; "
                     "the real dog's motors do not" % (STALE_S * 1e3))
ap.add_argument("--listen", default=HOST,
                help="address to take the brain's commands on (default %(default)s). For a brain on "
                     "another machine, this board's address on that one link, e.g. 192.168.55.1 "
                     "over USB-C: never 0.0.0.0, or anyone on the LAN can drive the dog")
ap.add_argument("--brain", default=HOST,
                help="where the brain listens for state and clouds (default %(default)s)")
args = ap.parse_args()
if args.listen in ("0.0.0.0", ""):
    ap.error("--listen 0.0.0.0 would take commands from every network the board is on")

walker = None
if args.policy:
    walker = PolicyWalker(args.policy)      # refuses a policy that fails its samples
    print(f"[board] walking policy {walker.name}: {walker.decimation}-tick period, "
          f"samples match to {walker.worst:.1e}")

bus = SimBus(args.iface) if args.iface else SimBus()
bus.wait_for_state()
lidar = None if args.no_lidar else Lidar(brain=(args.brain, BRAIN_CLOUD_PORT))
if lidar is not None and args.record:
    from go2_board import lidar as lidar_mod
    from go2_board.record import LioRecorder
    lidar.recorder = LioRecorder(args.record, clock=lidar.clock, meta={
        "mount_pitch_deg": lidar_mod.MOUNT_PITCH, "mount_x": lidar_mod.MOUNT_X,
        "sensor_height": lidar_mod.SENSOR_HEIGHT, "board_clock": "time.monotonic"})
    print(f"[board] recording Mid-360 frames, IMU and the bus pose to {args.record}")
if lidar is not None:
    lidar.start()
stale_s = STALE_S if args.stale_ms is None else args.stale_ms / 1e3
if stale_s != STALE_S:
    print(f"[board] WARNING: stale-state limit {stale_s * 1e3:.0f} ms, not {STALE_S * 1e3:.0f}: "
          "for a simulator only, never on the dog")
board = Board(bus, stale_s=stale_s)
link = BoardLink(clock=board.clock, listen=args.listen, brain_host=args.brain)
app = App(bus, link=link, lidar=lidar, walker=walker, board=board)
print(f"[board] up, LIMP; waiting for the brain on udp {args.listen}:9870, state to "
      f"{args.brain}:{BRAIN_PORT}"
      + ("" if lidar is None else f"; Mid-360 on udp 56301, clouds to {args.brain}:{BRAIN_CLOUD_PORT}"))
# `onboard.sh stop` is a SIGTERM: exit through the finally below (limp, the
# LiDAR stopped, a recording's last chunk written), not straight out of it.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
try:
    app.run(args.seconds)
finally:
    app.board.go_limp()
    if lidar is not None:
        lidar.stop()
        if lidar.recorder is not None:
            lidar.recorder.close()
    s = app.board.stats
    print(f"[board] stopped: ticks {s['ticks']}, sent {s['sent']}, mode {app.modes.mode}, "
          f"safety {app.board.safety.tripped or 'never tripped'}")
