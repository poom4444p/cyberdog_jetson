"""Stand the sim's Go2 up from the board, hold, lie down, go limp.

  python scripts/run_board.py [--iface lo0] [--hold 3]

The first end-to-end run of the board: SimBus, Safety and the Pose
controller, against unitree_mujoco (~/Work/unitree/go2_world/scripts/run_sim.py).
"""
import argparse
import time

from go2_board import go2
from go2_board.board import Board
from go2_board.controllers import Pose
from go2_board.sim_bus import SimBus

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default=None, help="DDS interface (default lo0 on macOS, lo on Linux)")
ap.add_argument("--hold", type=float, default=3.0, help="seconds standing")
args = ap.parse_args()

bus = SimBus(args.iface) if args.iface else SimBus()
bus.wait_for_state()
board = Board(bus)
t0 = time.monotonic()

for name, ctrl, seconds in [("stand up", Pose(go2.STAND, 1.5), 1.5),
                            ("hold", Pose(go2.STAND, 0.0), args.hold),
                            ("lie down", Pose(go2.LIE, 1.5), 1.5)]:
    print(f"t={time.monotonic() - t0:5.2f}s  {name}")
    board.run(ctrl, seconds=seconds)
    if board.safety.tripped:
        break

print(f"t={time.monotonic() - t0:5.2f}s  go limp")
board.go_limp()
s = board.stats
print(f"ticks {s['ticks']}, sent {s['sent']}, late {s['late']}, "
      f"safety: {board.safety.tripped or 'never tripped'}")
