"""Stand in for the brain: send a scripted sequence of modes, print the state.

  python scripts/brain_cli.py stand:3 walk:1 lie:2
  python scripts/brain_cli.py stand:3 walk:2 walk:5:0.5,0,0 walk:4:0,0,0.8 stand:2 lie:2

Each step is mode:seconds[:vx,vy,w], sent at 20 Hz; the velocity is zero
unless given. The state line the board sends back is printed twice a second.
"""
import math
import sys
import time

from go2_board.link import BrainLink



def parse(step):
    parts = step.split(":")
    v = tuple(float(x) for x in parts[2].split(",")) if len(parts) > 2 else (0.0, 0.0, 0.0)
    if len(v) != 3:
        sys.exit(f"{step}: velocity is vx,vy,w")
    return parts[0], float(parts[1]), v


steps = [parse(s) for s in sys.argv[1:]] or [("stand", 3, (0.0, 0.0, 0.0)), ("lie", 2, (0.0, 0.0, 0.0))]
brain = BrainLink()
t0 = time.monotonic()
for mode, seconds, (vx, vy, w) in steps:
    end = time.monotonic() + seconds
    shown = 0.0
    while time.monotonic() < end:
        brain.send(vx, vy, w, mode)
        st = brain.poll()
        t = time.monotonic() - t0
        if st and t - shown >= 0.5:
            shown = t
            z = "-" if st["z"] is None else f"{st['z']:.3f}"
            xy = "-" if st["x"] is None else f"{st['x']:+.2f},{st['y']:+.2f}"
            yaw = "-" if st["yaw"] is None else f"{math.degrees(st['yaw']):+4.0f}"
            print(f"t={t:5.2f}s  sent {mode:5s} {vx:+.1f},{vy:+.1f},{w:+.1f}  board {st['mode']:8s}  "
                  f"xy={xy} m  z={z} m  yaw={yaw} deg  tripped={st['tripped']}  walker={st['walker']}")
        time.sleep(0.05)
brain.close()
