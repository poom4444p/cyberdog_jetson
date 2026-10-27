"""Stand in for the brain: send a scripted sequence of modes, print the state.

  python scripts/brain_cli.py stand:3 walk:1 lie:2

Each step is mode:seconds, sent at 20 Hz with zero velocity. The state line
the board sends back is printed twice a second.
"""
import sys
import time

from go2_board.link import BrainLink

steps = [(s.split(":")[0], float(s.split(":")[1])) for s in sys.argv[1:]] or [("stand", 3), ("lie", 2)]
brain = BrainLink()
t0 = time.monotonic()
for mode, seconds in steps:
    end = time.monotonic() + seconds
    shown = 0.0
    while time.monotonic() < end:
        brain.send(0.0, 0.0, 0.0, mode)
        st = brain.poll()
        t = time.monotonic() - t0
        if st and t - shown >= 0.5:
            shown = t
            z = "-" if st["z"] is None else f"{st['z']:.3f}"
            print(f"t={t:5.2f}s  sent {mode:5s}  board {st['mode']:8s}  z={z} m  "
                  f"tripped={st['tripped']}  walker={st['walker']}")
        time.sleep(0.05)
brain.close()
