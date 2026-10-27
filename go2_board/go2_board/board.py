"""The board loop: read the motors, ask the controller, filter, write. 500 Hz.

One loop, one thread, so a command is never computed from a state another
thread is halfway through replacing. Anything slow (the brain, VAMOS) runs
elsewhere and hands this loop a target; nothing here waits on it.
"""
import time

from go2_board.bus import MotorCmd
from go2_board.safety import Safety

RATE_HZ = 500
LATE_S = 0.01             # a tick this far behind schedule is counted as late


class Board:
    def __init__(self, bus, rate_hz=RATE_HZ, clock=time.monotonic, sleep=time.sleep):
        self.bus = bus
        self.dt = 1.0 / rate_hz
        self.clock, self.sleep = clock, sleep
        self.safety = Safety()
        self.stats = {"ticks": 0, "late": 0, "sent": 0}

    def tick(self, controller):
        """One read-filter-write. Returns the command sent, or None."""
        now = self.clock()
        state = self.bus.read()
        cmd = None
        if state is not None:
            try:
                cmd = controller.step(state, now)
            except Exception as e:          # a crash must not stop the motors' loop
                print(f"[board] controller raised {type(e).__name__}: {e}")
        was = self.safety.tripped
        out = self.safety.filter(cmd, state, now)
        if self.safety.tripped and not was:
            print(f"[board] DAMPING: {self.safety.tripped}")
        if out is not None:
            self.bus.write(out)
            self.stats["sent"] += 1
        self.stats["ticks"] += 1
        return out

    def run(self, controller, seconds=None, until=None):
        """Tick on a fixed schedule for `seconds`, or until until(t) is true.
        A late tick restarts the schedule rather than firing the missed ones
        back to back."""
        t0 = nxt = self.clock()
        while True:
            t = self.clock()
            if (seconds is not None and t - t0 >= seconds) or (until and until(t)):
                return
            self.tick(controller)
            nxt += self.dt
            wait = nxt - self.clock()
            if wait > 0:
                self.sleep(wait)
            elif -wait > LATE_S:
                self.stats["late"] += 1
                nxt = self.clock()

    def go_limp(self, seconds=0.5):
        """Damping for a moment before letting go of the bus, so the legs are
        not left holding the last command."""
        state = self.bus.read()
        if state is None:
            return
        end = self.clock() + seconds
        while self.clock() < end:
            state = self.bus.read()
            self.bus.write(MotorCmd.damping(state.q))
            self.sleep(self.dt)
