"""The twin walked the way the board walks the dog: `--robot twin-board`.

MujocoRobot does exactly what it is told, at once, and nothing stops it. The
dog on go2_board does neither:

  - it turns late and short. Fitted on 57 HIL routes (11-09..11-11, 605
    windows of 2 s with turning; brain.log CMD lines against tracks.jsonl
    poses): about 0.1 s of smoothing after a 0.15 s delay, and a gain that
    depends on how fast it is walking, not on how hard it is asked to turn --
    a pivot on the spot gets about a fifth of the turn it asks for, walking
    at 0.6+ m/s about 0.70. The twin's 1.0 everywhere was 5.8 deg rms off over
    2 s; this, 2.6. Forward it walks what it is told.
  - its LiDAR guard slows, then stops, motion towards anything within
    SLOW_M..STOP_M of the body (go2_board.lidar: the same code, run here on
    the twin's own Mid-360 scan), and run_building works with it as it does
    on the board (respect_board, step_aside: GUARDED).

The fit is on the brain's clock, and the Mac's world runs ~0.83x real time,
so the gains below are the fitted ones / 0.83: the policy's own, which is what
a real Go2 would show. `clock` puts the HIL rig's slowdown back (0.83) to
compare with HIL logs directly. Simulated time throughout: runs repeat.
"""
import math

import numpy as np

from cyberdog.sim.robot.mujoco_robot import MujocoRobot

HIL_CLOCK = 0.83          # the Mac's world against the wall clock (Pace 0.81-0.88)
# Turn gain by forward speed (m/s, the guarded command): fitted on the brain's
# clock 0.20 / 0.48 / 0.62 / 0.70 for 0-0.05 / 0.2-0.4 / 0.4-0.6 / 0.6-0.9 m/s
# (n 26 / 53 / 203 / 311 windows), divided by HIL_CLOCK. The pivot figure has
# the fewest windows behind it.
TURN_GAIN_V = (0.0, 0.3, 0.5, 0.75)
TURN_GAIN = (0.24, 0.58, 0.75, 0.84)
TURN_TAU = 0.10           # s, first-order smoothing of the turn rate
TURN_DELAY = 0.10         # s before a turn command starts to act (fit: 0.15 s on
                          # the brain's clock; two 20 Hz ticks here)


def turn_gain(vx):
    """Fraction of a commanded turn rate the dog turns at, walking at `vx`."""
    return float(np.interp(abs(vx), TURN_GAIN_V, TURN_GAIN))


class BoardModelRobot(MujocoRobot):
    """MujocoRobot, turning as the board's dog turns and stopped by its guard."""
    MAX_W = 1.0               # rad/s: go2_flat_v5 is trained to +-1, as BoardTwinRobot
    GUARDED = True            # run_building meets the guard as on the board

    def __init__(self, scene_xml, start_xy=(45.0, 4.0), start_yaw=math.pi / 2,
                 start_z=0.0, clock=1.0):
        from go2_board import lidar as board_lidar
        self._board = board_lidar
        self.clock = float(clock)
        self._cmd = (0.0, 0.0, 0.0)
        self._w_real = 0.0
        self._w_queue = []
        self._scan, self._scan_t = None, None
        self.guarded_ticks = 0    # ticks the guard cut the forward or side command
        super().__init__(scene_xml, start_xy=start_xy, start_yaw=start_yaw, start_z=start_z)

    def reset(self, xy, yaw):
        super().reset(xy, yaw)
        self._w_real, self._w_queue = 0.0, []
        self._scan, self._scan_t = None, None

    def get_scan(self):
        """As MujocoRobot's, cast once per tick: the brain and the guard both read it."""
        if self._scan_t != self.sim_time or self._scan is None:
            self._scan, self._scan_t = MujocoRobot.get_scan(self), self.sim_time
        return self._scan

    def board_clearance(self):
        """What the board's guard reports: free distance front, back, left,
        right (inf where nothing), from the twin's scan in the body frame."""
        c = self._clearance()
        return {k: getattr(c, k) for k in ("front", "back", "left", "right")}

    def _clearance(self):
        x, y, yaw = self.get_pose()
        p = np.asarray(self.get_scan(), float).reshape(-1, 3)
        cs, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = p[:, 0] - x, p[:, 1] - y
        body = np.column_stack((cs * dx + sn * dy, -sn * dx + cs * dy, p[:, 2] - self.z))
        return self._board.clearance(body, self.sim_time)

    def set_velocity(self, vx, vy=0.0, w=0.0):
        super().set_velocity(vx, vy, w)                     # clipped to MAX_V / MAX_W
        self._cmd = (self._vx, self._vy, self._w)

    def step(self):
        vx, vy, w = self._cmd
        gx, gy, _ = self._board.guard(vx, vy, w, self._clearance(), self.sim_time)
        if (gx, gy) != (vx, vy):
            self.guarded_ticks += 1
        delay = int(round(TURN_DELAY / self.CONTROL_DT))
        self._w_queue.append(w)
        w_cmd = self._w_queue.pop(0) if len(self._w_queue) > delay else 0.0
        a = min(1.0, self.CONTROL_DT / TURN_TAU)
        self._w_real += a * (turn_gain(gx) * w_cmd - self._w_real)
        self._vx, self._vy, self._w = gx * self.clock, gy * self.clock, self._w_real * self.clock
        super().step()
        self._vx, self._vy, self._w = self._cmd

    def place(self, xy, yaw, z=None):
        """Scripted moves (the lift) are not walked: put there, turn state reset."""
        self._w_real, self._w_queue = 0.0, []
        super().place(xy, yaw, z)
