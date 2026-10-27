# go2_board

The Go2's one board: everything between the Cyberdog brain and the motor
drivers, on the J4012. It replaces what the Go2's own main board does.

```
J4012
  (1) brain           Cyberdog repo: vx, vy, w                  20 Hz
  (2) controller      Pose now; the walking policy next          500 Hz
  (3) safety          limits, and damping when it cannot trust   500 Hz
  (4) motor bus       MotorCmd out, BoardState in                500 Hz
        |
        |  sim:  DDS (rt/lowcmd, rt/lowstate) to unitree_mujoco
        |  real: straight to the 12 motor drivers (not written yet)
        v
  motor drivers: tau + kp*(q - q_measured) + kd*(dq - dq_measured)
```

| File | What |
|---|---|
| `go2_board/go2.py` | Joint order (FR, FL, RR, RL x hip, thigh, calf), ranges, peak torques, poses |
| `go2_board/bus.py` | `MotorBus`, `BoardState`, `MotorCmd`: the line every layer above stops at |
| `go2_board/sim_bus.py` | `SimBus`: the bus over DDS in Unitree's format |
| `go2_board/safety.py` | Clips targets to joint range and peak torque; damping, latched, on stale state, a fall or a failed controller |
| `go2_board/controllers.py` | `Pose`: move smoothly to a pose and hold it |
| `go2_board/board.py` | The 500 Hz loop: read, controller, safety, write |
| `scripts/run_board.py` | Stand up, hold, lie down, go limp, against the sim |

## Run against the sim (Mac)

```
conda activate unitree_sim
mjpython ~/Work/unitree/scripts/run_sim.py          # terminal 1: the sim's Go2
python scripts/run_board.py --hold 3                 # terminal 2: the board
python -m pytest -q                                  # no sim needed
```

macOS: the SDK's timer is Linux-only, so `_macos_timerfd.py` stands in for it
there, and the loopback is `lo0`. On the J4012 neither applies.

## Not yet

- The walking policy (a controller taking vx, vy, w).
- A bus to real motor drivers.
- Battery and temperature limits.
