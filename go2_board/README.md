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
| `go2_board/modes.py` | LIMP, STAND_UP, STAND, WALK, LIE_DOWN, and what may follow what |
| `go2_board/link.py` | The brain link: UDP on localhost, commands in (`vx, vy, w`, mode), state out at 50 Hz |
| `go2_board/app.py` | The board as a program: link, modes, safety and bus, one tick at a time |
| `go2_board/livox.py` | Mid-360 packets (Livox SDK2 UDP) to points: CRC, udp_cnt gaps, frames; same code for the real sensor |
| `scripts/board_main.py` | Run the board against the sim, waiting for the brain |
| `scripts/brain_cli.py` | A stand-in brain: `stand:3 walk:1 lie:2` |
| `scripts/run_board.py` | Stand up, hold, lie down, go limp, with no brain |
| `scripts/livox_watch.py` | Print what the Mid-360 (real or go2_world's) is sending, once a second |

## Run against the sim (Mac)

```
conda activate unitree_sim
mjpython ~/Work/unitree/scripts/run_sim.py          # terminal 1: the sim's Go2
python scripts/board_main.py                         # terminal 2: the board
python scripts/brain_cli.py stand:3 lie:2            # terminal 3: a stand-in brain
python -m pytest -q                                  # no sim needed
```

macOS: the SDK's timer is Linux-only, so `_macos_timerfd.py` stands in for it
there, and the loopback is `lo0`. On the J4012 neither applies.

In the sim the body position comes from `rt/sportmodestate`, which is the
simulator's ground truth. On the real dog it has to come from FAST-LIO.

## Mid-360

Points arrive on UDP 56301 and IMU on 56401, from go2_world's emulator on the
Mac or from the real sensor on the J4012; `livox.py` cannot tell which.

```
python ~/Work/unitree/go2_world/scripts/run_lidar.py --static   # Mac: a fake Mid-360
python scripts/livox_watch.py                                   # 10 frames/s, 200 imu/s, lost, bad
```

The real sensor sends to the host IP configured in it (Livox's quick start
puts the PC at 192.168.1.50; check the unit), so the J4012's Ethernet needs
that address, or the sensor needs telling another. `livox.py` listens on every
interface.

## Not yet

- The walking policy (a controller taking vx, vy, w): WALK is refused until one is loaded.
- A bus to real motor drivers.
- Battery and temperature limits.
