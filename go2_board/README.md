# go2_board

The Go2's one board: everything between the Cyberdog brain and the motor
drivers, on the J4012. It replaces what the Go2's own main board does.

```
J4012
  (1) brain           Cyberdog repo: vx, vy, w                  20 Hz
  (2) controller      Pose (stand, lie); PolicyWalker (WALK)     500 Hz, policy 50 Hz
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
| `go2_board/walker.py` | `PolicyWalker`: the policy go2_train exported (`policy_spec.json`, `policy.npz`), in numpy. Checks the spec's samples on load, blends from STAND to the policy's pose and gains, runs it every 10th tick |
| `go2_board/board.py` | The 500 Hz loop: read, controller, safety, write |
| `go2_board/modes.py` | LIMP, STAND_UP, STAND, WALK, LIE_DOWN, and what may follow what |
| `go2_board/link.py` | The brain link: UDP on localhost, commands in (`vx, vy, w`, mode), state out at 50 Hz |
| `go2_board/app.py` | The board as a program: link, modes, safety and bus, one tick at a time |
| `tools/sim2sim_mujoco.py` | A policy through Modes, PolicyWalker and safety in unitree_mujoco's Go2, headless, ~30 s: does it walk in a sim it did not train in? |
| `tools/check_j4012.sh` | What a J4012 has (JetPack, kernel, CUDA, torch, network), read-only: run it on the board first |
| `go2_board/livox.py` | Mid-360 packets (Livox SDK2 UDP) to points: CRC, udp_cnt gaps, frames; same code for the real sensor |
| `go2_board/lidar.py` | The Mid-360 in use: a thread that levels each frame with the sensor's IMU, works out the free distance each way for the loop (`guard` limits the brain's velocity by it), and sends the cloud to the brain on 9872 (`CloudLink` is the brain's end) |
| `scripts/board_main.py` | Run the board against the sim, waiting for the brain |
| `scripts/brain_cli.py` | A stand-in brain: `stand:3 walk:5:0.5,0,0 lie:2` (mode:seconds[:vx,vy,w]) |
| `scripts/run_board.py` | Stand up, hold, lie down, go limp, with no brain |
| `scripts/livox_watch.py` | Print what the Mid-360 (real or go2_world's) is sending, once a second |

## Run against the sim (Mac)

```
conda activate unitree_sim
mjpython ~/Work/unitree/go2_world/scripts/run_sim.py          # terminal 1: the sim's Go2
python scripts/board_main.py --policy policies/go2_flat_v1   # terminal 2: the board
python scripts/brain_cli.py stand:3 walk:2 walk:5:0.5,0,0 stand:2 lie:2   # terminal 3
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
Mid-360 --56301/56401--> Lidar thread --Clearance--> App.step: guard(vx, vy, w)  (500 Hz)
                                  \----9872-------> brain: CloudLink (levelled trunk frame)
```

In WALK the brain's vx and vy are scaled down from 1.0 m of free distance and
stopped at 0.3 m, each way; a LiDAR silent for 0.3 s stops translation but
not turning. The state message to the brain carries the clearance. The mount
(`lidar.MOUNT_PITCH`) must match go2_world's until the real
bracket is measured; the floor is found in each frame, so the dog's height
does not matter. `board_main.py --no-lidar` runs without it.

```
python ~/Work/unitree/go2_world/scripts/run_lidar.py --static   # Mac: a fake Mid-360
python scripts/livox_watch.py                                   # 10 frames/s, 200 imu/s, lost, bad
```

The real sensor sends to the host IP configured in it (Livox's quick start
puts the PC at 192.168.1.50; check the unit), so the J4012's Ethernet needs
that address, or the sensor needs telling another. `livox.py` listens on every
interface.

## Walking policies

`policies/<run>/` holds what go2_train exported (copy `logs/<run>/exported/`
there; not in git). `board_main.py --policy` loads one; without it, WALK is
refused. Before trusting a new one:

```
python tools/sim2sim_mujoco.py policies/<run>
```

`tools/sim2sim_mujoco.py` turns off the bars (x = 1.2, 1.6 m) and stairs
(from 2.3 m) that unitree_mujoco's go2 `scene.xml` has in front of the dog;
`--obstacles` keeps them. Earlier runs walked into the first bar and were
read as failing to walk.

| Command | go2_flat_v1 | go2_flat_v2 | go2_flat_v3 |
|---|---|---|---|
| vx 0.5 | spins | +0.56 | +0.43 |
| vx 1.0 | spins | +1.04 | +0.95 |
| vx -0.5 | spins | -0.54 | -0.43 |
| vy 0.3 | spins | +0.27 | +0.16 |
| w 0.8 | spins | +0.56 | +0.49 |
| still | spins (-0.6 rad/s) | 0.00 | 0.00 |
| max tilt | 21 deg | 4.7 deg | 6.0 deg |

v1 trained on Genesis' default joint armature (0.1, ten times the Go2
model's); v2 on the Go2's joint dynamics, randomized; v3 adds feet rewards
and tracks a little worse. **go2_flat_v2 is the one to use**; it turns at
70 % of what it is asked.

`run_sim.py` loads the same scene with the bars in it: in the full sim,
keep forward walks under ~1 m from the start.

## Not yet

- Turning tracks at 70 % (go2_flat_v2, w 0.8 -> 0.56).
- A bus to real motor drivers.
- Battery and temperature limits.
