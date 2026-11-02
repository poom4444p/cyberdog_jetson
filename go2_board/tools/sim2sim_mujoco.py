"""Does a policy walk in a simulator it did not train in? Headless, ~30 s.

go2_board's own Modes, PolicyWalker and Safety.limit drive unitree_mujoco's
Go2 model at 500 Hz, with torques from the motor drivers' PD law, as the
sim's DDS bridge would. No DDS, no viewer: a quick check to run on every
exported policy before the full sim.

  python tools/sim2sim_mujoco.py policies/go2_flat_v1
  python tools/sim2sim_mujoco.py policies/go2_flat_v1 --armature 0.1 --friction 1.0

--armature, --friction, --no-joint-friction, --condim, --no-self-collision change the MuJoCo model, to find
which difference from the training sim matters.
"""
import argparse
import math
import os

import mujoco
import numpy as np

from go2_board import go2
from go2_board.board import RATE_HZ
from go2_board.bus import BoardState
from go2_board.controllers import Pose
from go2_board.modes import Modes
from go2_board.safety import Safety, tilt
from go2_board.walker import PolicyWalker

XML = os.path.expanduser(os.environ.get(
    "GO2_MJCF", "~/Work/unitree/unitree_mujoco/unitree_robots/go2/scene.xml"))
COMMANDS = [(0.5, 0, 0), (1.0, 0, 0), (-0.5, 0, 0), (0, 0.3, 0), (0, 0, 0.8), (0, 0, 0)]
SETTLE_S, MEASURE_S = 2.0, 4.0
FELL_TILT_DEG, FELL_Z = 45.0, 0.15

ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("--armature", type=float, default=None, help="leg joints' armature (MuJoCo model: 0.01)")
ap.add_argument("--friction", type=float, default=None, help="every geom's sliding friction (model: 0.4)")
ap.add_argument("--no-joint-friction", action="store_true", help="leg joints' damping and frictionloss to 0")
ap.add_argument("--no-self-collision", action="store_true", help="robot geoms collide with the floor only (as Genesis trains)")
ap.add_argument("--condim", type=int, default=None, help="feet's contact dimension (model: 6, with torsional and rolling friction)")
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(XML)
m.opt.timestep = 1.0 / RATE_HZ
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
if args.armature is not None:
    m.dof_armature[6:] = args.armature
if args.friction is not None:
    m.geom_friction[:, 0] = args.friction
if args.no_self_collision:
    robot = m.geom_bodyid != 0                    # body 0 is the world: the floor
    m.geom_contype[robot] = 0
    m.geom_conaffinity[robot] = 1
if args.condim is not None:
    for leg in ("FR", "FL", "RR", "RL"):
        m.geom_condim[m.geom(leg).id] = args.condim
if args.no_joint_friction:
    m.dof_frictionloss[6:] = 0.0
    m.dof_damping[6:] = 0.0

names = [f"{n}_joint" for n in go2.NAMES]
qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
act = np.array([m.actuator(n).id for n in go2.NAMES])
gyro = m.sensor_adr[m.sensor("imu_gyro").id]
base = m.body("base_link").id


def read():
    return BoardState(q=d.qpos[qadr].copy(), dq=d.qvel[vadr].copy(), quat=d.qpos[3:7].copy(),
                      gyro=d.sensordata[gyro:gyro + 3].copy(), acc=np.zeros(3), stamp=d.time)


def run(modes, seconds, vel=(0.0, 0.0, 0.0), log=None):
    modes.velocity = vel
    end = d.time + seconds
    while d.time < end:
        s = read()
        cmd = Safety.limit(modes.step(s, d.time), s)
        tau = cmd.tau + cmd.kp * (cmd.q - s.q) + cmd.kd * (cmd.dq - s.dq)
        d.ctrl[act] = np.clip(tau, -go2.TAU_MAX, go2.TAU_MAX)
        mujoco.mj_step(m, d)
        if log is not None:
            R = d.xmat[base].reshape(3, 3)
            log.append((R.T @ d.qvel[0:3], d.qvel[5], math.degrees(tilt(d.qpos[3:7])), d.qpos[2]))


walker = PolicyWalker(args.policy)
print(f"{walker.name} in {os.path.basename(XML)}: armature {m.dof_armature[6]:.3f}, "
      f"friction {m.geom_friction[-1, 0]:.2f}, joint frictionloss {m.dof_frictionloss[6]:.2f}, "
      f"foot condim {m.geom_condim[m.geom('FR').id]}")
modes = Modes(walker)
modes.mode, modes.pose = "STAND", Pose(go2.STAND, 1.0)
run(modes, 1.5)
modes.request("walk", d.time)
run(modes, 1.5)

fell, err = False, []
for vel in COMMANDS:
    log = []
    run(modes, SETTLE_S, vel)
    run(modes, MEASURE_S, vel, log)
    v = np.mean([x[0] for x in log], axis=0)
    w = float(np.mean([x[1] for x in log]))
    tmax, zmin = max(x[2] for x in log), min(x[3] for x in log)
    bad = tmax > FELL_TILT_DEG or zmin < FELL_Z
    fell |= bad
    err.append(max(abs(v[0] - vel[0]), abs(v[1] - vel[1]), abs(w - vel[2])))
    print(f"cmd {vel}: got vx {v[0]:+.2f} vy {v[1]:+.2f} w {w:+.2f} | max tilt {tmax:4.1f} deg"
          f" | min z {zmin:.2f} m{'  FELL' if bad else ''}")
    if bad:
        break
modes.request("stand", d.time)
run(modes, 1.0)
print(f"back to {modes.mode}: z {d.qpos[2]:.3f} m")
print(f"RESULT {'FELL' if fell else 'stayed up'}, worst tracking error {max(err):.2f}")
