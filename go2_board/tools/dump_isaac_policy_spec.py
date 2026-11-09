"""Write down everything the board needs to run an Isaac Lab Go2 policy.

Runs inside Isaac Lab, on the GPU machine (UCloud). From the IsaacLab folder:

    ./isaaclab.sh -p /path/to/go2_board/tools/dump_isaac_policy_spec.py \
        --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
        --policy /path/to/exported/policy.pt --headless

Writes policy_spec.json next to the policy. Copy both files (and
policy.onnx if play.py wrote one) to the board. The board never guesses
these numbers: a joint order, a default pose or a gain that differs from
training by a little is a dog that falls over.

What it records:
  - joint names in the policy's order, and the default joint angles;
  - each actuator's stiffness, damping and effort limit (the PD the motor
    side must reproduce);
  - the action term: scale, and whether the default pose is added;
  - the observation terms of the "policy" group in order, with their sizes,
    scales and clips;
  - sim dt and decimation (the policy's rate), and the command ranges;
  - a few (observation, action) pairs from the live env, so the board can
    check that it loads the same network and gets the same answer.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Velocity-Flat-Unitree-Go2-Play-v0")
parser.add_argument("--policy", required=True, help="TorchScript policy.pt exported by play.py")
parser.add_argument("--out", default=None, help="default: policy_spec.json next to --policy")
parser.add_argument("--samples", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  registers the tasks with gym
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def as_list(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return x


def main():
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env = gym.make(args.task, cfg=cfg)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]

    spec = {"task": args.task, "policy": str(Path(args.policy).resolve())}
    spec["sim_dt"] = float(cfg.sim.dt)
    spec["decimation"] = int(cfg.decimation)
    spec["policy_dt"] = float(uenv.step_dt)

    spec["joint_names"] = list(robot.joint_names)
    spec["default_joint_pos"] = as_list(robot.data.default_joint_pos[0])

    spec["actuators"] = {}
    for name, act in robot.actuators.items():
        idx = act.joint_indices
        idx = list(range(len(robot.joint_names))) if isinstance(idx, slice) else as_list(idx)
        spec["actuators"][name] = {
            "class": type(act).__name__,
            "joints": [robot.joint_names[i] for i in idx],
            "stiffness": as_list(act.stiffness[0]),
            "damping": as_list(act.damping[0]),
            "effort_limit": as_list(act.effort_limit[0]),
        }

    a = cfg.actions.joint_pos
    spec["action"] = {
        "class": type(a).__name__,
        "joint_names": as_list(getattr(a, "joint_names", None)),
        "scale": as_list(getattr(a, "scale", None)),
        "offset": as_list(getattr(a, "offset", None)),
        "use_default_offset": getattr(a, "use_default_offset", None),
        "clip": as_list(getattr(a, "clip", None)),
    }

    om = uenv.observation_manager
    names = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]
    cfgs = om._group_obs_term_cfgs["policy"]
    spec["observations"] = [
        {"name": n, "dim": list(d), "func": getattr(c.func, "__name__", str(c.func)),
         "scale": as_list(c.scale), "clip": as_list(c.clip),
         "params": {k: str(v) for k, v in (c.params or {}).items()}}
        for n, d, c in zip(names, dims, cfgs)]
    spec["obs_dim"] = int(sum(int(torch.tensor(d).prod()) for d in dims))

    r = cfg.commands.base_velocity.ranges
    spec["command_ranges"] = {k: as_list(getattr(r, k)) for k in
                              ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")
                              if hasattr(r, k)}

    # Golden pairs: the board loads the same network and must give the same
    # action for these observations.
    policy = torch.jit.load(args.policy, map_location=uenv.device).eval()
    obs, _ = env.reset()
    pairs = []
    with torch.inference_mode():
        for i in range(200):
            o = obs["policy"]
            act = policy(o)
            if len(pairs) < args.samples and i % 40 == 0:
                pairs.append({"obs": as_list(o[0]), "action": as_list(act[0])})
            obs, *_ = env.step(act)
    spec["golden"] = pairs

    out = Path(args.out) if args.out else Path(args.policy).with_name("policy_spec.json")
    out.write_text(json.dumps(spec, indent=1))
    print(f"wrote {out}")
    print(f"  joints: {spec['joint_names']}")
    print(f"  obs: {[(o['name'], o['dim']) for o in spec['observations']]} = {spec['obs_dim']}")
    print(f"  policy rate: {1 / spec['policy_dt']:.0f} Hz, action scale {spec['action']['scale']}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
