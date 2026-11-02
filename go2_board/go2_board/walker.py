"""WALK: the trained policy as a controller. vx, vy, w in; 12 targets out.

Loads what go2_train exports (policies/<run>/policy_spec.json and
policy.npz) and runs the MLP in numpy: no torch on the J4012. Everything it
does with the policy's inputs and outputs comes from the spec, never from
numbers written here; on load it runs the spec's (obs, action) samples and
refuses a policy that does not give the same answers.

On entering WALK it first blends, over BLEND_S, from wherever the legs are
(STAND) to the pose the policy was trained around, with the gains easing
from STAND's to the policy's. Then, every `decimation` ticks (10 at 500 Hz
for a 50 Hz policy) it builds the observation, runs the policy and holds
the new targets until the next one.

Training applied each action one policy step late (Genesis' stand-in for
the real robot's delay); so does this walker, when the spec says so.
"""
import json
import math
import os

import numpy as np

from go2_board import go2
from go2_board.board import RATE_HZ
from go2_board.bus import MotorCmd

BLEND_S = 1.0
FORMAT = "go2_train/1"


def projected_gravity(quat):
    """(0, 0, -1) in the body frame, from the body's w x y z orientation."""
    w, x, y, z = quat
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def mlp(weights, x):
    """ELU between layers, nothing after the last: rsl_rl's MLP actor.
    One observation (45,) in, one action (12,) out. np.dot, not @: numpy 2.2's
    matmul on macOS (Accelerate) flags spurious overflows for these small numbers."""
    x = np.asarray(x, np.float32)
    for i, (w, b) in enumerate(weights):
        x = np.dot(w, x) + b
        if i < len(weights) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return x


class PolicyWalker:
    def __init__(self, folder, rate_hz=RATE_HZ, check_atol=1e-4):
        with open(os.path.join(folder, "policy_spec.json")) as f:
            spec = json.load(f)
        if spec.get("format") != FORMAT:
            raise ValueError(f"{folder}: spec format {spec.get('format')!r}, expected {FORMAT!r}")
        if spec["joint_names"] != [f"{n}_joint" for n in go2.NAMES]:
            raise ValueError(f"{folder}: joint order {spec['joint_names']} is not the board's")
        npz = np.load(os.path.join(folder, "policy.npz"))
        n = len([k for k in npz.files if k.startswith("w")])
        self.weights = [(npz[f"w{i}"].astype(np.float32), npz[f"b{i}"].astype(np.float32)) for i in range(n)]

        self.name = spec.get("run", os.path.basename(folder.rstrip("/")))
        self.default = np.array(spec["default_joint_angles"], float)
        self.kp, self.kd = float(spec["kp"]), float(spec["kd"])
        self.action_scale = float(spec["action"]["scale"])
        self.action_clip = float(spec["action"]["clip"])
        self.latency = bool(spec["action"]["trained_with_one_step_latency"])
        ratio = spec["policy_dt"] * rate_hz
        if abs(ratio - round(ratio)) > 1e-6 or round(ratio) < 1:
            raise ValueError(f"policy_dt {spec['policy_dt']} is not a whole number of {rate_hz} Hz ticks")
        self.decimation = int(round(ratio))
        self.obs_scale = {o["name"]: np.array(o["scale"], float) for o in spec["observation"]}
        self.obs_order = [(o["name"], o["size"]) for o in spec["observation"]]
        self.obs_size = spec["observation_size"]
        c = spec["commands"]
        self.cmd_lo = np.array([c["vx"][0], c["vy"][0], c["w"][0]])
        self.cmd_hi = np.array([c["vx"][1], c["vy"][1], c["w"][1]])
        if self.weights[0][0].shape[1] != self.obs_size or self.weights[-1][0].shape[0] != go2.N:
            raise ValueError(f"{folder}: network is {self.weights[0][0].shape[1]} -> "
                             f"{self.weights[-1][0].shape[0]}, spec says {self.obs_size} -> {go2.N}")
        self.worst = self._check(spec["samples"], check_atol)
        self.reset()

    def _check(self, samples, atol):
        if not samples:
            raise ValueError("spec has no samples to check the network against")
        worst = max(float(np.abs(mlp(self.weights, s["obs"]) - np.asarray(s["action"])).max())
                    for s in samples)
        if worst > atol:
            raise ValueError(f"policy.npz gives answers {worst:.1e} away from the spec's samples")
        return worst

    def reset(self):
        """Called by Modes on entering WALK."""
        self.ticks = 0
        self.t0 = None
        self.start = None
        self.action = np.zeros(go2.N)       # the policy's last output (an input to the next)
        self.applied = np.zeros(go2.N)      # the action the targets come from
        self.target = self.default.copy()
        self.command = np.zeros(3)

    def observation(self, state, command):
        parts = {
            "base_ang_vel": np.asarray(state.gyro, float),
            "projected_gravity": projected_gravity(state.quat),
            "commands": command,
            "dof_pos_minus_default": np.asarray(state.q, float) - self.default,
            "dof_vel": np.asarray(state.dq, float),
            "last_action": self.action,
        }
        obs = np.concatenate([parts[name] * self.obs_scale[name] for name, _ in self.obs_order])
        if obs.shape != (self.obs_size,):
            raise ValueError(f"observation is {obs.shape}, expected ({self.obs_size},)")
        return obs

    def blending(self, t):
        return self.t0 is None or t - self.t0 < BLEND_S

    def step(self, state, t, vx, vy, w):
        if self.t0 is None:
            self.t0, self.start = t, np.asarray(state.q, float).copy()
        if self.blending(t):
            a = min(1.0, (t - self.t0) / BLEND_S)
            a = 0.5 - 0.5 * math.cos(math.pi * a)          # smooth at both ends
            q = (1 - a) * self.start + a * self.default
            kp = (1 - a) * go2.KP_STAND + a * self.kp
            kd = (1 - a) * go2.KD_STAND + a * self.kd
            return MotorCmd.hold(q, kp, kd)

        if self.ticks % self.decimation == 0:
            self.command = np.clip([vx, vy, w], self.cmd_lo, self.cmd_hi)
            obs = self.observation(state, self.command)
            act = np.clip(mlp(self.weights, obs).astype(float), -self.action_clip, self.action_clip)
            if not np.all(np.isfinite(act)):
                raise FloatingPointError("policy output is not finite")
            self.applied = self.action if self.latency else act
            self.action = act
            self.target = self.default + self.action_scale * self.applied
        self.ticks += 1
        return MotorCmd.hold(self.target, self.kp, self.kd)
