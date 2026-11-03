"""PolicyWalker, with a small random network written as go2_train would."""
import json
import math
import os

import numpy as np
import pytest

from go2_board import go2
from go2_board.bus import BoardState
from go2_board.walker import BLEND_S, PolicyWalker, mlp, projected_gravity

DT = 0.002
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
DEFAULT = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]
LAYOUT = [("base_ang_vel", 3, 0.25), ("projected_gravity", 3, 1.0), ("commands", 3, None),
          ("dof_pos_minus_default", 12, 1.0), ("dof_vel", 12, 0.05), ("last_action", 12, 1.0)]
REAL = os.path.join(os.path.dirname(__file__), "..", "policies", "go2_flat_v1")


def write_policy(folder, seed=0, samples=None, latency=True, **spec_over):
    rng = np.random.default_rng(seed)
    sizes = [45, 32, 16, 12]
    weights = {}
    for i in range(3):
        weights[f"w{i}"] = rng.normal(0, 0.3, (sizes[i + 1], sizes[i])).astype(np.float32)
        weights[f"b{i}"] = rng.normal(0, 0.1, sizes[i + 1]).astype(np.float32)
    np.savez(os.path.join(folder, "policy.npz"), **weights)
    layers = [(weights[f"w{i}"], weights[f"b{i}"]) for i in range(3)]
    if samples is None:
        samples = []
        for _ in range(3):
            obs = rng.normal(0, 1, 45).tolist()
            samples.append({"obs": obs, "action": mlp(layers, obs).tolist()})
    spec = {
        "format": "go2_train/1", "run": "test",
        "joint_names": [f"{n}_joint" for n in go2.NAMES],
        "default_joint_angles": DEFAULT, "kp": 20.0, "kd": 0.5, "policy_dt": 0.02,
        "action": {"scale": 0.25, "clip": 100.0, "trained_with_one_step_latency": latency},
        "observation": [{"name": n, "size": k, "scale": [2.0, 2.0, 0.25] if s is None else [s] * k}
                        for n, k, s in LAYOUT],
        "observation_size": 45,
        "commands": {"vx": [-0.6, 1.0], "vy": [-0.4, 0.4], "w": [-1.0, 1.0]},
        "samples": samples,
    }
    spec.update(spec_over)
    with open(os.path.join(folder, "policy_spec.json"), "w") as f:
        json.dump(spec, f)
    return layers


def state(q=go2.STAND, dq=None, quat=UPRIGHT, gyro=(0, 0, 0)):
    return BoardState(q=np.array(q, float), dq=np.zeros(12) if dq is None else np.array(dq, float),
                      quat=np.array(quat, float), gyro=np.array(gyro, float), acc=np.zeros(3), stamp=0.0)


def past_blend(w, s, t=0.0):
    """Tick through the blend; return the time of the first policy tick."""
    w.step(s, t, 0, 0, 0)
    while w.blending(t):
        t += DT
        if w.blending(t):
            w.step(s, t, 0, 0, 0)
    return t


# -- the observation's pieces --------------------------------------------------

def test_gravity_upright_and_rolled():
    np.testing.assert_allclose(projected_gravity(UPRIGHT), [0, 0, -1], atol=1e-12)
    r = math.radians(30)                                 # rolled right 30 deg about x
    g = projected_gravity([math.cos(r / 2), math.sin(r / 2), 0, 0])
    np.testing.assert_allclose(g, [0, -math.sin(r), -math.cos(r)], atol=1e-12)
    p = math.radians(20)                                 # nose down 20 deg about y
    g = projected_gravity([math.cos(p / 2), 0, math.sin(p / 2), 0])
    np.testing.assert_allclose(g, [math.sin(p), 0, -math.cos(p)], atol=1e-12)


def test_observation_order_and_scales(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    w.action = np.arange(12) / 10
    q = np.array(DEFAULT) + 0.1
    obs = w.observation(state(q=q, dq=np.ones(12) * 2, gyro=(1, 2, 3)), np.array([0.5, -0.2, 0.4]))
    np.testing.assert_allclose(obs[0:3], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(obs[3:6], [0, 0, -1])
    np.testing.assert_allclose(obs[6:9], [1.0, -0.4, 0.1])
    np.testing.assert_allclose(obs[9:21], 0.1, atol=1e-12)
    np.testing.assert_allclose(obs[21:33], 0.1)
    np.testing.assert_allclose(obs[33:45], np.arange(12) / 10)


# -- loading -----------------------------------------------------------------------

def test_loads_and_checks_its_samples(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    assert w.decimation == 10 and w.worst < 1e-5


def test_refuses_a_network_that_disagrees_with_its_samples(tmp_path):
    write_policy(tmp_path, samples=[{"obs": [0.0] * 45, "action": [9.0] * 12}])
    with pytest.raises(ValueError, match="away from the spec's samples"):
        PolicyWalker(str(tmp_path))


def test_refuses_another_joint_order(tmp_path):
    names = [f"{leg}_{p}_joint" for p in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
    write_policy(tmp_path, joint_names=names)
    with pytest.raises(ValueError, match="joint order"):
        PolicyWalker(str(tmp_path))


def test_refuses_a_rate_the_board_cannot_keep(tmp_path):
    write_policy(tmp_path, policy_dt=0.0125)
    with pytest.raises(ValueError, match="whole number"):
        PolicyWalker(str(tmp_path))


# -- stepping ----------------------------------------------------------------------

def test_blends_from_stand_to_the_policy_pose_and_gains(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    s = state()
    first = w.step(s, 0.0, 0.5, 0, 0)
    np.testing.assert_allclose(first.q, go2.STAND)
    assert first.kp[0] == pytest.approx(go2.KP_STAND)
    last = w.step(s, BLEND_S - 1e-6, 0.5, 0, 0)
    np.testing.assert_allclose(last.q, DEFAULT, atol=1e-6)
    assert last.kp[0] == pytest.approx(20.0, abs=1e-3) and last.kd[0] == pytest.approx(0.5, abs=1e-3)


def test_policy_runs_every_tenth_tick_and_holds_between(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    t = past_blend(w, s)
    targets = []
    for i in range(30):
        s.dq = np.sin(np.arange(12) + i)                 # a state that keeps changing
        targets.append(w.step(s, t + i * DT, 0.5, 0, 0).q.copy())
    for i in range(30):
        if i % 10:
            np.testing.assert_array_equal(targets[i], targets[i - 1])
    assert not np.allclose(targets[10], targets[20])


def test_one_step_latency_as_trained(tmp_path):
    write_policy(tmp_path, latency=True)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    t = past_blend(w, s)
    first = w.step(s, t, 0.5, 0, 0)
    np.testing.assert_allclose(first.q, DEFAULT)          # the first action lands one step later
    a1 = w.action.copy()
    for i in range(1, 11):
        out = w.step(s, t + i * DT, 0.5, 0, 0)
    np.testing.assert_allclose(out.q, np.array(DEFAULT) + 0.25 * a1)


def test_without_latency_the_action_lands_at_once(tmp_path):
    write_policy(tmp_path, latency=False)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    out = w.step(s, past_blend(w, s), 0.5, 0, 0)
    np.testing.assert_allclose(out.q, np.array(DEFAULT) + 0.25 * w.action)


def ticks(w, s, t, n, vel):
    for _ in range(n):
        w.step(s, t, *vel)
        t += DT
    return t


def test_commands_are_clipped_to_what_it_trained_on(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    ticks(w, s, past_blend(w, s), 1000, (3.0, -2.0, 5.0))       # 2 s: ramped all the way
    np.testing.assert_allclose(w.command, [1.0, -0.4, 1.0])


def test_command_ramps_from_standing_and_on_every_step_change(tmp_path):
    from go2_board.walker import CMD_ACCEL
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    t = past_blend(w, s)
    per_policy_step = np.asarray(CMD_ACCEL) * w.decimation * DT
    w.step(s, t, 0.8, 0.0, 1.0)                     # a step from standing still
    np.testing.assert_allclose(w.command, [per_policy_step[0], 0.0, per_policy_step[2]])
    t = ticks(w, s, t + DT, 1000, (0.8, 0.0, 1.0))
    np.testing.assert_allclose(w.command, [0.8, 0.0, 1.0])
    ticks(w, s, t, w.decimation, (0.0, 0.0, 0.0))  # a stop is eased too, not slammed
    np.testing.assert_allclose(w.command, [0.8 - per_policy_step[0], 0.0, 1.0 - per_policy_step[2]])
    w.reset()
    assert not w.command.any()                      # every WALK starts from standing


def test_reset_starts_the_blend_again(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    past_blend(w, s)
    w.reset()
    assert w.blending(50.0) and not w.action.any()


def test_a_non_finite_output_raises(tmp_path):
    write_policy(tmp_path)
    w = PolicyWalker(str(tmp_path))
    s = state(q=DEFAULT)
    t = past_blend(w, s)
    s.dq[:] = np.nan
    with pytest.raises(FloatingPointError):
        w.step(s, t, 0.5, 0, 0)


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "policy_spec.json")),
                    reason="policies/go2_flat_v1 not copied in (not in git)")
def test_the_trained_policy_loads():
    w = PolicyWalker(REAL)
    assert w.decimation == 10 and w.obs_size == 45


def test_modes_stand_walk_stand_with_the_policy(tmp_path):
    from go2_board.modes import WALK_TO_STAND_S, Modes

    write_policy(tmp_path)
    m = Modes(PolicyWalker(str(tmp_path)))
    s = state(q=go2.STAND)
    m.mode, m.pose = "STAND", None
    from go2_board.controllers import Pose
    m.pose = Pose(go2.STAND, 0.0)
    m.request("walk", 0.0)
    assert m.mode == "WALK"
    m.velocity = (0.5, 0.0, 0.0)
    t = 0.0
    while t < BLEND_S + 0.1:
        cmd = m.step(s, t)
        s.q = cmd.q.copy()
        t += DT
    assert cmd.kp[0] == pytest.approx(20.0)
    m.request("stand", t)
    assert m.mode == "STAND"
    first = m.step(s, t)
    np.testing.assert_allclose(first.q, s.q)              # no jump: it blends back
    end = m.step(s, t + WALK_TO_STAND_S + DT)
    np.testing.assert_allclose(end.q, go2.STAND)
