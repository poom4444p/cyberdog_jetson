"""Tests for the affordance stage on VAMOS's candidates (--vamos-affordance).

VamosPolicy.deliver with a stand-in judge: what a refusal, a pass and "cannot
say" do to the verdicts and the stats. runtime.judge_paths with a fixed
random network: which paths get judged at all, and that the batched call
agrees with judging one at a time. And infer.py, the numpy network the J4012's
brain runs, against the torch one it was exported from. Only that last one
needs torch.
"""
import math

import numpy as np
import pytest

from cyberdog.sim.vamos_client import VamosPolicy

POSE = (0.0, 0.0, 0.0)
DEST = (5.0, 0.0)


def _line(a, b, n=6):
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            for k in range(n + 1)]


STRAIGHT = _line((0.0, 0.0), (3.0, 0.0))
LEFT = _line((0.0, 0.0), (3.0, 0.6))
RIGHT = _line((0.0, 0.0), (3.0, -0.6))


def _policy(judge):
    return VamosPolicy(cam=None, is_free=lambda x, y: True, url="http://127.0.0.1:1",
                       affordance=judge)


class TestDeliver:
    def test_a_refused_path_is_rejected_and_the_rest_go_on(self):
        p = _policy(lambda paths, pose: ["not walkable" if q == STRAIGHT else "walkable"
                                         for q in paths])
        chosen, _, _ = p.deliver([STRAIGHT, LEFT, RIGHT], POSE, DEST)
        # deliver() cuts each path to what is ahead (new lists), so by end point.
        verdict = {q[-1]: v for q, _, v in p.verdicts}
        assert verdict[STRAIGHT[-1]] == "affordance"
        assert verdict[LEFT[-1]] == verdict[RIGHT[-1]] == "pass"
        assert chosen != STRAIGHT
        assert p.stats["rejected_by_affordance"] == 1
        assert p.stats["rejected"] == 1

    def test_caution_is_not_a_refusal(self):
        p = _policy(lambda paths, pose: ["caution"] * len(paths))
        chosen, _, _ = p.deliver([STRAIGHT], POSE, DEST)
        assert chosen == STRAIGHT
        assert p.stats["rejected_by_affordance"] == 0

    def test_cannot_say_passes_and_is_counted(self):
        # No judgement is not a refusal: too little seen must not stop the dog.
        p = _policy(lambda paths, pose: [None] * len(paths))
        chosen, _, _ = p.deliver([STRAIGHT, LEFT], POSE, DEST)
        assert chosen is not None
        assert p.stats["affordance_unjudged"] == 2
        assert p.stats["rejected"] == 0

    def test_all_refused_is_nothing_chosen(self):
        p = _policy(lambda paths, pose: ["not walkable"] * len(paths))
        chosen, _, safety = p.deliver([STRAIGHT, LEFT], POSE, DEST)
        assert chosen is None and safety == 0.0
        assert p.stats["rejected_by_affordance"] == 2

    def test_only_paths_that_passed_traversable_are_asked(self):
        asked = []
        p = VamosPolicy(cam=None, is_free=lambda x, y: y <= 0.05, url="http://127.0.0.1:1",
                        affordance=lambda paths, pose: asked.extend(paths) or
                        ["walkable"] * len(paths))
        p.deliver([STRAIGHT, LEFT], POSE, DEST)
        assert asked == [STRAIGHT]

    def test_off_by_default(self):
        p = VamosPolicy(cam=None, is_free=lambda x, y: True, url="http://127.0.0.1:1")
        chosen, _, _ = p.deliver([STRAIGHT], POSE, DEST)
        assert chosen == STRAIGHT
        assert all(v != "affordance" for _, _, v in p.verdicts)


@pytest.fixture
def rt():
    from cyberdog.affordance import runtime
    return runtime


def _random_net(seed=0, hidden=(256, 128)):
    from cyberdog.affordance import data, infer
    rng = np.random.default_rng(seed)
    sizes = [infer.N_IN, *hidden, len(data.CLASS_NAMES)]
    return infer.NumpyMLP([rng.normal(0, 1 / np.sqrt(a), (b, a)) for a, b in zip(sizes, sizes[1:])],
                          [rng.normal(0, 0.1, b) for b in sizes[1:]])


@pytest.fixture
def aff():
    """An Affordance without the trained weights: the same call, a fixed net."""
    from cyberdog.affordance import infer

    class _Aff:
        net = _random_net()
        refuse_p = infer.REFUSE_P
    return _Aff()


def _patch(fill=0.0):
    from cyberdog.affordance import data
    return np.full((data.NY, data.NX), fill, np.float32)


class TestLeaves:
    def test_crossing_between_vertices(self, rt):
        # Vertices at 0 and 3 m: the crossing at 1.2 m is between them.
        p = rt.leaves([(0.0, 0.0), (3.0, 0.0)], (0.0, 0.0), r=1.2)
        assert math.isclose(p[0], 1.2, abs_tol=0.06) and p[1] == 0.0

    def test_short_path_gives_its_end(self, rt):
        assert rt.leaves([(0.0, 0.0), (0.5, 0.0)], (0.0, 0.0), r=1.2) == (0.5, 0.0)


class TestJudgePaths:
    def test_every_ahead_path_gets_a_class(self, rt, aff):
        from cyberdog.affordance import data
        out = rt.judge_paths(aff, _patch(), POSE, [STRAIGHT, LEFT, RIGHT])
        assert all(c in data.CLASS_NAMES for c in out)

    def test_too_little_seen_is_no_judgement(self, rt, aff):
        assert rt.judge_paths(aff, _patch(np.nan), POSE, [STRAIGHT]) == [None]

    def test_a_path_behind_or_sideways_is_not_judged(self, rt, aff):
        back = _line((0.0, 0.0), (-3.0, 0.0))
        side = _line((0.0, 0.0), (0.0, 3.0))
        assert rt.judge_paths(aff, _patch(), POSE, [back, side]) == [None, None]

    def test_batched_agrees_with_one_at_a_time(self, rt, aff):
        patch = _patch()
        # A step across the walk, so the answer is not the same for every bearing.
        patch[:, 12:] = 0.3
        paths = [STRAIGHT, LEFT, RIGHT]
        batched = rt.judge_paths(aff, patch, POSE, paths)
        single = [rt.judge_paths(aff, patch, POSE, [p])[0] for p in paths]
        assert batched == single


class TestNumpyNetwork:
    def test_numpy_matches_torch(self, tmp_path):
        torch = pytest.importorskip("torch")
        from cyberdog.affordance import data, infer
        from cyberdog.affordance import model as M
        torch.manual_seed(1)
        net = M.AffordanceMLP().eval()
        rng = np.random.default_rng(1)
        patch = rng.uniform(-0.3, 0.3, (8, data.NY, data.NX)).astype(np.float32)
        patch[rng.random(patch.shape) < 0.2] = np.nan
        target = np.column_stack([rng.uniform(0.6, 1.4, 8), rng.uniform(-0.5, 0.5, 8)])
        want_cls, want_p = M.classify(net, patch, target)
        M.save(net, tmp_path / "mlp.pt")
        np_net, info = infer.load(M.export(tmp_path / "mlp.pt"))
        got_cls, got_p = infer.classify(np_net, patch, target)
        assert np.allclose(got_p, want_p, atol=1e-5)
        assert (got_cls == want_cls).all()
        assert info["classes"] == tuple(data.CLASS_NAMES)

    def test_runtime_loads_the_npz_without_torch(self, tmp_path):
        from cyberdog.affordance import infer, runtime
        np.savez(tmp_path / "mlp.npz", **_random_net().arrays())
        a = runtime.Affordance(tmp_path / "mlp.npz")
        assert isinstance(a.net, infer.NumpyMLP)
