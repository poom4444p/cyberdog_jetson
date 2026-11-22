"""The VAMOS gate's chance constraint: P(safe) >= P_SAFE_MIN with confidence.

Rollouts are scripted (a stub `_roll`), so each test knows exactly how many
survive and can check when the dream stops, when it dreams more, and what the
gate makes of it.
"""
import pytest

from cyberdog.sim.sensing import dreaming as D
from cyberdog.sim.vamos_client import GATE, passes

PATH = [(0.0, 0.0), (2.0, 0.0)]
POSE = (0.0, 0.0, 0.0)


class Scripted(D.Dream):
    """A dream whose rollouts survive or hit by a script, all with full room."""

    def __init__(self, outcomes, **kw):
        super().__init__(lambda x, y: 1.0, 1.0, **kw)
        self.outcomes = iter(outcomes)
        self.rolled = 0

    def _roll(self, path, pose, steps):
        self.rolled += 1
        ok = next(self.outcomes)
        return ok, 1.0, ok


def script(first_hits, more_hits=0):
    """N_DREAMS rollouts with `first_hits` hits, then the rest with `more_hits`."""
    n, extra = D.N_DREAMS, D.N_DREAMS_MAX - D.N_DREAMS
    return ([False] * first_hits + [True] * (n - first_hits)
            + [False] * more_hits + [True] * (extra - more_hits))


def test_wilson_bounds():
    lo, hi = D.wilson(24, 24)
    assert 0.89 < lo < 0.91 and hi == 1.0
    lo, hi = D.wilson(0, 24)
    assert lo == 0.0 and hi < 0.11
    lo, hi = D.wilson(20, 24)
    assert lo < 20 / 24 < hi
    assert D.wilson(0, 0) == (0.0, 1.0)


def test_all_survive_passes_without_dreaming_more():
    d = Scripted(script(0))
    f, why = d.factor(PATH, POSE)
    assert d.rolled == D.N_DREAMS and why["n"] == D.N_DREAMS
    assert why["confident"] and passes(f, why)


def test_hopeless_is_refused_without_dreaming_more():
    # Upper bound already under P_SAFE_MIN: no point in more.
    d = Scripted(script(12))
    f, why = d.factor(PATH, POSE)
    assert d.rolled == D.N_DREAMS
    assert D.wilson(12, 24)[1] < D.P_SAFE_MIN
    assert not why["confident"] and not passes(f, why)


def test_one_in_four_hits_is_refused_though_the_factor_passes():
    # B9's case: P(safe) 0.75 with room to spare scores over GATE.
    d = Scripted(script(6, 18))
    f, why = d.factor(PATH, POSE)
    assert d.rolled == D.N_DREAMS_MAX and why["n"] == D.N_DREAMS_MAX
    assert why["p_safe"] == pytest.approx(0.75)
    assert f >= GATE
    assert not passes(f, why)


def test_lucky_first_batch_is_not_passed_on_it():
    # 23/24 meets the bound at 24, but the early pass is all-or-nothing:
    # the rest of the rollouts decide, and a 0.75 path fails on them.
    d = Scripted(script(1, 23))
    f, why = d.factor(PATH, POSE)
    assert d.rolled == D.N_DREAMS_MAX
    assert not passes(f, why)


def test_undecided_then_safe_passes():
    d = Scripted(script(2, 4))                  # 90 of 96 survive
    f, why = d.factor(PATH, POSE)
    assert d.rolled == D.N_DREAMS_MAX
    assert why["p_safe_lo"] >= D.P_SAFE_MIN
    assert why["confident"] and passes(f, why)


def test_n_max_equal_to_n_never_extends():
    d = Scripted(script(2), n_max=D.N_DREAMS)
    _, why = d.factor(PATH, POSE)
    assert d.rolled == D.N_DREAMS and why["n"] == D.N_DREAMS


def test_stand_in_dream_judged_on_factor():
    assert passes(0.6, {})
    assert not passes(0.4, {})
    assert not passes(0.9, {"confident": False})


def test_short_path():
    f, why = D.Dream(lambda x, y: 1.0, 1.0).factor([(0.0, 0.0)], POSE)
    assert f == 0.0 and not passes(f, why)
