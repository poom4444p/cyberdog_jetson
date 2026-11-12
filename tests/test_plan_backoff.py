"""CE-RRT* is not searched again on every tick after a search found nothing
(PLAN_RETRY_S), and each search tries one of ce_rrt.SPEEDS."""
import types

from cyberdog.planning import ce_rrt
from cyberdog.sim import run_building as rb


def stub(plan=None, failed=None):
    run = types.SimpleNamespace(plan=plan, plan_failed=failed)
    run.plan_backoff = rb.Run.plan_backoff.__get__(run)
    return run


def test_no_backoff_before_any_failure():
    assert not stub().plan_backoff(100)


def test_backoff_lasts_plan_retry_s_in_ticks():
    wait = round(rb.PLAN_RETRY_S * rb.FPS)
    assert wait >= 2 * round(ce_rrt.REPLAN_S * rb.FPS)   # slower than the replan rate
    run = stub(failed=100)
    assert all(run.plan_backoff(100 + k) for k in range(wait))
    assert not run.plan_backoff(100 + wait)


def test_a_plan_being_walked_is_never_held_back():
    assert not stub(plan=object(), failed=100).plan_backoff(101)


def test_speeds_still_offer_a_slower_pass():
    assert len(ce_rrt.SPEEDS) >= 2 and ce_rrt.SPEEDS[0] > ce_rrt.SPEEDS[-1]
