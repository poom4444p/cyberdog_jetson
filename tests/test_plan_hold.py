"""While a plan is walked, CE-RRT* is not searched again on every replan
(PLAN_HOLD_S): the plan is checked with ce_rrt.still_good and kept."""
import types

from cyberdog.planning import ce_rrt
from cyberdog.sim import run_building as rb


def stub(plan=None, searched=None):
    run = types.SimpleNamespace(plan=plan, searched_at=searched)
    run.plan_hold = rb.Run.plan_hold.__get__(run)
    return run


def test_no_hold_without_a_plan():
    # With no plan the first search (and PLAN_RETRY_S's backoff) is unchanged.
    assert not stub(searched=100).plan_hold(101)


def test_no_hold_before_any_search():
    assert not stub(plan=object()).plan_hold(101)


def test_hold_lasts_plan_hold_s_in_ticks():
    hold = round(rb.PLAN_HOLD_S * rb.FPS)
    replan = round(ce_rrt.REPLAN_S * rb.FPS)
    assert hold > replan          # still checked at the replan rate in between
    run = stub(plan=object(), searched=100)
    assert all(run.plan_hold(100 + k) for k in range(hold))
    assert not run.plan_hold(100 + hold)


def test_still_good_rejects_a_plan_that_has_turned_unsafe():
    # The hold rests on still_good: a plan whose way is now blocked is not kept.
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    def ok(x, y, yaw=0.0):
        return True
    assert ce_rrt.still_good((0.0, 0.0, 0.0), path, ok, [])
    def wall(x, y, yaw=0.0):
        return not 0.9 < x < 1.1
    assert not ce_rrt.still_good((0.0, 0.0, 0.0), path, wall, [])
