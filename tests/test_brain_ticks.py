"""The brain's late ticks on the board, broken down (BrainTicks).

HIL 11-23 12:02 counted 425 of 3253 ticks late, worst 10.2 s, and could not
say why. Now a stretch with no step() (the start, a lift ride: request())
is a gap, not a late tick, and each tick carries what ran on it: a CE-RRT*
search, a VAMOS ask, the worker dreaming during it, and the ms of each.
"""
import threading
import types

from cyberdog.sim import run_building as rb
from cyberdog.sim.robot.go2_board_robot import BrainTicks, Go2BoardRobot


class FakeLink:
    def send(self, *a, **k):
        pass

    def poll(self):
        return {"mode": "STAND"}


def board():
    t = [0.0]
    r = Go2BoardRobot.__new__(Go2BoardRobot)
    r.link, r.clock, r.sleep = FakeLink(), (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))
    r._next, r._cmd = 0.05, (0.0, 0.0, 0.0)
    r._state, r._got, r._history = None, None, []
    r.tick_log, r._ret, r._gap = BrainTicks(0.05), None, True
    return r, t


def test_a_stand_or_a_ride_is_a_gap_not_a_late_tick():
    r, t = board()
    log = r.tick_log
    r.step()                                  # the first step: after the start
    assert (log.ticks, log.late, log.gaps) == (0, 0, 1)
    r.step()                                  # on time
    r.request("stand")                        # the lift: no step() for 10 s
    t[0] += 10.0
    r.step()
    assert (log.ticks, log.late, log.gaps) == (1, 0, 2)
    assert abs(log.gap_s_max - 10.0) < 1e-6
    r.step()                                  # the schedule started afresh: on time
    assert (log.ticks, log.late) == (2, 0)
    t[0] += 0.08                              # the brain worked 80 ms on this tick
    r.step()
    assert (log.ticks, log.late) == (3, 1)
    late, work, _, _ = log.rows[-1]
    assert abs(late - 30.0) < 1e-6 and abs(work - 80.0) < 1e-6


def test_ticks_carry_their_tags_and_the_dream_across_ticks():
    log = BrainTicks(0.05)
    log.tag("plan")
    log.mark("plan", 70.0)
    log.mark("scan", 5.0)
    log.tick(20.0, 75.0)                      # a search: late
    log.begin("dream")                        # the worker starts dreaming
    log.tick(0.0, 10.0)
    log.tick(0.0, 10.0)                       # still dreaming
    log.end("dream")
    log.tick(0.0, 10.0)                       # it ended during this one
    log.tick(0.0, 10.0)                       # over
    log.begin("dream")
    log.end("dream")                          # a short one, inside one tick
    log.tick(0.0, 10.0)
    tags = [set(r[2]) for r in log.rows]
    assert tags == [{"plan"}, {"dream"}, {"dream"}, {"dream"}, set(), {"dream"}]
    assert log.rows[0][3] == {"plan": 70.0, "scan": 5.0} and log.rows[1][3] == {}


def test_the_summary_splits_late_ticks_by_size_and_tag():
    log = BrainTicks(0.05)
    for late in (5.0, 40.0, 300.0):
        log.tag("plan")
        log.mark("plan", late + 50)
        log.tick(late, late + 50)
    log.begin("dream")
    log.end("dream")                          # inside this tick only
    log.tick(15.0, 65.0)
    for _ in range(4):
        log.tick(0.0, 20.0)
    log.gap(9.5)
    lines = log.summary()
    assert lines[0] == ("BRAIN TICK: 4 of 8 ticks late, worst 300 ms past the 50 ms "
                        "deadline; 1 gaps not counted (start, lift rides), longest 9.5 s")
    assert lines[1] == "  late by (ms): 0-10 1, 10-25 1, 25-50 1, 50-100 0, 100-250 0, >250 1"
    assert lines[2].startswith("  nothing: 0 of 4 late, work mean 20.0")
    assert lines[3].startswith("  plan: 3 of 3 late (mean 115, worst 300 ms)")
    assert lines[4].startswith("  dream: 1 of 1 late (mean 15, worst 15 ms)")
    assert lines[5].startswith("  mean ms on late ticks: work ")
    assert "plan 123.8" in lines[5] and "plan 0.0" in lines[6]


def test_the_worker_dreams_inside_begin_and_end():
    log = BrainTicks(0.05)
    seen = []

    class Policy:
        def request(self, image, prompt, cam_pose):
            return [[(0.0, 0.0)]]

        def deliver(self, paths, pose, destination):
            seen.append(dict(log._active))
            return paths[0], paths, 1.0

    run = types.SimpleNamespace(robot=types.SimpleNamespace(board=types.SimpleNamespace(
        tick_log=log)), vlm_gen=0, vlm_now=((0.0, 0.0, 0.0), (1.0, 0.0)))
    t = threading.Thread(target=rb.Run.ask_and_judge,
                         args=(run, Policy(), None, "", (0, 0, 0), 0))
    t.start()
    t.join()
    assert seen == [{"dream": 1}] and not log._active
    log.tick(0.0, 1.0)
    assert log.rows[-1][2] == {"dream"}


def test_the_twin_has_no_tick_log_and_marks_nothing():
    log = rb.tick_log(types.SimpleNamespace())
    log.tag("plan")
    log.mark("plan", 1.0)
    log.begin("dream")
    log.end("dream")
