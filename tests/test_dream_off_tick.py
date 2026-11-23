"""On the board the dream runs on the VAMOS worker, not on the control tick.

HIL 11-23 11:17 on the J4012: the dream took 81-220 ms a frame on average and
462 ms at worst, inside the brain's 50 ms tick. Now the worker asks and
judges; the tick only picks the judged answer up. The clearance field the
worker dreams against is updated by the loop at the same time, so it must
never show a half-built (cleared) field. And the board counts late ticks.
"""
import concurrent.futures
import threading
import time
import types

import numpy as np

from cyberdog.sim import run_building as rb
from cyberdog.sim.sensing import perception as pc

from test_vlm_worker import Robot

PATH = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


class Policy:
    """request answers at once; deliver records the thread it ran on."""

    def __init__(self, deliver_s=0.0):
        self.deliver_s = deliver_s
        self.threads = []

    def request(self, image, prompt, cam_pose):
        return [PATH]

    def deliver(self, paths, pose, destination):
        self.threads.append(threading.current_thread().name)
        self.pose = pose
        time.sleep(self.deliver_s)
        return paths[0], paths, 0.8


def stub(policy, real_time=True):
    run = types.SimpleNamespace(robot=Robot(), real_time=real_time, vlm_pool=None,
                                vlm_worker=None, pending=None, vlm_latency=0.1,
                                policy=lambda floor: policy,
                                vlm={"asked": 0, "answers": 0, "dropped": 0, "age": 0.0,
                                     "moved": 0.0, "cap_min": None},
                                vamos_steer=None, vlm_asked_t=None, vlm_lat=[],
                                vlm_gen=0, vlm_now=((1.0, 2.0, 0.3), (5.0, 5.0)))
    for name in ("ask_vamos", "ask_and_judge", "take_answer", "vlm_busy"):
        setattr(run, name, getattr(rb.Run, name).__get__(run))
    return run


STATE = {"state": "TRACK", "loc": (500, 600), "destination": (5.0, 5.0)}


def test_board_dreams_on_the_worker_and_the_tick_does_not_wait():
    policy = Policy(deliver_s=0.3)              # a J4012-sized dream
    run = stub(policy)
    run.ask_vamos(1, 0, STATE)
    t0 = time.perf_counter()
    while run.take_answer(1, 1, *run.vlm_now) is None:
        time.sleep(0.01)
        assert time.perf_counter() - t0 < 2.0
    # Every take_answer returned at once while the worker dreamed.
    t0 = time.perf_counter()
    run.ask_vamos(1, 2, STATE)
    assert run.take_answer(1, 3, *run.vlm_now) is None
    assert (time.perf_counter() - t0) * 1000 < rb.TICK_MS
    run.vlm_worker.result(timeout=2.0)
    best, _, safety = run.take_answer(1, 4, *run.vlm_now)
    assert best == PATH and safety == 0.8
    assert policy.threads and all(t.startswith("vamos") for t in policy.threads)
    assert policy.pose == (1.0, 2.0, 0.3)       # the pose the loop published
    run.vlm_pool.shutdown()


def test_the_worker_judges_from_the_latest_published_pose():
    gate = threading.Event()

    class Held(Policy):
        def request(self, image, prompt, cam_pose):
            gate.wait(2.0)
            return [PATH]

    policy = Held()
    run = stub(policy)
    run.ask_vamos(1, 0, STATE)
    run.vlm_now = ((3.0, 4.0, 0.0), (5.0, 5.0))  # the dog walked on meanwhile
    gate.set()
    run.vlm_worker.result(timeout=2.0)
    assert policy.pose == (3.0, 4.0, 0.0)
    run.vlm_pool.shutdown()


def test_twin_still_dreams_on_the_tick():
    policy = Policy()
    run = stub(policy, real_time=False)
    run.ask_vamos(1, 0, STATE)                  # lands at tick 2 (0.1 s / 0.05)
    assert run.take_answer(1, 1, *run.vlm_now) is None
    assert run.take_answer(1, 2, *run.vlm_now)[0] == PATH
    assert policy.threads == [threading.current_thread().name]


def test_live_clearance_is_never_seen_cleared_mid_update():
    lc = pc.LiveClearance(1)
    pts = np.array([[21.0, 9.5, 0.5], [21.1, 9.5, 0.5]])
    lc.update(pts, 0.0, (20.0, 9.5), (20.0, 9.5, 0.0))
    old = lc.local
    assert old is not None
    seen = []
    build = lc._field

    def spying(*a):
        seen.append(lc.local)                   # what a reader sees meanwhile
        return build(*a)

    lc._field = spying
    lc.update(pts + [0.1, 0.0, 0.0], 0.0, (20.0, 9.5), (20.0, 9.5, 0.0))
    assert seen == [old]
    assert lc.local is not old and lc.local is not None
    lc.remember([], 1.0, (20.0, 9.5))
    assert lc.recent is not None
    rec, build_r = lc.recent, lc._remembered
    seen.clear()

    def spying_r(*a):
        seen.append(lc.recent)
        return build_r(*a)

    lc._remembered = spying_r
    lc.remember([], 1.1, (20.0, 9.5))
    assert seen == [rec] and lc.recent is not None


class FakeLink:
    def send(self, *a, **k):
        pass

    def close(self):
        pass


def test_board_counts_late_ticks():
    from cyberdog.sim.robot.go2_board_robot import Go2BoardRobot
    t = [0.0]
    r = Go2BoardRobot.__new__(Go2BoardRobot)
    r.link, r.clock, r.sleep = FakeLink(), (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))
    r._next, r._cmd = 0.05, (0.0, 0.0, 0.0)
    r.ticks, r.late, r.late_ms_max = 0, 0, 0.0
    t[0] = 0.01
    r.step()                                    # on time: sleeps to 0.05
    assert (r.ticks, r.late) == (1, 0) and abs(t[0] - 0.05) < 1e-9
    t[0] = 0.05 + 0.05 + 0.2                    # 200 ms past the next deadline
    r.step()
    assert (r.ticks, r.late) == (2, 1) and abs(r.late_ms_max - 200.0) < 1e-6
    assert r.tick_summary() == ("BRAIN TICK: 1 of 2 ticks late, worst 200 ms "
                                "past the 50 ms deadline")
