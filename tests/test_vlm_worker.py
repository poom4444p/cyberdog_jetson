"""On a real-time robot (go2_board) the VAMOS call runs on a worker thread:
the control loop goes on ticking while it runs, and the answer lands when done."""
import threading
import types

import numpy as np

from cyberdog.sim import run_building as rb


class SlowVamos:
    def __init__(self):
        self.go = threading.Event()
        self.started = threading.Event()

    def request(self, image, prompt, cam_pose):
        self.started.set()
        self.go.wait(2.0)
        return [[(1.0, 2.0), (3.0, 4.0)]]

    def deliver(self, paths, pose, destination):
        return paths[0], paths, 0.9


class Robot:
    CONTROL_DT = 0.05
    REAL_TIME = True

    def get_image(self):
        return np.zeros((4, 4, 3), np.uint8)

    def camera_pose(self):
        return (0.0, 0.0, 0.0)

    def get_pose(self):
        return (0.0, 0.0, 0.0)

    def now(self):
        return self.t

    t = 0.0


def stub():
    vamos = SlowVamos()
    run = types.SimpleNamespace(robot=Robot(), real_time=True, vlm_pool=None, vlm_worker=None,
                                pending=None, vlm_latency=None, policy=lambda floor: vamos,
                                vlm={"asked": 0, "answers": 0, "dropped": 0, "age": 0.0,
                                     "moved": 0.0, "cap_min": None},
                                vamos_steer=None, vlm_asked_t=None, vlm_lat=[])
    for name in ("ask_vamos", "take_answer", "vlm_busy", "vlm_speed_cap"):
        setattr(run, name, getattr(rb.Run, name).__get__(run))
    return run, vamos


def test_the_loop_is_not_held_while_vamos_thinks():
    run, vamos = stub()
    run.ask_vamos(1, 0, {"state": "TRACK", "loc": (500, 600), "destination": (5.0, 5.0)})
    assert vamos.started.wait(1.0)
    for n in range(1, 5):                       # ticks go on: no answer yet, not blocked
        assert run.take_answer(1, n, (0, 0, 0), (5, 5)) is None
    assert run.vlm_busy()
    vamos.go.set()
    run.vlm_worker.result(timeout=1.0)
    chosen, _, safety = run.take_answer(1, 9, (0, 0, 0), (5, 5))
    assert chosen == [(1.0, 2.0), (3.0, 4.0)] and safety == 0.9
    assert run.pending is None and not run.vlm_busy() and run.vlm["answers"] == 1
    run.vlm_pool.shutdown()


def test_a_dropped_answer_still_holds_the_worker():
    run, vamos = stub()
    run.ask_vamos(1, 0, {"state": "TRACK", "loc": (500, 600), "destination": (5.0, 5.0)})
    assert vamos.started.wait(1.0)
    run.pending = None                          # the leg ended: answer dropped
    assert run.vlm_busy()                       # but the GPU is still on it
    vamos.go.set()
    run.vlm_worker.result(timeout=1.0)
    assert not run.vlm_busy()
    run.vlm_pool.shutdown()


def test_vlm_latency_option_takes_seconds_or_measured():
    # The twin's default is the dog's latency, not the call's own wall-clock
    # time: from the Mac over the tunnel that was ~4.7 s and 19 of 25 paths
    # arrived already behind the dog (cafeteria, twin + VAMOS).
    assert rb.vlm_latency_arg("1.8") == 1.8
    assert rb.vlm_latency_arg("measured") is None
    assert 2.5 <= rb.VLM_LATENCY <= 3.2                  # HIL 11-11: 2.7-3.2 s


def test_latency_is_measured_on_robot_time():
    run, vamos = stub()
    run.robot.t = 10.0
    run.ask_vamos(1, 0, {"state": "TRACK", "loc": (500, 600), "destination": (5.0, 5.0)})
    vamos.go.set()
    run.vlm_worker.result(timeout=1.0)
    run.robot.t = 12.6
    assert run.take_answer(1, 30, (0, 0, 0), (5, 5)) is not None
    assert len(run.vlm_lat) == 1 and abs(run.vlm_lat[0] - 2.6) < 1e-9
    run.vlm_pool.shutdown()


def test_speed_cap_follows_the_vamos_cycle():
    run, _ = stub()
    # Before any answer: the expected latency, here VLM_LATENCY (2.8 s).
    assert abs(run.vlm_speed_cap() - 2.9 / 5.6) < 1e-9
    run.vlm_lat = [2.6]                         # J4012 with the shared prefill
    assert abs(run.vlm_speed_cap() - 2.9 / 5.2) < 1e-9      # ~0.56 m/s
    run.vlm_lat = [2.6, 4.6]                    # one slow answer: the slowest counts
    assert abs(run.vlm_speed_cap() - 2.9 / 9.2) < 1e-9
    assert run.vlm["cap_min"] == run.vlm_speed_cap()
    run.vlm_latency = 1.8                       # --vlm-latency is the default only
    run.vlm_lat = []
    assert abs(run.vlm_speed_cap() - 2.9 / 3.6) < 1e-9


def test_vamos_is_asked_back_to_back():
    # No tick schedule left: the only gate is a call already in flight.
    assert not hasattr(rb, "REPLAN_EVERY") and not hasattr(rb, "REPLAN_BLOCKED")
    assert rb.VLM_REACH_D - rb.VLM_STOP_D > 0
