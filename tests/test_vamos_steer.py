"""Tests for the language preference in the VAMOS prompt (--vamos-steer, spec L4 s3)."""
import sys

import pytest

from cyberdog.planning.checkpoint_projector import vamos_prompt
from cyberdog.sim import run_building as rb
from test_vlm_worker import stub

TRACK = {"state": "TRACK", "loc": (444, 526), "destination": (5.0, 5.0)}
GOAL = "Navigate to x=<loc0444>, y=<loc0526>."


class TestPrompt:
    def test_off_is_the_goal_alone(self):
        assert vamos_prompt(TRACK) == GOAL
        assert vamos_prompt(TRACK, None) == GOAL

    def test_blank_is_off(self):
        assert vamos_prompt(TRACK, "") == GOAL
        assert vamos_prompt(TRACK, "   \n ") == GOAL

    def test_preference_follows_the_goal_like_upstream(self):
        # vendor/VAMOS/vamos_demo.py's default prompt.
        steer = ("Make a sharp right arc to avoid the people in front. Move along that "
                 "arc towards the goal keeping the people to the left of you.")
        assert vamos_prompt(TRACK, steer) == f"{GOAL} {steer}"

    def test_missing_full_stop_is_added(self):
        assert vamos_prompt(TRACK, "keep to the left wall") == f"{GOAL} keep to the left wall."
        assert vamos_prompt(TRACK, "Go left!") == f"{GOAL} Go left!"

    def test_whitespace_is_collapsed(self):
        assert vamos_prompt(TRACK, "  keep  left\n now. ") == f"{GOAL} keep left now."


class Recorder:
    def __init__(self):
        self.prompts = []

    def request(self, image, prompt, cam_pose):
        self.prompts.append(prompt)
        return []

    def deliver(self, paths, pose, destination):   # the board's worker judges too
        return None, [], 0.0


@pytest.mark.parametrize("real_time", [True, False])
def test_ask_vamos_sends_the_preference(real_time):
    run, _ = stub()
    rec = Recorder()
    run.policy = lambda floor: rec
    run.real_time = real_time
    run.robot.REAL_TIME = real_time
    run.vamos_steer = "Keep to the left wall."
    run.ask_vamos(1, 0, TRACK)
    if real_time:
        run.vlm_worker.result(timeout=1.0)
        run.vlm_pool.shutdown()
    assert rec.prompts == [f"{GOAL} Keep to the left wall."]


def test_ask_vamos_without_preference_is_unchanged():
    run, _ = stub()
    rec = Recorder()
    run.policy = lambda floor: rec
    run.real_time = False
    run.ask_vamos(1, 0, TRACK)
    assert rec.prompts == [GOAL]


def test_cli_needs_vamos_or_shadow(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_building", "--no-vamos", "--vamos-steer", "left"])
    with pytest.raises(SystemExit) as e:
        rb.main()
    assert e.value.code == 2
    assert "--vamos-steer" in capsys.readouterr().err
