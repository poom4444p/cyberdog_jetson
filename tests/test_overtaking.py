"""People coming up behind the dog the way it is going are let past, not
waited for (HIL 11-10 11:24: 5 s stood still, stepping left and right)."""
import math
from types import SimpleNamespace

from cyberdog.sim import run_building as rb
from cyberdog.sim.run_building import overtaking

DOG = (32.09, 9.02)
EAST = math.radians(-2)


def track(fwd, lat, v_fwd, v_lat=0.0, moving=True, walked=True, dog=DOG, yaw=EAST):
    c, s = math.cos(yaw), math.sin(yaw)
    return SimpleNamespace(x=dog[0] + fwd * c - lat * s, y=dog[1] + fwd * s + lat * c,
                           vx=v_fwd * c - v_lat * s, vy=v_fwd * s + v_lat * c,
                           moving=moving, walked=walked, r=0.2)


def test_the_1124_overtakers_are_let_past():
    # as the track log had them at t 31.6-36.1
    assert overtaking(DOG, EAST, track(-2.4, +0.97, 0.46, moving=False), 0.8)
    assert overtaking(DOG, EAST, track(-3.0, +0.05, 1.18), 0.8)
    assert overtaking(DOG, EAST, track(-1.2, +0.22, 0.48, moving=False), 0.0)


def test_alongside_or_ahead_and_pulling_away_is_let_go():
    assert overtaking(DOG, EAST, track(+0.3, +0.6, 1.1), 0.8)
    assert overtaking(DOG, EAST, track(+1.0, +0.21, 1.32), 0.0)     # 11:45, t 36.4
    assert not overtaking(DOG, EAST, track(+0.3, +0.6, 0.6), 0.8)   # being passed by the dog
    assert not overtaking(DOG, EAST, track(+1.0, +0.2, 0.85), 0.8)  # ahead, not leaving: usual rules


def test_not_overtaking():
    assert not overtaking(DOG, EAST, track(-2.0, +1.0, -0.8), 0.8)          # coming towards
    assert not overtaking(DOG, EAST, track(-1.5, +1.2, 0.4, v_lat=-0.6), 0.8)  # cutting across
    assert not overtaking(DOG, EAST, track(-1.5, +0.5, 0.1), 0.8)           # standing behind
    assert not overtaking(DOG, EAST, track(-1.5, +0.5, 1.0, moving=False, walked=False), 0.8)


class Fake:
    def __init__(self):
        self.t, self.said = 0.0, []
        self.robot = SimpleNamespace(now=lambda: self.t)
        self.overtakers_said = []
        self.overtake_said_at = {}

    def say(self, line):
        self.said.append((self.t, line))


def test_passing_is_said_once_a_person_and_on_the_right_side():
    f, t = Fake(), track(-2.0, +0.8, 1.0)
    for k in range(200):                      # 10 s of one person beside the dog
        f.t = k * 0.05
        t.x, t.y = track(-2.0 + k * 0.01, +0.8, 1.0).x, track(-2.0 + k * 0.01, +0.8, 1.0).y
        rb.Run.mention_overtakers(f, [t], DOG, EAST)
    assert [line for _, line in f.said] == ["Someone passing on your left."]
    other = track(-1.0, -0.7, 1.0)            # somebody else, on the right
    rb.Run.mention_overtakers(f, [t, other], DOG, EAST)
    assert [line for _, line in f.said][-1] == "Someone passing on your right."


def test_straight_behind_is_not_said_until_a_side_shows():
    f = Fake()
    rb.Run.mention_overtakers(f, [track(-1.5, +0.1, 1.0)], DOG, EAST)
    assert f.said == []
    f.t = 0.5
    rb.Run.mention_overtakers(f, [track(-1.0, -0.6, 1.0)], DOG, EAST)
    assert [line for _, line in f.said] == ["Someone passing on your right."]


def test_a_track_that_breaks_and_comes_back_is_not_said_again():
    f = Fake()
    for k in range(6):                        # the same person, a new track each second
        f.t = float(k)
        rb.Run.mention_overtakers(f, [track(-1.5 + 0.2 * k, +0.8, 1.0)], DOG, EAST)
    assert [line for _, line in f.said] == ["Someone passing on your left."]
    f.t = 20.0                                # somebody else, much later
    rb.Run.mention_overtakers(f, [track(-1.5, +0.8, 1.0)], DOG, EAST)
    assert len(f.said) == 2
