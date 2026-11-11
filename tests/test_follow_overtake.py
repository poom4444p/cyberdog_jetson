"""Somebody slower ahead in the dog's lane, walking its way: followed at
FOLLOW_GAP, or overtaken on the left when that is worth it and clear --
not "Someone ahead. Waiting." (see run_building.FOLLOW_GAP)."""
import math
from types import SimpleNamespace

from cyberdog.sim import run_building as rb

LANE_Y = 8.87
ROUTE = [(20.0, LANE_Y), (43.6, LANE_Y), (45.0, 9.5)]   # floor 1, east, right-hand lane
TURN = [(20.0, LANE_Y), (43.6, LANE_Y), (43.6, 4.0)]    # the same, turning right at 43.6
CRUISE = 0.8
DT = 0.05


def person(x, y, vx, vy=0.0):
    return SimpleNamespace(x=x, y=y, vx=vx, vy=vy, r=0.22, moving=True, walked=True)


class Dog:
    """Run.follow's state, and the dog it steers: walks at the speed asked
    (cruise unless capped), towards the lane point shifted as asked."""

    def __init__(self, x, y=LANE_Y):
        self.x, self.y, self.yaw = x, y, 0.0
        self.t, self.said = 0.0, []
        self.robot = SimpleNamespace(now=lambda: self.t, CONTROL_DT=DT)
        self.passing_lead, self.follow_said_at, self.pass_said_at = None, -rb.INF, -rb.INF
        self.lead_track, self.lead_t, self.lead_v, self.lead_seen = None, 0.0, 0.0, 0.0

    def say(self, line):
        self.said.append(line)

    def tick(self, tracks, route=ROUTE, i=1, gates=()):
        cap, shift, lead = rb.Run.follow_lead(self, tracks, (self.x, self.y), self.yaw,
                                         route, i, gates, CRUISE)
        tx, ty = rb.lane_point((self.x, self.y), route, i)
        ty += shift                                    # lane is east: left is +y
        v = CRUISE if cap is None else min(CRUISE, cap)
        h = math.atan2(ty - self.y, tx - self.x)
        self.x += v * math.cos(h) * DT
        self.y += v * math.sin(h) * DT
        self.yaw = h
        self.t += DT
        return cap, shift, lead


def walk(p, dt=DT):
    p.x += p.vx * dt
    p.y += p.vy * dt


def test_follows_a_slightly_slower_person_at_the_gap():
    dog, p = Dog(22.0), person(26.0, LANE_Y, 0.65)     # 0.15 m/s slower: not worth passing
    for _ in range(int(25 / DT)):                       # they reach x 42: the route ends at 43.6
        dog.tick([p])
        walk(p)
    gap = p.x - dog.x
    assert abs(gap - rb.FOLLOW_GAP) < 0.1
    assert abs(dog.y - LANE_Y) < 0.05                  # stayed in its lane
    assert dog.said == ["Following someone ahead."]


def test_overtakes_on_the_left_and_moves_back_into_its_lane():
    dog, p = Dog(22.0), person(25.0, LANE_Y, 0.4)
    least, widest = rb.INF, 0.0
    for _ in range(int(22 / DT)):                       # the dog reaches x ~39
        dog.tick([p])
        walk(p)
        least = min(least, math.dist((dog.x, dog.y), (p.x, p.y)))
        widest = max(widest, dog.y - LANE_Y)
    assert dog.x > p.x + rb.OVERTAKE_BACK              # past them
    assert abs(dog.y - LANE_Y) < 0.1                   # and back in its lane
    assert widest > 1.0                                # went by on the left
    assert least >= rb.FOLLOW_MIN
    assert dog.said == ["Passing someone on your left."]


def test_someone_coming_the_other_way_on_the_left_blocks_the_overtake():
    dog, p = Dog(22.0), person(25.0, LANE_Y, 0.4)
    oncoming = person(31.0, LANE_Y + 1.2, -0.9)
    cap, shift, lead = dog.tick([p, oncoming])
    assert lead is p and shift == 0.0 and cap is not None
    for _ in range(int(25 / DT)):                       # once they are by, it goes
        dog.tick([p, oncoming])
        walk(p)
        walk(oncoming)
    assert dog.said == ["Following someone ahead.", "Passing someone on your left."]
    assert dog.x > p.x


def test_a_corner_within_the_pass_blocks_it():
    dog = Dog(38.0)                                     # 5.6 m before the corner at 43.6
    p = person(40.0, LANE_Y, 0.4)
    cap, shift, lead = dog.tick([p], route=TURN)
    assert lead is p and shift == 0.0
    assert dog.said == ["Following someone ahead."]


def test_a_door_within_the_pass_blocks_it():
    door = ((30.0, 9.6), (30.0, 10.4), (0.0, 1.0))     # entry, exit, direction
    dog, p = Dog(22.0), person(25.0, LANE_Y, 0.4)
    cap, shift, _ = dog.tick([p], gates=[door])
    assert shift == 0.0 and cap is not None


def test_never_closer_than_follow_min():
    assert rb.follow_speed(rb.FOLLOW_MIN, 0.5) == 0.0
    assert rb.follow_speed(0.6, 0.5) == 0.0
    assert rb.follow_speed(rb.FOLLOW_GAP, 0.5) == 0.5
    assert abs(rb.follow_speed(1.2, 0.2) - 0.05) < 1e-9           # closing too fast: slower than them
    # Somebody 1.3 m in front who slows from 0.5 to 0.3 m/s, with somebody
    # standing in the left lane so it cannot overtake: it drops back, never
    # nearer than FOLLOW_MIN (less one tick of CRUISE * DT).
    dog, p = Dog(22.0), person(23.3, LANE_Y, 0.5)
    left = person(25.5, LANE_Y + 1.1, 0.0)
    left.moving = False
    least = rb.INF
    for k in range(int(15 / DT)):
        if k == 10:
            p.vx = rb.CROSS_V
        cap, shift, lead = dog.tick([p, left])
        assert shift == 0.0
        walk(p)
        least = min(least, p.x - dog.x)
    assert least >= rb.FOLLOW_MIN - CRUISE * DT
    assert abs((p.x - dog.x) - rb.FOLLOW_GAP) < 0.1
    # (somebody who stops dead is not a leader any more -- leader() wants them
    # walking the dog's way -- and is the stand-still rules' business)


def test_who_is_a_leader():
    u = (1.0, 0.0)
    dog = (30.0, LANE_Y)
    assert rb.leader(dog, u, [person(33.0, LANE_Y + 0.2, 0.5)], CRUISE)
    assert not rb.leader(dog, u, [person(33.0, LANE_Y + 1.2, 0.5)], CRUISE)    # other lane
    assert not rb.leader(dog, u, [person(33.0, LANE_Y, -0.8)], CRUISE)         # oncoming
    assert not rb.leader(dog, u, [person(33.0, LANE_Y, 1.2)], CRUISE)          # pulling away
    assert not rb.leader(dog, u, [person(28.0, LANE_Y, 0.5)], CRUISE)          # behind
    assert not rb.leader(dog, u, [person(37.0, LANE_Y, 0.5)], CRUISE)          # beyond FOLLOW_LOOK
    assert not rb.leader(dog, u, [person(33.0, LANE_Y, 0.4, 0.6)], CRUISE)     # crossing
    near, far = person(32.0, LANE_Y, 0.5), person(34.0, LANE_Y, 0.5)
    assert rb.leader(dog, u, [far, near], CRUISE) is near
    assert not rb.leader(dog, u, [person(33.0, LANE_Y, CRUISE)], CRUISE)      # the dog's pace


def test_somebody_overtaking_level_with_the_dog_is_not_followed():
    # HIL 11-10 15:42, t 35.13: 0.09 m ahead, 0.68 m left, 0.86 m/s -- the
    # dog stood still for them. Slower and level is not a leader either.
    dog, u = (36.58, 8.92), (1.0, 0.0)
    assert not rb.leader(dog, u, [person(36.67, 9.60, 0.86, 0.01)], CRUISE, 0.05)
    assert not rb.leader(dog, u, [person(36.90, 9.30, 0.5)], CRUISE, 0.05)


def test_said_only_once_it_holds_the_dog_back():
    # HIL 11-10 15:42, t 21.0: 5.6 m ahead at 0.65 m/s -- no difference yet.
    dog = Dog(28.34)
    dog.y = 9.08
    p = person(33.92, LANE_Y + 0.3, 0.65)
    blocker = person(36.0, LANE_Y + 1.2, 0.0)          # no overtaking: something on the left
    blocker.moving = False
    cap, shift, lead = dog.tick([p, blocker])
    assert lead is p and cap > CRUISE and dog.said == []


def test_straight_ahead_stops_at_the_corner():
    assert abs(rb.straight_ahead((40.0, LANE_Y), TURN[1:]) - 3.6) < 1e-9
    assert rb.straight_ahead((40.0, LANE_Y), ROUTE[1:]) > 5.0     # 24 deg: not a corner
    assert rb.straight_ahead((40.0, LANE_Y), [(50.0, LANE_Y)]) == 10.0


def test_a_pass_under_way_is_not_called_off_by_a_noisy_pace():
    # HIL 11-10 16:11: somebody walking 0.42 read 0.3-0.6 m/s, and every tick
    # they read over 0.55 ended the pass, and the next began it again: said 8 times.
    dog, p = Dog(22.0), person(25.0, LANE_Y, 0.42)
    paces = [0.42, 0.6, 0.3, 0.58, 0.45]
    for k in range(int(25 / DT)):
        p.vx = paces[k % len(paces)]
        dog.tick([p])
        p.x += 0.42 * DT
    assert dog.said == ["Passing someone on your left."]
    assert dog.x > p.x + rb.OVERTAKE_BACK


def pace_after(seconds, v_cmd, achieved, w=0.0, dt=0.05):
    p, x = rb.Pace(), 0.0
    for k in range(int(seconds / dt)):
        p.update(k * dt, x, 0.0, 0.0, (v_cmd, 0.0, w))
        x += v_cmd * achieved * dt
    return p.ratio


def test_pace_is_what_the_dog_walks_for_what_it_is_told():
    assert abs(pace_after(10.0, 0.8, 0.83) - 0.83) < 0.01        # HIL: physics at 0.83x
    assert pace_after(10.0, 0.8, 1.0) == 1.0
    assert pace_after(1.0, 0.8, 0.83) == 1.0                      # not seen long enough
    assert pace_after(10.0, 0.8, 0.83, w=0.5) == 1.0              # turning: not counted
    assert pace_after(10.0, 0.2, 0.83) == 1.0                     # creeping: not counted
    assert pace_after(10.0, 0.8, 0.1) == rb.PACE_LO               # stalled: not a pace


def test_the_1613_walker_is_followed_judged_on_the_dogs_real_pace():
    # 0.65 m/s of sim time read 0.54 by the brain; the dog's 0.8 walked 0.67.
    u, dog = (1.0, 0.0), (29.5, LANE_Y)
    p = person(35.0, LANE_Y, 0.54)
    assert rb.overtake_plan(dog, u, p, [p], 0.8) is not None      # on the command: overtaken
    assert rb.overtake_plan(dog, u, p, [p], 0.8 * 0.83) is None   # on what it walks: followed


def test_nobody_is_overtaken_before_the_pace_is_known():
    dog, p = Dog(22.0), person(25.0, LANE_Y, 0.4)
    cap, shift, lead = rb.Run.follow_lead(dog, [p], (dog.x, dog.y), 0.0, ROUTE, 1, (),
                                          CRUISE, can_pass=False)
    assert lead is p and shift == 0.0
    assert not rb.Pace().known


def test_somebody_just_overtaken_falling_back_is_not_passing_the_dog():
    # HIL 11-10 16:19: "Someone passing on your right" -- the one it had passed.
    from types import SimpleNamespace as NS
    f = NS(said=[], overtakers_said=[], overtake_said_at={},
           robot=NS(now=lambda: 0.0))
    f.say = f.said.append
    behind = person(28.0, LANE_Y, 0.45)
    rb.Run.mention_overtakers(f, [behind], (29.0, LANE_Y + 1.0), 0.0, 0.66)
    assert f.said == []
    rb.Run.mention_overtakers(f, [person(28.0, LANE_Y, 1.0)], (29.0, LANE_Y + 1.0), 0.0, 0.66)
    assert f.said == ["Someone passing on your right."]


def test_one_slow_reading_of_their_pace_does_not_start_an_overtake():
    # HIL 11-10 16:26: somebody walking 0.54 (brain clock) read 0.37-0.80
    # tick to tick; the dog walking 0.66 passed on one 0.41.
    dog, p = Dog(22.0), person(27.7, LANE_Y, 0.54)
    paces = [0.61, 0.64, 0.80, 0.60, 0.41, 0.51, 0.60, 0.47, 0.45, 0.42, 0.48, 0.37, 0.53]
    for k in range(int(10 / DT)):
        p.vx = paces[k % len(paces)]
        rb.Run.follow_lead(dog, [p], (dog.x, dog.y), 0.0, ROUTE, 1, (), 0.66)
        dog.t += DT
        p.x += 0.54 * DT
        dog.x += 0.54 * DT
    assert "Passing someone on your left." not in dog.said
    assert abs(dog.lead_v - 0.54) < 0.06


def test_a_slow_walker_dipping_under_cross_v_is_still_overtaken():
    # HIL 11-11 09:37 (--walker 0.4): walking 0.33 (brain clock), read
    # 0.24-0.41 tick to tick; every reading under CROSS_V dropped them as the
    # leader, LEAD_SEEN restarted, and the dog followed them 16 s to the lift.
    dog, p = Dog(22.0), person(25.0, LANE_Y, 0.33)
    paces = [0.40, 0.30, 0.33, 0.24, 0.40, 0.31, 0.34, 0.26, 0.37, 0.29]
    for k in range(int(25 / DT)):
        p.vx = paces[k % len(paces)]
        dog.tick([p])
        p.x += 0.33 * DT
    assert dog.said[-1] == "Passing someone on your left."
    assert dog.said.count("Passing someone on your left.") == 1
    assert dog.x > p.x + rb.OVERTAKE_BACK


def test_only_the_one_being_followed_stays_leader_under_cross_v():
    u, dog = (1.0, 0.0), (30.0, LANE_Y)
    slow = person(33.0, LANE_Y, 0.2)
    assert rb.leader(dog, u, [slow], CRUISE) is None               # not a new leader
    assert rb.leader(dog, u, [slow], CRUISE, keep=slow) is slow    # kept as one
    slow.vx = 0.05
    assert rb.leader(dog, u, [slow], CRUISE, keep=slow) is None    # stopping: not any more


def test_a_missed_tick_does_not_restart_watching_the_leader():
    dog, p = Dog(22.0), person(26.0, LANE_Y, 0.5)
    for _ in range(10):
        dog.tick([p])
        walk(p)
    seen = dog.lead_seen
    dog.tick([])                                       # one tick without them
    walk(p)
    dog.tick([p])
    assert dog.lead_seen > seen                        # carried on, not restarted
    for _ in range(int(rb.LEAD_GRACE / DT) + 2):       # gone longer than LEAD_GRACE
        dog.tick([])
        walk(p)
    dog.tick([p])
    assert dog.lead_seen == 0.0
