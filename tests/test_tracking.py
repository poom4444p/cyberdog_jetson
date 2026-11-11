"""Tracker: coasting must not make a track moving on its own (HIL 11-11 phantoms)."""
import math

from cyberdog.sim.sensing.tracking import MISS, WINDOW, Tracker

DT = 0.05


def blob(x, y, r=0.25, n=12):
    """A person-sized ring of returns round (x, y)."""
    return [(x + r * math.cos(2 * math.pi * k / n), y + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


def test_server_room_phantom_not_moving():
    # 09:45 replay: a half-seen static thing whose centroid settles over three
    # ticks (47.02 -> 46.91, 5.12 -> 5.04), then drops out of the returns.
    tr = Tracker(DT)
    for x, y in [(47.02, 5.12), (47.02, 5.12), (46.91, 5.04), (46.91, 5.04)]:
        tr.update(blob(x, y))
    for _ in range(MISS):
        tracks = tr.update([])
        assert not any(t.moving for t in tracks)


def test_steady_walker_called_moving():
    tr = Tracker(DT)
    moving_at = None
    for i in range(20):
        tracks = tr.update(blob(40.0 + 1.0 * DT * i, 9.0))
        if moving_at is None and any(t.moving for t in tracks):
            moving_at = i
    assert moving_at == WINDOW - 1


def test_walked_person_coasts_along_velocity():
    # Someone already known to walk, hidden for a few ticks: still predicted
    # walking, and picked up again where they got to.
    tr = Tracker(DT)
    for i in range(WINDOW + 2):
        tr.update(blob(40.0 + 1.0 * DT * i, 9.0))
    i = WINDOW + 2
    for _ in range(3):
        tracks = tr.update([])
        i += 1
        assert len(tracks) == 1 and tracks[0].moving
    assert tracks[0].x > 40.0 + 1.0 * DT * (WINDOW + 1) + 0.1
    tracks = tr.update(blob(40.0 + 1.0 * DT * i, 9.0))
    assert len(tracks) == 1 and tracks[0].moving


def test_unproven_walker_lost_briefly_still_found_moving():
    # A real walker lost for 3 ticks before earning `moving`: held in place,
    # matched again, and moving once the window measures real ticks.
    tr = Tracker(DT)
    seen = [i for i in range(25) if i not in (4, 5, 6)]
    moving = False
    for i in range(25):
        pts = blob(40.0 + 1.0 * DT * i, 9.0) if i in seen else []
        tracks = tr.update(pts)
        moving = moving or any(t.moving for t in tracks)
        assert len(tracks) == 1
    assert moving and tracks[0].walked
