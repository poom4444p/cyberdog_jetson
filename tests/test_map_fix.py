"""The board's pose put back on the map (sensing/map_fix.py)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cyberdog.sim.sensing.map_fix import MapFix

RES = 0.05


def corridor():
    """A 16 x 3 m room inside 0.15 m walls, faces at y = 1 and 4, x = 2 and 18;
    `faces` built as clearance.wall_face_field builds it: an EDT of the wall
    cells, 0 all through the walls."""
    from scipy import ndimage
    W, H = int(20 / RES), int(6 / RES)
    xs = (np.arange(W) + 0.5) * RES
    ys = (H - 1.5 - np.arange(H)) * RES              # row centres, as LiveClearance reads them
    X, Y = np.meshgrid(xs, ys)
    inside = (X > 2.0) & (X < 18.0) & (Y > 1.0) & (Y < 4.0)
    shell = (X > 1.85) & (X < 18.15) & (Y > 0.85) & (Y < 4.15)
    occ = shell & ~inside
    d = ndimage.distance_transform_edt(~occ) * RES
    return SimpleNamespace(faces=d, res=RES, ox=0.0, oy=0.0, H=H)


def wall_scan(dog=(8.0, 2.5), n=3000, seed=0):
    """Returns on the walls, as a true pose would put them."""
    rng = np.random.default_rng(seed)
    k = n // 4
    xs = rng.uniform(2.0, 18.0, 2 * k)
    ys = rng.uniform(1.0, 4.0, 2 * k)
    pts = np.concatenate([np.column_stack([xs[:k], np.full(k, 1.0)]),
                          np.column_stack([xs[k:], np.full(k, 4.0)]),
                          np.column_stack([np.full(k, 2.0), ys[:k]]),
                          np.column_stack([np.full(k, 18.0), ys[k:]])])
    return np.column_stack([pts, np.full(len(pts), 0.8)])


def off(p, centre, dx, dy, dth):
    """p as a drifted pose puts it: turned dth about the dog, moved (dx, dy)."""
    c, s = math.cos(dth), math.sin(dth)
    q = p.copy()
    q[:, 0] = centre[0] + c * (p[:, 0] - centre[0]) - s * (p[:, 1] - centre[1]) + dx
    q[:, 1] = centre[1] + s * (p[:, 0] - centre[0]) + c * (p[:, 1] - centre[1]) + dy
    return q



def test_a_drifted_scan_is_put_back_on_the_walls():
    f, mf, dog = corridor(), MapFix(), (8.0, 2.5)
    truth = wall_scan(dog)
    seen = off(truth, dog, 0.15, -0.12, math.radians(2.0))
    assert np.median(np.hypot(seen[:, 0] - truth[:, 0], seen[:, 1] - truth[:, 1])) > 0.15
    mf.use(f)
    out = seen
    for _ in range(5):                       # five scans, the same drift
        out = mf.refine(mf.points(seen), dog)
    err = np.hypot(out[:, 0] - truth[:, 0], out[:, 1] - truth[:, 1])
    assert np.median(err) < 0.03
    assert mf.used == 5 and mf.resets == 0


def test_a_true_scan_is_left_where_it_is():
    f, mf, dog = corridor(), MapFix(), (8.0, 2.5)
    truth = wall_scan(dog)
    mf.use(f)
    out = mf.refine(truth, dog)
    assert np.abs(out[:, :2] - truth[:, :2]).max() < 0.02


def test_a_person_by_the_wall_stays_off_it():
    # Rigid: whatever moves the walls moves everybody with them. Somebody
    # 0.45 m off the south wall in a drifted scan is 0.45 m off it after.
    f, mf, dog = corridor(), MapFix(), (8.0, 2.5)
    person = np.array([[9.0 + 0.2 * math.cos(a), 1.45 + 0.2 * math.sin(a), 1.0]
                       for a in np.linspace(0, 2 * math.pi, 40)])
    truth = np.concatenate([wall_scan(dog), person])
    seen = off(truth, dog, 0.0, -0.2, 0.0)
    mf.use(f)
    for _ in range(4):
        out = mf.refine(mf.points(seen), dog)
    got = out[-40:, :2].mean(axis=0)
    assert got == pytest.approx((9.0, 1.45), abs=0.03)


def test_too_few_wall_returns_change_nothing():
    f, mf, dog = corridor(), MapFix(), (8.0, 2.5)
    few = wall_scan(dog, n=100)
    mf.use(f)
    out = mf.refine(off(few, dog, 0.1, 0.1, 0.0), dog)
    assert mf.used == 0 and mf.pose(1.0, 2.0, 0.3) == (1.0, 2.0, 0.3)
    assert out is not None


def test_no_field_no_correction():
    mf = MapFix()
    p = wall_scan()
    assert mf.refine(p, (8.0, 2.5)) is p
    assert mf.pose(1.0, 2.0, 0.3) == (1.0, 2.0, 0.3)


def test_a_new_floor_starts_from_the_correction_it_left_with():
    # One odometry frame for the building: the error stepping into the lift
    # is the error stepping out. The new floor then refines on its own.
    f1, f2, mf, dog = corridor(), corridor(), MapFix(), (8.0, 2.5)
    mf.use(f1)
    for _ in range(3):
        mf.refine(mf.points(off(wall_scan(dog), dog, 0.0, 0.15, 0.0)), dog)
    assert mf.pose(8.0, 2.5, 0.0)[1] == pytest.approx(2.35, abs=0.03)
    mf.use(None)                                   # in the lift
    assert mf.pose(8.0, 2.5, 0.0) == (8.0, 2.5, 0.0)
    mf.use(f2)
    assert mf.pose(8.0, 2.5, 0.0)[1] == pytest.approx(2.35, abs=0.03)
    for _ in range(3):                             # floor 2 sees its own drift
        mf.refine(mf.points(off(wall_scan(dog), dog, 0.0, 0.25, 0.0)), dog)
    assert mf.pose(8.0, 2.5, 0.0)[1] == pytest.approx(2.25, abs=0.03)
    assert mf.corr[id(f1)][1] == pytest.approx(-0.15, abs=0.03)   # floor 1's kept


def test_the_same_floor_every_tick_keeps_its_own_correction():
    f, mf, dog = corridor(), MapFix(), (8.0, 2.5)
    mf.use(f)
    for _ in range(3):
        mf.refine(mf.points(off(wall_scan(dog), dog, 0.0, 0.15, 0.0)), dog)
        mf.use(f)                                  # run_building calls use() every tick
    assert mf.pose(8.0, 2.5, 0.0)[1] == pytest.approx(2.35, abs=0.03)


def test_a_corridor_end_puts_back_drift_past_the_old_bound():
    # HIL 11-17, room 201: x drift down the corridor, seen all at once at its
    # end wall, needed 0.6+ m. The old 0.60 m bound reset C every few scans
    # (587 resets) and the walls jumped back and forth. Now it is taken.
    f, mf, dog = corridor(), MapFix(), (3.5, 2.5)          # 1.5 m from the west end
    truth = wall_scan(dog)
    seen = off(truth, dog, 0.9, 0.05, math.radians(1.0))
    mf.use(f)
    out = seen
    for _ in range(8):
        out = mf.refine(mf.points(seen), dog)
    err = np.hypot(out[:, 0] - truth[:, 0], out[:, 1] - truth[:, 1])
    assert np.median(err) < 0.03
    assert mf.resets == 0
    assert mf.pose(*dog, 0.0)[0] == pytest.approx(dog[0] - 0.9, abs=0.05)


def test_past_the_bound_it_still_lets_go():
    # 1.8 m of drift: C has followed it to -1.4 and the next scan would take
    # it past MAX_M. Lost, not trusted: back to the board's own pose.
    f, mf, dog = corridor(), MapFix(), (3.5, 2.5)
    seen = off(wall_scan(dog), dog, 1.8, 0.0, 0.0)
    mf.use(f)
    mf.corr[id(f)] = mf.carry = (-1.4, 0.0, 0.0)
    mf.refine(mf.points(seen), dog)
    assert mf.resets == 1 and mf.carry is None and id(f) not in mf.corr
