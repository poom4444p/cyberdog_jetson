"""LiveClearance.remember keeps its memory in a grid, not a dict of cells
(70-130 ms a tick on the J4012 beside dense returns). The field it builds
must be the dict version's, cell for cell."""
import numpy as np
from scipy import ndimage

from cyberdog.sim.sensing import perception as pc


def old_remember(lc, memory, people, now, centre):
    """The dict version, as it was (perception.py before the grid)."""
    if lc._scan is not None:
        px, py, _, origin = lc._scan
        keep = np.ones(len(px), dtype=bool)
        for mx, my, mr in people:
            keep &= np.hypot(px - mx, py - my) > mr + pc.CLUSTER_PAD
        px, py = lc._shadowed(px[keep], py[keep], origin)
        cells = np.unique(np.column_stack([(px / lc.res).astype(int),
                                           (py / lc.res).astype(int)]), axis=0)
        for c, r in cells:
            memory[(int(c), int(r))] = now
    memory = {k: t for k, t in memory.items() if now - t <= pc.MEMORY_S}
    if not memory:
        return memory, None
    x0, y0 = centre[0] - pc.WINDOW, centre[1] - pc.WINDOW
    cells = np.array(list(memory), dtype=float)
    c = ((cells[:, 0] + 0.5) * lc.res - x0) / lc.res
    r = ((cells[:, 1] + 0.5) * lc.res - y0) / lc.res
    c, r = c.astype(int), r.astype(int)
    ok = (r >= 0) & (r < lc.n) & (c >= 0) & (c < lc.n)
    if not ok.any():
        return memory, None
    occ = np.zeros((lc.n, lc.n), dtype=bool)
    occ[r[ok], c[ok]] = True
    return memory, (ndimage.distance_transform_edt(~occ) * lc.res, x0, y0)


def test_grid_memory_matches_the_dict():
    rng = np.random.default_rng(3)
    new = pc.LiveClearance(1)
    ref = pc.LiveClearance(1)
    memory = {}
    x, y = 2.0, 9.5
    for k in range(80):
        now = k * 0.25                       # 20 s: cells expire (MEMORY_S 6 s)
        x += 0.3                             # walk east along the corridor
        if k % 7 == 3:
            scan = None                      # a tick with no scan
        else:
            m = int(rng.integers(20, 400))
            px = x + rng.uniform(-6, 6, m)
            py = y + rng.uniform(-4, 4, m)
            scan = (px, py, None, (x, y, 0.5))
        people = [(x + 1.0, y + 0.5, 0.3)] if k % 3 == 0 else []
        new._scan = scan
        ref._scan = scan
        new.remember(people, now, (x, y))
        memory, recent = old_remember(ref, memory, people, now, (x, y))
        if recent is None:
            assert new.recent is None, k
        else:
            assert new.recent is not None, k
            assert np.array_equal(new.recent[0], recent[0]), k
            assert new.recent[1:] == recent[1:], k
            assert new.remembered_at(x - 0.8, y) == ref.__class__.remembered_at(
                type("R", (), {"recent": recent, "res": ref.res, "n": ref.n})(), x - 0.8, y)


def test_points_far_outside_the_map_are_ignored():
    lc = pc.LiveClearance(1)
    lc._scan = (np.array([-500.0, 2.5]), np.array([-500.0, 9.5]), None, (2.0, 9.5, 0.5))
    lc.remember([], 0.0, (2.0, 9.5))
    assert lc.recent is not None
    assert lc.remembered_at(2.5, 9.5) < 0.1
