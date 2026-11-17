"""Fixed furniture -- tables, shelves, counters -- and the building outline.

Both come from data/building/ (furniture.json, footprint.json), written by
scripts/mapping/generate_building.py, so the grid, the mesh and the scene read
one set of numbers.

Mapped furniture is in the planner's grid (mesh_to_grid --furniture stamps
it), which is right for planning and for clearance -- a table is something the
dog and the person on the handle must not touch. It is wrong for drawing:
build_scene turns every occupied grid cell into a full-height wall box, so a
table would come out as a 2 m pillar. `clear_mapped` takes the footprints back
out of the grid used for drawing walls, and `geoms` draws each piece at its
real height instead.

Unmapped furniture is walkable floor: the raised area in the cafeteria. It is
drawn with a "slab" name, so run_building does not count walking on it as a
collision.
"""
import json
import os

from cyberdog.paths import BUILDING_DIR
from cyberdog.sim.scene.levels import floor_z

COLOURS = {"table": "0.55 0.42 0.30 1", "shelf": "0.45 0.35 0.28 1",
           "counter": "0.62 0.62 0.66 1", "vending": "0.70 0.20 0.22 1",
           "cabinet": "0.50 0.52 0.56 1", "raised_area": "0.50 0.48 0.44 1"}
DEFAULT_COLOUR = "0.55 0.50 0.45 1"

_PIECES = None


def pieces(floor):
    """[{x0, x1, y0, y1, h, kind, mapped}] for one floor; [] without the file."""
    global _PIECES
    if _PIECES is None:
        path = os.path.join(BUILDING_DIR, "furniture.json")
        try:
            with open(path, encoding="utf-8") as f:
                _PIECES = json.load(f)["floors"]
        except FileNotFoundError:
            _PIECES = {}
    return _PIECES.get(str(floor), [])


def footprint():
    """The building outline as [(x0, x1, y0, y1)] slab rectangles (wall
    thickness included), or None for a building generated before the file."""
    path = os.path.join(BUILDING_DIR, "footprint.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    t = data.get("wall_t", 0.2) / 2
    return [(x0 - t, x1 + t, y0 - t, y1 + t) for x0, x1, y0, y1 in data["rects"]]


def clear_mapped(occ, res, ox, oy, floor):
    """`occ` with every mapped piece's footprint (plus a cell) set free."""
    occ = occ.copy()
    H, W = occ.shape
    for p in pieces(floor):
        if not p["mapped"]:
            continue
        c0 = max(int((p["x0"] - ox) / res) - 1, 0)
        c1 = min(int((p["x1"] - ox) / res) + 2, W)
        r0 = max(int(H - 1 - (p["y1"] - oy) / res) - 1, 0)
        r1 = min(int(H - 1 - (p["y0"] - oy) / res) + 2, H)
        occ[r0:r1, c0:c1] = False
    return occ


def geoms(floor, z0=None):
    """MuJoCo box geoms for floor `floor`'s furniture, on its walking surface."""
    z = floor_z(floor) if z0 is None else z0
    out = []
    for i, p in enumerate(pieces(floor)):
        kind, h = p["kind"], p["h"]
        # Walkable pieces are floor, and named as such (see the docstring).
        name = f"furn{floor}_{kind}{i}" if p["mapped"] else f"slab_{kind}{floor}_{i}"
        out.append(
            f'<geom name="{name}" type="box" '
            f'pos="{(p["x0"] + p["x1"]) / 2:.3f} {(p["y0"] + p["y1"]) / 2:.3f} {z + h / 2:.3f}" '
            f'size="{(p["x1"] - p["x0"]) / 2:.3f} {(p["y1"] - p["y0"]) / 2:.3f} {h / 2:.3f}" '
            f'rgba="{COLOURS.get(kind, DEFAULT_COLOUR)}"/>')
    return out
