"""Generate a synthetic multi-floor building that matches the location names
the language layer's outputs (cyberdog/language/generate_dataset.py).

PLACEHOLDER: replace with the real LumaAI scan + real room layout once
available. Floor/room counts are imported from generate_dataset.py so the
map and the NLU model can't drift apart.

Outputs (in data/building/):
    floor1.obj, floor2.obj, ...   Z-up meshes, same style as classroom_map.obj
    locations.json                name -> list of {floor, xy, door_xy}
    zones.json                    Behavior Layer polygons (the stairwells)
    furniture.json                tables, shelves, counters... per floor; the
                                  "mapped" ones go into the planner's grid
    footprint.json                the building outline, as rectangles, so the
                                  scene lays slabs only where there is building

Then build the grids with scripts/mapping/mesh_to_grid.py --up-axis z --points 4000000
--inflate 0.25 --furniture data/building/furniture.json --floor N
(see src/cyberdog/planning/README.md).

Layout of every floor (top-down, meters). An L-shaped teaching block: the main
block runs east-west, the wing runs north from it at the east end and ends in
an open hall.

                                       x=30    x=37.5 x=40.5   x=48
    y=49                                +---------------------+
                                        |  HALL   [] [] []    |  F1 cafeteria,
                                        |         [] [] [] ## |  F2 library,
                                        |         [] [] [] ## |  F3 study area
    y=37                                +--------+   +--------+  (## raised area, F1)
                                        | W2     ]   [     E2 |
                                        | W1     ]   [     E1 |  wing rooms, two
                                        | W0     ]   [     E0 |  doors each
    y=19 +------+------+-- ... ---------+--------+   +--------+
         | N0   | N1   |          | N4  | N5     | C | N7     |
    y=11 +--  --+--  --+-- ... ---+--  -+--  ----+   +---  ---+
         | S          main corridor (3 m)       T-junction  L |  S = stairs, L = lift
    y=8  +--  --+--  --+-- ... ---+--  -+--  --+--  --+--  ---+
         | S0   | S1   |          | S4  | S5   | S6   | S7    |  S7 on F1: the foyer,
    y=0  +------+------+-- ... ---+-----+------+------+--]  [-+  with the street door
        x=0                                                x=48

The wing corridor C (x 37.5 - 40.5) carries straight on north from the main
corridor where north slot 6 used to be, so the two meet at a T-junction. Rooms
N5 and N7 take the slot's leftover 1.5 m either side; their doors stay where
they were. The original single straight corridor meant VAMOS only ever saw
straight corridors, and the first wing (a parallel corridor at the end of a
blank passage) was not a building anyone would build.

Everything south of y=19 keeps its old coordinates on purpose: obstacles.py,
pedestrians.py, stairs.py, lift.py and STAIRS_ZONE all pin themselves to the
main corridor band (y 8 - 11), and the grid origin is the building's south-west
corner, which the wing does not move.
"""
import json
import os

from cyberdog.language.generate_dataset import NUM_FLOORS, ROOMS_PER_FLOOR  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "building")

SLOTS_PER_SIDE = 8
ROOM_W = 6.0
ROOM_D = 8.0
CORRIDOR_W = 3.0
WALL_T = 0.2
WALL_H = 3.0
DOOR_W = 1.2
STREET_DOOR_W = 1.8                                   # double doors into the foyer

BUILDING_L = SLOTS_PER_SIDE * ROOM_W                  # 48 m
CORR_Y0 = ROOM_D                                      # 8
CORR_Y1 = ROOM_D + CORRIDOR_W                         # 11
BUILDING_W = 2 * ROOM_D + CORRIDOR_W                  # 19
CORR_MID_Y = (CORR_Y0 + CORR_Y1) / 2

# The wing corridor: north slot BRANCH_SLOT, narrowed to CORRIDOR_W and
# centred on the slot, so the junction sits where that room's door used to be.
BRANCH_SLOT = 6
WING_CX0 = BRANCH_SLOT * ROOM_W + ROOM_W / 2 - CORRIDOR_W / 2   # 37.5
WING_CX1 = WING_CX0 + CORRIDOR_W                                # 40.5
WING_MID_X = (WING_CX0 + WING_CX1) / 2                          # 39

# The wing itself: flush with the main block's east wall, as wide as slots
# 5-7, rooms on both sides of the corridor.
WING_X0 = (BRANCH_SLOT - 1) * ROOM_W                  # 30
WING_X1 = BUILDING_L                                  # 48
WING_ROOMS = 3                                        # per side
WING_Y0 = BUILDING_W                                  # 19
WING_Y1 = WING_Y0 + WING_ROOMS * ROOM_W               # 37
# Two doors per wing room, a quarter of the way in from each end -- labs have
# a second way out. The south one (nearer the junction) is the one
# locations.json points the router at.
WING_DOOR_OFFSETS = (1.5, ROOM_W - 1.5)

# The hall at the north end of the wing: the corridor opens straight into it.
HALL_Y0 = WING_Y1                                     # 37
HALL_Y1 = HALL_Y0 + 12.0                              # 49

FOOTPRINT = [(0.0, BUILDING_L, 0.0, BUILDING_W),      # main block
             (WING_X0, WING_X1, WING_Y0, HALL_Y1)]    # wing + hall

STAIRS_XY = (1.0, CORR_MID_Y)
# The lift sits at the opposite, east dead-end of the corridor. Stairs are a
# hazard this robot refuses (blind user, no hand on a rail), so every floor
# change goes through here -- putting the two at opposite ends means a route
# that heads the wrong way is obvious on sight, in the map and in the video.
ELEVATOR_XY = (BUILDING_L - 1.4, CORR_MID_Y)

# The stairwell, as the Behavior Layer sees it: a STOP zone, not a route. It
# spans the full corridor width -- a no-go zone with a walkable channel down
# the middle would just be an invitation to thread between two flights -- and
# it stops at x=2.5, half a metre short of the doors of the westernmost rooms
# (slot 0, gaps centred on x=3), so sealing the stairwell does not also seal
# room 101 and room 106 off the map. sim/scene/stairs.py's shaft is sized to match.
STAIRS_ZONE = {"x": (0.0, 2.5), "y": (8.3, 10.7)}

# Named (non-numbered) locations, pinned to fixed places so the same place
# sits in the same spot on every floor it appears on. Names must match the
# canonical names in cyberdog/language/generate_dataset.py exactly.
#   ("S"/"N", slot)  main block, south/north of the corridor, slot 0-7 west to
#                    east. Numbered rooms take slots 0-4; north slot 6 is the
#                    wing corridor.
#   ("W"/"E", slot)  wing, west/east of its corridor, slot 0-2 south to north
#   ("H", 0)         the hall
# PLACEHOLDER assignment until the real building layout is known.
#
#        F1                     F2                           F3
#   S5   restroom               restroom                     restroom
#   S6   office                 computer engineering lab     chemistry lab
#   S7   main entrance (foyer)  -                            server room
#   N5   -                      electrical engineering lab   biology lab
#   N7   library                -                            -
#   W1   -                      mechanical engineering lab   physics lab
#   H    cafeteria              library                      (study area, unnamed)
NAMED_ROOMS = {
    1: {("S", 5): "restroom", ("S", 6): "office", ("S", 7): "main entrance",
        ("N", 7): "library", ("H", 0): "cafeteria"},
    2: {("S", 5): "restroom", ("S", 6): "computer engineering lab",
        ("N", 5): "electrical engineering lab", ("W", 1): "mechanical engineering lab",
        ("H", 0): "library"},
    3: {("S", 5): "restroom", ("S", 6): "chemistry lab", ("S", 7): "server room",
        ("N", 5): "biology lab", ("W", 1): "physics lab"},
}

# Where a hall destination points the router: the middle of the hall's
# central aisle, which every hall layout below keeps clear.
HALL_POINT = (WING_MID_X, 43.5)


# ---------------------------------------------------------------- furniture --
# (x0, x1, y0, y1, height, kind, mapped). "mapped" furniture is in the planner's
# grid (mesh_to_grid --furniture stamps it): a building map has its fixed
# furniture in it. Unmapped things are drawn but left for the dog to perceive:
# the corridor clutter and the half-open door live in sim/scene/obstacles.py,
# and the raised area here is walkable floor, not an obstacle.

def _grid(xs, ys, h, kind):
    return [(x0, x1, y0, y1, h, kind, True) for x0, x1 in xs for y0, y1 in ys]


# A narrow passage on every floor of the wing corridor: a vending machine on
# the west wall and a cabinet on the east, leaving a door's width (1.2 m)
# between them, on the wall between W1/W2's doors (y 30.1 - 31.9).
PINCH = [(37.6, 38.4, 30.3, 31.7, 1.8, "vending", True),
         (39.6, 40.4, 30.3, 31.7, 1.2, "cabinet", True)]

# Table columns: the west block has a 0.9 m aisle in it; the central aisle
# (x 37.2 - 40.8) is the way in from the corridor and stays clear.
_TABLE_X = [(31.6, 32.8), (33.7, 34.9), (36.0, 37.2), (40.8, 42.0), (43.0, 44.2)]
_TABLE_Y = [(39.0, 39.8), (41.4, 42.2), (43.8, 44.6)]

# The kerb: a raised seating area in the cafeteria's north-east corner,
# 0.12 m up. Walkable, so it is not in the grid -- A* can route over it,
# which is the case the affordance work (backlog item 2) has to handle.
RAISED = (44.6, 47.8, 44.6, 48.8, 0.12, "raised_area", False)

FURNITURE = {
    1: (_grid(_TABLE_X, _TABLE_Y, 0.75, "table")
        + [(30.2, 30.9, 40.0, 47.0, 1.0, "counter", True), RAISED] + PINCH),
    2: (_grid([(31.6, 33.2), (34.2, 35.8)], [(40.0, 40.9), (42.6, 43.5), (45.2, 46.1)],
              0.75, "table")
        + _grid([(41.0, 46.5)], [(41.0, 41.4), (43.0, 43.4), (45.0, 45.4), (47.0, 47.4)],
                1.8, "shelf")
        + PINCH),
    3: (_grid(_TABLE_X[:2] + _TABLE_X[3:], _TABLE_Y[:2], 0.75, "table") + PINCH),
}

# Which floors get the street door: only the ground floor's foyer (S7).
STREET_DOOR_X = 7 * ROOM_W + ROOM_W / 2                # 45


def _check_furniture():
    """Nothing may block a doorway, the hall entrance or the hall point."""
    t = WALL_T / 2
    keep_clear = [(WING_CX0 + 0.3, WING_CX1 - 0.3, HALL_Y0 - 1.0, HALL_Y0 + 1.5),  # hall entrance
                  (HALL_POINT[0] - 0.8, HALL_POINT[0] + 0.8, HALL_POINT[1] - 0.8, HALL_POINT[1] + 0.8)]
    for side_x in (WING_CX0, WING_CX1):
        for k in range(WING_ROOMS):
            for off in WING_DOOR_OFFSETS:
                y = WING_Y0 + k * ROOM_W + off
                keep_clear.append((side_x - 1.0, side_x + 1.0, y - DOOR_W / 2, y + DOOR_W / 2))
    for floor, items in FURNITURE.items():
        for x0, x1, y0, y1, _h, kind, _m in items:
            assert x0 < x1 and y0 < y1, (floor, kind)
            inside = any(fx0 + t <= x0 and x1 <= fx1 - t and fy0 + t <= y0 and y1 <= fy1 - t
                         for fx0, fx1, fy0, fy1 in FOOTPRINT)
            assert inside, f"{kind} on floor {floor} is outside the building"
            for cx0, cx1, cy0, cy1 in keep_clear:
                assert x1 <= cx0 or x0 >= cx1 or y1 <= cy0 or y0 >= cy1, \
                    f"{kind} on floor {floor} at ({x0}, {y0}) blocks a door or the hall way in"


_check_furniture()


# -------------------------------------------------------------------- rooms --

def write_box(f, x1, y1, z1, x2, y2, z2, v_idx):
    """Same triangulated box writer as generate_obj.py."""
    verts = [
        (x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1),
        (x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)
    ]
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for a, b, c in [(1, 4, 3), (1, 3, 2), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
                    (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]:
        f.write(f"f {v_idx+a} {v_idx+b} {v_idx+c}\n")
    return v_idx + 8


def slot_positions(side, slot):
    """Room point and the corridor point in front of its (first) door.

    Main-block doors stay on the slot's centre line even where the wing
    corridor has widened N5 and N7 by 1.5 m, so a room's point is its door
    axis rather than the middle of its floor -- still well inside the room.
    """
    if side == "H":
        return HALL_POINT, (WING_MID_X, HALL_Y0 - 1.0)
    if side in ("W", "E"):
        y0 = WING_Y0 + slot * ROOM_W
        x_mid = (WING_X0 + WING_CX0) / 2 if side == "W" else (WING_CX1 + WING_X1) / 2
        return (x_mid, y0 + ROOM_W / 2), (WING_MID_X, y0 + WING_DOOR_OFFSETS[0])
    x_mid = slot * ROOM_W + ROOM_W / 2
    y_room = ROOM_D / 2 if side == "S" else CORR_Y1 + ROOM_D / 2
    return (x_mid, y_room), (x_mid, CORR_MID_Y)


def floor_slots(floor):
    """(name, side, slot) for every labeled room on one floor."""
    half = (ROOMS_PER_FLOOR + 1) // 2
    used = {"S": half, "N": ROOMS_PER_FLOOR - half, "W": 0, "E": 0, "H": 0}
    n_slots = {"S": SLOTS_PER_SIDE, "N": SLOTS_PER_SIDE, "W": WING_ROOMS, "E": WING_ROOMS, "H": 1}
    for (side, slot), name in NAMED_ROOMS.get(floor, {}).items():
        if slot < used[side] or slot >= n_slots[side]:
            raise ValueError(f"'{name}' on floor {floor} ({side}, {slot}) overlaps a numbered "
                             f"room or is off the building; adjust NAMED_ROOMS.")
        if (side, slot) == ("N", BRANCH_SLOT):
            raise ValueError(f"'{name}' on floor {floor} is in north slot {BRANCH_SLOT}, which "
                             f"is the wing corridor; move it to a wing room.")
    out = []
    for n in range(1, ROOMS_PER_FLOOR + 1):
        side, slot = ("S", n - 1) if n <= half else ("N", n - 1 - half)
        out.append((f"room {floor}{n:02d}", side, slot))
    out += [(name, side, slot) for (side, slot), name in NAMED_ROOMS.get(floor, {}).items()]
    return out


def floor_walls(floor):
    """Every wall on a floor as (x0, x1, y0, y1) footprints, WALL_T thick.

    Walls are listed along their centre lines and padded by half a thickness
    at both ends, so corners and T-joins close instead of leaving a pinhole
    the grid would read as a gap.
    """
    t = WALL_T / 2
    walls = []

    def hwall(y, x0, x1):
        walls.append((x0 - t, x1 + t, y - t, y + t))

    def vwall(x, y0, y1):
        walls.append((x - t, x + t, y0 - t, y1 + t))

    def doored(along, at, a0, a1, doors, width=DOOR_W):
        """A wall from a0 to a1 at `at`, with a gap of `width` centred on each
        of `doors`. along="x" runs east-west (y = at), "y" north-south."""
        cuts = sorted((d - width / 2, d + width / 2) for d in doors)
        start = a0 - t
        for g0, g1 in cuts + [(a1 + t, None)]:
            if g0 > start:
                if along == "x":
                    walls.append((start, g0, at - t, at + t))
                else:
                    walls.append((at - t, at + t, start, g0))
            start = g1 if g1 is not None else start

    # Main block outer walls. The north one stops either side of the wing
    # corridor; the south one has the street door into the foyer on floor 1.
    doored("x", 0, 0, BUILDING_L, [STREET_DOOR_X] if floor == 1 else [], STREET_DOOR_W)
    hwall(BUILDING_W, 0, WING_X0)                     # west of the wing: outside wall
    vwall(0, 0, BUILDING_W)                           # the east wall is the wing's, below

    # South rooms: partitions, and the corridor wall with one door per room.
    for k in range(1, SLOTS_PER_SIDE):
        vwall(k * ROOM_W, 0, CORR_Y0)
    doored("x", CORR_Y0, 0, BUILDING_L, [k * ROOM_W + ROOM_W / 2 for k in range(SLOTS_PER_SIDE)])

    # North rooms. The wing corridor's walls replace the partitions either
    # side of slot BRANCH_SLOT, and that slot has no corridor wall -- it is
    # the T-junction.
    for k in range(1, SLOTS_PER_SIDE):
        if k not in (BRANCH_SLOT, BRANCH_SLOT + 1):
            vwall(k * ROOM_W, CORR_Y1, BUILDING_W)
    doors = [k * ROOM_W + ROOM_W / 2 for k in range(SLOTS_PER_SIDE) if k != BRANCH_SLOT]
    doored("x", CORR_Y1, 0, WING_CX0, [d for d in doors if d < WING_CX0])
    doored("x", CORR_Y1, WING_CX1, BUILDING_L, [d for d in doors if d > WING_CX1])
    # N5 / N7 back walls, which are also the wing's first rooms' south walls.
    hwall(BUILDING_W, WING_X0, WING_CX0)
    hwall(BUILDING_W, WING_CX1, WING_X1)

    # The wing: outer walls, corridor walls with two doors per room, room
    # partitions. The corridor runs on into the hall with no wall across it.
    vwall(WING_X0, BUILDING_W, HALL_Y1)
    vwall(WING_X1, 0, HALL_Y1)                        # whole east side, main block included
    wing_doors = [WING_Y0 + k * ROOM_W + off for k in range(WING_ROOMS) for off in WING_DOOR_OFFSETS]
    for x in (WING_CX0, WING_CX1):
        vwall(x, CORR_Y1, BUILDING_W)                 # alongside N5 / N7: no doors
        doored("y", x, WING_Y0, WING_Y1, wing_doors)
    for k in range(1, WING_ROOMS + 1):
        y = WING_Y0 + k * ROOM_W
        hwall(y, WING_X0, WING_CX0)
        hwall(y, WING_CX1, WING_X1)

    # The hall: north wall only; its south side is the last wing rooms' walls.
    hwall(HALL_Y1, WING_X0, WING_X1)
    return walls


def write_floor_obj(path, floor):
    t = WALL_T / 2
    with open(path, "w") as f:
        f.write("# Synthetic building floor (placeholder), Z-up, meters\n")
        v = 0
        for x0, x1, y0, y1 in FOOTPRINT:
            v = write_box(f, x0 - t, y0 - t, -0.1, x1 + t, y1 + t, 0, v)
        for x0, x1, y0, y1 in floor_walls(floor):
            v = write_box(f, x0, y0, 0, x1, y1, WALL_H, v)
        # Furniture at its true height. Most of it is under mesh_to_grid's
        # 0.8 - 1.3 m slice, which is why the mapped pieces are also stamped
        # into the grid from furniture.json.
        for x0, x1, y0, y1, h, _kind, _mapped in FURNITURE.get(floor, []):
            v = write_box(f, x0, y0, 0, x1, y1, h, v)


def write_zones(path):
    """Behavior Layer polygons for the building.

    Only the stairwells so far. They are written from the same constants the
    stairs are drawn from, so the zone cannot end up somewhere the flight
    isn't. Action rules are left empty on purpose -- BehaviorLayer.from_config
    fills them from config/map_config.yaml, which is where the
    "stairs = full stop, announce, requires confirmation" rule lives.

    The zone repeats per floor even though the geometry is identical, because
    the stairwell is a real hazard on each of them and the metadata is what
    tells you which floor a trigger fired on.
    """
    (zx0, zx1), (zy0, zy1) = STAIRS_ZONE["x"], STAIRS_ZONE["y"]
    zones = [{
        "zone_id": f"stairs_f{floor}",
        "zone_type": "stairs",
        "polygon": [[zx0, zy0], [zx1, zy0], [zx1, zy1], [zx0, zy1]],
        "metadata": {"name": f"Floor {floor} stairwell", "floor": floor},
    } for floor in range(1, NUM_FLOORS + 1)]

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"zones": zones}, f, indent=2)
    print(f"Wrote {path} ({len(zones)} zones)")


def write_furniture(path):
    data = {"floors": {str(n): [
        {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "h": h, "kind": kind, "mapped": mapped}
        for x0, x1, y0, y1, h, kind, mapped in FURNITURE.get(n, [])]
        for n in range(1, NUM_FLOORS + 1)}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path} ({sum(len(v) for v in data['floors'].values())} pieces)")


def write_footprint(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rects": [list(r) for r in FOOTPRINT], "wall_t": WALL_T}, f, indent=2)
    print(f"Wrote {path}")


def main():
    locations = {}

    def add(name, entry):
        locations.setdefault(name, []).append(entry)

    for floor in range(1, NUM_FLOORS + 1):
        obj_path = os.path.join(OUT_DIR, f"floor{floor}.obj")
        write_floor_obj(obj_path, floor)
        print(f"Wrote {obj_path}")

        for name, side, slot in floor_slots(floor):
            xy, door_xy = slot_positions(side, slot)
            add(name, {"floor": floor, "xy": list(xy), "door_xy": list(door_xy)})

        add("hallway", {"floor": floor, "xy": [BUILDING_L / 2, CORR_MID_Y],
                        "door_xy": [BUILDING_L / 2, CORR_MID_Y]})
        # Every word for a staircase, same point. BuildingRouter.hazard_named
        # matches the command against these names before the NLU model runs;
        # a synonym missing here goes to the model, which maps it onto the
        # nearest place it knows ("staircase" -> "hallway") without a word.
        for name in ("stairs", "staircase", "stairway", "stairwell"):
            add(name, {"floor": floor, "xy": list(STAIRS_XY), "door_xy": list(STAIRS_XY)})
        # Both names, same point: the destination lookup in cyberdog/sim/run_building.py
        # and the planning layer match on the literal word the user said.
        for name in ("elevator", "lift"):
            add(name, {"floor": floor, "xy": list(ELEVATOR_XY),
                       "door_xy": list(ELEVATOR_XY)})

    loc_path = os.path.join(OUT_DIR, "locations.json")
    with open(loc_path, "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=2)
    print(f"Wrote {loc_path} ({len(locations)} names)")

    write_zones(os.path.join(OUT_DIR, "zones.json"))
    write_furniture(os.path.join(OUT_DIR, "furniture.json"))
    write_footprint(os.path.join(OUT_DIR, "footprint.json"))


if __name__ == "__main__":
    main()
