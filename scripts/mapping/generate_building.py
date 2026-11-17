"""Generate a synthetic multi-floor building that matches the location names
the language layer's outputs (cyberdog/language/generate_dataset.py).

PLACEHOLDER: replace with the real LumaAI scan + real room layout once
available. Floor/room counts are imported from generate_dataset.py so the
map and the NLU model can't drift apart.

Outputs (in data/building/):
    floor1.obj, floor2.obj, ...   Z-up meshes, same style as classroom_map.obj
    locations.json                name -> list of {floor, xy, door_xy}
    zones.json                    Behavior Layer polygons (the stairwells)

Then build the grids with scripts/mapping/mesh_to_grid.py --up-axis z --points 4000000
--inflate 0.25 (see src/cyberdog/planning/README.md).

Layout of every floor (top-down, meters):

                        x=16.5                  x=40.5
    y=30                    +-----+-----+-----+-----+
                            | W0  | W1  | W2  | W3  |      wing rooms (4 slots, 6 m)
    y=22                    +-  --+-  --+-  --+-  --+
                            |  wing corridor      B |      L-turn at the top of B
    y=19 +------+-- ... ----+---------------+---+   +------+
         | N0   | N1  ...          | N5     | B |   N7     |  north rooms; slot 6 is
    y=11 +--  --+--  ... ----------+---  ---+   +---  ----+  the branch corridor B
         | S         corridor (3 m wide)    T-junction   L |  S = stairs, L = lift
    y=8  +--  --+--  ... --+--  --+--  --+--  --+--  --+----+
         | S0   | S1  ...  | S4   | S5   | S6   | S7        |  south rooms
    y=0  +------+--- ... --+------+------+------+-----------+
        x=0                                             x=48

The branch corridor B (x 37.5 - 40.5) replaces north slot 6: the main
corridor meets it at a T-junction, and it turns west into the wing corridor
at the top (an L-turn). The slot's leftover 1.5 m either side goes to rooms
N5 and N7, whose doors stay where they were. The original single straight
corridor meant VAMOS only ever saw straight corridors.

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

BUILDING_L = SLOTS_PER_SIDE * ROOM_W                  # 48 m
CORR_Y0 = ROOM_D                                      # 8
CORR_Y1 = ROOM_D + CORRIDOR_W                         # 11
BUILDING_W = 2 * ROOM_D + CORRIDOR_W                  # 19
CORR_MID_Y = (CORR_Y0 + CORR_Y1) / 2

# The branch corridor: north slot BRANCH_SLOT, narrowed to CORRIDOR_W and
# centred on the slot, so the junction sits where that room's door used to be.
BRANCH_SLOT = 6
BRANCH_X0 = BRANCH_SLOT * ROOM_W + ROOM_W / 2 - CORRIDOR_W / 2   # 37.5
BRANCH_X1 = BRANCH_X0 + CORRIDOR_W                               # 40.5

# The wing: a corridor running west from the top of the branch along the
# building's north wall, with WING_SLOTS rooms on its north side. Its east end
# is the branch's east wall, so the branch turns into it as an L.
WING_SLOTS = 4
WING_X1 = BRANCH_X1                                   # 40.5
WING_X0 = WING_X1 - WING_SLOTS * ROOM_W               # 16.5
WING_CORR_Y0 = BUILDING_W                             # 19
WING_CORR_Y1 = WING_CORR_Y0 + CORRIDOR_W              # 22
WING_Y1 = WING_CORR_Y1 + ROOM_D                       # 30
WING_MID_Y = (WING_CORR_Y0 + WING_CORR_Y1) / 2

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

# Named (non-numbered) locations, pinned to fixed slots so the same place
# sits in the same spot on every floor it appears on. Names must match the
# canonical names in cyberdog/language/generate_dataset.py exactly.
# (side, slot): side "S"/"N" of the corridor, slot 0-7 from west to east.
# Numbered rooms take slots 0-4 on both sides; these use slots 5-7, except
# north slot 6, which is the branch corridor. Side "W" is the wing.
# PLACEHOLDER assignment until the real building layout is known.
#
#            slot 5            slot 6                      slot 7
#   N  F1: cafeteria         (branch corridor)           F1: library
#      F2: electrical eng                                F2: library (2nd floor of it)
#      F3: biology lab                                   F3: -
#   S  all: restroom         F1: office                  F1: main entrance
#      (one per floor,       F2: computer eng lab        F2: -
#       stacked)             F3: chemistry lab           F3: server room
#   W  F2: wing slot 1 mechanical eng, F3: wing slot 1 physics lab (both used
#      to be north slot 6); the other wing rooms are unnamed.
NAMED_ROOMS = {
    1: {("S", 5): "restroom", ("S", 6): "office", ("S", 7): "main entrance",
        ("N", 5): "cafeteria", ("N", 7): "library"},
    2: {("S", 5): "restroom", ("S", 6): "computer engineering lab",
        ("N", 5): "electrical engineering lab", ("N", 7): "library",
        ("W", 1): "mechanical engineering lab"},
    3: {("S", 5): "restroom", ("S", 6): "chemistry lab", ("S", 7): "server room",
        ("N", 5): "biology lab", ("W", 1): "physics lab"},
}


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
    """Room center and the corridor point in front of its door.

    Doors stay on the slot's centre line even where the branch corridor has
    widened N5 and N7 by 1.5 m, so a room's centre point is its door axis
    rather than the middle of its floor -- still well inside the room.
    """
    if side == "W":
        x_mid = WING_X0 + slot * ROOM_W + ROOM_W / 2
        return (x_mid, WING_CORR_Y1 + ROOM_D / 2), (x_mid, WING_MID_Y)
    x_mid = slot * ROOM_W + ROOM_W / 2
    y_room = ROOM_D / 2 if side == "S" else CORR_Y1 + ROOM_D / 2
    return (x_mid, y_room), (x_mid, CORR_MID_Y)


def floor_slots(floor):
    """(name, side, slot) for every labeled room on one floor."""
    half = (ROOMS_PER_FLOOR + 1) // 2
    used = {"S": half, "N": ROOMS_PER_FLOOR - half, "W": 0}  # numbered rooms take slots 0..used-1
    n_slots = {"S": SLOTS_PER_SIDE, "N": SLOTS_PER_SIDE, "W": WING_SLOTS}
    for (side, slot), name in NAMED_ROOMS.get(floor, {}).items():
        if slot < used[side] or slot >= n_slots[side]:
            raise ValueError(f"'{name}' on floor {floor} ({side}, {slot}) overlaps a numbered "
                             f"room or is off the building; adjust NAMED_ROOMS/SLOTS_PER_SIDE.")
        if (side, slot) == ("N", BRANCH_SLOT):
            raise ValueError(f"'{name}' on floor {floor} is in north slot {BRANCH_SLOT}, which "
                             f"is the branch corridor to the wing; move it to a wing slot.")
    out = []
    for n in range(1, ROOMS_PER_FLOOR + 1):
        side, slot = ("S", n - 1) if n <= half else ("N", n - 1 - half)
        out.append((f"room {floor}{n:02d}", side, slot))
    out += [(name, side, slot) for (side, slot), name in NAMED_ROOMS.get(floor, {}).items()]
    return out


def floor_walls():
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

    def doored(y, x0, x1, door_x):
        """A wall from x0 to x1 along y, with one DOOR_W gap centred on door_x."""
        g0, g1 = door_x - DOOR_W / 2, door_x + DOOR_W / 2
        walls.append((x0 - t, g0, y - t, y + t))
        walls.append((g1, x1 + t, y - t, y + t))

    # Outer walls of the main block. The north one stops either side of the
    # branch, which is the only way into the wing.
    hwall(0, 0, BUILDING_L)
    hwall(BUILDING_W, 0, BRANCH_X0)
    hwall(BUILDING_W, BRANCH_X1, BUILDING_L)
    vwall(0, 0, BUILDING_W)
    vwall(BUILDING_L, 0, BUILDING_W)

    # South rooms: partitions, and the corridor wall with one door per room.
    for k in range(1, SLOTS_PER_SIDE):
        vwall(k * ROOM_W, 0, CORR_Y0)
    for k in range(SLOTS_PER_SIDE):
        doored(CORR_Y0, k * ROOM_W, (k + 1) * ROOM_W, k * ROOM_W + ROOM_W / 2)

    # North rooms. The branch walls replace the partitions either side of
    # slot BRANCH_SLOT, and that slot has no corridor wall at all -- it is the
    # T-junction. The branch's east wall carries on up as the wing's east end.
    for k in range(1, SLOTS_PER_SIDE):
        if k not in (BRANCH_SLOT, BRANCH_SLOT + 1):
            vwall(k * ROOM_W, CORR_Y1, BUILDING_W)
    vwall(BRANCH_X0, CORR_Y1, BUILDING_W)
    vwall(BRANCH_X1, CORR_Y1, WING_Y1)
    for k in range(SLOTS_PER_SIDE):
        if k == BRANCH_SLOT:
            continue
        x0 = BRANCH_X1 if k == BRANCH_SLOT + 1 else k * ROOM_W
        x1 = BRANCH_X0 if k == BRANCH_SLOT - 1 else (k + 1) * ROOM_W
        doored(CORR_Y1, x0, x1, k * ROOM_W + ROOM_W / 2)

    # The wing. Its corridor's south side is the main block's north wall.
    vwall(WING_X0, BUILDING_W, WING_Y1)
    hwall(WING_Y1, WING_X0, WING_X1)
    for k in range(1, WING_SLOTS):
        vwall(WING_X0 + k * ROOM_W, WING_CORR_Y1, WING_Y1)
    for k in range(WING_SLOTS):
        x0 = WING_X0 + k * ROOM_W
        doored(WING_CORR_Y1, x0, x0 + ROOM_W, x0 + ROOM_W / 2)
    return walls


def write_floor_obj(path):
    t = WALL_T / 2
    with open(path, "w") as f:
        f.write("# Synthetic building floor (placeholder), Z-up, meters\n")
        v = 0
        # Floor slabs: the main block, then the wing.
        v = write_box(f, -t, -t, -0.1, BUILDING_L + t, BUILDING_W + t, 0, v)
        v = write_box(f, WING_X0 - t, BUILDING_W - t, -0.1, WING_X1 + t, WING_Y1 + t, 0, v)
        for x0, x1, y0, y1 in floor_walls():
            v = write_box(f, x0, y0, 0, x1, y1, WALL_H, v)


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


def main():
    locations = {}

    def add(name, entry):
        locations.setdefault(name, []).append(entry)

    for floor in range(1, NUM_FLOORS + 1):
        obj_path = os.path.join(OUT_DIR, f"floor{floor}.obj")
        write_floor_obj(obj_path)
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


if __name__ == "__main__":
    main()
