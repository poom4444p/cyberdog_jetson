# Architecture

How the four layers actually connect, and why the boundaries fall where they do.
For what each layer contains, see its own README; for the specification this was
built against, see [`spec.md`](spec.md).

---

## The path a command takes

```
  "take me upstairs to the chemistry lab"
              |
              v
  +-------------------------------------------------------------+
  |  language/                                                   |
  |    command_splitter  "A, then B" -> ["A", "B"]   (rules)     |
  |    floor_parser      "upstairs"  -> floor + 1    (rules)     |
  |    infer             text -> {destination, floor}  (LoRA)    |
  +-------------------------------------------------------------+
              |  destination name + floor
              v
  +-------------------------------------------------------------+
  |  planning/building_router                                    |
  |    locations.json -> where is it, on which floor             |
  |    per-floor A* (mapping/) -> checkpoints                    |
  |    across floors -> legs joined by the lift                  |
  |    stairs -> NoAccessibleRoute (a refusal, spoken)           |
  +-------------------------------------------------------------+
              |  legs of checkpoints, in map coordinates
              v
  +-------------------------------------------------------------+
  |  sim/run_building   THE CONTROL LOOP, 20 Hz                  |
  |                                                              |
  |   every tick:                                                |
  |     lidar.scan()            -> ~2000 world points            |
  |     perception.update()     -> which of them the map cannot  |
  |                                explain (crates kept 6 s)     |
  |     tracking.update()       -> which of those are walking    |
  |     perception.free_destination() -> the goal to aim at      |
  |     control.advance()       -> which checkpoint we are on    |
  |     a person in the way:                                     |
  |       --planner cerrt (default) -> ce_rrt.plan() round them, |
  |                                    replanned every 0.1 s     |
  |       no safe plan, or --planner map -> wait for them        |
  |     spare_handler()         -> ease a turn that would swing  |
  |                                the person on the handle      |
  |                                into something                |
  |                                                              |
  |   every 20 ticks (1 Hz), if --vamos:                         |
  |     checkpoint_projector.project_route()  -> goal PIXEL      |
  |     vamos_client.plan()     -> 5 candidate paths -> 1        |
  |       |- traversable()      cheap geometry test              |
  |       \- dreaming.factor()  imagined rollouts -> safety      |
  |     (asynchronous: the answer lands --vlm-latency s later,   |
  |      and is re-dreamed from where the dog is by then)        |
  +-------------------------------------------------------------+
              |  target point
              v
        control law (control.py) -> handle.SafetyMux -> velocity -> MuJoCo
                                    (a tug stops it on the tick)
```

---

## The two ideas that decide everything

### 1. Map decides WHERE, VLM decides HOW

`planning/` produces checkpoints in map coordinates.
`checkpoint_projector` turns the next one into a **pixel** in the dog's camera
frame. VAMOS is handed the image and that pixel, and answers with five candidate
paths *drawn on the image* — its job is the local manoeuvre, never the
destination.

This is why `free_destination` exists. VAMOS drives at whatever goal pixel it is
given. A* plans on a map with no crate in it, so the pixel can land *on* a
crate — and the model obligingly draws five paths into it (measured: at 0.8 m the
crate fills 60% of the frame and all five candidates still go straight through).
That is not a model that cannot avoid obstacles; it is a model being told to walk
into one. So the map's half of the division of labour has to use what the sensor
found, and hand VAMOS a goal on open floor.

### 2. AI proposes, simple code disposes

Nothing the VLM returns is followed on trust. Every candidate passes two
deterministic filters before it can steer:

| Filter | Question | Where |
|---|---|---|
| `traversable()` | Does the drawn line itself stay in free space? | `vamos_client.py` |
| `Dream.factor()` | If we actually chased it with our own controller, 24 times, with noise — how often do we survive, and with how much room? | `sensing/dreaming.py` |

The second returns `P(no collision) × (room left)`. Below `GATE = 0.5` the
candidate is rejected; above it, the factor also **throttles speed**, so a path
the dog only just believes it can walk is walked at half pace.

`dreaming.py` imports its gains from `sim/control.py` — the same constants the
live loop uses. A dream of a controller the dog does not have would tell us
nothing, so that shared import is load-bearing, not incidental.

---

## Clearance: the one number every safety decision reads

"Metres to the nearest thing the dog must not touch." Two sources, fused:

```
  clearance.py                 sensing/perception.py
  ------------                 --------------------
  occupancy grid walls         LiDAR returns the static
  + Behavior Layer             field cannot explain
    full-stop zones            + SHADOW behind each one
        |                              |
        +--------------+---------------+
                       v
              LiveClearance(x, y)  ->  min(static, detected)
```

`LiveClearance` has the same call signature as the static closure it replaced,
which is why the safety gate and the imagined rollouts needed no changes to start
seeing obstacles — they are handed this object instead and cannot tell the
difference.

Two details that are not obvious:

- **Zones matter as much as walls.** The grid is built from the floor mesh, so it
  is flat and knows nothing about the stairwell standing on it. Without the
  Behavior Layer's full-stop zones, a path into the stairs reads as wide open.
- **A range sensor sees surfaces, not solids.** A crate returns only the face
  pointing at the dog; the space behind that face is *unknown*, not free.
  Treating unknown as free is how a robot plans confidently into the inside of a
  box, so each return also shadows `SHADOW = 1.0 m` of the ray behind it. This is
  deliberately conservative and can over-claim when the real object is thin.

---

## Why the obstacles are not on the map

Every other solid in the scene — walls, slabs, lift, stairwells — is *also* in
the occupancy grid. That would make the entire perception half decorative: VAMOS
proposes paths from a picture, and those paths get scored against the same ground
truth the route already came from, so the camera could never contribute anything
the map did not already have.

`sim/scene/obstacles.py` is the exception, and the exception is the point. Its
crates are drawn into the scene and deliberately **absent from `data/building/`**
— no grid cell, no zone, no location. The map is stale on purpose. A* routes
straight through them; the static clearance field reports 0.85–1.10 m of open
floor exactly where they stand. That gap is what gives the perception layer a
job, and it is what `tests/selftest.py` measures.

The module asserts its own placement: each crate must straddle the centreline
the dog actually walks (pure pursuit cuts corners off the checkpoint polyline),
and must leave at least `MIN_GAP = 0.6 m` on its wider side to get past. Size is
asserted the other way round, by being modest: these are the objects they are
named after — a 1.2 x 0.7 m trolley, a 0.6 m square stack of cartons — and not
the half-corridor chicanes an earlier version built, which made getting past a
manoeuvre rather than a detour. A crate that sealed the corridor would be a
wall the planner cannot see, and every route on that floor would fail for reasons
no error message explains.

---

## Module dependencies

Arrows point from importer to imported. No cycles.

```
  main_planner ---> language ---> (torch)
       |
       +---------> planning ---> mapping
                      ^
                      |
  sim/run_building ---+
       |
       +--> planning/ce_rrt     (numpy only)
       +--> sim/control ------> planning/building_router
       +--> sim/handle
       +--> sim/clearance ----> mapping/behavior_layer
       |                        sim/scene/build_scene
       +--> sim/overlay ------> planning/checkpoint_projector
       +--> sim/vamos_client -> (HTTP :8009)
       +--> sim/sensing/lidar ------> (mujoco)
       +--> sim/sensing/perception -> sim/clearance
       +--> sim/sensing/tracking
       +--> sim/sensing/dreaming ---> sim/control
       +--> sim/robot/mujoco_robot -> sim/scene/build_scene
       +--> sim/scene/* -----------> cyberdog.paths

  everything --> cyberdog.paths     (filesystem locations, one place)
```

Two of these edges were recently *removed*, and the reasons generalise:

- `sensing/perception.py` used to do `from record_demo import clearance_field` —
  the live safety layer importing a video recorder. The clearance field now lives
  in `sim/clearance.py`.
- `sensing/dreaming.py` and `run_building.py` used to do
  `from run_demo import K_W, TURN_ONLY` — two modules reaching into a demo script
  for the control gains. Those now live in `sim/control.py`.

In both cases the shared thing was real and the location was wrong. If you find
yourself importing an entry point, that is the signal to extract.

---

## Timing budget

Measured by `tests/selftest.py latency` on an M-series Mac:

| Stage | Cost | Budget |
|---|---|---|
| LiDAR scan (1980 rays) | ~1.5 ms | — |
| Perception + `free_destination` | 2.4 ms | 50 ms (one 20 Hz tick) — 5% |
| `Dream.factor()` × 5 candidates | 7.4 ms | ~1.8 s (a VLM call) |
| Static clearance field (once per floor) | ~125 ms | cached — cannot be rebuilt per tick |
| CE-RRT\* plan (only with a person in the way) | ~15–40 ms | 100 ms (`ce_rrt.REPLAN_S`): 10 Hz, as the spec asks |
| VAMOS call (real server, M-series `mps`) | ~1.6–1.8 s | spec: ≤ 2–3 s per plan |
| VAMOS call (J4012, 25W, bf16, 5 beams, `--loc_head`) | ~2.3 s (`generate()`: 2.8 s) | spec: ≤ 2–3 s per plan; ≥ 0.5 Hz (≤ 2 s) not yet met |

The control loop runs at 20 Hz; VAMOS is asked back to back, as soon as the
last answer lands, from the newest frame (one call at a time, ~2.45 s on the
J4012 with `--loc_head`). Under `--vamos` the dog's pace is capped by that cycle: a hazard only
VAMOS sees is known at worst 2 L after it comes into view, so
`v <= (VLM_SEE_D - VLM_STOP_D) / 2 L` (2.0 m and 0.4 m, estimates; ~0.32 m/s
at L = 2.45 s), L the slowest of the last few answers.

On the HIL rig VAMOS runs in shadow by default (`VAMOS_MODE=shadow` in
`scripts/hil/onboard.sh` / `run_hil.sh`; `steer` is `--vamos` as above). It is
asked back to back as under `--vamos` and every answer is logged
(`output/shadow_log.jsonl`, copied to the run's folder), but the map route,
LiDAR and CE-RRT\* drive and there is no VLM speed cap, so the dog walks at up
to `MAX_V` (0.8 m/s). Why: no speed-up of the model on the 25W J4012 brings the
cap near walking pace (0.8 m/s needs L ~1.0 s; int8 + tensor cores reached
~2.0 s), zero-shot VAMOS does not yet steer better than the map (see the
comment above `on_vamos` in `run_building.py`), and the puddle it is there to
see is untested (`VLM_SEE_D` is a guess). The cost: in shadow nothing watches
for flat hazards the LiDAR cannot see -- the same as map-only. A fast floor
check is what would let the dog walk at pace with that covered; until then,
shadow is for the rig, not for walking with people.

The floor guard (`src/cyberdog/sim/sensing/floor_guard.py`, `--floor-guard sim`)
is that fast floor check, simulated: the twin has no water, so `--puddles N`
places N puddles on each leg's route (on open floor, out of doorways, leaving
0.9 m beside them), and the guard sees one when it would be in the camera's
image, within `--fg-see` (2.0 m) of the lens and not behind a wall, answering
`--fg-latency` (0.25 s) later, frames back to back, each puddle missed with
`--fg-miss` per frame. Known puddles become LiDAR-like returns, so the dog
goes round them as round a crate; the pace cap is the guard's own
`(see - 0.4) / 2 L` (3.2 m/s at the defaults, so `MAX_V` rules) in place of
VAMOS's. A real guard (a model on the camera image) replaces `FloorGuard.look`.
Twin, server room from the HIL start, 2 puddles a leg, seeds 1-3: no guard
63 s and both puddles walked into every time; guard 66-70 s, none walked
into, each known 1.7-2.0 m before its edge (the same with a 30% miss rate per
frame: at 0.25 s each puddle is in view for tens of frames). HIL (shadow,
4 people): 96 s, 2/2 seen 2.1-2.2 m out, none walked into.
Known puddles occupy ground without the LiDAR's SHADOW (`LiveClearance.update`'s
`flat`): the camera sees the floor beyond a puddle, and a shadowed puddle
closed the way round one. A puddle the guard has just found resets the
detour's side, as a new obstacle does.

The rules, for anything in the way: a way round -> go round and say so; no way
round (checked) -> stop and say so; a person and no way round -> stop and ask
them ("Excuse me, may we pass?", every `EXCUSE_S`). A dog going round
something that walks under `PROGRESS_M` in `BLOCKED_S` says "I cannot get past
this. Stopping here." and stops (it used to stand there silent). Stopped, it
may not turn on the spot towards a wall with its user (`UNPIN_DT`), and a
detour line may not cut a wall corner (`perception.SIDE_WALL`: it once aimed
into a side-door recess and hit the jamb). `--puddle-at X,Y,R[,F]` places one
exactly, for the blocked-way cases. Simulated people (`--crowd aware`, the
default) now stop for a dog right in front of them, as the oblivious ones did:
before, a head-on walker with no room either side walked to 0.19 m of a dog
standing still (`--pedestrians 3 --seed 2`); Gate D's aware numbers predate this.

Twin, server room from the HIL start, `--puddles 2 --pedestrians 3`, seeds
1-6: all arrive, no puddle walked into, no collision, the user never on a
wall; seed 1 has a walker passing the stopped dog at 0.54 m (`PED_NEAR` 0.55).
Known limit: selftest's `room 201 (with people, stress test)` (`--crowd
oblivious --planner map --seed 1`) now reports CONTACT, 0.35 m for 0.1 s on
floor 2: the dog had stood still for over 3 s and an oblivious crosser walked
past it. The run differs from before because `SIDE_WALL` turns down aims into
the north door recess by room 201's cartons; `SIDE_WALL = 0` gives the old run
tick for tick. Kept on the user's decision (2026-11-16): not cutting corners
is the safer rule.

Where a VAMOS call goes on the J4012 (25W caps the GPU at 408 MHz, so it
reads memory at ~40 GB/s, not the module's ~102 GB/s; GPU ~99% busy, the CPU
is not the limit -- a CUDA graph of the decode step saved ~1%): prefill (image
+ prompt, once, shared by the 5 beams) ~0.67 s, then 10 decode steps (10
`<loc>` tokens + EOS) at ~150-177 ms, each reading the 5.2 GB of weights once.
`--loc_head` (`vendor/VAMOS/server/loc_head.py`, on by default in
`scripts/hil/onboard.sh`, `VAMOS_LOC_HEAD=0` for `generate()`) scores only the
1,025 tokens VAMOS answers with instead of all 257,216: it skips most of the
output head (1.2 GB of the weights) and beam search's softmax/top-k over 5 x
257k scores. 2.80 -> 2.30 s a call; the top path is the same as
`generate()`'s in 32/32 test cases (all 5 paths in 31/32: near-tied lower
beams can swap, since the softmax is over the 1,025 tokens only).
