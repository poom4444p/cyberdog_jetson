"""Collect VAMOS LoRA training data (spec L4 step 5) with the auto-labeller.

Map-only runs of `run_building --autolabel`, every named destination that is
not held out, in an empty building and with crowds of 4, 6 and 8 people a
floor (--people, one size per seed): a crowded corridor is part of what the
model has to see. Each frame is labelled with the path
the dog walked from it (sim/autolabel.py), and only clean runs -- arrived,
nothing touched -- go into the dataset.

    python scripts/vamos/collect.py                       # train set
    python scripts/vamos/collect.py --test                # the held-out routes
    python scripts/vamos/collect.py --seeds 0 1 --people 0 6 --pairs 0  # smaller

Held out (HOLDOUT): the Gate D routes and the HIL steer set (room 201 -- also
a Gate D route --, mechanical engineering lab, server room). The adapter is
judged on them offline (scripts/vamos/evaluate.py), closed-loop in the twin
(scripts/gate_d.py) and on the board (HIL steer), and a model that has seen
those exact walks would be judged on its memory. They share corridors with
the training routes -- there is one building -- so this is a held-out
*route*, not a held-out layout, and Gate A's grouped split is still owed.

The crowd and planner are the live defaults (aware people, CE-RRT* round
them), not the 2026-10 set's oblivious crowd and map planner: the label is
the path walked, and going round a person is what the fine-tune is for
(no-candidate answers come in detours and with the goal off the camera axis,
HANDOFF 11-18). Every sample carries `bulge` (m off the straight line) and
`off` (deg between the camera axis and the map's goal); index.json counts
frames by evaluate.py's parts, and train_lora.py repeats the detour and
off > 15 deg frames (--oversample, --oversample-off).

Writes datasets/vamos_L/<split>/<run>/ (frames, samples.jsonl, run.json, log.txt)
and datasets/vamos_L/<split>/index.json. No VAMOS server needed. The old
building's 2026-10 set stays in datasets/vamos/.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gate_d import ROUTES  # noqa: E402

from cyberdog import paths  # noqa: E402

from cyberdog.sim.autolabel import DETOUR_M  # noqa: E402
from cyberdog.sim.scene import pedestrians  # noqa: E402

OUT = paths.DATASETS_DIR / "vamos_L"
# Judged on, so never trained on: the Gate D routes and the HIL steer set.
HOLDOUT = list(dict.fromkeys(ROUTES + ["room 201", "mechanical engineering lab",
                                       "server room"]))
OFF_BUCKETS = ((0, 5), (5, 15), (15, 90))   # deg, as scripts/vamos/evaluate.py
PEOPLE = [0, 4, 6, 8]   # per floor, one for each of the default seeds 0-3
# Names that are not somewhere to walk to: the stairs end in a refusal in the
# corridor outside, and the lift and the hallway are the corridor itself.
NOT_PLACES = {"stairs", "staircase", "stairway", "stairwell", "elevator", "lift",
              "hallway", "main entrance"}


def destinations():
    names = json.loads((paths.ROOT / "data" / "building" / "locations.json").read_text())
    return [n for n in names if n not in NOT_PLACES and n not in HOLDOUT]


def parts(samples):
    """Frames per evaluate.py part: all, detour, heading off 0-5 / 5-15 / >15."""
    out = {"all": len(samples), "detour": sum(s["bulge"] > DETOUR_M for s in samples)}
    for lo, hi in OFF_BUCKETS:
        key = f"off >{lo}" if hi == OFF_BUCKETS[-1][1] else f"off {lo}-{hi}"
        out[key] = sum(lo <= s["off"] < hi for s in samples)
    return out


def run_id(command, seed, people):
    return f"{command.replace(', then ', '+').replace(' ', '_')}__seed{seed}_p{people}"


def drive(command, seed, people, split_dir, crowd, planner):
    rid = run_id(command, seed, people)
    out = split_dir / rid
    cmd = [sys.executable, "-m", "cyberdog.sim.run_building", command,
           "--auto-confirm", "--no-video", "--autolabel", str(out),
           "--pedestrians", str(people), "--seed", str(seed),
           "--crowd", crowd, "--planner", planner]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text(p.stdout + p.stderr)
    run = out / "run.json"
    r = json.loads(run.read_text()) if run.exists() else {"clean": False, "samples": 0,
                                                         "outcome": f"exit {p.returncode}"}
    samples = out / "samples.jsonl"
    lines = [json.loads(l) for l in samples.read_text().splitlines()] if samples.exists() else []
    return {"run": rid, "command": command, "seed": seed, "people": people, "clean": r["clean"],
            "samples": r["samples"], "outcome": r["outcome"], "parts": parts(lines),
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", action="store_true",
                    help="collect the held-out routes (HOLDOUT) instead of the train set")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="which crowd: the same seed is the same people")
    ap.add_argument("--people", nargs="+", type=int, default=PEOPLE,
                    help=f"people per floor for each seed, in order (0-{pedestrians.POOL}); "
                         "one value = the same for every seed")
    ap.add_argument("--pairs", type=int, default=20,
                    help="two-stop commands added to the train set, so legs start "
                         "somewhere other than the main entrance")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--crowd", choices=["oblivious", "aware"], default="aware",
                    help="people: aware make room (live default); oblivious = the 2026-10 set")
    ap.add_argument("--planner", choices=["map", "cerrt"], default="cerrt",
                    help="cerrt goes round people (live default); map waits = the 2026-10 set")
    ap.add_argument("--out", default=str(OUT), help="dataset root (<out>/<split>/...)")
    ap.add_argument("--only", nargs="+", metavar="COMMAND",
                    help="run just these commands (a quick look; index.json then covers only them)")
    args = ap.parse_args()

    if not paths.BUILDING_SCENE.exists():
        raise SystemExit("no building scene yet -- run: "
                         "python -m cyberdog.sim.scene.build_scene --building")
    if args.test:
        split, commands = "test", list(HOLDOUT)
    else:
        split, places = "train", destinations()
        rng = random.Random(0)
        commands = places + [", then ".join(rng.sample(places, 2)) for _ in range(args.pairs)]
    if len(args.people) == 1:
        args.people = args.people * len(args.seeds)
    if len(args.people) != len(args.seeds):
        ap.error("--people: one value, or one per seed")
    if max(args.people) > pedestrians.POOL:
        ap.error(f"--people: at most {pedestrians.POOL} a floor (scene/pedestrians.POOL)")
    if args.only:
        commands = args.only
    split_dir = Path(args.out) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s, n) for c in commands for s, n in zip(args.seeds, args.people)]
    print(f"{split}: {len(commands)} commands x seeds {args.seeds} "
          f"(people {args.people}) = {len(jobs)} runs, "
          f"crowd {args.crowd}, planner {args.planner} -> {split_dir}")

    t0, results = time.time(), []
    with ThreadPoolExecutor(args.jobs) as pool:
        for k, r in enumerate(pool.map(
                lambda j: drive(*j, split_dir, args.crowd, args.planner), jobs), 1):
            results.append(r)
            flag = "" if r["clean"] else f"  NOT CLEAN ({r['outcome']})"
            print(f"  [{k}/{len(jobs)}] {r['run']}: {r['samples']} frames, "
                  f"{r['wall_s']:.0f}s{flag}", flush=True)

    clean = [r for r in results if r["clean"]]
    totals = {k: sum(r["parts"][k] for r in clean) for k in parts([])}
    index = {"split": split, "seeds": args.seeds, "people": args.people, "crowd": args.crowd,
             "planner": args.planner, "holdout": HOLDOUT, "runs": results,
             "clean_runs": [r["run"] for r in clean],
             "samples": sum(r["samples"] for r in clean), "parts": totals}
    (split_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{len(clean)}/{len(results)} runs clean, {index['samples']} labelled frames "
          f"usable, {time.time() - t0:.0f}s -> {split_dir / 'index.json'}")
    print("  " + ", ".join(f"{k} {v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
