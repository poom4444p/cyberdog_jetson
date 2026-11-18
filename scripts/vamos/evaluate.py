"""Score VAMOS, base or fine-tuned, on the held-out Gate D routes (offline).

Sends each held-out frame to the VAMOS server exactly as the live client does
(5-beam search, temperature 0) and compares the five answers with the path
the dog really walked from that frame, on the floor, in metres:

    path   mean distance from each walked point to the candidate's line
    end    distance between the candidate's end and the walked path's end
    best   the better of the five candidates, by `path`; top is beam 1
    spread how far the five candidates' ends lie from their mean -- the
           README's +/-0.25 m, which a fine-tune is supposed to widen where
           the walk needs it, not everywhere
    none   % of frames where NO candidate passes traversable() against the
           floor's static map -- the live "safety 0.00 ... chosen end -"
           answer, which leaves the dog on the map route. The KPI the LoRA
           is for (HIL 11-18: 23% of steer answers, 31% in detours, ~40% with
           the subgoal more than 15 deg off the camera axis)
    pass   candidates per frame that pass traversable(), of five

Reported for all frames, for *detour* frames -- those whose walked path
leaves the straight line to its own end by more than DETOUR m, the frames
that are about going round something and the reason for the fine-tune --
and by *heading off*: how far the map's goal pixel lies from the image
centre, atan((u - cx) / fx) with the camera level, in 0-5, 5-15 and >15 deg
buckets. `label ok` is the share of frames whose own walked path passes the
same test: the ceiling, since the static map is a little tighter than the
walk (doorways at ROBOT_R).

    conda activate cyberdog_sim                  # server in another terminal
    python scripts/vamos/evaluate.py --name base
    python scripts/vamos/evaluate.py --name lora --load <repo>/models/vamos_lora
    python scripts/vamos/evaluate.py --compare base lora
    python scripts/vamos/evaluate.py --rescore base   # re-judge saved answers, no server

Results go to output/vamos_eval/<name>.json, every frame's answer included.
This is the offline half of the judgement. The closed-loop half is Gate D,
with the server running the adapter.
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests

from cyberdog import paths
from cyberdog.planning.checkpoint_projector import load_camera_config
from cyberdog.sim.sensing.dreaming import ROBOT_R
from cyberdog.sim.sensing.perception import LiveClearance
from cyberdog.sim.vamos_client import DEFAULT_URL, VamosPolicy, ahead, pixel_to_ground

DATA = paths.DATASETS_DIR / "vamos" / "test"
OUT = paths.OUTPUT_DIR / "vamos_eval"
DETOUR = 0.15        # m off the straight line that makes a frame a detour (= train_lora.DETOUR_M)
OFF_BUCKETS = ((0, 5), (5, 15), (15, 90))   # heading off, degrees
LOC_X = re.compile(r"x=<loc(\d{4})>")


def seg_dist(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def line_dist(p, line):
    return min(seg_dist(p, a, b) for a, b in zip(line, line[1:]))


def bulge(path):
    """How far the walked path strays from the straight line to its end."""
    start = path[0]
    return max(seg_dist(p, start, path[-1]) for p in path)


def heading_off(prompt, cam):
    """Degrees between the camera axis and the goal in `prompt` (its pixel
    column, at the bin's centre), unsigned. The camera is level, so the
    column alone gives the bearing."""
    m = LOC_X.search(prompt)
    if not m:
        return None
    u = (int(m.group(1)) + 0.5) * cam["width"] / 1024
    return abs(math.degrees(math.atan((u - cam["cx"]) / cam["fx"])))


def bucket(off):
    if off is None:
        return None
    for lo, hi in OFF_BUCKETS[:-1]:
        if lo <= off < hi:
            return f"off {lo}-{hi}"
    return f"off >{OFF_BUCKETS[-1][0]}"


BUCKETS = [bucket(lo) for lo, _ in OFF_BUCKETS]
PARTS = ["all", "detour"] + BUCKETS


class StaticFree:
    """The brain's traversable() test against each floor's static map. Live,
    the gate is `LiveClearance(floor)(x, y) >= ROBOT_R` with the LiDAR laid
    over it; offline there is no LiDAR -- people and crates in the frame are
    on no map -- so this is the walls-and-furniture half of the live
    rejections (most of them: 48 of 100 off-map verdicts were inside a wall)."""

    def __init__(self, cam):
        self.cam = cam
        self.policy = {}

    def __call__(self, floor):
        if floor not in self.policy:
            live = LiveClearance(floor)
            self.policy[floor] = VamosPolicy(self.cam, lambda x, y: live(x, y) >= ROBOT_R)
        return self.policy[floor]


def frames(data_dir, limit, seed):
    index = json.loads((data_dir / "index.json").read_text())
    out = []
    for rid in index["clean_runs"]:
        for l in (data_dir / rid / "samples.jsonl").read_text().splitlines():
            s = json.loads(l)
            s["run"], s["image"] = rid, str(data_dir / rid / s["frame"])
            # The walk starts at the lens, not at the first visible point.
            s["detour"] = bulge([s["cam_pose"][:2]] + s["ground"]) > DETOUR
            out.append(s)
    if limit and len(out) > limit:
        # Every detour frame first, up to half: they are rarer and they are
        # what this is for. The rest drawn at random, the same draw each time.
        rng = random.Random(seed)
        det = [s for s in out if s["detour"]]
        rest = [s for s in out if not s["detour"]]
        rng.shuffle(det)
        rng.shuffle(rest)
        det = det[:limit // 2]
        out = det + rest[:limit - len(det)]
    return out


def candidates(s, answers, cam):
    """The answers on the floor, in map metres -- as VamosPolicy.request()."""
    cands = []
    for pp in answers:
        g = [pixel_to_ground(u, v, cam, s["cam_pose"]) for u, v in pp]
        g = [p for p in g if p is not None]
        g = [p for k, p in enumerate(g) if k == 0 or math.dist(p, g[k - 1]) > 0.05]
        if len(g) >= 2:
            cands.append(g)
    return cands


def passing(s, cands, free):
    """How many candidates the brain would keep before the dream, as
    VamosPolicy.deliver() does: cut to what lies ahead of the body, then
    traversable(). Judged from the pose the frame was taken at."""
    pol = free(s["floor"])
    pose = s.get("pose", s["cam_pose"])[:2]
    rest = [ahead(c, pose) for c in cands]
    return sum(1 for c in rest if len(c) >= 2 and pol.traversable(c))


def judge(rows, todo, cam, free):
    """Adds the traversable() KPI to each row; `todo` holds the rows' frames."""
    for r, s in zip(rows, todo):
        cands = candidates(s, r["answers"], cam)
        walked = [tuple(s["cam_pose"][:2])] + [tuple(p) for p in s["ground"]]
        r["n_cands"] = len(cands)
        r["n_pass"] = passing(s, cands, free)
        r["label_ok"] = passing(s, [walked], free) == 1
        r["off"] = heading_off(s["map_prompt"], cam)


def score(s, answers, cam):
    walked = [tuple(p) for p in s["ground"]]
    cands = candidates(s, answers, cam)
    if not cands:
        return None
    path = [statistics.mean(line_dist(p, c) for p in walked) for c in cands]
    end = [math.dist(c[-1], walked[-1]) for c in cands]
    ends = np.array([c[-1] for c in cands])
    spread = float(np.linalg.norm(ends - ends.mean(0), axis=1).mean())
    b = int(np.argmin(path))
    return {"top_path": path[0], "top_end": end[0], "best_path": path[b],
            "best_end": end[b], "spread": spread, "n": len(cands)}


def summary(rows):
    sels = {"all": rows, "detour": [r for r in rows if r["detour"]]}
    sels.update({b: [r for r in rows if bucket(r.get("off")) == b] for b in BUCKETS})
    out = {}
    for name, sel in sels.items():
        ok = [r["score"] for r in sel if r["score"]]
        out[name] = {"frames": len(sel), "answered": len(ok), **{
            k: round(statistics.mean(x[k] for x in ok), 3) if ok else None
            for k in ("top_path", "top_end", "best_path", "best_end", "spread")}}
        judged = [r for r in sel if "n_pass" in r]
        if judged:
            n = len(judged)
            out[name].update({
                "none": round(100 * sum(r["n_pass"] == 0 for r in judged) / n, 1),
                "pass": round(statistics.mean(r["n_pass"] for r in judged), 2),
                "label_ok": round(100 * sum(r["label_ok"] for r in judged) / n, 1)})
    return out


def table(results):
    keys = ("top_path", "best_path", "top_end", "best_end", "spread")
    for part in ("all", "detour"):
        print(f"\n{part} frames  " + "  ".join(f"{k:>9s}" for k in keys) + "   (metres)")
        for name, r in results.items():
            s = r["summary"][part]
            print(f"  {name:12s}" + "  ".join(
                f"{s[k]:9.3f}" if s[k] is not None else f"{'-':>9s}" for k in keys)
                  + f"   {s['answered']}/{s['frames']}")
    print("\nno candidate passes traversable() (static map): % frames, "
          "candidates passing of 5, walked path itself passing")
    print(f"  {'':12s}" + "".join(f"{p:>22s}" for p in PARTS))
    for name, r in results.items():
        cells = []
        for p in PARTS:
            s = r["summary"].get(p) or {}
            cells.append(f"{s['none']:5.1f}% {s['pass']:4.2f} ({s['label_ok']:.0f}%)"
                         if "none" in s else "-")
        print(f"  {name:12s}" + "".join(f"{c:>22s}" for c in cells))
        print(f"  {'  frames':12s}" + "".join(
            f"{(r['summary'].get(p) or {}).get('frames', 0):>22d}" for p in PARTS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", help="what to call this result (base, lora, ...)")
    ap.add_argument("--load", metavar="MODEL", help="ask the server to load this model first")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--limit", type=int, default=400, help="frames to send (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", choices=["map", "hindsight"], default="map",
                    help="goal in the prompt: the projector's, as asked live (default), "
                         "or the walked path's end, which gives the answer's end away")
    ap.add_argument("--compare", nargs="+", metavar="NAME", help="print saved results side by side")
    ap.add_argument("--rescore", metavar="NAME",
                    help="re-judge a saved result's answers against the static map, no server")
    args = ap.parse_args()

    cam = load_camera_config()
    if args.rescore:
        path = OUT / f"{args.rescore}.json"
        result = json.loads(path.read_text())
        by = {(s["run"], s["frame"]): s for s in frames(Path(result["data"]), 0, 0)}
        judge(result["frames"], [by[(r["run"], r["frame"])] for r in result["frames"]],
              cam, StaticFree(cam))
        result["summary"] = summary(result["frames"])
        path.write_text(json.dumps(result))
        table({args.rescore: result})
        return
    if args.compare:
        table({n: json.loads((OUT / f"{n}.json").read_text()) for n in args.compare})
        return
    if not args.name:
        ap.error("--name is required (or --compare)")

    if args.load:
        r = requests.post(f"{args.url}/load_model", json={"model_path": args.load}, timeout=900)
        r.raise_for_status()
        if not r.json().get("success"):
            raise SystemExit(r.json().get("message"))
        print(r.json()["message"])

    policy = VamosPolicy(cam, is_free=lambda x, y: True, url=args.url)
    if not policy.available():
        raise SystemExit(f"VAMOS server is not answering on {args.url} (main README, step 6)")
    todo = frames(Path(args.data), args.limit, args.seed)
    print(f"{len(todo)} frames ({sum(s['detour'] for s in todo)} detours) -> {args.url}")

    from PIL import Image
    rows, t0 = [], time.time()
    for k, s in enumerate(todo, 1):
        image = np.asarray(Image.open(s["image"]).convert("RGB"))
        try:
            answers = policy._request(image, s["map_prompt"] if args.prompt == "map" else s["prompt"])
        except (requests.RequestException, ValueError) as e:
            print(f"  {s['frame']}: {e}", file=sys.stderr)
            answers = []
        rows.append({"run": s["run"], "frame": s["frame"], "detour": s["detour"],
                     "answers": answers, "score": score(s, answers, cam)})
        if k % 25 == 0:
            print(f"  {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    judge(rows, todo, cam, StaticFree(cam))
    result = {"name": args.name, "load": args.load, "data": args.data,
              "map_prompt": args.prompt == "map", "summary": summary(rows), "frames": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(result))
    table({args.name: result})
    print(f"\n-> {OUT / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
