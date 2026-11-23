"""VAMOS in the loop: image + goal pixel -> candidate paths -> one chosen path.

The VLM runs as a separate process (VAMOS/server/vlm_server.py) because it
needs torch and this environment deliberately does not have it -- the same
split the real robot has, where the VLM is a service rather than an import.

"AI proposes, simple code disposes": VAMOS returns five candidate paths and a
deterministic scorer picks one. On the real system that scorer is the
affordance MLP; here it is the ground-truth map, which is honest for a PoC
and gives the MLP something to be measured against.

Up to three stages to the disposing. First a cheap geometric test throws out
anything whose drawn line leaves free space -- no point imagining a path that
is already through a wall. Then, if one is given, the affordance network
(spec L5, cyberdog.affordance) judges the ground each path walks onto from
the LiDAR's elevation patch, and refuses what it calls not walkable: the
map's clearance knows walls, not a step or a kerb the map never had. This
is the paper's affordance re-ranking, as a filter in front of the dream
rather than in place of it. What survives is handed to dreaming.py, which rolls
the dog's own controller forward along it with noise and comes back with a
safety factor: the probability of getting down it without hitting anything,
weighted by how much room it left. That factor gates (below GATE, rejected --
and so does a path the dream cannot show is P_SAFE_MIN safe, see passes()),
ranks (safest first, progress second) and throttles (the chosen path's factor
scales the dog's speed, so a tight path is walked slowly rather than briskly).
"""
import io
import json
import math
import os
import time

import requests
from PIL import Image

DEFAULT_URL = "http://127.0.0.1:8009"

# The spec's confidence gate. A candidate the dog cannot imagine surviving at
# least this well is not offered, however much progress it would have made.
GATE = 0.5


def passes(factor, why):
    """The gate: factor >= GATE, and P(safe) >= P_SAFE_MIN with confidence.

    The factor alone does not bound collision: P(safe) x (0.4 + 0.6 room)
    lets a roomy path that hits one time in four score 0.525. So the dream's
    own verdict on P_SAFE_MIN (dreaming.Dream.factor's `confident`, the
    Wilson lower bound over up to N_DREAMS_MAX rollouts) has to agree too.
    A `why` without that verdict (a stand-in dream) is judged on the factor.
    """
    return factor >= GATE and why.get("confident", True)

# Safety is the first sort key, but not to three decimal places: two paths
# whose factors differ by a hundredth are equally safe, and progress should
# decide between them.
BAND = 0.05

# The brain's control tick, in ms. Dreaming runs inside it, on the tick that
# delivers VAMOS's answer: a frame whose dreams take longer than this holds
# that tick up. Counted in stats (dream_over_tick) and printed per floor.
TICK_MS = 50.0


def pixel_to_ground(u, v, cam, cam_pose):
    """Inverse of checkpoint_projector.project_to_pixel.

    A pixel only maps back to a ground point if it is below the horizon
    (v > cy); anything at or above it is infinitely far away.
    """
    cy, fy, fx, cx = cam["cy"], cam["fy"], cam["fx"], cam["cx"]
    if v <= cy + 1e-6:
        return None
    z_cam = fy * cam["camera_height"] / (v - cy)      # forward distance
    x_cam = (u - cx) * z_cam / fx
    x_fwd, y_left = z_cam, -x_cam

    rx, ry, yaw = cam_pose
    return (rx + math.cos(yaw) * x_fwd - math.sin(yaw) * y_left,
            ry + math.sin(yaw) * x_fwd + math.cos(yaw) * y_left)


def ahead(path, xy):
    """The part of `path` still in front of a dog standing at `xy`.

    An answer is about the frame it was asked from, and the dog has walked on
    since. path_target steers at the first point a lookahead away, and a
    point a lookahead *behind* qualifies -- so an unwalked stale path turned
    the dog round to chase its own start. Cut at the nearest point instead.
    """
    if not path:
        return []
    k = min(range(len(path)), key=lambda j: math.dist(path[j], xy))
    return path[k:]


class FrameLog:
    """--frame-log: every frame sent to VAMOS, saved with what came back.

    Measurement only, nothing reads it back. Per call, in `root`:
    NNNNN.jpg (the image as VAMOS got it) and NNNNN.json (wall time, floor,
    lens pose, prompt, VAMOS's pixel paths, the same projected to the map,
    and -- once judged -- each candidate's verdict and which was chosen).
    scripts/vamos/frames.py draws the paths on the frames.
    """

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.seq = 0

    def save(self, image, record):
        """Write one frame and its request; returns its sequence number."""
        n, self.seq = self.seq, self.seq + 1
        Image.fromarray(image).save(os.path.join(self.root, f"{n:05d}.jpg"), quality=90)
        self.write(n, record)
        return n

    def write(self, n, record):
        with open(os.path.join(self.root, f"{n:05d}.json"), "w") as f:
            json.dump(record, f)


def _rnd(path):
    return [[round(a, 3), round(b, 3)] for a, b in path]


class VamosPolicy:
    """Asks the VLM for paths, scores them, returns the winner in map frame."""

    def __init__(self, cam, is_free, dream=None, url=DEFAULT_URL, num_samples=5,
                 timeout=120, sample=False, affordance=None, frame_log=None, floor=None):
        self.cam = cam
        # FrameLog or None; the record of the call deliver() will judge next
        # (one call at a time: the brain never asks while one is running).
        self.frame_log, self.floor = frame_log, floor
        self.frame_rec = None
        self.is_free = is_free          # (x, y) -> bool, stands in for the affordance MLP
        self.dream = dream              # dreaming.Dream, or None for geometry only
        # (paths, pose) -> one class name per path ("walkable", "caution",
        # "not walkable") or None where it cannot say; None is no affordance
        # stage. A callable, so this module stays free of torch.
        self.affordance = affordance
        self.url = url
        self.num_samples = num_samples
        self.timeout = timeout
        # Beam search unless asked to sample. The server's default is
        # temperature 1.0 with no seed, so the same frame gave five different
        # paths every call and no two --vamos runs of one route were alike --
        # the same command brushed a wall on one run and not the next, and a
        # paired A/B (Gate D) compared noise. num_beams=num_samples at
        # temperature 0 returns the five most likely paths, the same five
        # every time, still distinct, at the same 1.6 s (measured, room 201
        # from the floor-2 lift). Plain temperature 0 is greedy and can only
        # return one.
        self.sample = sample
        self.last = None
        self.safety = 1.0               # factor of the path currently being followed
        # Last call's candidates, each with what became of it: (path, factor,
        # verdict), verdict "behind", "off-map", "affordance", "gate" or
        # "pass". Factor is None where
        # the dream never ran. Shadow mode logs this; nothing steers by it.
        self.verdicts = []
        self.stats = {"calls": 0, "failures": 0, "rejected": 0,
                      "rejected_by_gate": 0, "rejected_behind": 0,
                      "rejected_by_affordance": 0, "affordance_unjudged": 0,
                      "safety_sum": 0.0, "chosen": 0,
                      # wall-clock ms dreaming per answered call (frames that
                      # dreamed at least one candidate): count, sum, worst, and
                      # how many took longer than TICK_MS
                      "dream_frames": 0, "dream_ms_sum": 0.0, "dream_ms_max": 0.0,
                      "dream_over_tick": 0}
        self.dream_ms = None            # last call's dreaming time, None if it dreamed nothing
        self.last_error = None     # why the latest failed call failed, for the log

    def available(self):
        try:
            return requests.get(f"{self.url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    def _request(self, image, prompt):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        buf.seek(0)
        r = requests.post(
            f"{self.url}/predict",
            files={"image": ("frame.png", buf, "image/png")},
            data={"text_prompt": prompt, "num_samples": self.num_samples,
                  "max_tokens": 20,
                  **({} if self.sample else {"temperature": 0,
                                             "num_beams": self.num_samples})},
            timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return d.get("trajectories") or [] if d.get("success") else []

    def traversable(self, path, step=0.1):
        """Does the drawn line itself stay in free space?

        The cheap first pass. Sampling along each segment, not just at the
        returned points: they are 0.5-2 m apart, so a point-only test happily
        passes a path that cuts straight through a wall between two free
        samples.
        """
        if len(path) < 2:
            return False
        for a, b in zip(path, path[1:]):
            n = max(int(math.dist(a, b) / step), 1)
            for k in range(n + 1):
                x = a[0] + (b[0] - a[0]) * k / n
                y = a[1] + (b[1] - a[1]) * k / n
                if not self.is_free(x, y):
                    return False
        return True

    def progress(self, path, destination):
        """How much closer to the map's destination this path's end would get us."""
        return -math.hypot(path[-1][0] - destination[0], path[-1][1] - destination[1])

    def plan(self, image, prompt, cam_pose, destination, pose=None):
        """Returns (chosen_path, all_paths, safety) in map coordinates.

        `pose` is the body pose the imagined rollouts start from; it defaults
        to the camera pose, which is the same heading half a head further on.

        Ask and judge in one go -- the answer judged from the pose it was asked
        from. The live loop splits the two (request now, deliver when the
        answer lands) so it never waits on the VLM; this is that with no gap.
        """
        pose = pose if pose is not None else cam_pose
        return self.deliver(self.request(image, prompt, cam_pose), pose, destination)

    def request(self, image, prompt, cam_pose):
        """The slow half: ask the VLM. Candidate paths in map coordinates, or
        None if the server failed. Nothing is judged or counted here.

        Map coordinates are fixed at the moment of asking, from the lens pose
        the image was taken at -- which is what lets deliver() judge them from
        wherever the dog has got to by the time they arrive.
        """
        rec = None
        if self.frame_log is not None:
            rec = {"wall_t": round(time.time(), 3), "floor": self.floor,
                   "cam_pose": [round(v, 4) for v in cam_pose], "prompt": prompt}
            rec["seq"] = self.frame_log.save(image, rec)
            self.frame_rec = rec
        try:
            pixel_paths = self._request(image, prompt)
        except (requests.RequestException, ValueError) as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            if rec is not None:
                rec["error"] = self.last_error
                self.frame_log.write(rec["seq"], rec)
            return None

        paths, kept = [], []       # kept[i]: pixel path i's map path, or None
        for pp in pixel_paths:
            pts = [g for g in (pixel_to_ground(u, v, self.cam, cam_pose) for u, v in pp)
                   if g is not None]
            # The decoder pads with the last token, so drop repeats.
            dedup = [p for k, p in enumerate(pts) if k == 0 or math.dist(p, pts[k - 1]) > 0.05]
            if len(dedup) >= 2:
                paths.append(dedup)
            kept.append(dedup if len(dedup) >= 2 else None)
        if rec is not None:
            rec["latency_s"] = round(time.time() - rec["wall_t"], 3)
            rec["pixel_paths"] = [[[round(u, 1), round(v, 1)] for u, v in pp]
                                  for pp in pixel_paths]
            # Aligned with pixel_paths; None where too little was below the horizon.
            rec["map_paths"] = [None if p is None else _rnd(p) for p in kept]
            self.frame_log.write(rec["seq"], rec)
        return paths

    def _count_dream(self, ms):
        """One frame's dreaming time, into stats."""
        self.dream_ms = ms
        st = self.stats
        st["dream_frames"] += 1
        st["dream_ms_sum"] += ms
        st["dream_ms_max"] = max(st["dream_ms_max"], ms)
        st["dream_over_tick"] += ms > TICK_MS

    def dream_summary(self):
        """One line on dreaming time per frame, worst against TICK_MS."""
        st = self.stats
        k = st["dream_frames"]
        if not k:
            return "dream time: no frame dreamed"
        return (f"dream time: {k} frames, mean {st['dream_ms_sum'] / k:.1f} ms, "
                f"worst {st['dream_ms_max']:.1f} ms, {st['dream_over_tick']} over the "
                f"{TICK_MS:.0f} ms tick")

    def deliver(self, paths, pose, destination):
        """See _deliver; with --frame-log, also records what became of the
        candidates in the frame's record."""
        out = self._deliver(paths, pose, destination)
        rec, self.frame_rec = self.frame_rec, None
        if rec is not None:
            chosen = out[0]
            rec["judged_pose"] = [round(v, 4) for v in pose]
            rec["verdicts"] = [{"path": _rnd(p), "verdict": v,
                                "factor": None if f is None else round(f, 4),
                                "chosen": p is chosen} for p, f, v in self.verdicts]
            rec["safety"] = round(out[2], 4)
            rec["dream_ms"] = None if self.dream_ms is None else round(self.dream_ms, 2)
            self.frame_log.write(rec["seq"], rec)
        return out

    def _deliver(self, paths, pose, destination):
        """The fast half: judge a request's answer from `pose`, the body pose
        *now*. Returns (chosen_path, all_paths, safety), as plan() does.

        Each path is cut to what is still ahead of the dog, then tested and
        dreamed against the clearance field as it is now -- so a path that was
        fine from where the image was taken, and runs into something the LiDAR
        has found since, is rejected on arrival rather than followed.
        """
        self.stats["calls"] += 1
        self.verdicts = []
        self.dream_ms = None
        if paths is None:
            self.stats["failures"] += 1
            self.safety = 0.0
            return None, [], 0.0

        # Walked past already: nothing left of it to follow.
        rest = [ahead(p, pose[:2]) for p in paths]
        spent = [p for p in rest if len(p) < 2]
        paths = [p for p in rest if len(p) >= 2]
        self.stats["rejected"] += len(spent)
        self.stats["rejected_behind"] += len(spent)
        self.verdicts = [(p, None, "behind") for p in spent]

        # Cheap test first: no sense imagining a path already through a wall.
        walkable = [p for p in paths if self.traversable(p)]
        self.stats["rejected"] += len(paths) - len(walkable)
        self.verdicts += [(p, None, "off-map") for p in paths if p not in walkable]

        # The ground itself: what the map cannot know is there. A path the
        # network cannot judge (too little seen, or turning) goes on to the
        # dream unjudged -- no judgement is not a refusal.
        if self.affordance is not None and walkable:
            classes = self.affordance(walkable, pose)
            kept = []
            for p, c in zip(walkable, classes):
                if c == "not walkable":
                    self.stats["rejected_by_affordance"] += 1
                    self.stats["rejected"] += 1
                    self.verdicts.append((p, None, "affordance"))
                else:
                    self.stats["affordance_unjudged"] += c is None
                    kept.append(p)
            walkable = kept

        scored = []
        t0 = time.perf_counter()
        for p in walkable:
            if self.dream is None:
                scored.append((1.0, p, {}))
                self.verdicts.append((p, None, "pass"))
                continue
            factor, why = self.dream.factor(p, pose)
            if not passes(factor, why):
                self.stats["rejected_by_gate"] += 1
                self.stats["rejected"] += 1
                self.verdicts.append((p, factor, "gate"))
                continue
            scored.append((factor, p, why))
            self.verdicts.append((p, factor, "pass"))
        if self.dream is not None and walkable:
            self._count_dream((time.perf_counter() - t0) * 1000.0)

        if not scored:
            self.safety = 0.0
            return None, rest, 0.0

        # Safest first, progress second -- banded, so a hundredth of safety
        # does not outrank getting somewhere.
        factor, best, _ = max(scored, key=lambda t: (round(t[0] / BAND),
                                                     self.progress(t[1], destination)))
        self.last, self.safety = best, factor
        self.stats["safety_sum"] += factor
        self.stats["chosen"] += 1
        return best, rest, factor
