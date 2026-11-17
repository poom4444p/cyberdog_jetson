"""The affordance MLP at run time, in numpy: no torch on the robot's brain.

The brain's environment on the J4012 (go2env) deliberately has no torch: the
GPU and its memory are VAMOS's, and the network is three small Linear layers
that numpy runs in well under a millisecond. Training stays in model.py,
with torch; this file holds what both need -- the input features and the
refusal rule -- so the network is fed the same numbers either way, and the
weights move across as a .npz (model.py's `export`).

    features(patch, target)   the network's input, as in training
    decide(probs, refuse_p)   probabilities -> class, with the refusal rule
    load(path)                an NumpyMLP from mlp.npz (or mlp.pt, if torch is there)
    classify(net, ...)        the runtime's call
"""
import re
from pathlib import Path

import numpy as np

from cyberdog.affordance import data

H_CLIP = 0.5            # metres: anything higher or lower is "a lot", the same to us
N_CELLS = data.NY * data.NX
N_IN = 4 * N_CELLS + 2

# Refuse when the network gives "not walkable" at least this probability,
# whatever else it rates higher. Chosen for the people this is for: on the
# held-out tiles of run_20261006_132140, 0.5 (plain argmax) called 10.8% of
# not-walkable trials followable; 0.2 calls 5.4%, at the price of refusing
# 10.5% of followable ones (and 92.0% right on followable-or-not, spec > 90%).
# Too careful is a detour; not careful enough is a fall.
REFUSE_P = 0.2
JUMP_SCALE = 10.0       # a 0.1 m jump reads as 1
VIEW_HALF_W = 0.5       # metres either side of the walk the network may see


def jumps(patch):
    """(N, NY, NX): each cell's largest height jump to a seen 4-neighbour, metres.

    Only between two seen cells: next to an unseen one is "don't know", not an
    edge -- the border of what the LiDAR saw would otherwise read as a cliff."""
    p = np.asarray(patch, dtype=np.float32)
    out = np.zeros_like(p)
    with np.errstate(invalid="ignore"):
        for axis in (1, 2):
            d = np.abs(np.diff(p, axis=axis))
            d = np.where(np.isfinite(d), d, 0.0)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            out[tuple(lo)] = np.maximum(out[tuple(lo)], d)
            out[tuple(hi)] = np.maximum(out[tuple(hi)], d)
    return out


def features(patch, target):
    """(N, N_IN) float32 network input from (N, NY, NX) patches and (N, 2) targets."""
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    patch = np.where(data.path_cells(target, VIEW_HALF_W),
                     np.asarray(patch, dtype=np.float32), np.nan)
    seen = np.isfinite(patch)
    h = np.clip(np.where(seen, patch, 0.0), -H_CLIP, H_CLIP) / H_CLIP
    on = data.path_cells(target)
    jump = np.clip(jumps(patch) / H_CLIP, 0.0, 1.0) * (H_CLIP * JUMP_SCALE)
    n = len(patch)
    return np.concatenate([h.reshape(n, -1), seen.reshape(n, -1).astype(np.float32),
                           on.reshape(n, -1).astype(np.float32), jump.reshape(n, -1),
                           target], axis=1)


def decide(probs, refuse_p=REFUSE_P):
    """(N,) class per row of (N, 3) probabilities, refusing at refuse_p.

    NOT_WALKABLE when its probability reaches refuse_p; otherwise the more
    likely of walkable and caution."""
    probs = np.asarray(probs)
    out = np.where(probs[:, data.CAUTION] > probs[:, data.WALKABLE],
                   data.CAUTION, data.WALKABLE)
    out[probs[:, data.NOT_WALKABLE] >= refuse_p] = data.NOT_WALKABLE
    return out.astype(np.int8)


class NumpyMLP:
    """Linear, ReLU, ..., Linear: the trained network with dropout gone, as in eval()."""

    def __init__(self, weights, biases):
        self.weights = [np.asarray(w, np.float32) for w in weights]
        self.biases = [np.asarray(b, np.float32) for b in biases]

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if k < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x

    def arrays(self):
        """{name: array} for np.savez -- what `load` reads back."""
        out = {}
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            out[f"w{k}"], out[f"b{k}"] = w, b
        return out


def from_state(state):
    """A NumpyMLP from an AffordanceMLP state_dict: its Linear layers, in order."""
    idx = sorted(int(m.group(1)) for k in state
                 if (m := re.fullmatch(r"net\.(\d+)\.weight", k)))
    as_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return NumpyMLP([as_np(state[f"net.{i}.weight"]) for i in idx],
                    [as_np(state[f"net.{i}.bias"]) for i in idx])


def load(path):
    """(NumpyMLP, saved info) from mlp.npz; from mlp.pt if torch is installed."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        n = sum(1 for k in z.files if re.fullmatch(r"w\d+", k))
        net = NumpyMLP([z[f"w{k}"] for k in range(n)], [z[f"b{k}"] for k in range(n)])
        info = {"classes": tuple(str(c) for c in z["classes"])} if "classes" in z.files else {}
        return net, info
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return from_state(ckpt["state"]), ckpt


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classify(net, patch, target, refuse_p=REFUSE_P):
    """(classes, probabilities) for raw patches and targets -- the runtime's call."""
    probs = softmax(net(features(patch, target)))
    return decide(probs, refuse_p), probs
