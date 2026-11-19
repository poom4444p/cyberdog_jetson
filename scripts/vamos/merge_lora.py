"""Merge a VAMOS LoRA adapter into its base and save a plain model.

The server loads any path with "lora" in it as base + PeftModel; unmerged,
every call pays for the adapter's extra matmuls (lora_L on the J4012: 2.6 s a
call against 1.8 s for the base). Merged, the weights are the base's shape
again, so the out path must NOT contain "lora": the server then loads it as a
plain model.

    conda activate vamos_mac
    python scripts/vamos/merge_lora.py models/vamos_lora_L models/vamos_L_merged
    python scripts/vamos/merge_lora.py models/vamos_lora_L models/vamos_L_merged --check 8

--check N first answers N test frames (greedy, the server's prompt) with the
adapter, then merges in place and answers them again with the merged model,
and prints both. Identical is expected; bf16 rounding of W + BA may flip a
token now and then, and evaluate.py on the board is the real check. One
model in memory at a time (bf16, ~6 GB): the merge is done on the loaded
adapter, the delta in fp32 (peft does so on the CPU).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "datasets" / "vamos_L" / "test"
DETOUR_M = 0.15     # = train_lora.DETOUR_M (vamos_mac has no cyberdog package)


def test_frames(n, seed=0):
    """n test frames, half of them detours, drawn the same way each time
    (as evaluate.py draws): run starts alone all say "straight ahead"."""
    index = json.loads((TEST / "index.json").read_text())
    det, rest = [], []
    for rid in index["clean_runs"]:
        for l in (TEST / rid / "samples.jsonl").read_text().splitlines():
            s = json.loads(l)
            item = (TEST / rid / s["frame"], s["map_prompt"])
            (det if s.get("bulge", 0) > DETOUR_M else rest).append(item)
    rng = random.Random(seed)
    rng.shuffle(det)
    rng.shuffle(rest)
    det = det[:n // 2]
    return det + rest[:n - len(det)]


@torch.no_grad()
def answer(model, processor, frames):
    out = []
    for image, prompt in frames:
        enc = processor(text=["<image><bos>" + prompt], images=[Image.open(image).convert("RGB")],
                        return_tensors="pt").to(model.device)
        enc["pixel_values"] = enc["pixel_values"].to(model.dtype)
        ids = model.generate(**enc, max_new_tokens=25, do_sample=False)
        out.append(processor.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="compare adapter vs merged answers on N test frames")
    a = ap.parse_args()
    if "lora" in str(a.out).lower():
        sys.exit(f"{a.out}: the server would load a path with 'lora' in it as an adapter")

    base = PeftConfig.from_pretrained(str(a.adapter)).base_model_name_or_path
    print(f"base {base} + adapter {a.adapter}")
    processor = AutoProcessor.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(a.adapter)).eval()

    frames = test_frames(a.check) if a.check else []
    before = answer(model, processor, frames)
    model = model.merge_and_unload()
    after = answer(model, processor, frames)
    same = sum(b == c for b, c in zip(before, after))
    for (image, _), b, c in zip(frames, before, after):
        print(f"{'same' if b == c else 'DIFF'}  {image.parent.name}/{image.name}\n  adapter {b}\n  merged  {c}")
    if frames:
        print(f"{same}/{len(frames)} answers identical")

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    (a.out / "merged_from.json").write_text(json.dumps(
        {"base": base, "adapter": str(a.adapter), "check": [same, len(frames)]}, indent=1) + "\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
