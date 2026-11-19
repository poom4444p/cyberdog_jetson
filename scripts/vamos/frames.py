"""Draw VAMOS's answers on the frames it was sent (--frame-log / FRAME_LOG=1).

    python scripts/vamos/frames.py OUT/frames [--sheet N] [--only 12,40,...]

For each NNNNN.jpg + NNNNN.json in the directory, writes NNNNN_paths.png
next to it: every candidate VAMOS drew, in the pixels it drew it, coloured
by what the brain made of it --

    green   chosen          orange  passed, not chosen
    red     safety gate     grey    off the map / walked past / unjudged

-- and a caption (seq, floor, wall time, safety, verdicts). Then contact
sheets of N frames each (sheet_KK.png, default 20) to look through quickly.
Measurement only: nothing here feeds back into the dog.
"""
import argparse
import glob
import json
import os

from PIL import Image, ImageDraw

COLOUR = {"chosen": (40, 220, 60), "pass": (255, 160, 30), "gate": (230, 40, 40),
          "off-map": (150, 150, 150), "behind": (150, 150, 150),
          "affordance": (180, 60, 220), None: (150, 150, 150)}


def verdict_of(rec):
    """Pixel path i -> verdict name ("chosen", "pass", "gate", ...) or None.

    A verdict's path is the map path cut to what was still ahead when it was
    judged, so it is matched to its pixel path by the end point.
    """
    out = []
    ends = [(tuple(v["path"][-1]) if v["path"] else None, v) for v in rec.get("verdicts", [])]
    for mp in rec.get("map_paths") or []:
        name = None
        if mp is not None:
            for end, v in ends:
                if end is not None and abs(end[0] - mp[-1][0]) < 1e-3 \
                        and abs(end[1] - mp[-1][1]) < 1e-3:
                    name = "chosen" if v["chosen"] else v["verdict"]
                    break
        out.append(name)
    return out


def draw(jpg, rec):
    img = Image.open(jpg).convert("RGB")
    d = ImageDraw.Draw(img)
    pix = rec.get("pixel_paths") or []
    names = verdict_of(rec) + [None] * len(pix)
    # Chosen last, so it is on top.
    order = sorted(range(len(pix)), key=lambda i: names[i] == "chosen")
    for i in order:
        pts = [tuple(p) for p in pix[i]]
        c = COLOUR.get(names[i], COLOUR[None])
        w = 4 if names[i] == "chosen" else 2
        if len(pts) >= 2:
            d.line(pts, fill=c, width=w)
        for u, v in pts:
            d.ellipse((u - 3, v - 3, u + 3, v + 3), outline=c)
    counts = {}
    for n in names[:len(pix)]:
        counts[n or "?"] = counts.get(n or "?", 0) + 1
    cap = (f"#{rec.get('seq')} f{rec.get('floor')} safety {rec.get('safety', '-')} "
           + " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
           + (f" ERROR {rec['error'][:40]}" if rec.get("error") else "")
           + ("" if "verdicts" in rec else " (never judged)"))
    d.rectangle((0, 0, img.width, 16), fill=(0, 0, 0))
    d.text((4, 2), cap, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--sheet", type=int, default=20, help="frames per contact sheet (0: none)")
    ap.add_argument("--only", default="", help="comma-separated seq numbers")
    a = ap.parse_args()
    only = {int(s) for s in a.only.split(",") if s}
    drawn = []
    for js in sorted(glob.glob(os.path.join(a.dir, "[0-9]" * 5 + ".json"))):
        with open(js) as f:
            rec = json.load(f)
        if only and rec.get("seq") not in only:
            continue
        img = draw(js[:-5] + ".jpg", rec)
        img.save(js[:-5] + "_paths.png")
        drawn.append(img)
    print(f"{len(drawn)} frames drawn in {a.dir}")
    if a.sheet and drawn:
        cols, w, h = 5, drawn[0].width // 2, drawn[0].height // 2
        for k in range(0, len(drawn), a.sheet):
            group = drawn[k:k + a.sheet]
            rows = (len(group) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
            for i, im in enumerate(group):
                sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
            path = os.path.join(a.dir, f"sheet_{k // a.sheet:02d}.png")
            sheet.save(path)
            print(f"  {path}")


if __name__ == "__main__":
    main()
