"""--frame-log: the frame VAMOS got and what became of its answer, on disk."""
import importlib.util
import json
import pathlib

import numpy as np

from cyberdog.sim.vamos_client import FrameLog, VamosPolicy

CAM = {"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0, "camera_height": 0.32}
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _policy(tmp_path, answer, error=None):
    p = VamosPolicy(cam=CAM, is_free=lambda x, y: y > -0.5, url="http://127.0.0.1:1",
                    frame_log=FrameLog(str(tmp_path)), floor=2)

    def fake(image, prompt):
        if error:
            raise ValueError(error)
        return answer
    p._request = fake
    return p


def test_frame_and_answer_are_saved_and_the_chosen_path_marked(tmp_path):
    # Two straight-ahead paths and one far to the right (off the map: y < -0.5).
    answer = [[[80, 120], [80, 80]], [[78, 120], [76, 80]], [[100, 120], [160, 70]]]
    p = _policy(tmp_path, answer)
    img = np.full((120, 160, 3), 128, np.uint8)
    paths = p.request(img, "go to room 201.", (0.0, 0.0, 0.0))
    chosen, _, safety = p.deliver(paths, (0.0, 0.0, 0.0), (5.0, 0.0))
    assert (tmp_path / "00000.jpg").exists()
    rec = json.loads((tmp_path / "00000.json").read_text())
    assert rec["floor"] == 2 and rec["prompt"] == "go to room 201."
    assert rec["pixel_paths"] == [[[float(u), float(v)] for u, v in pp] for pp in answer]
    assert len(rec["map_paths"]) == 3
    assert sum(v["chosen"] for v in rec["verdicts"]) == 1
    assert {v["verdict"] for v in rec["verdicts"]} == {"pass", "off-map"}
    assert rec["safety"] == safety == 1.0
    assert chosen is not None and p.frame_rec is None

    spec = importlib.util.spec_from_file_location("frames", ROOT / "scripts/vamos/frames.py")
    frames = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frames)
    names = frames.verdict_of(rec)
    assert names.count("chosen") == 1 and names[2] == "off-map"
    frames.draw(str(tmp_path / "00000.jpg"), rec).save(tmp_path / "x.png")


def test_a_failed_call_is_recorded(tmp_path):
    p = _policy(tmp_path, None, error="server down")
    assert p.request(np.zeros((120, 160, 3), np.uint8), "go.", (0, 0, 0)) is None
    p.deliver(None, (0, 0, 0), (5, 0))
    rec = json.loads((tmp_path / "00000.json").read_text())
    assert "server down" in rec["error"] and rec["verdicts"] == []


def test_off_by_default_writes_nothing(tmp_path):
    p = VamosPolicy(cam=CAM, is_free=lambda x, y: True, url="http://127.0.0.1:1")
    p._request = lambda image, prompt: [[[80, 120], [80, 80]]]
    p.deliver(p.request(np.zeros((120, 160, 3), np.uint8), "go.", (0, 0, 0)), (0, 0, 0), (5, 0))
    assert p.frame_log is None and not list(tmp_path.iterdir())
