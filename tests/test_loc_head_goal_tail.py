"""The VAMOS server's goal tail (vendor/VAMOS/server/loc_head.py, --goal_tail).

A scripted stand-in for the model: logits depend only on the step, so the
answer is known. Needs torch (the Mac env; skipped on the brain's ~/go2env).
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]
                       / "vendor" / "VAMOS" / "server"))
import loc_head  # noqa: E402
from loc_head import LocHeadDecoder  # noqa: E402

EOS, PAD, LOC0 = 1, 0, 10
V = LOC0 + 1024                                          # pad, eos, 8 others, 1024 <loc>
P = 6                                                    # prompt length


def loc(i):
    return LOC0 + i


class Tokenizer:
    unk_token_id = 2

    def convert_tokens_to_ids(self, tokens):
        return [loc(int(t[4:8])) if t.startswith("<loc") else self.unk_token_id for t in tokens]


class Cache:
    def batch_repeat_interleave(self, n):
        pass

    def reorder_cache(self, idx):
        pass


class Body:
    """Logits for generated token k (1-based): `script[k]` is a list of
    (token id, logit); everything else -20. Ignores the beam's history."""

    def __init__(self, script):
        self.script = script
        self.forwards = 0

    def _row(self, k):
        row = torch.full((V,), -20.0)
        for t, v in self.script.get(k, [(EOS, 0.0)]):
            row[t] = v
        return row

    def __call__(self, input_ids=None, position_ids=None, use_cache=True, **kw):
        if position_ids is None:                          # prefill: logits for token 1
            rows = self._row(1)[None, None, :].expand(input_ids.shape[0], input_ids.shape[1], V)
        else:                                             # decode at 1-indexed position P + k
            self.forwards += 1
            k = int(position_ids[0, 0]) - P + 1
            rows = self._row(k)[None, None, :].expand(input_ids.shape[0], 1, V)
        return types.SimpleNamespace(last_hidden_state=rows.clone(), past_key_values=Cache())


class Model:
    def __init__(self, script):
        self.model = Body(script)
        self.generation_config = types.SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD)
        self._head = torch.nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self._head.weight.copy_(torch.eye(V))         # logits = hidden state

    def get_output_embeddings(self):
        return self._head


def five_points(last=(512, 600)):
    """VAMOS's shape: 4 points with two near-tied choices per token (so the 5
    beams differ), then `last` and EOS near-certain, as VAMOS copies its goal."""
    path = [500, 700, 500, 650, 500, 620, 500, 605]
    script = {k + 1: [(loc(t), 0.0), (loc(t + 3), -0.7), (loc(t - 3), -1.1)] for k, t in enumerate(path)}
    script[9], script[10] = [(loc(last[0]), 10.0)], [(loc(last[1]), 10.0)]
    script[11] = [(EOS, 10.0)]
    return script


GOAL = (loc(512), loc(600))
INPUT = {"input_ids": torch.zeros((1, P), dtype=torch.long)}


def decode(script, goal=None, max_new_tokens=20, subset=True):
    m = Model(script)
    dec = LocHeadDecoder(m, Tokenizer(), num_beams=5, subset=subset)
    with torch.no_grad():
        seq = dec.generate(dict(INPUT), max_new_tokens=max_new_tokens, num_return_sequences=5, goal=goal)
    return seq, dec, m.model.forwards


def answers(seq):
    return [[t - LOC0 if t >= LOC0 else ("EOS" if t == EOS else t) for t in s[P:].tolist()] for s in seq]


def test_off_by_default_is_the_full_decode():
    seq, dec, fw = decode(five_points())
    assert dec.last_tail is None and dec.tail_used == dec.tail_fallback == 0
    assert fw == 10 and all(a[-3:] == [512, 600, "EOS"] for a in answers(seq))


def test_tail_gives_the_same_paths_with_three_steps_fewer():
    full, _, fw_full = decode(five_points())
    tail, dec, fw_tail = decode(five_points(), goal=GOAL)
    assert dec.last_tail == "used" and dec.tail_used == 1
    assert fw_full - fw_tail == 3                          # 10 decode forwards -> 7
    # The same five paths and the same best one. The script gives exact ties
    # (one -0.7 choice at token 2 or at token 4), which may come out in either order.
    assert sorted(answers(full)) == sorted(answers(tail))
    assert torch.equal(full[0], tail[0])


def test_tail_ends_an_off_goal_path_at_the_goal():
    # The model's last point is (900, 900), not the goal: the J4012 saw this
    # with a goal at x=100, y=900 (x copied from y). Same first 4 points.
    full, _, _ = decode(five_points(last=(900, 900)))
    tail, dec, _ = decode(five_points(last=(900, 900)), goal=GOAL)
    assert dec.last_tail == "used"
    assert all(a[-3:] == [900, 900, "EOS"] for a in answers(full))
    assert all(a[-3:] == [512, 600, "EOS"] for a in answers(tail))
    assert sorted(a[:8] for a in answers(full)) == sorted(a[:8] for a in answers(tail))


@pytest.mark.parametrize("eos, early_returned", [(0.3, False), (0.6, False), (2.0, True)])
def test_early_end_falls_back_to_the_full_decode(eos, early_returned):
    # A model that may end after 2 points: not the 5-point shape. Falls back
    # whenever a kept beam ended early, even if the full decode would not
    # return it (0.3, 0.6); at 2.0 every answer ends early and the search
    # stops before the tail.
    script = five_points()
    script[5] = [(EOS, eos)] + script[5]
    full, _, _ = decode(script)
    tail, dec, fw = decode(script, goal=GOAL)
    assert dec.last_tail == "fallback" and dec.tail_fallback == 1 and dec.tail_used == 0
    assert torch.equal(full, tail)
    assert any("EOS" in a[:5] for a in answers(tail)) == early_returned


def test_no_room_for_the_tail_falls_back():
    full, _, _ = decode(five_points(), max_new_tokens=10)
    tail, dec, _ = decode(five_points(), goal=GOAL, max_new_tokens=10)
    assert dec.last_tail == "fallback" and torch.equal(full, tail)


def test_full_head_falls_back():
    full, _, _ = decode(five_points(), subset=False)
    tail, dec, _ = decode(five_points(), goal=GOAL, subset=False)
    assert dec.last_tail == "fallback" and torch.equal(full, tail)


def test_tail_starts_after_four_points():
    assert loc_head.GOAL_TAIL_AT == 8


class Peft:
    """A PeftModel's shape: get_base_model() is the LM; `.model` (attribute
    forwarding) is the whole LM too, whose forward has no last_hidden_state."""

    def __init__(self, lm):
        self._lm = lm
        self.model = types.SimpleNamespace()               # the LM: logits only
        self.generation_config = lm.generation_config

    def get_base_model(self):
        return self._lm

    def get_output_embeddings(self):
        return self._lm.get_output_embeddings()


def test_lora_model_is_unwrapped_to_the_body():
    full, _, _ = decode(five_points(), goal=GOAL)
    lm = Model(five_points())
    dec = LocHeadDecoder(Peft(lm), Tokenizer(), num_beams=5)
    with torch.no_grad():
        seq = dec.generate(dict(INPUT), max_new_tokens=20, num_return_sequences=5, goal=GOAL)
    assert dec.last_tail == "used"
    assert torch.equal(full, seq)
