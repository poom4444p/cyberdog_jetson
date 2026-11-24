"""The voice loop's decisions (cyberdog/voice/text.py) and its hooks in
run_building, with no microphone, speaker or model: text and synthetic audio."""
import concurrent.futures
import types

import numpy as np
import pytest

from cyberdog.planning.building_router import BuildingRouter
from cyberdog.sim.run_building import (Run, VOICE_TRIES, listen_for_stops,
                                       spoken_prompt)
from cyberdog.voice import text as T


# --- the wake phrase ----------------------------------------------------------
@pytest.mark.parametrize("said, rest", [
    ("Hey dog, take me to room 201.", "take me to room 201"),
    ("Hey Dog!", ""),
    ("Hey Doug, the library", "the library"),
    ("hey dawg take me to the cafeteria", "take me to the cafeteria"),
    ("Hi dog. I need to pee.", "I need to pee"),
    ("Okay, hey dog, the server room", "the server room"),
    ('"They dog, I need to pee on the second floor. "', "I need to pee on the second floor"),
    ('"Hey dog, I need to pee on the second floor"', "I need to pee on the second floor"),
])
def test_wake_heard(said, rest):
    assert T.split_wake(said) == (True, rest)


@pytest.mark.parametrize("said", [
    "the dog is over there", "take me to room 201", "hey", "", "[BLANK_AUDIO]",
    "I said hey to the dog over there",          # the call has to come first
])
def test_wake_not_heard(said):
    assert T.split_wake(said) == (False, "")


def test_wake_phrase_is_a_setting():
    assert T.split_wake("Hey Cyberdog, room 101", "hey cyberdog") == (True, "room 101")
    assert T.split_wake("Hey dog, room 101", "hey cyberdog")[0] is False


# --- yes / no -----------------------------------------------------------------
@pytest.mark.parametrize("said, answer", [
    ("Yes.", True), ("Yes, please.", True), ("Yeah", True), ("okay", True),
    ("No.", False), ("Nope", False), ("No, the library.", False), ("Stop", False),
    ("Yes no", None), ("Wait.", None), ("the library", None), ("", None),
    ("[BLANK_AUDIO]", None), ("Thank you.", None),
])
def test_yes_no(said, answer):
    assert T.parse_yes_no(said) is answer


def test_clean_drops_whispers_silence_markers():
    assert T.clean("[BLANK_AUDIO]") == ""
    assert T.clean("(music) Yes. [noise]") == "Yes."
    assert T.clean("Thank you.") == ""                 # a phantom on its own...
    assert T.clean("Thank you, room 101") == "Thank you, room 101"   # ...not inside speech


def test_clean_keeps_a_repeated_sentence_once():
    assert T.clean("Hey dog, take me to room 201. Hey dog, take me to room 201.") == \
        "Hey dog, take me to room 201."
    assert T.clean("Yes. Yes.") == "Yes."
    assert T.clean("Room 101. Then room 102.") == "Room 101. Then room 102."


# --- the segmenter ------------------------------------------------------------
def _tone(seconds, level):
    n = int(seconds * T.RATE)
    t = np.arange(n) / T.RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def _feed(stream):
    seg, out = T.Segmenter(), []
    for i in range(0, len(stream) - T.FRAME + 1, T.FRAME):
        u = seg.feed(stream[i:i + T.FRAME])
        if u is not None:
            out.append(u)
    return out


def test_segmenter_finds_one_utterance_in_noise():
    rng = np.random.default_rng(0)
    noise = lambda s: rng.normal(0, 40, int(s * T.RATE)).astype(np.int16)
    out = _feed(np.concatenate([noise(1.0), _tone(1.2, 4000), noise(1.5)]))
    assert len(out) == 1
    assert 1.2 <= len(out[0]) / T.RATE <= 1.2 + T.PRE_S + T.SILENCE_S + 0.2


def test_segmenter_ignores_a_click():
    quiet = np.zeros(int(1.5 * T.RATE), np.int16)
    assert _feed(np.concatenate([quiet, _tone(0.08, 8000), quiet])) == []


def test_segmenter_cuts_a_monologue_at_max():
    out = _feed(np.concatenate([np.zeros(T.RATE, np.int16), _tone(T.MAX_S + 3, 4000)]))
    assert out and len(out[0]) / T.RATE <= T.MAX_S + 0.1


# --- run_building's hooks -----------------------------------------------------
def test_spoken_prompt():
    assert spoken_prompt('I heard "libery". Did you mean the library? Press continue '
                         "to go there.") == ('I heard "libery". Did you mean the library? '
                                             "Say yes to go there.")
    assert spoken_prompt("Press the call button for floor 2, then press continue.") == \
        "Press the call button for floor 2, then say yes."


class FakeListener:
    """Hands out scripted answers, the way Listener.yes_no_async would."""

    def __init__(self, answers=(), commands=()):
        self.answers, self.commands = list(answers), list(commands)

    def yes_no_async(self, timeout=None):
        f = concurrent.futures.Future()
        said = self.answers.pop(0) if self.answers else None
        f.set_result((None if not said else T.parse_yes_no(said), said))
        return f

    def command(self, timeout=None):
        return self.commands.pop(0)


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, line):
        self.said.append(line)


def _stub(answers):
    run = types.SimpleNamespace(listener=FakeListener(answers), real_time=False, said=[])
    run.say = run.said.append
    return run


def test_ask_voice_yes_and_no():
    assert Run.ask_voice(_stub(["yes"]), "destination") is True
    assert Run.ask_voice(_stub(["no"]), "destination") is False


def test_ask_voice_asks_again_until_understood():
    run = _stub(["the library", None, "yeah"])
    assert Run.ask_voice(run, "destination") is True
    assert run.said == ["Please say yes or no."] * 2


def test_ask_voice_silence_is_a_no():
    run = _stub([None] * VOICE_TRIES)
    assert Run.ask_voice(run, "lift") is False
    assert len(run.said) == VOICE_TRIES - 1


def test_listen_for_stops_asks_again_on_an_unknown_place():
    args = types.SimpleNamespace(wake="hey dog", nlu=False, command=None)
    speaker, guessed = FakeSpeaker(), []
    listener = FakeListener(commands=["take me to narnia", "take me to room 201"])
    stops = listen_for_stops(BuildingRouter(), speaker, listener, args, guessed)
    assert stops == [("room 201", None)]
    assert args.command == "take me to room 201" and guessed == []
    assert any("try again" in s for s in speaker.said)
    assert speaker.said[-1] == "Taking you to room 201."


def test_listen_for_stops_leaves_a_near_miss_to_be_confirmed():
    args = types.SimpleNamespace(wake="hey dog", nlu=False, command=None)
    speaker, guessed = FakeSpeaker(), []
    stops = listen_for_stops(BuildingRouter(), speaker,
                             FakeListener(commands=["take me to the libery"]), args, guessed)
    assert stops == [("library", None)]
    assert guessed and guessed[0][1] == "library"          # asked before the dog moves
    assert not any(s.startswith("Taking you") for s in speaker.said)
