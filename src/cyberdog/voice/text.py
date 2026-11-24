"""The voice loop's decisions, as pure functions: no microphone, no model.

What the microphone and whisper hand over is a sentence, and what the dog
needs from it is one of three small answers: was the dog called ("hey dog"),
is this a yes or a no, and -- for the recorder -- where one utterance ends.
They live here, apart from the audio, so they can be tested on text and on
synthetic levels, and so the J4012 and the Mac decide them the same way.

    split_wake(text, phrase)   -> (called?, the command said after the call)
    parse_yes_no(text)         -> True, False, or None when it is neither
    clean(text)                -> whisper's text without its silence markers
    Segmenter                  -> 16 kHz frames in, one utterance out
"""
import difflib
import re

import numpy as np

# whisper writes these for silence and noise; none of them is speech.
_MARKERS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
# ...and on a near-silent clip it tends to "hear" a sign-off. Only the whole
# utterance is dropped, never the words inside a real sentence.
_PHANTOMS = {"thank you", "thanks for watching", "you", "bye"}

_WORD = re.compile(r"[A-Za-z0-9']+")


_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


def clean(text):
    """whisper's text with [BLANK_AUDIO], (music) and the like taken out,
    and a sentence said twice in a row kept once: tiny.en repeats itself on
    a short clip ("Hey dog, room 201. Hey dog, room 201."), seen on the J4012."""
    text = _MARKERS.sub(" ", text or "")
    text = " ".join(text.split())
    if text.lower().strip(" .!?,") in _PHANTOMS:
        return ""
    kept = []
    for sentence in (m.strip() for m in _SENTENCE.findall(text)):
        key = sentence.lower().strip(" .!?,")
        if sentence and not (kept and kept[-1][0] == key):
            kept.append((key, sentence))
    return " ".join(sentence for _, sentence in kept)


# The call is short and said to a dog, so whisper spells it many ways:
# "Hey, dog." "Hey Doug" "Hey dawg" "Hi dog". The first word is matched from a
# list (difflib scores "hi" against "hey" too low to trust), the rest by
# spelling, the way place_match matches a near miss.
_HEY = {"hey", "hay", "hei", "heh", "hi", "hej", "ey",
        "they"}        # tiny.en on the J4012: "They dog, I need to pee..."
_NEAR = 0.6
_LEAD = 3          # words of noise whisper may put before the call ("okay, ...")


def _close(word, target):
    if target == "hey":
        return word in _HEY
    if difflib.SequenceMatcher(None, word, target).ratio() >= _NEAR:
        return True
    # Same ends, a vowel's worth between: "dawg" for "dog" (0.57 by spelling),
    # where "door" (0.57 too) is not.
    return (len(word) > 2 and word[0] == target[0] and word[-1] == target[-1]
            and abs(len(word) - len(target)) <= 1)


def split_wake(text, phrase="hey dog"):
    """(True, the rest) if `text` starts with the wake phrase, else (False, "").

    "Hey dog, take me to room 201." -> (True, "take me to room 201")
    "Hey Doug."                     -> (True, "")
    "the dog is over there"         -> (False, "")
    """
    words = _WORD.findall(clean(text))
    want = phrase.lower().split()
    low = [w.lower() for w in words]
    for i in range(min(_LEAD + 1, len(low) - len(want) + 1)):
        if all(_close(low[i + k], want[k]) for k in range(len(want))):
            return True, " ".join(words[i + len(want):])
    return False, ""


_YES = {"yes", "yeah", "yep", "yup", "yah", "ya", "sure", "correct", "right",
        "ok", "okay", "affirmative", "go", "continue", "please"}
# "wait" is neither: at the lift it means "not yet", and a no there ends the trip.
_NO = {"no", "nope", "nah", "stop", "wrong", "cancel", "don't", "dont", "not",
       "negative"}


def parse_yes_no(text):
    """True for a yes, False for a no, None for neither -- or for both.

    "Yes, please." -> True; "No, the library." -> False; "Yes no" -> None.
    A yes and a no in one answer is a question this cannot settle, so None,
    and the caller asks again.
    """
    words = {w.lower() for w in _WORD.findall(clean(text))}
    yes, no = bool(words & _YES), bool(words & _NO)
    if yes == no:
        return None
    return yes


# --- one utterance out of a stream of frames --------------------------------
RATE = 16000              # whisper's rate, and the microphone's
FRAME = 1280              # 80 ms a frame
MIN_RMS = 250.0           # int16 level below which nothing counts as speech
START_RATIO = 3.0         # speech starts this far above the room's noise...
END_RATIO = 2.0           # ...and is over once it falls below this
SILENCE_S = 0.8           # this long quiet ends an utterance
MAX_S = 8.0               # nobody gives a destination for longer
MIN_SPEECH_S = 0.25       # a click or a cough is shorter
PRE_S = 0.3               # kept from before the start, for the first syllable


def rms(frame):
    f = np.asarray(frame, dtype=np.float32)
    return float(np.sqrt(np.mean(f * f))) if f.size else 0.0


class Segmenter:
    """Frames in, one utterance (int16 array) out when it ends.

    An energy gate against a running estimate of the room's noise: plain on
    purpose. The J4012's cores are the brain's, and a corridor's noise moves
    slowly enough for a running floor to follow it.
    """

    def __init__(self, rate=RATE, frame=FRAME, min_rms=MIN_RMS):
        self.rate, self.frame, self.min_rms = rate, frame, min_rms
        self.floor = min_rms / START_RATIO
        self.pre = []
        self.speech = None
        self.quiet = 0
        self.loud = 0

    def _s(self, n_frames):
        return n_frames * self.frame / self.rate

    def feed(self, frame):
        """The utterance as int16 if this frame ended one, else None."""
        level = rms(frame)
        if self.speech is None:
            if level > max(self.min_rms, self.floor * START_RATIO):
                self.speech, self.quiet, self.loud = list(self.pre) + [frame], 0, 1
                self.pre = []
            else:
                self.floor = 0.95 * self.floor + 0.05 * level
                self.pre.append(frame)
                keep = max(1, int(PRE_S * self.rate / self.frame))
                self.pre = self.pre[-keep:]
            return None
        self.speech.append(frame)
        if level < max(self.min_rms, self.floor * END_RATIO):
            self.quiet += 1
        else:
            self.quiet, self.loud = 0, self.loud + 1
        if self._s(self.quiet) >= SILENCE_S or self._s(len(self.speech)) >= MAX_S:
            out, loud = self.speech, self.loud
            self.speech, self.quiet, self.loud = None, 0, 0
            if self._s(loud) < MIN_SPEECH_S:
                return None
            return np.concatenate(out).astype(np.int16)
        return None
