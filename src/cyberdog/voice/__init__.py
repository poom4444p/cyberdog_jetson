"""Voice in and out, on the dog itself: the person talks, the dog answers.

    text.py     the decisions, pure: wake phrase, yes/no, where speech ends
    listen.py   microphone -> utterance -> whisper.cpp (CPU) -> text
    speak.py    Run.say() -> Piper (CPU), on its own thread

Both models run on the CPU and together take ~0.5 GB, so they sit beside
VAMOS on the J4012, where the Gemma parser (4.7 GB) does not. Commands are
still read by the rules in cyberdog.language; this package only replaces the
keyboard and the terminal with a microphone and a speaker.

Needs the voice extra: pip install -e ".[voice]". run_building --voice.
"""
from cyberdog.voice.speak import Speaker
from cyberdog.voice.text import parse_yes_no, split_wake


def Listener(*args, **kwargs):
    """listen.Listener, imported on first use: it opens the microphone."""
    from cyberdog.voice.listen import Listener as _Listener
    return _Listener(*args, **kwargs)


__all__ = ["Speaker", "Listener", "parse_yes_no", "split_wake"]
