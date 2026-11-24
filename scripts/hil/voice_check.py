"""Can the dog hear and speak on this machine, and what does it cost?

    python scripts/hil/voice_check.py                 # devices, speed, memory
    python scripts/hil/voice_check.py --play          # ...and say a line out loud
    python scripts/hil/voice_check.py --listen 5      # ...and transcribe 5 s of the mic

Run it on the J4012 in go2env, with VAMOS up (onboard.sh vamos): the numbers
that matter are the ones beside it. No microphone is needed for the timing:
the test sentences are spoken by Piper itself and read back by whisper.cpp.

Prints, per whisper model: seconds to read each sentence (the wait between
the person finishing and the dog answering) and what it heard; Piper's time
to speak a line; and this process's resident memory.
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

from cyberdog import paths
from cyberdog.voice import text as T

LINES = ["Hey dog, take me to room two oh one.",
         "Hey dog, I need to pee on the second floor.",
         "Yes, please."]


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20   # Linux: kB, macOS: bytes


def to16k(audio, rate):
    from scipy.signal import resample_poly
    a = resample_poly(audio, T.RATE, rate)
    return (np.clip(a, -1, 1) * 30000).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["tiny.en", "base.en"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--play", action="store_true", help="say a line on the speaker")
    ap.add_argument("--listen", type=float, default=0, metavar="S",
                    help="record S seconds from the mic and transcribe them")
    ap.add_argument("--mic", default=None)
    args = ap.parse_args()

    try:
        import sounddevice as sd
        print("audio devices:")
        print(sd.query_devices())
    except OSError as e:                 # no libportaudio2: timing still works
        sd = None
        print(f"audio: none usable ({e}); timing only, --play/--listen skipped")
    print(f"cpus: {os.cpu_count()}, whisper threads: {args.threads}")

    from piper import PiperVoice
    voice = PiperVoice.load(str(paths.VOICE_MODELS / "en_US-lessac-medium.onnx"))
    clips = []
    for line in LINES:
        t0 = time.monotonic()
        chunks = list(voice.synthesize(line))
        dt = time.monotonic() - t0
        audio = np.concatenate([c.audio_float_array for c in chunks])
        rate = chunks[0].sample_rate
        # framed as the Segmenter cuts speech: PRE_S before, SILENCE_S after
        pad = lambda sec: np.zeros(int(sec * T.RATE), np.int16)
        clips.append((line, np.concatenate([pad(T.PRE_S), to16k(audio, rate),
                                            pad(T.SILENCE_S)])))
        print(f"piper: {dt:.2f} s to speak {len(audio) / rate:.1f} s of {line!r}")
    print(f"memory after piper: {rss_mb():.0f} MB")

    from cyberdog.voice.listen import Stt
    for name in args.models:
        t0 = time.monotonic()
        stt = Stt(name, args.threads)
        print(f"\nwhisper {name}: loaded in {time.monotonic() - t0:.1f} s, "
              f"memory {rss_mb():.0f} MB")
        stt(clips[0][1])                                   # warm-up
        for line, clip in clips:
            t0 = time.monotonic()
            said = stt(clip)
            dt = time.monotonic() - t0
            print(f"  {dt:5.2f} s for {len(clip) / T.RATE:.1f} s: {said!r}"
                  f"  wake={T.split_wake(said)[0]} yes/no={T.parse_yes_no(said)}")

    if args.play and sd is not None:
        from cyberdog.voice import Speaker
        sp = Speaker("piper")
        sp.say("Hello. Say hey dog, and where you would like to go.")
        sp.close()
        print("\nplayed a line on the default speaker")

    if args.listen and sd is not None:
        mic = int(args.mic) if args.mic and args.mic.isdigit() else args.mic
        print(f"\nrecording {args.listen:.0f} s -- speak now")
        rec = sd.rec(int(args.listen * T.RATE), samplerate=T.RATE, channels=1,
                     dtype="int16", device=mic)
        sd.wait()
        rec = rec[:, 0]
        print(f"level: rms {T.rms(rec):.0f} (speech starts above {T.MIN_RMS:.0f})")
        t0 = time.monotonic()
        said = stt(rec)
        print(f"heard in {time.monotonic() - t0:.2f} s: {said!r}")


if __name__ == "__main__":
    main()
