"""The dog's ears: a microphone, the wake phrase, and whisper.cpp.

    "Hey dog"                         -> "Yes?"  -> "take me to room 201"
    "Hey dog, take me to room 201"    -> the command at once

The command then goes through the same resolve_stops() as a typed one, so
everything the rules do with a near miss ("libery") -- and the yes/no question
before the dog moves -- is unchanged; only where the text comes from is new.

whisper.cpp runs on the CPU (pywhispercpp), never the GPU: on the J4012 the
GPU and most of the shared memory are VAMOS's, and base.en needs ~0.4 GB.

The wake phrase is spotted by transcribing each utterance and matching its
first words (text.split_wake), which needs no wake-word model of its own and
works for any phrase. The price is a whisper call for every stretch of speech
the microphone picks up while the dog waits to be called; a trained
openWakeWord model is the cheaper path if that proves too much for the
J4012's cores. Measure before deciding (BrainTicks in the brain's log).
"""
import concurrent.futures
import queue
import time

import numpy as np

from cyberdog import paths
from cyberdog.voice import text as T

# tiny.en: 0.84 s a command on the J4012's CPU, against base.en's 2.0 s, and
# the same words on the test sentences. base.en hears accents better; judge
# it again with real voices on the real microphone (--stt-model base.en).
STT_MODEL = "tiny.en"
THREADS = 2              # whisper.cpp threads: the brain keeps the rest of its cores
AUDIO_CTX_MAX = 1500     # whisper's full 30 s window
AUDIO_CTX_PAD = 64       # encoder frames past the clip's end
FOLLOW_S = 6.0           # after "Yes?", how long to wait for the command
ANSWER_S = 10.0          # how long one yes/no question waits for an answer


class Mic:
    """16 kHz mono int16 frames from the default (or given) input device.

    Frames that arrive while the dog is talking are dropped at the source.
    """

    def __init__(self, speaker=None, device=None):
        import sounddevice as sd
        self.speaker = speaker
        self.q = queue.Queue(maxsize=int(30 * T.RATE / T.FRAME))   # 30 s of audio
        self.stream = sd.InputStream(samplerate=T.RATE, channels=1, dtype="int16",
                                     blocksize=T.FRAME, device=device, callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, when, status):
        if self.speaker is not None and self.speaker.busy():
            return
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def read(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def close(self):
        self.stream.stop()
        self.stream.close()


class Stt:
    """whisper.cpp, English only, on THREADS CPU threads."""

    def __init__(self, model=STT_MODEL, threads=THREADS):
        from pywhispercpp.model import Model
        paths.VOICE_MODELS.mkdir(parents=True, exist_ok=True)
        # The model file is downloaded into models/voice the first time.
        self.model = Model(model, models_dir=str(paths.VOICE_MODELS), n_threads=threads,
                           redirect_whispercpp_logs_to=None, language="en",
                           print_progress=False, print_realtime=False, no_context=True)

    def __call__(self, audio):
        """int16 audio at 16 kHz -> its text, cleaned of silence markers.

        whisper pads every clip to 30 s and encodes all of it; audio_ctx cuts
        the encoder to the clip's own length (50 per second, plus a margin).
        On the J4012's CPU, 2 threads, a 3 s command: tiny.en 5.5 s -> 0.84 s,
        base.en 13.7 s -> 2.0 s, the same words (scripts/hil/voice_check.py).
        """
        f = audio.astype(np.float32) / 32768.0
        ctx = min(AUDIO_CTX_MAX, int(len(f) / T.RATE * 50) + AUDIO_CTX_PAD)
        segments = self.model.transcribe(f, audio_ctx=ctx)
        return T.clean(" ".join(s.text for s in segments))


class Listener:
    def __init__(self, speaker, wake="hey dog", model=STT_MODEL, device=None,
                 threads=THREADS, mic=None, stt=None):
        self.speaker = speaker
        self.wake = wake
        self.stt = stt or Stt(model, threads)
        self.mic = mic or Mic(speaker, device)
        self.pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="listener")

    def utterance(self, timeout=None):
        """The next thing said, as int16 audio, or None after `timeout` s."""
        seg = T.Segmenter()
        end = None if timeout is None else time.monotonic() + timeout
        while end is None or time.monotonic() < end or seg.speech is not None:
            frame = self.mic.read(0.2)
            if frame is None:
                continue
            out = seg.feed(frame)
            if out is not None:
                return out
        return None

    def hear(self, timeout=None):
        """The next thing said, as text ("" if whisper made nothing of it),
        or None after `timeout` s of nobody speaking."""
        audio = self.utterance(timeout)
        if audio is None:
            return None
        t0 = time.monotonic()
        said = self.stt(audio)
        print(f'    [heard] "{said}" ({len(audio) / T.RATE:.1f} s of speech, '
              f"{time.monotonic() - t0:.2f} s to read)")
        return said

    def _quiet_then_flush(self):
        """Let the dog finish talking, then forget what the mic held meanwhile."""
        self.speaker.wait()
        self.mic.flush()

    def command(self, timeout=None):
        """Wait to be called, and return the command that follows ("" never).

        None only if `timeout` s pass with no command at all.
        """
        end = None if timeout is None else time.monotonic() + timeout
        while end is None or time.monotonic() < end:
            self._quiet_then_flush()
            said = self.hear(None if end is None else max(0.1, end - time.monotonic()))
            if said is None:
                continue
            called, rest = T.split_wake(said, self.wake)
            if not called:
                continue
            if rest:
                return rest
            self.speaker.say("Yes?")
            self._quiet_then_flush()
            said = self.hear(FOLLOW_S)
            if said:
                return said
            self.speaker.say("I did not catch that.")
        return None

    def yes_no(self, timeout=ANSWER_S):
        """(True / False / None, what was heard) for the next answer."""
        self._quiet_then_flush()
        said = self.hear(timeout)
        return (None if not said else T.parse_yes_no(said)), said

    def yes_no_async(self, timeout=ANSWER_S):
        """yes_no on the listener's own thread: a Future, so the caller can
        keep the dog's loop ticking while the person answers."""
        return self.pool.submit(self.yes_no, timeout)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.mic.close()
