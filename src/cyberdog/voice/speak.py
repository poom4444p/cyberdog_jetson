"""The dog's voice: Run.say() made audible, without holding up the loop.

Run.say() is called from inside the 20 Hz control loop ("Something ahead.
Stepping around it to your left."), and a sentence takes a second or two to
speak. So speaking happens on a thread of its own: say() only queues the line
and returns. If lines arrive faster than they can be spoken, the oldest
waiting ones are dropped -- a warning about the corridor two seconds ago is
worth less than the one about now -- and the same line said again within
REPEAT_S is not repeated.

Backends, best first: Piper (a small neural voice, onnxruntime on the CPU, so
the GPU stays VAMOS's), macOS `say` while developing on the Mac, and print
alone (the twin's batch runs).

The Listener reads busy() and does not listen while the dog is talking or
just after, or the dog would hear its own "Did you mean the library?" as the
person's answer.
"""
import collections
import shutil
import subprocess
import sys
import threading
import time

from cyberdog import paths

VOICE = "en_US-lessac-medium"
MAX_PENDING = 2          # lines waiting to be spoken; older ones are dropped
REPEAT_S = 6.0           # the same line again within this is not spoken again
ECHO_S = 0.4             # still "busy" this long after the last word, for the room's echo


class Speaker:
    def __init__(self, backend="auto", voice=VOICE, device=None):
        self.device = device
        self.backend, self.voice = self._pick(backend, voice)
        self.lines = collections.deque()
        self.cv = threading.Condition()
        self.speaking = False
        self.quiet_from = 0.0
        self.last = {}               # line -> when it was last queued
        self.dropped = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self.thread.start()

    @staticmethod
    def _pick(backend, voice):
        if backend in ("auto", "piper"):
            model = paths.VOICE_MODELS / f"{voice}.onnx"
            try:
                from piper import PiperVoice
                if model.exists():
                    return "piper", PiperVoice.load(str(model))
                if backend == "piper":
                    raise SystemExit(f"no Piper voice at {model}: python -m piper.download_voices "
                                     f"{voice} --download-dir {paths.VOICE_MODELS}")
            except ImportError:
                if backend == "piper":
                    raise SystemExit('Piper is not installed: pip install -e ".[voice]"')
        if backend in ("auto", "say") and sys.platform == "darwin" and shutil.which("say"):
            return "say", None
        return "print", None

    def say(self, line):
        """Queue `line` to be spoken. Returns at once."""
        now = time.monotonic()
        with self.cv:
            if now - self.last.get(line, -1e9) < REPEAT_S:
                return
            self.last[line] = now
            self.lines.append(line)
            while len(self.lines) > MAX_PENDING:
                self.lines.popleft()
                self.dropped += 1
            self.cv.notify_all()

    def busy(self):
        """Talking, about to, or the echo of it still in the room."""
        with self.cv:
            return (self.speaking or bool(self.lines)
                    or time.monotonic() < self.quiet_from)

    def wait(self, timeout=None):
        """Block until everything queued has been said (or `timeout` s)."""
        end = None if timeout is None else time.monotonic() + timeout
        while self.busy():
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.05)
        return True

    def close(self, timeout=10.0):
        """Say what is still queued (up to `timeout` s), then stop."""
        self.wait(timeout)
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def _run(self):
        while True:
            with self.cv:
                while not self.lines and not self.closed:
                    self.cv.wait()
                if self.closed and not self.lines:
                    return
                line = self.lines.popleft()
                self.speaking = True
            try:
                self._speak(line)
            except Exception as e:           # a voice that fails must not stop the dog
                print(f"    [voice] speaker failed ({type(e).__name__}: {e}); printing only")
                self.backend = "print"
            finally:
                with self.cv:
                    self.speaking = False
                    self.quiet_from = time.monotonic() + ECHO_S

    def _speak(self, line):
        if self.backend == "piper":
            import numpy as np
            import sounddevice as sd
            chunks = list(self.voice.synthesize(line))
            if not chunks:
                return
            audio = np.concatenate([c.audio_float_array for c in chunks])
            sd.play(audio, chunks[0].sample_rate, device=self.device)
            sd.wait()
        elif self.backend == "say":
            subprocess.run(["say", line], check=False)
