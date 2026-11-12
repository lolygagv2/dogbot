#!/usr/bin/env python3
"""Regression: the bark-suppression monitor must never outlive the sound it guards.

Live failure (treatbot1, 2026-11-12): voice clips and music share the single
`pygame.mixer.music` stream, and pygame keeps reporting `get_busy() == True` while
that stream is PAUSED. _start_bark_suppression_monitor() used get_busy() as its only
"is the speaker making sound" test, so a paused track left bark detection suppressed
indefinitely: Silent Guardian went deaf while the speaker sat silent (the sink monitor
measured RMS 0) and /audio/status still read playing:false, which is why it hid.
Symptom in the journal: "Bark detection suppressed for 0.4s (speaker echo)" every
200 ms, forever, with nothing audible.

Run: /home/morgan/dogbot/env_new/bin/python tests/media/test_usb_audio_bark_suppression.py
Standalone (no pytest). pygame is stubbed before import on purpose — importing
usb_audio for real runs pygame.mixer.init() at module scope and would seize the
robot's USB audio device.
"""
import logging
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.CRITICAL)


class FakeMusic:
    """Stand-in for pygame.mixer.music with a scriptable get_busy()."""

    def __init__(self):
        self.busy = False
        self.paused = False

    def get_busy(self):
        return self.busy

    def load(self, path):
        pass

    def play(self, loops=0):
        self.busy = True
        self.paused = False

    def pause(self):
        self.paused = True          # the crux: get_busy() stays True

    def unpause(self):
        self.paused = False

    def stop(self):
        self.busy = False
        self.paused = False


FAKE_MUSIC = FakeMusic()
_fake_pygame = types.ModuleType("pygame")
_fake_pygame.mixer = types.SimpleNamespace(
    music=FAKE_MUSIC,
    init=lambda **kw: None,
    quit=lambda: None,
    get_init=lambda: True,
)
sys.modules["pygame"] = _fake_pygame

from services.media.usb_audio import USBAudioService  # noqa: E402


class Recorder:
    """Captures suppress_detection() calls made from the monitor thread."""

    def __init__(self):
        self.calls = []

    def suppress_detection(self, seconds):
        self.calls.append(seconds)


def make_service():
    """A USBAudioService shell, skipping __init__ (which inits real hardware)."""
    svc = object.__new__(USBAudioService)
    svc.logger = logging.getLogger("test.usb_audio")
    svc._paused = False
    svc._music_playing = False
    rec = Recorder()
    fake_bd = types.ModuleType("services.perception.bark_detector")
    fake_bd.peek_bark_detector_service = lambda: rec
    sys.modules["services.perception.bark_detector"] = fake_bd
    return svc, rec


def monitor_finished(timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(t.name == "BarkSuppressMonitor" and t.is_alive()
                   for t in threading.enumerate()):
            return True
        time.sleep(0.05)
    return False


def test_paused_stream_releases_suppression():
    """THE REGRESSION: a paused track must not pin bark detection off forever."""
    svc, _ = make_service()
    FAKE_MUSIC.busy = True          # pygame still claims "busy"...
    svc._paused = True              # ...but it is paused, so the speaker is silent

    svc._start_bark_suppression_monitor()

    assert monitor_finished(), (
        "monitor never exited with a paused stream — bark detection would stay "
        "suppressed indefinitely (the 2026-11-12 tb1 failure)")


def test_voice_clip_still_suppresses_while_really_playing():
    """Guards the fix itself: _music_playing is False for voice clips.

    A tempting but WRONG fix is `if not self._music_playing: break`. That flag is set
    only for playlist music (usb_audio.py:334) while the monitor is started only for
    non-music (usb_audio.py:357), so it would abort instantly on every voice clip and
    silently destroy the echo suppression this monitor exists to provide.
    """
    svc, rec = make_service()
    FAKE_MUSIC.busy = True
    svc._paused = False
    svc._music_playing = False      # a voice clip, not music
    svc.MAX_BARK_SUPPRESS_S = 1.0   # keep the test quick

    svc._start_bark_suppression_monitor()
    time.sleep(0.5)
    ongoing = len(rec.calls)

    assert ongoing >= 2, f"voice clip got no ongoing suppression: {rec.calls}"
    assert monitor_finished()


def test_monitor_never_outlives_its_ceiling():
    """A permanently-busy stream must still release via MAX_BARK_SUPPRESS_S."""
    svc, _ = make_service()
    FAKE_MUSIC.busy = True          # never goes false
    svc._paused = False
    svc.MAX_BARK_SUPPRESS_S = 1.0

    started = time.time()
    svc._start_bark_suppression_monitor()

    assert monitor_finished(timeout=8.0), "monitor ignored its ceiling"
    elapsed = time.time() - started
    assert elapsed < 5.0, f"ceiling took too long to fire: {elapsed:.1f}s"


def test_finished_stream_exits_with_reverb_tail():
    """Normal path: the clip ends, monitor exits and still leaves the 0.5s tail."""
    svc, rec = make_service()
    FAKE_MUSIC.busy = False
    svc._paused = False

    svc._start_bark_suppression_monitor()

    assert monitor_finished()
    assert rec.calls[0] == 1.0, f"missing the initial 1s window: {rec.calls}"
    assert rec.calls[-1] == 0.5, f"missing the reverb/AGC tail: {rec.calls}"


def test_stop_clears_the_paused_flag():
    """stop() must clear _paused, else the next clip's monitor exits at once."""
    svc, _ = make_service()
    FAKE_MUSIC.busy = True
    svc.initialized = True
    svc._playlist_track = "songs/x.mp3"
    svc._is_looping = False
    svc._loading = False
    svc._send_audio_event = lambda *a, **k: None

    svc.pause()
    assert svc._paused is True, "pause() did not record the paused state"

    svc.stop()
    assert svc._paused is False, "stop() left _paused set; next voice clip would not suppress"


def test_resume_clears_the_paused_flag():
    """resume() must clear _paused so suppression works again after a resume."""
    svc, _ = make_service()
    FAKE_MUSIC.busy = True
    svc.initialized = True
    svc._playlist_track = "songs/x.mp3"
    svc._send_audio_event = lambda *a, **k: None

    svc.pause()
    svc.resume()
    assert svc._paused is False, "resume() left _paused set"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
