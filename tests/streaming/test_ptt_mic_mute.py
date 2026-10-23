#!/usr/bin/env python3
"""PTT playback must keep the WebRTC mic muted for the WHOLE clip — 2026-10-23.

Morgan: push-to-talk clip plays, the live robot mic streams it to the app, the
phone plays that, the robot hears the phone... "endless loop". The mic track
was muted for only 0.2 s of a 5 s clip. Now: muted at playback start (30 s
cap), released with a 0.7 s tail when the speaker is actually done, and the
stale mic frames captured during the mute are dropped on unmute.

No audio hardware: the track is built with __new__ (sounddevice untouched) and
the PTT service's speaker/ffmpeg paths are stubbed.

Run: /home/morgan/dogbot/env_new/bin/python tests/streaming/test_ptt_mic_mute.py
"""
import logging
import queue
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from services.streaming import audio_track as at  # noqa: E402
from services.media import push_to_talk as ptt  # noqa: E402


def _bare_track():
    t = at.WIMZAudioTrack.__new__(at.WIMZAudioTrack)
    t.logger = logging.getLogger("t")
    t._muted = False
    t._ptt_muted = False
    t._ptt_mute_until = 0
    t._audio_queue = queue.Queue(maxsize=50)
    t._stream = object()   # "capturing"
    return t


def test_release_shortens_mute_to_tail_and_unmute_drains_queue():
    t = _bare_track()
    t.mute_for_ptt(30.0)
    assert t.is_muted()
    for _ in range(10):
        t._audio_queue.put_nowait(np.ones((at.SAMPLES_PER_FRAME, 1), dtype='int16'))
    t.release_ptt_mute(0.05)
    assert t.is_muted(), "still muted during the tail"
    time.sleep(0.08)
    assert not t.is_muted(), "unmuted after the tail"
    assert t._audio_queue.qsize() == 0, "stale clip audio must be dropped on unmute"


def test_release_never_extends_an_earlier_deadline():
    t = _bare_track()
    t.mute_for_ptt(0.01)
    t.release_ptt_mute(5.0)
    time.sleep(0.03)
    assert not t.is_muted()


def test_release_without_mute_is_noop():
    t = _bare_track()
    t.release_ptt_mute(1.0)
    assert not t.is_muted()


def test_play_single_mutes_for_whole_clip_then_releases():
    t = _bare_track()
    at.set_audio_track(t)
    svc = ptt.PushToTalkService.__new__(ptt.PushToTalkService)
    svc.logger = logging.getLogger("ptt")
    svc._on_play_done = None
    ptt.TEMP_DIR.mkdir(parents=True, exist_ok=True)  # __new__ skipped __init__

    busy = {"n": 6}  # ~0.6 s of "playback" at the service's 0.1 s poll
    class FakeUSB:
        def is_busy(self):
            busy["n"] -= 1
            return busy["n"] > 0
        def stop(self): pass
        def play_file(self, p): return {"success": True}

    fake_mod = types.ModuleType("services.media.usb_audio")
    fake_mod.get_usb_audio_service = lambda: FakeUSB()
    sys.modules["services.media.usb_audio"] = fake_mod
    try:
        ptt.PTT_MUTE_TAIL_S = 0.05
        start = time.time()
        svc._play_single(b"RIFF" + b"\0" * 64, "wav")
        played_for = time.time() - start
        assert played_for >= 0.4, played_for
        # Mute covered the whole playback (deadline still in the future at the
        # moment of release, so release pulled it down to the tail).
        assert t._ptt_muted and t._ptt_mute_until <= time.time() + 0.06
        time.sleep(0.08)
        assert not t.is_muted()
    finally:
        del sys.modules["services.media.usb_audio"]
        at.set_audio_track(None)


if __name__ == "__main__":
    fails = 0
    for name in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[name]()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
