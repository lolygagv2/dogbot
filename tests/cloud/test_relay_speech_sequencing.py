#!/usr/bin/env python3
"""Relay voice clips vs. treats: no cut-off, no overlap (2026-11-12).

Seen live 2026-11-11: a second TTS clip stopped the first 6 s into a 15 s sentence,
and dispense_treat started the carousel while a clip was still speaking.

Checks (audio, PTT queue, relay and dispenser are fakes; no sound, no motor):
  - a tts clip arriving while another relay clip is speaking queues behind it;
  - a tts clip still stops music / other audio when no relay clip is speaking;
  - explicit interrupt=true still cuts speech, interrupt=false never stops anything;
  - dispense_treat waits for the speech to finish, then dispenses;
  - the wait is bounded and does not wait for music.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_relay_speech_sequencing.py
Standalone (no pytest).
"""
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='speechtest_')

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakePTT:
    def __init__(self):
        self.playing = False
        self._play_queue = []
        self.cleared = 0
        self.played = []

    def is_playing(self):
        return self.playing

    def clear_queue(self):
        self.cleared += 1
        self._play_queue.clear()
        return 0

    def play_audio(self, data, fmt):
        self.played.append(data)
        return {'success': True}


class FakeAudio:
    def __init__(self):
        self.busy = False
        self.stops = 0

    def is_busy(self):
        return self.busy

    def stop(self):
        self.stops += 1
        self.busy = False


class FakeDispenser:
    treats_remaining = 5
    treats_dispensed_today = 1

    def __init__(self, ptt):
        self.ptt = ptt
        self.speaking_at_dispense = None

    def dispense_treat(self, **kw):
        self.speaking_at_dispense = self.ptt.is_playing()
        return True


def main():
    ptt, audio = FakePTT(), FakeAudio()
    disp = FakeDispenser(ptt)
    # Stand-ins for the real modules, which open the sound card on import.
    for name, attrs in (
            ('services.media.push_to_talk', {'get_push_to_talk_service': lambda: ptt, 'UTTERANCE_MAX_S': 30}),
            ('services.media.usb_audio', {'get_usb_audio_service': lambda: audio}),
            ('services.reward.dispenser', {'get_dispenser_service': lambda: disp})):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    import services.cloud.tool_dispatcher as td
    d = td.ToolDispatcher()
    d._relay = types.SimpleNamespace(download_sync=lambda url: b'mp3bytes')
    cancel = threading.Event()
    clip = {'url': 'https://x/a.mp3', 'kind': 'tts'}

    # --- second clip while the first is speaking
    ptt.playing, audio.busy = True, True
    res = d._tool_play_sound(dict(clip), cancel)
    check(res.get('ok') and audio.stops == 0 and ptt.cleared == 0 and len(ptt.played) == 1,
          'tts clip during relay speech is queued, nothing is stopped')

    # --- clip queued but not started yet counts as speech too
    ptt.playing, ptt._play_queue = False, [(b'x', 'mp3')]
    d._tool_play_sound(dict(clip), cancel)
    check(audio.stops == 0 and ptt.cleared == 0, 'tts clip behind a queued clip does not clear the queue')
    ptt._play_queue = []

    # --- music playing, no relay speech: tts still takes over
    ptt.playing, audio.busy = False, True
    d._tool_play_sound(dict(clip), cancel)
    check(audio.stops == 1 and ptt.cleared == 1, 'tts clip still stops music when no relay clip is speaking')

    # --- explicit flags
    ptt.playing, audio.busy = True, True
    d._tool_play_sound(dict(clip, interrupt=True), cancel)
    check(audio.stops == 2 and ptt.cleared == 2, 'interrupt=true still cuts speech')
    ptt.playing, audio.busy = False, True
    d._tool_play_sound(dict(clip, interrupt=False), cancel)
    check(audio.stops == 2, 'interrupt=false never stops audio')

    # --- treat waits for the sentence
    ptt.playing, audio.busy = True, True
    threading.Timer(0.6, lambda: setattr(ptt, 'playing', False)).start()
    t0 = time.time()
    res = td.ToolDispatcher._tool_dispense_treat({'source': 'agent'}, cancel)
    waited = time.time() - t0
    check(res.get('ok') and 0.5 <= waited < 1.5 and disp.speaking_at_dispense is False,
          f'dispense_treat waits for speech to end, then dispenses (waited {waited:.1f}s)')

    # --- bounded
    td.SPEECH_WAIT_MAX_S = 0.5
    ptt.playing = True
    t0 = time.time()
    res = td.ToolDispatcher._tool_dispense_treat({'source': 'agent'}, cancel)
    check(res.get('ok') and 0.4 <= time.time() - t0 < 1.2, 'the wait is bounded: treat still dispensed after the cap')

    # --- music alone is not waited for
    ptt.playing, audio.busy = False, True
    t0 = time.time()
    td.ToolDispatcher._tool_dispense_treat({'source': 'agent'}, cancel)
    check(time.time() - t0 < 0.2, 'music alone does not delay a treat')

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
