#!/usr/bin/env python3
"""Replayed relay events must reach the speaker (2026-11-18).

The robot has no replay path of its own. When the relay replays events, the
agent speech they trigger arrives as an ordinary command frame
(play_sound {url, kind:"tts", source:"replay"}). Checks that nothing on the
robot drops it because of `source`: the relay client hands it to the
dispatcher, the dispatcher plays it and answers command_result ok, exactly as
for source "agent". The only robot-side gate is the timestamp age check
(30 s), which is the same for every source.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_replay_source_not_filtered.py
"""
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='replaytest_')

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


played = []


class FakePTT:
    _play_queue = []
    def is_playing(self): return False
    def clear_queue(self): return 0
    def play_audio(self, data, fmt):
        played.append((len(data), fmt))
        return {'success': True}


class FakeAudio:
    def is_busy(self): return False
    def stop(self): pass


for name, attr, obj in (('services.media.push_to_talk', 'get_push_to_talk_service', FakePTT()),
                        ('services.media.usb_audio', 'get_usb_audio_service', FakeAudio())):
    m = types.ModuleType(name)
    setattr(m, attr, lambda obj=obj: obj)
    sys.modules[name] = m

import services.cloud.tool_dispatcher as td  # noqa: E402
td.log_tool_call = lambda *a, **k: None


class FakeRelay:
    def download_sync(self, url, timeout=20.0):
        return b'ID3' + b'\0' * 2000


d = td.ToolDispatcher()
d._relay = FakeRelay()
results = {}
for source in ('agent', 'replay'):
    played.clear()
    f = d.call('play_sound', {'url': f'https://relay.example/tts/{source}.mp3', 'kind': 'tts', 'source': source},
               f'c-{source}', wait_s=5)
    results[source] = (f['success'], f['result'].get('played'), f['result'].get('kind'), len(played))
check(results['replay'] == results['agent'] == (True, 'url', 'tts', 1),
      f"play_sound kind=tts source=replay plays exactly like source=agent ({results})")

# Relay client: a replay-sourced frame is routed to the dispatcher like any other.
from services.cloud.relay_client import RelayClient  # noqa: E402
sent, seen = [], []
rc = RelayClient.__new__(RelayClient)
rc.logger = __import__('logging').getLogger('t')
rc.config = types.SimpleNamespace(device_id='tb')
rc.send_message = lambda m: sent.append(m)
saved = td._instance
td._instance = d
try:
    data = {'type': 'command', 'command': 'play_sound', 'command_id': 'c-r2', 'source': 'replay',
            'url': 'https://relay.example/tts/x.mp3', 'kind': 'tts'}
    ok = rc._dispatch_v2_command('play_sound', data, {k: v for k, v in data.items() if k not in ('type', 'command')})
    t_end = time.time() + 3
    while not sent and time.time() < t_end:
        time.sleep(0.02)
    check(ok and sent and sent[0]['success'] is True and sent[0]['command_id'] == 'c-r2',
          f"relay frame with source=replay -> dispatched, command_result ok ({sent[:1] and sent[0].get('error')})")
finally:
    td._instance = saved

# The age gate is source-blind: same limits whatever the source.
import inspect  # noqa: E402
src = inspect.getsource(RelayClient._handle_command)
gate = src[:src.index('_dispatch_v2_command')]
check('source' not in gate.replace('data source', ''), "nothing before dispatch in _handle_command looks at `source`")
check(RelayClient.COMMAND_MAX_AGE_SEC == 30,
      "only gate is timestamp age (30 s): a replayed command must carry the send time, not the original event time")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
