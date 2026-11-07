#!/usr/bin/env python3
"""Relay `listen` tool (2026-11-07): PTT capture with bark detection kept
running -> YAMNet top-5 + mean_db -> WAV uploaded -> reply shape.

Uses the real YAMNet model (ai/models/yamnet.tflite) on a synthetic clip, a
fake PTT (records a generated 1 kHz tone + noise WAV) and a fake relay.
Checks: seconds clamped to 15, capture_audio called with pause_bark=False,
labels is a 5-list of {label, confidence} sorted desc, mean_db is a dBFS
number, audio_url comes from upload_media_sync with audio/wav, capture
failure -> ok:false capture_failed, classifier helpers (wav reader /
dBFS), and that the real PTT.capture_audio accepts pause_bark.

Run: /home/morgan/dogbot/env_new/bin/python tests/audio/test_listen_tool.py
"""
import base64
import inspect
import io
import os
import sys
import tempfile
import threading
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='listentest_')

import numpy as np  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


def tone_wav(seconds=2.0, rate=16000, amp=0.1):
    t = np.arange(int(seconds * rate)) / rate
    x = amp * np.sin(2 * np.pi * 1000 * t) + 0.01 * np.random.randn(len(t))
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue(), x


# --- classifier helpers + real model
from ai.yamnet_classifier import YAMNetClassifier, read_wav_mono16k, mean_dbfs, get_yamnet  # noqa: E402
wav_bytes, x = tone_wav()
p = Path(tempfile.mkdtemp()) / 't.wav'; p.write_bytes(wav_bytes)
xr = read_wav_mono16k(str(p))
check(len(xr) == len(x) and abs(float(np.max(np.abs(xr - x.astype(np.float32)))) ) < 0.01, "wav reader returns float32 mono 16k")
db = mean_dbfs(xr)
check(-25 < db < -20, f"mean_dbfs of 0.1-amplitude tone ~ -23 dBFS ({db})")
check(mean_dbfs(np.zeros(100, np.float32)) <= -90, "silence -> <= -90 dBFS")
y = get_yamnet()
check(y.available(), "yamnet.tflite + class map present")
res = y.classify_wav(str(p), top_k=5)
labs = res['labels']
check(len(labs) == 5 and all(set(l) == {'label', 'confidence'} for l in labs)
      and all(labs[i]['confidence'] >= labs[i+1]['confidence'] for i in range(4)),
      f"top-5 labels sorted desc ({[l['label'] for l in labs]})")
check(any(k in ' '.join(l['label'] for l in labs).lower() for k in ('sine', 'tone', 'beep', 'whistle', 'alarm', 'tuning')),
      f"1 kHz tone tagged as a tone-like class ({labs[0]})")

# --- PTT signature
from services.media.push_to_talk import PushToTalkService  # noqa: E402
check('pause_bark' in inspect.signature(PushToTalkService.capture_audio).parameters, "PTT.capture_audio accepts pause_bark")

# --- tool via dispatcher with fakes
import services.cloud.tool_dispatcher as td  # noqa: E402
import services.media.push_to_talk as ptt_mod  # noqa: E402
td.log_tool_call = lambda *a, **k: None
calls = []
class FakePTT:
    def capture_audio(self, duration, max_duration=None, pause_bark=True):
        calls.append({'duration': duration, 'max': max_duration, 'pause_bark': pause_bark})
        return {'success': True, 'data': base64.b64encode(wav_bytes).decode(), 'format': 'wav'}
ptt_mod.get_push_to_talk_service = lambda: FakePTT()
class FakeRelay:
    def __init__(self): self.uploads = []
    def upload_media_sync(self, filepath, content_type='image/jpeg', timeout=20.0, filename=None):
        self.uploads.append((filepath, content_type, filename)); return f"https://relay/media/{filename}"
d = td.ToolDispatcher(); fr = FakeRelay(); d._relay = fr
check(d.has_tool('listen') and d._tools['listen'].timeout_s >= 30, "dispatcher registers 'listen' with >=30 s timeout")
out = d._tool_listen({'seconds': 40}, threading.Event())
check(calls and calls[0]['duration'] == 15.0 and calls[0]['pause_bark'] is False,
      f"seconds clamped to 15 and bark detector NOT paused ({calls})")
check(out['ok'] and len(out['labels']) == 5 and isinstance(out['mean_db'], float) and out['seconds'] == 15.0
      and out['bytes'] == len(wav_bytes), f"reply has labels/mean_db/seconds/bytes ({out.get('mean_db')})")
check(fr.uploads and fr.uploads[0][1] == 'audio/wav' and fr.uploads[0][2].endswith('.wav')
      and out['audio_url'] == f"https://relay/media/{fr.uploads[0][2]}", f"WAV uploaded as audio/wav, audio_url returned ({out['audio_url']})")
check(Path(out['filepath']).exists() and '/media/' in out['filepath'], f"WAV parked in media tree ({out['filepath']})")
class BadPTT:
    def capture_audio(self, *a, **k): return {'success': False, 'error': 'Microphone not available'}
ptt_mod.get_push_to_talk_service = lambda: BadPTT()
out = d._tool_listen({'seconds': 3}, threading.Event())
check(out['ok'] is False and out['error']['code'] == 'capture_failed', f"capture failure -> ok:false capture_failed ({out})")
check(d._tool_listen({'seconds': 'long'}, threading.Event())['error']['code'] == 'bad_args', "bad seconds -> bad_args")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
