#!/usr/bin/env python3
"""Relay `take_photo` command frame -> command_result within budget (2026-11-07).

Regression for command_id 4eba3833: the relay sends `timestamp` as an ISO
string and the staleness check did int - str, so the handler crashed before
the dispatcher and no command_result was ever sent. Checks: timestamp
parsing (ms/s/numeric string/ISO/garbage), a string-timestamp frame reaches
the dispatcher and gets exactly one command_result, a crashing handler still
yields a command_result, fast capture returns jpeg_b64 + photo_id, slow
capture returns ok:true + photo_id (pending) inside the budget and then
uploads <photo_id>.jpg and sends photo_ready.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_take_photo_command_frame.py
"""
import asyncio
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='phototest_')

import services.cloud.tool_dispatcher as td  # noqa: E402
from services.cloud.tool_dispatcher import ToolDispatcher  # noqa: E402
from services.cloud.relay_client import RelayClient, RelayConfig  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


td.log_tool_call = lambda *a, **k: None

# ---- timestamp parsing
P = RelayClient._parse_cmd_ts_ms
now_ms = int(time.time() * 1000)
check(abs(P(now_ms) - now_ms) < 2, "epoch ms int")
check(abs(P(now_ms / 1000.0) - now_ms) < 2000, "epoch s float")
check(abs(P(str(now_ms)) - now_ms) < 2, "numeric string ms")
check(P('2026-11-07T05:22:20.123Z') == 1794028940123, f"ISO Z -> ms ({P('2026-11-07T05:22:20.123Z')})")
check(P('2026-11-07T00:22:20-05:00') == 1794028940000, "ISO with offset")
check(P('yesterday') is None and P('') is None and P(None) is None and P(-5) is None, "garbage -> None (check skipped)")

# ---- string-timestamp command frame -> exactly one command_result
rc = RelayClient(RelayConfig(enabled=True, device_id='tb-test', device_secret='s'))
sent = []
rc.send_message = lambda m: sent.append(m)
disp = td.get_tool_dispatcher()
disp.register_tool('take_photo', lambda args, cancel: {'ok': True, 'jpeg_b64': 'AAAA', 'photo_id': 'p1'}, timeout_s=10.0)
from datetime import datetime, timezone
iso_now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
frame = {'type': 'command', 'command': 'take_photo', 'command_id': 'cmd-1',
         'timestamp': iso_now, 'data': {'with_hud': False}}
asyncio.run(rc._handle_message(frame) if hasattr(rc, '_handle_message') else rc._handle_command(frame))
for _ in range(50):
    if sent:
        break
    time.sleep(0.05)
res = [m for m in sent if m.get('type') == 'command_result']
check(len(res) == 1 and res[0]['command_id'] == 'cmd-1' and res[0]['ok'] is True
      and res[0]['result'].get('jpeg_b64') == 'AAAA',
      f"ISO-timestamp frame -> one command_result ok:true with jpeg_b64 ({res})")

# ---- stale frame with command_id -> command_result ok:false (not a bare ack)
sent.clear()
old = {'type': 'command', 'command': 'take_photo', 'command_id': 'cmd-old',
       'timestamp': now_ms - 120_000, 'data': {}}
asyncio.run(rc._handle_command(old))
check(len(sent) == 1 and sent[0]['type'] == 'command_result' and sent[0]['ok'] is False
      and sent[0]['error']['code'] == 'stale', f"stale command_id frame -> command_result stale ({sent})")

# ---- handler crash -> command_result exception (via the message router)
sent.clear()
rc._handle_message_sync = None
async def boom(data):
    raise RuntimeError('kaboom')
rc._message_handlers['command'] = boom
router = getattr(rc, '_handle_message', None) or getattr(rc, '_process_message', None) or getattr(rc, '_on_message', None)
if router is None:
    import inspect
    cands = [n for n, f in inspect.getmembers(rc, inspect.ismethod)
             if 'Handler error for' in (inspect.getsource(f) if True else '')]
    router = getattr(rc, cands[0]) if cands else None
if router is not None:
    r = router({'type': 'command', 'command': 'take_photo', 'command_id': 'cmd-crash'})
    if asyncio.iscoroutine(r):
        asyncio.run(r)
    check(len(sent) == 1 and sent[0]['type'] == 'command_result' and sent[0]['ok'] is False
          and sent[0]['error']['code'] == 'exception', f"handler crash -> command_result exception ({sent})")
else:
    check(False, "message router not found")

# ---- take_photo: fast + slow capture via a fresh dispatcher
d = ToolDispatcher()
class FakeRelay:
    def __init__(self):
        self.uploads, self.events = [], []
        self.config = type('C', (), {'device_id': 'tb-test'})()
    def upload_media_sync(self, filepath, content_type='image/jpeg', timeout=20.0, filename=None):
        self.uploads.append((filepath, filename)); return f"https://relay/media/{filename}"
    def send_event(self, et, data):
        self.events.append((et, data))
fr = FakeRelay(); d._relay = fr
d.PHOTO_REPLY_BUDGET_S = 0.5

fake_res = {'success': True, 'data': 'QUJD', 'filename': 'x.jpg', 'filepath': '/tmp/x.jpg',
            'resolution': '640x480', 'timestamp': 't'}
d._capture_photo_blocking = staticmethod(lambda with_hud: dict(fake_res))
out = d._tool_take_photo({'_command_id': 'c-fast', 'with_hud': False}, threading.Event())
check(out['ok'] and out['jpeg_b64'] == 'QUJD' and out.get('photo_id') and out['width'] == 640
      and 'pending' not in out, f"fast capture -> jpeg_b64 + photo_id ({sorted(out)})")

def slow(with_hud):
    time.sleep(1.5); return dict(fake_res)
d._capture_photo_blocking = staticmethod(slow)
t0 = time.monotonic()
out = d._tool_take_photo({'_command_id': 'c-slow', 'with_hud': True}, threading.Event())
dt = time.monotonic() - t0
check(out['ok'] is True and out.get('pending') is True and out.get('photo_id') and dt < 1.0
      and out['media_filename'] == out['photo_id'] + '.jpg',
      f"slow capture -> immediate ok:true photo_id pending in {dt:.2f}s ({sorted(out)})")
for _ in range(60):
    if fr.events:
        break
    time.sleep(0.05)
check(fr.uploads and fr.uploads[0][1] == out['photo_id'] + '.jpg', f"JPEG uploaded as <photo_id>.jpg ({fr.uploads})")
check(fr.events and fr.events[0][0] == 'photo_ready' and fr.events[0][1]['command_id'] == 'c-slow'
      and fr.events[0][1]['photo_id'] == out['photo_id'] and fr.events[0][1]['image_url'].endswith('.jpg'),
      f"photo_ready {{command_id, photo_id, image_url}} sent ({fr.events})")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
