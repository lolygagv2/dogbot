#!/usr/bin/env python3
"""PhotoCaptureService.capture_photo_safe never blocks past its timeout and
never raises (2026-11-07).

Checks: fast capture passes the result through; a capture that hangs returns
success:false 'timed out' inside timeout+0.5 s and leaves the caller running;
a capture that raises (even BaseException) returns success:false; a detector
with no frame returns success:false; the wrapper never calls anything on the
detector other than get_last_frame (no camera handle access); the tool
dispatcher's take_photo answers ok:false capture_failed when the capture
times out, within its reply budget.

Run: /home/morgan/dogbot/env_new/bin/python tests/media/test_photo_capture_safe.py
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='photosafe_')

import numpy as np  # noqa: E402
from services.media.photo_capture import PhotoCaptureService  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class SpyDetector:
    """Records every attribute touched; only get_last_frame is legitimate."""
    def __init__(self, frame=None, hang=0.0, raise_exc=None):
        self._frame, self._hang, self._raise = frame, hang, raise_exc
        self.touched = set()

    def __getattr__(self, name):
        self.touched.add(name)
        if name == 'get_last_frame':
            def _f():
                if self._hang:
                    time.sleep(self._hang)
                if self._raise:
                    raise self._raise
                return self._frame
            return _f
        raise AttributeError(name)


svc = PhotoCaptureService()
frame = np.zeros((48, 64, 3), dtype=np.uint8)

# fast path
det = SpyDetector(frame)
res = svc.capture_photo_safe(with_hud=False, detector=det, timeout_s=2.0)
check(res.get('success') is True and res.get('resolution') == '64x48', f"fast capture passes through ({res.get('error')})")
check(det.touched <= {'get_last_frame'}, f"only get_last_frame touched on detector ({det.touched})")

# hang -> timeout, caller not blocked
det = SpyDetector(frame, hang=3.0)
t0 = time.monotonic()
res = svc.capture_photo_safe(with_hud=False, detector=det, timeout_s=0.5)
dt = time.monotonic() - t0
check(res.get('success') is False and 'timed out' in res.get('error', '') and dt < 1.0,
      f"hung capture -> success:false timed out in {dt:.2f}s ({res})")

# exception inside capture -> success:false, never raises
det = SpyDetector(frame, raise_exc=RuntimeError('camera exploded'))
res = svc.capture_photo_safe(with_hud=False, detector=det, timeout_s=2.0)
check(res.get('success') is False and 'camera exploded' in res.get('error', ''), f"exception -> success:false ({res})")

# BaseException (e.g. KeyboardInterrupt-class) inside capture -> still a dict
class Nasty(BaseException):
    pass
orig = svc.capture_photo
svc.capture_photo = lambda **kw: (_ for _ in ()).throw(Nasty('base'))
res = svc.capture_photo_safe(with_hud=False, detector=SpyDetector(frame), timeout_s=2.0)
svc.capture_photo = orig
check(res.get('success') is False and 'Nasty' in res.get('error', ''), f"BaseException -> success:false ({res})")

# no frame -> success:false
res = svc.capture_photo_safe(with_hud=False, detector=SpyDetector(None), timeout_s=2.0)
check(res.get('success') is False and 'No camera frame' in res.get('error', ''), f"no frame -> success:false ({res})")

# dispatcher take_photo: capture timeout -> ok:false capture_failed inside budget
import services.cloud.tool_dispatcher as td  # noqa: E402
td.log_tool_call = lambda *a, **k: None
d = td.ToolDispatcher()
d._relay = None
d.PHOTO_REPLY_BUDGET_S = 2.0
d._capture_photo_blocking = staticmethod(
    lambda with_hud: svc.capture_photo_safe(with_hud=with_hud, detector=SpyDetector(frame, hang=3.0), timeout_s=0.5))
t0 = time.monotonic()
out = d._tool_take_photo({'_command_id': 'c-timeout', 'with_hud': False}, threading.Event())
dt = time.monotonic() - t0
check(out.get('ok') is False and out['error']['code'] == 'capture_failed' and 'timed out' in out['error']['message']
      and dt < 1.5, f"take_photo on capture timeout -> ok:false capture_failed in {dt:.2f}s ({out})")

# real default cap is 5 s on both layers
check(PhotoCaptureService.CAPTURE_TIMEOUT_S == 5.0 and td.ToolDispatcher.PHOTO_CAPTURE_TIMEOUT_S == 5.0,
      "default capture timeout is 5 s")

print('\nALL PASS' if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
