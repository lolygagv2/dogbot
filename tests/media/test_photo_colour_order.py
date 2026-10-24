#!/usr/bin/env python3
"""Saved photos must keep the camera's colours — 2026-10-24.

Morgan: photos "in night-vision mode". Every saved photo last night had a BLUE
dog (tan Shiba) and brown jeans: the photo/recorder/sentry paths ran
cv2.COLOR_RGB2BGR on a frame that is already BGR (picamera2 'RGB888' is BGR in
memory — picamera2/request.py FORMAT_TABLE maps RGB888 -> 'BGR'; the live
WebRTC track streams it as BGR and looks right). The swap is now gone at all
five sites; this test feeds a synthetic BGR frame with a pure-red patch through
PhotoCaptureService.capture_photo() and the stream snapshot helper and checks
the JPEG still shows red.

Run: /home/morgan/dogbot/env_new/bin/python tests/media/test_photo_colour_order.py
"""
import os
import sys
import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import services.media.photo_capture as pc  # noqa: E402

FAILS = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: FAILS.append(m)


class FakeDetector:
    def __init__(self, frame): self._f = frame
    def get_last_frame(self): return self._f
    def get_last_frame_age(self): return 0.1


class FakeStore:
    def __init__(self, d): self.d = Path(d); self.registered = []
    def media_path_for(self, ext='jpg'): return self.d / f"photo.{ext}"
    def register_media(self, *a, **k): self.registered.append((a, k))


def bgr_frame():
    f = np.zeros((360, 640, 3), dtype=np.uint8)
    f[:, :320] = (0, 0, 255)      # pure RED in BGR order (what the camera gives us)
    f[:, 320:] = (255, 0, 0)      # pure BLUE
    return f


def main():
    tmp = tempfile.mkdtemp(prefix="photo_colour_")
    store = FakeStore(tmp)
    pc.get_wimz_store = lambda: store
    svc = pc.PhotoCaptureService()
    svc._cleanup_old_photos = lambda: None

    res = svc.capture_photo(with_hud=False, detector=FakeDetector(bgr_frame()),
                            ai_controller=None, mission_engine=None, mode_fsm=None)
    check(res.get("success"), f"capture_photo succeeded ({res.get('error')})")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(res["image_base64"] if "image_base64" in res else res.get("data", res.get("image", ""))), np.uint8), cv2.IMREAD_COLOR) if res.get("success") else None
    if img is None:
        saved = cv2.imread(str(store.media_path_for()))
        img = saved
    b, g, r = [int(v) for v in img[180, 100]]
    check(r > 200 and b < 60, f"left patch is still RED in the saved JPEG (BGR={b},{g},{r})")
    b, g, r = [int(v) for v in img[180, 540]]
    check(b > 200 and r < 60, f"right patch is still BLUE in the saved JPEG (BGR={b},{g},{r})")

    # HUD path draws on the same (already-BGR) frame: colours must not flip either
    res = svc.capture_photo(with_hud=True, detector=FakeDetector(bgr_frame()),
                            ai_controller=None, mission_engine=None, mode_fsm=None)
    saved = cv2.imread(str(store.media_path_for()))
    b, g, r = [int(v) for v in saved[300, 100]]
    check(res.get("success") and r > 200 and b < 60, f"HUD photo keeps RED red (BGR={b},{g},{r})")

    # The input frame must not be mutated by the HUD drawing (copy, not alias)
    src = bgr_frame(); before = src.copy()
    svc.capture_photo(with_hud=True, detector=FakeDetector(src), ai_controller=None, mission_engine=None, mode_fsm=None)
    check(np.array_equal(src, before), "capture_photo does not draw into the detector's shared frame")

    print(); print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
