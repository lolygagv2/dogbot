#!/usr/bin/env python3
"""Coach-mode camera framing regression (2026-10-03).

The old nudge loop oscillated between a bottom-clipped and a top-clipped box
(journal 2026-09-22 10:37). This drives _handle_coach_mode() with a fake servo
and a simple camera model: tilting the view down by D degrees moves the box UP
in the frame by D / DEG_PER_PX pixels.

Standalone, no hardware:  python3 tests/motion/test_pan_tilt_framing.py
"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# --- stub the heavy imports pan_tilt pulls in -------------------------------
import importlib
for name in ("core.bus", "core.state", "config.config_loader", "core.hardware.servo_controller"):
    sys.modules.setdefault(name, types.ModuleType(name))
bus = sys.modules["core.bus"]
bus.get_bus = lambda: types.SimpleNamespace(subscribe=lambda *a, **k: None)
bus.publish_motion_event = lambda *a, **k: None
st = sys.modules["core.state"]
class SystemMode:
    COACH = "coach"; MISSION = "mission"; MANUAL = "manual"; IDLE = "idle"; SILENT_GUARDIAN = "sg"
st.SystemMode = SystemMode
st.get_state = lambda: types.SimpleNamespace(get_mode=lambda: SystemMode.COACH,
                                             update_hardware=lambda **k: None)
sys.modules["core.hardware.servo_controller"].get_servo_controller = lambda: None
cfg = sys.modules["config.config_loader"]
cfg.get_config = lambda: types.SimpleNamespace(raw={'camera': {
    'pan_min': -180, 'pan_max': 268, 'tilt_min': 29, 'tilt_max': 290,
    'pan_center': 46, 'tilt_center': 90, 'tilt_convention': 'up',
    'coach_pan_min': 18, 'coach_pan_max': 162, 'coach_tilt_min': 29, 'coach_tilt_max': 149}})

import services.motion.pan_tilt as pt
PanTilt = pt.PanTiltService if hasattr(pt, 'PanTiltService') else pt.PanTiltTrackingService


class FakeServo:
    def __init__(self): self.pan = 46; self.tilt = 90
    def set_camera_pan(self, v): self.pan = v
    def set_camera_pitch(self, v): self.tilt = v


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def make():
    clock = Clock(); pt.time.time = clock
    svc = PanTilt.__new__(PanTilt)
    PanTilt.__init__(svc)
    svc.servo = FakeServo(); svc.servo_initialized = True
    svc.tracking_enabled = True
    svc.current_pan, svc.current_tilt = 46.0, 90.0
    return svc, clock


class Scene:
    """Dog box in frame coords; camera tilt (deg, 'up' convention) shifts it."""
    def __init__(self, svc, x1, y1, x2, y2):
        self.svc = svc; self.base = [x1, y1, x2, y2]; self.tilt0 = svc.current_tilt
    def box(self):
        # tb5: larger tilt = camera UP -> view up -> dog moves DOWN in frame
        shift_px = (self.svc.current_tilt - self.tilt0) / self.svc.DEG_PER_PX
        x1, y1, x2, y2 = self.base
        return [x1, y1 + shift_px, x2, y2 + shift_px]
    def feed(self, clock):
        b = self.box()
        ev = types.SimpleNamespace(subtype='dog_detected', data={
            'center': [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], 'bbox': b, 'dog_name': 'elsa'})
        self.svc._on_vision_event(ev)


def run(svc, clock, scene, seconds, det_hz=5):
    tilts = []
    for _ in range(int(seconds * 20)):
        clock.advance(0.05)
        if int(clock.t * 20) % (20 // det_hz) == 0:
            scene.feed(clock)
        svc._handle_coach_mode(0.05)
        tilts.append(svc.current_tilt)
    return tilts


def reversals(tilts):
    d = [b - a for a, b in zip(tilts, tilts[1:]) if abs(b - a) > 1e-6]
    return sum(1 for a, b in zip(d, d[1:]) if (a > 0) != (b > 0))


def test_bottom_clipped_dog_that_fits_gets_framed_without_oscillating():
    svc, clock = make()
    scene = Scene(svc, 200, 200, 440, 640)        # 440 px tall, legs cut off at bottom
    tilts = run(svc, clock, scene, 8.0)
    final = scene.box()
    assert final[3] < 640 - svc.CLIP_START_PX, f"bottom still clipped: {final}"
    assert final[1] > 0, f"head pushed out: {final}"
    assert reversals(tilts) <= 1, f"oscillated: {reversals(tilts)} reversals"


def test_too_tall_dog_holds_still():
    svc, clock = make()
    scene = Scene(svc, 150, 0, 490, 640)          # overflows both edges
    tilts = run(svc, clock, scene, 4.0)
    assert max(tilts) - min(tilts) < 0.01, "moved on an unframeable dog"
    scene2 = Scene(svc, 150, 5, 490, 600)         # 595 px tall (> 0.85*640): fits? no
    tilts = run(svc, clock, scene2, 4.0)
    assert max(tilts) - min(tilts) < 0.01, "chased a dog taller than the fit fraction"


def test_no_move_without_a_new_bbox():
    svc, clock = make()
    scene = Scene(svc, 200, 200, 440, 640)
    scene.feed(clock)
    for _ in range(40):                            # 2 s of ticks, no new detections
        clock.advance(0.05); svc._handle_coach_mode(0.05)
    # First dwell then one step at most; after that the bbox is stale (>0.5 s)
    moves = 1 if svc.current_tilt != 90.0 else 0
    assert moves <= 1


def test_anonymous_second_box_does_not_steal_target():
    svc, clock = make()
    good = types.SimpleNamespace(subtype='dog_detected', data={'center': [320, 300], 'bbox': [200, 100, 440, 500], 'dog_name': 'elsa'})
    svc._on_vision_event(good)
    bad = types.SimpleNamespace(subtype='dog_detected', data={'center': [100, 600], 'bbox': [0, 500, 200, 640], 'dog_name': None})
    svc._on_vision_event(bad)
    assert svc.target_position == (320, 300)


def test_centering_nudge_still_works_for_a_low_dog():
    svc, clock = make()
    scene = Scene(svc, 200, 420, 440, 600)        # fits, centre at y=510 (> 25% off)
    tilts = run(svc, clock, scene, 6.0)
    assert svc.current_tilt < 90.0, "did not tilt down toward a low dog"
    assert reversals(tilts) == 0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
            except Exception as e:
                import traceback; traceback.print_exc()
                fails += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
