#!/usr/bin/env python3
"""Relay `look` command: one bounded, gentle camera step (relay brief 2026-11-12).

A PanTiltService shell with a fake servo (no hardware, no bus). Checks: direction
of each action for both tilt conventions, gentle stepping (several small servo
writes, about 30 deg/s), clamping + limit_reached, center, reported pan/tilt
relative to centre, gimbal_busy right after a joystick command and while another
look runs, invalid action, and the tool's argument validation / error codes.

Run: /home/morgan/dogbot/env_new/bin/python tests/motion/test_look_step.py
Standalone (no pytest).
"""
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.motion.pan_tilt import PanTiltService  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class FakeServo:
    def __init__(self):
        self.writes = []

    def set_camera_pan(self, a, smooth=False):
        self.writes.append(('pan', a))
        return True

    def set_camera_pitch(self, a, smooth=False):
        self.writes.append(('tilt', a))
        return True


def make(tilt_down_sign=1.0):
    pt = PanTiltService.__new__(PanTiltService)
    pt.logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None,
                                      error=lambda *a, **k: None, warning=lambda *a, **k: None)
    pt.servo, pt.servo_initialized = FakeServo(), True
    pt.pan_limits, pt.tilt_limits = (40, 180), (30, 150)
    pt.center_pan, pt.center_tilt = 110, 90
    pt.current_pan, pt.current_tilt = 110, 90
    pt._tilt_down_sign = tilt_down_sign
    pt.last_servo_command_time, pt.min_command_interval = 0.0, 0.05
    pt._manual_camera_control = False
    pt._last_manual_move = 0.0
    pt._look_lock = threading.Lock()
    pt.smoothing_factor, pt.pending_pan, pt.pending_tilt = 0.3, None, None
    pt.state = types.SimpleNamespace(update_hardware=lambda **k: None)
    return pt


def main():
    import services.motion.pan_tilt as mod
    mod.publish_motion_event = lambda *a, **k: None

    pt = make()
    t0 = time.time()
    r = pt.look_step('pan_left', 15)
    took = time.time() - t0
    pans = [a for k, a in pt.servo.writes if k == 'pan']
    check(r['ok'] and pt.current_pan == 125 and r['pan'] == -15 and r['moved_degrees'] == 15
          and r['limit_reached'] is False, f'pan_left 15: servo 110 -> 125, reported pan -15 ({r})')
    check(len(pans) >= 8 and max(b - a for a, b in zip([110] + pans, pans)) <= 1.6 and 0.3 <= took <= 1.2,
          f'gentle: {len(pans)} small steps over {took:.2f}s (about 30 deg/s)')
    r = pt.look_step('pan_right', 30)
    check(pt.current_pan == 95 and r['pan'] == 15, 'pan_right 30: reported pan +15 (right is positive)')
    r = pt.look_step('tilt_up', 20)
    check(pt.current_tilt == 70 and r['tilt'] == 20, 'tilt_up with "larger = down" servo: tilt 90 -> 70, reported +20')
    r = pt.look_step('tilt_down', 10)
    check(pt.current_tilt == 80 and r['tilt'] == 10, 'tilt_down 10: reported tilt +10')
    check(pt._manual_camera_control is False, 'tracking hold released after the step')

    up = make(tilt_down_sign=-1.0)
    r = up.look_step('tilt_up', 20)
    check(up.current_tilt == 110 and r['tilt'] == 20, 'tilt_up with "larger = up" servo (tb5): tilt 90 -> 110, reported +20')

    r = pt.look_step('pan_right', 45)
    r = pt.look_step('pan_right', 45)
    check(pt.current_pan == 40 and r['limit_reached'] is True and r['moved_degrees'] == 10 and r['pan'] == 70,
          f'clamped at the pan limit: limit_reached, moved_degrees 10 ({r})')
    r = pt.look_step('pan_right', 15)
    check(r['ok'] and r['limit_reached'] is True and r['moved_degrees'] == 0, 'already at the limit: moved 0')
    r = pt.look_step('center')
    check(pt.current_pan == 110 and pt.current_tilt == 90 and r['pan'] == 0 and r['tilt'] == 0, 'center returns to 0,0')

    check(pt.look_step('spin', 15) == {'ok': False, 'code': 'invalid_action'}, 'unknown action -> invalid_action')
    pt.move_camera(pan=120, smooth=False)        # the owner's joystick
    before = pt.current_pan
    r = pt.look_step('pan_left', 15)
    check(r == {'ok': False, 'code': 'gimbal_busy'} and pt.current_pan == before,
          'right after a joystick command -> gimbal_busy, camera untouched')
    pt._last_manual_move = time.time() - 10
    check(pt.look_step('pan_left', 5)['ok'], 'allowed again once the joystick has been idle')

    results = []
    th = threading.Thread(target=lambda: results.append(pt.look_step('pan_right', 45)))
    th.start()
    time.sleep(0.2)
    second = pt.look_step('tilt_up', 10)
    th.join()
    check(second == {'ok': False, 'code': 'gimbal_busy'} and results[0]['ok'], 'second look during a look -> gimbal_busy')

    # ---- tool argument handling
    fake = types.ModuleType('services.motion.pan_tilt_fake')
    import services.cloud.tool_dispatcher as td
    d = td.ToolDispatcher()
    mod.get_pantilt_service = lambda: pt
    pt._last_manual_move = 0.0
    out = d._tool_look({'action': 'pan_left', 'degrees': 10, 'source': 'connector'}, None)
    check(out.get('ok') and set(out) >= {'pan', 'tilt', 'limit_reached', 'moved_degrees'}, f'tool: result fields ({out})')
    check(d._tool_look({'action': 'up'}, None)['error']['code'] == 'invalid_action', 'tool: invalid_action')
    check(d._tool_look({'action': 'pan_left', 'degrees': 60}, None)['error']['code'] == 'invalid_value'
          and d._tool_look({'action': 'pan_left', 'degrees': 'far'}, None)['error']['code'] == 'invalid_value',
          'tool: degrees outside 5-45 or not a number -> invalid_value')
    check(d._tool_look({'action': 'center', 'degrees': 999}, None).get('ok'), 'tool: degrees ignored for center')
    d.running = lambda group=None: [{'name': 'move'}]
    check(d._tool_look({'action': 'pan_left'}, None)['error']['code'] == 'gimbal_busy', 'tool: gimbal_busy while a base move runs')
    del fake

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
