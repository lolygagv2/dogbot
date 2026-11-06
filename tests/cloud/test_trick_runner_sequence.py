#!/usr/bin/env python3
"""TrickRunner (run_trick command) sequence + mode-restore guarantee (2026-11-06).

Fakes for state/fsm/coaching engine/interpreter/audio/dispenser/LED/voice.
Checks: success path (cue played, one treat, 'good'), no_dog / wrong_pose /
timeout classification, invalid cue, busy in SG without force, engine
hold+release, prior mode restored even when the dispenser raises.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_trick_runner_sequence.py
Standalone (no pytest).
"""
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core.state as core_state  # noqa: E402  (real SystemMode)
from core.state import SystemMode  # noqa: E402
import core.bus as core_bus  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


# ---------------------------------------------------------------- fakes
class FakeState:
    def __init__(self, mode=SystemMode.IDLE, locked=False):
        self.mode, self.locked = mode, locked
    def get_mode(self):
        return self.mode
    def is_mode_locked(self):
        return self.locked


class FakeFSM:
    def __init__(self, state):
        self.state, self.override_mode, self.calls = state, None, []
    def set_mode_override(self, mode, duration=None):
        self.calls.append(('override', mode.value)); self.state.mode = mode; self.override_mode = mode; return True
    def clear_override(self, reason=''):
        self.calls.append(('clear', reason)); self.override_mode = None
    def force_mode(self, mode, reason=''):
        self.calls.append(('force', mode.value)); self.state.mode = mode; return True


class FakeEngine:
    def __init__(self):
        self.held, self.log, self.current_session = False, [], None
    def hold(self, r):
        self.held = True; self.log.append('hold')
    def release(self):
        self.held = False; self.log.append('release')
    def reset_session_cooldown(self):
        self.log.append('reset')


class Chk:
    def __init__(self, completed=False, behavior=None, conf=0.0, hold=0.0):
        self.completed, self.behavior_detected, self.confidence, self.hold_duration = completed, behavior, conf, hold
        self.reason = ''


class FakeInterp:
    def __init__(self, script):
        self.script = list(script)       # list of Chk per check_trick call
        self.target, self.resets = None, 0
    def get_all_tricks(self):
        return ['sit', 'laydown', 'come', 'spin', 'speak']
    def get_trick_rules(self, t):
        return {'required_behavior': 'sit', 'detection_window_sec': 0.6, 'hold_duration_sec': 0.1}
    def reset_tracking(self):
        self.resets += 1
    def set_target_behavior(self, b):
        self.target = b
    def check_trick(self, t, dog_id=None):
        return self.script.pop(0) if self.script else Chk()


class FakeAudio:
    def __init__(self):
        self.played = []
    def play_file(self, p, loop=False):
        self.played.append(p); return {'success': True}
    def wait_for_completion(self, timeout=5.0):
        return True


class FakeDispenser:
    def __init__(self, raise_=False):
        self.calls, self.raise_ = [], raise_
    def dispense_treat(self, **kw):
        self.calls.append(kw)
        if self.raise_:
            raise RuntimeError('jam')
        return True


def install_fakes(state, fsm, engine, interp, audio, disp):
    core_state.get_state = lambda: state
    m = types.ModuleType('orchestrators.mode_fsm'); m.get_mode_fsm = lambda: fsm; sys.modules['orchestrators.mode_fsm'] = m
    m = types.ModuleType('orchestrators.coaching_engine'); m.get_coaching_engine = lambda: engine; sys.modules['orchestrators.coaching_engine'] = m
    m = types.ModuleType('core.behavior_interpreter'); m.get_behavior_interpreter = lambda: interp; sys.modules['core.behavior_interpreter'] = m
    m = types.ModuleType('services.media.usb_audio'); m.get_usb_audio_service = lambda: audio; sys.modules['services.media.usb_audio'] = m
    m = types.ModuleType('services.reward.dispenser'); m.get_dispenser_service = lambda: disp; sys.modules['services.reward.dispenser'] = m
    m = types.ModuleType('services.media.led'); m.get_led_service = lambda: types.SimpleNamespace(celebration_sequence=lambda d: True); sys.modules['services.media.led'] = m
    m = types.ModuleType('services.media.voice_lookup'); m.resolve_voice_file = lambda cmd, dog_id_override=None: f'/talks/{cmd}.mp3'; sys.modules['services.media.voice_lookup'] = m


import orchestrators.trick_runner as tr  # noqa: E402
tr.POSE_WARMUP_S, tr.ANNOUNCE_WAIT_S, tr.CUE_WAIT_S = 0.2, 0.0, 0.0


def see_dog(runner, delay=0.05):
    def _go():
        time.sleep(delay)
        core_bus.publish_vision_event('dog_detected', {'dog_id': 'dog_0'}, 'test')
    threading.Thread(target=_go, daemon=True).start()


def run(args, state=None, interp=None, disp=None, engine=None, dog=True):
    state = state or FakeState()
    fsm = FakeFSM(state)
    engine = engine or FakeEngine()
    interp = interp or FakeInterp([])
    audio = FakeAudio()
    disp = disp or FakeDispenser()
    install_fakes(state, fsm, engine, interp, audio, disp)
    r = tr.TrickRunner()
    if dog:
        see_dog(r)
    out = r.run(args, threading.Event())
    return out, fsm, engine, audio, disp, interp, state


def main():
    # success
    out, fsm, eng, audio, disp, interp, st = run(
        {'cue': 'sit'}, interp=FakeInterp([Chk(False, 'stand', 0.7), Chk(True, 'sit', 0.9, 1.1)]))
    check(out['ok'] and out['result'] == 'success' and out['behavior_detected'] == 'sit', f"success ({out['result']})")
    check(len(disp.calls) == 1 and disp.calls[0]['reason'] == 'connector_trick', "exactly one treat, reason connector_trick")
    check(audio.played[:2] == ['/talks/sit.mp3', '/talks/good.mp3'], f"cue then good ({audio.played})")
    check(('override', 'coach') in fsm.calls and st.mode == SystemMode.IDLE and out['mode_restored'],
          f"entered coach and restored IDLE ({fsm.calls})")
    check(eng.log == ['hold', 'release'] and not eng.held, "engine held then released")
    check(interp.target is None and interp.resets >= 1, "target cleared, tracking reset")

    # no dog at all
    out, *_ = run({'cue': 'sit', 'retry': False}, dog=False)
    check(out['result'] == 'no_dog' and not out['ok'], f"no detections -> no_dog ({out['result']})")

    # dog seen, wrong pose only
    out, fsm, eng, audio, disp, interp, st = run(
        {'cue': 'sit', 'retry': False}, interp=FakeInterp([Chk(False, 'stand', 0.8)] * 50))
    check(out['result'] == 'wrong_pose' and audio.played[-1] == '/talks/no.mp3' and not disp.calls,
          f"wrong pose -> wrong_pose, 'no' played, no treat ({out['result']})")

    # dog seen, no behavior -> timeout; retry gives two cues
    out, fsm, eng, audio, disp, interp, st = run({'cue': 'sit', 'retry': True})
    check(out['result'] == 'timeout' and out['attempts'] == 2 and audio.played.count('/talks/sit.mp3') == 2,
          f"timeout after retry, two cues ({out['result']}, {out['attempts']})")

    # invalid cue
    out, *_ = run({'cue': 'backflip'})
    check(out['result'] == 'invalid_cue' and out['error']['code'] == 'invalid_cue', "invalid cue rejected")

    # SG without force -> busy, nothing touched
    out, fsm, eng, *_ = run({'cue': 'sit'}, state=FakeState(SystemMode.SILENT_GUARDIAN))
    check(out['result'] == 'busy' and not fsm.calls and not eng.log, "SG without force -> busy, no side effects")
    out, fsm, eng, audio, disp, interp, st = run(
        {'cue': 'sit', 'force': True}, state=FakeState(SystemMode.SILENT_GUARDIAN),
        interp=FakeInterp([Chk(True, 'sit', 0.9, 1.0)]))
    check(out['result'] == 'success' and st.mode == SystemMode.SILENT_GUARDIAN, "force=true runs and restores SG")

    # already in COACH: no mode change, no restore
    out, fsm, eng, audio, disp, interp, st = run(
        {'cue': 'sit'}, state=FakeState(SystemMode.COACH), interp=FakeInterp([Chk(True, 'sit', 0.9, 1.0)]))
    check(out['result'] == 'success' and not fsm.calls and out['mode_restored'] is None, "in COACH: no mode switch")

    # dispenser raises -> still restored + released
    out, fsm, eng, audio, disp, interp, st = run(
        {'cue': 'sit'}, interp=FakeInterp([Chk(True, 'sit', 0.9, 1.0)]), disp=FakeDispenser(raise_=True))
    check(out['result'] == 'success' and out['treat_dispensed'] is False and 'dispense failed' in out['error']['message']
          and st.mode == SystemMode.IDLE and not eng.held, "dispenser exception: reported, mode restored, engine released")


if __name__ == '__main__':
    main()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
