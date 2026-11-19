#!/usr/bin/env python3
"""ROBOT-STATE v2 (2026-11-19), robot side. No hardware, no relay, no live files.

Covers: the five modes and nothing else (sleep is a flag, never a mode);
session_dog_id rules R1-R5; select_dog as the only writer of the selected
dog; the command_result frame (success / error string / duration_ms) for
set_mode, select_dog and set_sleep and the 2000 ms reply budget; mode_change
stamping (source / reason / command_id / session_dog_id); dog attribution
A1..A6; and the grep proof that nothing emits "last_dog", "vision" as an
id_method, or "mode_changed".

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_robot_state_v2_contract.py
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TMP = tempfile.mkdtemp(prefix='robotstate_')
os.environ['WIMZ_DATA_ROOT'] = TMP

import core.state as state_mod  # noqa: E402
from core.state import StateManager, SystemMode, CONTRACT_MODES, contract_mode  # noqa: E402

# Never read or write the robot's real persisted selection.
_ORIG_LOAD = StateManager._load_persisted_state
StateManager._load_persisted_state = lambda self: None
st = state_mod.get_state()
st._STATE_FILE = os.path.join(TMP, 'robot_state.json')

import core.dog_profile_manager as dpm  # noqa: E402
import services.cloud.tool_dispatcher as td  # noqa: E402
from services.power import sleep_manager as sm  # noqa: E402

ELSA = '019f58ad-03c9-7aff-adae-b697fb058ffc'
BEZIK = '01a183fe-0169-7d8b-ac62-5d58727473bf'
FAILS = []
td.log_tool_call = lambda *a, **k: None


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class Prof:
    def __init__(self, name, dog_id):
        self.name, self.dog_id = name, dog_id


class FakeProfiles:
    """Stands in for DogProfileManager: two paired dogs unless told otherwise."""

    def __init__(self):
        self.profiles = {'elsa': Prof('Elsa', ELSA), 'bezik': Prof('Bezik', BEZIK)}

    def resolve_app_dog_id(self, dog_id=None, dog_name=None, aruco_id=None):
        for key in (dog_id, dog_name):
            if key is None:
                continue
            s = str(key)
            for p in self.profiles.values():
                if p.dog_id == s:
                    return s
            if s.lower() in self.profiles:
                return self.profiles[s.lower()].dog_id
        return None

    def sole_app_dog_id(self):
        ids = [p.dog_id for p in self.profiles.values()]
        return ids[0] if len(ids) == 1 else None

    def get_all_profiles(self):
        return list(self.profiles.values())


profiles = FakeProfiles()
dpm._profile_manager = profiles

changes = []
st.subscribe('mode_change', changes.append)


def wait_changes(n, timeout=2.0):
    end = time.time() + timeout
    while len(changes) < n and time.time() < end:
        time.sleep(0.01)
    return len(changes) >= n


def reset(selected=None):
    """Back to idle, awake, with the given selected dog and no marker reads."""
    with st.mode_context(source='robot', reason='user test reset'):
        st.set_mode(SystemMode.IDLE, 'user test reset', force=True)
    st._marker_reads.clear()
    st._last_aruco_dog_id, st._last_aruco_time = None, 0.0
    st.set_current_dog(selected, None)
    time.sleep(0.05)
    changes.clear()


def fake_sleep_manager():
    m = sm.SleepManager()

    def sessions(source):
        st.set_mode(SystemMode.IDLE, f"user sleep ({source})", force=True)
    m._sleep_sessions = sessions
    for name in ('motors', 'gimbal', 'camera', 'hailo', 'audio_in', 'amp', 'leds', 'governor'):
        setattr(m, f'_sleep_{name}', lambda: None)
    for name in ('governor', 'leds', 'amp', 'audio_in', 'hailo', 'camera', 'gimbal'):
        setattr(m, f'_wake_{name}', lambda: None)
    m._health_check = lambda: {'camera': {'ok': True}, 'hailo': {'ok': True}, 'audio': {'ok': True}}
    m._emit_status = lambda: None
    m._start_status_loop = lambda: None
    m.read_power = lambda: {}
    m.SETTLE_S = 0
    return m


def wait_state(m, want, timeout=3.0):
    end = time.time() + timeout
    while m.state != want and time.time() < end:
        time.sleep(0.01)
    return m.state == want


FRAME_KEYS = {'type', 'command_id', 'command', 'success', 'result', 'error', 'duration_ms'}


def main():
    d = td.ToolDispatcher()

    # ---------------------------------------------------------------- modes
    check(CONTRACT_MODES == ('idle', 'silent_guardian', 'coach', 'mission', 'manual')
          and 'sleep' not in [m.value for m in SystemMode], "five contract modes; sleep is not a SystemMode")
    check([contract_mode(m) for m in (SystemMode.NIGHT_SENTRY, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY,
                                      SystemMode.SHUTDOWN)] == ['idle'] * 4
          and contract_mode('docking') == 'idle' and contract_mode(SystemMode.MANUAL) == 'manual',
          "R6: every internal state outside the five reports idle")

    # ---------------------------------------------------------------- C2 select_dog
    reset()
    t0 = time.time()
    f = d.call('select_dog', {'dog_id': BEZIK, 'source': 'app'}, 'c-sel-1', 3)
    check(set(f) == FRAME_KEYS and f['type'] == 'command_result' and f['command_id'] == 'c-sel-1'
          and f['command'] == 'select_dog' and f['success'] is True and f['error'] is None
          and f['result'] == {'selected_dog_id': BEZIK} and isinstance(f['duration_ms'], int)
          and 'ok' not in f and time.time() - t0 < 2.0,
          f"T1 select_dog Bezik -> command_result success, result selected_dog_id, <2000 ms ({f})")
    saved = json.load(open(st._STATE_FILE))
    check(saved.get('current_dog_id') == BEZIK and saved.get('current_dog_name') == 'Bezik',
          "selected dog persisted (with the profile's name)")
    f = d.call('select_dog', {'dog_id': None, 'source': 'app'}, 'c-sel-2', 3)
    check(f['success'] and f['result'] == {'selected_dog_id': None} and st.get_current_dog_id() is None,
          "select_dog null clears the selection")

    # ---------------------------------------------------------------- C1 set_mode, R2-R4, E1 stamp (T5/T6)
    reset(selected=BEZIK)
    t0 = time.time()
    f = d.call('set_mode', {'mode': 'silent_guardian', 'dog_id': ELSA, 'source': 'connector', 'reason': None},
               'c-mode-1', 3)
    dt = time.time() - t0
    check(set(f) == FRAME_KEYS and f['success'] is True and f['error'] is None
          and f['result'] == {'mode': 'silent_guardian', 'session_dog_id': ELSA} and dt < 2.0,
          f"T5 set_mode silent_guardian dog Elsa -> result mode + session_dog_id Elsa in {dt * 1000:.0f} ms")
    check(st.get_session_dog_id() == ELSA and st.get_current_dog_id() == BEZIK,
          "T5 session_dog_id Elsa, selected_dog_id still Bezik (R2: a mode never changes it)")
    wait_changes(1)
    c = changes[-1] if changes else {}
    check(c.get('mode') == 'silent_guardian' and c.get('contract_previous_mode') == 'idle'
          and c.get('session_dog_id') == ELSA and c.get('source') == 'connector'
          and c.get('command_id') == 'c-mode-1' and c.get('contract_reason') is None,
          f"T5 mode change stamped session Elsa, source connector, command_id ({c})")
    got = dpm.resolve_relay_dog(dog_id='dog_0', dog_name=None, id_method=None)
    check(got == (ELSA, 'session'), f"T5 next automatic treat -> dog Elsa, id_method session ({got})")

    f = d.call('set_mode', {'mode': 'silent_guardian', 'dog_id': BEZIK, 'source': 'app'}, 'c-mode-1b', 3)
    check(f['success'] and f['result']['session_dog_id'] == ELSA, "R3: session dog is fixed for the session")

    changes.clear()
    f = d.call('set_mode', {'mode': 'idle', 'dog_id': None, 'source': 'app', 'reason': 'dropdown'}, 'c-mode-2', 3)
    wait_changes(1)
    c = changes[-1] if changes else {}
    check(f['success'] and f['result'] == {'mode': 'idle', 'session_dog_id': None}
          and c.get('source') == 'app' and c.get('contract_reason') == 'dropdown' and c.get('session_dog_id') is None,
          f"T6 Silent Guardian off -> idle, session_dog_id null, reason dropdown ({f['result']})")

    f = d.call('set_mode', {'mode': 'coach', 'source': 'app'}, 'c-mode-3', 3)
    check(f['success'] and f['result'] == {'mode': 'coach', 'session_dog_id': BEZIK},
          "R3: no dog_id -> session_dog_id = selected dog")
    f = d.call('set_mode', {'mode': 'manual', 'source': 'app', 'reason': 'drive_enter'}, 'c-mode-4', 3)
    check(f['success'] and f['result'] == {'mode': 'manual', 'session_dog_id': None}, "R4: manual -> session null")

    reset(selected=None)
    f = d.call('set_mode', {'mode': 'silent_guardian', 'dog_id': None, 'source': 'app'}, 'c-mode-5', 3)
    check(f['success'] and f['result'] == {'mode': 'silent_guardian', 'session_dog_id': None},
          "T7/R1 no selected dog, no dog_id -> Silent Guardian still starts, session null")

    reset(selected=BEZIK)
    for bad in ('sleep', 'guardian', 'training', 'night_sentry', 'Idle', '', None):
        f = d.call('set_mode', {'mode': bad, 'source': 'app'}, 'c-bad', 3)
        if not (f['success'] is False and f['error'] == 'invalid_mode'):
            check(False, f"invalid mode {bad!r} -> invalid_mode ({f})")
            break
    else:
        check(st.get_contract_mode() == 'idle' and not changes,
              "any value outside the five (sleep, aliases, night_sentry) -> invalid_mode, state unchanged")

    # internal mode outside the five: reported idle, no session
    st.set_mode(SystemMode.NIGHT_SENTRY, 'Night Sentry started')
    check(st.get_contract_mode() == 'idle' and st.get_session_dog_id() is None,
          "R6: night_sentry reports mode idle")

    # mission start names the dog
    reset(selected=BEZIK)
    st.set_mode(SystemMode.MISSION, reason='Mission: sit', dog_id=ELSA)
    check(st.get_session_dog_id() == ELSA, "R3: mission started for Elsa -> session_dog_id Elsa")
    reset(selected=BEZIK)
    with st.mode_context(source='schedule', reason='schedule', dog_id='unknown'):
        st.set_mode(SystemMode.COACH, 'User override')
    wait_changes(1)
    check(st.get_session_dog_id() == BEZIK and changes and changes[-1].get('source') == 'schedule',
          "scheduled mode with no real dog -> selected dog, source schedule")

    # ---------------------------------------------------------------- C3 set_sleep, R5 (T10)
    reset(selected=BEZIK)
    mgr = fake_sleep_manager()
    sm._instance = mgr
    d.call('set_mode', {'mode': 'silent_guardian', 'dog_id': ELSA, 'source': 'app'}, 'c-s0', 3)
    wait_changes(1)
    changes.clear()
    t0 = time.time()
    f = d.call('set_sleep', {'asleep': True, 'source': 'app', 'reason': 'sleep_button'}, 'c-sleep-1', 3)
    dt = time.time() - t0
    check(set(f) == FRAME_KEYS and f['success'] is True and f['result'] == {'asleep': True, 'mode': 'idle'}
          and dt < 2.0, f"T10 set_sleep true -> asleep true, mode idle in {dt * 1000:.0f} ms ({f['result']})")
    check(wait_state(mgr, sm.ASLEEP) and st.get_contract_mode() == 'idle' and st.get_session_dog_id() is None,
          "R5: asleep forced mode idle and session_dog_id null")
    wait_changes(1)
    c = changes[-1] if changes else {}
    check(c.get('mode') == 'idle' and c.get('contract_previous_mode') == 'silent_guardian'
          and c.get('source') == 'app' and c.get('contract_reason') == 'sleep_button'
          and c.get('command_id') == 'c-sleep-1',
          f"going to sleep from Silent Guardian is a mode change to idle, stamped with the command ({c})")
    f = d.call('set_mode', {'mode': 'idle', 'source': 'app'}, 'c-s1', 3)
    check(f['success'] and f['result']['mode'] == 'idle' and mgr.state == sm.ASLEEP,
          "set_mode idle while asleep: stays asleep (idle is not a wake)")
    changes.clear()
    f = d.call('set_sleep', {'asleep': False, 'source': 'app', 'reason': 'wake_button'}, 'c-wake-1', 3)
    check(f['success'] and f['result'] == {'asleep': False, 'mode': 'idle'} and wait_state(mgr, sm.AWAKE)
          and st.get_contract_mode() == 'idle' and not changes,
          "R5: waking -> asleep false, mode stays idle, no mode_change")
    f = d.call('set_sleep', {'asleep': 'yes'}, 'c-sleep-bad', 3)
    check(f['success'] is False and f['error'] == 'bad_request', "set_sleep needs a boolean")
    d.call('set_sleep', {'asleep': True, 'source': 'app'}, 'c-sleep-2', 3)
    wait_state(mgr, sm.ASLEEP)
    f = d.call('set_mode', {'mode': 'coach', 'source': 'connector'}, 'c-s2', 5)
    check(f['success'] and f['result']['mode'] == 'coach' and mgr.state == sm.AWAKE,
          "a session mode on a sleeping robot wakes it first, then enters the mode")
    check(not d.has_tool('sleep') and not d.has_tool('wake') and d.has_tool('set_sleep')
          and d.has_tool('select_dog'), "hard cut: no sleep / wake commands, only set_sleep")
    # ---------------------------------------------------------------- command budget (2026-11-19)
    # From receipt on the monotonic clock: a command not BEGUN within 1500 ms
    # is not executed ("expired") and the wake still completes; set_sleep
    # false answers when awake, within 5000 ms.
    reset(selected=BEZIK)
    slow = fake_sleep_manager()
    slow._wake_camera = lambda: time.sleep(1.8)          # wake longer than the 1500 ms budget
    sm._instance = slow
    slow.sleep('test')
    changes.clear()
    t0 = time.monotonic()
    f = d.call('set_mode', {'mode': 'silent_guardian', 'source': 'connector'}, 'c-exp', 6)
    dt = time.monotonic() - t0
    check(f['success'] is False and f['error'] == 'expired' and f['result'].get('executed') is False
          and 1.45 <= dt <= 1.75 and 1450 <= f['duration_ms'] <= 1750,
          f"set_mode to a sleeping robot whose wake takes 1.8 s -> expired at {dt * 1000:.0f} ms")
    check(st.get_contract_mode() == 'idle' and not changes, "the expired command was not executed")
    check(wait_state(slow, sm.AWAKE, 4.0), "the abandoned command still completed the wake")
    time.sleep(0.3)
    check(st.get_contract_mode() == 'idle' and not changes, "and it is never run late after the wake")
    f = d.call('set_mode', {'mode': 'silent_guardian', 'source': 'connector'}, 'c-retry', 3)
    check(f['success'] and f['result']['mode'] == 'silent_guardian', "the caller's retry succeeds")

    d.call('set_mode', {'mode': 'idle', 'source': 'app'}, 'c-b0', 3)
    slow.sleep('test')
    t0 = time.monotonic()
    f = d.call('set_sleep', {'asleep': False, 'source': 'app'}, 'c-wake-slow', 8)
    dt = time.monotonic() - t0
    check(f['success'] and f['result'] == {'asleep': False, 'mode': 'idle'} and slow.state == sm.AWAKE
          and 1.7 <= dt < 5.0, f"set_sleep false answers once awake ({dt * 1000:.0f} ms, budget 5000)")
    slow.sleep('test')
    t0 = time.monotonic()
    f = d.call('select_dog', {'dog_id': ELSA, 'source': 'app'}, 'c-passive', 3)
    check(f['success'] and time.monotonic() - t0 < 0.5 and slow.state == sm.ASLEEP,
          "select_dog on a sleeping robot runs at once and does not wake it")
    slow.wake('test', greet=False)
    stuck = fake_sleep_manager()
    stuck._wake_camera = lambda: time.sleep(6.0)
    sm._instance = stuck
    stuck.sleep('test')
    t0 = time.monotonic()
    f = d.call('set_sleep', {'asleep': False, 'source': 'app'}, 'c-wake-stuck', 8)
    dt = time.monotonic() - t0
    check(f['success'] is False and f['error'] == 'wake_timeout' and dt < 5.0,
          f"a wake that does not finish still replies inside 5000 ms ({dt * 1000:.0f} ms, wake_timeout)")
    wait_state(stuck, sm.AWAKE, 5.0)
    sm._instance = None

    # ---------------------------------------------------------------- attribution A1..A6
    r = dpm.resolve_relay_dog
    reset(selected=BEZIK)
    d.call('set_mode', {'mode': 'silent_guardian', 'dog_id': ELSA, 'source': 'app'}, 'c-a', 3)
    st.update_aruco_dog('bezik')
    check(r(dog_id=ELSA, id_method='explicit') == (ELSA, 'explicit'),
          "A1 command carried a dog -> that dog, explicit (beats a fresh marker)")
    check(r(dog_id='dog_0') == (BEZIK, 'qr'), "A2 marker read within 5000 ms -> that dog, qr (beats session)")
    st.update_aruco_dog('elsa')
    check(r(dog_id='bezik', dog_name='Bezik') == (BEZIK, 'qr') and r() == (ELSA, 'qr'),
          "A2 two fresh markers: the event's own dog if its marker is fresh, else the latest read")
    st._marker_reads['bezik'] = st._marker_reads['elsa'] = time.time() - 5.5
    check(r(dog_id='bezik', dog_name='Bezik') == (ELSA, 'session'),
          "marker older than 5000 ms + vision-only match naming Bezik -> session dog Elsa (vision never attributes)")
    reset(selected=BEZIK)
    check(r(dog_name='Elsa', id_method='color') == (BEZIK, 'owner_selected'), "A4 no session -> selected dog, owner_selected")
    reset(selected=None)
    check(r() == (None, 'none'), "A6 two profiles, nothing selected -> dog_id null, none")
    del profiles.profiles['bezik']
    check(r() == (ELSA, 'sole_dog'), "A5 exactly one profile -> that dog, sole_dog")
    profiles.profiles['bezik'] = Prof('Bezik', BEZIK)

    # Inside Coach the order is the contract order: command dog_id, then a
    # marker read within 5000 ms, then the session dog. The Coach engine's own
    # name for a dog (colour / vision-only) never attributes.
    reset(selected=BEZIK)
    d.call('set_mode', {'mode': 'coach', 'dog_id': ELSA, 'source': 'app'}, 'c-coach', 3)
    st.set_coach_dog('dog_1')
    check(r(dog_id='dog_1', dog_name='Bezik') == (ELSA, 'session')
          and r(dog_id='dog_1', dog_name='Bezik', id_method='color') == (ELSA, 'session'),
          "Coach session for Elsa, engine named its dog Bezik by colour -> Elsa, session")
    check(r(dog_id='dog_1', dog_name='Bezik', id_method='explicit') == (ELSA, 'session'),
          "a dog NAME marked explicit is not a command dog_id -> still the session dog")
    st.update_aruco_dog('bezik')
    check(r(dog_id='dog_1', dog_name='Bezik') == (BEZIK, 'qr'),
          "Coach's dog identified by a marker read within 5000 ms -> Bezik, qr")
    check(r(dog_id=ELSA, id_method='explicit') == (ELSA, 'explicit'),
          "run_trick / command dog_id inside Coach -> explicit (beats the marker)")
    check(r(dog_id='0a000000-0000-7000-8000-000000000001', id_method='explicit')
          == ('0a000000-0000-7000-8000-000000000001', 'explicit'),
          "a command's app UUID with no local profile yet is still explicit, verbatim")
    st.set_coach_dog('bezik')
    check(st.get_coach_dog_id() == 'bezik' and st.get_active_dog_id(aruco_ttl=0) == 'bezik'
          and st.get_session_dog_id() == ELSA,
          "Coach keeps _coach_dog_id for voice clips and spoken names (session dog untouched)")
    st.clear_coach_dog()
    from services.reward.dispenser import COMMAND_REASONS as _CR
    check('coaching_reward' not in _CR, "a Coach reward is never stamped explicit by the dispenser")

    # ---------------------------------------------------------------- selected_dog_changed_locally
    reset(selected=None)
    check(st.selected_dog_changed_locally() is False, "flag starts false")
    d.dispatch({'name': 'select_dog', 'id': None, 'args': {'dog_id': BEZIK, 'source': 'app'}, 'transport': 'local'},
               reply=lambda f: None)
    time.sleep(0.3)
    check(st.get_current_dog_id() == BEZIK and st.selected_dog_changed_locally() is True
          and json.load(open(st._STATE_FILE)).get('selected_dog_changed_locally') is True,
          "select_dog over the local / AP connection -> flag true, persisted")
    real_load = state_mod.StateManager._load_persisted_state
    state_mod.StateManager._load_persisted_state = _ORIG_LOAD
    reborn = state_mod.StateManager.__new__(state_mod.StateManager)
    import threading as _th, logging as _lg
    reborn._lock, reborn.logger = _th.RLock(), _lg.getLogger('reborn')
    reborn._current_dog_id = reborn._current_dog_name = None
    reborn._selected_dog_changed_locally = False
    reborn._STATE_FILE = st._STATE_FILE
    reborn._load_persisted_state()
    state_mod.StateManager._load_persisted_state = real_load
    check(reborn._current_dog_id == BEZIK and reborn._selected_dog_changed_locally is True,
          "after a restart following a local select_dog -> selection and flag true are restored")
    d.dispatch({'name': 'select_dog', 'id': 'r-1', 'args': {'dog_id': BEZIK, 'source': 'app'}, 'transport': 'relay'},
               reply=lambda f: None)
    time.sleep(0.3)
    check(st.selected_dog_changed_locally() is False
          and json.load(open(st._STATE_FILE)).get('selected_dog_changed_locally') is False,
          "select_dog from the relay (same value) -> flag false, persisted")

    from services.reward.dispenser import COMMAND_REASONS
    check({'cloud_command', 'agent', 'connector', 'schedule', 'manual'} <= COMMAND_REASONS
          and not ({'quiet_reward', 'coach', 'mission'} & COMMAND_REASONS),
          "treats: only command-caused dispenses with a dog are stamped explicit")

    # ---------------------------------------------------------------- relay frame routing
    from services.cloud.relay_client import RelayClient
    rc = RelayClient.__new__(RelayClient)
    import logging
    rc.logger = logging.getLogger('test')
    seen = []
    real_get = td.get_tool_dispatcher

    class Disp:
        def has_tool(self, name):
            return d.has_tool(name)

        def dispatch(self, msg, reply):
            seen.append(msg)
    td.get_tool_dispatcher = lambda: Disp()
    rc.send_message = lambda m: None
    frame = {'type': 'command', 'command': 'set_mode', 'command_id': 'u-1',
             'data': {'mode': 'coach', 'dog_id': ELSA, 'source': 'agent', 'reason': None,
                      'timestamp': '2026-11-19T18:47:32.551Z'}}
    ok = rc._dispatch_v2_command('set_mode', frame, frame['data'])
    check(ok and seen and (seen[-1]['id'], seen[-1]['name']) == ('u-1', 'set_mode')
          and seen[-1]['args'] == {'mode': 'coach', 'dog_id': ELSA, 'source': 'agent', 'reason': None},
          f"C1 frame: parameters read from data, timestamp not passed on ({seen[-1:]})")
    frame = {'type': 'command', 'command': 'select_dog', 'command_id': 'u-2', 'data': {'dog_id': None}}
    rc._dispatch_v2_command('select_dog', frame, frame['data'])
    check(seen[-1]['args'] == {'dog_id': None, 'source': 'relay'}, "a frame with no source is stamped relay")
    td.get_tool_dispatcher = real_get

    # ---------------------------------------------------------------- T11 grep proof
    skip = ('tests', 'archive', 'Archive', 'env_new', 'hailov2', 'driver_extract', '__pycache__', 'docs',
            'patches', 'programs')
    live = [p for p in ROOT.rglob('*.py')
            if not (set(p.relative_to(ROOT).parts[:-1]) & set(skip)) and len(p.relative_to(ROOT).parts) > 1]
    live.append(ROOT / 'main_treatbot.py')
    hits = []
    pats = [re.compile(r"""['"]last_dog['"]"""),
            re.compile(r"""id_method['"]?\s*[:=]\s*['"]vision['"]|return\s+['"]vision['"]|,\s*['"]vision['"]\s*\)?\s*$"""),
            re.compile(r"""['"]mode_changed['"]"""),
            re.compile(r"""SystemMode\.SLEEP|['"]mode['"]\s*:\s*['"]sleep['"]""")]
    allowed = {('core/treat_dispenser_robot.py', 2), ('services/logging/dog_event_logger.py', 2)}
    for p in live:
        rel = str(p.relative_to(ROOT))
        for n, line in enumerate(p.read_text(errors='ignore').splitlines(), 1):
            code = line.split('#', 1)[0]
            for i, pat in enumerate(pats):
                if pat.search(code) and (rel, i) not in allowed:
                    hits.append(f"{rel}:{n}: {line.strip()[:90]}")
    check(not hits, "T11 no code path emits last_dog, vision as id_method, mode_changed, or sleep as a mode "
          + (f"-> {hits[:6]}" if hits else f"({len(live)} files scanned)"))

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILURE(S): ' + str(FAILS)}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
