#!/usr/bin/env python3
"""Sleep mode manager: command classification, step order, wake gate and the
asleep-only battery shutdown. Hardware steps are replaced by recorders.
Standalone: python3 tests/power/test_sleep_manager.py
"""
import os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from services.power import sleep_manager as sm

SLEEP_ORDER = ['sessions', 'motors', 'gimbal', 'camera', 'hailo', 'audio_in', 'amp', 'leds', 'governor']


def _manager(calls):
    m = sm.SleepManager()
    for name in SLEEP_ORDER:
        setattr(m, f'_sleep_{name}', (lambda n=name: calls.append(f'sleep:{n}')) if name != 'sessions'
                else (lambda source, n=name: calls.append(f'sleep:{n}')))
    for name in ('governor', 'leds', 'amp', 'audio_in', 'hailo', 'camera', 'gimbal'):
        setattr(m, f'_wake_{name}', lambda n=name: calls.append(f'wake:{n}'))
    m._wake_mode = lambda source: calls.append('wake:sessions')
    m._health_check = lambda: (calls.append('wake:health') or
                               {'camera': {'ok': True}, 'hailo': {'ok': True}, 'audio': {'ok': True}})
    m._emit_status = lambda: calls.append('status')
    m._start_status_loop = lambda: None
    m.read_power = lambda: {'battery_current_sensor': False}
    m.SETTLE_S = 0
    return m


def test_action_command_classification():
    a = sm.is_action_command
    for cmd in ('dispense_treat', 'treat', 'move', 'look', 'take_photo', 'play_sound', 'start_mission',
                'motor_command', 'servo', 'call_dog', 'listen', 'led', 'audio'):
        assert a(cmd, {}), cmd
    for cmd in ('set_sleep', 'select_dog', 'ping', 'standby', 'stop', 'emergency_stop', 'schedule_sync', 'schedule_mission', 'cancel_mission',
                'sg_config', 'audio_volume', 'smart_plug', 'local_mode', 'cloud_mode'):
        assert not a(cmd, {}), cmd
    assert a('set_mode', {'mode': 'coach'}) and a('set_mode', {'mode': 'silent_guardian'})
    assert a('set_mode', {'mode': 'mission'}) and a('set_mode', {'mode': 'manual'})
    assert not a('set_mode', {'mode': 'idle'}) and not a('set_mode', {'mode': 'sleep'})
    assert not a('move', {'action': 'stop'}) and a('move', {'action': 'forward'})
    assert not a(None)


def test_sleep_is_not_a_mode():
    """ROBOT-STATE v2: sleep is a flag (asleep), never a mode value."""
    from core.state import SystemMode, CONTRACT_MODES
    assert 'sleep' not in [m.value for m in SystemMode] and 'sleep' not in CONTRACT_MODES
    assert not hasattr(sm, 'sleep_intent') and not hasattr(sm, 'run_intent')
    calls = []
    m = _manager(calls)
    res = m.sleep('test')
    assert res['asleep'] is True and res['mode'] == 'idle' and m.is_asleep_target() is True
    assert m.sleep('again')['mode'] == 'idle'
    res = m.wake('test')
    assert res['asleep'] is False and res['mode'] == 'idle' and m.is_asleep_target() is False


def test_sleep_order_then_wake_in_reverse():
    calls = []
    m = _manager(calls)
    greeted = threading.Event()
    m._greeting_hook = greeted.set
    res = m.sleep('test')
    assert res['ok'] and m.state == sm.ASLEEP and m.is_asleep()
    assert [c for c in calls if c.startswith('sleep:')] == [f'sleep:{n}' for n in SLEEP_ORDER]
    assert m.sleep('again').get('already')

    calls.clear()
    res = m.wake('test')
    assert res['ok'] and m.state == sm.AWAKE and res['healthy'] and 'ready_s' in res
    woke = [c.split(':')[1] for c in calls if c.startswith('wake:')]
    # hailo loads on a side thread next to the camera; everything else is strictly reversed
    assert [w for w in woke if w != 'hailo'] == ['governor', 'leds', 'amp', 'audio_in', 'camera',
                                                 'gimbal', 'health', 'sessions']
    assert woke.index('hailo') < woke.index('gimbal')
    assert greeted.wait(2.0), 'boot greeting hook must run after wake'
    assert m.wake('again').get('already')


def test_failed_step_does_not_stop_the_sequence():
    calls = []
    m = _manager(calls)
    def boom():
        raise RuntimeError('no camera')
    m._sleep_camera = boom
    res = m.sleep('test')
    assert res['failed'] == ['camera'] and m.state == sm.ASLEEP
    assert 'sleep:governor' in calls


def test_ensure_awake_wakes_only_when_asleep():
    calls = []
    m = _manager(calls)
    assert m.ensure_awake('cmd') and not calls          # awake: no-op
    m.sleep('test')
    calls.clear()
    assert m.ensure_awake('cmd') and m.state == sm.AWAKE
    assert 'wake:sessions' in calls


def test_battery_shutdown_only_while_asleep():
    calls = []
    m = _manager(calls)
    ran = []
    real = sm.subprocess.Popen
    sm.subprocess.Popen = lambda cmd, *a, **k: ran.append(cmd)
    try:
        assert m.battery_shutdown('Battery voltage critical') is False and not ran
        m.sleep('test')
        assert m.battery_shutdown('Battery voltage critical') is True
        assert ran == [['sudo', '-n', 'shutdown', '-h', 'now']]
    finally:
        sm.subprocess.Popen = real


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
