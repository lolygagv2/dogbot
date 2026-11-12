#!/usr/bin/env python3
"""ultrasonic.burst() safety logic — no hardware, no sound.

The amp switch, amixer/aplay and the event store are faked. Checks: amp on -> volume
-> play -> amp off order; amp is switched off when playback crashes; refusals
(cooldown, unknown frequency) never power the amp; the cooldown floor is 20 s; the
2 kHz test tone neither obeys nor starts the cooldown; pulsed episodes; Silent
Guardian's three styles only play with the flag on, with per-dog overrides.

Run: /home/morgan/dogbot/env_new/bin/python tests/hardware/test_ultrasonic_burst.py
Standalone (no pytest).
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ultrasonic as u  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


def main():
    calls, logged = [], []
    boom = {'on': False}

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        if boom['on'] and cmd[0] == 'aplay':
            raise RuntimeError('aplay crashed')

    u._set_amp = lambda on: calls.append('amp_on' if on else 'amp_off')
    u.subprocess.run = fake_run
    u._log_burst = lambda payload, *a: logged.append(payload)
    u.time.sleep = lambda s: None
    u.STATE_FILE = Path(tempfile.mkdtemp(prefix='ultratest_')) / 'ultrasonic_last'
    u.ULTRA_CARD = 'card0'   # any card that exists, so the presence check passes

    res = u.burst(25000, 40)
    check(res['ok'] and calls == ['amp_on', 'amixer', 'aplay', 'amp_off'],
          f'order amp on -> volume -> play -> amp off ({calls})')
    check(logged and logged[-1]['freq_hz'] == 25000 and logged[-1]['level_pct'] == 40,
          'burst logged with freq and level')

    calls.clear()
    res = u.burst(25000, 40, cooldown_s=0)
    check(not res['ok'] and 'cooldown' in res['reason'] and calls == [],
          'second burst refused: cooldown floor is 20 s even when asked for 0, amp untouched')

    calls.clear()
    res = u.burst(u.TEST_FREQ, 40)
    check(res['ok'] and calls[-1] == 'amp_off', '2 kHz test tone ignores the cooldown')

    calls.clear()
    res = u.burst(12345, 40)
    check(not res['ok'] and calls == [], 'unknown frequency refused, amp untouched')

    old = os.path.getmtime(u.STATE_FILE) - 60
    os.utime(u.STATE_FILE, (old, old))
    calls.clear()
    boom['on'] = True
    res = u.burst(32000, 150)
    check(not res['ok'] and calls[0] == 'amp_on' and calls[-1] == 'amp_off',
          f'amp switched off when playback crashes ({calls})')
    check(logged[-1]['ok'] is False and logged[-1]['level_pct'] == 100,
          'failed burst logged; level clamped to 100')
    boom['on'] = False

    # --- pulsed episode: one aplay call, PCM = alternating pulses with gaps
    old = os.path.getmtime(u.STATE_FILE) - 60
    os.utime(u.STATE_FILE, (old, old))
    pcm = u._pulsed_pcm([20000, 25000], 5.0, 0.3, 0.2)
    check(abs(len(pcm) / 2 / u.SAMPLE_RATE - 4.8) < 0.01, '5 s pulsed episode = 10 pulses of 0.3 s (4.8 s of audio)')
    calls.clear()
    res = u.burst(20000, 80, duration_s=5, pulse_s=0.3, gap_s=0.2, alt_freq_hz=25000)
    check(res['ok'] and res['freqs'] == [20000, 25000] and calls == ['amp_on', 'amixer', 'aplay', 'amp_off'],
          'pulsed burst: amp on once, one playback, amp off')
    check(u.burst(25000, 40, duration_s=60)['reason'].startswith('cooldown'), 'cooldown applies after a pulsed burst')

    # --- Silent Guardian styles
    from modes.silent_guardian import SilentGuardianMode
    from config import config_loader
    raw = {}
    config_loader.get_config = lambda: types.SimpleNamespace(raw=raw)
    bursts = []
    u.burst = lambda **kw: bursts.append(kw) or {'ok': True}
    sg = SilentGuardianMode.__new__(SilentGuardianMode)
    sg.config = {'escalation': {'max_level': 4}}
    sg.current_escalation_level, sg.intervention_step = 4, 1
    sg.intervention_dog_id, sg.intervention_dog_name, sg.wimz_session_id = 'dog_1', 'Elsa', 's1'

    check(sg._ultrasonic('before_quiet') is None and bursts == [],
          'SG: nothing plays when ultrasonic_enabled is absent')
    raw['ultrasonic_enabled'] = True
    for _ in range(30):
        sg._ultrasonic('before_quiet')
    check({b['freq_hz'] for b in bursts} == {20000, 25000}
          and {b['duration_s'] for b in bursts} == {1, 2, 3}
          and all(b['level_pct'] == 80 and not b['pulse_s'] for b in bursts),
          'SG before_quiet: random mix of 20/25 kHz and 1-3 s at 80%')
    st = sg._ultrasonic('before_music')
    check(bursts[-1]['duration_s'] == 3 and bursts[-1]['level_pct'] == 60 and st['lead_s'] == 10,
          'SG before_music: 3 s at 60%, music 10 s later')
    n = len(bursts)
    sg.current_escalation_level = 3
    sg._maybe_ultrasonic_burst()
    sg.current_escalation_level, sg.intervention_step = 4, 0
    sg._maybe_ultrasonic_burst()
    check(len(bursts) == n, 'SG nonstop_barking: not below the top level, not before the music')
    sg.intervention_step = 1
    sg._maybe_ultrasonic_burst()
    b = bursts[-1]
    check(len(bursts) == n + 1 and b['duration_s'] == 5 and b['pulse_s'] == 0.3
          and b['freq_hz'] == 20000 and b['alt_freq_hz'] == 25000 and b['level_pct'] == 80,
          'SG nonstop_barking: 5 s pulsed 20/25 kHz at 80% on a bark during the music')
    raw['ultrasonic'] = {'dogs': {'Elsa': {'before_quiet': {'level_pct': 50, 'freqs': [25000]}}},
                         'before_music': {'enabled': False}}
    sg._ultrasonic('before_quiet')
    check(bursts[-1]['level_pct'] == 50 and bursts[-1]['freq_hz'] == 25000, 'SG: per-dog override of a style')
    check(sg._ultrasonic('before_music') is None, 'SG: a style can be disabled on its own')

    # --- relay tool
    import services.cloud.tool_dispatcher as td
    tool = td.ToolDispatcher._tool_ultrasonic
    raw.clear()
    check(u.play_style('before_quiet')['code'] == 'disabled', 'play_style refuses with code disabled')
    st = tool({'action': 'status'}, None)
    check(st['ok'] and st['enabled'] is False and 'nonstop_barking' in st['styles'],
          'tool: status works while off')
    check(tool({'style': 'before_quiet'}, None)['error']['code'] == 'ultrasonic_disabled',
          'tool: refused while ultrasonic_enabled is off')
    raw['ultrasonic_enabled'] = True
    n = len(bursts)
    res = tool({'style': 'nonstop_barking', 'dog_id': 'dog_1'}, None)
    check(res['ok'] and len(bursts) == n + 1 and bursts[-1]['pulse_s'] == 0.3
          and bursts[-1]['source'] == 'relay:nonstop_barking', 'tool: plays a named style')
    tool({'freq_hz': 25000, 'level_pct': 100, 'duration_s': 2}, None)
    check(bursts[-1]['level_pct'] == 80 and bursts[-1]['duration_s'] == 2.0,
          'tool: explicit burst, level capped at 80')
    u.burst = lambda **kw: {'ok': False, 'code': 'cooldown', 'retry_after_s': 12.0,
                            'reason': 'cooldown (12s left)'}
    err = tool({'freq_hz': 25000}, None)
    check(err['error']['code'] == 'cooldown' and err['retry_after_s'] == 12.0,
          'tool: cooldown reported with retry_after_s')
    check(tool({}, None)['error']['code'] == 'bad_request'
          and tool({'style': 'nope'}, None)['error']['code'] == 'bad_request',
          'tool: bad requests rejected')

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
