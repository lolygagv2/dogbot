#!/usr/bin/env python3
"""RemoteSettings apply/persist (CONTRACT v2 FINAL, 2026-11-06).

Checks the contract shapes land on the right seams: treat_schedule ->
TreatPolicy (daily_cap/default_multiplier/max_per_hour/windows),
sg_sensitivity {level, min_episode_s, escalate_after_s} -> SG
bark_detection profile + bark tracker window + bark_episode_min_s,
music_plan / quiet_hours / notify_rules validated and stored, single-key
settings_updated apply with version, persistence + load_cached, bad input
rejected without poisoning the rest.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_remote_settings_apply.py
Standalone (no pytest).
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='rstest_')

# Fake SG + behavior-events modules (no hardware)
sg_mod = types.ModuleType('modes.silent_guardian')
class _SG:
    config = {'bark_detection': {'threshold': 2, 'loudness_threshold_db': -28, 'confidence_minimum': 0.35}}
_sg = _SG()
sg_mod.get_silent_guardian_mode = lambda: _sg
sys.modules['modes.silent_guardian'] = sg_mod
be_mod = types.ModuleType('services.cloud.behavior_events')
class _BE:
    config = {'bark_episode_min_s': 0.0}
_be = _BE()
be_mod.get_behavior_event_service = lambda: _be
sys.modules['services.cloud.behavior_events'] = be_mod

from services.cloud.remote_settings import RemoteSettings  # noqa: E402
from services.reward.treat_policy import get_treat_policy  # noqa: E402
from core.bark_frequency_tracker import get_bark_frequency_tracker  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakeStore:
    def __init__(self):
        self.kv = {}
    def set_setting(self, k, v):
        self.kv[k] = v
    def get_setting(self, k, default=None):
        return self.kv.get(k, default)


def main():
    tp = get_treat_policy()
    tp._get_dispenser = lambda: None
    store = FakeStore()
    rs = RemoteSettings()
    rs._store = store

    doc = {
        'treat_schedule': {'daily_cap': 12, 'default_multiplier': 0.5, 'max_per_hour': 3,
                           'windows': [{'start': '22:00', 'end': '06:00', 'multiplier': 0}]},
        'sg_sensitivity': {'level': 'high', 'min_episode_s': 4, 'escalate_after_s': 45},
        'music_plan': {'enabled': True, 'volume': 40,
                       'on_departure': {'track': 'songs/default/bach.mp3', 'delay_s': 120, 'duration_min': 20},
                       'schedule': []},
        'quiet_hours': {'start': '22:00', 'end': '07:00'},
        'notify_rules': {'rules': [{'id': 'bark_long', 'enabled': True, 'event_type': 'bark_episode',
                                    'when': {'min_duration_s': 60.0, 'severity_in': ['med', 'high']},
                                    'cooldown_s': 900, 'attach': 'still', 'app_type': 'bark'}]},
        'future_key': {'x': 1},
    }
    res = rs.apply(doc, source='relay_get', dog_id='dog-1')
    check(not res['rejected'] and set(res['applied']) == set(doc), f"full document applied ({res})")
    eff = tp.get_settings()
    check(eff['daily_cap'] == 12 and eff['default_multiplier'] == 0.5 and eff['max_per_hour'] == 3
          and eff['windows'][0]['multiplier'] == 0, f"treat_schedule landed in TreatPolicy ({eff})")
    bd = _sg.config['bark_detection']
    check(bd['threshold'] == 1 and bd['loudness_threshold_db'] == -32 and bd['profile'] == 'sensitive',
          f"level=high -> sensitive profile ({bd})")
    check(bd['window_seconds'] == 45 and get_bark_frequency_tracker().window_seconds == 45,
          "escalate_after_s -> bark window + tracker")
    check(_be.config['bark_episode_min_s'] == 4.0, "min_episode_s -> bark_episode_min_s")
    snap = rs.get()
    check(snap['music_plan']['on_departure']['duration_min'] == 20 and snap['quiet_hours']['end'] == '07:00'
          and snap['notify_rules'] and snap['future_key'] == {'x': 1}, "music_plan/quiet_hours/notify_rules/unknown stored")
    check(rs.get_notify_rules()[0]['id'] == 'bark_long', "notify_rules {rules: [...]} unwrapped by getter")
    check(snap['_meta']['dog_id'] == 'dog-1' and snap['_meta']['source'] == 'relay_get', "meta records dog and source")
    check('remote_settings' in store.kv and json.loads(store.kv['remote_settings'])['settings']['treat_schedule']['daily_cap'] == 12,
          "persisted to store")

    # single-key push with version
    res = rs.apply({'sg_sensitivity': {'level': 'low'}}, source='relay_push', dog_id='dog-1', version=7)
    check(res['applied'] == ['sg_sensitivity'] and _sg.config['bark_detection']['threshold'] == 4
          and rs.get()['_versions']['sg_sensitivity'] == 7, "settings_updated single key applied with version")

    # bad input rejected, good keys still applied
    res = rs.apply({'treat_schedule': {'windows': [{'start': '9:99', 'end': '10:00'}]},
                    'sg_sensitivity': {'level': 'extreme'},
                    'music_plan': {'volume': 150},
                    'quiet_hours': 'nope',
                    'notify_rules': [1]}, source='relay_push')
    check(set(res['rejected']) == {'treat_schedule', 'sg_sensitivity', 'music_plan', 'quiet_hours'}
          and res['applied'] == ['notify_rules'], f"invalid values rejected individually ({res})")
    res = rs.apply({'notify_rules': {'rules': 'nope'}}, source='relay_push')
    check('notify_rules' in res['rejected'] and rs.get_notify_rules() == [1],
          f"notify_rules with non-list rules rejected, previous kept ({res})")
    check(tp.get_settings()['daily_cap'] == 12, "rejected treat_schedule left the previous policy intact")

    # load_cached in a fresh instance
    rs2 = RemoteSettings()
    rs2._store = store
    ok = rs2.load_cached()
    check(ok and rs2.get()['treat_schedule']['daily_cap'] == 12 and rs2.get()['_versions'].get('sg_sensitivity') == 7,
          "load_cached restores settings + versions")
    rs3 = RemoteSettings()
    rs3._store = FakeStore()
    check(rs3.load_cached() is False, "load_cached with empty store returns False")

    # legacy flat daily_cap still accepted
    res = rs.apply({'daily_cap': 20}, source='local_ws')
    check(res['applied'] == ['daily_cap'] and tp.get_settings()['daily_cap'] == 20, "legacy flat daily_cap")

    check(rs.apply('nope') == {'applied': [], 'rejected': {'_': 'settings must be an object'}}, "non-object rejected")


if __name__ == '__main__':
    main()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
