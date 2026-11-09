#!/usr/bin/env python3
"""Treat cap from ANY source (2026-11-08).

The robot used to take the daily treat cap only from the relay's
treat_schedule settings document. Now the app's sg_config slider, the
local API (/treat/limit), the connector and the relay all land on one
setter, RemoteSettings.set_treat_limits(), which applies locally, persists,
and writes back to the relay (PUT /api/dogs/{dog}/settings/treat_schedule).

Checks:
  * extract_treat_limits() finds the cap under the loosely-named fields the
    app / relay may use (top level or nested), ignores junk, keeps partials
  * set_treat_limits() merges into the current treat_schedule, updates the
    TreatPolicy + dispenser, persists, and starts the upstream PUT
  * while the PUT is pending, a stale relay push with the OLD value is
    deferred (local wins); an equal relay push clears the pending state;
    an expired pending falls back to the relay value
  * a 2xx PUT clears pending; 404 retries then is superseded by a newer write
  * daily_cap 0 is accepted and blocks dispensing (dispenser limit 0)

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_treat_limit_any_source.py
Standalone (no pytest, no hardware).
"""
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='tltest_')

# --- fakes: SG, behavior events, dispenser, relay client -------------------
sg_mod = types.ModuleType('modes.silent_guardian')
class _SG:
    config = {'bark_detection': {}}
sg_mod.get_silent_guardian_mode = lambda: _SG()
sys.modules['modes.silent_guardian'] = sg_mod
be_mod = types.ModuleType('services.cloud.behavior_events')
class _BE:
    config = {'bark_episode_min_s': 0.0}
be_mod.get_behavior_event_service = lambda: _BE()
sys.modules['services.cloud.behavior_events'] = be_mod

class FakeDispenser:
    def __init__(self):
        self.daily_limit = 300
        self.treats_dispensed_today = 0
    def set_daily_limit(self, limit):
        self.daily_limit = max(0, min(300, int(limit)))
    def reset_daily_counters(self):
        self.treats_dispensed_today = 0
fake_disp = FakeDispenser()
disp_mod = types.ModuleType('services.reward.dispenser')
disp_mod.get_dispenser_service = lambda: fake_disp
sys.modules['services.reward.dispenser'] = disp_mod

class FakeRelay:
    """Scripted PUT responses: pop from `responses`, record calls."""
    def __init__(self):
        self.calls = []
        self.responses = []
        self.ev = threading.Event()
    def http_sync(self, method, path, *, json_body=None, data=None, timeout=15.0):
        self.calls.append((method, path, json_body))
        self.ev.set()
        if self.responses:
            return self.responses.pop(0)
        return 200, {'ok': True}
fake_relay = FakeRelay()
rc_mod = types.ModuleType('services.cloud.relay_client')
rc_mod.get_relay_client = lambda *a, **k: fake_relay
rc_mod.RelayClient = FakeRelay  # services.cloud package re-exports it
sys.modules['services.cloud.relay_client'] = rc_mod

import services.cloud.remote_settings as rs_mod  # noqa: E402
from services.cloud.remote_settings import RemoteSettings  # noqa: E402
from services.reward.treat_policy import (extract_treat_limits,  # noqa: E402
                                          get_treat_policy)

rs_mod.UPSTREAM_BACKOFF_S = (0.05, 0.05, 0.05)   # fast retries for the test

results = []
def check(name, cond, detail=''):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f" ({detail})" if detail else ''))

def wait_for(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return pred()

DOG = '019f58ad-03c9-7aff-adae-b697fb058ffc'
tp = get_treat_policy()

# ---------------------------------------------------------------- sniffing
check('extract: canonical daily_cap', extract_treat_limits({'daily_cap': 12}) == {'daily_cap': 12, 'matched': ['daily_cap']})
check('extract: app-style max_treats_per_day', extract_treat_limits({'bark_threshold': -28, 'max_treats_per_day': '15'}).get('daily_cap') == 15)
check('extract: nested treat_schedule + hourly',
      extract_treat_limits({'treat_schedule': {'daily_cap': 7, 'max_per_hour': 2}}) == {'daily_cap': 7, 'max_per_hour': 2, 'matched': ['daily_cap', 'max_per_hour']})
check('extract: hourly alias only (partial)', extract_treat_limits({'treats_per_hour': 4}) == {'max_per_hour': 4, 'matched': ['treats_per_hour']})
check('extract: junk ignored', extract_treat_limits({'daily_cap': 'lots', 'fast_escalation_bpm': 0}) == {})
check('extract: non-dict', extract_treat_limits(None) == {} and extract_treat_limits([1]) == {})

# ---------------------------------------------------------------- baseline relay doc
rs = RemoteSettings()
r = rs.apply({'treat_schedule': {'daily_cap': 10, 'default_multiplier': 1.0, 'max_per_hour': 3, 'windows': []}},
             source='relay_get', dog_id=DOG)
check('relay doc applied (cap 10)', 'treat_schedule' in r['applied'] and tp.get_settings()['daily_cap'] == 10 and fake_disp.daily_limit == 10)
check('dog id tracked', rs.get_treat_sync_status()['dog_ids'] == [DOG])

# ---------------------------------------------------------------- local set -> policy, dispenser, upstream PUT
fake_relay.calls.clear(); fake_relay.ev.clear()
res = rs.set_treat_limits(daily_cap=25, source='local_api')
check('set_treat_limits applied', 'treat_schedule' in res['applied'] and res['upstream'] == 'pending', str(res))
check('policy + dispenser updated to 25', tp.get_settings()['daily_cap'] == 25 and fake_disp.daily_limit == 25)
check('other fields kept (max_per_hour 3)', res['treat_schedule']['max_per_hour'] == 3 and tp.get_settings()['max_per_hour'] == 3)
check('persisted locally', rs.get()['settings']['treat_schedule']['daily_cap'] == 25 if 'settings' in rs.get() else rs._settings['treat_schedule']['daily_cap'] == 25)
got = wait_for(lambda: len(fake_relay.calls) >= 1)
check('upstream PUT sent', got and fake_relay.calls[0][0] == 'PUT'
      and fake_relay.calls[0][1] == f'/api/dogs/{DOG}/settings/treat_schedule'
      and fake_relay.calls[0][2]['value']['daily_cap'] == 25, str(fake_relay.calls[:1]))
check('2xx clears pending', wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None))

# ---------------------------------------------------------------- no-op on same value
res = rs.set_treat_limits(daily_cap=25, source='local_api')
check('same value -> no_change, no PUT', res['upstream'] == 'no_change' and len(fake_relay.calls) == 1)

# ---------------------------------------------------------------- account-wide: two dogs -> ONE PUT (relay fans out)
DOG2 = '019f58ad-0000-7aff-adae-000000000002'
rs.apply({'treat_schedule': {'daily_cap': 25, 'default_multiplier': 1.0, 'max_per_hour': 3, 'windows': []}},
         source='relay_push', dog_id=DOG2)
check('second dog tracked', rs.get_treat_sync_status()['dog_ids'] == sorted([DOG, DOG2]))
fake_relay.calls.clear(); fake_relay.ev.clear()
res = rs.set_treat_limits(daily_cap=30, source='local_api')
check('two-dog write pending', res['upstream'] == 'pending', str(res))
check('2xx clears pending (two dogs)', wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None))
time.sleep(0.2)
check('account-wide: exactly one PUT for two dogs', len(fake_relay.calls) == 1, str(fake_relay.calls))
# first dog fails -> falls back to the second, still one accepted write
fake_relay.calls.clear(); fake_relay.ev.clear()
fake_relay.responses = [(500, 'boom')]
res = rs.set_treat_limits(daily_cap=25, source='local_api')
check('fallback clears pending', wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None))
time.sleep(0.2)
check('account-wide: failed dog then other dog -> 2 calls, stops after accept',
      len(fake_relay.calls) == 2 and fake_relay.calls[0][1] != fake_relay.calls[1][1], str(fake_relay.calls))

# ---------------------------------------------------------------- stale relay push while pending (404 relay)
fake_relay.calls.clear()
fake_relay.responses = [(404, None)] * 10     # relay without the PUT endpoint
res = rs.set_treat_limits(daily_cap=40, source='app_sg_config')
check('pending while relay 404s', rs.get_treat_sync_status()['pending_local'] is not None)
r = rs.apply({'treat_schedule': {'daily_cap': 10, 'default_multiplier': 1.0, 'max_per_hour': 3, 'windows': []}},
             source='relay_push', dog_id=DOG)
check('stale relay push (10) deferred, local 40 kept',
      'treat_schedule' in r['rejected'] and tp.get_settings()['daily_cap'] == 40 and fake_disp.daily_limit == 40, str(r))
r = rs.apply({'treat_schedule': {'daily_cap': 40, 'default_multiplier': 1.0, 'max_per_hour': 3, 'windows': []}},
             source='relay_push', dog_id=DOG)
check('equal relay push acks: applied + pending cleared',
      'treat_schedule' in r['applied'] and rs.get_treat_sync_status()['pending_local'] is None)
wait_for(lambda: not (rs._upstream_thread and rs._upstream_thread.is_alive()), 2.0)
check('retry thread stopped after ack', not (rs._upstream_thread and rs._upstream_thread.is_alive()))

# ---------------------------------------------------------------- expired pending -> relay regains authority
fake_relay.responses = [(500, 'boom')] * 10
rs.set_treat_limits(daily_cap=33, source='local_api')
with rs._lock:
    rs._pending_local['since'] -= (rs_mod.LOCAL_OVERRIDE_TTL_S + 1)
r = rs.apply({'treat_schedule': {'daily_cap': 10, 'default_multiplier': 1.0, 'max_per_hour': 3, 'windows': []}},
             source='relay_get', dog_id=DOG)
check('expired pending -> relay value accepted', 'treat_schedule' in r['applied'] and tp.get_settings()['daily_cap'] == 10)
wait_for(lambda: not (rs._upstream_thread and rs._upstream_thread.is_alive()), 2.0)

# ---------------------------------------------------------------- newer local write supersedes in-flight retry
fake_relay.calls.clear()
fake_relay.responses = [(404, None)] * 3 + [(200, {})] * 5
rs.set_treat_limits(daily_cap=50, source='local_api')
time.sleep(0.02)
rs.set_treat_limits(daily_cap=60, source='local_api')
wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None, 3.0)
check('newer write wins; relay ends with 60',
      rs.get_treat_sync_status()['pending_local'] is None and fake_relay.calls
      and fake_relay.calls[-1][2]['value']['daily_cap'] == 60 and tp.get_settings()['daily_cap'] == 60,
      f"last PUT {fake_relay.calls[-1][2] if fake_relay.calls else None}")

# ---------------------------------------------------------------- cap 0 and None
fake_relay.responses = []
res = rs.set_treat_limits(daily_cap=0, source='local_api')
check('cap 0 accepted -> dispenser limit 0 (block all)', 'treat_schedule' in res['applied'] and fake_disp.daily_limit == 0)
check('policy check blocks at cap 0', tp.check('cloud_command', treats_today=0).allowed is False
      if hasattr(tp.check('cloud_command', treats_today=0), 'allowed') else tp.check('cloud_command', treats_today=0)[0] is False)
res = rs.set_treat_limits(daily_cap=None, source='local_api')
check('cap None -> no cap, dispenser back to 300', tp.get_settings()['daily_cap'] is None and fake_disp.daily_limit == 300)
try:
    rs.set_treat_limits(daily_cap=-1, source='local_api')
    r = rs._settings['treat_schedule']['daily_cap']
    check('negative cap rejected', r is None)
except ValueError:
    check('negative cap rejected', True)

# ---------------------------------------------------------------- hourly only (partial)
rs.set_treat_limits(daily_cap=20, source='local_api')
wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None, 2.0)
res = rs.set_treat_limits(max_per_hour=5, source='local_api')
check('hourly-only write keeps cap 20', tp.get_settings()['max_per_hour'] == 5 and tp.get_settings()['daily_cap'] == 20)
wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None, 2.0)

# ---------------------------------------------------------------- daily_cap_enabled (relay brief 2026-11-09 §1)
r = rs.apply({'treat_schedule': {'daily_cap': 65, 'daily_cap_enabled': True, 'max_per_hour': 0,
                                 'default_multiplier': 1.0, 'windows': []}}, source='relay_push', dog_id=DOG)
check('relay doc with daily_cap_enabled true -> cap 65, per_hour 0 = off',
      tp.get_settings()['daily_cap'] == 65 and fake_disp.daily_limit == 65 and tp.get_settings()['max_per_hour'] in (0, None))
check('max_per_hour 0 does not block protocol treats',
      tp.check('silent_guardian_reward', treats_today=0).allowed is True, str(tp.check('silent_guardian_reward', treats_today=0)))
r = rs.apply({'treat_schedule': {'daily_cap': 65, 'daily_cap_enabled': False, 'max_per_hour': 0,
                                 'default_multiplier': 1.0, 'windows': []}}, source='relay_push', dog_id=DOG)
check('daily_cap_enabled false -> unlimited (policy None, dispenser 300)',
      tp.get_settings()['daily_cap'] is None and fake_disp.daily_limit == 300 and rs._settings['treat_schedule']['daily_cap'] == 65)
check('unlimited: 70th treat allowed', tp.check('cloud_command', treats_today=70).allowed is True)
res = rs.set_treat_limits(daily_cap_enabled=True, source='local_api')
check('local re-enable keeps parked cap 65', tp.get_settings()['daily_cap'] == 65 and res['treat_schedule']['daily_cap_enabled'] is True)
check('71st treat refused at cap 65', tp.check('cloud_command', treats_today=70).allowed is False
      and tp.check('cloud_command', treats_today=70).code == 'daily_cap')
wait_for(lambda: rs.get_treat_sync_status()['pending_local'] is None, 2.0)
check('enabled-only write sent full value upstream',
      fake_relay.calls[-1][2]['value'].get('daily_cap_enabled') is True and fake_relay.calls[-1][2]['value']['daily_cap'] == 65)

# ---------------------------------------------------------------- wire codes + reset time
from services.reward.treat_policy import wire_code, TreatPolicy
check('wire codes', (wire_code('daily_cap'), wire_code('cooldown'), wire_code('max_per_hour'), wire_code('window_blocked'))
      == ('treat_cap_reached', 'treat_rate', 'treat_rate', 'treat_window_closed'))
from services.cloud.remote_settings import get_remote_settings
get_remote_settings().apply({'timezone': 'America/New_York'}, source='relay_push', dog_id=DOG)
ra = TreatPolicy.resets_at_iso()
check('resets_at is next local midnight in dog tz', ra.endswith('00:00:00-05:00') or ra.endswith('00:00:00-04:00'), ra)
check('seconds_until_midnight sane', 0 < TreatPolicy.seconds_until_midnight() <= 86400)

print()
print('ALL PASS' if all(results) else f'{results.count(False)} FAILED')
sys.exit(0 if all(results) else 1)
