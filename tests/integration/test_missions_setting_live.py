#!/usr/bin/env python3
"""LIVE: relay `missions` setting -> scheduler -> coach mode run -> mission_run (2026-11-07).

Against the running service (no restart). Pushes a settings document through
POST /settings exactly as the relay's settings_updated{key:"missions"} does:
  - one `coach` mission scheduled for the next whole minute (dog-local HH:MM,
    days mon..sun), duration_s 20, reward_treats 0
  - one `sit` mission at a far time (must show in schedule_sync, not fire)
  - one `departure` mission (must be ignored)
Then checks schedule_sync entries {id, enabled, next_fire_at}, waits for the
coach run: robot mode == coach, `active` in schedule_sync, back to idle after
duration_s, mission_run {run_id, result:success, rewards_given:0,
schedule_mission_id} in the relay outbox. Finally pushes missions: [] and
checks the set is cleared (replace semantics).

Run: /home/morgan/dogbot/env_new/bin/python tests/integration/test_missions_setting_live.py [--duration 20]
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from urllib import request

API = os.environ.get('WIMZ_API', 'http://127.0.0.1:8000')
DOG = 'livetest-dog'
FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def http(method, path, body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    req = request.Request(f"{API}{path}", data=data, headers={'Content-Type': 'application/json'}, method=method)
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wimz_db():
    root = os.environ.get('WIMZ_DATA_ROOT') or ('/data' if os.path.isdir('/data') and os.access('/data', os.W_OK)
                                                else '/home/morgan/dogbot/data')
    return os.path.join(root, 'wimz.db')


def mission_runs(since_ms):
    con = sqlite3.connect(f"file:{wimz_db()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT payload FROM relay_outbox WHERE event_type='mission_run' AND created_at>=? ORDER BY seq",
                           (since_ms,)).fetchall()
    finally:
        con.close()
    return [json.loads(r[0]) for r in rows]


def sync():
    return http('POST', '/tools/schedule_sync', {'dog_id': DOG})['result']


def mode():
    return http('GET', '/mode').get('current_mode')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=int, default=20)
    a = ap.parse_args()

    # next whole minute at least 20 s away (the slot is HH:MM:00)
    now = datetime.now().astimezone()
    slot = (now + timedelta(seconds=80)).replace(second=0, microsecond=0)
    hhmm = slot.strftime('%H:%M')
    far = (now + timedelta(hours=5)).strftime('%H:%M')
    doc = {"missions": {"missions": [
        {"id": "lt-coach", "type": "coach", "name": "LT coach", "enabled": True,
         "trigger": {"kind": "schedule", "times": [hhmm], "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]},
         "duration_s": a.duration, "reward_treats": 0},
        {"id": "lt-sit", "type": "sit", "name": "LT sit", "enabled": True,
         "trigger": {"kind": "schedule", "times": [far]}, "duration_s": 120, "reward_treats": 1},
        {"id": "lt-dep", "type": "recall", "name": "LT departure", "enabled": True,
         "trigger": {"kind": "departure"}, "duration_s": 120, "reward_treats": 1},
    ]}}
    t_push = time.time()
    # POST /settings applies via RemoteSettings.apply(source=local_api) — same code path as the relay push,
    # but with no dog_id, so pass it inside the document the way the relay's GET wraps keys.
    res = http('POST', '/settings', {"missions": {"key": "missions", "value": doc["missions"], "version": 99}})
    check('missions' in json.dumps(res), f"POST /settings missions applied ({str(res)[:160]})")

    s = sync()
    # the local API has no dog_id; the loader tags it 'unknown' — look across all dogs
    allrows = http('POST', '/tools/schedule_sync', {})['result']['missions']
    rows = {r['id']: r for r in allrows if r['id'].startswith('lt-')}
    check(set(rows) == {'lt-coach', 'lt-sit'}, f"schedule_sync lists the two schedule missions, not the departure one ({sorted(rows)})")
    rc = rows.get('lt-coach', {})
    check(rc.get('enabled') is True and rc.get('next_fire_at') and
          abs(datetime.fromisoformat(rc['next_fire_at']).timestamp() - slot.timestamp()) < 1,
          f"lt-coach next_fire_at == {hhmm} today in robot offset ({rc.get('next_fire_at')})")
    check(datetime.fromisoformat(rc['next_fire_at']).utcoffset() == now.utcoffset(), "next_fire_at carries the robot's UTC offset")
    check(rc.get('kind') == 'mode' and rc.get('mode') == 'coach' and rc.get('reward_treats') == 0, "coach -> mode schedule, no treats")
    check(rows.get('lt-sit', {}).get('next_fire_at') is not None, "lt-sit armed for its far time")

    # wait for the fire
    print(f"waiting for {hhmm} ...", flush=True)
    t_end = slot.timestamp() + 40
    active = None
    while time.time() < t_end:
        r = http('POST', '/tools/schedule_sync', {})['result']
        if r.get('active') and r['active'].get('id') == 'lt-coach':
            active = r['active']
            break
        time.sleep(2)
    check(active is not None, f"schedule_sync.active shows lt-coach ({active})")
    if active is None:
        http('POST', '/settings', {"missions": {"key": "missions", "value": {"missions": []}, "version": 100}})
        sys.exit(1)
    run_id = active['run_id']
    m = mode()
    check(m == 'coach', f"robot mode is coach while the run executes (mode={m})")
    fired_at = time.time()

    # wait for the end
    t_end = slot.timestamp() + a.duration + 45
    while time.time() < t_end:
        r = http('POST', '/tools/schedule_sync', {})['result']
        if not r.get('active'):
            break
        time.sleep(2)
    r = http('POST', '/tools/schedule_sync', {})['result']
    check(not r.get('active'), "run ended: schedule_sync.active is null")
    time.sleep(2)
    m = mode()
    check(m == 'idle', f"robot back to idle after duration_s (mode={m})")
    row = next((x for x in r['missions'] if x['id'] == 'lt-coach'), {})
    check(row.get('last_result') == 'success' and str(row.get('last_run_id')) == str(run_id),
          f"lt-coach last_result=success last_run_id={row.get('last_run_id')}")
    check(row.get('next_fire_at') and datetime.fromisoformat(row['next_fire_at']) > now + timedelta(hours=23),
          f"daily coach re-armed for tomorrow ({row.get('next_fire_at')})")

    runs = [x for x in mission_runs(int((fired_at - 60) * 1000)) if str(x.get('data', {}).get('run_id')) == str(run_id)]
    t_end = time.time() + 15
    while not runs and time.time() < t_end:
        time.sleep(2)
        runs = [x for x in mission_runs(int((fired_at - 60) * 1000)) if str(x.get('data', {}).get('run_id')) == str(run_id)]
    check(len(runs) == 1, f"one mission_run event for run_id={run_id}")
    if runs:
        d = runs[0]['data']
        check(d.get('mission_id') == 'coach' and d.get('result') == 'success' and d.get('rewards_given') == 0
              and d.get('schedule_mission_id') == 'lt-coach' and d.get('kind') == 'mode',
              f"mission_run {{mission_id={d.get('mission_id')}, result={d.get('result')}, rewards={d.get('rewards_given')}, "
              f"schedule_mission_id={d.get('schedule_mission_id')}}}")

    # replace with empty set
    http('POST', '/settings', {"missions": {"key": "missions", "value": {"missions": []}, "version": 100}})
    left = [x for x in http('POST', '/tools/schedule_sync', {})['result']['missions'] if x['id'].startswith('lt-')]
    check(not left, "missions: [] clears the set (replace semantics)")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
