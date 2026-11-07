#!/usr/bin/env python3
"""LIVE: a mission scheduled 2 min out fires after a treatbot restart (2026-11-07).

Runs against the live service on this robot (restarts it with sudo):
  1. POST /tools/schedule_mission  {mission: sit, fire_at: now+120 s}  -> command_result ok
  2. sudo systemctl restart treatbot.service, wait for the API to come back
  3. POST /tools/schedule_sync until the row shows last_run_at (fired) —
     asserts |last_run_at - fire_at| <= 30 s and that it fired AFTER the restart
  4. waits for the run to end (or cancels it after --cancel-after s) and asserts
     a `mission_run` event with run_id == last_run_id landed in the relay outbox
     (data/wimz.db relay_outbox) with result success|fail|aborted matching
     schedule_sync.last_result.
  5. cleans up the schedule (cancel_mission).

Run: /home/morgan/dogbot/env_new/bin/python tests/integration/test_scheduler_restart_fires_live.py
     [--lead 120] [--cancel-after 90] [--mission sit] [--no-restart]
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from urllib import request, error

API = os.environ.get('WIMZ_API', 'http://127.0.0.1:8000')


def _wimz_db():
    """Same resolution as core.data.wimz_store (WIMZ_DATA_ROOT > /data > dogbot/data)."""
    root = os.environ.get('WIMZ_DATA_ROOT')
    if not root:
        root = '/data' if os.path.isdir('/data') and os.access('/data', os.W_OK) else '/home/morgan/dogbot/data'
    return os.path.join(root, 'wimz.db')


WIMZ_DB = _wimz_db()
FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def post(name, body=None, timeout=20):
    req = request.Request(f"{API}/tools/{name}", data=json.dumps(body or {}).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_up():
    try:
        with request.urlopen(f"{API}/tools", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def wait_api(timeout):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if api_up():
            return True
        time.sleep(2)
    return False


def sync_row(sid):
    f = post('schedule_sync')
    return next((r for r in f['result']['missions'] if r['schedule_id'] == sid), None), f


def mission_run_rows(since_ms):
    con = sqlite3.connect(f'file:{WIMZ_DB}?mode=ro', uri=True)
    try:
        rows = con.execute("SELECT payload FROM relay_outbox WHERE event_type='mission_run' AND created_at>=? "
                           "ORDER BY seq", (since_ms,)).fetchall()
    finally:
        con.close()
    return [json.loads(r[0]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lead', type=float, default=120.0, help='seconds from now to fire_at')
    ap.add_argument('--cancel-after', type=float, default=90.0,
                    help='if the run is still going this long after it fired, cancel it (proves mission_run on abort)')
    ap.add_argument('--mission', default='sit')
    ap.add_argument('--no-restart', action='store_true')
    a = ap.parse_args()

    check(api_up(), f"API reachable at {API}")
    if FAILS:
        sys.exit(1)
    pre = post('schedule_sync')
    check(pre.get('ok') and pre['result']['scheduler']['enabled'], "scheduler enabled before the test")

    t_sched = time.time()
    fire_at = datetime.now().astimezone() + timedelta(seconds=a.lead)
    f = post('schedule_mission', {'mission': a.mission, 'dog_id': 'livetest', 'fire_at': fire_at.isoformat(),
                                  'mission_id': f"livetest-{int(t_sched)}", 'name': 'restart-fire test'})
    check(f.get('type') == 'command_result' and f.get('ok') is True, f"schedule_mission -> command_result ok ({f.get('error')})")
    if not f.get('ok'):
        sys.exit(1)
    sid = f['result']['schedule_id']
    nfa = f['result']['next_fire_at']
    check(nfa and abs(datetime.fromisoformat(nfa).timestamp() - fire_at.timestamp()) < 1,
          f"next_fire_at == fire_at ({nfa})")
    print(f"schedule {sid} fire_at={fire_at.isoformat()}")

    restart_done_at = None
    if not a.no_restart:
        print("restarting treatbot.service ...", flush=True)
        t_r = time.time()
        subprocess.run(['sudo', 'systemctl', 'restart', 'treatbot.service'], check=True)
        time.sleep(3)
        check(wait_api(a.lead + 60), "API back after restart")
        restart_done_at = time.time()
        print(f"API back {restart_done_at - t_r:.0f}s after restart command", flush=True)
        row, _ = sync_row(sid)
        check(row is not None and row['enabled'] and row['next_fire_at'] and row['last_run_at'] is None,
              f"schedule re-armed after boot: next_fire_at={row and row['next_fire_at']}")

    # wait for the fire
    t_end = fire_at.timestamp() + 45
    row = None
    while time.time() < t_end:
        row, _ = sync_row(sid)
        if row and row.get('last_run_at'):
            break
        time.sleep(2)
    check(row is not None and row.get('last_run_at'), f"fired (last_run_at={row and row.get('last_run_at')})")
    if not (row and row.get('last_run_at')):
        post('cancel_mission', {'mission_id': sid}); sys.exit(1)
    fired_at = datetime.fromisoformat(row['last_run_at']).timestamp()
    delta = fired_at - fire_at.timestamp()
    check(abs(delta) <= 30, f"fired within 30 s of fire_at (delta {delta:+.1f}s)")
    if restart_done_at:
        check(fired_at > restart_done_at - 60, "fired by the NEW process (after restart)")
    check(row['enabled'] is False and row['next_fire_at'] is None, "one-shot disabled after firing")
    check(row['last_result'] in ('running', 'success', 'fail', 'aborted'), f"last_result={row['last_result']}")
    run_id = row.get('last_run_id')
    check(run_id is not None, f"last_run_id={run_id}")

    # wait for the run to end, cancelling if it drags on
    t_cancel = fired_at + a.cancel_after
    cancelled = False
    while True:
        row, f = sync_row(sid)
        if row['last_result'] in ('success', 'fail', 'aborted'):
            break
        if time.time() > t_cancel and not cancelled:
            c = post('cancel_mission', {'mission_id': sid})
            check(c.get('ok') is True and c['result'].get('stopped_run') is True,
                  f"cancel_mission stops the scheduled run -> command_result ok ({c.get('error')})")
            cancelled = True
            sid_deleted = True
            # schedule is gone now; read the final result from the outbox instead
            break
        time.sleep(2)
    else:
        pass

    rows = [r for r in mission_run_rows(int((fired_at - 5) * 1000))
            if str(r.get('data', {}).get('run_id')) == str(run_id)]
    t_end = time.time() + 15
    while not rows and time.time() < t_end:
        time.sleep(2)
        rows = [r for r in mission_run_rows(int((fired_at - 5) * 1000))
                if str(r.get('data', {}).get('run_id')) == str(run_id)]
    check(len(rows) == 1, f"exactly one mission_run event with run_id={run_id} in the relay outbox")
    if rows:
        d = rows[0]['data']
        check(d.get('mission_id') == a.mission and d.get('result') in ('success', 'fail', 'aborted'),
              f"mission_run {{mission_id={d.get('mission_id')}, result={d.get('result')}, rewards={d.get('rewards_given')}}}")
        if cancelled:
            check(d.get('result') == 'aborted', "cancelled run -> mission_run.result aborted")
        else:
            check(d.get('result') == row['last_result'], f"schedule_sync.last_result == mission_run.result ({row['last_result']})")

    if not cancelled:
        c = post('cancel_mission', {'mission_id': sid})
        check(c.get('ok') is True, "cleanup: cancel_mission removes the consumed schedule")
    row, _ = sync_row(sid)
    check(row is None, "schedule gone from schedule_sync")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
