#!/usr/bin/env python3
"""LIVE: coach_session + trick_attempt relay events (2026-11-08).

Against the running treatbot.service (new code must already be loaded):
  1. POST /mode/set coach, hold --coach-hold s, POST /mode/set idle
     -> exactly one `coach_session` row {start, end, cues_given, successes,
        treats} in the relay outbox for that activation (zero counts if no
        dog turned up; any coach cues given meanwhile appear as
        `trick_attempt` source=coach and are counted in it).
  2. POST /tools/run_trick {cue, retry:false, timeout_s}
     -> one `trick_attempt` source=run_trick whose result/confidence match
        the tool reply, and NO additional coach_session (COACH was entered
        only for the tool with the engine held).

Plays cue audio on the robot and dispenses a treat if the dog performs.

Run: /home/morgan/dogbot/env_new/bin/python tests/integration/test_coach_events_live.py
     [--coach-hold 15] [--cue sit] [--trick-timeout 20] [--skip-trick]
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request

API = os.environ.get('WIMZ_API', 'http://127.0.0.1:8000')


def _wimz_db():
    root = os.environ.get('WIMZ_DATA_ROOT')
    if not root:
        root = '/data' if os.path.isdir('/data') and os.access('/data', os.W_OK) else '/home/morgan/dogbot/data'
    return os.path.join(root, 'wimz.db')


WIMZ_DB = _wimz_db()
FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


def post(path, body=None, timeout=90):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(API + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or '{}')


def get(path, timeout=10):
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read().decode() or '{}')


def rows(event_type, since_ms):
    con = sqlite3.connect(f'file:{WIMZ_DB}?mode=ro', uri=True)
    try:
        out = con.execute("SELECT payload FROM relay_outbox WHERE event_type=? AND created_at>=? ORDER BY seq",
                          (event_type, since_ms)).fetchall()
    finally:
        con.close()
    return [json.loads(r[0]) for r in out]


def wait_rows(event_type, since_ms, n=1, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        r = rows(event_type, since_ms)
        if len(r) >= n:
            return r
        time.sleep(0.5)
    return rows(event_type, since_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coach-hold', type=float, default=15.0)
    ap.add_argument('--cue', default='sit')
    ap.add_argument('--trick-timeout', type=float, default=20.0)
    ap.add_argument('--skip-trick', action='store_true')
    a = ap.parse_args()

    print(f"API={API} DB={WIMZ_DB}")
    mode0 = get('/mode/get').get('mode')
    check(mode0 == 'idle', f"robot idle before the test (mode={mode0})")

    # ---- 1. coach activation
    t0 = int(time.time() * 1000) - 1000
    r = post('/mode/set', {'mode': 'coach'})
    check(r.get('success') is True, f"POST /mode/set coach -> {r}")
    time.sleep(a.coach_hold)
    r = post('/mode/set', {'mode': 'idle'})
    check(r.get('success') is True, f"POST /mode/set idle -> {r}")
    cs = wait_rows('coach_session', t0, n=1, timeout=20)
    check(len(cs) == 1, f"exactly one coach_session row for the activation ({len(cs)})")
    if cs:
        d = cs[0].get('data', {})
        print("   coach_session:", json.dumps(d))
        check(all(k in d for k in ('start', 'end', 'cues_given', 'successes', 'treats')),
              "coach_session carries start, end, cues_given, successes, treats")
        check(isinstance(d.get('duration_s'), (int, float)) and a.coach_hold - 3 <= d['duration_s'] <= a.coach_hold + 10,
              f"coach_session duration ~{a.coach_hold}s (got {d.get('duration_s')})")
        coach_cues = [x for x in rows('trick_attempt', t0) if x.get('data', {}).get('source') == 'coach']
        check(d['cues_given'] == len(coach_cues),
              f"cues_given ({d['cues_given']}) == coach trick_attempt rows in the window ({len(coach_cues)})")
        check(d['successes'] == sum(1 for x in coach_cues if x['data'].get('result') == 'success'),
              "successes == coach trick_attempt rows with result=success")
        check(d['successes'] <= d['cues_given'] and d['treats'] <= d['successes'],
              "treats <= successes <= cues_given")
    mc = rows('mode_change', t0)
    check(any(m.get('data', {}).get('mode') == 'coach' for m in mc),
          f"mode_change row for the coach stay still emitted ({[m['data']['mode'] for m in mc]})")

    if a.skip_trick:
        return

    # ---- 2. run_trick from idle (settle first so part 1's rows are outside the window)
    time.sleep(2.0)
    t1 = int(time.time() * 1000)
    reply = post('/tools/run_trick', {'cue': a.cue, 'retry': False, 'timeout_s': a.trick_timeout},
                 timeout=a.trick_timeout + 60)
    res = reply.get('result') or {}
    print("   run_trick reply:", json.dumps({k: res.get(k) for k in ('result', 'confidence', 'attempts',
                                                                   'behavior_detected', 'treat_dispensed',
                                                                   'mode_restored')}))
    ta = wait_rows('trick_attempt', t1, n=1, timeout=20)
    ta = [x for x in ta if x.get('data', {}).get('source') == 'run_trick']
    check(len(ta) == 1, f"exactly one trick_attempt source=run_trick ({len(ta)})")
    if ta:
        d = ta[0]['data']
        print("   trick_attempt:", json.dumps(d))
        check(d.get('cue') == a.cue, f"trick_attempt.cue == {a.cue}")
        check(d.get('result') in ('success', 'no_dog', 'wrong_pose', 'timeout'), f"result on contract ({d.get('result')})")
        check(d.get('result') == res.get('result'), f"trick_attempt.result == tool reply result ({res.get('result')})")
        conf_ok = (d.get('confidence') is None and res.get('confidence') is None) or (
            d.get('confidence') is not None and res.get('confidence') is not None
            and abs(d['confidence'] - res['confidence']) < 0.01)
        check(conf_ok, f"confidence matches tool reply ({d.get('confidence')} vs {res.get('confidence')})")
        check(d.get('attempt') == res.get('attempts'), "attempt == tool attempts")
    time.sleep(3)
    check(len(rows('coach_session', t1)) == 0, "run_trick's COACH entry produced no coach_session row")
    check(get('/mode/get').get('mode') == 'idle', "robot back to idle after run_trick")


if __name__ == '__main__':
    try:
        main()
    finally:
        print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
