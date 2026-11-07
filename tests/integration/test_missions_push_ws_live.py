#!/usr/bin/env python3
"""LIVE: relay settings_updated{key:"missions"} over the WebSocket -> schedule_sync (2026-11-07).

Replays the relay's exact push frame on ws://127.0.0.1:8000/ws/local, which
routes it through the same apply_settings_updated_frame() the relay socket
uses, then sends schedule_sync 40 ms later on the SAME socket (the relay's
confirmation pattern) and reads the command_result:
  - sit / coach / silent_guardian / spin entries, times as trigger.times
    and also trigger.time (single string) and "HH:MM:SS"
  - each entry has id, enabled, next_fire_at (robot offset) and `reason`
    (spin -> unsupported reason; no time -> reason, never fires)
  - a second push without the coach mission removes it (cancellation path)
  - missions: [] clears the dog's set; the other dog's set is untouched
No restart, nothing fires (times are 5 h away).

Run: /home/morgan/dogbot/env_new/bin/python tests/integration/test_missions_push_ws_live.py
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta

import websockets

WS = os.environ.get('WIMZ_WS', 'ws://127.0.0.1:8000/ws/local')
DOG_A, DOG_B = 'wstest-dog-a', 'wstest-dog-b'
FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


async def recv_until(ws, pred, timeout=10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, t_end - time.time()))
        except asyncio.TimeoutError:
            break
        try:
            f = json.loads(raw)
        except ValueError:
            continue
        if pred(f):
            return f
    return None


async def push_then_sync(ws, dog_id, missions, version):
    """Relay pattern: settings_updated, then schedule_sync ~40 ms later; returns (ack, sync_result)."""
    await ws.send(json.dumps({"type": "settings_updated", "dog_id": dog_id, "key": "missions",
                              "value": {"missions": missions}, "version": version}))
    await asyncio.sleep(0.04)
    cid = f"ws-{version}-{int(time.time() * 1000)}"
    await ws.send(json.dumps({"type": "command", "command": "schedule_sync", "command_id": cid,
                              "dog_id": dog_id, "source": "agent", "timestamp": time.time()}))
    ack = await recv_until(ws, lambda f: f.get("type") == "settings_applied" and f.get("version") == version)
    res = await recv_until(ws, lambda f: f.get("type") == "command_result" and f.get("command_id") == cid)
    return ack, (res or {}).get("result") or {}


async def main():
    now = datetime.now().astimezone()
    t_far = (now + timedelta(hours=5)).replace(second=0, microsecond=0)
    hhmm = t_far.strftime('%H:%M')
    hhmm2 = (t_far + timedelta(hours=1)).strftime('%H:%M')
    missions = [
        {"id": "m_sit_ws", "type": "sit", "name": "WS sit", "enabled": True,
         "trigger": {"kind": "schedule", "times": [hhmm, hhmm2], "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]},
         "duration_s": 120, "reward_treats": 1},
        {"id": "m_coach_ws", "type": "coach", "name": "WS coach", "enabled": True,
         "trigger": {"kind": "schedule", "time": hhmm}, "duration_s": 600, "reward_treats": 0},
        {"id": "m_sg_ws", "type": "silent_guardian", "name": "WS guard", "enabled": True,
         "trigger": {"kind": "schedule", "times": [hhmm + ":00"]}, "duration_s": 43200, "reward_treats": 0},
        {"id": "m_spin_ws", "type": "spin", "name": "WS spin", "enabled": True,
         "trigger": {"kind": "schedule", "times": [hhmm]}, "duration_s": 60, "reward_treats": 1},
        {"id": "m_notime_ws", "type": "sit", "name": "WS no time", "enabled": True,
         "trigger": {"kind": "schedule"}, "duration_s": 60, "reward_treats": 1},
        {"id": "m_dep_ws", "type": "recall", "name": "WS departure", "enabled": True,
         "trigger": {"kind": "departure"}, "duration_s": 60, "reward_treats": 1},
    ]
    async with websockets.connect(WS, max_size=8 * 1024 * 1024) as ws:
        hello = await recv_until(ws, lambda f: f.get("type") in ("connected", "settings_state") or "mode" in f, 5)
        check(hello is not None, f"connected to {WS}")

        ack, r = await push_then_sync(ws, DOG_A, missions, 101)
        check(ack and 'missions' in (ack.get('applied') or []), f"push acked: applied={ack and ack.get('applied')}")
        rows = {x['id']: x for x in r.get('missions', [])}
        check(set(rows) == {'m_sit_ws', 'm_coach_ws', 'm_sg_ws', 'm_spin_ws', 'm_notime_ws'},
              f"schedule_sync 40 ms after the push lists all 5 schedule missions ({sorted(rows)})")
        for mid in rows:
            check({'id', 'enabled', 'next_fire_at', 'reason'} <= set(rows[mid]), f"{mid} has id/enabled/next_fire_at/reason")
        sit = rows.get('m_sit_ws', {})
        check(sit.get('times') == sorted([hhmm, hhmm2]) and sit.get('next_fire_at')
              and abs(datetime.fromisoformat(sit['next_fire_at']).timestamp() - t_far.timestamp()) < 1
              and datetime.fromisoformat(sit['next_fire_at']).utcoffset() == now.utcoffset(),
              f"sit: two times kept, next_fire_at == {hhmm} today with robot offset ({sit.get('next_fire_at')})")
        coach = rows.get('m_coach_ws', {})
        check(coach.get('kind') == 'mode' and coach.get('times') == [hhmm] and coach.get('next_fire_at'),
              f"coach: trigger.time (single string) parsed, mode schedule armed ({coach.get('next_fire_at')})")
        sg = rows.get('m_sg_ws', {})
        check(sg.get('times') == [hhmm] and sg.get('duration_s') == 43200 and sg.get('next_fire_at'),
              "silent_guardian: HH:MM:SS parsed, 12 h duration kept, armed")
        spin = rows.get('m_spin_ws', {})
        check(spin.get('next_fire_at') is None and spin.get('reason') and 'spin' in spin['reason'],
              f"spin: next_fire_at null, reason={spin.get('reason')!r}")
        nt = rows.get('m_notime_ws', {})
        check(nt.get('next_fire_at') is None and nt.get('reason') and 'time' in nt['reason'],
              f"no-time mission: listed with reason={nt.get('reason')!r}, never fires")

        # other dog
        _, rb = await push_then_sync(ws, DOG_B, [missions[0] | {"id": "m_sit_b"}], 102)
        check([x['id'] for x in rb.get('missions', [])] == ['m_sit_b'], "dog B push lists only dog B's mission")
        _, ra = await push_then_sync(ws, DOG_A, missions, 103)
        check(len(ra.get('missions', [])) == 5, "re-push for dog A leaves 5 (dog B untouched)")

        # cancellation arrives as a push without the mission
        _, r2 = await push_then_sync(ws, DOG_A, [m for m in missions if m['id'] != 'm_coach_ws'], 104)
        ids = {x['id'] for x in r2.get('missions', [])}
        check('m_coach_ws' not in ids and len(ids) == 4, f"push without coach removes it ({sorted(ids)})")

        # empty set clears
        _, r3 = await push_then_sync(ws, DOG_A, [], 105)
        check(r3.get('missions') == [] and r3.get('count') == 0, "missions: [] clears dog A")
        _, rb2 = await push_then_sync(ws, DOG_B, [], 106)
        check(rb2.get('missions') == [], "cleanup: dog B cleared")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    asyncio.run(main())
