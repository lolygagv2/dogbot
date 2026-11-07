#!/usr/bin/env python3
"""Relay outbox: durable, strictly ordered, batched HTTP delivery (2026-11-06).

CONTRACT v2 FINAL: events are rows in wimz.db `relay_outbox`; the relay
client's pump POSTs them to /events in seq order in batches of <=200, marks
rows sent on accepted/duplicate, and stops + backs off on HTTP failure.
This guards: ordering, no duplicates after a mid-batch failure, durability
across a process restart, prune cap, and the pump's stop-on-failure.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_relay_outbox_ordered_flush.py
Standalone (no pytest).
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TMP = tempfile.mkdtemp(prefix='outboxtest_')
os.environ['WIMZ_DATA_ROOT'] = TMP

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


from services.cloud.relay_outbox import RelayOutbox, PENDING_CAP, PENDING_TRIM_TO  # noqa: E402
import services.cloud.relay_outbox as ob_mod  # noqa: E402


def ev(i, etype='treat'):
    return {'event_id': f'id-{i:05d}', 'type': etype, 'device_id': 'tb-test', 'dog_id': 'dog-1',
            'timestamp': '2026-11-06T00:00:00.000Z', 'data': {'n': i}}


def test_order_and_mark_sent():
    ob = RelayOutbox(Path(TMP) / 'wimz.db')
    seqs = [ob.enqueue(ev(i)) for i in range(300)]
    check(seqs == sorted(seqs) and None not in seqs, "enqueue returns increasing seq")
    check(ob.pending_count() == 300, f"300 pending ({ob.pending_count()})")
    batch = ob.pending(200)
    check(len(batch) == 200 and [e['event_id'] for _, e in batch] == [f'id-{i:05d}' for i in range(200)],
          "pending(200) is the oldest 200 in order")
    # mid-batch failure: only first 149 sent
    ob.mark_sent([s for s, _ in batch[:149]])
    rest = ob.pending(500)
    check(len(rest) == 151 and rest[0][1]['event_id'] == 'id-00149', "unsent rows resume at the failure point")
    ob.mark_sent([s for s, _ in rest])
    check(ob.pending_count() == 0, "all sent after resume")
    # duplicate id is ignored
    check(ob.enqueue(ev(5)) is None, "duplicate event id ignored (relay dedupe key)")
    ob.close()


def test_durability_across_process():
    ob = RelayOutbox(Path(TMP) / 'wimz.db')
    for i in range(1000, 1005):
        ob.enqueue(ev(i))
    ob.close()
    code = (
        "import os,sys; sys.path.insert(0,%r); os.environ['WIMZ_DATA_ROOT']=%r;"
        "from pathlib import Path; from services.cloud.relay_outbox import RelayOutbox;"
        "ob=RelayOutbox(Path(%r)/'wimz.db'); print([e['event_id'] for _,e in ob.pending(10)])"
        % (str(ROOT), TMP, TMP))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    check("'id-01000'" in out.stdout and "'id-01004'" in out.stdout,
          f"pending rows visible from a new process ({out.stdout.strip()[-80:]} {out.stderr.strip()[-120:]})")
    ob = RelayOutbox(Path(TMP) / 'wimz.db')
    ob.mark_sent([s for s, _ in ob.pending(10)])
    ob.close()


def test_prune_cap():
    ob_mod.PENDING_CAP, ob_mod.PENDING_TRIM_TO = 50, 30
    try:
        ob = RelayOutbox(Path(TMP) / 'wimz.db')
        for i in range(2000, 2040):
            ob.enqueue(ev(i, 'status'))
        for i in range(3000, 3020):
            ob.enqueue(ev(i, 'treat'))
        ob.prune()
        pend = ob.pending(500)
        types = [e['type'] for _, e in pend]
        check(len(pend) == 30, f"pending trimmed to 30 ({len(pend)})")
        check(types.count('treat') == 20, "low-value status rows dropped first, treats kept")
        ob.mark_sent([s for s, _ in pend])
        ob.close()
    finally:
        ob_mod.PENDING_CAP, ob_mod.PENDING_TRIM_TO = PENDING_CAP, PENDING_TRIM_TO


def test_pump_stop_on_failure_then_resume():
    from services.cloud.relay_client import RelayClient, RelayConfig
    rc = RelayClient(RelayConfig(enabled=True, device_id='tb-test', device_secret='s3cret'))
    ob = RelayOutbox(Path(TMP) / 'wimz.db')
    rc._outbox = ob
    for i in range(4000, 4450):
        ob.enqueue(ev(i))
    posts = []
    state = {'fail_next': True}

    async def fake_http(method, path, json_body=None, data=None, timeout=15.0):
        posts.append((method, path, [e['event_id'] for e in json_body['events']]))
        if state['fail_next']:
            state['fail_next'] = False
            return 503, None
        return 200, {'accepted': [e['event_id'] for e in json_body['events']]}

    rc._http = fake_http
    rc._running = True

    async def run():
        rc._loop = asyncio.get_running_loop()
        task = asyncio.create_task(rc._outbox_pump())
        await asyncio.sleep(2.5)       # first batch fails, 1 s backoff, then drains
        rc._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(run())
    sizes = [len(ids) for _, _, ids in posts]
    check(posts and posts[0][1] == '/events' and posts[0][0] == 'POST', "pump POSTs to /events")
    check(sizes and sizes[0] == 200 and max(sizes) <= 200, f"batches of <=200 ({sizes})")
    check(posts[0][2][0] == 'id-04000' and posts[1][2][0] == 'id-04000',
          "failed batch is retried from the same first row (stop on failure)")
    sent_ids = [i for _, _, ids in posts[1:] for i in ids]
    check(sent_ids == [f'id-{i:05d}' for i in range(4000, 4450)], "all rows sent exactly once, in order")
    check(ob.pending_count() == 0, "outbox drained")
    check(len(posts) == 4, f"1 failed + 3 successful posts ({len(posts)})")
    ob.close()



def test_422_keeps_rows_and_per_row_rejected_drops():
    """A whole-batch 422 (validation) must NOT drop rows; only a per-row
    'rejected' verdict does. Legacy rows keyed 'id' ship as 'event_id'."""
    from services.cloud.relay_client import RelayClient, RelayConfig
    rc = RelayClient(RelayConfig(enabled=True, device_id='tb-test', device_secret='s3cret'))
    ob = RelayOutbox(Path(TMP) / 'wimz.db')
    rc._outbox = ob
    nodog = dict(ev(4999)); nodog['dog_id'] = None                   # dog-scoped row, no dog: held
    ob.enqueue(nodog)
    status = dict(ev(4998, 'status')); status['dog_id'] = None       # device-scoped: ships w/o dog_id
    ob.enqueue(status)
    legacy = dict(ev(5000)); legacy['id'] = legacy.pop('event_id')   # queued by an older build
    legacy['timestamp'] = legacy.pop('ts', legacy.get('timestamp', '2026-11-06T00:00:00Z'))
    ob.enqueue(legacy)
    for i in range(5001, 5004):
        ob.enqueue(ev(i))
    posts = []
    wire_rows = []
    state = {'n': 0}

    async def fake_http(method, path, json_body=None, data=None, timeout=15.0):
        rows = json_body['events']
        posts.append([e.get('event_id') for e in rows])
        wire_rows.append(rows[0])
        state['n'] += 1
        if state['n'] == 1:
            return 422, {'detail': [{'type': 'missing', 'loc': ['body', 'events', 0, 'event_id']}]}
        ids = [e['event_id'] for e in rows]
        # accept first two, reject the third, say nothing about the fourth
        # accept the status + legacy rows, reject the first treat, say nothing about the rest
        return 200, {'accepted': ids[:2], 'rejected': [{'event_id': ids[2], 'reason': 'bad'}]}

    rc._http = fake_http
    rc._running = True

    async def run():
        rc._loop = asyncio.get_running_loop()
        task = asyncio.create_task(rc._outbox_pump())
        await asyncio.sleep(2.5)
        rc._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(run())
    check(posts and posts[0][:2] == ['id-04998', 'id-05000'],
          f"dog-less status row ships, dog-less treat row skipped, legacy 'id' row POSTed as 'event_id' ({posts[:1]})")
    check(wire_rows and set(wire_rows[0]) == {'event_id','type','ts','data','device_id'},
          f"status wire row has no dog_id key ({sorted(wire_rows[0])})")
    check(len(posts) >= 2 and posts[1] == posts[0], f"422 kept all 5 rows and retried the same batch ({posts[:2]})")
    pend = [e['event_id'] for _, e in ob.pending(10)]
    check(pend == ['id-04999', 'id-05002', 'id-05003'],
          f"dog-less treat held, status+legacy accepted, 5001 rejected+gone, unmentioned rows pending ({pend})")
    ob.close()

def test_headers():
    from services.cloud.relay_client import RelayClient, RelayConfig
    rc = RelayClient(RelayConfig(enabled=True, device_id='tb-test', device_secret='s3cret',
                                 relay_url='wss://relay.example/ws/device'))
    h = rc._http_headers()
    check(set(h) == {'X-Device-ID', 'X-Timestamp', 'X-Signature'} and h['X-Device-ID'] == 'tb-test',
          "HMAC headers present")
    check(rc._http_url('/events') == 'https://relay.example/events', "URL derived from relay_url host")


if __name__ == '__main__':
    test_order_and_mark_sent()
    test_422_keeps_rows_and_per_row_rejected_drops()
    test_durability_across_process()
    test_prune_cap()
    test_pump_stop_on_failure_then_resume()
    test_headers()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
