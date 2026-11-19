#!/usr/bin/env python3
"""ToolDispatcher: command_id -> command_result, single-flight, timeout (2026-11-06).

CONTRACT v2 FINAL: every command with a command_id gets exactly one
{type:"command_result", command_id, ok, result, error}. Checks: happy path
frame shape, unknown command, busy while another runs, timeout sets the
cancel event and the slot is released only when the handler returns,
preemptive command cancels the running one, every call logged, and the
relay client's _dispatch_v2_command only intercepts frames with command_id
for registered commands.

Run: /home/morgan/dogbot/env_new/bin/python tests/cloud/test_tool_dispatcher_single_flight.py
Standalone (no pytest).
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='tooltest_')

import services.cloud.tool_dispatcher as td  # noqa: E402
from services.cloud.tool_dispatcher import ToolDispatcher, ToolSpec, MOTION  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


logged = []
_orig_log = td.log_tool_call
td.log_tool_call = lambda *a, **k: logged.append(a)


def collect():
    frames = []
    done = threading.Event()

    def reply(f):
        frames.append(f)
        done.set()
    return frames, done, reply


def main():
    d = ToolDispatcher()
    d._tools.clear()      # built-ins need hardware; use fakes
    d.register(ToolSpec('echo', lambda args, cancel: {'ok': True, 'echo': args.get('x')}, 5.0))
    d.register(ToolSpec('fail', lambda args, cancel: {'ok': False, 'error': {'code': 'nope', 'message': 'bad'}}, 5.0))
    d.register(ToolSpec('boom', lambda args, cancel: 1 / 0, 5.0))

    def slow(args, cancel):
        t_end = time.time() + args.get('s', 1.0)
        while time.time() < t_end and not cancel.is_set():
            time.sleep(0.02)
        return {'ok': True, 'cancelled': cancel.is_set()}
    d.register(ToolSpec('slow', slow, 0.5, group=MOTION))
    d.register(ToolSpec('slow2', slow, 5.0, group=MOTION))
    d.register(ToolSpec('stop', lambda args, cancel: {'ok': True}, 5.0, preemptive=True))
    d.register(ToolSpec('echo_motion', lambda args, cancel: {'ok': True, 'echo': args.get('x')}, 5.0, group=MOTION))

    frames, done, reply = collect()
    d.dispatch({'command': 'echo', 'command_id': 'c-1', 'args': {'x': 42}}, reply)
    done.wait(2)
    f = frames[0]
    check(f['type'] == 'command_result' and f['command_id'] == 'c-1' and f['success'] is True
          and f['result'] == {'echo': 42} and f['error'] is None,
          f"command_result frame shape ({f})")

    f = d.call('nope', {}, 'c-2')
    check(f['success'] is False and f['error'] == 'unknown_command' and 'echo' in f['result']['available'],
          "unknown command -> command_result with unknown_command")
    f = d.call('fail', {}, 'c-3')
    check(f['success'] is False and f['error'] == 'nope' and f['result']['message'] == 'bad', "handler error passes through")
    f = d.call('boom', {}, 'c-4')
    check(f['success'] is False and f['error'] == 'exception', "exception -> command_result exception")

    # busy: only within the motion group (2026-11-07: audio must not block photos)
    frames, done, reply = collect()
    d.dispatch({'command': 'slow', 'command_id': 'c-5', 'args': {'s': 0.3}}, reply)
    time.sleep(0.05)
    f = d.call('echo_motion', {}, 'c-6')
    check(f['success'] is False and f['error'] == 'busy' and f['result']['running'] == 'slow'
          and f['result']['group'] == MOTION, "second MOTION command while one runs -> busy")
    f = d.call('echo', {'x': 7}, 'c-6b')
    check(f['success'] is True and f['result']['echo'] == 7, "ungrouped command runs while a motion command is running")
    lt = d.list_tools()
    check(lt['busy'] is True and lt['current']['name'] == 'slow' and [r['name'] for r in lt['running']] == ['slow'],
          f"list_tools reports motion busy + running ({lt['running']})")
    done.wait(2)
    check(frames and frames[0]['success'] is True and frames[0]['command_id'] == 'c-5', "slow completes normally")

    # two ungrouped slow commands overlap
    def slow_free(args, cancel):
        time.sleep(0.3); return {'ok': True}
    d.register(ToolSpec('slow_free', slow_free, 5.0))
    fr1, dn1, rp1 = collect(); fr2, dn2, rp2 = collect()
    t0 = time.time()
    d.dispatch({'command': 'slow_free', 'command_id': 'c-6c'}, rp1)
    d.dispatch({'command': 'slow_free', 'command_id': 'c-6d'}, rp2)
    dn1.wait(2); dn2.wait(2)
    check(fr1 and fr2 and fr1[0]['success'] and fr2[0]['success'] and time.time() - t0 < 0.5,
          f"two ungrouped commands run concurrently ({time.time() - t0:.2f}s)")

    # timeout: handler ignores cancel for 1.2 s, spec timeout 0.5 s
    def stubborn(args, cancel):
        time.sleep(1.2)
        return {'ok': True}
    d.register(ToolSpec('stubborn', stubborn, 0.5, group=MOTION))
    t0 = time.time()
    f = d.call('stubborn', {}, 'c-7', wait_s=3)
    check(f['success'] is False and f['error'] == 'timeout' and time.time() - t0 < 1.0,
          "timeout answered at 0.5 s without waiting for the handler")
    f2 = d.call('echo_motion', {}, 'c-8')
    check(f2['success'] is False and f2['error'] == 'busy', "motion slot stays held until the timed-out handler returns")
    time.sleep(1.0)
    f3 = d.call('echo_motion', {}, 'c-9')
    check(f3['success'] is True, "motion slot released after the handler returned")

    # preemptive
    frames, done, reply = collect()
    d.dispatch({'command': 'slow', 'command_id': 'c-10', 'args': {'s': 5}}, reply)
    time.sleep(0.05)
    f = d.call('stop', {}, 'c-11', wait_s=4)
    done.wait(2)
    check(f['success'] is True and frames and frames[0]['result'].get('cancelled') is True,
          "preemptive command cancels the running one and then runs")

    # ---- queueing (2026-11-18): queue=True commands wait their turn, FIFO
    order = []

    def step(args, cancel):
        order.append(('start', args['n']))
        t_end = time.time() + 0.25
        while time.time() < t_end and not cancel.is_set():
            time.sleep(0.01)
        order.append(('end', args['n']))
        return {'ok': True, 'n': args['n'], 'cut': cancel.is_set()}
    d.register(ToolSpec('qmove', step, 5.0, group=MOTION, queue=True, busy_code='robot_busy',
                        urgent=lambda a: a.get('n') == 'stop'))
    d.register(ToolSpec('qlook', step, 5.0, group=MOTION, queue=True, busy_code='gimbal_busy'))
    cols = [collect() for _ in range(3)]
    d.dispatch({'command': 'qmove', 'command_id': 'q-1', 'args': {'n': 1}}, cols[0][2])
    d.dispatch({'command': 'qmove', 'command_id': 'q-2', 'args': {'n': 2}}, cols[1][2])
    d.dispatch({'command': 'qlook', 'command_id': 'q-3', 'args': {'n': 3}}, cols[2][2])
    t0 = time.time()
    cols[0][1].wait(1)
    first_done = time.time() - t0
    check(d.list_tools()['queued'] and not cols[1][0] or len(order) >= 3, "second command is queued, not answered early")
    for c in cols:
        c[1].wait(3)
    check(order == [('start', 1), ('end', 1), ('start', 2), ('end', 2), ('start', 3), ('end', 3)],
          f"three back-to-back commands run one after the other in arrival order ({order})")
    check(all(c[0] and c[0][0]['success'] for c in cols) and first_done >= 0.2
          and cols[2][0][0]['result'].get('queued_ms', 0) >= 400,
          f"each command_result only after its own motion ended (3rd queued {cols[2][0][0]['result'].get('queued_ms')}ms)")

    # not queueable behind a long non-queue command -> honest busy codes at once
    fr, dn, rp = collect()
    d.dispatch({'command': 'slow2', 'command_id': 'q-4', 'args': {'s': 0.4}}, rp)
    time.sleep(0.05)
    t0 = time.time()
    f1 = d.call('qmove', {'n': 9}, 'q-5')
    f2 = d.call('qlook', {'n': 9}, 'q-6')
    check(f1['error'] == 'robot_busy' and f2['error'] == 'gimbal_busy' and time.time() - t0 < 0.2,
          "behind a non-queueable command: move -> robot_busy, look -> gimbal_busy, immediately")
    dn.wait(2)

    # queue full -> robot_busy
    import services.cloud.tool_dispatcher as tdq
    order.clear()
    cols = [collect() for _ in range(tdq.QUEUE_MAX_DEPTH + 2)]
    for i, c in enumerate(cols):
        d.dispatch({'command': 'qmove', 'command_id': f'q-f{i}', 'args': {'n': i}}, c[2])
    cols[-1][1].wait(1)
    check(cols[-1][0] and cols[-1][0][0]['error'] and cols[-1][0][0]['error'] == 'robot_busy',
          f"queue full ({tdq.QUEUE_MAX_DEPTH} waiting) -> robot_busy")
    # urgent stop: cuts the running one, drops the queue, runs at once
    f = d.call('qmove', {'n': 'stop'}, 'q-stop', wait_s=3)
    for c in cols[:-1]:
        c[1].wait(2)
    codes = [c[0][0]['error'] for c in cols[1:-1]]
    check(f['success'] and codes.count('cancelled') >= tdq.QUEUE_MAX_DEPTH - 1 and cols[0][0][0]['result'].get('cut') is True,
          f"stop jumps the queue: running one cut, queued ones answer cancelled ({codes})")
    time.sleep(0.1)
    check(d.call('qmove', {'n': 5}, 'q-after')['success'] and not d.list_tools()['queued'], "slot free again after stop")

    check(len(logged) >= 10 and all(len(a) >= 6 for a in logged), f"every call logged ({len(logged)})")
    names = {a[0] for a in logged}
    check({'echo', 'nope', 'fail', 'boom', 'slow', 'stubborn', 'stop'} <= names, "log covers unknown/busy/timeout too")

    # built-in grouping: only move/go_to/run_trick are motion; audio/photo/listen/treat are free
    real = ToolDispatcher()
    try:
        from orchestrators.trick_runner import register_run_trick
        register_run_trick(real)
    except Exception as e:
        print('note: run_trick not registered in test env:', e)
    groups = {n: t.group for n, t in real._tools.items()}
    check(all(groups.get(n) == MOTION for n in ('move', 'go_to') if n in groups)
          and groups.get('run_trick', MOTION) == MOTION
          and all(groups.get(n) is None for n in ('play_sound', 'take_photo', 'listen', 'dispense_treat', 'set_mode')),
          f"built-in groups: motion only for move/go_to/run_trick ({groups})")

    # relay interception rule
    from services.cloud.relay_client import RelayClient, RelayConfig
    rc = RelayClient(RelayConfig(enabled=True, device_id='tb', device_secret='s'))
    import services.cloud.tool_dispatcher as tdm
    saved = tdm._instance
    tdm._instance = d
    try:
        sent = []
        rc.send_message = lambda m: sent.append({**m, 'device_id': m.get('device_id', 'tb')})
        check(rc._dispatch_v2_command('echo', {'command': 'echo'}, {}) is False, "no command_id -> legacy path")
        check(rc._dispatch_v2_command('motor', {'command': 'motor', 'command_id': 'c-12'}, {}) is False,
              "unregistered command -> legacy path even with command_id")
        ok = rc._dispatch_v2_command('echo', {'command': 'echo', 'command_id': 'c-13', 'x': 1, 'timestamp': 5},
                                     {'x': 1, 'timestamp': 5})
        time.sleep(0.3)
        check(ok and sent and sent[0]['type'] == 'command_result' and sent[0]['command_id'] == 'c-13'
              and sent[0]['result']['echo'] == 1 and sent[0]['device_id'] == 'tb',
              f"registered command with command_id -> dispatched + command_result sent ({sent[:1]})")
    finally:
        tdm._instance = saved


if __name__ == '__main__':
    main()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1 if FAILS else 0)
