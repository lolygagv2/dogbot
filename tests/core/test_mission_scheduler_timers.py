#!/usr/bin/env python3
"""MissionScheduler persisted timers (2026-11-07).

Checks, without hardware: next_fire math for once/daily/weekly incl. grace
and "missed"; a one-shot persisted to disk fires on a NEW scheduler
instance (simulated restart) within one tick and is then disabled; the
daily slot re-arms to tomorrow after firing; mission.completed/stopped on
the bus stamp last_result success|fail|aborted via last_run_id; cancel by
schedule_id / relay mission_id / mission_name and stop of the running one;
the three dispatcher tools answer command_result frames.

Run: /home/morgan/dogbot/env_new/bin/python tests/core/test_mission_scheduler_timers.py
Standalone (no pytest).
"""
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_SCHEDULES_DIR'] = tempfile.mkdtemp(prefix='schedtest_')
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='schedtest_data_')

import core.mission_scheduler as ms                                   # noqa: E402
from core.mission_scheduler import MissionScheduler, compute_next_fire  # noqa: E402
import core.schedule_manager as sm                                     # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


class FakeSession:
    def __init__(self, run_id, name, dog_id):
        self.mission_id, self.dog_id = run_id, dog_id
        self.mission = type('M', (), {'name': name})()


class FakeEngine:
    """Only what the scheduler touches."""
    def __init__(self):
        self.missions = {'sit': object(), 'evening_settle': object()}
        self.active_session = None
        self.started = []
        self.stopped = []
        self.last_failure_reason = ''
        self._next_run = 100

    def start_mission(self, name, dog_id=None):
        self._next_run += 1
        self.started.append((name, dog_id, time.time()))
        self.active_session = FakeSession(self._next_run, name, dog_id)
        return True

    def stop_mission(self, reason='x'):
        if not self.active_session:
            return False
        self.stopped.append((self.active_session.mission_id, reason))
        self.active_session = None
        return True


def fresh_manager():
    """Simulate a process restart: brand-new ScheduleManager from disk."""
    sm._manager = None
    return sm.get_schedule_manager()


def new_scheduler(engine):
    s = MissionScheduler()
    s._mission_engine = engine
    s._schedule_manager = fresh_manager()
    s.FIRST_TICK_DELAY_S = 0.0
    s.check_interval = 0.2
    return s


def now():
    return datetime.now().astimezone()


def main():
    # mission_engine import is needed by ScheduleManager.create_schedule validation;
    # stub it so no hardware singletons are built.
    import types
    fake_me = types.ModuleType('orchestrators.mission_engine')
    eng = FakeEngine()
    fake_me.get_mission_engine = lambda: eng
    fake_me.MISSION_ALIASES = {'Basic Sit': 'sit'}
    sys.modules['orchestrators.mission_engine'] = fake_me
    import orchestrators  # noqa: F401  (package must exist for the alias)
    orchestrators.mission_engine = fake_me

    t0 = now()

    # ---------------------------------------------------------------- math
    one = {'type': 'once', 'fire_at': (t0 + timedelta(minutes=2)).isoformat(), 'created_at': t0.isoformat()}
    nxt, missed = compute_next_fire(one, t0)
    check(nxt is not None and abs((nxt - (t0 + timedelta(minutes=2))).total_seconds()) < 1 and not missed,
          "once/fire_at -> next_fire_at == fire_at")
    nxt, missed = compute_next_fire(one, t0 + timedelta(minutes=2, seconds=30))
    check(nxt is not None, "once/fire_at 30 s late -> still due (inside 1 h grace)")
    nxt, missed = compute_next_fire(one, t0 + timedelta(hours=2))
    check(nxt is None and len(missed) == 1, "once/fire_at 2 h late -> missed, no next")
    nxt, missed = compute_next_fire(one, t0 + timedelta(hours=2), last_fired_at=t0 + timedelta(minutes=2, seconds=5))
    check(nxt is None and not missed, "once/fire_at already fired -> nothing")

    hhmm = (t0 + timedelta(minutes=1)).strftime('%H:%M')
    daily = {'type': 'daily', 'start_time': hhmm, 'end_time': (t0 + timedelta(minutes=31)).strftime('%H:%M'),
             'created_at': t0.isoformat()}
    nxt, missed = compute_next_fire(daily, t0)
    check(nxt is not None and 0 < (nxt - t0).total_seconds() <= 60 and not missed,
          "daily -> next slot today, no missed for slots before created_at")
    slot = nxt
    nxt2, _ = compute_next_fire(daily, slot + timedelta(seconds=3), last_fired_at=slot + timedelta(seconds=3))
    check(nxt2 is not None and abs((nxt2 - (slot + timedelta(days=1))).total_seconds()) < 1,
          "daily fired at slot+3 s -> tomorrow")
    nxt3, missed3 = compute_next_fire(daily, slot + timedelta(minutes=45))
    check(nxt3 is not None and abs((nxt3 - (slot + timedelta(days=1))).total_seconds()) < 1 and missed3 == [slot],
          "daily 45 min late (30 min window) -> today missed, tomorrow armed")
    nxt4, _ = compute_next_fire(daily, slot + timedelta(minutes=10))
    check(nxt4 == slot, "daily 10 min late (inside window) -> today's slot still due")

    dayname = (t0 + timedelta(days=3)).strftime('%A').lower()
    weekly = {'type': 'weekly', 'start_time': '09:00', 'days_of_week': [dayname], 'created_at': t0.isoformat()}
    nxt, _ = compute_next_fire(weekly, t0)
    check(nxt is not None and nxt.strftime('%A').lower() == dayname and nxt.hour == 9,
          f"weekly -> next {dayname} 09:00")

    # ---------------------------------------------------------------- persistence + restart
    mgr = fresh_manager()
    fire_at = now() + timedelta(seconds=0.6)
    res = mgr.create_schedule({'mission_name': 'sit', 'dog_id': 'd1', 'fire_at': fire_at.isoformat(),
                               'mission_id': 'relay-m-1'})
    check(res['success'] and res['schedule']['type'] == 'once' and res['schedule']['mission_id'] == 'relay-m-1',
          "create_schedule with fire_at -> once, relay mission_id kept")
    sid = res['schedule']['schedule_id']
    path = Path(os.environ['WIMZ_SCHEDULES_DIR']) / f"{sid}.json"
    check(path.exists(), "schedule persisted to disk")

    # "restart": brand-new scheduler + manager read from disk, then enable
    sched = new_scheduler(eng)
    sched.enable()
    row = next(r for r in sched.get_scheduled_missions() if r['schedule_id'] == sid)
    check(row['next_fire_at'] is not None and row['last_run_at'] is None and row['mission_id'] == 'relay-m-1',
          f"after boot: armed next_fire_at={row['next_fire_at']}")
    deadline = time.time() + 5
    while time.time() < deadline and not eng.started:
        time.sleep(0.05)
    check(len(eng.started) == 1 and eng.started[0][0] == 'sit' and eng.started[0][1] == 'd1',
          "one-shot fired after restart with the schedule's dog_id")
    check(eng.started and abs(eng.started[0][2] - fire_at.timestamp()) < 1.0,
          f"fired within 1 s of fire_at (delta {eng.started[0][2] - fire_at.timestamp():+.2f}s)" if eng.started else "fired?")
    time.sleep(0.3)
    on_disk = sm.get_schedule_manager().get_schedule(sid)
    check(on_disk['enabled'] is False and on_disk['last_result'] == 'running' and on_disk['last_run_id'] == 101
          and on_disk['last_run_at'] is not None, f"persisted: disabled, running, run_id=101 ({on_disk['last_result']})")
    row = next(r for r in sched.get_scheduled_missions() if r['schedule_id'] == sid)
    check(row['next_fire_at'] is None and row['enabled'] is False, "sync row: once disabled, no next_fire_at")

    # result via bus-shaped call
    sched.record_result(101, 'mission.completed', {'success': True, 'rewards_given': 2})
    check(sm.get_schedule_manager().get_schedule(sid)['last_result'] == 'success', "mission.completed -> success")
    sched.record_result(101, 'mission.completed', {'success': False})
    check(sm.get_schedule_manager().get_schedule(sid)['last_result'] == 'fail', "mission.completed success=false -> fail")
    sched.record_result(101, 'mission.stopped', {})
    check(sm.get_schedule_manager().get_schedule(sid)['last_result'] == 'aborted', "mission.stopped -> aborted")
    eng.active_session = None

    # no double fire on a second restart
    sched.disable()
    sched2 = new_scheduler(eng)
    sched2.enable()
    time.sleep(0.6)
    check(len(eng.started) == 1, "second restart does not re-fire the consumed one-shot")

    # daily: fires at slot, re-arms tomorrow, survives restart without re-firing
    slot = (now() + timedelta(seconds=0.8)).replace(microsecond=0)
    res = sm.get_schedule_manager().create_schedule({'mission_name': 'evening_settle', 'dog_id': 'd1',
                                                     'type': 'daily', 'start_time': slot.strftime('%H:%M')})
    dsid = res['schedule']['schedule_id']
    # start_time is HH:MM so the slot is the top of the current minute: force the
    # slot to be "now" by backdating created_at and pretending it is due.
    sm.get_schedule_manager().update_schedule(dsid, {'start_time': now().strftime('%H:%M')})
    sched2.poke()
    deadline = time.time() + 3
    while time.time() < deadline and len(eng.started) < 2:
        time.sleep(0.05)
    check(len(eng.started) == 2 and eng.started[1][0] == 'evening_settle', "daily slot inside window fires once")
    time.sleep(0.3)
    d = sm.get_schedule_manager().get_schedule(dsid)
    nxt = datetime.fromisoformat(d['next_fire_at'])
    check(d['enabled'] is True and (nxt - now()).total_seconds() > 23 * 3600, f"daily re-armed for tomorrow ({d['next_fire_at']})")
    sched2.disable()
    eng.active_session = None
    sched3 = new_scheduler(eng)
    sched3.enable()
    time.sleep(0.6)
    check(len(eng.started) == 2, "restart after a daily fire does not re-fire today's slot")

    # ---------------------------------------------------------------- cancel
    eng.start_mission('sit', 'd1')                    # run 103 is "active" but not from a schedule
    out = sched3.cancel('nope')
    check(out['ok'] is False and out['error']['code'] == 'not_found', "cancel unknown -> not_found")
    out = sched3.cancel('active')
    check(out['ok'] and out['stopped_run'] and eng.active_session is None, "cancel 'active' stops the running mission")

    res = sm.get_schedule_manager().create_schedule({'mission_name': 'sit', 'dog_id': 'd1', 'type': 'daily',
                                                     'start_time': '03:00', 'mission_id': 'relay-m-9'})
    csid = res['schedule']['schedule_id']
    out = sched3.cancel('relay-m-9')
    check(out['ok'] and out['cancelled'] == csid and sm.get_schedule_manager().get_schedule(csid) is None
          and not (Path(os.environ['WIMZ_SCHEDULES_DIR']) / f"{csid}.json").exists(),
          "cancel by relay mission_id deletes the schedule file")
    out = sched3.cancel('evening_settle')
    check(out['ok'] and out['cancelled'] == dsid, "cancel by mission_name")
    # cancel the schedule whose run is active -> stops it too
    res = sm.get_schedule_manager().create_schedule({'mission_name': 'sit', 'dog_id': 'd1',
                                                     'fire_at': (now() + timedelta(seconds=0.3)).isoformat()})
    rsid = res['schedule']['schedule_id']
    sched3.poke()
    deadline = time.time() + 3
    while time.time() < deadline and eng.active_session is None:
        time.sleep(0.05)
    time.sleep(0.2)
    out = sched3.cancel(rsid)
    check(out['ok'] and out['stopped_run'] and eng.active_session is None and eng.stopped[-1][1] == 'schedule_cancelled',
          "cancel of the schedule whose run is executing stops that run")
    sched3.disable()

    # ---------------------------------------------------------------- dispatcher tools
    import services.cloud.tool_dispatcher as td
    td.log_tool_call = lambda *a, **k: None
    ms._scheduler = sched3
    sched3.enable()
    d = td.ToolDispatcher()
    f = d.call('schedule_mission', {'mission': 'Basic Sit', 'dog_id': 'd2',
                                    'fire_at': (now() + timedelta(hours=1)).isoformat(), 'mission_id': 'relay-m-2'}, 'c-1', 3)
    check(f.get('type') == 'command_result' and f['ok'] and f['result']['mission_name'] == 'sit'
          and f['result']['mission_id'] == 'relay-m-2' and f['result']['next_fire_at'],
          f"schedule_mission -> command_result ok (alias resolved) {f.get('error')}")
    f = d.call('schedule_mission', {'mission': 'sit', 'dog_id': 'd2', 'start_time': '07:30'}, 'c-2', 3)
    check(f['ok'] and f['result']['type'] == 'daily', "schedule_mission start_time only -> daily")
    f = d.call('schedule_mission', {'mission': 'nope', 'dog_id': 'd2', 'start_time': '07:30'}, 'c-3', 3)
    check(f['ok'] is False and f['error']['code'] == 'unknown_mission', "schedule_mission unknown mission -> error")
    f = d.call('schedule_sync', {}, 'c-4', 3)
    rows = f['result']['missions']
    check(f['ok'] and f['result']['count'] == len(rows) >= 3
          and all({'mission_id', 'next_fire_at', 'last_run_at', 'last_result'} <= set(r) for r in rows),
          f"schedule_sync -> {len(rows)} rows with next_fire_at/last_run_at/last_result")
    f = d.call('cancel_mission', {'mission_id': 'relay-m-2'}, 'c-5', 3)
    check(f['ok'] and f['result']['cancelled'], "cancel_mission by relay mission_id -> command_result ok")
    f = d.call('cancel_mission', {'mission_id': 'relay-m-2'}, 'c-6', 3)
    check(f['ok'] is False and f['error']['code'] == 'not_found', "cancel_mission again -> not_found")
    sched3.disable()

    # ---------------------------------------------------------------- relay `missions` setting
    sched4 = new_scheduler(eng)
    sched4.enable()
    mgr = sm.get_schedule_manager()
    for s_ in mgr.list_schedules():
        mgr.delete_schedule(s_['schedule_id'])
    t1 = now() + timedelta(minutes=30)
    doc = {"missions": [
        {"id": "m-sit", "type": "sit", "name": "Morning sit", "enabled": True,
         "trigger": {"kind": "schedule", "times": [t1.strftime('%H:%M'), "23:59"], "days": ["mon", "Tue", "wed", "thu", "fri", "sat", "sun"]},
         "duration_s": 120, "reward_treats": 1},
        {"id": "m-sg", "type": "silent_guardian", "name": "Guard", "enabled": True,
         "trigger": {"kind": "schedule", "times": ["09:00"]}, "duration_s": 99999, "reward_treats": 0},
        {"id": "m-dep", "type": "recall", "trigger": {"kind": "departure"}},
        {"id": "m-spin", "type": "spin", "trigger": {"kind": "schedule", "times": ["10:00"]}},
        {"id": "m-off", "type": "down", "enabled": False, "trigger": {"kind": "schedule", "times": ["11:00"]}},
    ]}
    res = sched4.apply_missions_setting('dogA', {"key": "missions", "value": doc, "version": 7})
    check(len(res['loaded']) == 4 and len(res['skipped']) == 1 and res['skipped'][0]['id'] == 'm-dep',
          f"missions setting: 4 schedule missions loaded, departure skipped ({res})")
    rows = {r['id']: r for r in sched4.get_scheduled_missions(dog_id='dogA')}
    check(set(rows) == {'m-sit', 'm-sg', 'm-spin', 'm-off'}, f"schedule_sync ids are the setting's ids ({sorted(rows)})")
    r = rows['m-sit']
    check(r['enabled'] and r['type'] == 'weekly' and r['days_of_week'] == ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
          and r['times'] == sorted([t1.strftime('%H:%M'), '23:59']) and r['mission_name'] == 'sit',
          "mon..sun normalized, both times kept, type sit -> mission sit")
    nfa = datetime.fromisoformat(r['next_fire_at'])
    check(abs((nfa - t1.replace(second=0, microsecond=0)).total_seconds()) < 1 and nfa.utcoffset() == now().utcoffset(),
          f"next_fire_at = earliest time today with the robot's UTC offset ({r['next_fire_at']})")
    check(rows['m-sg']['kind'] == 'mode' and rows['m-sg']['mode'] == 'silent_guardian' and rows['m-sg']['reward_treats'] == 0
          and rows['m-sg']['next_fire_at'], "silent_guardian -> mode schedule, no treats, armed")
    check(rows['m-spin']['unsupported'] and rows['m-spin']['next_fire_at'] is None, "unsupported type spin -> next_fire_at null")
    check(rows['m-off']['enabled'] is False and rows['m-off']['next_fire_at'] is None, "disabled mission -> enabled false, next_fire_at null")
    # mark runtime, re-push same doc -> replaced but history carried, no new fire
    sid_sit = rows['m-sit']['schedule_id']
    mgr.set_runtime(sid_sit, {'last_run_at': '2026-01-01T08:00:00-05:00', 'last_result': 'success', 'last_run_id': 55})
    res = sched4.apply_missions_setting('dogA', doc)
    rows2 = {r['id']: r for r in sched4.get_scheduled_missions(dog_id='dogA')}
    check(res['replaced'] == 4 and rows2['m-sit']['schedule_id'] != sid_sit and rows2['m-sit']['last_result'] == 'success'
          and rows2['m-sit']['last_run_id'] == 55, "re-push replaces the dog's set and carries last_run_*")
    # other dog untouched
    far = (now() + timedelta(hours=3)).strftime('%H:%M')
    sched4.apply_missions_setting('dogB', {"missions": [{"id": "b-1", "type": "sit", "trigger": {"kind": "schedule", "times": [far]}}]})
    check(len(sched4.get_scheduled_missions(dog_id='dogA')) == 4 and len(sched4.get_scheduled_missions(dog_id='dogB')) == 1,
          "per-dog replace leaves the other dog's missions alone")
    sched4.apply_missions_setting('dogA', {"missions": []})
    check(len(sched4.get_scheduled_missions(dog_id='dogA')) == 0 and len(sched4.get_scheduled_missions(dog_id='dogB')) == 1,
          "empty missions for dogA clears only dogA")
    # dog-local timezone: 09:00 in Los_Angeles == 12:00 New_York
    sched4.apply_missions_setting('dogB', {"missions": []})
    import services.cloud.remote_settings as rsm
    rs = rsm.get_remote_settings()
    rs._settings['timezone'] = 'America/Los_Angeles'
    sched4.apply_missions_setting('dogC', {"missions": [{"id": "c-1", "type": "sit", "trigger": {"kind": "schedule", "times": ["09:00"]}}]})
    rc = sched4.get_scheduled_missions(dog_id='dogC')[0]
    nfa = datetime.fromisoformat(rc['next_fire_at'])
    from zoneinfo import ZoneInfo
    check(rc['tz'] == 'America/Los_Angeles' and nfa.astimezone(ZoneInfo('America/Los_Angeles')).hour == 9
          and nfa.utcoffset() == now().utcoffset(),
          f"times are the dog's local HH:MM (tz setting), reported in the robot's offset ({rc['next_fire_at']})")
    rs._settings.pop('timezone', None)
    sched4.apply_missions_setting('dogC', {"missions": []})

    # ---------------------------------------------------------------- mode run (coach for 1 s)
    eng.active_session = None
    ms._scheduler = sched4
    modes = []
    sched4._mode_set = lambda mode, dur: modes.append(('set', mode, dur)) or True
    sched4._mode_clear = lambda: modes.append(('clear',))
    sched4._new_run_id = lambda name, cfg: 777
    sched4._end_run_record = lambda *a, **k: None
    events = []
    ms.publish_system_event = lambda st, data, *a: events.append((st, data))
    doc = {"missions": [{"id": "m-coach", "type": "coach", "name": "Coach hour", "enabled": True,
                         "trigger": {"kind": "schedule", "times": [now().strftime('%H:%M')]}, "duration_s": 1}]}
    sched4.apply_missions_setting('dogA', doc)
    deadline = time.time() + 4
    while time.time() < deadline and not modes:
        time.sleep(0.05)
    check(modes and modes[0] == ('set', 'coach', 1.0), f"coach mission fires -> set_mode coach for duration_s ({modes})")
    f = td.ToolDispatcher().call('schedule_sync', {}, 'c-7', 3)
    check(f['result']['active'] and f['result']['active'].get('kind') == 'mode' and f['result']['active']['run_id'] == 777,
          f"schedule_sync.active shows the mode run ({f['result']['active']})")
    check(any(st == 'mission.started' and d.get('mission_id') == 777 and d.get('kind') == 'mode' for st, d in events),
          "mission.started published for the mode run")
    deadline = time.time() + 4
    while time.time() < deadline and ('clear',) not in modes:
        time.sleep(0.05)
    check(('clear',) in modes, "mode cleared (back to idle) after duration_s")
    done_ev = [d for st, d in events if st == 'mission.completed' and d.get('mission_id') == 777]
    check(done_ev and done_ev[0]['rewards_given'] == 0 and done_ev[0]['schedule_mission_id'] == 'm-coach',
          "mission.completed with rewards 0 + schedule_mission_id -> mission_run goes out")
    time.sleep(0.3)
    rowc = sched4.get_scheduled_missions(dog_id='dogA')[0]
    check(rowc['last_result'] == 'success' and rowc['last_run_id'] == 777, f"mode run stamped success ({rowc['last_result']})")
    check(sched4.run_meta(777) == {'schedule_id': rowc['schedule_id'], 'mission_id': 'm-coach', 'dog_id': 'dogA',
                                   'kind': 'mode', 'mission_type': 'coach'}, "run_meta resolves the relay mission id")
    # cancel an executing mode run
    modes.clear(); events.clear()
    sched4._new_run_id = lambda name, cfg: 778
    doc['missions'][0]['duration_s'] = 600
    doc['missions'][0]['trigger']['times'] = [now().strftime('%H:%M')]
    mgr2 = sm.get_schedule_manager()
    sched4.apply_missions_setting('dogA', doc)
    # same mission id carried last_fired_at -> would not refire this minute; force a fresh slot
    sidc = sched4.get_scheduled_missions(dog_id='dogA')[0]['schedule_id']
    mgr2.set_runtime(sidc, {'last_fired_at': None})
    sched4.poke()
    deadline = time.time() + 4
    while time.time() < deadline and not modes:
        time.sleep(0.05)
    check(modes and modes[0][1] == 'coach', "second coach run started")
    f = td.ToolDispatcher().call('cancel_mission', {'mission_id': 'm-coach'}, 'c-8', 3)
    check(f['ok'] and f['result']['stopped_run'] and f['result']['run_id'] == 778 and ('clear',) in modes,
          f"cancel_mission on an executing mode run stops it ({f.get('result')})")
    check(any(st == 'mission.stopped' and d.get('mission_id') == 778 for st, d in events), "mission.stopped -> mission_run aborted")
    check(sched4.active_mode_run() is None, "no active mode run after cancel")
    sched4.disable()

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
