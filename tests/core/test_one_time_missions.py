#!/usr/bin/env python3
"""One-time relay missions (trigger.date) — relay brief 2026-11-12.

Fake engine + temp schedules dir; no hardware, no relay. Checks:
  - date present: next_fire_at is that date's time, days are ignored, once=true in
    schedule_sync; date absent: weekly behaviour unchanged;
  - a date far in the future is armed (not dropped by the 9-day look-ahead);
  - it fires on its date, is NOT deleted when it starts, and is deleted with one
    schedule_deleted {mission_id, dog_id} after the run finished;
  - a mode mission (silent_guardian) is deleted only after the mode run ends;
  - date+time already past at load: never runs, schedule_sync shows
    next_fire_at null + reason "date_passed", then deleted + schedule_deleted;
  - invalid date: kept as unsupported with a reason, never fires;
  - re-push with a new date clears the old date_passed verdict.

Run: /home/morgan/dogbot/env_new/bin/python tests/core/test_one_time_missions.py
Standalone (no pytest).
"""
import os
import sys
import tempfile
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['WIMZ_SCHEDULES_DIR'] = tempfile.mkdtemp(prefix='oncetest_')
os.environ['WIMZ_DATA_ROOT'] = tempfile.mkdtemp(prefix='oncetest_data_')

import core.mission_scheduler as ms                    # noqa: E402
from core.mission_scheduler import MissionScheduler    # noqa: E402
import core.schedule_manager as sm                     # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class FakeSession:
    def __init__(self, run_id, name):
        self.mission_id = run_id
        self.mission = types.SimpleNamespace(name=name)


class FakeEngine:
    def __init__(self):
        self.missions = {'sit': object(), 'quiet_progressive': object()}
        self.active_session = None
        self.started = []
        self._run = 100

    def start_mission(self, name, dog_id=None):
        self._run += 1
        self.started.append(name)
        self.active_session = FakeSession(self._run, name)
        return True

    def stop_mission(self, reason='x'):
        self.active_session = None
        return True


def now():
    return datetime.now().astimezone()


def mission(mid, when: datetime, mtype='sit', **extra):
    trig = {"kind": "schedule", "time": when.strftime("%H:%M"), "date": when.strftime("%Y-%m-%d"),
            "days": [when.strftime("%a").lower()], "delay_s": 0, "cooldown_s": 3600}
    trig.update(extra.pop('trigger', {}))
    return {"id": mid, "type": mtype, "name": mid, "enabled": True, "trigger": trig, **extra}


def row(sched, mid):
    return next((r for r in sched.get_scheduled_missions() if r['mission_id'] == mid), None)


def main():
    eng = FakeEngine()
    fake_me = types.ModuleType('orchestrators.mission_engine')
    fake_me.get_mission_engine = lambda: eng
    fake_me.MISSION_ALIASES = {}
    sys.modules['orchestrators.mission_engine'] = fake_me
    import orchestrators  # noqa: F401
    orchestrators.mission_engine = fake_me

    sm._manager = None
    sched = MissionScheduler()
    sched._mission_engine = eng
    sched._schedule_manager = sm.get_schedule_manager()
    sched.FIRST_TICK_DELAY_S = 0.0
    sched.check_interval = 0.1
    deleted = []
    sched.deleted_notifier = deleted.append
    mode_calls = []
    sched._mode_set = lambda mode, dur: mode_calls.append(('set', mode)) or True
    sched._mode_clear = lambda: mode_calls.append(('clear',))
    ms.ONCE_EXPIRED_LINGER_S = 0.6

    t = now()
    # A weekday name that is NOT the date's weekday: with a date it must be ignored.
    far = (t + timedelta(days=40)).replace(hour=15, minute=0, second=0, microsecond=0)
    wrong_day = (far + timedelta(days=1)).strftime("%a").lower()
    soon = t + timedelta(days=3)
    sched.apply_missions_setting('dogA', {"missions": [
        mission('far-one', far, trigger={"days": [wrong_day]}),
        {"id": "weekly-one", "type": "sit", "name": "weekly",
         "trigger": {"kind": "schedule", "time": soon.strftime("%H:%M"), "days": [soon.strftime("%a").lower()]}},
    ]})
    r = row(sched, 'far-one')
    check(r and r['once'] is True and r['date'] == far.strftime("%Y-%m-%d")
          and r['next_fire_at'] and datetime.fromisoformat(r['next_fire_at']) == far,
          f"date 40 days out: next_fire_at is that date at its time, days ignored ({r and r['next_fire_at']})")
    w = row(sched, 'weekly-one')
    check(w and w['once'] is False and w['date'] is None and w['type'] == 'weekly' and w['next_fire_at'],
          'no date: still a weekly mission')

    # ---- fires on its date; deleted only after the run finished
    sched.enable()
    due = now() + timedelta(seconds=1)
    past_slot = due.replace(second=0, microsecond=0)        # this minute: already "now" within grace
    sched.apply_missions_setting('dogA', {"missions": [mission('today-one', past_slot)]})
    t_end = time.time() + 5
    while time.time() < t_end and not eng.started:
        time.sleep(0.05)
    check(eng.started == ['sit'], 'one-time mission fired on its date/time')
    time.sleep(0.5)
    r = row(sched, 'today-one')
    check(r is not None and r['next_fire_at'] is None and not deleted,
          'still listed while running (next_fire_at null), not deleted at start')
    run_id = eng.active_session.mission_id
    eng.active_session = None
    sched.record_result(run_id, 'mission.completed', {'success': True})
    t_end = time.time() + 5
    while time.time() < t_end and not deleted:
        time.sleep(0.05)
    check(len(deleted) == 1 and deleted[0].get('mission_id') == 'today-one'
          and deleted[0].get('dog_id') == 'dogA' and deleted[0].get('reason') == 'completed',
          f'after the run: deleted + schedule_deleted with mission_id and dog_id ({deleted})')
    check(row(sched, 'today-one') is None, 'gone from schedule_sync afterwards')
    time.sleep(0.4)
    check(len(deleted) == 1 and eng.started == ['sit'], 'deleted once, never fired again')

    # ---- mode mission: only after the mode run ends
    deleted.clear()
    sched.apply_missions_setting('dogA', {"missions": [
        mission('sg-one', now().replace(second=0, microsecond=0), mtype='silent_guardian', duration_s=1.5)]})
    t_end = time.time() + 5
    while time.time() < t_end and not sched.active_mode_run():
        time.sleep(0.05)
    check(sched.active_mode_run() is not None and ('set', 'silent_guardian') in mode_calls,
          'one-time Silent Guardian mission started its mode run')
    time.sleep(0.6)
    check(not deleted and row(sched, 'sg-one') is not None, 'not deleted while the mode run is going')
    t_end = time.time() + 6
    while time.time() < t_end and not deleted:
        time.sleep(0.05)
    check(len(deleted) == 1 and deleted[0].get('mission_id') == 'sg-one' and sched.active_mode_run() is None,
          'deleted after the mode run ended')

    # ---- date already passed at load
    deleted.clear()
    started_before = list(eng.started)
    old = now() - timedelta(hours=3)
    sched.apply_missions_setting('dogB', {"missions": [mission('late-one', old)]})
    r = row(sched, 'late-one')
    check(r is not None and r['next_fire_at'] is None and r['reason'] == 'date_passed',
          f"past date at load: schedule_sync next_fire_at null + reason date_passed ({r and r['reason']})")
    t_end = time.time() + 5
    while time.time() < t_end and not deleted:
        time.sleep(0.05)
    check(len(deleted) == 1 and deleted[0].get('mission_id') == 'late-one'
          and deleted[0].get('reason') == 'date_passed' and eng.started == started_before,
          'then deleted + schedule_deleted, and it never ran late')

    # ---- invalid date
    sched.apply_missions_setting('dogC', {"missions": [
        mission('bad-one', now() + timedelta(days=1), trigger={"date": "14/11/2026"})]})
    r = row(sched, 'bad-one')
    check(r is not None and r['next_fire_at'] is None and 'invalid date' in str(r['reason']),
          f"invalid date kept as unsupported with a reason ({r and r['reason']})")

    # ---- re-push with a new date clears date_passed
    deleted.clear()
    ms.ONCE_EXPIRED_LINGER_S = 30.0
    sched.apply_missions_setting('dogD', {"missions": [mission('moved-one', now() - timedelta(hours=2))]})
    check(row(sched, 'moved-one')['reason'] == 'date_passed', 'moved-one starts as date_passed')
    sched.apply_missions_setting('dogD', {"missions": [mission('moved-one', now() + timedelta(days=2))]})
    r = row(sched, 'moved-one')
    check(r['reason'] is None and r['next_fire_at'] is not None and not deleted,
          'pushed again with a future date: armed, verdict cleared')

    sched.disable()
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
