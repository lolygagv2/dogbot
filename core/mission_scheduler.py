#!/usr/bin/env python3
"""
Mission Scheduler for WIM-Z — persisted, re-armed on boot (2026-11-07).

Fires the user/relay schedules kept by core.schedule_manager (JSON files in
/home/morgan/dogbot/schedules). Every schedule carries its own timer state
on disk (last_fired_at, last_run_at, last_run_id, last_result, next_fire_at)
so a restart re-arms from the same facts: a slot that came due while the
service was down (or restarting) fires on the first tick after boot, as
long as it is still inside the schedule's grace window.

Timing: 5 s tick, so a timer fires within ~5 s of its instant, and within
~30 s across a restart (service init is the only latency).

Schedule types
  once   `fire_at` (absolute ISO/epoch) or `start_time` HH:MM (next
         occurrence). Disabled after it fires (or is missed).
  daily  `start_time` every day.
  weekly `start_time` on `days_of_week`.
Grace: `grace_s` if set, else the start_time..end_time window (min 60 s),
else 1 h for `fire_at` one-shots. A slot missed by more than the grace is
recorded as last_result="missed" and the schedule moves on to its next slot.

Results: the scheduler records the engine's store mission id as
`last_run_id` and listens on the system bus for mission.completed /
mission.stopped to stamp `last_result` success|fail|aborted — the same
vocabulary as the relay `mission_run` event, whose run_id == last_run_id.

Legacy: missions/*.json may embed {"schedule": {"auto_start": true, ...}}.
Those were never armed at boot before this rewrite, so they stay off unless
`legacy_embedded_schedules` is flipped on — enabling them would start the
multi-hour quiet missions on every unit.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.bus import get_bus, publish_system_event


# Singleton instance
_scheduler = None
_scheduler_lock = threading.Lock()

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DEFAULT_ONESHOT_GRACE_S = 3600.0
MIN_GRACE_S = 60.0


# ------------------------------------------------------------------ pure timer math

def _parse_hhmm(txt: Any, default: str) -> Tuple[int, int]:
    try:
        h, m = str(txt or default).split(":")[:2]
        return int(h), int(m)
    except (ValueError, AttributeError):
        h, m = default.split(":")
        return int(h), int(m)


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        from core.schedule_manager import parse_fire_at
        return parse_fire_at(value)
    except Exception:
        return None


def grace_seconds(schedule: Dict[str, Any]) -> float:
    """How late a slot may still fire."""
    g = schedule.get("grace_s")
    if g is not None:
        try:
            return max(MIN_GRACE_S, float(g))
        except (TypeError, ValueError):
            pass
    if schedule.get("fire_at"):
        return DEFAULT_ONESHOT_GRACE_S
    sh, sm = _parse_hhmm(schedule.get("start_time"), "00:00")
    eh, em = _parse_hhmm(schedule.get("end_time"), "23:59")
    start_min, end_min = sh * 60 + sm, eh * 60 + em
    span = (end_min - start_min) if end_min >= start_min else (end_min + 24 * 60 - start_min)
    return max(MIN_GRACE_S, float(span * 60))


def compute_next_fire(schedule: Dict[str, Any], now: datetime,
                      last_fired_at: Optional[datetime] = None) -> Tuple[Optional[datetime], List[datetime]]:
    """Return (next_fire_at, missed_slots).

    next_fire_at is the earliest slot that (a) has not been fired yet
    (last_fired_at < slot) and (b) is not past its grace. Slots that are
    past grace and unfired are returned in missed_slots (oldest first) so
    the caller can record them. `now` must be tz-aware local time.
    """
    stype = schedule.get("type", "daily")
    grace = timedelta(seconds=grace_seconds(schedule))
    missed: List[datetime] = []

    # Slots older than the schedule itself are not "missed" — the schedule
    # simply did not exist yet.
    created = _parse_iso(schedule.get("created_at"))

    def fired(slot: datetime) -> bool:
        return last_fired_at is not None and last_fired_at >= slot - timedelta(seconds=1)

    if stype == "once" and schedule.get("fire_at"):
        slot = _parse_iso(schedule["fire_at"])
        if slot is None or fired(slot):
            return None, missed
        if now > slot + grace:
            missed.append(slot.astimezone())
            return None, missed
        return slot.astimezone(), missed

    days = [str(d).lower() for d in (schedule.get("days_of_week") or [])] if stype == "weekly" else []
    times = schedule.get("times") or [schedule.get("start_time") or "00:00"]
    tz = _tz_of(schedule)
    local_now = now.astimezone(tz)

    # Walk day by day from yesterday (a slot can still be inside its grace
    # after midnight) until a live slot is found; several times a day are
    # independent slots and the earliest live one wins.
    best: Optional[datetime] = None
    for t in times:
        sh, sm = _parse_hhmm(t, "00:00")
        base = local_now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        for offset in range(-1, 9):
            slot = base + timedelta(days=offset)
            if days and slot.strftime("%A").lower() not in days:
                continue
            if fired(slot) or (created is not None and slot + grace < created):
                continue
            if now > slot + grace:
                # Only report the most recent missed slot; older ones predate the
                # schedule's own history and are noise.
                if offset <= 0 and (not missed or slot > missed[-1]):
                    missed = [slot]
                continue
            if best is None or slot < best:
                best = slot
            break
    # report in the robot's local offset (relay contract), whatever tz the
    # HH:MM were written in
    return (best.astimezone() if best else None), [m.astimezone() for m in missed]


def _tz_of(schedule: Dict[str, Any]):
    """IANA zone the schedule's HH:MM are written in (dog's local); robot local if unset/unknown."""
    name = schedule.get("tz")
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(str(name))
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo


# ------------------------------------------------------------------ scheduler

class MissionScheduler:
    """Persisted mission timer. One background thread, 5 s tick."""

    check_interval = 5.0            # seconds between ticks
    legacy_embedded_schedules = False

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bus = get_bus()

        self.enabled = False
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._started_at = time.time()

        # In-memory mirror of what fired (schedule_id -> datetime); the
        # persisted copy in the schedule file is the source of truth.
        self.last_started: Dict[str, datetime] = {}

        self._mission_engine = None
        self._schedule_manager = None
        self._bus_subscribed = False
        self._mode_run: Optional[Dict[str, Any]] = None

        self.logger.info("Mission scheduler initialized")

    # ------------------------------------------------------------- lazy deps

    @property
    def mission_engine(self):
        if self._mission_engine is None:
            from orchestrators.mission_engine import get_mission_engine
            self._mission_engine = get_mission_engine()
        return self._mission_engine

    @property
    def schedule_manager(self):
        if self._schedule_manager is None:
            from core.schedule_manager import get_schedule_manager
            self._schedule_manager = get_schedule_manager()
        return self._schedule_manager

    # ------------------------------------------------------------- lifecycle

    def enable(self) -> bool:
        with self._lock:
            if self.enabled:
                return True
            self.enabled = True
            self.running = True
            self._subscribe_bus()
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True, name="MissionScheduler")
            self.scheduler_thread.start()
        armed = self.rearm()
        self.logger.info(f"Mission scheduler enabled: {armed} schedule(s) armed")
        publish_system_event("scheduler.enabled", {"armed": armed})
        return True

    def disable(self) -> bool:
        with self._lock:
            if not self.enabled:
                return True
            self.enabled = False
            self.running = False
            self._wake.set()
        self.logger.info("Mission scheduler disabled")
        publish_system_event("scheduler.disabled", {})
        return True

    def poke(self) -> None:
        """Run a tick now (after a schedule was created/changed)."""
        self._wake.set()

    def _subscribe_bus(self) -> None:
        if self._bus_subscribed:
            return
        try:
            self.bus.subscribe("system", self._on_system_event)
            self._bus_subscribed = True
        except Exception as e:
            self.logger.error(f"Scheduler bus subscribe failed: {e}")

    # ------------------------------------------------------------- status

    def rearm(self) -> int:
        """Recompute + persist next_fire_at for every enabled schedule (boot).
        Returns the number of schedules with a pending slot."""
        armed = 0
        now = _now()
        for s in self.schedule_manager.get_active_schedules():
            nxt, missed = self._next_for(s, now)
            self._record_missed(s, missed)
            fields = {"next_fire_at": nxt.isoformat() if nxt else None}
            if s.get("next_fire_at") != fields["next_fire_at"]:
                self.schedule_manager.set_runtime(s["schedule_id"], fields)
            if nxt:
                armed += 1
                self.logger.info(f"[SCHED] armed {s['schedule_id']} {s.get('mission_name')} "
                                 f"type={s.get('type')} next_fire_at={nxt.isoformat()}")
        return armed

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "running": self.running,
            "check_interval": self.check_interval,
            "now": _now().isoformat(),
            "last_started": {k: v.isoformat() for k, v in self.last_started.items()},
            "scheduled_missions": self.get_scheduled_missions(),
        }

    def get_scheduled_missions(self, dog_id: Optional[str] = None,
                               include_disabled: bool = True) -> List[Dict[str, Any]]:
        """schedule_sync rows: every schedule with next_fire_at / last_run_at / last_result."""
        now = _now()
        rows = []
        for s in self.schedule_manager.list_schedules():
            if dog_id and str(s.get("dog_id")) != str(dog_id):
                continue
            if not include_disabled and not s.get("enabled", True):
                continue
            rows.append(self._sync_row(s, now))
        rows.sort(key=lambda r: (r["next_fire_at"] is None, r["next_fire_at"] or "", r["schedule_id"]))
        return rows

    def _sync_row(self, s: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        nxt = None
        if s.get("enabled", True):
            nxt, _ = self._next_for(s, now)
        if s.get("unsupported"):
            nxt = None
        return {
            "id": s.get("mission_id") or s["schedule_id"],
            "mission_id": s.get("mission_id") or s["schedule_id"],
            "schedule_id": s["schedule_id"],
            "kind": s.get("kind") or "mission",
            "mode": s.get("mode"),
            "mission_type": s.get("mission_type"),
            "duration_s": s.get("duration_s"),
            "reward_treats": s.get("reward_treats"),
            "source": s.get("source"),
            "tz": s.get("tz"),
            "times": s.get("times") or ([s.get("start_time")] if s.get("start_time") else []),
            "unsupported": s.get("unsupported"),
            "mission_name": s.get("mission_name"),
            "name": s.get("name"),
            "dog_id": s.get("dog_id"),
            "type": s.get("type"),
            "enabled": bool(s.get("enabled", True)),
            "start_time": s.get("start_time"),
            "end_time": s.get("end_time"),
            "days_of_week": s.get("days_of_week") or [],
            "fire_at": s.get("fire_at"),
            "grace_s": grace_seconds(s),
            "next_fire_at": nxt.isoformat() if nxt else None,
            "last_run_at": s.get("last_run_at"),
            "last_run_end": s.get("last_run_end"),
            "last_run_id": s.get("last_run_id"),
            "last_result": s.get("last_result"),
            "missed_at": s.get("missed_at"),
        }

    # ------------------------------------------------------------- loop

    FIRST_TICK_DELAY_S = 3.0   # let the rest of init settle before the first fire

    def _scheduler_loop(self):
        self.logger.info("Scheduler loop started")
        self._wake.wait(self.FIRST_TICK_DELAY_S)
        self._wake.clear()
        while self.running:
            try:
                self._check_schedules()
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}", exc_info=True)
            self._wake.wait(self.check_interval)
            self._wake.clear()
        self.logger.info("Scheduler loop stopped")

    def _next_for(self, s: Dict[str, Any], now: datetime):
        return compute_next_fire(s, now, _parse_iso(s.get("last_fired_at")))

    def _record_missed(self, s: Dict[str, Any], missed: List[datetime]) -> None:
        if not missed:
            return
        slot = missed[-1]
        if s.get("missed_at") == slot.isoformat():
            return
        fields = {"missed_at": slot.isoformat(), "last_result": "missed"}
        self.logger.warning(f"[SCHED] missed {s['schedule_id']} {s.get('mission_name')} slot {slot.isoformat()} "
                            f"(grace {grace_seconds(s):.0f}s)")
        if s.get("type") == "once":
            self.schedule_manager.update_schedule(s["schedule_id"], {"enabled": False})
        self.schedule_manager.set_runtime(s["schedule_id"], fields)

    def _check_schedules(self):
        """One tick: fire every due schedule (one per tick — the engine runs
        one mission at a time)."""
        now = _now()
        due: List[Tuple[datetime, Dict[str, Any]]] = []
        for s in self.schedule_manager.get_active_schedules():
            if not s.get("mission_name"):
                continue
            nxt, missed = self._next_for(s, now)
            self._record_missed(s, missed)
            if nxt is None:
                if s.get("type") == "once" and not missed and s.get("enabled", True) and s.get("last_fired_at"):
                    # fired earlier but never disabled (crash between the two writes)
                    self.schedule_manager.update_schedule(s["schedule_id"], {"enabled": False})
                continue
            if nxt <= now:
                due.append((nxt, s))

        if self.legacy_embedded_schedules:
            self._check_legacy_embedded(now)

        if not due:
            return
        due.sort(key=lambda t: t[0])
        if self.mission_engine.active_session or self.active_mode_run():
            # Still inside grace: try again next tick. Beyond grace it is
            # recorded as missed by the loop above.
            return
        slot, s = due[0]
        self._fire(s, slot, now)

    def _fire(self, s: Dict[str, Any], slot: datetime, now: datetime) -> bool:
        sid = s["schedule_id"]
        name = s["mission_name"]
        late = (now - slot).total_seconds()
        self.logger.info(f"[SCHED] fire {sid} {name} slot={slot.isoformat()} late={late:.1f}s dog_id={s.get('dog_id')}")

        # Persist "fired" BEFORE starting so a crash in start_mission cannot
        # re-fire the same slot forever.
        fields = {"last_fired_at": now.isoformat(), "last_run_at": now.isoformat(),
                  "last_run_id": None, "last_run_end": None, "last_result": "starting"}
        self.schedule_manager.set_runtime(sid, fields)
        self.last_started[sid] = now

        if s.get("kind") == "mode":
            return self._fire_mode(s, sid, now, late)

        if name not in self.mission_engine.missions:
            self.schedule_manager.set_runtime(sid, {"last_result": "start_failed:unknown_mission"})
            if s.get("type") == "once":
                self.schedule_manager.update_schedule(sid, {"enabled": False})
            self.logger.warning(f"[SCHED] {sid}: mission {name!r} not loaded")
            return False

        ok = False
        try:
            ok = bool(self.mission_engine.start_mission(name, dog_id=s.get("dog_id")))
        except Exception as e:
            self.logger.error(f"[SCHED] start_mission raised: {e}", exc_info=True)

        if ok:
            run_id = None
            try:
                sess = self.mission_engine.active_session
                run_id = sess.mission_id if sess else None
            except Exception:
                pass
            self.schedule_manager.set_runtime(sid, {"last_run_id": run_id, "last_result": "running"})
            publish_system_event("scheduler.mission_started", {
                "mission_name": name, "schedule_id": sid, "run_id": run_id,
                "mission_id": s.get("mission_id") or sid, "scheduled": True, "late_s": round(late, 1)})
        else:
            reason = getattr(self.mission_engine, "last_failure_reason", "") or "start_failed"
            self.schedule_manager.set_runtime(sid, {"last_result": f"start_failed:{reason}"})
            self.logger.warning(f"[SCHED] failed to start scheduled mission {name}: {reason}")

        if s.get("type") == "once":
            self.schedule_manager.update_schedule(sid, {"enabled": False})
        nxt, _ = self._next_for(self.schedule_manager.get_schedule(sid) or s, _now())
        self.schedule_manager.set_runtime(sid, {"next_fire_at": nxt.isoformat() if nxt else None})
        return ok

    # ------------------------------------------------------------- results

    def _on_system_event(self, event) -> None:
        try:
            st = getattr(event, "subtype", None)
            if st not in ("mission.completed", "mission.stopped"):
                return
            d = getattr(event, "data", None) or {}
            self.record_result(d.get("mission_id"), st, d)
        except Exception as e:
            self.logger.error(f"[SCHED] result handler: {e}")

    def record_result(self, run_id: Any, subtype: str, data: Dict[str, Any]) -> Optional[str]:
        """Stamp last_result on the schedule whose last_run_id == run_id.
        Vocabulary matches the relay mission_run event."""
        if run_id is None:
            return None
        if subtype == "mission.stopped":
            result = "aborted"
        else:
            result = "success" if data.get("success", True) else "fail"
        for s in self.schedule_manager.list_schedules():
            if s.get("last_run_id") is not None and str(s.get("last_run_id")) == str(run_id):
                self.schedule_manager.set_runtime(s["schedule_id"], {
                    "last_result": result, "last_run_end": _now().isoformat()})
                self.logger.info(f"[SCHED] {s['schedule_id']} run {run_id} -> {result} "
                                 f"(reason={data.get('reason')}, rewards={data.get('rewards_given')})")
                return s["schedule_id"]
        return None

    def run_meta(self, run_id: Any) -> Optional[Dict[str, Any]]:
        """{schedule_id, mission_id (relay), dog_id, kind} for a run the scheduler started."""
        if run_id is None:
            return None
        for s in self.schedule_manager.list_schedules():
            if s.get("last_run_id") is not None and str(s.get("last_run_id")) == str(run_id):
                return {"schedule_id": s["schedule_id"], "mission_id": s.get("mission_id") or s["schedule_id"],
                        "dog_id": s.get("dog_id"), "kind": s.get("kind") or "mission",
                        "mission_type": s.get("mission_type")}
        return None

    # ------------------------------------------------------------- mode runs
    # A "mode" mission (relay types silent_guardian / coach) is a timed mode
    # override: set_mode(mode) for duration_s, then back to idle. It is
    # reported exactly like an engine mission (mission.started/.completed on
    # the bus -> mission_run event; active in schedule_sync) with no treats.

    MODE_MAX_DURATION_S = 43200.0     # 12 h cap (relay contract)
    MODE_DEFAULT_DURATION_S = 1800.0

    def _mode_set(self, mode: str, duration_s: float) -> bool:
        from core.state import SystemMode
        from orchestrators.mode_fsm import get_mode_fsm
        fsm = get_mode_fsm()
        m = SystemMode(mode)
        ok = fsm.set_mode_override(m, float(duration_s))
        if not ok:
            ok = fsm.force_mode(m, "scheduled mode mission")
        return bool(ok)

    def _mode_clear(self) -> None:
        from core.state import SystemMode, get_state
        from orchestrators.mode_fsm import get_mode_fsm
        fsm = get_mode_fsm()
        try:
            fsm.clear_override("scheduled mode mission ended")
        except Exception:
            pass
        if not fsm.set_mode_override(SystemMode.IDLE):
            get_state().set_mode(SystemMode.IDLE, reason="scheduled mode mission ended")

    def _new_run_id(self, name: str, config: Dict[str, Any]) -> Any:
        try:
            from core.store import get_store
            rid = get_store().start_mission(name=name, target_rewards=0, config=config)
            if rid is not None and rid >= 0:
                return rid
        except Exception as e:
            self.logger.debug(f"store.start_mission failed: {e}")
        return f"mode-{int(time.time())}"

    def _end_run_record(self, run_id: Any, status: str, results: Dict[str, Any]) -> None:
        if not isinstance(run_id, int):
            return
        try:
            from core.store import get_store
            get_store().end_mission(run_id, status=status, results=results)
        except Exception:
            pass

    def active_mode_run(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._mode_run
            return dict(r) if r else None

    def _fire_mode(self, s: Dict[str, Any], sid: str, now: datetime, late: float) -> bool:
        mode = s.get("mode")
        try:
            duration = float(s.get("duration_s") or self.MODE_DEFAULT_DURATION_S)
        except (TypeError, ValueError):
            duration = self.MODE_DEFAULT_DURATION_S
        duration = max(1.0, min(duration, self.MODE_MAX_DURATION_S))
        run_id = self._new_run_id(f"mode:{mode}", {"schedule_id": sid, "mission_id": s.get("mission_id"),
                                                     "duration_s": duration})
        ok = False
        try:
            ok = self._mode_set(mode, duration)
        except Exception as e:
            self.logger.error(f"[SCHED] mode set {mode} raised: {e}")
        if not ok:
            self._end_run_record(run_id, "failed", {"reason": "set_mode_failed"})
            self.schedule_manager.set_runtime(sid, {"last_run_id": run_id, "last_result": "start_failed:set_mode"})
            if s.get("type") == "once":
                self.schedule_manager.update_schedule(sid, {"enabled": False})
            return False

        ends_at = time.time() + duration
        with self._lock:
            self._mode_run = {"run_id": run_id, "schedule_id": sid, "mode": mode, "dog_id": s.get("dog_id"),
                              "mission_id": s.get("mission_id") or sid, "started": time.time(),
                              "ends_at": ends_at, "duration_s": duration}
        self.schedule_manager.set_runtime(sid, {"last_run_id": run_id, "last_result": "running"})
        publish_system_event("mission.started", {"mission_name": mode, "mission_id": run_id,
                                                 "dog_id": s.get("dog_id"), "kind": "mode",
                                                 "schedule_mission_id": s.get("mission_id") or sid,
                                                 "duration_s": duration})
        publish_system_event("scheduler.mission_started", {
            "mission_name": mode, "schedule_id": sid, "run_id": run_id, "kind": "mode",
            "mission_id": s.get("mission_id") or sid, "scheduled": True, "late_s": round(late, 1)})
        self.logger.info(f"[SCHED] mode run {run_id}: {mode} for {duration:.0f}s (schedule {sid})")
        threading.Thread(target=self._mode_run_wait, args=(run_id, ends_at), daemon=True,
                         name=f"ModeRun-{run_id}").start()
        if s.get("type") == "once":
            self.schedule_manager.update_schedule(sid, {"enabled": False})
        nxt, _ = self._next_for(self.schedule_manager.get_schedule(sid) or s, _now())
        self.schedule_manager.set_runtime(sid, {"next_fire_at": nxt.isoformat() if nxt else None})
        return True

    def _mode_run_wait(self, run_id: Any, ends_at: float) -> None:
        while self.running:
            with self._lock:
                cur = self._mode_run
            if not cur or cur.get("run_id") != run_id:
                return                      # cancelled / replaced
            if time.time() >= ends_at:
                break
            time.sleep(min(1.0, max(0.05, ends_at - time.time())))
        self.end_mode_run(run_id, reason="duration_elapsed")

    def end_mode_run(self, run_id: Any = None, reason: str = "duration_elapsed") -> bool:
        """Finish the active mode run: back to idle, mission.completed (or
        mission.stopped when cancelled) so mission_run goes out."""
        with self._lock:
            cur = self._mode_run
            if not cur or (run_id is not None and cur.get("run_id") != run_id):
                return False
            self._mode_run = None
        try:
            self._mode_clear()
        except Exception as e:
            self.logger.error(f"[SCHED] mode clear raised: {e}")
        cancelled = reason not in ("duration_elapsed",)
        results = {"reason": reason, "rewards_given": 0, "duration": time.time() - cur["started"],
                   "mode": cur["mode"]}
        self._end_run_record(cur["run_id"], "stopped" if cancelled else "completed", results)
        self.logger.info(f"[SCHED] mode run {cur['run_id']} {cur['mode']} ended ({reason})")
        subtype = "mission.stopped" if cancelled else "mission.completed"
        self.record_result(cur["run_id"], subtype, {"success": not cancelled})
        publish_system_event(subtype, {
            "mission_name": cur["mode"], "mission_id": cur["run_id"], "dog_id": cur.get("dog_id"),
            "success": not cancelled, "reason": reason, "rewards_given": 0, "kind": "mode",
            "schedule_mission_id": cur.get("mission_id")})
        return True

    # ------------------------------------------------------------- relay missions setting

    # relay mission `type` -> missions/*.json program (2026-11-07). Anything
    # not here (and not a loaded mission name itself) is kept but marked
    # unsupported: it shows in schedule_sync with next_fire_at null.
    TYPE_TO_MISSION = {
        "sit": "sit", "sit_stay": "sit_sustained", "stay": "sit_sustained",
        "down": "down_sustained", "down_stay": "down_sustained", "place": "down_sustained",
        "recall": "come_and_sit", "quiet": "quiet_progressive",
    }

    def _known_mission_names(self) -> set:
        """Mission programs available, WITHOUT constructing the engine early
        (the cached settings apply at boot before hardware services exist)."""
        if self._mission_engine is not None:
            return set(getattr(self._mission_engine, "missions", {}) or {})
        try:
            import orchestrators.mission_engine as me
            if getattr(me, "_mission_engine", None) is not None:
                return set(me._mission_engine.missions)
        except Exception:
            pass
        from pathlib import Path
        d = Path("/home/morgan/dogbot/missions")
        return {p.stem for p in d.glob("*.json")} if d.is_dir() else set()

    def apply_missions_setting(self, dog_id: Optional[str], value: Any) -> Dict[str, Any]:
        """Load the relay `missions` setting for one dog: every mission with
        trigger.kind == "schedule" becomes a persisted schedule (one per
        mission, all its times), replacing that dog's previous relay set.
        Runtime state (last_run_*, last_fired_at) carries over when the same
        mission id is still present, so a re-push never re-fires a slot."""
        missions = _missions_list(value)
        mgr = self.schedule_manager
        tz = None
        try:
            from services.cloud.remote_settings import get_remote_settings
            tz = get_remote_settings().get_timezone()
        except Exception:
            pass

        old = mgr.list_by_source("relay_settings", dog_id) if dog_id else mgr.list_by_source("relay_settings")
        carry = {str(s.get("mission_id")): {k: s.get(k) for k in
                                           ("last_fired_at", "last_run_at", "last_run_id", "last_run_end",
                                            "last_result", "missed_at")}
                 for s in old if s.get("mission_id")}
        active = self.active_mode_run()
        for s in old:
            # keep the file of a mission whose mode run is executing so the
            # run can still finish/cancel cleanly; it is rewritten below
            mgr.delete_schedule(s["schedule_id"])

        loaded, skipped = [], []
        loaded_names = self._known_mission_names()
        for m in missions:
            if not isinstance(m, dict):
                continue
            trig = m.get("trigger") or {}
            if isinstance(trig, str):
                trig = {"kind": trig}
            kind = str(trig.get("kind") or trig.get("type") or "").lower()
            mid = str(m.get("id") or m.get("mission_id") or "")
            if kind != "schedule" or not mid:
                skipped.append({"id": mid or None, "reason": f"trigger {kind or 'none'}"})
                continue
            times = trig.get("times") or m.get("times") or []
            if isinstance(times, str):
                times = [times]
            days = trig.get("days") or m.get("days") or []
            mtype = str(m.get("type") or "").lower()
            data = {
                "mission_id": mid, "dog_id": str(dog_id or m.get("dog_id") or "unknown"),
                "name": m.get("name") or mid, "times": list(times), "days_of_week": list(days),
                "type": "weekly" if days else "daily", "enabled": bool(m.get("enabled", True)),
                "source": "relay_settings", "tz": tz, "mission_type": mtype,
                "duration_s": m.get("duration_s"), "reward_treats": m.get("reward_treats"),
                "grace_s": m.get("grace_s"),
            }
            if mtype in ("silent_guardian", "coach"):
                data.update({"kind": "mode", "mode": mtype, "reward_treats": 0})
            else:
                prog = self.TYPE_TO_MISSION.get(mtype, mtype)
                if prog not in loaded_names:
                    data.update({"mission_name": prog or mtype or "unknown", "allow_unknown_mission": True,
                                 "unsupported": f"no mission program for type {mtype!r}"})
                else:
                    data["mission_name"] = prog
            res = mgr.create_schedule(data)
            if not res.get("success"):
                skipped.append({"id": mid, "reason": res.get("error")})
                self.logger.warning(f"[SCHED] missions setting: {mid} rejected: {res.get('error')}")
                continue
            sid = res["schedule"]["schedule_id"]
            if mid in carry and any(v is not None for v in carry[mid].values()):
                mgr.set_runtime(sid, {k: v for k, v in carry[mid].items() if v is not None})
            if active and str(active.get("mission_id")) == mid:
                with self._lock:
                    if self._mode_run:
                        self._mode_run["schedule_id"] = sid
            loaded.append(sid)

        armed = self.rearm() if self.enabled else 0
        self.logger.info(f"[SCHED] missions setting dog={dog_id}: {len(missions)} mission(s) -> "
                         f"{len(loaded)} scheduled, {len(skipped)} skipped, {len(old)} replaced; armed={armed}")
        self.poke()
        return {"loaded": loaded, "skipped": skipped, "replaced": len(old)}

    # ------------------------------------------------------------- commands

    def cancel(self, mission_id: str, stop_running: bool = True) -> Dict[str, Any]:
        """cancel_mission {mission_id}: remove the schedule that id names
        (schedule_id | relay mission_id | mission_name) and stop its run if
        it is the mission currently executing. With no matching schedule,
        stop the active mission when its name/id matches ("active" /
        "current" stop whatever is running)."""
        me = self.mission_engine
        sess = me.active_session
        active_name = sess.mission.name if sess else None
        active_run = sess.mission_id if sess else None

        mode_run = self.active_mode_run()
        s = self.schedule_manager.find_by_mission_id(mission_id)
        stopped = False
        if s is not None:
            sid = s["schedule_id"]
            if stop_running and sess and s.get("last_run_id") is not None \
                    and str(s.get("last_run_id")) == str(active_run):
                stopped = bool(me.stop_mission(reason="schedule_cancelled"))
            elif stop_running and mode_run and str(mode_run.get("run_id")) == str(s.get("last_run_id")):
                stopped = self.end_mode_run(mode_run["run_id"], reason="schedule_cancelled")
                active_run = mode_run["run_id"]
            res = self.schedule_manager.delete_schedule(sid)
            self.last_started.pop(sid, None)
            self.logger.info(f"[SCHED] cancel {mission_id!r} -> deleted {sid} stopped_run={stopped}")
            return {"ok": bool(res.get("success")), "cancelled": sid,
                    "mission_id": s.get("mission_id") or sid, "mission_name": s.get("mission_name"),
                    "stopped_run": stopped, "run_id": active_run if stopped else None}

        if sess and stop_running and (str(mission_id) in ("active", "current")
                                      or str(mission_id) in (str(active_name), str(active_run))):
            stopped = bool(me.stop_mission(reason="schedule_cancelled"))
            return {"ok": stopped, "cancelled": None, "mission_name": active_name,
                    "stopped_run": stopped, "run_id": active_run}
        if mode_run and stop_running and (str(mission_id) in ("active", "current")
                                          or str(mission_id) in (str(mode_run["mode"]), str(mode_run["run_id"]),
                                                                 str(mode_run.get("mission_id")))):
            stopped = self.end_mode_run(mode_run["run_id"], reason="schedule_cancelled")
            return {"ok": stopped, "cancelled": None, "mission_name": mode_run["mode"],
                    "stopped_run": stopped, "run_id": mode_run["run_id"]}

        return {"ok": False, "error": {"code": "not_found",
                                       "message": f"no schedule or running mission matches {mission_id!r}"},
                "active": active_name}

    def force_start(self, mission_name: str) -> bool:
        if mission_name not in self.mission_engine.missions:
            return False
        ok = self.mission_engine.start_mission(mission_name)
        if ok:
            self.last_started[mission_name] = _now()
        return ok

    # ------------------------------------------------------------- legacy

    def _check_legacy_embedded(self, now: datetime) -> None:
        """missions/*.json embedded auto_start windows (off by default)."""
        if self.mission_engine.active_session:
            return
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%A").lower()
        for name, mission in self.mission_engine.missions.items():
            schedule = mission.schedule
            if not mission.enabled or not isinstance(schedule, dict) or not schedule.get("auto_start"):
                continue
            days = schedule.get("days_of_week", [])
            if days and current_day not in days:
                continue
            if not _time_in_window(current_time, schedule.get("start_time", "00:00"), schedule.get("end_time", "23:59")):
                continue
            last = self.last_started.get(name)
            if last and (now - last).total_seconds() < schedule.get("cooldown_hours", 24) * 3600:
                continue
            if self.mission_engine.start_mission(name, dog_id=schedule.get("dog_id")):
                self.last_started[name] = now
                publish_system_event("scheduler.mission_started", {"mission_name": name, "scheduled": True})
            return


def _now() -> datetime:
    return datetime.now().astimezone()


def _missions_list(value: Any) -> List[Dict[str, Any]]:
    """Accept {"missions":[...]}, [...], or the relay's {"key","value"} wrapper."""
    if isinstance(value, dict) and "value" in value and "key" in value:
        value = value.get("value")
    if isinstance(value, dict):
        value = value.get("missions") or value.get("items") or []
    return list(value) if isinstance(value, list) else []


def _time_in_window(current: str, start: str, end: str) -> bool:
    try:
        c = _parse_hhmm(current, "00:00"); s = _parse_hhmm(start, "00:00"); e = _parse_hhmm(end, "23:59")
        cm, sm, em = c[0] * 60 + c[1], s[0] * 60 + s[1], e[0] * 60 + e[1]
        return (cm >= sm or cm <= em) if em < sm else (sm <= cm <= em)
    except Exception:
        return False


def get_mission_scheduler() -> MissionScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = MissionScheduler()
        return _scheduler
