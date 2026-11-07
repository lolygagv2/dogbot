#!/usr/bin/env python3
"""
Schedule Manager for WIM-Z
Handles CRUD operations for training schedules

Handles CRUD operations for training schedules created via the app.
"""

import os
import json
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Singleton
_manager = None
_manager_lock = threading.Lock()

# Overridable so tests never touch the live schedules (2026-11-07).
SCHEDULES_DIR = os.environ.get("WIMZ_SCHEDULES_DIR") or "/home/morgan/dogbot/schedules"

# Runtime fields the MissionScheduler persists next to the user's fields so
# timers survive a restart (2026-11-07). They are never set by the app.
RUNTIME_FIELDS = ("last_fired_at", "last_run_at", "last_run_id", "last_run_end",
                  "last_result", "missed_at", "next_fire_at")

VALID_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
_DAY_ABBR = {d[:3]: d for d in VALID_DAYS}
_DAY_ABBR.update({'tues': 'tuesday', 'thur': 'thursday', 'thurs': 'thursday'})

# Mission kinds. "mission" runs a missions/*.json program through the engine;
# "mode" (relay types silent_guardian / coach, 2026-11-07) switches the robot
# into that mode for duration_s and back to idle — no treats.
MODE_KINDS = ('silent_guardian', 'coach')


def normalize_days(days: Any) -> List[str]:
    """['mon','Tue','wednesday'] -> ['monday','tuesday','wednesday']; raises on junk."""
    out: List[str] = []
    for d in (days or []):
        key = str(d).strip().lower()
        full = _DAY_ABBR.get(key, key)
        if full not in VALID_DAYS:
            raise ValueError(f"Invalid day {d!r}. Use: {VALID_DAYS} or mon..sun")
        if full not in out:
            out.append(full)
    return out


def normalize_times(times: Any, start_time: Any = None) -> List[str]:
    """Validate a list of "HH:MM" (or the single start_time) -> sorted unique list."""
    raw = list(times) if isinstance(times, (list, tuple)) else ([start_time] if start_time else [])
    out: List[str] = []
    for t in raw:
        txt = str(t).strip()
        datetime.strptime(txt, "%H:%M")
        h, m = txt.split(":")
        norm = f"{int(h):02d}:{int(m):02d}"
        if norm not in out:
            out.append(norm)
    return sorted(out)


def _write_json_atomic(filepath: str, obj: Dict[str, Any]) -> None:
    """tmp + fsync + os.replace so a power cut mid-write can't leave a torn file."""
    # unique per writer: two threads persisting the same schedule at once
    # must not share a tmp name (one os.replace would ENOENT the other)
    tmp = f"{filepath}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filepath)


def parse_fire_at(value: Any) -> Optional[datetime]:
    """ISO-8601 (with or without offset) or epoch seconds -> aware local datetime.
    Naive ISO strings are taken as local wall-clock time (same as start_time)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value)).astimezone()
    txt = str(value).strip()
    if txt.endswith('Z'):
        txt = txt[:-1] + '+00:00'
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.astimezone()          # interpret as local
    return dt.astimezone()


class ScheduleManager:
    """
    Manages training schedules stored as JSON files

    Each schedule links a mission to a time window and days of week.
    The MissionScheduler checks these when deciding what to run.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._schedules: Dict[str, Dict[str, Any]] = {}

        # Ensure schedules directory exists
        os.makedirs(SCHEDULES_DIR, exist_ok=True)

        # Load existing schedules
        self._load_schedules()

        self.logger.info(f"ScheduleManager initialized with {len(self._schedules)} schedules")

    def _load_schedules(self):
        """Load all schedules from disk"""
        with self._lock:
            self._schedules.clear()

            if not os.path.isdir(SCHEDULES_DIR):
                return

            for filename in os.listdir(SCHEDULES_DIR):
                if not filename.endswith('.json'):
                    if filename.endswith('.tmp'):
                        try:
                            os.remove(os.path.join(SCHEDULES_DIR, filename))   # torn write leftover
                        except OSError:
                            pass
                    continue

                filepath = os.path.join(SCHEDULES_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        schedule = json.load(f)
                        # Support both old 'id' and new 'schedule_id' fields
                        schedule_id = schedule.get('schedule_id', schedule.get('id', filename[:-5]))
                        self._schedules[schedule_id] = schedule
                except Exception as e:
                    self.logger.error(f"Failed to load schedule {filename}: {e}")

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all schedules"""
        with self._lock:
            return list(self._schedules.values())

    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific schedule by ID"""
        with self._lock:
            return self._schedules.get(schedule_id)

    def create_schedule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new schedule

        Required fields:
        - name: Display name for the schedule
        - mission_name: Name of mission to run (must exist)
        - dog_id: ID of the dog this schedule is for
        - start_time: Start time in HH:MM format
        - end_time: End time in HH:MM format

        Optional fields:
        - type: Schedule type - "once", "daily", or "weekly" (default: "daily")
        - days_of_week: List of day names (required for "weekly" type)
        - enabled: Whether schedule is active (default: True)
        - cooldown_hours: Hours between runs (default: 24)
        """
        # One-shot at an absolute instant (relay schedule_mission / tests):
        # fire_at wins over start_time and forces type "once".
        fire_at_dt = None
        if data.get('fire_at') or data.get('at'):
            try:
                fire_at_dt = parse_fire_at(data.get('fire_at') or data.get('at'))
            except (ValueError, TypeError) as e:
                return {"success": False, "error": f"Invalid fire_at: {e}"}
            data['type'] = 'once'
            data['start_time'] = fire_at_dt.strftime("%H:%M")
            data['fire_at'] = fire_at_dt.isoformat()

        # Several fire times a day (relay missions setting: trigger.times).
        try:
            times = normalize_times(data.get('times'), data.get('start_time'))
        except ValueError:
            return {"success": False, "error": "Invalid time format in times/start_time. Use HH:MM"}
        if times and not data.get('start_time'):
            data['start_time'] = times[0]
        kind = data.get('kind') or ('mode' if data.get('mode') in MODE_KINDS else 'mission')
        if kind == 'mode':
            if data.get('mode') not in MODE_KINDS:
                return {"success": False, "error": f"mode must be one of {list(MODE_KINDS)}"}
            data.setdefault('mission_name', f"mode:{data['mode']}")

        # Validate required fields (name and end_time are optional with defaults)
        required = ['mission_name', 'dog_id', 'start_time']
        missing = [f for f in required if f not in data or not data[f]]
        if missing:
            return {
                "success": False,
                "error": f"Missing required fields: {', '.join(missing)}"
            }

        # Auto-generate name if not provided
        if not data.get('name'):
            data['name'] = f"{data['mission_name']} Schedule"

        # Auto-generate end_time if not provided (start_time + 1 hour)
        if not data.get('end_time'):
            try:
                start = datetime.strptime(data['start_time'], "%H:%M")
                end = start + timedelta(hours=1)
                data['end_time'] = end.strftime("%H:%M")
            except ValueError:
                # Let the time format validation below catch this
                pass

        # Validate type field
        schedule_type = data.get('type', 'daily')
        valid_types = ['once', 'daily', 'weekly']
        if schedule_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid type '{schedule_type}'. Must be one of: {valid_types}"
            }

        # For weekly type, days_of_week is required
        if schedule_type == 'weekly':
            if 'days_of_week' not in data or not data['days_of_week']:
                return {
                    "success": False,
                    "error": "days_of_week is required for 'weekly' type schedules"
                }

        # Validate mission exists (mode schedules do not run a mission program)
        if kind != 'mode' and not data.get('allow_unknown_mission'):
            from orchestrators.mission_engine import get_mission_engine
            engine = get_mission_engine()
            if data['mission_name'] not in engine.missions:
                available = list(engine.missions.keys())
                return {
                    "success": False,
                    "error": f"Mission '{data['mission_name']}' not found. Available: {available[:5]}..."
                }

        # Validate time format
        for field in ['start_time', 'end_time']:
            try:
                datetime.strptime(data[field], "%H:%M")
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid time format for {field}. Use HH:MM"
                }

        # Validate days of week if provided (full names or mon..sun)
        try:
            days_of_week = normalize_days(data.get('days_of_week') or data.get('days') or [])
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # Generate ID and timestamps
        schedule_id = str(uuid.uuid4())[:8]
        schedule = {
            "schedule_id": schedule_id,
            "dog_id": data['dog_id'],
            "name": data['name'],
            "mission_name": data['mission_name'],
            "type": schedule_type,
            "start_time": data['start_time'],
            "end_time": data['end_time'],
            "days_of_week": days_of_week,
            "times": times,
            "enabled": data.get('enabled', True),
            "cooldown_hours": data.get('cooldown_hours', 24),
            # mission | mode; mode schedules carry mode + duration_s
            "kind": kind,
            "mode": data.get('mode') if kind == 'mode' else None,
            "mission_type": data.get('mission_type'),
            "duration_s": data.get('duration_s'),
            "reward_treats": data.get('reward_treats'),
            # IANA zone the HH:MM times are expressed in (dog's local); None = robot local
            "tz": data.get('tz') or data.get('timezone'),
            # app | relay_settings | agent ... — relay_settings sets are replaced wholesale
            "source": data.get('source') or 'app',
            "unsupported": data.get('unsupported'),
            # Relay-side mission id (agent drafts) so cancel_mission/schedule_sync
            # can address the schedule by the id the owner's app knows.
            "mission_id": data.get('mission_id') or None,
            "fire_at": data.get('fire_at') or None,
            # How late a fire may be (restart / robot busy) and still run.
            "grace_s": data.get('grace_s'),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "last_fired_at": None,
            "last_run_at": None,
            "last_run_id": None,
            "last_result": None,
        }

        # Save to disk
        filepath = os.path.join(SCHEDULES_DIR, f"{schedule_id}.json")
        try:
            _write_json_atomic(filepath, schedule)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save schedule: {e}"
            }

        # Add to memory cache
        with self._lock:
            self._schedules[schedule_id] = schedule

        self.logger.info(f"Created schedule: {schedule['name']} ({schedule_id}) "
                         f"type={schedule_type} fire_at={schedule['fire_at']} start={schedule['start_time']}")

        return {
            "success": True,
            "schedule": schedule
        }

    def update_schedule(self, schedule_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing schedule"""
        with self._lock:
            if schedule_id not in self._schedules:
                return {
                    "success": False,
                    "error": f"Schedule '{schedule_id}' not found"
                }

            schedule = self._schedules[schedule_id].copy()

        # Validate mission if being updated
        if 'mission_name' in data:
            from orchestrators.mission_engine import get_mission_engine
            engine = get_mission_engine()
            if data['mission_name'] not in engine.missions:
                return {
                    "success": False,
                    "error": f"Mission '{data['mission_name']}' not found"
                }

        # Validate type if being updated
        if 'type' in data:
            valid_types = ['once', 'daily', 'weekly']
            if data['type'] not in valid_types:
                return {
                    "success": False,
                    "error": f"Invalid type '{data['type']}'. Must be one of: {valid_types}"
                }

        # Validate times if being updated
        for field in ['start_time', 'end_time']:
            if field in data:
                try:
                    datetime.strptime(data[field], "%H:%M")
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid time format for {field}. Use HH:MM"
                    }

        # Validate days if being updated
        if 'days_of_week' in data:
            try:
                data['days_of_week'] = normalize_days(data['days_of_week'])
            except ValueError as e:
                return {"success": False, "error": str(e)}
        if 'times' in data:
            try:
                data['times'] = normalize_times(data['times'])
            except ValueError:
                return {"success": False, "error": "Invalid time format in times. Use HH:MM"}
            if data['times']:
                data.setdefault('start_time', data['times'][0])

        # Update allowed fields
        updatable = ['name', 'mission_name', 'dog_id', 'type', 'start_time', 'end_time', 'days_of_week',
                     'times', 'enabled', 'cooldown_hours', 'mission_id', 'grace_s', 'kind', 'mode',
                     'mission_type', 'duration_s', 'reward_treats', 'tz', 'source', 'unsupported']
        for field in updatable:
            if field in data:
                schedule[field] = data[field]

        schedule['updated_at'] = datetime.utcnow().isoformat() + "Z"

        # Save to disk
        filepath = os.path.join(SCHEDULES_DIR, f"{schedule_id}.json")
        try:
            _write_json_atomic(filepath, schedule)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save schedule: {e}"
            }

        # Update memory cache
        with self._lock:
            self._schedules[schedule_id] = schedule

        self.logger.info(f"Updated schedule: {schedule['name']} ({schedule_id})")

        return {
            "success": True,
            "schedule": schedule
        }

    def delete_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """Delete a schedule"""
        with self._lock:
            if schedule_id not in self._schedules:
                return {
                    "success": False,
                    "error": f"Schedule '{schedule_id}' not found"
                }

            schedule = self._schedules.pop(schedule_id)

        # Remove from disk
        filepath = os.path.join(SCHEDULES_DIR, f"{schedule_id}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            self.logger.error(f"Failed to delete schedule file: {e}")
            # Already removed from memory, continue

        self.logger.info(f"Deleted schedule: {schedule['name']} ({schedule_id})")

        return {
            "success": True,
            "deleted": schedule_id
        }

    def get_active_schedules(self) -> List[Dict[str, Any]]:
        """Get all enabled schedules for the scheduler to check"""
        with self._lock:
            return [s for s in self._schedules.values() if s.get('enabled', True)]

    def disable_schedule(self, schedule_id: str) -> bool:
        """
        Disable a schedule (used by scheduler for 'once' type after execution)
        Returns True if successfully disabled
        """
        result = self.update_schedule(schedule_id, {"enabled": False})
        return result.get("success", False)

    def set_runtime(self, schedule_id: str, fields: Dict[str, Any]) -> bool:
        """Persist scheduler runtime state (last_fired_at, last_result, ...)
        for one schedule. Only RUNTIME_FIELDS are accepted; written through
        to disk at once so a restart re-arms from the same facts."""
        bad = [k for k in fields if k not in RUNTIME_FIELDS]
        if bad:
            raise ValueError(f"not runtime fields: {bad}")
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if schedule is None:
                return False
            schedule.update(fields)
            snapshot = dict(schedule)
        filepath = os.path.join(SCHEDULES_DIR, f"{schedule_id}.json")
        try:
            _write_json_atomic(filepath, snapshot)
            return True
        except Exception as e:
            self.logger.error(f"Failed to persist runtime state for {schedule_id}: {e}")
            return False

    def find_by_mission_id(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Resolve the id a relay/agent uses: schedule_id, the relay mission_id
        stored at creation, or (last) a mission_name. Enabled schedules win."""
        if not mission_id:
            return None
        mid = str(mission_id)
        with self._lock:
            rows = list(self._schedules.values())
        for key in ('schedule_id', 'mission_id', 'mission_name'):
            hits = [s for s in rows if str(s.get(key) or '') == mid]
            if hits:
                hits.sort(key=lambda s: (not s.get('enabled', True), s.get('created_at') or ''))
                return hits[0]
        return None

    def list_by_source(self, source: str, dog_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return [s for s in self._schedules.values()
                    if s.get('source') == source and (dog_id is None or str(s.get('dog_id')) == str(dog_id))]

    def list_schedules_for_dog(self, dog_id: str) -> List[Dict[str, Any]]:
        """List schedules for a specific dog"""
        with self._lock:
            return [s for s in self._schedules.values() if s.get('dog_id') == dog_id]


def get_schedule_manager() -> ScheduleManager:
    """Get or create schedule manager singleton"""
    global _manager

    with _manager_lock:
        if _manager is None:
            _manager = ScheduleManager()
        return _manager
