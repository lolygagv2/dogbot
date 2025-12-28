#!/usr/bin/env python3
"""
Mission Scheduler for WIM-Z
Automatically starts missions based on schedule configuration
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from core.bus import get_bus, publish_system_event
from core.store import get_store


# Singleton instance
_scheduler = None
_scheduler_lock = threading.Lock()


class MissionScheduler:
    """
    Automatic mission scheduler

    Checks mission schedules and starts missions based on:
    - Time windows (start_time, end_time)
    - Days of week
    - auto_start flag
    - Daily limits and cooldowns
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bus = get_bus()
        self.store = get_store()

        # Scheduler state
        self.enabled = False
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Track last execution per mission
        self.last_started: Dict[str, datetime] = {}

        # Check interval (seconds)
        self.check_interval = 60  # Check every minute

        # Mission engine reference (set later to avoid circular import)
        self._mission_engine = None

        self.logger.info("Mission scheduler initialized")

    @property
    def mission_engine(self):
        """Lazy load mission engine to avoid circular import"""
        if self._mission_engine is None:
            from orchestrators.mission_engine import get_mission_engine
            self._mission_engine = get_mission_engine()
        return self._mission_engine

    def enable(self) -> bool:
        """Enable auto-scheduling"""
        with self._lock:
            if self.enabled:
                self.logger.warning("Scheduler already enabled")
                return True

            self.enabled = True
            self.running = True
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="MissionScheduler"
            )
            self.scheduler_thread.start()

            self.logger.info("Mission scheduler enabled")
            publish_system_event("scheduler.enabled", {})
            return True

    def disable(self) -> bool:
        """Disable auto-scheduling"""
        with self._lock:
            if not self.enabled:
                return True

            self.enabled = False
            self.running = False

            self.logger.info("Mission scheduler disabled")
            publish_system_event("scheduler.disabled", {})
            return True

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "check_interval": self.check_interval,
            "last_started": {
                name: ts.isoformat()
                for name, ts in self.last_started.items()
            },
            "scheduled_missions": self.get_scheduled_missions()
        }

    def get_scheduled_missions(self) -> List[Dict[str, Any]]:
        """Get list of missions with auto-start schedules"""
        missions = []

        for name, mission in self.mission_engine.missions.items():
            schedule = mission.schedule

            # Skip non-scheduled missions
            if isinstance(schedule, str):
                if schedule in ("manual", "continuous"):
                    continue

            # Parse schedule config
            if isinstance(schedule, dict):
                if not schedule.get("auto_start", False):
                    continue

                missions.append({
                    "name": name,
                    "description": mission.description,
                    "enabled": mission.enabled,
                    "schedule_type": schedule.get("type", "unknown"),
                    "start_time": schedule.get("start_time", "00:00"),
                    "end_time": schedule.get("end_time", "23:59"),
                    "days_of_week": schedule.get("days_of_week", []),
                    "next_run": self._calculate_next_run(schedule),
                    "last_run": self.last_started.get(name, None)
                })

        return missions

    def _scheduler_loop(self):
        """Main scheduler loop - runs in background thread"""
        self.logger.info("Scheduler loop started")

        while self.running:
            try:
                self._check_schedules()
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}", exc_info=True)

            # Sleep in small increments to respond to stop quickly
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

        self.logger.info("Scheduler loop stopped")

    def _check_schedules(self):
        """Check all mission schedules and start if conditions met"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%A").lower()

        # Don't start if mission already active
        if self.mission_engine.active_session:
            return

        for name, mission in self.mission_engine.missions.items():
            if not mission.enabled:
                continue

            schedule = mission.schedule

            # Handle dictionary schedule (auto-start config)
            if isinstance(schedule, dict):
                if not schedule.get("auto_start", False):
                    continue

                if self._should_start_mission(name, schedule, now, current_time, current_day):
                    self._start_scheduled_mission(name)
                    break  # Only start one mission at a time

    def _should_start_mission(
        self,
        name: str,
        schedule: Dict[str, Any],
        now: datetime,
        current_time: str,
        current_day: str
    ) -> bool:
        """Check if mission should be started based on schedule"""

        # Check day of week
        days = schedule.get("days_of_week", [])
        if days and current_day not in days:
            return False

        # Check time window
        start_time = schedule.get("start_time", "00:00")
        end_time = schedule.get("end_time", "23:59")

        if not self._time_in_window(current_time, start_time, end_time):
            return False

        # Check cooldown (don't start same mission twice in same day by default)
        if name in self.last_started:
            last = self.last_started[name]

            # Default: once per day
            cooldown_hours = schedule.get("cooldown_hours", 24)
            if (now - last).total_seconds() < cooldown_hours * 3600:
                return False

        # Check daily limits (via mission engine)
        if self.mission_engine._check_daily_limits(None):
            return False

        return True

    def _time_in_window(self, current: str, start: str, end: str) -> bool:
        """Check if current time is within window"""
        try:
            curr_parts = [int(x) for x in current.split(":")]
            start_parts = [int(x) for x in start.split(":")]
            end_parts = [int(x) for x in end.split(":")]

            curr_mins = curr_parts[0] * 60 + curr_parts[1]
            start_mins = start_parts[0] * 60 + start_parts[1]
            end_mins = end_parts[0] * 60 + end_parts[1]

            # Handle overnight windows (e.g., 22:00 - 06:00)
            if end_mins < start_mins:
                return curr_mins >= start_mins or curr_mins <= end_mins
            else:
                return start_mins <= curr_mins <= end_mins

        except (ValueError, IndexError):
            return False

    def _calculate_next_run(self, schedule: Dict[str, Any]) -> Optional[str]:
        """Calculate next scheduled run time"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%A").lower()

        start_time = schedule.get("start_time", "08:00")
        days = schedule.get("days_of_week", [])

        # If we're within the window today, next run could be now
        if current_day in days or not days:
            if current_time < start_time:
                return f"Today at {start_time}"

        # Find next scheduled day
        if days:
            day_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            current_idx = day_order.index(current_day)

            for i in range(1, 8):
                next_idx = (current_idx + i) % 7
                next_day = day_order[next_idx]
                if next_day in days:
                    return f"{next_day.capitalize()} at {start_time}"

        return f"Tomorrow at {start_time}"

    def _start_scheduled_mission(self, name: str):
        """Start a scheduled mission"""
        self.logger.info(f"Auto-starting scheduled mission: {name}")

        success = self.mission_engine.start_mission(name)

        if success:
            self.last_started[name] = datetime.now()
            publish_system_event("scheduler.mission_started", {
                "mission_name": name,
                "scheduled": True
            })
        else:
            self.logger.warning(f"Failed to start scheduled mission: {name}")

    def force_start(self, mission_name: str) -> bool:
        """Force start a mission regardless of schedule"""
        if mission_name not in self.mission_engine.missions:
            return False

        success = self.mission_engine.start_mission(mission_name)
        if success:
            self.last_started[mission_name] = datetime.now()
        return success


def get_mission_scheduler() -> MissionScheduler:
    """Get or create mission scheduler singleton"""
    global _scheduler

    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = MissionScheduler()
        return _scheduler
