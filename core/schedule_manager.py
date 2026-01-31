#!/usr/bin/env python3
"""
Schedule Manager for WIM-Z
Handles CRUD operations for training schedules

BUILD 35: Added to support app schedule creation/management
"""

import os
import json
import uuid
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

# Singleton
_manager = None
_manager_lock = threading.Lock()

SCHEDULES_DIR = "/home/morgan/dogbot/schedules"


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
        # Validate required fields
        required = ['name', 'mission_name', 'dog_id', 'start_time', 'end_time']
        missing = [f for f in required if f not in data or not data[f]]
        if missing:
            return {
                "success": False,
                "error": f"Missing required fields: {', '.join(missing)}"
            }

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

        # Validate mission exists
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

        # Validate days of week if provided
        days_of_week = data.get('days_of_week', [])
        if days_of_week:
            valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            invalid_days = [d for d in days_of_week if d.lower() not in valid_days]
            if invalid_days:
                return {
                    "success": False,
                    "error": f"Invalid days: {invalid_days}. Use: {valid_days}"
                }
            # Normalize days to lowercase
            days_of_week = [d.lower() for d in days_of_week]

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
            "enabled": data.get('enabled', True),
            "cooldown_hours": data.get('cooldown_hours', 24),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }

        # Save to disk
        filepath = os.path.join(SCHEDULES_DIR, f"{schedule_id}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(schedule, f, indent=2)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save schedule: {e}"
            }

        # Add to memory cache
        with self._lock:
            self._schedules[schedule_id] = schedule

        self.logger.info(f"Created schedule: {schedule['name']} ({schedule_id})")

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
            valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            invalid_days = [d for d in data['days_of_week'] if d.lower() not in valid_days]
            if invalid_days:
                return {
                    "success": False,
                    "error": f"Invalid days: {invalid_days}"
                }
            data['days_of_week'] = [d.lower() for d in data['days_of_week']]

        # Update allowed fields
        updatable = ['name', 'mission_name', 'dog_id', 'type', 'start_time', 'end_time', 'days_of_week', 'enabled', 'cooldown_hours']
        for field in updatable:
            if field in data:
                schedule[field] = data[field]

        schedule['updated_at'] = datetime.utcnow().isoformat() + "Z"

        # Save to disk
        filepath = os.path.join(SCHEDULES_DIR, f"{schedule_id}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(schedule, f, indent=2)
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
