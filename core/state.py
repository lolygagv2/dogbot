#!/usr/bin/env python3
"""
Global state manager for TreatBot
Thread-safe state tracking for system mode, missions, and hardware
"""

import threading
import time
from typing import Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import json
import logging


class SystemMode(Enum):
    """System operation modes"""
    IDLE = "idle"
    SILENT_GUARDIAN = "silent_guardian"  # Primary mode - bark-focused passive monitoring
    COACH = "coach"                       # Opportunistic trick training when dog approaches
    MISSION = "mission"                   # Running a structured mission/program
    PHOTOGRAPHY = "photography"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

    # Backward compatibility aliases (deprecated)
    @classmethod
    def _missing_(cls, value):
        """Handle deprecated mode names"""
        deprecated_map = {
            "detection": cls.COACH,
            "vigilant": cls.SILENT_GUARDIAN,
            "guardian": cls.SILENT_GUARDIAN,  # app alias
            "training": cls.COACH,  # app alias
        }
        if value in deprecated_map:
            import logging
            logging.getLogger('SystemMode').warning(
                f"Deprecated mode name '{value}' used. Use '{deprecated_map[value].value}' instead."
            )
            return deprecated_map[value]
        return None


class MissionState(Enum):
    """Mission execution states"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class HardwareState:
    """Hardware component states"""
    motors_initialized: bool = False
    servos_initialized: bool = False
    audio_initialized: bool = False
    leds_initialized: bool = False
    camera_initialized: bool = False

    motor_moving: bool = False
    servo_positions: Dict[str, float] = field(default_factory=dict)
    audio_playing: bool = False
    led_pattern: str = "off"

    battery_voltage: float = 0.0
    temperature: float = 0.0

    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'motors_initialized': self.motors_initialized,
            'servos_initialized': self.servos_initialized,
            'audio_initialized': self.audio_initialized,
            'leds_initialized': self.leds_initialized,
            'camera_initialized': self.camera_initialized,
            'motor_moving': self.motor_moving,
            'servo_positions': self.servo_positions,
            'audio_playing': self.audio_playing,
            'led_pattern': self.led_pattern,
            'battery_voltage': self.battery_voltage,
            'temperature': self.temperature,
            'last_updated': self.last_updated
        }


@dataclass
class MissionStatus:
    """Current mission status"""
    name: str = ""
    state: MissionState = MissionState.INACTIVE
    progress: float = 0.0
    current_stage: str = ""
    rewards_given: int = 0
    max_rewards: int = 5
    start_time: float = 0.0
    duration_limit: float = 1800.0  # 30 minutes default

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'state': self.state.value,
            'progress': self.progress,
            'current_stage': self.current_stage,
            'rewards_given': self.rewards_given,
            'max_rewards': self.max_rewards,
            'start_time': self.start_time,
            'duration_limit': self.duration_limit,
            'elapsed_time': time.time() - self.start_time if self.start_time > 0 else 0
        }


@dataclass
class DetectionStatus:
    """Current detection status"""
    dogs_detected: int = 0
    active_dog_id: Optional[str] = None
    last_detection_time: float = 0.0
    current_behavior: str = ""
    behavior_confidence: float = 0.0
    behavior_duration: float = 0.0
    pose_stable: bool = False
    # Dog identification
    dog_name: str = ""  # Identified dog name (e.g., "Bezik", "Elsa", "Dog")
    id_method: str = ""  # Identification method: aruco, color, persistence, unknown

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dogs_detected': self.dogs_detected,
            'active_dog_id': self.active_dog_id,
            'dog_name': self.dog_name,
            'id_method': self.id_method,
            'last_detection_time': self.last_detection_time,
            'current_behavior': self.current_behavior,
            'behavior_confidence': self.behavior_confidence,
            'behavior_duration': self.behavior_duration,
            'pose_stable': self.pose_stable,
            'time_since_detection': time.time() - self.last_detection_time if self.last_detection_time > 0 else 999
        }


class StateManager:
    """
    Thread-safe global state manager for TreatBot
    Tracks system mode, mission status, hardware state, and detection status
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.logger = logging.getLogger('StateManager')

        # Core state
        self.mode = SystemMode.IDLE
        self.previous_mode = SystemMode.IDLE
        self.mode_changed_time = time.time()

        # Component states
        self.hardware = HardwareState()
        self.mission = MissionStatus()
        self.detection = DetectionStatus()

        # State change listeners
        self._listeners: Dict[str, Set[callable]] = {
            'mode_change': set(),
            'mission_change': set(),
            'hardware_change': set(),
            'detection_change': set()
        }

        # Emergency flags
        self.emergency_stop = False
        self.emergency_reason = ""

        # Performance tracking
        self.state_updates = 0
        self.start_time = time.time()

        # Mode locking (for mission protection)
        self._mode_locked = False
        self._mode_lock_reason = None

        # Current dog selection (C3.1 - persisted across restarts)
        # Priority: ArUco (5s) > session dog_id > select_dog > None
        self._current_dog_id: Optional[str] = None
        self._current_dog_name: Optional[str] = None
        self._last_aruco_dog_id: Optional[str] = None
        self._last_aruco_time: float = 0.0
        self._session_dog_id: Optional[str] = None  # From start_coach/start_mission
        self._STATE_FILE = "/home/morgan/dogbot/data/robot_state.json"
        self._load_persisted_state()

    def lock_mode(self, reason: str):
        """Lock mode - only mission system should call this."""
        with self._lock:
            self._mode_locked = True
            self._mode_lock_reason = reason
            self.logger.info(f"[MODE] Locked: {reason}")

    def unlock_mode(self):
        """Unlock mode - only mission system should call this."""
        with self._lock:
            self._mode_locked = False
            self._mode_lock_reason = None
            self.logger.info("[MODE] Unlocked")

    @property
    def is_mode_locked(self) -> bool:
        """Check if mode is locked."""
        with self._lock:
            return self._mode_locked

    # === Dog Selection (C3.1) ===

    def _load_persisted_state(self):
        """Load persisted state from disk (called on init)."""
        try:
            with open(self._STATE_FILE, 'r') as f:
                data = json.load(f)
                self._current_dog_id = data.get('current_dog_id')
                self._current_dog_name = data.get('current_dog_name')
                if self._current_dog_id:
                    self.logger.info(f"[STATE] Loaded persisted dog: {self._current_dog_name} ({self._current_dog_id})")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning(f"[STATE] Could not load persisted state: {e}")

    def _save_persisted_state(self):
        """Save current_dog_id to disk for restart persistence."""
        try:
            data = {
                'current_dog_id': self._current_dog_id,
                'current_dog_name': self._current_dog_name,
            }
            with open(self._STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"[STATE] Could not save persisted state: {e}")

    def set_current_dog(self, dog_id: str, dog_name: str = None):
        """Set current dog from app's select_dog command. Persists across restarts."""
        with self._lock:
            self._current_dog_id = dog_id
            self._current_dog_name = dog_name or dog_id
            self._save_persisted_state()
            self.logger.info(f"[DOG] Selected: {self._current_dog_name} ({dog_id})")

    def set_session_dog(self, dog_id: str):
        """Set dog_id for active coach/mission session. Clears on session end."""
        with self._lock:
            self._session_dog_id = dog_id
            self.logger.info(f"[DOG] Session dog: {dog_id}")

    def clear_session_dog(self):
        """Clear session dog_id when coach/mission ends."""
        with self._lock:
            self._session_dog_id = None

    def update_aruco_dog(self, dog_id: str):
        """Update last ArUco-identified dog (called on ArUco detection)."""
        with self._lock:
            self._last_aruco_dog_id = dog_id
            self._last_aruco_time = time.time()

    def get_active_dog_id(self, aruco_ttl: float = 5.0) -> Optional[str]:
        """Get the currently active dog_id using fallback chain.

        Priority (highest wins):
        (a) ArUco-identified dog within aruco_ttl seconds
        (b) Session dog_id (from start_coach/start_mission)
        (c) select_dog from app (persisted)
        (d) None

        Args:
            aruco_ttl: Seconds ArUco detection remains valid (default 5s)

        Returns:
            dog_id string or None
        """
        with self._lock:
            # (a) ArUco within TTL
            if self._last_aruco_dog_id and (time.time() - self._last_aruco_time) < aruco_ttl:
                return self._last_aruco_dog_id
            # (b) Session dog
            if self._session_dog_id:
                return self._session_dog_id
            # (c) App-selected dog
            if self._current_dog_id:
                return self._current_dog_id
            # (d) None
            return None

    @property
    def current_dog_id(self) -> Optional[str]:
        """Get app-selected dog_id (select_dog). Use get_active_dog_id() for full fallback."""
        with self._lock:
            return self._current_dog_id

    @property
    def current_dog_name(self) -> Optional[str]:
        """Get app-selected dog name."""
        with self._lock:
            return self._current_dog_name

    def set_mode(self, new_mode: SystemMode, reason: str = "", force: bool = False) -> bool:
        """
        Change system mode with validation and lock protection.

        Args:
            new_mode: New system mode
            reason: Reason for mode change
            force: Override mode lock (for emergencies only)

        Returns:
            bool: True if mode changed successfully
        """
        with self._lock:
            if new_mode == self.mode:
                return True

            # Check mode lock (unless force override for emergencies)
            if self._mode_locked and not force:
                self.logger.warning(f"MODE_CHANGE BLOCKED: {self.mode.value} -> {new_mode.value} | "
                                    f"trigger={reason} | mode locked: {self._mode_lock_reason}")
                # Publish rejection event for app notification
                from core.bus import publish_system_event
                publish_system_event('mode_change_rejected', {
                    'requested': new_mode.value,
                    'current': self.mode.value,
                    'reason': self._mode_lock_reason
                })
                return False

            # SG PROTECTION: Silent Guardian is "sticky" — only explicit user action
            # or safety emergency can take it out of SG. This prevents bark events,
            # battery events, mission completions, timeouts, WebRTC disconnects, etc.
            # from silently reverting SG to IDLE.
            if self.mode == SystemMode.SILENT_GUARDIAN and not force:
                # Always allow transitions TO emergency/shutdown (safety)
                # Always allow transitions TO manual/mission/coach (explicit user actions)
                if new_mode == SystemMode.IDLE:
                    # Only explicit user action can revert SG to IDLE
                    _user_keywords = ['user', 'app', 'api', 'override', 'dropdown', 'user_action']
                    is_user_action = any(kw in reason.lower() for kw in _user_keywords)
                    if not is_user_action:
                        self.logger.warning(
                            f"MODE_CHANGE BLOCKED: silent_guardian -> idle | "
                            f"trigger={reason} | SG is protected — only explicit user action can revert to idle"
                        )
                        return False

            # Validate mode transition
            if not self._is_valid_transition(self.mode, new_mode):
                self.logger.warning(f"MODE_CHANGE INVALID: {self.mode.value} -> {new_mode.value} | trigger={reason}")
                return False

            self.previous_mode = self.mode
            self.mode = new_mode
            self.mode_changed_time = time.time()

            # Comprehensive mode change logging with trigger and source
            import traceback
            stack = traceback.extract_stack()
            # Find the caller outside of state.py
            caller_str = "unknown"
            for frame in reversed(stack[:-1]):
                if 'state.py' not in frame.filename:
                    caller_str = f"{frame.filename}:{frame.lineno}"
                    break
            self.logger.info(
                f"MODE_CHANGE: {self.previous_mode.value} -> {new_mode.value} | "
                f"trigger={reason} | source={caller_str} | "
                f"timestamp={self.mode_changed_time:.3f}"
            )

            # Notify listeners
            self._notify_listeners('mode_change', {
                'previous_mode': self.previous_mode.value,
                'new_mode': new_mode.value,
                'reason': reason,
                'timestamp': self.mode_changed_time,
                'locked': self._mode_locked
            })

            return True

    def _is_valid_transition(self, from_mode: SystemMode, to_mode: SystemMode) -> bool:
        """Validate mode transitions"""
        # Emergency and shutdown can be reached from any state
        if to_mode in [SystemMode.EMERGENCY, SystemMode.SHUTDOWN]:
            return True

        # Can't transition from emergency unless to shutdown
        if from_mode == SystemMode.EMERGENCY and to_mode != SystemMode.SHUTDOWN:
            return False

        # Can't transition from shutdown
        if from_mode == SystemMode.SHUTDOWN:
            return False

        return True

    def get_mode(self) -> SystemMode:
        """Get current system mode"""
        with self._lock:
            return self.mode

    def get_mode_duration(self) -> float:
        """Get time since last mode change"""
        with self._lock:
            return time.time() - self.mode_changed_time

    def update_hardware(self, **kwargs) -> None:
        """Update hardware state"""
        with self._lock:
            changed = False
            for key, value in kwargs.items():
                if hasattr(self.hardware, key):
                    if getattr(self.hardware, key) != value:
                        setattr(self.hardware, key, value)
                        changed = True

            if changed:
                self.hardware.last_updated = time.time()
                self.state_updates += 1
                self._notify_listeners('hardware_change', self.hardware.to_dict())

    def update_mission(self, **kwargs) -> None:
        """Update mission status"""
        with self._lock:
            changed = False
            for key, value in kwargs.items():
                if hasattr(self.mission, key):
                    old_value = getattr(self.mission, key)
                    if old_value != value:
                        setattr(self.mission, key, value)
                        changed = True

            if changed:
                self.state_updates += 1
                self._notify_listeners('mission_change', self.mission.to_dict())

    def update_detection(self, **kwargs) -> None:
        """Update detection status"""
        with self._lock:
            changed = False
            for key, value in kwargs.items():
                if hasattr(self.detection, key):
                    if getattr(self.detection, key) != value:
                        setattr(self.detection, key, value)
                        changed = True

            if changed:
                self.state_updates += 1
                self._notify_listeners('detection_change', self.detection.to_dict())

    def set_emergency(self, reason: str) -> None:
        """Set emergency state"""
        with self._lock:
            self.emergency_stop = True
            self.emergency_reason = reason
            self.set_mode(SystemMode.EMERGENCY, reason)
            self.logger.error(f"EMERGENCY: {reason}")

    def clear_emergency(self) -> None:
        """Clear emergency state - restore previous mode (not always IDLE)"""
        with self._lock:
            self.emergency_stop = False
            self.emergency_reason = ""
            # Restore mode from before emergency, not always IDLE
            restore_mode = self.previous_mode if self.previous_mode not in (
                SystemMode.EMERGENCY, SystemMode.SHUTDOWN
            ) else SystemMode.IDLE
            self.set_mode(restore_mode, f"Emergency cleared, restoring {restore_mode.value}")
            self.logger.info(f"Emergency cleared, restored to {restore_mode.value}")

    def is_emergency(self) -> bool:
        """Check if in emergency state"""
        with self._lock:
            return self.emergency_stop

    def subscribe(self, event_type: str, callback: callable) -> None:
        """Subscribe to state changes"""
        with self._lock:
            if event_type in self._listeners:
                self._listeners[event_type].add(callback)
                self.logger.debug(f"Subscribed to {event_type} events")

    def unsubscribe(self, event_type: str, callback: callable) -> None:
        """Unsubscribe from state changes"""
        with self._lock:
            if event_type in self._listeners:
                self._listeners[event_type].discard(callback)

    def _notify_listeners(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify state change listeners"""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    threading.Thread(target=callback, args=(data,), daemon=True).start()
                except Exception as e:
                    self.logger.error(f"Listener callback failed: {e}")

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete system state"""
        with self._lock:
            return {
                'mode': self.mode.value,
                'previous_mode': self.previous_mode.value,
                'mode_duration': self.get_mode_duration(),
                'emergency': self.emergency_stop,
                'emergency_reason': self.emergency_reason,
                'hardware': self.hardware.to_dict(),
                'mission': self.mission.to_dict(),
                'detection': self.detection.to_dict(),
                'uptime': time.time() - self.start_time,
                'state_updates': self.state_updates,
                'timestamp': time.time()
            }

    def get_status_summary(self) -> Dict[str, Any]:
        """Get condensed status summary"""
        with self._lock:
            return {
                'mode': self.mode.value,
                'emergency': self.emergency_stop,
                'mission_active': self.mission.state != MissionState.INACTIVE,
                'mission_progress': self.mission.progress,
                'dogs_detected': self.detection.dogs_detected,
                'current_behavior': self.detection.current_behavior,
                'battery': self.hardware.battery_voltage,
                'temperature': self.hardware.temperature,
                'uptime': time.time() - self.start_time
            }

    def reset_mission(self) -> None:
        """Reset mission state"""
        with self._lock:
            self.mission = MissionStatus()
            self._notify_listeners('mission_change', self.mission.to_dict())

    def reset_detection(self) -> None:
        """Reset detection state"""
        with self._lock:
            self.detection = DetectionStatus()
            self._notify_listeners('detection_change', self.detection.to_dict())


# Global state manager instance
_state_instance = None
_state_lock = threading.Lock()

def get_state() -> StateManager:
    """Get the global state manager instance (singleton)"""
    global _state_instance
    if _state_instance is None:
        with _state_lock:
            if _state_instance is None:
                _state_instance = StateManager()
    return _state_instance


if __name__ == "__main__":
    # Test the state manager
    state = get_state()

    # Test mode changes
    print("Initial mode:", state.get_mode())

    state.set_mode(SystemMode.SILENT_GUARDIAN, "Starting Silent Guardian mode")
    print("New mode:", state.get_mode())
    print("Mode duration:", state.get_mode_duration())

    # Test hardware updates
    state.update_hardware(battery_voltage=13.8, temperature=45.2)

    # Test mission updates
    state.update_mission(name="test_mission", state=MissionState.ACTIVE, rewards_given=2)

    # Test detection updates
    state.update_detection(dogs_detected=1, current_behavior="sitting", behavior_confidence=0.87)

    # Show full state
    print("\nFull State:")
    import json
    print(json.dumps(state.get_full_state(), indent=2))