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
    NIGHT_SENTRY = "night_sentry"         # Demo watch mode — body sweep + dog detection alerts
    MANUAL = "manual"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"
    # ROBOT-STATE v2: sleep is NOT a mode. A sleeping robot is in IDLE with
    # services/power/sleep_manager reporting asleep = true.

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


# ROBOT-STATE v2 (2026-11-19). state.mode on the wire is exactly one of these
# five. Every other internal mode (night_sentry, photography, emergency,
# shutdown) is reported as "idle" (R6).
CONTRACT_MODES = ("idle", "silent_guardian", "coach", "mission", "manual")
SESSION_MODES = ("silent_guardian", "coach", "mission")
CONTRACT_SOURCES = ("app", "connector", "agent", "schedule", "robot", "admin", "relay")


def contract_mode(mode) -> str:
    """Internal mode (SystemMode or its value) -> the contract's mode."""
    value = mode.value if isinstance(mode, SystemMode) else str(mode or "")
    return value if value in CONTRACT_MODES else "idle"


def contract_source(source, default: str = "robot") -> str:
    s = str(source or "").lower()
    return s if s in CONTRACT_SOURCES else default


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

        # Dog selection (ROBOT-STATE v2).
        #   _current_dog_id  = the contract's selected_dog_id: the dog chosen
        #                      in the app. Changes ONLY through select_dog.
        #                      Persisted across restarts.
        #   _session_dog_id  = the contract's session_dog_id: fixed when
        #                      silent_guardian / coach / mission is entered,
        #                      null in idle / manual (set in set_mode()).
        #   _coach_dog_id    = the dog the coach engine is working with right
        #                      now (from vision; used for per-dog voice clips).
        #   _marker_reads    = last time each dog's ArUco/QR marker was read.
        self._current_dog_id: Optional[str] = None
        self._current_dog_name: Optional[str] = None
        # True when the last select_dog came over the local / AP connection
        # (the relay has not seen it); False after any select_dog from the
        # relay. Persisted with the selection. Starts False.
        self._selected_dog_changed_locally: bool = False
        self._last_aruco_dog_id: Optional[str] = None
        self._last_aruco_time: float = 0.0
        self._marker_reads: Dict[str, float] = {}
        self._session_dog_id: Optional[str] = None
        self._coach_dog_id: Optional[str] = None
        # Who/what caused the mode change in progress on this thread
        # (see mode_context()).
        self._mode_ctx = threading.local()
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
                self._selected_dog_changed_locally = bool(data.get('selected_dog_changed_locally', False))
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
                'selected_dog_changed_locally': bool(self._selected_dog_changed_locally),
            }
            with open(self._STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"[STATE] Could not save persisted state: {e}")

    def set_current_dog(self, dog_id: Optional[str], dog_name: str = None,
                        changed_locally: Optional[bool] = None):
        """select_dog (ROBOT-STATE v2 C2): the contract's selected_dog_id.
        None clears the selection. Persists across restarts. This is the ONLY
        writer of the selected dog.

        changed_locally: True = arrived over the local / AP connection,
        False = arrived from the relay (whatever the value), None = leave the
        flag as it is."""
        with self._lock:
            self._current_dog_id = dog_id or None
            self._current_dog_name = (dog_name or dog_id) if dog_id else None
            if changed_locally is not None:
                self._selected_dog_changed_locally = bool(changed_locally)
            self._save_persisted_state()
            self.logger.info(f"[DOG] Selected: {self._current_dog_name} ({self._current_dog_id}) "
                             f"changed_locally={self._selected_dog_changed_locally}")

    def selected_dog_changed_locally(self) -> bool:
        with self._lock:
            return bool(self._selected_dog_changed_locally)

    def set_coach_dog(self, dog_id: str):
        """Dog the coach engine is working with right now (voice fallback chain)."""
        with self._lock:
            self._coach_dog_id = dog_id
            self.logger.info(f"[DOG] Coach dog: {dog_id}")

    def clear_coach_dog(self):
        with self._lock:
            self._coach_dog_id = None

    def get_coach_dog_id(self) -> Optional[str]:
        with self._lock:
            return self._coach_dog_id

    # === Mode change context (ROBOT-STATE v2 E1) ===

    def mode_context(self, source: Optional[str] = None, reason: Optional[str] = None,
                     command_id: Optional[str] = None, dog_id: Optional[str] = None):
        """Context manager: every set_mode() on this thread inside the block
        is stamped with who caused it (mode_change data.source / reason /
        command_id) and, for a session mode, the command's dog_id (R3).
        Nested blocks inherit what they do not set."""
        return _ModeContext(self, source, reason, command_id, dog_id)

    def _current_mode_ctx(self) -> Dict[str, Any]:
        return getattr(self._mode_ctx, 'value', None) or {}

    @staticmethod
    def _is_real_dog_id(dog_id: str) -> bool:
        """Check if dog_id is a real profile ID vs generic tracker ID.

        Rejects generic auto-tracker ids like dog_0, dog_-1000, dog_5.
        Real profile ids come from the relay/app and look like
        dog_1777167142852 (timestamp-based, not small integers).
        """
        if not dog_id:
            return False
        if dog_id.startswith("dog_"):
            try:
                suffix = dog_id.split("_", 1)[1]
                val = int(suffix)
                # Generic tracker IDs are small integers or negative
                # Real profile IDs are timestamps (13+ digits)
                if val < 1000000000000:  # Before year 2001 in ms
                    return False
            except (ValueError, IndexError):
                pass
        return True

    def update_aruco_dog(self, dog_id: str):
        """A marker (ArUco / QR) mapped to a dog was read just now.
        Only accepts real profile IDs, not generic tracker IDs like dog_0."""
        if not self._is_real_dog_id(dog_id):
            return  # Ignore generic tracker IDs
        with self._lock:
            now = time.time()
            self._last_aruco_dog_id = dog_id
            self._last_aruco_time = now
            self._marker_reads[dog_id] = now

    def marker_dogs_within(self, seconds: float = 5.0) -> list:
        """Dogs whose marker was read within `seconds`, most recent first
        (the keys update_aruco_dog() was given: profile names / ids)."""
        with self._lock:
            now = time.time()
            rows = [(t, d) for d, t in self._marker_reads.items() if now - t < seconds]
        return [d for _, d in sorted(rows, reverse=True)]

    def get_active_dog_id(self, aruco_ttl: float = 5.0) -> Optional[str]:
        """Dog to use for local decisions (voice clips, controller), by
        fallback chain. NOT the relay attribution rule (that is
        core.dog_profile_manager.resolve_relay_dog).

        Priority (highest wins):
        (a) ArUco-identified dog within aruco_ttl seconds
        (b) the coach engine's current dog
        (c) session_dog_id (fixed at mode entry)
        (d) selected dog from the app (persisted)
        (e) None
        """
        with self._lock:
            if self._last_aruco_dog_id and (time.time() - self._last_aruco_time) < aruco_ttl:
                return self._last_aruco_dog_id
            for d in (self._coach_dog_id, self._session_dog_id, self._current_dog_id):
                if d and self._is_real_dog_id(d):
                    return d
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

    def get_aruco_dog_within(self, seconds: float = 5.0) -> Optional[str]:
        """Get ArUco-identified dog if detected within seconds."""
        with self._lock:
            if self._last_aruco_dog_id and (time.time() - self._last_aruco_time) < seconds:
                return self._last_aruco_dog_id
            return None

    def get_session_dog_id(self) -> Optional[str]:
        """Contract session_dog_id: fixed on entering silent_guardian / coach /
        mission, None in idle / manual."""
        with self._lock:
            return self._session_dog_id

    def get_current_dog_id(self) -> Optional[str]:
        """Contract selected_dog_id (select_dog)."""
        with self._lock:
            return self._current_dog_id

    def get_contract_mode(self) -> str:
        """state.mode as the contract reports it: one of the five."""
        with self._lock:
            return contract_mode(self.mode)

    def set_mode(self, new_mode: SystemMode, reason: str = "", force: bool = False,
                 dog_id: Optional[str] = None) -> bool:
        """
        Change system mode with validation and lock protection.

        Args:
            new_mode: New system mode
            reason: Reason for mode change (internal trigger text)
            force: Override mode lock (for emergencies only)
            dog_id: dog named by whatever started a session mode (R3); falls
                back to the enclosing mode_context(), then to the selected dog

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

            ctx = self._current_mode_ctx()
            self.previous_mode = self.mode
            self.mode = new_mode
            self.mode_changed_time = time.time()

            # ROBOT-STATE v2 R3/R4: session_dog_id is fixed here, on entry.
            new_contract = contract_mode(new_mode)
            prev_contract = contract_mode(self.previous_mode)
            if new_contract in SESSION_MODES:
                named = dog_id or ctx.get('dog_id')
                if not named or str(named).lower() in ('unknown', 'none', 'null'):
                    named = None
                self._session_dog_id = named or self._current_dog_id or None
            else:
                self._session_dog_id = None

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
                'locked': self._mode_locked,
                # ROBOT-STATE v2 E1 fields
                'mode': new_contract,
                'contract_previous_mode': prev_contract,
                'session_dog_id': self._session_dog_id,
                'source': contract_source(ctx.get('source')),
                'contract_reason': ctx.get('reason') if ctx else (reason or None),
                'command_id': ctx.get('command_id'),
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


class _ModeContext:
    """See StateManager.mode_context()."""

    def __init__(self, state, source, reason, command_id, dog_id):
        self._state = state
        self._new = {'source': source, 'reason': reason, 'command_id': command_id, 'dog_id': dog_id}
        self._prev = None

    def __enter__(self):
        local = self._state._mode_ctx
        self._prev = getattr(local, 'value', None)
        merged = dict(self._prev or {})
        merged.update({k: v for k, v in self._new.items() if v is not None})
        local.value = merged
        return self

    def __exit__(self, *exc):
        self._state._mode_ctx.value = self._prev
        return False


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