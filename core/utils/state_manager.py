#!/usr/bin/env python3
"""
State Manager for tracking robot and environment state
Provides centralized state management with event notifications
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from .event_bus import EventBus

class SystemState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class DetectionState(Enum):
    NO_DOG = "no_dog"
    DOG_DETECTED = "dog_detected"
    TRACKING = "tracking"
    LOST_TRACK = "lost_track"

@dataclass
class RobotState:
    """Complete robot state information"""
    system_state: SystemState = SystemState.INITIALIZING
    detection_state: DetectionState = DetectionState.NO_DOG

    # Hardware states
    motors_active: bool = False
    camera_tracking: bool = False
    audio_playing: bool = False

    # Detection info
    current_dog_count: int = 0
    last_detection_time: Optional[float] = None
    current_behavior: Optional[str] = None
    behavior_confidence: float = 0.0

    # Statistics
    total_detections: int = 0
    total_treats_dispensed: int = 0
    session_start_time: float = 0.0

    # Safety
    emergency_stop_active: bool = False
    last_health_check: float = 0.0

class StateManager:
    """Centralized state management with event notifications"""

    def __init__(self, event_bus: EventBus):
        self.logger = logging.getLogger('StateManager')
        self.event_bus = event_bus
        self.lock = threading.RLock()

        # Initialize state
        self.state = RobotState()
        self.state.session_start_time = time.time()
        self.state.last_health_check = time.time()

        # State change history (for debugging)
        self.state_history = []
        self.max_history = 100

    def get_state(self) -> RobotState:
        """Get current robot state (thread-safe copy)"""
        with self.lock:
            return RobotState(**asdict(self.state))

    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary"""
        with self.lock:
            return asdict(self.state)

    def update_system_state(self, new_state: SystemState):
        """Update system state with event notification"""
        with self.lock:
            if self.state.system_state != new_state:
                old_state = self.state.system_state
                self.state.system_state = new_state
                self._record_state_change('system_state', old_state.value, new_state.value)

                self.event_bus.publish('system_state_changed', {
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'timestamp': time.time()
                })

    def update_detection_state(self, new_state: DetectionState, **kwargs):
        """Update detection state with additional data"""
        with self.lock:
            if self.state.detection_state != new_state:
                old_state = self.state.detection_state
                self.state.detection_state = new_state
                self.state.last_detection_time = time.time()

                # Update related fields
                if 'dog_count' in kwargs:
                    self.state.current_dog_count = kwargs['dog_count']
                if 'behavior' in kwargs:
                    self.state.current_behavior = kwargs['behavior']
                if 'confidence' in kwargs:
                    self.state.behavior_confidence = kwargs['confidence']

                self._record_state_change('detection_state', old_state.value, new_state.value)

                # Publish specific events
                if new_state == DetectionState.DOG_DETECTED:
                    self.state.total_detections += 1
                    self.event_bus.publish('dog_detected', {
                        'dog_count': self.state.current_dog_count,
                        'timestamp': self.state.last_detection_time,
                        **kwargs
                    })
                elif new_state == DetectionState.NO_DOG:
                    self.event_bus.publish('dog_lost', {
                        'last_seen': self.state.last_detection_time,
                        'timestamp': time.time()
                    })

    def update_hardware_state(self, **kwargs):
        """Update hardware-related state"""
        with self.lock:
            if 'motors_active' in kwargs:
                self.state.motors_active = kwargs['motors_active']
            if 'camera_tracking' in kwargs:
                self.state.camera_tracking = kwargs['camera_tracking']
            if 'audio_playing' in kwargs:
                self.state.audio_playing = kwargs['audio_playing']

    def record_treat_dispensed(self, behavior: str):
        """Record that a treat was dispensed"""
        with self.lock:
            self.state.total_treats_dispensed += 1
            self.event_bus.publish('treat_dispensed', {
                'behavior': behavior,
                'total_count': self.state.total_treats_dispensed,
                'timestamp': time.time()
            })

    def set_emergency_stop(self, active: bool):
        """Set emergency stop state"""
        with self.lock:
            if self.state.emergency_stop_active != active:
                self.state.emergency_stop_active = active
                self._record_state_change('emergency_stop', not active, active)

                if active:
                    self.event_bus.publish('emergency_stop_activated', {
                        'timestamp': time.time()
                    })
                else:
                    self.event_bus.publish('emergency_stop_cleared', {
                        'timestamp': time.time()
                    })

    def update_health_check(self):
        """Update last health check timestamp"""
        with self.lock:
            self.state.last_health_check = time.time()

    def get_session_duration(self) -> float:
        """Get current session duration in seconds"""
        with self.lock:
            return time.time() - self.state.session_start_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self.lock:
            duration = self.get_session_duration()
            return {
                'session_duration_seconds': duration,
                'session_duration_formatted': self._format_duration(duration),
                'total_detections': self.state.total_detections,
                'total_treats_dispensed': self.state.total_treats_dispensed,
                'treats_per_hour': (self.state.total_treats_dispensed / duration * 3600) if duration > 0 else 0,
                'detections_per_hour': (self.state.total_detections / duration * 3600) if duration > 0 else 0,
                'current_dog_count': self.state.current_dog_count,
                'current_behavior': self.state.current_behavior,
                'behavior_confidence': self.state.behavior_confidence
            }

    def _record_state_change(self, field: str, old_value: Any, new_value: Any):
        """Record state change in history"""
        change_record = {
            'timestamp': time.time(),
            'field': field,
            'old_value': old_value,
            'new_value': new_value
        }

        self.state_history.append(change_record)

        # Trim history if too long
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

        self.logger.debug(f"State change: {field} {old_value} -> {new_value}")

    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent state change history"""
        with self.lock:
            return self.state_history[-limit:]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"