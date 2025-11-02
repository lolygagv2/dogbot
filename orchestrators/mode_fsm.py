#!/usr/bin/env python3
"""
Camera mode state machine
Handles automatic mode transitions between IDLE, DETECTION, VIGILANT, PHOTOGRAPHY
"""

import threading
import time
import logging
from typing import Dict, Any, Optional
from enum import Enum

from core.bus import get_bus, publish_system_event
from core.state import get_state, SystemMode
from core.store import get_store


class ModeTransition(Enum):
    """Mode transition triggers"""
    MOTION_DETECTED = "motion_detected"
    DOG_DETECTED = "dog_detected"
    DOG_LOST = "dog_lost"
    NO_MOTION = "no_motion"
    USER_OVERRIDE = "user_override"
    TIMEOUT = "timeout"
    EMERGENCY = "emergency"
    MANUAL_INPUT = "manual_input"
    MANUAL_TIMEOUT = "manual_timeout"


class ModeFSM:
    """
    Camera mode finite state machine
    Automatically transitions between modes based on events and timeouts
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.logger = logging.getLogger('ModeFSM')

        # FSM state
        self.current_mode = SystemMode.IDLE
        self.mode_start_time = time.time()
        self.last_motion_time = 0.0
        self.last_detection_time = 0.0
        self.last_manual_input_time = 0.0
        self.override_mode = None
        self.override_until = 0.0

        # Transition timeouts (seconds)
        self.timeouts = {
            'no_motion_timeout': 10.0,      # DETECTION -> VIGILANT
            'vigilant_timeout': 30.0,       # VIGILANT -> IDLE
            'detection_timeout': 5.0,       # No dogs -> VIGILANT
            'manual_timeout': 120.0,        # MANUAL -> AUTO (2 minutes)
            'override_timeout': 300.0       # Max override duration (5 min)
        }

        # Transition rules
        self.valid_transitions = {
            SystemMode.IDLE: [SystemMode.DETECTION, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY],
            SystemMode.DETECTION: [SystemMode.VIGILANT, SystemMode.IDLE, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY],
            SystemMode.VIGILANT: [SystemMode.DETECTION, SystemMode.IDLE, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY],
            SystemMode.PHOTOGRAPHY: [SystemMode.IDLE, SystemMode.DETECTION, SystemMode.EMERGENCY],
            SystemMode.MANUAL: [SystemMode.IDLE, SystemMode.DETECTION, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY],
            SystemMode.EMERGENCY: [SystemMode.SHUTDOWN]
        }

        # FSM control
        self.fsm_running = False
        self.fsm_thread = None
        self._stop_event = threading.Event()

        # Subscribe to relevant events
        self.bus.subscribe('vision', self._on_vision_event)
        self.bus.subscribe('motion', self._on_motion_event)
        self.bus.subscribe('system', self._on_system_event)

    def start_fsm(self) -> bool:
        """Start the mode FSM"""
        if self.fsm_running:
            self.logger.warning("Mode FSM already running")
            return True

        self.fsm_running = True
        self._stop_event.clear()

        self.fsm_thread = threading.Thread(
            target=self._fsm_loop,
            daemon=True,
            name="ModeFSM"
        )
        self.fsm_thread.start()

        self.logger.info("Mode FSM started")
        publish_system_event('mode_fsm_started', {}, 'mode_fsm')
        return True

    def stop_fsm(self) -> None:
        """Stop the mode FSM"""
        if not self.fsm_running:
            return

        self.fsm_running = False
        self._stop_event.set()

        if self.fsm_thread and self.fsm_thread.is_alive():
            self.fsm_thread.join(timeout=2.0)

        self.logger.info("Mode FSM stopped")
        publish_system_event('mode_fsm_stopped', {}, 'mode_fsm')

    def _fsm_loop(self) -> None:
        """Main FSM loop"""
        while not self._stop_event.wait(1.0):  # Check every second
            try:
                self._evaluate_transitions()
            except Exception as e:
                self.logger.error(f"FSM loop error: {e}")

    def _evaluate_transitions(self) -> None:
        """Evaluate possible mode transitions"""
        now = time.time()
        current_state_mode = self.state.get_mode()

        # Check for override expiration
        if self.override_mode and now > self.override_until:
            self.clear_override("timeout")

        # Don't auto-transition if overridden
        if self.override_mode:
            return

        # Don't transition from emergency or shutdown
        if current_state_mode in [SystemMode.EMERGENCY, SystemMode.SHUTDOWN]:
            return

        # Check for emergency conditions
        if self.state.is_emergency():
            self._transition_to(SystemMode.EMERGENCY, ModeTransition.EMERGENCY)
            return

        time_in_mode = now - self.mode_start_time
        time_since_motion = now - self.last_motion_time if self.last_motion_time > 0 else 999
        time_since_detection = now - self.last_detection_time if self.last_detection_time > 0 else 999

        # Mode-specific transition logic
        if current_state_mode == SystemMode.IDLE:
            self._evaluate_idle_transitions(time_since_motion, time_since_detection)

        elif current_state_mode == SystemMode.DETECTION:
            self._evaluate_detection_transitions(time_since_motion, time_since_detection, time_in_mode)

        elif current_state_mode == SystemMode.VIGILANT:
            self._evaluate_vigilant_transitions(time_since_motion, time_since_detection, time_in_mode)

        elif current_state_mode == SystemMode.PHOTOGRAPHY:
            # Photography mode should be manually controlled
            pass

        elif current_state_mode == SystemMode.MANUAL:
            self._evaluate_manual_transitions(now)

    def _evaluate_idle_transitions(self, time_since_motion: float, time_since_detection: float) -> None:
        """Evaluate transitions from IDLE mode"""
        # Motion detected -> DETECTION
        if time_since_motion < 2.0:
            self._transition_to(SystemMode.DETECTION, ModeTransition.MOTION_DETECTED)

        # Recent dog detection -> DETECTION
        elif time_since_detection < 5.0:
            self._transition_to(SystemMode.DETECTION, ModeTransition.DOG_DETECTED)

    def _evaluate_detection_transitions(self, time_since_motion: float,
                                      time_since_detection: float, time_in_mode: float) -> None:
        """Evaluate transitions from DETECTION mode"""
        # No motion for a while -> VIGILANT
        if time_since_motion > self.timeouts['no_motion_timeout']:
            self._transition_to(SystemMode.VIGILANT, ModeTransition.NO_MOTION)

        # No detection for a while -> VIGILANT
        elif time_since_detection > self.timeouts['detection_timeout'] and time_in_mode > 10.0:
            self._transition_to(SystemMode.VIGILANT, ModeTransition.DOG_LOST)

    def _evaluate_vigilant_transitions(self, time_since_motion: float,
                                     time_since_detection: float, time_in_mode: float) -> None:
        """Evaluate transitions from VIGILANT mode"""
        # Motion detected -> DETECTION
        if time_since_motion < 2.0:
            self._transition_to(SystemMode.DETECTION, ModeTransition.MOTION_DETECTED)

        # Dog detected -> DETECTION
        elif time_since_detection < 3.0:
            self._transition_to(SystemMode.DETECTION, ModeTransition.DOG_DETECTED)

        # Long time in vigilant -> IDLE
        elif time_in_mode > self.timeouts['vigilant_timeout']:
            self._transition_to(SystemMode.IDLE, ModeTransition.TIMEOUT)

    def _evaluate_manual_transitions(self, now: float) -> None:
        """Evaluate transitions from MANUAL mode"""
        if self.last_manual_input_time == 0:
            return  # No manual input recorded yet

        time_since_manual = now - self.last_manual_input_time

        # Manual timeout -> return to appropriate autonomous mode
        if time_since_manual > self.timeouts['manual_timeout']:
            # Choose autonomous mode based on recent activity
            time_since_motion = now - self.last_motion_time if self.last_motion_time > 0 else 999
            time_since_detection = now - self.last_detection_time if self.last_detection_time > 0 else 999

            if time_since_detection < 10.0:
                target_mode = SystemMode.DETECTION
            elif time_since_motion < 30.0:
                target_mode = SystemMode.VIGILANT
            else:
                target_mode = SystemMode.IDLE

            self._transition_to(target_mode, ModeTransition.MANUAL_TIMEOUT)

    def _transition_to(self, new_mode: SystemMode, trigger: ModeTransition) -> None:
        """Execute mode transition"""
        current_mode = self.state.get_mode()

        # Validate transition
        if new_mode not in self.valid_transitions.get(current_mode, []):
            self.logger.warning(f"Invalid transition: {current_mode.value} -> {new_mode.value}")
            return

        # Execute transition
        success = self.state.set_mode(new_mode, f"FSM: {trigger.value}")

        if success:
            self.current_mode = new_mode
            self.mode_start_time = time.time()

            # Log transition
            self.store.log_event('system', 'mode_transition', 'mode_fsm', {
                'from_mode': current_mode.value,
                'to_mode': new_mode.value,
                'trigger': trigger.value,
                'timestamp': time.time()
            })

            publish_system_event('mode_transition', {
                'from_mode': current_mode.value,
                'to_mode': new_mode.value,
                'trigger': trigger.value
            }, 'mode_fsm')

            self.logger.info(f"Mode transition: {current_mode.value} -> {new_mode.value} ({trigger.value})")

    def _on_vision_event(self, event) -> None:
        """Handle vision events"""
        if event.subtype == 'dog_detected':
            self.last_detection_time = time.time()

        elif event.subtype == 'detection_started':
            self.last_motion_time = time.time()

    def _on_motion_event(self, event) -> None:
        """Handle motion events"""
        if event.subtype in ['motor_started', 'servo_moved']:
            self.last_motion_time = time.time()

    def _on_system_event(self, event) -> None:
        """Handle system events"""
        if event.subtype == 'emergency':
            self._transition_to(SystemMode.EMERGENCY, ModeTransition.EMERGENCY)

        elif event.subtype == 'manual_input_detected':
            # Xbox controller input detected
            self.last_manual_input_time = time.time()
            current_mode = self.state.get_mode()

            # Switch to manual mode if not already
            if current_mode != SystemMode.MANUAL:
                self._transition_to(SystemMode.MANUAL, ModeTransition.MANUAL_INPUT)

        elif event.subtype == 'controller_connected':
            # Controller connected - if no recent activity, stay in current mode
            self.logger.info("Xbox controller connected")

        elif event.subtype == 'controller_disconnected':
            # Controller disconnected - if in manual mode, switch to appropriate autonomous mode
            current_mode = self.state.get_mode()
            if current_mode == SystemMode.MANUAL:
                self.logger.info("Controller disconnected while in manual mode, switching to autonomous")

                # Choose appropriate autonomous mode based on recent activity
                now = time.time()
                time_since_motion = now - self.last_motion_time if self.last_motion_time > 0 else 999
                time_since_detection = now - self.last_detection_time if self.last_detection_time > 0 else 999

                if time_since_detection < 10.0:
                    target_mode = SystemMode.DETECTION
                elif time_since_motion < 30.0:
                    target_mode = SystemMode.VIGILANT
                else:
                    target_mode = SystemMode.IDLE

                self._transition_to(target_mode, ModeTransition.USER_OVERRIDE)

    def set_mode_override(self, mode: SystemMode, duration: float = None) -> bool:
        """Override automatic mode transitions"""
        if mode not in [SystemMode.PHOTOGRAPHY, SystemMode.MANUAL, SystemMode.DETECTION, SystemMode.IDLE]:
            self.logger.error(f"Cannot override to mode: {mode.value}")
            return False

        duration = duration or self.timeouts['override_timeout']
        self.override_mode = mode
        self.override_until = time.time() + duration

        # Force transition to override mode
        success = self.state.set_mode(mode, f"User override for {duration}s")

        if success:
            self.current_mode = mode
            self.mode_start_time = time.time()

            publish_system_event('mode_override_set', {
                'mode': mode.value,
                'duration': duration
            }, 'mode_fsm')

            self.logger.info(f"Mode override set: {mode.value} for {duration}s")

        return success

    def clear_override(self, reason: str = "manual") -> None:
        """Clear mode override"""
        if not self.override_mode:
            return

        old_override = self.override_mode
        self.override_mode = None
        self.override_until = 0.0

        publish_system_event('mode_override_cleared', {
            'previous_override': old_override.value,
            'reason': reason
        }, 'mode_fsm')

        self.logger.info(f"Mode override cleared: {old_override.value} ({reason})")

    def force_mode(self, mode: SystemMode, reason: str = "manual") -> bool:
        """Force immediate mode change (bypass FSM)"""
        success = self.state.set_mode(mode, f"Force: {reason}")

        if success:
            self.current_mode = mode
            self.mode_start_time = time.time()

            publish_system_event('mode_forced', {
                'mode': mode.value,
                'reason': reason
            }, 'mode_fsm')

            self.logger.info(f"Mode forced: {mode.value} ({reason})")

        return success

    def get_status(self) -> Dict[str, Any]:
        """Get FSM status"""
        now = time.time()

        return {
            'fsm_running': self.fsm_running,
            'current_mode': self.current_mode.value,
            'time_in_mode': now - self.mode_start_time,
            'override_active': self.override_mode is not None,
            'override_mode': self.override_mode.value if self.override_mode else None,
            'override_remaining': max(0, self.override_until - now) if self.override_mode else 0,
            'last_motion': self.last_motion_time,
            'last_detection': self.last_detection_time,
            'last_manual_input': self.last_manual_input_time,
            'time_since_motion': now - self.last_motion_time if self.last_motion_time > 0 else 999,
            'time_since_detection': now - self.last_detection_time if self.last_detection_time > 0 else 999,
            'time_since_manual': now - self.last_manual_input_time if self.last_manual_input_time > 0 else 999,
            'timeouts': self.timeouts.copy()
        }

    def set_timeout(self, timeout_name: str, value: float) -> bool:
        """Set FSM timeout"""
        if timeout_name in self.timeouts:
            self.timeouts[timeout_name] = max(1.0, value)
            self.logger.info(f"Timeout updated: {timeout_name} = {value}s")
            return True
        return False

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_fsm()
        self.logger.info("Mode FSM cleaned up")


# Global mode FSM instance
_mode_fsm_instance = None
_mode_fsm_lock = threading.Lock()

def get_mode_fsm() -> ModeFSM:
    """Get the global mode FSM instance (singleton)"""
    global _mode_fsm_instance
    if _mode_fsm_instance is None:
        with _mode_fsm_lock:
            if _mode_fsm_instance is None:
                _mode_fsm_instance = ModeFSM()
    return _mode_fsm_instance