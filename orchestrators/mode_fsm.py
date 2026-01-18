#!/usr/bin/env python3
"""
WIM-Z Mode State Machine
Handles automatic mode transitions between IDLE, SILENT_GUARDIAN, COACH, PHOTOGRAPHY, MANUAL

Mode Hierarchy:
- SILENT_GUARDIAN: Primary mode - bark-focused passive monitoring (boot default)
- COACH: Opportunistic trick training when dog approaches
- IDLE: True standby mode
- MANUAL: Xbox controller control
- PHOTOGRAPHY: High-res capture mode
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

        # Track mode before entering MANUAL (to return to it when controller disconnects)
        self.pre_manual_mode = SystemMode.IDLE

        # Transition timeouts (seconds)
        self.timeouts = {
            'silent_guardian_timeout': 0,      # SILENT_GUARDIAN: No auto-timeout (persistent)
            'coach_timeout': 0,                # COACH: No auto-timeout (persistent training mode)
            'manual_timeout': 120.0,           # MANUAL -> previous mode (2 minutes no input)
            'override_timeout': 86400.0        # Max override duration (24 hours - effectively indefinite)
        }

        # Transition rules
        # Note: SILENT_GUARDIAN <-> COACH requires manual switch (API/schedule)
        self.valid_transitions = {
            SystemMode.IDLE: [SystemMode.SILENT_GUARDIAN, SystemMode.COACH, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY, SystemMode.MANUAL],
            SystemMode.SILENT_GUARDIAN: [SystemMode.IDLE, SystemMode.COACH, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY, SystemMode.MANUAL],
            SystemMode.COACH: [SystemMode.IDLE, SystemMode.SILENT_GUARDIAN, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY, SystemMode.MANUAL],
            SystemMode.PHOTOGRAPHY: [SystemMode.IDLE, SystemMode.SILENT_GUARDIAN, SystemMode.COACH, SystemMode.EMERGENCY, SystemMode.MANUAL],
            SystemMode.MANUAL: [SystemMode.IDLE, SystemMode.SILENT_GUARDIAN, SystemMode.COACH, SystemMode.PHOTOGRAPHY, SystemMode.EMERGENCY],
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

        elif current_state_mode == SystemMode.SILENT_GUARDIAN:
            self._evaluate_silent_guardian_transitions(time_in_mode)

        elif current_state_mode == SystemMode.COACH:
            self._evaluate_coach_transitions(time_since_detection, time_in_mode)

        elif current_state_mode == SystemMode.PHOTOGRAPHY:
            # Photography mode should be manually controlled
            pass

        elif current_state_mode == SystemMode.MANUAL:
            self._evaluate_manual_transitions(now)

    def _evaluate_idle_transitions(self, time_since_motion: float, time_since_detection: float) -> None:
        """Evaluate transitions from IDLE mode"""
        # IDLE is a true standby - no automatic transitions
        # User must explicitly switch to SILENT_GUARDIAN or COACH via API/schedule
        pass

    def _evaluate_silent_guardian_transitions(self, time_in_mode: float) -> None:
        """Evaluate transitions from SILENT_GUARDIAN mode"""
        # SILENT_GUARDIAN is the primary mode - stays active until:
        # 1. User switches to COACH or IDLE
        # 2. Very long inactivity timeout (5 min with no barks/interventions)
        # Note: The Silent Guardian handler manages bark-based behavior internally

        # For now, no auto-transitions from Silent Guardian
        # The mode handler (modes/silent_guardian.py) will manage internal state
        pass

    def _evaluate_coach_transitions(self, time_since_detection: float, time_in_mode: float) -> None:
        """Evaluate transitions from COACH mode"""
        # COACH mode is a persistent training mode - no auto-timeout
        # Setting coach_timeout to 0 disables automatic transition
        coach_timeout = self.timeouts.get('coach_timeout', 0)
        if coach_timeout > 0 and time_since_detection > coach_timeout and time_in_mode > 30.0:
            self._transition_to(SystemMode.IDLE, ModeTransition.TIMEOUT)
        # Otherwise stay in Coach mode indefinitely until manually switched

    def _evaluate_manual_transitions(self, now: float) -> None:
        """Evaluate transitions from MANUAL mode"""
        if self.last_manual_input_time == 0:
            return  # No manual input recorded yet

        # Check if Xbox controller is connected - if so, NEVER timeout
        # The Xbox controller will send periodic manual_input_detected events
        # but we'll also check for the controller process
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', 'xbox_hybrid_controller'],
                                  capture_output=True, text=True, timeout=0.5)
            if result.returncode == 0:
                # Xbox controller is running, stay in MANUAL mode
                self.last_manual_input_time = now  # Reset timeout
                return
        except:
            pass  # If check fails, continue with normal timeout logic

        time_since_manual = now - self.last_manual_input_time

        # Manual timeout -> return to previous mode (before entering MANUAL)
        if time_since_manual > self.timeouts['manual_timeout']:
            # Return to the mode we were in before entering MANUAL
            target_mode = self.pre_manual_mode
            self.logger.info(f"Manual timeout, returning to previous mode: {target_mode.value}")
            self._transition_to(target_mode, ModeTransition.MANUAL_TIMEOUT)

    def _transition_to(self, new_mode: SystemMode, trigger: ModeTransition) -> None:
        """Execute mode transition"""
        current_mode = self.state.get_mode()

        # Validate transition
        if new_mode not in self.valid_transitions.get(current_mode, []):
            self.logger.warning(f"Invalid transition: {current_mode.value} -> {new_mode.value}")
            return

        # CRITICAL: Set manual input time BEFORE mode change to avoid race condition
        if new_mode == SystemMode.MANUAL:
            self.last_manual_input_time = time.time()

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
                # Save current mode so we can return to it when controller disconnects
                self.pre_manual_mode = current_mode
                self.logger.info(f"Saving pre-manual mode: {current_mode.value}")
                self._transition_to(SystemMode.MANUAL, ModeTransition.MANUAL_INPUT)

        elif event.subtype == 'controller_connected':
            # Controller connected - if no recent activity, stay in current mode
            self.logger.info("Xbox controller connected")

        elif event.subtype == 'controller_disconnected':
            # Controller disconnected - if in manual mode, return to previous mode
            current_mode = self.state.get_mode()
            if current_mode == SystemMode.MANUAL:
                # Return to the mode we were in before entering MANUAL
                target_mode = self.pre_manual_mode
                self.logger.info(f"Controller disconnected, returning to previous mode: {target_mode.value}")
                self._transition_to(target_mode, ModeTransition.USER_OVERRIDE)

    def set_mode_override(self, mode: SystemMode, duration: float = None) -> bool:
        """Override automatic mode transitions"""
        if mode not in [SystemMode.PHOTOGRAPHY, SystemMode.MANUAL, SystemMode.SILENT_GUARDIAN, SystemMode.COACH, SystemMode.IDLE]:
            self.logger.error(f"Cannot override to mode: {mode.value}")
            return False

        duration = duration or self.timeouts['override_timeout']
        self.override_mode = mode
        self.override_until = time.time() + duration

        # CRITICAL: Set manual input time BEFORE mode change to avoid race condition
        if mode == SystemMode.MANUAL:
            self.last_manual_input_time = time.time()

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
        # CRITICAL: Set manual input time BEFORE mode change to avoid race condition
        # The FSM thread could check timeout between set_mode() and time update
        if mode == SystemMode.MANUAL:
            self.last_manual_input_time = time.time()

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