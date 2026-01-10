#!/usr/bin/env python3
"""
Behavior Interpreter - Layer 1
==============================
Simple wrapper that checks if behavior detections meet trick requirements.

This is the SINGLE location for:
- Trick rule definitions (from YAML config)
- Confidence thresholds per behavior
- Hold duration tracking

Other modules (coaching_engine, mission_engine) call this - they don't
implement their own detection logic.
"""

import time
import logging
import threading
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrickCheckResult:
    """Result of checking if a trick was completed"""
    completed: bool
    behavior_detected: Optional[str] = None
    hold_duration: float = 0.0
    confidence: float = 0.0
    reason: str = ""


class BehaviorInterpreter:
    """
    Simple behavior interpreter - checks if detections meet trick requirements.

    Tracks recent detections to calculate hold duration.
    All config comes from trick_rules.yaml (Layer 2).
    """

    BEHAVIORS = ['stand', 'sit', 'lie', 'cross', 'spin']

    def __init__(self, config_path: str = None):
        """Initialize behavior interpreter"""
        self._lock = threading.Lock()

        # Simple state: last detection per source
        self._last_behavior: Optional[str] = None
        self._last_confidence: float = 0.0
        self._behavior_start_time: float = 0.0
        self._last_update_time: float = 0.0
        self._reset_timestamp: float = 0.0  # Track when reset was called (for stale event filtering)

        # Default confidence thresholds (MUST be defined before _load_trick_rules)
        # Note: lie/cross raised to 0.75 to prevent sitting from triggering false positives
        self.confidence_thresholds = {
            'stand': 0.70,
            'sit': 0.65,
            'lie': 0.75,    # Raised from 0.65 - sitting was triggering false positives
            'cross': 0.75,  # Raised from 0.60 - sitting was triggering false positives
            'spin': 0.70,
        }

        # Load config (may override confidence_thresholds above)
        self.config_path = config_path or '/home/morgan/dogbot/configs/trick_rules.yaml'
        self.trick_rules = self._load_trick_rules()

        # Subscribe to behavior events
        self._setup_event_subscription()

        logger.info(f"BehaviorInterpreter initialized with {len(self.trick_rules)} tricks")

    def _load_trick_rules(self) -> Dict[str, Any]:
        """Load trick rules from YAML config"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded trick rules from {self.config_path}")

                    # Store config sections for other modules
                    self.coaching_config = config.get('coaching', {})
                    self.audio_config = config.get('audio', {})
                    self.detection_config = config.get('detection', {})
                    self.rewards_config = config.get('rewards', {})

                    # Apply confidence overrides from config
                    if self.detection_config:
                        overrides = self.detection_config.get('confidence_overrides', {})
                        for behavior, threshold in overrides.items():
                            self.confidence_thresholds[behavior] = threshold

                    return config.get('tricks', {})
        except Exception as e:
            logger.warning(f"Failed to load trick rules: {e}, using defaults")

        # Defaults
        self.coaching_config = {}
        self.audio_config = {}
        self.detection_config = {}
        self.rewards_config = {}

        return {
            'sit': {'required_behavior': 'sit', 'hold_duration_sec': 1.0, 'detection_window_sec': 10, 'alternative_behaviors': [], 'confidence_threshold': 0.65, 'audio_command': 'sit.mp3'},
            'down': {'required_behavior': 'lie', 'hold_duration_sec': 1.5, 'detection_window_sec': 10, 'alternative_behaviors': ['cross'], 'confidence_threshold': 0.65, 'audio_command': 'lie_down.mp3'},
            'crosses': {'required_behavior': 'cross', 'hold_duration_sec': 1.5, 'detection_window_sec': 10, 'alternative_behaviors': [], 'confidence_threshold': 0.60, 'audio_command': 'crosses.mp3'},
            'spin': {'required_behavior': 'spin', 'hold_duration_sec': 0.3, 'detection_window_sec': 15, 'alternative_behaviors': [], 'confidence_threshold': 0.70, 'audio_command': 'spin.mp3'},
            'stand': {'required_behavior': 'stand', 'hold_duration_sec': 2.0, 'detection_window_sec': 10, 'alternative_behaviors': [], 'confidence_threshold': 0.70, 'audio_command': 'stand.mp3'},
            'speak': {'required_behavior': 'bark', 'hold_duration_sec': 0, 'detection_window_sec': 5, 'alternative_behaviors': [], 'confidence_threshold': 0.60, 'audio_command': 'speak.mp3', 'min_barks': 1, 'max_barks': 2},
        }

    def _setup_event_subscription(self):
        """Subscribe to behavior events from AI controller"""
        try:
            from core.bus import get_bus
            bus = get_bus()
            bus.subscribe('vision', self._on_vision_event)
        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    def _on_vision_event(self, event):
        """Handle vision events - track behavior detections"""
        if event.subtype == 'behavior_detected':
            behavior = event.data.get('behavior')
            confidence = event.data.get('confidence', 0.0)

            if behavior:
                # Pass event timestamp to filter out stale threaded callbacks
                self._update_detection(behavior, confidence, event.timestamp)

    def _update_detection(self, behavior: str, confidence: float, event_timestamp: float = None):
        """Update current detection state

        Args:
            behavior: Detected behavior name
            confidence: Detection confidence
            event_timestamp: When the event was published (for stale event filtering)
        """
        with self._lock:
            now = time.time()

            # CRITICAL: Reject stale events from before the last reset
            # This prevents race conditions where threaded callbacks from old events
            # execute after reset_tracking() was called
            if event_timestamp is not None and event_timestamp < self._reset_timestamp:
                return  # Ignore stale event - it was published before reset

            threshold = self.confidence_thresholds.get(behavior, 0.7)

            if confidence >= threshold:
                if behavior != self._last_behavior:
                    # New behavior - reset timer
                    self._last_behavior = behavior
                    self._behavior_start_time = now
                    self._last_confidence = confidence
                else:
                    # Same behavior - update confidence (keep max)
                    self._last_confidence = max(self._last_confidence, confidence)

                self._last_update_time = now

    def reset_tracking(self):
        """
        Reset behavior tracking state.

        Call this when starting a new coaching session to prevent false
        positives from accumulated hold time before the session started.
        Also records timestamp to filter out stale threaded events.
        """
        with self._lock:
            self._last_behavior = None
            self._behavior_start_time = 0.0
            self._last_confidence = 0.0
            self._last_update_time = 0.0
            self._reset_timestamp = time.time()  # Events before this are stale
            logger.debug(f"BehaviorInterpreter tracking reset at {self._reset_timestamp:.3f}")

    def check_trick(self, trick_name: str, dog_id: str = None) -> TrickCheckResult:
        """
        Check if current detection meets trick requirements.

        Args:
            trick_name: Name of trick ('sit', 'down', etc.)
            dog_id: Ignored (kept for API compatibility)

        Returns:
            TrickCheckResult with completion status
        """
        rules = self.trick_rules.get(trick_name, {})
        if not rules:
            return TrickCheckResult(completed=False, reason=f"Unknown trick: {trick_name}")

        required = rules.get('required_behavior', trick_name)
        alternatives = rules.get('alternative_behaviors', [])
        hold_required = rules.get('hold_duration_sec', 1.0)
        conf_threshold = rules.get('confidence_threshold', 0.65)

        with self._lock:
            # Check if detection is stale (>2 sec old)
            if time.time() - self._last_update_time > 2.0:
                return TrickCheckResult(completed=False, reason="No recent detection")

            if not self._last_behavior:
                return TrickCheckResult(completed=False, reason="No behavior detected")

            # Check behavior matches
            valid = [required] + alternatives
            if self._last_behavior.lower() not in [b.lower() for b in valid]:
                return TrickCheckResult(
                    completed=False,
                    behavior_detected=self._last_behavior,
                    confidence=self._last_confidence,
                    reason=f"Wrong behavior: {self._last_behavior}"
                )

            # Check confidence
            if self._last_confidence < conf_threshold:
                return TrickCheckResult(
                    completed=False,
                    behavior_detected=self._last_behavior,
                    confidence=self._last_confidence,
                    reason=f"Low confidence: {self._last_confidence:.2f}"
                )

            # Check hold duration
            hold_duration = time.time() - self._behavior_start_time
            if hold_duration < hold_required:
                return TrickCheckResult(
                    completed=False,
                    behavior_detected=self._last_behavior,
                    confidence=self._last_confidence,
                    hold_duration=hold_duration,
                    reason=f"Hold: {hold_duration:.1f}s / {hold_required}s"
                )

            # Success!
            return TrickCheckResult(
                completed=True,
                behavior_detected=self._last_behavior,
                confidence=self._last_confidence,
                hold_duration=hold_duration,
                reason="Trick completed"
            )

    def get_trick_rules(self, trick_name: str) -> Dict[str, Any]:
        """Get rules for a specific trick"""
        return self.trick_rules.get(trick_name, {})

    def get_all_tricks(self) -> List[str]:
        """Get list of all defined tricks"""
        return list(self.trick_rules.keys())

    def get_status(self) -> Dict[str, Any]:
        """Get current interpreter status (for logging/debug)"""
        with self._lock:
            hold = time.time() - self._behavior_start_time if self._last_behavior else 0
            return {
                'current_behavior': self._last_behavior,
                'confidence': self._last_confidence,
                'hold_duration': hold,
                'last_update': self._last_update_time,
                'tricks_defined': list(self.trick_rules.keys()),
                'thresholds': self.confidence_thresholds,
            }


# Singleton
_interpreter_instance = None
_interpreter_lock = threading.Lock()


def get_behavior_interpreter() -> BehaviorInterpreter:
    """Get the global behavior interpreter instance"""
    global _interpreter_instance
    if _interpreter_instance is None:
        with _interpreter_lock:
            if _interpreter_instance is None:
                _interpreter_instance = BehaviorInterpreter()
    return _interpreter_instance
