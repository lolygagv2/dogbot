#!/usr/bin/env python3
"""
Dog Event Logger — subscribes to bus events and persists to dog_events table.
Captures barks, treats, detections, tricks, missions, SG escalations, quiet periods.
"""

import logging
import uuid
from typing import Dict, Any

from core.bus import get_bus, Event
from core.store import get_store
from core.state import get_state


class DogEventLogger:
    """Listens to EventBus and logs all dog-relevant events to SQLite."""

    def __init__(self):
        self.logger = logging.getLogger('DogEventLogger')
        self._session_id = str(uuid.uuid4())[:8]
        self._running = False

    def start(self):
        """Subscribe to all relevant event types on the bus."""
        if self._running:
            return
        self._running = True

        bus = get_bus()
        bus.subscribe('audio', self._on_audio_event)
        bus.subscribe('vision', self._on_vision_event)
        bus.subscribe('reward', self._on_reward_event)
        bus.subscribe('system', self._on_system_event)
        self.logger.info(f"DogEventLogger started (session={self._session_id})")

    def stop(self):
        """Unsubscribe from events."""
        if not self._running:
            return
        self._running = False

        bus = get_bus()
        bus.unsubscribe('audio', self._on_audio_event)
        bus.unsubscribe('vision', self._on_vision_event)
        bus.unsubscribe('reward', self._on_reward_event)
        bus.unsubscribe('system', self._on_system_event)
        self.logger.info("DogEventLogger stopped")

    def _current_mode(self) -> str:
        try:
            return get_state().get_mode().value
        except Exception:
            return "unknown"

    def _log(self, event_type: str, dog_id: str = "unknown",
             dog_name: str = "", details: Dict[str, Any] = None):
        """Write event to store (fire-and-forget)."""
        try:
            get_store().log_dog_event(
                event_type=event_type,
                dog_id=dog_id,
                dog_name=dog_name,
                details=details or {},
                mode=self._current_mode(),
                session_id=self._session_id
            )
        except Exception as e:
            self.logger.error(f"Failed to log dog event: {e}")

    # ---- Event handlers ----

    def _on_audio_event(self, event: Event):
        """Handle bark and audio events."""
        if event.subtype in ('bark_detected', 'bark'):
            self._log(
                'bark',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'confidence': event.data.get('confidence', 0),
                    'count': event.data.get('bark_count', 1),
                    'duration': event.data.get('duration', 0),
                }
            )
        elif event.subtype == 'quiet_period':
            self._log(
                'quiet',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'duration': event.data.get('duration', 0),
                }
            )

    def _on_vision_event(self, event: Event):
        """Handle detection and behavior events."""
        if event.subtype == 'dog_detected':
            self._log(
                'detection',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'confidence': event.data.get('confidence', 0),
                    'behavior': event.data.get('behavior', ''),
                    'id_method': event.data.get('id_method', ''),
                }
            )
        elif event.subtype in ('trick_success', 'trick_detected'):
            self._log(
                'trick_success',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'trick': event.data.get('trick', event.data.get('behavior', '')),
                    'confidence': event.data.get('confidence', 0),
                    'duration': event.data.get('duration', 0),
                }
            )
        elif event.subtype == 'trick_fail':
            self._log(
                'trick_fail',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'trick': event.data.get('trick', ''),
                    'reason': event.data.get('reason', ''),
                }
            )

    def _on_reward_event(self, event: Event):
        """Handle treat dispensing events."""
        if event.subtype in ('dispensed', 'treat_dispensed', 'reward_given'):
            self._log(
                'treat',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'behavior': event.data.get('behavior', ''),
                    'treats': event.data.get('treats_dispensed', 1),
                    'reason': event.data.get('reason', ''),
                }
            )

    def _on_system_event(self, event: Event):
        """Handle mode changes, missions, SG escalations."""
        if event.subtype == 'mode_changed':
            self._log(
                'mode_change',
                details={
                    'previous': event.data.get('previous_mode', ''),
                    'new': event.data.get('new_mode', ''),
                    'reason': event.data.get('reason', ''),
                }
            )
        elif event.subtype == 'mission_started':
            self._log(
                'mission_start',
                details={
                    'mission': event.data.get('name', ''),
                    'config': event.data.get('config', {}),
                }
            )
        elif event.subtype in ('mission_completed', 'mission_stopped', 'mission_failed'):
            self._log(
                'mission_end',
                details={
                    'mission': event.data.get('name', ''),
                    'status': event.subtype.replace('mission_', ''),
                    'rewards_given': event.data.get('rewards_given', 0),
                }
            )
        elif event.subtype == 'sg_escalation':
            self._log(
                'sg_escalation',
                dog_id=event.data.get('dog_id', 'unknown'),
                dog_name=event.data.get('dog_name', ''),
                details={
                    'level': event.data.get('escalation_level', 0),
                    'barks': event.data.get('barks_triggering', 0),
                    'action': event.data.get('action', ''),
                }
            )


# Singleton
_instance = None

def get_dog_event_logger() -> DogEventLogger:
    global _instance
    if _instance is None:
        _instance = DogEventLogger()
    return _instance
