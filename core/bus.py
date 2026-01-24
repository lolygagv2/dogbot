#!/usr/bin/env python3
"""
Event bus for inter-component communication
Thread-safe publish/subscribe pattern for TreatBot services
"""

import threading
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Event Types
class EventType(Enum):
    VISION = "vision"
    AUDIO = "audio"
    MOTION = "motion"
    SYSTEM = "system"
    REWARD = "reward"
    SAFETY = "safety"
    CLOUD = "cloud"

@dataclass
class Event:
    """Base event class"""
    type: EventType
    subtype: str
    timestamp: float
    data: Dict[str, Any]
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'subtype': self.subtype,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# Specific Event Types
@dataclass
class VisionEvent(Event):
    """Vision system events"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "vision"):
        super().__init__(EventType.VISION, subtype, time.time(), data, source)

@dataclass
class AudioEvent(Event):
    """Audio system events"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "audio"):
        super().__init__(EventType.AUDIO, subtype, time.time(), data, source)

@dataclass
class MotionEvent(Event):
    """Motion/motor events"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "motion"):
        super().__init__(EventType.MOTION, subtype, time.time(), data, source)

@dataclass
class SystemEvent(Event):
    """System status events"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "system"):
        super().__init__(EventType.SYSTEM, subtype, time.time(), data, source)

@dataclass
class RewardEvent(Event):
    """Reward system events"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "reward"):
        super().__init__(EventType.REWARD, subtype, time.time(), data, source)

@dataclass
class SafetyEvent(Event):
    """Safety monitoring events"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "safety"):
        super().__init__(EventType.SAFETY, subtype, time.time(), data, source)

@dataclass
class CloudEvent(Event):
    """Cloud relay events (commands from app)"""
    def __init__(self, subtype: str, data: Dict[str, Any], source: str = "cloud"):
        super().__init__(EventType.CLOUD, subtype, time.time(), data, source)


class EventBus:
    """
    Thread-safe event bus for TreatBot
    Supports publish/subscribe pattern with filtering
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
        self._lock = threading.RLock()
        self.logger = logging.getLogger('EventBus')

        # Initialize subscriber lists for all event types
        for event_type in EventType:
            self.subscribers[event_type.value] = []

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to events of a specific type

        Args:
            event_type: Event type to subscribe to (e.g., 'vision', 'audio')
            callback: Function to call when event occurs
        """
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []

            if callback not in self.subscribers[event_type]:
                self.subscribers[event_type].append(callback)
                self.logger.debug(f"Subscribed to {event_type} events")

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events"""
        with self._lock:
            if event_type in self.subscribers:
                if callback in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(callback)
                    self.logger.debug(f"Unsubscribed from {event_type} events")

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers

        Args:
            event: Event to publish
        """
        with self._lock:
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)

            # Notify subscribers
            event_type = event.type.value
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    try:
                        # Call subscriber in separate thread to avoid blocking
                        threading.Thread(
                            target=self._safe_callback,
                            args=(callback, event),
                            daemon=True
                        ).start()
                    except Exception as e:
                        self.logger.error(f"Failed to notify subscriber: {e}")

            self.logger.debug(f"Published {event_type}.{event.subtype} event")

    def _safe_callback(self, callback: Callable, event: Event) -> None:
        """Execute callback with error handling"""
        try:
            callback(event)
        except Exception as e:
            self.logger.error(f"Subscriber callback failed: {e}")

    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 50) -> List[Event]:
        """Get recent events, optionally filtered by type"""
        with self._lock:
            events = self.event_history[-limit:]
            if event_type:
                events = [e for e in events if e.type.value == event_type]
            return events

    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type"""
        with self._lock:
            return len(self.subscribers.get(event_type, []))

    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self.event_history.clear()
            self.logger.info("Event history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics"""
        with self._lock:
            return {
                'total_events': len(self.event_history),
                'subscribers': {k: len(v) for k, v in self.subscribers.items()},
                'event_types': [e.type.value for e in self.event_history[-10:]],
                'last_event': self.event_history[-1].to_dict() if self.event_history else None
            }


# Global event bus instance
_bus_instance = None
_bus_lock = threading.Lock()

def get_bus() -> EventBus:
    """Get the global event bus instance (singleton)"""
    global _bus_instance
    if _bus_instance is None:
        with _bus_lock:
            if _bus_instance is None:
                _bus_instance = EventBus()
    return _bus_instance

# Convenience functions for common event patterns
def publish_vision_event(subtype: str, data: Dict[str, Any], source: str = "vision") -> None:
    """Publish a vision event"""
    get_bus().publish(VisionEvent(subtype, data, source))

def publish_audio_event(subtype: str, data: Dict[str, Any], source: str = "audio") -> None:
    """Publish an audio event"""
    get_bus().publish(AudioEvent(subtype, data, source))

def publish_motion_event(subtype: str, data: Dict[str, Any], source: str = "motion") -> None:
    """Publish a motion event"""
    get_bus().publish(MotionEvent(subtype, data, source))

def publish_system_event(subtype: str, data: Dict[str, Any], source: str = "system") -> None:
    """Publish a system event"""
    get_bus().publish(SystemEvent(subtype, data, source))

def publish_reward_event(subtype: str, data: Dict[str, Any], source: str = "reward") -> None:
    """Publish a reward event"""
    get_bus().publish(RewardEvent(subtype, data, source))

def publish_safety_event(subtype: str, data: Dict[str, Any], source: str = "safety") -> None:
    """Publish a safety event"""
    get_bus().publish(SafetyEvent(subtype, data, source))


if __name__ == "__main__":
    # Test the event bus
    import time

    # Create bus
    bus = get_bus()

    # Test subscriber
    def test_handler(event):
        print(f"Received: {event.type.value}.{event.subtype} - {event.data}")

    # Subscribe to vision events
    bus.subscribe('vision', test_handler)

    # Publish test events
    publish_vision_event('dog_detected', {'confidence': 0.95, 'bbox': [100, 100, 200, 200]})
    publish_vision_event('pose', {'behavior': 'sitting', 'keypoints': 24})

    # Wait for async handlers
    time.sleep(0.1)

    # Show stats
    print("Bus stats:", bus.get_stats())