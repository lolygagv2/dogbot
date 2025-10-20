#!/usr/bin/env python3
"""
Event Bus System for inter-module communication
Allows decoupled communication between robot subsystems
"""

import threading
import logging
from typing import Dict, List, Callable, Any
from collections import defaultdict

class EventBus:
    """Thread-safe event bus for pub/sub communication between modules"""

    def __init__(self):
        self.logger = logging.getLogger('EventBus')
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.lock = threading.RLock()

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to an event type

        Args:
            event_type: Name of the event to subscribe to
            callback: Function to call when event occurs
        """
        with self.lock:
            self.subscribers[event_type].append(callback)
            self.logger.debug(f"Subscribed to '{event_type}', {len(self.subscribers[event_type])} total subscribers")

    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from an event type

        Args:
            event_type: Name of the event to unsubscribe from
            callback: The callback function to remove
        """
        with self.lock:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                self.logger.debug(f"Unsubscribed from '{event_type}'")

    def publish(self, event_type: str, event_data: Dict[str, Any] = None):
        """
        Publish an event to all subscribers

        Args:
            event_type: Name of the event to publish
            event_data: Data to send with the event
        """
        if event_data is None:
            event_data = {}

        with self.lock:
            subscribers = self.subscribers[event_type].copy()

        if subscribers:
            self.logger.debug(f"Publishing '{event_type}' to {len(subscribers)} subscribers")

            for callback in subscribers:
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.error(f"Error in event callback for '{event_type}': {e}")

    def get_subscriber_count(self, event_type: str) -> int:
        """Get number of subscribers for an event type"""
        with self.lock:
            return len(self.subscribers[event_type])

    def get_all_event_types(self) -> List[str]:
        """Get list of all event types with subscribers"""
        with self.lock:
            return list(self.subscribers.keys())

    def clear_subscribers(self, event_type: str = None):
        """Clear subscribers for an event type or all event types"""
        with self.lock:
            if event_type:
                self.subscribers[event_type].clear()
                self.logger.debug(f"Cleared all subscribers for '{event_type}'")
            else:
                self.subscribers.clear()
                self.logger.debug("Cleared all subscribers for all events")