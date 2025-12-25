#!/usr/bin/env python3
"""
Bark Frequency Tracker - Real-time bark counting per time window
Used for mission triggers (e.g., "3 barks in 1 minute" triggers quiet training)
"""

import time
import threading
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DogBarkState:
    """Tracking state for a single dog"""
    bark_times: List[float] = field(default_factory=list)
    daily_events: int = 0  # How many times threshold exceeded today
    last_event_time: float = 0
    escalation_level: int = 0  # 0=normal, 1+=escalated


class BarkFrequencyTracker:
    """
    Tracks bark frequency per dog for mission triggering

    Features:
    - Per-dog bark counting with time window
    - Threshold detection (e.g., 3 barks in 60 seconds)
    - Daily event counting for escalation
    - Thread-safe operations
    """

    def __init__(self, window_seconds: int = 60, default_threshold: int = 3):
        """
        Initialize tracker

        Args:
            window_seconds: Time window for counting barks
            default_threshold: Default bark count to trigger event
        """
        self.window_seconds = window_seconds
        self.default_threshold = default_threshold
        self.dogs: Dict[str, DogBarkState] = {}
        self._lock = threading.Lock()
        self._last_reset_day = datetime.now().date()

        logger.info(f"BarkFrequencyTracker initialized: window={window_seconds}s, threshold={default_threshold}")

    def _get_dog_state(self, dog_id: str) -> DogBarkState:
        """Get or create dog state"""
        if dog_id not in self.dogs:
            self.dogs[dog_id] = DogBarkState()
        return self.dogs[dog_id]

    def _cleanup_old(self, dog_id: str):
        """Remove barks outside the time window"""
        cutoff = time.time() - self.window_seconds
        state = self.dogs.get(dog_id)
        if state:
            state.bark_times = [t for t in state.bark_times if t > cutoff]

    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today != self._last_reset_day:
            logger.info("New day - resetting daily bark event counters")
            for dog_id, state in self.dogs.items():
                state.daily_events = 0
                state.escalation_level = 0
            self._last_reset_day = today

    def record_bark(self, dog_id: str = 'unknown') -> Dict:
        """
        Record a bark event

        Args:
            dog_id: Dog identifier (ArUco ID or "unknown")

        Returns:
            Dict with current state: bark_count, threshold_exceeded, escalation_level
        """
        with self._lock:
            self._check_daily_reset()
            self._cleanup_old(dog_id)

            state = self._get_dog_state(dog_id)
            state.bark_times.append(time.time())

            bark_count = len(state.bark_times)
            exceeded = bark_count >= self.default_threshold

            result = {
                'dog_id': dog_id,
                'bark_count': bark_count,
                'window_seconds': self.window_seconds,
                'threshold': self.default_threshold,
                'threshold_exceeded': exceeded,
                'daily_events': state.daily_events,
                'escalation_level': state.escalation_level
            }

            logger.debug(f"Bark recorded for {dog_id}: {bark_count}/{self.default_threshold} in {self.window_seconds}s")

            return result

    def get_barks_per_minute(self, dog_id: str) -> int:
        """
        Get bark count in the time window

        Args:
            dog_id: Dog identifier

        Returns:
            Number of barks in window
        """
        with self._lock:
            self._cleanup_old(dog_id)
            state = self.dogs.get(dog_id)
            if state:
                return len(state.bark_times)
            return 0

    def check_threshold(self, dog_id: str, threshold: int = None) -> bool:
        """
        Check if dog exceeded bark threshold

        Args:
            dog_id: Dog identifier
            threshold: Custom threshold (uses default if None)

        Returns:
            True if threshold exceeded
        """
        if threshold is None:
            threshold = self.default_threshold

        count = self.get_barks_per_minute(dog_id)
        return count >= threshold

    def record_event_triggered(self, dog_id: str):
        """
        Record that a threshold event was triggered (for daily counting)

        Args:
            dog_id: Dog identifier
        """
        with self._lock:
            state = self._get_dog_state(dog_id)
            state.daily_events += 1
            state.last_event_time = time.time()

            # Check for escalation (after 5 events)
            if state.daily_events >= 5 and state.escalation_level == 0:
                state.escalation_level = 1
                logger.info(f"Escalation triggered for {dog_id}: {state.daily_events} events today")

            logger.info(f"Event triggered for {dog_id}: daily_events={state.daily_events}, "
                       f"escalation={state.escalation_level}")

    def get_escalation_level(self, dog_id: str) -> int:
        """
        Get current escalation level for dog

        Args:
            dog_id: Dog identifier

        Returns:
            Escalation level (0=normal, 1+=escalated)
        """
        with self._lock:
            state = self.dogs.get(dog_id)
            if state:
                return state.escalation_level
            return 0

    def get_daily_events(self, dog_id: str) -> int:
        """
        Get daily event count for dog

        Args:
            dog_id: Dog identifier

        Returns:
            Number of threshold events today
        """
        with self._lock:
            self._check_daily_reset()
            state = self.dogs.get(dog_id)
            if state:
                return state.daily_events
            return 0

    def get_quiet_duration_requirement(self, dog_id: str, base_seconds: int = 5,
                                        max_seconds: int = 20) -> int:
        """
        Get required quiet duration based on daily progression

        Args:
            dog_id: Dog identifier
            base_seconds: Starting quiet requirement
            max_seconds: Maximum quiet requirement

        Returns:
            Required seconds of silence
        """
        with self._lock:
            state = self.dogs.get(dog_id)
            if not state:
                return base_seconds

            # Scale from base to max based on daily events
            # 0 events = 5s, 5+ events = 20s
            events = state.daily_events
            if events >= 5:
                return max_seconds

            # Linear interpolation
            progress = events / 5.0
            return int(base_seconds + (max_seconds - base_seconds) * progress)

    def get_status(self) -> Dict:
        """Get overall tracker status"""
        with self._lock:
            self._check_daily_reset()

            status = {
                'window_seconds': self.window_seconds,
                'default_threshold': self.default_threshold,
                'dogs_tracked': len(self.dogs),
                'per_dog': {}
            }

            for dog_id, state in self.dogs.items():
                self._cleanup_old(dog_id)
                status['per_dog'][dog_id] = {
                    'barks_in_window': len(state.bark_times),
                    'daily_events': state.daily_events,
                    'escalation_level': state.escalation_level
                }

            return status

    def reset_dog(self, dog_id: str):
        """Reset tracking for a specific dog"""
        with self._lock:
            if dog_id in self.dogs:
                del self.dogs[dog_id]
                logger.info(f"Reset tracking for {dog_id}")

    def reset_all(self):
        """Reset all tracking"""
        with self._lock:
            self.dogs.clear()
            logger.info("Reset all bark tracking")


# Singleton instance
_tracker_instance = None


def get_bark_frequency_tracker() -> BarkFrequencyTracker:
    """Get or create bark frequency tracker singleton"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = BarkFrequencyTracker()
    return _tracker_instance
