"""
Stage 3: BarkAnalytics - Per-dog bark session tracking

Tracks bark patterns for each dog during a session:
- Bark count and rate (barks per hour)
- Silence intervals (top 5 longest)
- Total barking vs silence time
- Distance distribution

Ported from Arduino Telegram reporting system.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class BarkSession:
    """
    Per-dog bark tracking for a session.

    Tracks all barking activity for one dog, providing analytics
    for reporting (dashboard, Telegram, etc.).
    """
    dog_id: str
    dog_name: str = ""
    session_start: float = field(default_factory=time.time)

    # Counters
    bark_count: int = 0
    barks_by_distance: Dict[str, int] = field(default_factory=lambda: {'close': 0, 'mid': 0, 'far': 0})
    barks_by_emotion: Dict[str, int] = field(default_factory=dict)

    # Time tracking (in milliseconds)
    total_silence_ms: int = 0
    total_barking_ms: int = 0
    last_bark_time: float = 0

    # Intervals
    bark_intervals: List[Tuple[int, float]] = field(default_factory=list)  # (interval_ms, timestamp)
    top_silences: List[Tuple[int, float]] = field(default_factory=list)    # Top 5 longest silences

    def record_bark(self, timestamp: float, duration_ms: int, distance: str,
                   peak_energy: float = 0.0, emotion: str = None):
        """
        Record a bark event.

        Args:
            timestamp: Unix timestamp of bark
            duration_ms: Bark duration in milliseconds
            distance: 'close', 'mid', or 'far'
            peak_energy: Peak audio energy during bark
            emotion: Optional emotion classification
        """
        self.bark_count += 1
        self.barks_by_distance[distance] = self.barks_by_distance.get(distance, 0) + 1

        if emotion:
            self.barks_by_emotion[emotion] = self.barks_by_emotion.get(emotion, 0) + 1

        # Track intervals between barks
        if self.last_bark_time > 0:
            interval_ms = int((timestamp - self.last_bark_time) * 1000)
            self.bark_intervals.append((interval_ms, timestamp))

            # Classify interval as silence or barking period
            # >2 minutes between barks = silence period
            if interval_ms > 120000:
                self.total_silence_ms += interval_ms
                self._update_top_silences(interval_ms, timestamp)
            else:
                self.total_barking_ms += interval_ms

        self.last_bark_time = timestamp

        logger.debug(f"Recorded bark for {self.dog_id}: #{self.bark_count}, "
                    f"{distance}, {duration_ms}ms, emotion={emotion}")

    def _update_top_silences(self, interval_ms: int, timestamp: float):
        """Keep track of top 5 longest silence intervals"""
        self.top_silences.append((interval_ms, timestamp))
        self.top_silences.sort(key=lambda x: x[0], reverse=True)
        self.top_silences = self.top_silences[:5]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get session summary for reporting.

        Returns:
            Dict with session analytics
        """
        now = time.time()
        duration_seconds = now - self.session_start
        duration_hours = duration_seconds / 3600

        # Calculate barks per hour
        barks_per_hour = self.bark_count / max(duration_hours, 0.01)

        # Format durations
        summary = {
            'dog_id': self.dog_id,
            'dog_name': self.dog_name,
            'session_start': self.session_start,
            'session_duration': self._format_duration(int(duration_seconds * 1000)),
            'session_duration_hours': round(duration_hours, 2),

            'total_barks': self.bark_count,
            'barks_per_hour': round(barks_per_hour, 1),

            'by_distance': self.barks_by_distance.copy(),
            'by_emotion': self.barks_by_emotion.copy(),

            'total_silence': self._format_duration(self.total_silence_ms),
            'total_silence_ms': self.total_silence_ms,
            'total_barking': self._format_duration(self.total_barking_ms),
            'total_barking_ms': self.total_barking_ms,

            'top_5_silences': [
                {
                    'duration': self._format_duration(s[0]),
                    'duration_ms': s[0],
                    'at': self._format_duration(int((s[1] - self.session_start) * 1000))
                }
                for s in self.top_silences
            ]
        }

        return summary

    @staticmethod
    def _format_duration(ms: int) -> str:
        """Format milliseconds as human-readable duration"""
        if ms < 1000:
            return f"{ms}ms"
        elif ms < 60000:
            return f"{ms // 1000}s"
        elif ms < 3600000:
            minutes = ms // 60000
            seconds = (ms % 60000) // 1000
            return f"{minutes}m {seconds}s"
        else:
            hours = ms // 3600000
            minutes = (ms % 3600000) // 60000
            return f"{hours}h {minutes}m"

    def reset(self):
        """Reset session (start fresh)"""
        self.session_start = time.time()
        self.bark_count = 0
        self.barks_by_distance = {'close': 0, 'mid': 0, 'far': 0}
        self.barks_by_emotion = {}
        self.total_silence_ms = 0
        self.total_barking_ms = 0
        self.last_bark_time = 0
        self.bark_intervals = []
        self.top_silences = []


class BarkAnalytics:
    """
    Multi-dog bark analytics manager.

    Maintains BarkSession for each dog and provides
    aggregated reporting.
    """

    def __init__(self):
        """Initialize analytics manager"""
        self.sessions: Dict[str, BarkSession] = {}
        self.global_start = time.time()

        logger.info("BarkAnalytics initialized")

    def get_or_create_session(self, dog_id: str, dog_name: str = "") -> BarkSession:
        """
        Get existing session or create new one for dog.

        Args:
            dog_id: Dog identifier (e.g., ArUco marker ID)
            dog_name: Human-readable name (e.g., "Elsa")

        Returns:
            BarkSession for the dog
        """
        if dog_id not in self.sessions:
            self.sessions[dog_id] = BarkSession(
                dog_id=dog_id,
                dog_name=dog_name or dog_id
            )
            logger.info(f"Created bark session for {dog_name or dog_id}")

        return self.sessions[dog_id]

    def record_bark(self, dog_id: str, duration_ms: int, distance: str,
                   peak_energy: float = 0.0, emotion: str = None,
                   dog_name: str = ""):
        """
        Record a bark for a specific dog.

        Args:
            dog_id: Dog identifier
            duration_ms: Bark duration
            distance: 'close', 'mid', or 'far'
            peak_energy: Peak audio energy
            emotion: Optional emotion classification
            dog_name: Human-readable name
        """
        session = self.get_or_create_session(dog_id, dog_name)
        session.record_bark(
            timestamp=time.time(),
            duration_ms=duration_ms,
            distance=distance,
            peak_energy=peak_energy,
            emotion=emotion
        )

    def get_dog_summary(self, dog_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for specific dog"""
        if dog_id in self.sessions:
            return self.sessions[dog_id].get_summary()
        return None

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all dogs"""
        return {
            dog_id: session.get_summary()
            for dog_id, session in self.sessions.items()
        }

    def get_global_summary(self) -> Dict[str, Any]:
        """Get aggregated summary across all dogs"""
        total_barks = sum(s.bark_count for s in self.sessions.values())
        duration_hours = (time.time() - self.global_start) / 3600

        by_distance = {'close': 0, 'mid': 0, 'far': 0}
        by_emotion = {}

        for session in self.sessions.values():
            for dist, count in session.barks_by_distance.items():
                by_distance[dist] += count
            for emotion, count in session.barks_by_emotion.items():
                by_emotion[emotion] = by_emotion.get(emotion, 0) + count

        return {
            'total_dogs': len(self.sessions),
            'total_barks': total_barks,
            'barks_per_hour': round(total_barks / max(duration_hours, 0.01), 1),
            'session_duration_hours': round(duration_hours, 2),
            'by_distance': by_distance,
            'by_emotion': by_emotion,
            'dogs': list(self.sessions.keys())
        }

    def reset_dog(self, dog_id: str):
        """Reset session for specific dog"""
        if dog_id in self.sessions:
            self.sessions[dog_id].reset()

    def reset_all(self):
        """Reset all sessions"""
        self.sessions.clear()
        self.global_start = time.time()
