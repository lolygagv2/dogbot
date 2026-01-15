"""
Stage 1: BarkGate - Signal processing bark detection (no ML)

Ported from proven Arduino approach. Uses energy thresholds, duration checks,
and cooldowns to reliably detect barks without machine learning.

This answers: "Is this a bark? Yes/No"
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class BarkGateConfig:
    """Configuration for bark gate thresholds"""
    # Energy thresholds - tuned for USB mic with ~0.02 ambient baseline
    # Barks should produce energy 0.10+ (5x ambient)
    base_threshold: float = 0.12      # Just above ambient noise floor
    thresh_close: float = 0.35        # Close/loud bark
    thresh_mid: float = 0.20          # Medium distance bark
    thresh_far: float = 0.12          # Far/quiet bark

    # Timing (in milliseconds)
    min_bark_duration_ms: int = 80    # Barks are typically 80-500ms
    max_bark_duration_ms: int = 2000  # Cap duration (barks rarely exceed 2s)
    grace_period_ms: int = 100        # Wait for bark "tail" before ending
    bark_cooldown_ms: int = 1000      # Cooldown between barks


class BarkGate:
    """
    Stage 1: Signal processing bark detection.

    Uses energy thresholds and timing to detect barks without ML.
    Ported from Arduino implementation.

    Usage:
        gate = BarkGate()
        result = gate.process_audio_chunk(energy=0.45, timestamp_ms=12345)
        if result['is_bark']:
            print(f"Bark detected! Distance: {result['distance']}")
    """

    def __init__(self, config: Optional[BarkGateConfig] = None):
        """
        Initialize bark gate.

        Args:
            config: Optional configuration override
        """
        self.config = config or BarkGateConfig()

        # State tracking
        self.in_bark = False
        self.bark_start_time = 0
        self.peak_energy = 0.0
        self.waiting_grace = False
        self.grace_start_time = 0
        self.last_valid_bark_time = 0

        # Statistics
        self.total_barks = 0
        self.barks_by_distance = {'close': 0, 'mid': 0, 'far': 0}

        logger.info(f"BarkGate initialized: threshold={self.config.base_threshold}, "
                   f"cooldown={self.config.bark_cooldown_ms}ms")

    def process_audio_chunk(self, energy: float, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Process audio energy level and detect barks.

        This implements the Arduino state machine:
        1. Energy above threshold starts a bark
        2. Track peak energy during bark
        3. When energy drops, start grace period
        4. After grace period, validate and emit bark event

        Args:
            energy: Audio energy level (0.0 to 1.0+, typically RMS)
            timestamp_ms: Current timestamp in ms (auto-generated if not provided)

        Returns:
            dict with keys:
                - is_bark: bool - True if a complete bark was detected
                - distance: str - 'close', 'mid', or 'far' (only if is_bark)
                - peak: float - Peak energy during bark (only if is_bark)
                - duration_ms: int - Bark duration (only if is_bark)
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        result = {'is_bark': False}
        cfg = self.config

        if energy > cfg.base_threshold:
            if not self.in_bark:
                # Bark starting
                self.in_bark = True
                self.bark_start_time = timestamp_ms
                self.peak_energy = energy
                self.waiting_grace = False
                logger.debug(f"Bark start: energy={energy:.3f}")
            else:
                # Bark continuing - track peak
                if energy > self.peak_energy:
                    self.peak_energy = energy
                # Reset grace period if energy spiked again
                self.waiting_grace = False

        elif self.in_bark and not self.waiting_grace:
            # Energy dropped below threshold - start grace period
            self.waiting_grace = True
            self.grace_start_time = timestamp_ms
            logger.debug(f"Grace period started: peak={self.peak_energy:.3f}")

        elif self.waiting_grace:
            if timestamp_ms - self.grace_start_time > cfg.grace_period_ms:
                # Grace period ended - bark is complete
                duration = self.grace_start_time - self.bark_start_time
                self.in_bark = False
                self.waiting_grace = False

                # Cap duration to reasonable max (handles chunked input)
                duration = min(duration, cfg.max_bark_duration_ms)

                # Validate: duration check
                if duration < cfg.min_bark_duration_ms:
                    logger.debug(f"Rejected: too short ({duration}ms < {cfg.min_bark_duration_ms}ms)")
                    return result

                # Validate: cooldown check
                if timestamp_ms - self.last_valid_bark_time < cfg.bark_cooldown_ms:
                    logger.debug(f"Rejected: cooldown active")
                    return result

                # Valid bark! Classify distance by peak energy
                if self.peak_energy >= cfg.thresh_close:
                    distance = 'close'
                elif self.peak_energy >= cfg.thresh_mid:
                    distance = 'mid'
                else:
                    distance = 'far'

                self.last_valid_bark_time = timestamp_ms
                self.total_barks += 1
                self.barks_by_distance[distance] += 1

                result = {
                    'is_bark': True,
                    'distance': distance,
                    'peak': self.peak_energy,
                    'duration_ms': duration
                }

                logger.info(f"BARK detected: {distance}, peak={self.peak_energy:.3f}, "
                           f"duration={duration}ms")

        return result

    def reset(self):
        """Reset state (e.g., when switching dogs or modes)"""
        self.in_bark = False
        self.bark_start_time = 0
        self.peak_energy = 0.0
        self.waiting_grace = False
        self.grace_start_time = 0
        # Don't reset last_valid_bark_time - keep cooldown active

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'total_barks': self.total_barks,
            'by_distance': self.barks_by_distance.copy(),
            'config': {
                'base_threshold': self.config.base_threshold,
                'thresh_close': self.config.thresh_close,
                'thresh_mid': self.config.thresh_mid,
                'thresh_far': self.config.thresh_far,
                'cooldown_ms': self.config.bark_cooldown_ms
            }
        }

    def update_thresholds(self, base: float = None, close: float = None,
                         mid: float = None, far: float = None):
        """
        Update energy thresholds (for runtime calibration).

        Args:
            base: New base threshold
            close: New close bark threshold
            mid: New mid bark threshold
            far: New far bark threshold
        """
        if base is not None:
            self.config.base_threshold = base
            self.config.thresh_far = base  # Far threshold = base
        if close is not None:
            self.config.thresh_close = close
        if mid is not None:
            self.config.thresh_mid = mid
        if far is not None:
            self.config.thresh_far = far

        logger.info(f"Thresholds updated: base={self.config.base_threshold}, "
                   f"close={self.config.thresh_close}, mid={self.config.thresh_mid}, "
                   f"far={self.config.thresh_far}")
