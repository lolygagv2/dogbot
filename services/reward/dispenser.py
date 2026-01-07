#!/usr/bin/env python3
"""
Treat dispensing service
Uses ServoController carousel servo to dispense treats
This is for testing and development purposes
"""

import threading
import time
import logging
from typing import Dict, Any, Optional

from core.bus import get_bus, publish_reward_event
from core.state import get_state
from core.store import get_store
from core.hardware.servo_controller import ServoController
from config.config_loader import get_config


class DispenserService:
    """
    Treat dispensing service using carousel servo
    Tracks treats dispensed per dog and implements portion control
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.logger = logging.getLogger('DispenserService')

        # Servo controller
        self.servo = None
        self.servo_initialized = False

        # Dispensing state
        self.treats_dispensed_today = 0
        self.last_dispense_time = 0.0
        self.min_dispense_interval = 0.0  # TESTING: removed cooldown - was 1.0 seconds between dispenses
        self.daily_limit = 300  # max treats per day

        # Per-dog tracking
        self.dog_treat_counts = {}  # dog_id -> count
        self.dog_cooldowns = {}     # dog_id -> last_dispense_time

        # Dispensing parameters - loaded from robot-specific config
        robot_config = get_config()
        self.dispense_pulse = 1300  # NOT USED? microseconds - legacy value
        self.dispense_duration = robot_config.dispenser.dispense_duration  # Robot-specific duration

        # Thread safety
        self._dispense_lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize servo controller"""
        try:
            self.servo = ServoController()

            if self.servo.is_initialized():
                self.servo_initialized = True
                self.logger.info("Treat dispenser initialized")
                return True
            else:
                self.logger.error("Servo controller initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"Dispenser initialization error: {e}")
            return False

    def dispense_treat(self, dog_id: Optional[str] = None, reason: str = "manual",
                      behavior: str = "", confidence: float = 0.0) -> bool:
        """
        Dispense a single treat

        Args:
            dog_id: ID of dog receiving treat (optional)
            reason: Reason for dispensing (manual, reward, etc.)
            behavior: Behavior that triggered reward
            confidence: Confidence of behavior detection

        Returns:
            bool: True if treat was dispensed successfully
        """
        with self._dispense_lock:
            # Auto-initialize if not already done
            if not self.servo_initialized:
                self.logger.info("Auto-initializing dispenser on first use")
                if not self.initialize():
                    self.logger.error("Dispenser auto-initialization failed")
                    return False

            # Check daily limit
            if self.treats_dispensed_today >= self.daily_limit:
                self.logger.warning(f"Daily treat limit reached ({self.daily_limit})")
                return False

            # Check minimum interval
            now = time.time()
            if now - self.last_dispense_time < self.min_dispense_interval:
                self.logger.warning("Dispense too soon after last dispense")
                return False

            # Check dog-specific cooldown - TESTING: disabled for troubleshooting
            # if dog_id and dog_id in self.dog_cooldowns:
            #     dog_cooldown = 20.0  # seconds between treats for same dog
            #     if now - self.dog_cooldowns[dog_id] < dog_cooldown:
            #         self.logger.warning(f"Dog {dog_id} on cooldown")
            #         return False

            try:
                # Dispense treat using proven method
                success = self._rotate_carousel()

                if success:
                    # Update counters
                    self.treats_dispensed_today += 1
                    self.last_dispense_time = now

                    if dog_id:
                        self.dog_treat_counts[dog_id] = self.dog_treat_counts.get(dog_id, 0) + 1
                        self.dog_cooldowns[dog_id] = now

                    # Log to store
                    self.store.log_reward(
                        dog_id=dog_id,
                        behavior=behavior,
                        confidence=confidence,
                        success=True,
                        treats_dispensed=1,
                        mission_name=self.state.mission.name
                    )

                    # Publish event
                    publish_reward_event('treat_dispensed', {
                        'dog_id': dog_id,
                        'reason': reason,
                        'behavior': behavior,
                        'confidence': confidence,
                        'treats_dispensed': 1,
                        'total_today': self.treats_dispensed_today,
                        'timestamp': now
                    }, 'dispenser_service')

                    self.logger.info(f"Treat dispensed for {dog_id or 'unknown'} ({reason})")
                    return True

                else:
                    self.logger.error("Carousel rotation failed")
                    return False

            except Exception as e:
                self.logger.error(f"Dispense error: {e}")
                return False

    def _rotate_carousel(self) -> bool:
        """Rotate carousel to dispense one treat"""
        try:
            # Use the proven method from test scripts - using 'slow' for 1000us pulse
            success = self.servo.rotate_winch('slow', self.dispense_duration)
            return success

        except Exception as e:
            self.logger.error(f"Carousel rotation error: {e}")
            return False

    def dispense_multiple(self, count: int, dog_id: Optional[str] = None,
                         reason: str = "manual") -> int:
        """
        Dispense multiple treats with pauses

        Args:
            count: Number of treats to dispense
            dog_id: Dog receiving treats
            reason: Reason for dispensing

        Returns:
            int: Number of treats successfully dispensed
        """
        dispensed = 0
        pause_between = 0.5  # seconds

        for i in range(count):
            if self.dispense_treat(dog_id, reason, f"multi_{i+1}"):
                dispensed += 1
                if i < count - 1:  # Don't pause after last treat
                    time.sleep(pause_between)
            else:
                break  # Stop if dispense fails

        self.logger.info(f"Dispensed {dispensed}/{count} treats")
        return dispensed

    def can_dispense_for_dog(self, dog_id: str) -> bool:
        """Check if we can dispense a treat for a specific dog"""
        if not self.servo_initialized:
            return False

        if self.treats_dispensed_today >= self.daily_limit:
            return False

        # Check dog-specific cooldown - TESTING: disabled for troubleshooting
        now = time.time()
        # if dog_id in self.dog_cooldowns:
        #     dog_cooldown = 20.0  # seconds
        #     if now - self.dog_cooldowns[dog_id] < dog_cooldown:
        #         return False

        return True

    def get_dog_treat_count(self, dog_id: str) -> int:
        """Get number of treats dispensed for a specific dog today"""
        return self.dog_treat_counts.get(dog_id, 0)

    def reset_daily_counters(self) -> None:
        """Reset daily treat counters (called at midnight)"""
        self.treats_dispensed_today = 0
        self.dog_treat_counts.clear()
        self.logger.info("Daily treat counters reset")

        publish_reward_event('daily_reset', {
            'timestamp': time.time()
        }, 'dispenser_service')

    def set_daily_limit(self, limit: int) -> None:
        """Set daily treat limit"""
        self.daily_limit = max(1, min(100, limit))  # Reasonable bounds
        self.logger.info(f"Daily limit set to {self.daily_limit}")

    def test_dispense(self) -> bool:
        """Test dispense (for calibration/testing)"""
        return self.dispense_treat(None, "test", "test")

    def get_status(self) -> Dict[str, Any]:
        """Get dispenser service status"""
        return {
            'initialized': self.servo_initialized,
            'treats_dispensed_today': self.treats_dispensed_today,
            'daily_limit': self.daily_limit,
            'last_dispense_time': self.last_dispense_time,
            'time_since_last_dispense': time.time() - self.last_dispense_time if self.last_dispense_time > 0 else 999,
            'dog_treat_counts': self.dog_treat_counts.copy(),
            'active_cooldowns': {
                dog_id: max(0, 20.0 - (time.time() - last_time))
                for dog_id, last_time in self.dog_cooldowns.items()
                if time.time() - last_time < 20.0
            }
        }

    def cleanup(self) -> None:
        """Clean shutdown"""
        # No ongoing operations to stop for dispenser
        self.logger.info("Dispenser service cleaned up")


# Global dispenser service instance
_dispenser_instance = None
_dispenser_lock = threading.Lock()

def get_dispenser_service() -> DispenserService:
    """Get the global dispenser service instance (singleton)"""
    global _dispenser_instance
    if _dispenser_instance is None:
        with _dispenser_lock:
            if _dispenser_instance is None:
                _dispenser_instance = DispenserService()
                # Auto-initialize the dispenser hardware
                if not _dispenser_instance.servo_initialized:
                    _dispenser_instance.initialize()
    return _dispenser_instance