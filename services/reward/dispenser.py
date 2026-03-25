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

import lgpio

from core.bus import get_bus, publish_reward_event
from core.state import get_state
from core.store import get_store
from core.hardware.servo_controller import get_servo_controller
from config.config_loader import get_config
from config.pins import TreatBotPins


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

        # Vibrator motor (GPIO16 via MOSFET)
        self.gpio_chip = None
        self.vibrator_pin = TreatBotPins.VIBRATOR
        self.vibrator_initialized = False

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

        # Treat counter — persisted to SQLite
        self.treats_loaded = 0
        self.treats_dispensed_session = 0
        self._load_treat_counter()

        # Thread safety
        self._dispense_lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize servo controller and vibrator motor"""
        # Servo init
        if not self.servo_initialized:
            try:
                self.servo = get_servo_controller()
                if self.servo.is_initialized():
                    self.servo_initialized = True
                    self.logger.info("Treat dispenser initialized")
                else:
                    self.logger.error("Servo controller initialization failed")
                    return False
            except Exception as e:
                self.logger.error(f"Dispenser initialization error: {e}")
                return False

        # Vibrator init — only once
        if not self.vibrator_initialized and self.gpio_chip is None:
            try:
                self.gpio_chip = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_output(self.gpio_chip, self.vibrator_pin, lgpio.SET_PULL_NONE)
                lgpio.gpio_write(self.gpio_chip, self.vibrator_pin, 0)  # Start OFF
                self.vibrator_initialized = True
                self.logger.info(f"Vibrator motor initialized on GPIO{self.vibrator_pin}")
            except Exception as e:
                self.logger.warning(f"Vibrator motor init failed (dispensing will work without it): {e}")
                self.vibrator_initialized = False

        return True

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
                    self._decrement_treat_counter()

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

    def _vibrator_on(self):
        """Turn vibrator motor ON"""
        if self.vibrator_initialized:
            try:
                lgpio.gpio_write(self.gpio_chip, self.vibrator_pin, 1)
                self.logger.debug("VIBRATOR: GPIO16 HIGH")
            except Exception as e:
                self.logger.warning(f"Vibrator ON failed: {e}")

    def _vibrator_off(self):
        """Turn vibrator motor OFF"""
        if self.vibrator_initialized:
            try:
                lgpio.gpio_write(self.gpio_chip, self.vibrator_pin, 0)
                self.logger.debug("VIBRATOR: GPIO16 LOW")
            except Exception as e:
                self.logger.warning(f"Vibrator OFF failed: {e}")

    def _rotate_carousel(self) -> bool:
        """Rotate carousel to dispense one treat with vibration assist"""
        try:
            # 1. Pre-shake to loosen stuck treats
            self._vibrator_on()
            time.sleep(0.3)
            self._vibrator_off()
            time.sleep(0.1)  # Let power settle before servo

            # 2. Rotate carousel (vibrator OFF during rotation)
            success = self.servo.rotate_winch('slow', self.dispense_duration)
            time.sleep(0.1)  # Let servo fully stop

            # 3. Post-vibrate to help treat fall
            self._vibrator_on()
            time.sleep(0.5)
            self._vibrator_off()

            return success

        except Exception as e:
            self._vibrator_off()  # Safety: ensure vibrator stops on error
            self.logger.error(f"Carousel rotation error: {e}")
            return False

    def anti_jam_wiggle(self) -> bool:
        """Anti-jam wiggle sequence: vibrate + carousel wiggle to clear jams"""
        with self._dispense_lock:
            if not self.servo_initialized:
                self.logger.error("DISPENSER_UNJAM: Servo not initialized")
                return False
            return self._anti_jam_wiggle_inner()

    def _anti_jam_wiggle_inner(self) -> bool:
        """Inner anti-jam logic (must be called with _dispense_lock held)"""
        self.logger.info("DISPENSER_UNJAM: Starting anti-jam wiggle sequence")
        try:
            for cycle in range(3):
                # Vibrate
                self._vibrator_on()
                time.sleep(0.4)

                # Forward rotation
                self.servo.rotate_winch('slow', 0.3)
                time.sleep(0.2)

                # Brief reverse (stop vibrator during reverse)
                self._vibrator_off()
                time.sleep(0.1)

                # Vibrate again
                self._vibrator_on()
                time.sleep(0.3)

                self.logger.info(f"DISPENSER_UNJAM: Wiggle cycle {cycle + 1}/3 complete")

            # Final forward rotation to clear
            self.servo.rotate_winch('slow', self.dispense_duration)
            time.sleep(0.3)
            self._vibrator_off()

            self.logger.info("DISPENSER_UNJAM: Anti-jam sequence complete")
            publish_reward_event('unjam_complete', {
                'timestamp': time.time()
            }, 'dispenser_service')
            return True

        except Exception as e:
            self._vibrator_off()  # Safety
            self.logger.error(f"Anti-jam wiggle error: {e}")
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

    def _load_treat_counter(self):
        """Load treat counter from SQLite persistence"""
        stored = self.store.get_setting('treats_loaded')
        if stored is not None:
            self.treats_loaded = int(stored)
        else:
            self.treats_loaded = 0
        self.logger.info(f"Treat counter loaded: {self.treats_loaded} treats loaded")

    def _save_treat_counter(self):
        """Persist treat counter to SQLite"""
        self.store.set_setting('treats_loaded', str(self.treats_loaded))

    def _decrement_treat_counter(self):
        """Decrement treat counter after successful dispense, warn if low"""
        self.treats_loaded -= 1
        self.treats_dispensed_session += 1
        self._save_treat_counter()

        remaining = self.treats_remaining
        self.logger.info(f"TREAT_COUNTER: {remaining} remaining")

        if remaining <= 0:
            publish_reward_event('treats_empty', {
                'treats_loaded': self.treats_loaded,
                'treats_dispensed': self.treats_dispensed_session,
                'treats_remaining': remaining,
                'timestamp': time.time()
            }, 'dispenser_service')
        elif remaining < 5:
            publish_reward_event('treats_low', {
                'treats_remaining': remaining,
                'timestamp': time.time()
            }, 'dispenser_service')

    @property
    def treats_remaining(self) -> int:
        """Get treats remaining — can go negative (never blocks dispensing)"""
        return self.treats_loaded

    def set_treat_count(self, count: int) -> None:
        """Set the number of treats loaded (called when user refills)"""
        self.treats_loaded = max(0, count)
        self.treats_dispensed_session = 0
        self._save_treat_counter()
        self.logger.info(f"TREAT_COUNTER: set to {self.treats_loaded}")
        publish_reward_event('treats_loaded', {
            'treats_loaded': self.treats_loaded,
            'treats_dispensed': self.treats_dispensed_session,
            'treats_remaining': self.treats_remaining,
            'timestamp': time.time()
        }, 'dispenser_service')

    def reset_treat_counter(self) -> None:
        """Reset treat counter to 0"""
        self.treats_loaded = 0
        self.treats_dispensed_session = 0
        self._save_treat_counter()
        self.logger.info("TREAT_COUNTER: Reset to 0")

    def test_dispense(self) -> bool:
        """Test dispense (for calibration/testing)"""
        return self.dispense_treat(None, "test", "test")

    def get_status(self) -> Dict[str, Any]:
        """Get dispenser service status"""
        return {
            'initialized': self.servo_initialized,
            'vibrator_initialized': self.vibrator_initialized,
            'treats_dispensed_today': self.treats_dispensed_today,
            'daily_limit': self.daily_limit,
            'last_dispense_time': self.last_dispense_time,
            'time_since_last_dispense': time.time() - self.last_dispense_time if self.last_dispense_time > 0 else 999,
            'treats_loaded': self.treats_loaded,
            'treats_dispensed': self.treats_dispensed_session,
            'treats_remaining': self.treats_remaining,
            'treats_low': self.treats_remaining < 5,
            'dog_treat_counts': self.dog_treat_counts.copy(),
            'active_cooldowns': {
                dog_id: max(0, 20.0 - (time.time() - last_time))
                for dog_id, last_time in self.dog_cooldowns.items()
                if time.time() - last_time < 20.0
            }
        }

    def cleanup(self) -> None:
        """Clean shutdown — ensure vibrator is OFF"""
        self._vibrator_off()
        if self.gpio_chip is not None:
            try:
                lgpio.gpiochip_close(self.gpio_chip)
            except Exception:
                pass
            self.gpio_chip = None
        self.vibrator_initialized = False
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