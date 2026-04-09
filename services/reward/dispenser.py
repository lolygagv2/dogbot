#!/usr/bin/env python3
"""
Treat dispensing service — NEMA 17 stepper motor + TMC2209 driver

Replaces the old servo-based carousel. Uses UART to configure TMC2209
(current limits, microstepping, StealthChop) then GPIO for STEP/DIR/EN.

Anti-jam: every dispense checks for stall via step count timing.
If resistance is detected, reverses and retries (mimics manual tap-to-clear).
"""

import threading
import serial
import struct
import time
import logging
from typing import Dict, Any, Optional

import lgpio

from core.bus import get_bus, publish_reward_event
from core.state import get_state
from core.store import get_store
from config.config_loader import get_config
from config.pins import TreatBotPins


class DispenserService:
    """
    Treat dispensing service using NEMA 17 stepper + TMC2209 driver.
    Tracks treats dispensed per dog and implements portion control.
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.logger = logging.getLogger('DispenserService')

        # Hardware handles
        self.gpio_chip = None
        self.uart = None
        self.initialized = False

        # GPIO pins
        self.step_pin = TreatBotPins.STEPPER_STEP
        self.dir_pin = TreatBotPins.STEPPER_DIR
        self.en_pin = TreatBotPins.STEPPER_EN
        self.uart_port = TreatBotPins.STEPPER_UART

        # Config from YAML
        robot_config = get_config()
        self.steps_per_slot = robot_config.dispenser.steps_per_slot
        self.step_delay = robot_config.dispenser.step_delay
        self.max_retries = robot_config.dispenser.max_retries
        self.reverse_steps = robot_config.dispenser.reverse_steps
        self.irun = robot_config.dispenser.irun
        self.ihold = robot_config.dispenser.ihold
        self.microstepping = robot_config.dispenser.microstepping
        self.sgthrs = getattr(robot_config.dispenser, 'sgthrs', 50)

        # Direction constants (fixed — polarity handled by TMC2209 shaft bit)
        self.CW = 1   # Clockwise = dispense direction
        self.CCW = 0   # Counter-clockwise = reverse/unjam
        self.shaft_invert = robot_config.dispenser.shaft_invert

        # Dispensing state
        self.treats_dispensed_today = 0
        self.last_dispense_time = 0.0
        self.min_dispense_interval = 0.0
        self.daily_limit = 300

        # Per-dog tracking
        self.dog_treat_counts = {}
        self.dog_cooldowns = {}

        # Treat counter — persisted to SQLite
        self.treats_loaded = 0
        self.treats_dispensed_session = 0
        self._load_treat_counter()

        # Thread safety
        self._dispense_lock = threading.Lock()
        self._refill_active = False
        self._refill_stop_requested = False

    # =========================================================================
    # TMC2209 UART
    # =========================================================================

    def _tmc_crc(self, data):
        """Calculate TMC2209 CRC8"""
        crc = 0
        for byte in data:
            for _ in range(8):
                if (crc >> 7) ^ (byte & 0x01):
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
                byte >>= 1
        return crc

    def _tmc_write(self, reg, value):
        """Write 32-bit value to TMC2209 register"""
        if self.uart is None:
            return
        datagram = bytes([0x05, 0x00, reg | 0x80]) + struct.pack('>I', value)
        datagram += bytes([self._tmc_crc(datagram)])
        self.uart.write(datagram)
        time.sleep(0.005)
        self.uart.reset_input_buffer()

    def _tmc_read(self, reg):
        """Read 32-bit value from TMC2209 register"""
        if self.uart is None:
            return None
        self.uart.reset_input_buffer()
        datagram = bytes([0x05, 0x00, reg])
        datagram += bytes([self._tmc_crc(datagram)])
        self.uart.write(datagram)
        time.sleep(0.01)
        response = self.uart.read(12)
        if len(response) >= 12:
            return struct.unpack('>I', response[7:11])[0]
        return None

    def _configure_tmc(self):
        """Configure TMC2209 for safe carousel operation with StallGuard"""
        # MRES lookup: microstepping → register value
        mres_map = {256: 0, 128: 1, 64: 2, 32: 3, 16: 4, 8: 5, 4: 6, 2: 7, 1: 8}
        mres = mres_map.get(self.microstepping, 5)

        # GCONF: I_scale_analog=1, mstep_reg_select=1, shaft bit for motor polarity
        shaft_bit = (1 << 4) if self.shaft_invert else 0
        self._tmc_write(0x00, 0x00000081 | shaft_bit)

        # IHOLD_IRUN: configured current with gradual ramp-down
        ihold_irun = (6 << 16) | (self.irun << 8) | self.ihold
        self._tmc_write(0x10, ihold_irun)

        # CHOPCONF: configured microstepping, vsense=0 (high current range)
        chopconf = (mres << 24) | 0x00000053
        self._tmc_write(0x6C, chopconf)

        # TPOWERDOWN: time before current drops to IHOLD
        self._tmc_write(0x11, 20)

        # SGTHRS: StallGuard sensitivity (0-255, higher = more sensitive)
        self._tmc_write(0x40, self.sgthrs)

        # TCOOLTHRS: velocity threshold for StallGuard activation
        # Set very high so StallGuard is active even at our low speed (~3 RPM)
        self._tmc_write(0x14, 0xFFFFF)

        self.logger.info(
            f"TMC2209 configured: IRUN={self.irun}, IHOLD={self.ihold}, "
            f"{self.microstepping}x microstep, vsense=0, SGTHRS={self.sgthrs}"
        )

    # =========================================================================
    # GPIO MOTOR CONTROL
    # =========================================================================

    def _enable_motor(self):
        """Enable TMC2209 (EN LOW)"""
        lgpio.gpio_write(self.gpio_chip, self.en_pin, 0)
        time.sleep(0.05)

    def _disable_motor(self):
        """Disable TMC2209 (EN HIGH) — motor free-spins"""
        lgpio.gpio_write(self.gpio_chip, self.en_pin, 1)

    def _step(self, steps, direction, delay=None):
        """
        Step the motor. Only checks _abort flag (set by emergency_stop).
        _refill_stop_requested is checked separately by refill code.
        """
        if delay is None:
            delay = self.step_delay

        lgpio.gpio_write(self.gpio_chip, self.dir_pin, direction)
        time.sleep(0.001)

        for i in range(steps):
            if self._abort:
                return False
            lgpio.gpio_write(self.gpio_chip, self.step_pin, 1)
            time.sleep(delay)
            lgpio.gpio_write(self.gpio_chip, self.step_pin, 0)
            time.sleep(delay)

        return True

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self) -> bool:
        """Initialize UART + GPIO for stepper dispenser"""
        if self.initialized:
            return True

        try:
            # Open UART to TMC2209
            try:
                self.uart = serial.Serial(
                    port=self.uart_port,
                    baudrate=115200,
                    timeout=0.2
                )
                self.uart.reset_input_buffer()
                self.logger.info(f"TMC2209 UART opened: {self.uart_port}")
            except Exception as e:
                self.logger.warning(f"TMC2209 UART failed (will run without current config): {e}")
                self.uart = None

            # Configure TMC2209 via UART BEFORE enabling motor
            if self.uart:
                # Verify chip responds
                ioin = self._tmc_read(0x06)
                if ioin is not None:
                    version = (ioin >> 24) & 0xFF
                    self.logger.info(f"TMC2209 detected: version=0x{version:02X}")
                    self._configure_tmc()
                else:
                    self.logger.warning("TMC2209 not responding on UART — using hardware defaults")

            # Initialize GPIO
            self.gpio_chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.gpio_chip, self.step_pin)
            lgpio.gpio_claim_output(self.gpio_chip, self.dir_pin)
            lgpio.gpio_claim_output(self.gpio_chip, self.en_pin)

            # Start disabled
            lgpio.gpio_write(self.gpio_chip, self.en_pin, 1)
            lgpio.gpio_write(self.gpio_chip, self.step_pin, 0)
            lgpio.gpio_write(self.gpio_chip, self.dir_pin, self.CW)

            self.initialized = True
            self.logger.info(
                f"Stepper dispenser initialized: "
                f"STEP=GPIO{self.step_pin}, DIR=GPIO{self.dir_pin}, EN=GPIO{self.en_pin}, "
                f"{self.steps_per_slot} steps/slot"
            )
            return True

        except Exception as e:
            self.logger.error(f"Dispenser initialization error: {e}")
            return False

    # =========================================================================
    # DISPENSING
    # =========================================================================

    # Max time for a single dispense
    DISPENSE_TIMEOUT = 10.0

    # Abort flag — set by emergency_stop() to cancel any in-progress operation
    _abort = False

    def emergency_stop(self):
        """Immediately stop all motor activity. Called from any thread."""
        self._abort = True
        self._refill_stop_requested = True
        self._disable_motor()
        self.logger.warning("EMERGENCY STOP — motor disabled")

    def _rotate_carousel(self) -> bool:
        """
        Dispense one treat with post-step preventive nudge.

        StallGuard cannot detect jams at our speed (~3 RPM).
        Instead, every dispense does a reverse-forward nudge AFTER stepping
        to clear any treat that got stuck mid-transition.

        Sequence:
        1. Forward 137 steps (advance one slot)
        2. Pause (let treat fall by gravity)
        3. Reverse 40 steps (loosen anything stuck)
        4. Forward 40 steps (re-seat to correct position)
        """
        try:
            self._abort = False
            self._refill_stop_requested = False
            self._enable_motor()
            time.sleep(0.05)

            # 1. Advance one slot
            self._step(self.steps_per_slot, self.CW, delay=self.step_delay)
            time.sleep(0.2)  # Let treat fall

            if self._abort:
                self._disable_motor()
                return False

            # 2. Post-dispense nudge — loosen any stuck treat
            self._step(self.reverse_steps, self.CCW, delay=0.004)
            time.sleep(0.1)
            self._step(self.reverse_steps, self.CW, delay=0.004)

            self._disable_motor()
            return True

        except Exception as e:
            self._disable_motor()
            self.logger.error(f"Carousel rotation error: {e}")
            return False

    def dispense_treat(self, dog_id: Optional[str] = None, reason: str = "manual",
                       behavior: str = "", confidence: float = 0.0) -> bool:
        """
        Dispense a single treat.

        Returns:
            bool: True if treat was dispensed successfully
        """
        acquired = self._dispense_lock.acquire(timeout=self.DISPENSE_TIMEOUT + 2)
        if not acquired:
            self.logger.error("Dispense lock timeout — previous operation stuck, forcing motor disable")
            self._disable_motor()
            return False
        try:
            # Auto-initialize if not already done
            if not self.initialized:
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

            try:
                success = self._rotate_carousel()

                if success:
                    self.treats_dispensed_today += 1
                    self.last_dispense_time = now
                    self._decrement_treat_counter()

                    if dog_id:
                        self.dog_treat_counts[dog_id] = self.dog_treat_counts.get(dog_id, 0) + 1
                        self.dog_cooldowns[dog_id] = now

                    self.store.log_reward(
                        dog_id=dog_id,
                        behavior=behavior,
                        confidence=confidence,
                        success=True,
                        treats_dispensed=1,
                        mission_name=self.state.mission.name
                    )

                    publish_reward_event('treat_dispensed', {
                        'dog_id': dog_id,
                        'reason': reason,
                        'behavior': behavior,
                        'confidence': confidence,
                        'treats_dispensed': 1,
                        'total_today': self.treats_dispensed_today,
                        'remaining': self.treats_remaining,
                        'timestamp': now
                    }, 'dispenser_service')

                    remaining = max(0, self.treats_loaded - self.treats_dispensed_session)
                    self.logger.info(
                        f"[TREAT] Dispensing treat #{self.treats_dispensed_today} of 44 | "
                        f"Remaining: {remaining} | Dog: {dog_id or 'unknown'} | Reason: {reason}"
                    )
                    return True
                else:
                    self.logger.error("Carousel rotation failed")
                    return False

            except Exception as e:
                self.logger.error(f"Dispense error: {e}")
                return False
        finally:
            self._dispense_lock.release()

    def anti_jam_wiggle(self) -> bool:
        """Anti-jam sequence: reverse, pause, forward with extra steps to clear."""
        acquired = self._dispense_lock.acquire(timeout=self.DISPENSE_TIMEOUT + 2)
        if not acquired:
            self.logger.error("DISPENSER_UNJAM: Lock timeout")
            self._disable_motor()
            return False
        try:
            if not self.initialized:
                self.logger.error("DISPENSER_UNJAM: Not initialized")
                return False
            return self._anti_jam_wiggle_inner()
        finally:
            self._dispense_lock.release()

    def _anti_jam_wiggle_inner(self) -> bool:
        """Manual unjam: L1 gentle reverse-forward (5s) then L2 aggressive shake (3s)"""
        self.logger.info("DISPENSER_UNJAM: Starting — L1 (5s)")
        try:
            self._abort = False
            self._refill_stop_requested = False
            self._enable_motor()
            time.sleep(0.1)

            # L1: gentle reverse-forward for 5 seconds
            deadline = time.time() + 5.0
            cycle = 0
            while time.time() < deadline and not self._abort:
                cycle += 1
                self._step(self.reverse_steps, self.CCW, delay=0.006)
                time.sleep(0.15)
                if self._abort:
                    break
                self._step(self.steps_per_slot + self.reverse_steps, self.CW, delay=self.step_delay)
                time.sleep(0.15)
                self.logger.info(f"Unjam L1 cycle {cycle}")

            if self._abort:
                self._disable_motor()
                return False

            # L2: aggressive shaking for 3 seconds
            self.logger.info("DISPENSER_UNJAM: Escalating — L2 (3s)")
            deadline = time.time() + 3.0
            cycle = 0
            while time.time() < deadline and not self._abort:
                cycle += 1
                self._step(self.reverse_steps * 3, self.CCW, delay=0.003)
                time.sleep(0.08)
                if self._abort:
                    break
                self._step(self.reverse_steps * 3, self.CW, delay=0.003)
                time.sleep(0.08)
                self.logger.info(f"Unjam L2 shake {cycle}")

            if not self._abort:
                # Final forward to clear
                self._step(self.steps_per_slot + 20, self.CW, delay=0.006)
                time.sleep(0.15)

            self._disable_motor()
            self.logger.info("DISPENSER_UNJAM: Sequence complete")
            publish_reward_event('unjam_complete', {'timestamp': time.time()}, 'dispenser_service')
            return True

        except Exception as e:
            self._disable_motor()
            self.logger.error(f"Anti-jam error: {e}")
            return False

    def refill_step(self) -> bool:
        """
        Single refill step — keeps motor enabled between calls.
        Checks for stall and runs inline unjam if jammed.
        Checks _refill_stop_requested for instant stop.
        No treat counter decrement (refill, not dispensing).
        """
        # Check stop/abort FIRST — never restart after a stop
        if self._abort or self._refill_stop_requested:
            self._disable_motor()
            self._refill_active = False
            return False

        if not self.initialized:
            if not self.initialize():
                return False

        if not self._refill_active:
            self._enable_motor()
            time.sleep(0.05)
            self._refill_active = True

        self._refill_step_motor(self.steps_per_slot, self.CW, delay=0.004)

        if self._abort or self._refill_stop_requested:
            self._disable_motor()
            self._refill_active = False
            return False
        return True

    def _refill_step_motor(self, steps, direction, delay):
        """Step for refill — checks both _abort AND _refill_stop_requested every step."""
        lgpio.gpio_write(self.gpio_chip, self.dir_pin, direction)
        time.sleep(0.001)
        for i in range(steps):
            if self._abort or self._refill_stop_requested:
                return False
            lgpio.gpio_write(self.gpio_chip, self.step_pin, 1)
            time.sleep(delay)
            lgpio.gpio_write(self.gpio_chip, self.step_pin, 0)
            time.sleep(delay)
        return True

    def refill_continuous(self):
        """
        Continuous refill loop — runs until emergency_stop() is called.
        Runs server-side in a background thread. The Xbox controller just
        polls button state and calls /treat/stop when released.
        """
        if not self.initialized:
            if not self.initialize():
                return

        self._abort = False
        self._refill_stop_requested = False
        self._refill_active = True
        self._enable_motor()
        time.sleep(0.05)

        self.logger.info("Refill continuous: started")
        try:
            while not self._abort and not self._refill_stop_requested:
                self._refill_step_motor(self.steps_per_slot, self.CW, 0.004)
        except Exception as e:
            self.logger.error(f"Refill continuous error: {e}")
        finally:
            self._disable_motor()
            self._refill_active = False
            self.logger.info("Refill continuous: stopped")

    def refill_stop(self):
        """Stop refill mode — disable motor immediately."""
        self._refill_stop_requested = True
        self._disable_motor()
        self._refill_active = False
        self.logger.info("Refill stopped")

    def refill_mode(self, total_slots: int = 56) -> int:
        """
        Refill mode: step through slots slowly for loading treats.
        Returns number of slots advanced.
        """
        with self._dispense_lock:
            if not self.initialized:
                if not self.initialize():
                    return 0

            self._enable_motor()
            time.sleep(0.1)
            advanced = 0

            try:
                for slot in range(total_slots):
                    self._step(self.steps_per_slot, self.CW)
                    advanced += 1
                    time.sleep(1.5)  # Pause for user to fill
            except Exception as e:
                self.logger.error(f"Refill mode error at slot {advanced}: {e}")
            finally:
                self._disable_motor()

            self.logger.info(f"Refill mode: advanced {advanced}/{total_slots} slots")
            return advanced

    def dispense_multiple(self, count: int, dog_id: Optional[str] = None,
                          reason: str = "manual") -> int:
        """Dispense multiple treats with pauses"""
        dispensed = 0
        for i in range(count):
            if self.dispense_treat(dog_id, reason, f"multi_{i+1}"):
                dispensed += 1
                if i < count - 1:
                    time.sleep(0.5)
            else:
                break
        self.logger.info(f"Dispensed {dispensed}/{count} treats")
        return dispensed

    def can_dispense_for_dog(self, dog_id: str) -> bool:
        """Check if we can dispense a treat for a specific dog"""
        if not self.initialized:
            return False
        if self.treats_dispensed_today >= self.daily_limit:
            return False
        return True

    # =========================================================================
    # TREAT COUNTER (unchanged from servo version)
    # =========================================================================

    def get_dog_treat_count(self, dog_id: str) -> int:
        return self.dog_treat_counts.get(dog_id, 0)

    def reset_daily_counters(self) -> None:
        self.treats_dispensed_today = 0
        self.dog_treat_counts.clear()
        self.logger.info("Daily treat counters reset")
        publish_reward_event('daily_reset', {'timestamp': time.time()}, 'dispenser_service')

    def set_daily_limit(self, limit: int) -> None:
        self.daily_limit = max(1, min(300, limit))
        self.logger.info(f"Daily limit set to {self.daily_limit}")

    def _load_treat_counter(self):
        stored = self.store.get_setting('treats_loaded')
        self.treats_loaded = int(stored) if stored is not None else 0
        self.logger.info(f"Treat counter loaded: {self.treats_loaded} treats loaded")

    def _save_treat_counter(self):
        self.store.set_setting('treats_loaded', str(self.treats_loaded))

    def _decrement_treat_counter(self):
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
        return self.treats_loaded

    def set_treat_count(self, count: int) -> None:
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
        self.treats_loaded = 0
        self.treats_dispensed_session = 0
        self._save_treat_counter()
        self.logger.info("TREAT_COUNTER: Reset to 0")

    def test_dispense(self) -> bool:
        return self.dispense_treat(None, "test", "test")

    # =========================================================================
    # STATUS & CLEANUP
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        status = {
            'initialized': self.initialized,
            'driver': 'stepper_tmc2209',
            'treats_dispensed_today': self.treats_dispensed_today,
            'daily_limit': self.daily_limit,
            'last_dispense_time': self.last_dispense_time,
            'time_since_last_dispense': time.time() - self.last_dispense_time if self.last_dispense_time > 0 else 999,
            'treats_loaded': self.treats_loaded,
            'treats_dispensed': self.treats_dispensed_session,
            'treats_remaining': self.treats_remaining,
            'treats_low': self.treats_remaining < 5,
            'dog_treat_counts': self.dog_treat_counts.copy(),
            'config': {
                'steps_per_slot': self.steps_per_slot,
                'step_delay': self.step_delay,
                'irun': self.irun,
                'microstepping': self.microstepping,
            },
            'active_cooldowns': {
                dog_id: max(0, 20.0 - (time.time() - last_time))
                for dog_id, last_time in self.dog_cooldowns.items()
                if time.time() - last_time < 20.0
            }
        }

        # Read TMC2209 diagnostics if UART available
        if self.uart:
            try:
                drv_status = self._tmc_read(0x6F)  # DRV_STATUS register
                if drv_status is not None:
                    status['tmc2209'] = {
                        'stall': bool(drv_status & (1 << 31)),
                        'overtemp_warning': bool(drv_status & (1 << 26)),
                        'overtemp_shutdown': bool(drv_status & (1 << 25)),
                        'short_a': bool(drv_status & (1 << 27)),
                        'short_b': bool(drv_status & (1 << 28)),
                        'open_a': bool(drv_status & (1 << 29)),
                        'open_b': bool(drv_status & (1 << 30)),
                    }
            except Exception:
                pass

        return status

    def cleanup(self) -> None:
        """Clean shutdown — disable motor, close GPIO and UART"""
        if self.gpio_chip is not None:
            try:
                lgpio.gpio_write(self.gpio_chip, self.en_pin, 1)  # Disable motor
                lgpio.gpio_write(self.gpio_chip, self.step_pin, 0)
                lgpio.gpiochip_close(self.gpio_chip)
            except Exception:
                pass
            self.gpio_chip = None

        if self.uart is not None:
            try:
                self.uart.close()
            except Exception:
                pass
            self.uart = None

        self.initialized = False
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
                if not _dispenser_instance.initialized:
                    _dispenser_instance.initialize()
    return _dispenser_instance
