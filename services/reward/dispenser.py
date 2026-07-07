#!/usr/bin/env python3
"""
Treat dispensing service — NEMA 17 stepper motor + TMC2209 driver

Replaces the old servo-based carousel. Uses UART to configure TMC2209
(current limits, microstepping, StealthChop) then GPIO for STEP/DIR/EN.

Anti-jam: every dispense checks for stall via step count timing.
If resistance is detected, reverses and retries (mimics manual tap-to-clear).
"""

import threading
from collections import deque
import serial
import struct
import time
import logging
from typing import Dict, Any, Optional

import lgpio

from core.bus import get_bus, publish_reward_event
from core.state import get_state
from core.store import get_store
from core.data import get_wimz_store
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
        # TMC2209 chopper mode: "stealthchop" (default, quiet, ~50-70% torque)
        # or "spreadcycle" (audible whine, full rated torque). Per-robot opt-in.
        self.chopper_mode = (getattr(robot_config.dispenser, 'chopper_mode', None)
                             or 'stealthchop').lower()

        # Direction constants (fixed — polarity handled by TMC2209 shaft bit)
        self.CW = 1   # Clockwise = dispense direction
        self.CCW = 0   # Counter-clockwise = reverse/unjam
        self.shaft_invert = robot_config.dispenser.shaft_invert

        # IR through-beam dispense confirmation (spec dispensed_confirmed).
        # Config-gated: beam_enabled=false until the sensor is physically wired.
        self.beam_enabled = robot_config.dispenser.beam_enabled
        self.beam_pin = robot_config.dispenser.beam_pin
        self.beam_timeout_s = robot_config.dispenser.beam_timeout_s
        self.beam_active_low = robot_config.dispenser.beam_active_low
        self._beam_callback = None
        self._last_beam_break_mono = 0.0  # monotonic ts of most recent beam break
        # DISPENSE-VERIFY: every break timestamped for windowed counting.
        # ~20ms debounce merges flickers from a single tumbling treat; two
        # stacked treats arrive further apart and count as 2.
        self._beam_breaks = deque(maxlen=256)  # monotonic timestamps
        self.BEAM_DEBOUNCE_S = 0.020
        self.COUNT_WINDOW_S = 6.0     # count window after rotation completes
        self.DISPENSE_CAP = 2         # max treats credited per normal dispense
        self.ANTIJAM_CAP = 6          # max treats credited per anti-jam run
        # Sticky failure state — set when the full retry ladder yields zero
        # treats. Cleared by: service restart, treat-counter update/reset, or
        # any later beam-confirmed dispense. Surfaced via get_status().
        self.empty_or_jammed = False
        self.last_unjam_count = 0  # treats beam-counted by last manual unjam

        # Last spec dispense_log id + confirmation — read by callers linking
        # training_attempt (spec §6: reward_dispensed reads dispensed_confirmed
        # when the beam is present, motor completion otherwise)
        self.last_wimz_dispense_id = None
        self.last_dispense_confirmed = 0

        # Dispensing state
        self.treats_dispensed_today = 0
        self.last_dispense_time = 0.0       # wall-clock — for status/display
        self.last_dispense_mono = 0.0       # monotonic — for elapsed gates (battery idle check)
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

        # GCONF: I_scale_analog=1 (bit 0), mstep_reg_select=1 (bit 7), shaft bit (bit 4) for polarity
        # bit 2 (en_spreadcycle): 0 = stealthChop (quiet, ~50-70% torque)
        #                        1 = spreadCycle (audible, full rated torque)
        shaft_bit = (1 << 4) if self.shaft_invert else 0
        spreadcycle_bit = (1 << 2) if self.chopper_mode == 'spreadcycle' else 0
        self._tmc_write(0x00, 0x00000081 | shaft_bit | spreadcycle_bit)

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
            f"{self.microstepping}x microstep, vsense=0, SGTHRS={self.sgthrs}, "
            f"chopper={self.chopper_mode}"
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

            # Through-beam sensor: persistent edge alert; the callback just
            # timestamps the latest beam break (dispense logic compares it
            # against the fire time).
            if self.beam_enabled:
                try:
                    edge = lgpio.FALLING_EDGE if self.beam_active_low else lgpio.RISING_EDGE
                    lgpio.gpio_claim_alert(self.gpio_chip, self.beam_pin, edge,
                                           lgpio.SET_PULL_UP)
                    self._beam_callback = lgpio.callback(
                        self.gpio_chip, self.beam_pin, edge, self._on_beam_edge)
                    self.logger.info(f"Through-beam sensor armed on GPIO{self.beam_pin} "
                                     f"(active_{'low' if self.beam_active_low else 'high'})")
                except Exception as e:
                    self.logger.error(f"Through-beam init failed (continuing unconfirmed): {e}")
                    self.beam_enabled = False

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

    # =========================================================================
    # THROUGH-BEAM CONFIRMATION + TMC DIAGNOSTICS (BEAM work item)
    # =========================================================================

    def _on_beam_edge(self, chip, gpio, level, timestamp):
        """lgpio alert callback — timestamp the beam break (treat ejected)."""
        now = time.monotonic()
        self._last_beam_break_mono = now
        self._beam_breaks.append(now)

    def _count_breaks_since(self, start_mono: float) -> int:
        """Debounced count of beam breaks after start_mono (20ms merge)."""
        count = 0
        last_counted = None
        for ts in list(self._beam_breaks):
            if ts <= start_mono:
                continue
            if last_counted is None or (ts - last_counted) > self.BEAM_DEBOUNCE_S:
                count += 1
                last_counted = ts
        return count

    def _count_beam_window(self, fire_mono: float, cap: int) -> tuple:
        """Count treats crossing the beam after fire_mono (DISPENSE-VERIFY).

        The window runs until COUNT_WINDOW_S after the rotation has finished
        (i.e. after this call starts), finalizing early once `cap` treats are
        counted — nothing more to learn. Breaks <20ms apart merge into one
        treat (single tumble); stacked-but-separate treats count individually.

        Returns (counted 0..cap, first_latency_ms or None, overage 0/1).
        With no beam sensor configured: (0, None, 0) — never fake a confirm.
        """
        if not self.beam_enabled:
            return 0, None, 0
        deadline = time.monotonic() + self.COUNT_WINDOW_S
        while time.monotonic() < deadline:
            if self._count_breaks_since(fire_mono) >= cap:
                break
            time.sleep(0.02)
        # small settle so a break racing the deadline still lands
        time.sleep(0.05)
        raw = self._count_breaks_since(fire_mono)
        first = next((ts for ts in list(self._beam_breaks) if ts > fire_mono), None)
        latency_ms = int((first - fire_mono) * 1000) if first is not None else None
        overage = 1 if raw > cap else 0
        if overage:
            self.logger.warning(
                f"Beam counted {raw} breaks (cap {cap}) — treat counter likely "
                f"stale or treats crumbling; crediting {cap}")
        return min(raw, cap), latency_ms, overage

    def _check_tmc_diagnostics(self, context: str) -> None:
        """StallGuard/driver flags — DIAGNOSTIC ONLY, never gates success.

        StallGuard is proven unreliable at our ~1.9-3 RPM dispense speed
        (sgthrs=0 fleet-wide); the through-beam is the confirmation. Here we
        only read DRV_STATUS after a rotation: any stall/thermal/short flag
        becomes a spec `error` event and triggers the anti-jam wiggle.
        """
        if not self.uart:
            return
        try:
            drv_status = self._tmc_read(0x6F)
            if drv_status is None:
                return
            flags = {
                'stall': bool(drv_status & (1 << 31)),
                'overtemp_warning': bool(drv_status & (1 << 26)),
                'overtemp_shutdown': bool(drv_status & (1 << 25)),
                'short_a': bool(drv_status & (1 << 27)),
                'short_b': bool(drv_status & (1 << 28)),
            }
            raised = [k for k, v in flags.items() if v]
            if not raised:
                return
            self.logger.warning(f"TMC2209 diagnostic flags after {context}: {raised}")
            try:
                wimz = get_wimz_store()
                wimz.log_event(
                    None, 'error',
                    {'code': 'dispenser_stall',
                     'detail': f"DRV_STATUS flags {raised} after {context}"},
                    label_source='auto_rule')
            except Exception:
                pass
            if 'stall' in raised:
                # Lock is already held by the dispense path — use the inner wiggle
                self.logger.info("Stall flag set — running anti-jam wiggle")
                self._anti_jam_wiggle_inner()
        except Exception as e:
            self.logger.debug(f"TMC diagnostic read failed: {e}")

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

    # reason -> spec dispense_log.trigger vocabulary
    _TRIGGER_MAP = {
        'coaching_reward': 'attempt',
        'silent_guardian_reward': 'attempt',
        'mission_reward': 'attempt',
        'manual': 'manual_pilot',
        'schedule': 'schedule',
    }

    def dispense_treat(self, dog_id: Optional[str] = None, reason: str = "manual",
                       behavior: str = "", confidence: float = 0.0,
                       wimz_session_id: Optional[str] = None) -> bool:
        """
        Dispense a single treat.

        Args:
            wimz_session_id: spec session for the dispense_log row (modes pass
                theirs; falls back to the ambient session). The resulting
                dispense_id is exposed as self.last_wimz_dispense_id.

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
                self.last_wimz_dispense_id = None
                fire_mono = time.monotonic()
                counted = 0
                confirm_latency_ms = None
                overage = 0
                attempts = 0
                success = False

                # DISPENSE-VERIFY ladder: rotate and count treats through the
                # beam (cap 2 — stacked pairs are real, more is a stale treat
                # counter). Zero treats -> re-dispense, up to 3 rotations.
                for attempt in range(1, 4):
                    attempts = attempt
                    attempt_mono = time.monotonic()
                    success = self._rotate_carousel()
                    if not success:
                        break  # motor-level failure — not a jam, stop the ladder
                    counted, latency, overage = self._count_beam_window(
                        attempt_mono, self.DISPENSE_CAP)
                    if latency is not None and confirm_latency_ms is None:
                        # latency from the ORIGINAL fire, for jam trending
                        confirm_latency_ms = latency + int((attempt_mono - fire_mono) * 1000)
                    if not self.beam_enabled or counted >= 1:
                        break
                    if attempt < 3:
                        self.logger.warning(
                            f"Dispense attempt {attempt}: motor OK but no treat "
                            f"crossed beam — auto re-dispensing")

                # Rung 4: three empty rotations -> full anti-jam, still counting
                if self.beam_enabled and success and counted == 0:
                    self.logger.warning(
                        "3 dispenses, zero treats counted — running anti-jam procedure")
                    attempts = 4
                    aj_mono = time.monotonic()
                    self._anti_jam_wiggle_inner()
                    counted, latency, overage = self._count_beam_window(
                        aj_mono, self.ANTIJAM_CAP)
                    if latency is not None and confirm_latency_ms is None:
                        confirm_latency_ms = latency + int((aj_mono - fire_mono) * 1000)

                # StallGuard/driver flags — diagnostic only, never gates success
                self._check_tmc_diagnostics('dispense rotation')

                confirmed = 1 if counted >= 1 else 0
                # Without a beam sensor, motor completion remains the only truth
                credit = counted if self.beam_enabled else (1 if success else 0)
                self.last_dispense_confirmed = confirmed if self.beam_enabled else int(success)

                # Spec dispense_log row (+ paired treat_dispensed event)
                try:
                    wimz = get_wimz_store()
                    wimz_dog = wimz.get_or_create_dog(legacy_id=dog_id) if dog_id else None
                    self.last_wimz_dispense_id = wimz.log_dispense(
                        wimz_session_id,
                        trigger=self._TRIGGER_MAP.get(reason, 'manual_pilot'),
                        dog_id=wimz_dog,
                        dispensed_confirmed=self.last_dispense_confirmed,
                        confirm_latency_ms=confirm_latency_ms,
                        dispensed_count=counted,
                        attempts=max(attempts, 1),
                        overage=overage)
                except Exception as e:
                    self.logger.warning(f"wimz dispense log failed: {e}")

                if self.beam_enabled and success and credit == 0:
                    # Full ladder exhausted, zero treats — out of treats or jam
                    # beyond anti-jam's reach. Sticky until refill/restart.
                    self.empty_or_jammed = True
                    self.logger.error(
                        "DISPENSER EMPTY/JAMMED: full retry ladder yielded no "
                        "treats — carousel empty or needs repair (sticky; "
                        "clears on treat-counter update or restart)")
                    try:
                        get_wimz_store().log_event(
                            None, 'error',
                            {'code': 'dispenser_empty', 'attempts': attempts})
                    except Exception:
                        pass
                    publish_reward_event('dispenser_empty', {
                        'dog_id': dog_id,
                        'reason': reason,
                        'attempts': attempts,
                        'treats_loaded': self.treats_loaded,
                        'timestamp': now
                    }, 'dispenser_service')
                    return False

                if success and credit > 0:
                    self.treats_dispensed_today += credit
                    self.last_dispense_time = now
                    self.last_dispense_mono = time.monotonic()
                    for _ in range(credit):
                        self._decrement_treat_counter()
                    # A physically-confirmed dispense proves we're not empty
                    if confirmed:
                        self.empty_or_jammed = False

                    if dog_id:
                        self.dog_treat_counts[dog_id] = self.dog_treat_counts.get(dog_id, 0) + credit
                        self.dog_cooldowns[dog_id] = now

                    self.store.log_reward(
                        dog_id=dog_id,
                        behavior=behavior,
                        confidence=confidence,
                        success=True,
                        treats_dispensed=credit,
                        mission_name=self.state.mission.name
                    )

                    publish_reward_event('treat_dispensed', {
                        'dog_id': dog_id,
                        'reason': reason,
                        'behavior': behavior,
                        'confidence': confidence,
                        'treats_dispensed': credit,
                        'confirmed': self.last_dispense_confirmed,
                        'attempts': max(attempts, 1),
                        'total_today': self.treats_dispensed_today,
                        'remaining': self.treats_remaining,
                        'timestamp': now
                    }, 'dispenser_service')

                    remaining = max(0, self.treats_loaded - self.treats_dispensed_session)
                    self.logger.info(
                        f"[TREAT] Dispensed {credit} treat(s) "
                        f"(confirmed={self.last_dispense_confirmed}, "
                        f"attempts={max(attempts, 1)}) #{self.treats_dispensed_today} today | "
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
        """Anti-jam sequence: reverse, pause, forward with extra steps to clear.

        DISPENSE-VERIFY: treats shaken loose are beam-counted (cap 6) into
        self.last_unjam_count, credited to the treat counters, and logged as
        a dispense_log row. Any treat out clears empty_or_jammed.
        """
        acquired = self._dispense_lock.acquire(timeout=self.DISPENSE_TIMEOUT + 2)
        if not acquired:
            self.logger.error("DISPENSER_UNJAM: Lock timeout")
            self._disable_motor()
            return False
        try:
            if not self.initialized:
                self.logger.error("DISPENSER_UNJAM: Not initialized")
                return False
            aj_mono = time.monotonic()
            result = self._anti_jam_wiggle_inner()
            counted, latency_ms, overage = self._count_beam_window(
                aj_mono, self.ANTIJAM_CAP)
            self.last_unjam_count = counted
            if self.beam_enabled:
                self.logger.info(f"DISPENSER_UNJAM: {counted} treat(s) crossed beam")
                try:
                    get_wimz_store().log_dispense(
                        None, trigger='manual_pilot',
                        dispensed_confirmed=1 if counted else 0,
                        confirm_latency_ms=latency_ms,
                        dispensed_count=counted, attempts=1, overage=overage)
                except Exception as e:
                    self.logger.warning(f"wimz unjam log failed: {e}")
                if counted > 0:
                    self.empty_or_jammed = False
                    self.treats_dispensed_today += counted
                    for _ in range(counted):
                        self._decrement_treat_counter()
            return result
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
        # Refill/counter update = human intervention — clear the sticky
        # empty-or-jammed state (DISPENSE-VERIFY reset rule)
        self.empty_or_jammed = False
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
        self.empty_or_jammed = False  # DISPENSE-VERIFY reset rule
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
            'beam_enabled': self.beam_enabled,
            'empty_or_jammed': self.empty_or_jammed,
            'last_dispense_confirmed': self.last_dispense_confirmed,
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
        if self._beam_callback is not None:
            try:
                self._beam_callback.cancel()
            except Exception:
                pass
            self._beam_callback = None
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
