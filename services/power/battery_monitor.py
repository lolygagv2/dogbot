#!/usr/bin/env python3
"""
Battery Monitor Service - ADS1115 ADC on I2C
Monitors 4S LiPo battery voltage via voltage divider on A0
"""

import threading
import time
import logging
from typing import Dict, Any, Optional

# ADS1115 imports
try:
    import board
    from adafruit_ads1x15.ads1115 import ADS1115
    from adafruit_ads1x15.analog_in import AnalogIn
    from core.hardware.i2c_bus import get_i2c_bus
    ADS1115_AVAILABLE = True
except ImportError:
    ADS1115_AVAILABLE = False

from core.bus import get_bus, publish_system_event
from core.state import get_state
from services.media.usb_audio import get_usb_audio_service


class BatteryMonitorService:
    """
    Battery voltage monitor using ADS1115 ADC
    Reads from A0 channel with calibrated voltage divider
    """

    # Calibration: empirically determined from actual measurements
    # Updated 2026-01-05: Recalibrated voltage divider factor
    CALIBRATION_FACTOR = 4.308

    # 4S LiPo voltage thresholds
    VOLTAGE_MIN = 12.0      # Empty (3.0V per cell)
    VOLTAGE_NOMINAL = 14.8  # Nominal (3.7V per cell)
    VOLTAGE_MAX = 16.8      # Full (4.2V per cell)
    VOLTAGE_CRITICAL = 12.8 # Critical low (3.2V per cell)
    VOLTAGE_LOW = 13.6      # Low warning (3.4V per cell)

    # I2C address for ADS1115
    I2C_ADDRESS = 0x48

    # Noise reduction. Median of N ADS1115 samples rejects motor/servo PWM
    # transients within a single read; EMA across reads kills the slower
    # drift / divider-amplified ADC noise. alpha=0.3 -> ~25s time constant
    # at the current 5s poll interval. Both apply uniformly to every unit
    # so the same recipe works regardless of voltage-divider ratio.
    SAMPLES_PER_READ = 8
    EMA_ALPHA = 0.3

    def __init__(self):
        self.logger = logging.getLogger('BatteryMonitor')
        self.bus = get_bus()
        self.state = get_state()

        # Per-device voltage-divider calibration. Profile yaml overrides class default
        # so each unit can carry its own divider ratio (treatbot3 has a ~54:1 divider
        # vs treatbot1's ~4.3:1). Falls back silently if config is unavailable.
        self.calibration_factor = self.CALIBRATION_FACTOR
        try:
            from config.config_loader import get_config
            cfg_factor = get_config().raw.get('battery', {}).get('calibration_factor')
            if cfg_factor is not None:
                self.calibration_factor = float(cfg_factor)
                self.logger.info(f"Battery calibration override from profile: {self.calibration_factor}")
        except Exception as e:
            self.logger.debug(f"Battery calibration: using class default ({self.CALIBRATION_FACTOR}); profile read skipped: {e}")

        # Hardware
        self.i2c = None
        self.ads = None
        self.channel = None
        self.initialized = False

        # Monitoring
        self.monitor_thread = None
        self.running = False
        self.update_interval = 5.0  # seconds

        # Current readings
        self.voltage = 0.0
        self.percentage = 0
        self.adc_voltage = 0.0
        self.last_update = 0.0

        # Smoothed-voltage state (EMA across reads). Seeded by the first read.
        self._smoothed_voltage = 0.0
        self._first_read = True

        # Warning state
        self.low_warning_sent = False
        self.critical_warning_sent = False

        # Charging detection
        self.voltage_history = []  # Rolling window for trend detection
        self.charging_detected = False
        self.charging_audio_path = '/home/morgan/dogbot/VOICEMP3/wimz/Wimz_charging.mp3'
        self.last_charging_announce = 0.0
        self.charging_announce_cooldown = 300.0  # 5 minutes cooldown
        # Motor-idle gate: voltage bouncing back after motor unload faked "charging".
        # Skip trend evaluation while motors have been active recently.
        # 4S LiPo recovery curve takes 1-2 min to flatten after sustained load.
        self.motor_idle_required_s = 120.0

    def initialize(self) -> bool:
        """Initialize ADS1115 ADC for battery monitoring"""
        if not ADS1115_AVAILABLE:
            self.logger.error("ADS1115 library not available")
            return False

        try:
            # Use shared I2C bus to avoid conflicts with servo controller
            self.i2c = get_i2c_bus()
            self.ads = ADS1115(self.i2c, address=self.I2C_ADDRESS)
            self.channel = AnalogIn(self.ads, 0)  # A0 channel

            # Take initial reading
            self._read_voltage()

            self.initialized = True
            self.logger.info(f"Battery monitor initialized: {self.voltage:.2f}V ({self.percentage}%)")

            return True

        except Exception as e:
            self.logger.error(f"Battery monitor initialization failed: {e}")
            return False

    def _read_voltage(self) -> float:
        """Read battery voltage from ADC with median+EMA smoothing.

        Single ADS1115 samples carry motor/servo PWM noise and rail-bounce.
        On high-ratio dividers (e.g. treatbot3's 54x) that noise is amplified
        ~12x vs treatbot1's 4.3x. We take SAMPLES_PER_READ samples and use the
        median to reject outliers, then EMA across reads to smooth slow drift.
        Result: percentage is stable for any consumer (HTTP, telemetry,
        charging detection) regardless of divider ratio.
        """
        if not self.initialized and not self.channel:
            return 0.0

        try:
            # Burst-sample then median (8 reads ~= 64ms at 128 SPS default)
            samples = [self.channel.voltage for _ in range(self.SAMPLES_PER_READ)]
            samples.sort()
            n = len(samples)
            mid = n // 2
            adc_median = samples[mid] if n % 2 else (samples[mid - 1] + samples[mid]) / 2.0
            raw_voltage = adc_median * self.calibration_factor

            # EMA across reads. Seed on first read so we don't bias from 0.
            if self._first_read:
                self._smoothed_voltage = raw_voltage
                self._first_read = False
            else:
                self._smoothed_voltage = (
                    self.EMA_ALPHA * raw_voltage
                    + (1.0 - self.EMA_ALPHA) * self._smoothed_voltage
                )

            self.adc_voltage = adc_median
            self.voltage = self._smoothed_voltage
            self.percentage = self._calculate_percentage(self.voltage)
            self.last_update = time.time()

            # Update system state
            self.state.update_hardware(battery_voltage=self.voltage)

            return self.voltage

        except Exception as e:
            self.logger.error(f"Battery read error: {e}")
            return self.voltage  # Return last known value

    def _calculate_percentage(self, voltage: float) -> int:
        """Calculate battery percentage from voltage"""
        if voltage <= self.VOLTAGE_MIN:
            return 0
        elif voltage >= self.VOLTAGE_MAX:
            return 100
        else:
            pct = (voltage - self.VOLTAGE_MIN) / (self.VOLTAGE_MAX - self.VOLTAGE_MIN) * 100
            return int(max(0, min(100, pct)))

    def _check_warnings(self) -> None:
        """Check for low battery warnings"""
        if self.voltage <= self.VOLTAGE_CRITICAL and not self.critical_warning_sent:
            self.logger.critical(f"CRITICAL: Battery voltage {self.voltage:.2f}V - SHUTDOWN RECOMMENDED")
            publish_system_event('battery_critical', {
                'voltage': self.voltage,
                'percentage': self.percentage
            }, 'battery_monitor')
            self.critical_warning_sent = True
            self.low_warning_sent = True

        elif self.voltage <= self.VOLTAGE_LOW and not self.low_warning_sent:
            self.logger.warning(f"LOW BATTERY: {self.voltage:.2f}V ({self.percentage}%)")
            publish_system_event('battery_low', {
                'voltage': self.voltage,
                'percentage': self.percentage
            }, 'battery_monitor')
            self.low_warning_sent = True

        elif self.voltage > self.VOLTAGE_LOW + 0.5:
            # Reset warnings when voltage recovers (with hysteresis)
            self.low_warning_sent = False
            self.critical_warning_sent = False

    def _motor_recently_active(self, now: float) -> bool:
        """True if drive motors OR dispenser were active within motor_idle_required_s.

        Motor load causes large voltage sag/rebound that the old oldest-vs-newest
        trend check falsely read as 'charging.' A stop command (0,0) still implies
        recent activity — the motors were moving up until that command — so a fresh
        stop should still keep the gate engaged while the rail rebounds.
        """
        # Drive motors: any recent command (including stops) implies recent load
        try:
            from core.motor_command_bus import get_motor_bus
            cmd = get_motor_bus().last_command
            if cmd is not None and (now - cmd.mono) < self.motor_idle_required_s:
                return True
        except Exception:
            pass

        # Dispenser: stepper pulls ~1.1A peak from the same rail
        try:
            from services.reward.dispenser import get_dispenser_service
            last_dispense = get_dispenser_service().last_dispense_mono
            if last_dispense > 0 and (now - last_dispense) < self.motor_idle_required_s:
                return True
        except Exception:
            pass

        return False

    def _check_charging(self) -> None:
        """Detect charger connect by requiring monotonic-ish rise while motors idle.

        Why: voltage-trend alone false-positives on motor unload bounce-back (a
        ~0.2V swing on the rail is normal under driving). And on units with high
        voltage-divider ratios (e.g. treatbot3's 54:1), ADC noise alone exceeds
        the old 0.05V threshold. Fix combines:
          1. Skip trend eval while motors active or recently active (<30s).
          2. Require every step in the window to not significantly drop.
          3. Require larger total rise (~0.15V) before announcing.
        """
        now = time.monotonic()  # gate uses monotonic stamps (cmd.mono / last_dispense_mono)

        # Don't sample at all while motors are loading the rail; clear stale
        # history so the next eval window starts fresh from a quiet rail.
        if self._motor_recently_active(now):
            if self.voltage_history:
                self.voltage_history.clear()
            return

        self.voltage_history.append(self.voltage)
        if len(self.voltage_history) > 5:
            self.voltage_history.pop(0)

        was_charging = self.charging_detected

        # Need full 5-sample window (~25s of quiet rail) for confident decision.
        if len(self.voltage_history) < 5:
            return

        # Step-wise check: no sample may drop by more than ~0.01V from its
        # predecessor. Tightened from 0.02 now that input is median+EMA smoothed
        # (raw single-sample drift could exceed 0.02V even on a flat rail).
        max_drop = max(
            self.voltage_history[i - 1] - self.voltage_history[i]
            for i in range(1, len(self.voltage_history))
        )
        voltage_trend = self.voltage_history[-1] - self.voltage_history[0]

        # Trend threshold 0.20 -> 0.35V. The 0.20V gate still false-triggered
        # on natural LiPo rebound after motor load (~0.2-0.3V recovery is
        # routine on a 4S after sustained current draw). Real charging delivers
        # ~1V/min on a 4S, so 0.35V over 25s is still well within legitimate
        # detection range while staying above passive rebound plateau.
        if voltage_trend >= 0.35 and max_drop <= 0.01:
            self.charging_detected = True
        elif voltage_trend < -0.05:
            self.charging_detected = False
        # else: ambiguous — keep current state (hysteresis)

        # Announce charging started
        if self.charging_detected and not was_charging:
            if now - self.last_charging_announce >= self.charging_announce_cooldown:
                self.logger.info(f"Charging detected: {self.voltage:.2f}V ({self.percentage}%), trend=+{voltage_trend:.3f}V")
                self._play_charging_audio()
                self.last_charging_announce = now

                publish_system_event('battery_charging', {
                    'voltage': self.voltage,
                    'percentage': self.percentage
                }, 'battery_monitor')

        # Log charging stopped
        if was_charging and not self.charging_detected:
            self.logger.info(f"Charging stopped: voltage trend {voltage_trend:+.3f}V")

    def _play_charging_audio(self) -> None:
        """Play charging started audio"""
        try:
            audio = get_usb_audio_service()
            if audio and audio.is_initialized:
                audio.play_file(self.charging_audio_path)
                self.logger.info("Played charging audio")
        except Exception as e:
            self.logger.debug(f"Could not play charging audio: {e}")

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        self.logger.info("Battery monitoring started")
        last_telemetry = 0

        while self.running:
            try:
                self._read_voltage()
                self._check_warnings()
                self._check_charging()

                # Log periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self.logger.debug(f"Battery: {self.voltage:.2f}V ({self.percentage}%)")

                # Send battery telemetry every 30 seconds for app display
                now = time.time()
                if now - last_telemetry >= 30:
                    # Include current mode as authoritative state for app sync
                    current_mode = self.state.get_mode().value if self.state else 'idle'
                    publish_system_event('battery_status', {
                        'percentage': self.percentage,
                        'voltage': self.voltage,
                        'charging': self.charging_detected,
                        'mode': current_mode,
                    }, 'battery_monitor')
                    last_telemetry = now

            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")

            time.sleep(self.update_interval)

        self.logger.info("Battery monitoring stopped")

    def start_monitoring(self) -> bool:
        """Start background battery monitoring"""
        if not self.initialized:
            if not self.initialize():
                return False

        if self.running:
            return True

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="BatteryMonitor"
        )
        self.monitor_thread.start()

        return True

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

    def get_voltage(self) -> float:
        """Get current battery voltage (triggers fresh read)"""
        return self._read_voltage()

    def get_percentage(self) -> int:
        """Get current battery percentage"""
        if time.time() - self.last_update > self.update_interval:
            self._read_voltage()
        return self.percentage

    def get_status(self) -> Dict[str, Any]:
        """Get full battery status"""
        if time.time() - self.last_update > self.update_interval:
            self._read_voltage()

        # Determine status string
        if self.voltage <= self.VOLTAGE_CRITICAL:
            status = "CRITICAL"
        elif self.voltage <= self.VOLTAGE_LOW:
            status = "LOW"
        elif self.percentage >= 80:
            status = "GOOD"
        else:
            status = "OK"

        return {
            'voltage': round(self.voltage, 2),
            'percentage': self.percentage,
            'status': status,
            'charging': self.charging_detected,
            'adc_voltage': round(self.adc_voltage, 3),
            'initialized': self.initialized,
            'monitoring': self.running,
            'last_update': self.last_update,
            'thresholds': {
                'min': self.VOLTAGE_MIN,
                'low': self.VOLTAGE_LOW,
                'critical': self.VOLTAGE_CRITICAL,
                'max': self.VOLTAGE_MAX
            }
        }

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_monitoring()
        self.logger.info("Battery monitor cleaned up")


# Singleton instance
_battery_instance = None
_battery_lock = threading.Lock()


def get_battery_monitor() -> BatteryMonitorService:
    """Get the global battery monitor instance"""
    global _battery_instance
    if _battery_instance is None:
        with _battery_lock:
            if _battery_instance is None:
                _battery_instance = BatteryMonitorService()
    return _battery_instance
