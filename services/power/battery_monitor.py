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

    def __init__(self):
        self.logger = logging.getLogger('BatteryMonitor')
        self.bus = get_bus()
        self.state = get_state()

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

        # Warning state
        self.low_warning_sent = False
        self.critical_warning_sent = False

        # Charging detection
        self.voltage_history = []  # Last 3 readings for trend detection
        self.charging_detected = False
        self.charging_audio_path = '/home/morgan/dogbot/VOICEMP3/wimz/Wimz_charging.mp3'
        self.last_charging_announce = 0.0
        self.charging_announce_cooldown = 300.0  # 5 minutes cooldown

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
        """Read battery voltage from ADC"""
        if not self.initialized and not self.channel:
            return 0.0

        try:
            self.adc_voltage = self.channel.voltage
            self.voltage = self.adc_voltage * self.CALIBRATION_FACTOR
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

    def _check_charging(self) -> None:
        """Check if charging started (voltage increasing)"""
        # Track voltage history
        self.voltage_history.append(self.voltage)
        if len(self.voltage_history) > 3:
            self.voltage_history.pop(0)

        # Need at least 3 readings to detect trend
        if len(self.voltage_history) < 3:
            return

        # Check if voltage is consistently increasing (charging)
        v1, v2, v3 = self.voltage_history
        voltage_increase = v3 - v1

        # Detect charging: voltage increased by 0.05V+ over last 3 readings (~15 seconds)
        # Even small voltage increases indicate charger is connected
        now = time.time()
        if voltage_increase >= 0.05 and not self.charging_detected:
            self.charging_detected = True

            # Play charging audio if cooldown elapsed
            if now - self.last_charging_announce >= self.charging_announce_cooldown:
                self.logger.info(f"Charging detected: voltage rose {voltage_increase:.2f}V")
                self._play_charging_audio()
                self.last_charging_announce = now

                publish_system_event('battery_charging', {
                    'voltage': self.voltage,
                    'percentage': self.percentage,
                    'voltage_increase': voltage_increase
                }, 'battery_monitor')

        # Reset charging flag if voltage drops (charger disconnected)
        elif voltage_increase < 0 and self.charging_detected:
            self.charging_detected = False
            self.logger.info("Charging stopped: voltage decreasing")

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

        while self.running:
            try:
                self._read_voltage()
                self._check_warnings()
                self._check_charging()

                # Log periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self.logger.debug(f"Battery: {self.voltage:.2f}V ({self.percentage}%)")

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
