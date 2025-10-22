#!/usr/bin/env python3
"""
Safety monitoring and emergency stops for TreatBot
Monitors battery, temperature, and system health
"""

import threading
import time
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .bus import get_bus, publish_safety_event
from .state import get_state, SystemMode


class SafetyLevel(Enum):
    """Safety alert levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyThresholds:
    """Safety monitoring thresholds"""
    # Battery thresholds (volts)
    battery_emergency: float = 11.0    # Immediate shutdown
    battery_critical: float = 11.5     # Stop all missions
    battery_warning: float = 12.0      # Reduce performance
    battery_normal: float = 12.5       # Normal operation

    # Temperature thresholds (Celsius)
    temp_emergency: float = 85.0       # Immediate shutdown
    temp_critical: float = 75.0        # Stop motors/AI
    temp_warning: float = 65.0         # Reduce performance
    temp_normal: float = 55.0          # Normal operation

    # CPU usage thresholds (%)
    cpu_critical: float = 95.0         # Stop non-essential processes
    cpu_warning: float = 85.0          # Reduce AI inference rate

    # Memory usage thresholds (%)
    memory_critical: float = 95.0      # Clear caches
    memory_warning: float = 85.0       # Reduce buffer sizes

    # Disk usage thresholds (%)
    disk_critical: float = 95.0        # Stop logging
    disk_warning: float = 85.0         # Clean old data

    # Timing thresholds
    max_no_heartbeat: float = 30.0     # Max time without main loop heartbeat
    max_continuous_runtime: float = 3600.0  # Max runtime (1 hour)


class SafetyMonitor:
    """
    Safety monitoring system for TreatBot
    Monitors battery, temperature, CPU, memory, disk usage
    Triggers emergency stops and graceful shutdowns
    """

    def __init__(self, thresholds: SafetyThresholds = None):
        self.thresholds = thresholds or SafetyThresholds()
        self.bus = get_bus()
        self.state = get_state()
        self.logger = logging.getLogger('SafetyMonitor')

        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Safety status
        self.current_level = SafetyLevel.NORMAL
        self.last_heartbeat = time.time()
        self.start_time = time.time()
        self.emergency_callbacks: list[Callable] = []

        # Measurement tracking
        self.last_measurements = {
            'battery_voltage': 0.0,
            'temperature': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'timestamp': time.time()
        }

        # Alert suppression (prevent spam)
        self.last_alerts = {}
        self.alert_cooldown = 60.0  # seconds

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start safety monitoring"""
        with self._lock:
            if self.monitoring:
                self.logger.warning("Safety monitoring already running")
                return

            self.monitoring = True
            self._stop_event.clear()
            self.start_time = time.time()
            self.last_heartbeat = time.time()

            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True,
                name="SafetyMonitor"
            )
            self.monitor_thread.start()

            self.logger.info("Safety monitoring started")
            self._publish_safety_event('monitoring_started', {'interval': interval})

    def stop_monitoring(self) -> None:
        """Stop safety monitoring"""
        with self._lock:
            if not self.monitoring:
                return

            self.monitoring = False
            self._stop_event.set()

            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)

            self.logger.info("Safety monitoring stopped")
            self._publish_safety_event('monitoring_stopped', {})

    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop"""
        while not self._stop_event.wait(interval):
            try:
                self._check_system_health()
                self._check_heartbeat()
                self._check_runtime_limits()
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")

    def _check_system_health(self) -> None:
        """Check all system health metrics"""
        measurements = self._get_system_measurements()
        self.last_measurements.update(measurements)

        # Check each metric
        self._check_battery(measurements.get('battery_voltage', 0.0))
        self._check_temperature(measurements.get('temperature', 0.0))
        self._check_cpu_usage(measurements.get('cpu_usage', 0.0))
        self._check_memory_usage(measurements.get('memory_usage', 0.0))
        self._check_disk_usage(measurements.get('disk_usage', 0.0))

        # Update state telemetry
        self.state.update_hardware(
            battery_voltage=measurements.get('battery_voltage', 0.0),
            temperature=measurements.get('temperature', 0.0)
        )

    def _get_system_measurements(self) -> Dict[str, Any]:
        """Get current system measurements"""
        measurements = {
            'timestamp': time.time(),
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

        # Try to get battery voltage (if available)
        try:
            # This would need actual hardware interface
            # For now, use state value or simulate
            measurements['battery_voltage'] = self.state.hardware.battery_voltage
        except Exception:
            measurements['battery_voltage'] = 0.0

        # Try to get temperature (if available)
        try:
            # This would need actual temperature sensor
            # For now, use state value or CPU temp
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Use first available temperature
                    temp_info = list(temps.values())[0][0]
                    measurements['temperature'] = temp_info.current
                else:
                    measurements['temperature'] = self.state.hardware.temperature
            else:
                measurements['temperature'] = self.state.hardware.temperature
        except Exception:
            measurements['temperature'] = 0.0

        return measurements

    def _check_battery(self, voltage: float) -> None:
        """Check battery voltage levels"""
        if voltage <= 0:
            return  # No valid reading

        if voltage <= self.thresholds.battery_emergency:
            self._trigger_emergency("Battery voltage critical", {'voltage': voltage})
        elif voltage <= self.thresholds.battery_critical:
            self._trigger_alert(SafetyLevel.CRITICAL, 'battery_critical',
                              f"Battery critically low: {voltage:.1f}V", {'voltage': voltage})
        elif voltage <= self.thresholds.battery_warning:
            self._trigger_alert(SafetyLevel.WARNING, 'battery_warning',
                              f"Battery low: {voltage:.1f}V", {'voltage': voltage})

    def _check_temperature(self, temp: float) -> None:
        """Check system temperature"""
        if temp <= 0:
            return  # No valid reading

        if temp >= self.thresholds.temp_emergency:
            self._trigger_emergency("Temperature critical", {'temperature': temp})
        elif temp >= self.thresholds.temp_critical:
            self._trigger_alert(SafetyLevel.CRITICAL, 'temp_critical',
                              f"Temperature critical: {temp:.1f}°C", {'temperature': temp})
        elif temp >= self.thresholds.temp_warning:
            self._trigger_alert(SafetyLevel.WARNING, 'temp_warning',
                              f"Temperature high: {temp:.1f}°C", {'temperature': temp})

    def _check_cpu_usage(self, cpu_pct: float) -> None:
        """Check CPU usage"""
        if cpu_pct >= self.thresholds.cpu_critical:
            self._trigger_alert(SafetyLevel.CRITICAL, 'cpu_critical',
                              f"CPU usage critical: {cpu_pct:.1f}%", {'cpu_usage': cpu_pct})
        elif cpu_pct >= self.thresholds.cpu_warning:
            self._trigger_alert(SafetyLevel.WARNING, 'cpu_warning',
                              f"CPU usage high: {cpu_pct:.1f}%", {'cpu_usage': cpu_pct})

    def _check_memory_usage(self, mem_pct: float) -> None:
        """Check memory usage"""
        if mem_pct >= self.thresholds.memory_critical:
            self._trigger_alert(SafetyLevel.CRITICAL, 'memory_critical',
                              f"Memory usage critical: {mem_pct:.1f}%", {'memory_usage': mem_pct})
        elif mem_pct >= self.thresholds.memory_warning:
            self._trigger_alert(SafetyLevel.WARNING, 'memory_warning',
                              f"Memory usage high: {mem_pct:.1f}%", {'memory_usage': mem_pct})

    def _check_disk_usage(self, disk_pct: float) -> None:
        """Check disk usage"""
        if disk_pct >= self.thresholds.disk_critical:
            self._trigger_alert(SafetyLevel.CRITICAL, 'disk_critical',
                              f"Disk usage critical: {disk_pct:.1f}%", {'disk_usage': disk_pct})
        elif disk_pct >= self.thresholds.disk_warning:
            self._trigger_alert(SafetyLevel.WARNING, 'disk_warning',
                              f"Disk usage high: {disk_pct:.1f}%", {'disk_usage': disk_pct})

    def _check_heartbeat(self) -> None:
        """Check main loop heartbeat"""
        time_since_heartbeat = time.time() - self.last_heartbeat
        if time_since_heartbeat > self.thresholds.max_no_heartbeat:
            self._trigger_emergency("Main loop not responding",
                                   {'time_since_heartbeat': time_since_heartbeat})

    def _check_runtime_limits(self) -> None:
        """Check runtime limits"""
        runtime = time.time() - self.start_time
        if runtime > self.thresholds.max_continuous_runtime:
            self._trigger_alert(SafetyLevel.WARNING, 'runtime_limit',
                              f"Runtime limit reached: {runtime/3600:.1f} hours",
                              {'runtime_hours': runtime/3600})

    def _trigger_alert(self, level: SafetyLevel, alert_type: str, message: str, data: Dict[str, Any]) -> None:
        """Trigger a safety alert"""
        # Check alert cooldown
        now = time.time()
        if alert_type in self.last_alerts:
            if now - self.last_alerts[alert_type] < self.alert_cooldown:
                return  # Suppress duplicate alert

        self.last_alerts[alert_type] = now
        self.current_level = max(self.current_level, level, key=lambda x: x.value)

        self.logger.warning(f"SAFETY ALERT [{level.value.upper()}]: {message}")

        # Publish safety event
        self._publish_safety_event('alert', {
            'level': level.value,
            'type': alert_type,
            'message': message,
            'data': data
        })

        # Take appropriate action based on level
        if level == SafetyLevel.CRITICAL:
            self._handle_critical_alert(alert_type, data)

    def _trigger_emergency(self, reason: str, data: Dict[str, Any]) -> None:
        """Trigger emergency shutdown"""
        self.current_level = SafetyLevel.EMERGENCY
        self.logger.error(f"EMERGENCY SHUTDOWN: {reason}")

        # Set emergency state
        self.state.set_emergency(reason)

        # Publish emergency event
        self._publish_safety_event('emergency', {
            'reason': reason,
            'data': data,
            'timestamp': time.time()
        })

        # Call emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                threading.Thread(target=callback, args=(reason, data), daemon=True).start()
            except Exception as e:
                self.logger.error(f"Emergency callback failed: {e}")

    def _handle_critical_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Handle critical safety alerts"""
        if alert_type.startswith('battery_'):
            # Stop all missions, reduce performance
            self.state.set_mode(SystemMode.IDLE, "Battery critical")

        elif alert_type.startswith('temp_'):
            # Stop motors and AI to reduce heat
            self.state.set_mode(SystemMode.IDLE, "Temperature critical")

        elif alert_type.startswith('cpu_') or alert_type.startswith('memory_'):
            # Reduce AI inference rate, clear caches
            pass  # Handled by service layer

    def heartbeat(self) -> None:
        """Update heartbeat timestamp (called by main loop)"""
        self.last_heartbeat = time.time()

    def add_emergency_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for emergency events"""
        self.emergency_callbacks.append(callback)

    def remove_emergency_callback(self, callback: Callable) -> None:
        """Remove emergency callback"""
        if callback in self.emergency_callbacks:
            self.emergency_callbacks.remove(callback)

    def _publish_safety_event(self, subtype: str, data: Dict[str, Any]) -> None:
        """Publish safety event to bus"""
        publish_safety_event(subtype, data, 'safety_monitor')

    def get_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        with self._lock:
            return {
                'monitoring': self.monitoring,
                'level': self.current_level.value,
                'uptime': time.time() - self.start_time,
                'last_heartbeat': self.last_heartbeat,
                'time_since_heartbeat': time.time() - self.last_heartbeat,
                'measurements': self.last_measurements.copy(),
                'thresholds': {
                    'battery_warning': self.thresholds.battery_warning,
                    'battery_critical': self.thresholds.battery_critical,
                    'temp_warning': self.thresholds.temp_warning,
                    'temp_critical': self.thresholds.temp_critical,
                    'max_runtime': self.thresholds.max_continuous_runtime
                }
            }

    def is_safe_to_operate(self) -> bool:
        """Check if system is safe for normal operation"""
        return self.current_level in [SafetyLevel.NORMAL, SafetyLevel.WARNING]

    def reset_alerts(self) -> None:
        """Reset safety alert level"""
        with self._lock:
            self.current_level = SafetyLevel.NORMAL
            self.last_alerts.clear()
            self.logger.info("Safety alerts reset")


# Global safety monitor instance
_safety_instance = None
_safety_lock = threading.Lock()

def get_safety_monitor() -> SafetyMonitor:
    """Get the global safety monitor instance (singleton)"""
    global _safety_instance
    if _safety_instance is None:
        with _safety_lock:
            if _safety_instance is None:
                _safety_instance = SafetyMonitor()
    return _safety_instance


def emergency_shutdown_handler(reason: str, data: Dict[str, Any]) -> None:
    """Default emergency shutdown handler"""
    logger = logging.getLogger('EmergencyShutdown')
    logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")

    # Stop all hardware (this would be implemented by hardware services)
    # For now, just log
    logger.critical("Stopping all hardware systems...")

    # Could trigger systemd shutdown, GPIO emergency stops, etc.


if __name__ == "__main__":
    # Test the safety monitor
    safety = get_safety_monitor()

    # Add emergency callback
    safety.add_emergency_callback(emergency_shutdown_handler)

    # Start monitoring
    safety.start_monitoring(interval=2.0)

    # Simulate main loop heartbeats
    try:
        for i in range(10):
            time.sleep(1)
            safety.heartbeat()
            print(f"Heartbeat {i+1}, Status: {safety.get_status()['level']}")

        # Test emergency trigger
        print("Triggering test emergency...")
        safety._trigger_emergency("Test emergency", {'test': True})

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        safety.stop_monitoring()