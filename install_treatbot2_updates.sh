#!/bin/bash
# TreatBot2 Updates Installation Script
# Installs battery monitor (ADS1115) and I2C bus sharing fixes
# Run this on the other treatbot device to apply the same changes

set -e

echo "=========================================="
echo "TreatBot2 Updates Installer"
echo "=========================================="

# Check we're in the right directory
if [ ! -f "main_treatbot.py" ]; then
    echo "ERROR: Run this script from the dogbot directory"
    echo "  cd /home/morgan/dogbot && ./install_treatbot2_updates.sh"
    exit 1
fi

echo ""
echo "1. Installing ADS1115 library..."
pip3 install adafruit-circuitpython-ads1x15

echo ""
echo "2. Creating directories..."
mkdir -p services/power
mkdir -p core/hardware
touch services/power/__init__.py

echo ""
echo "3. Creating shared I2C bus singleton..."
cat > core/hardware/i2c_bus.py << 'I2C_EOF'
#!/usr/bin/env python3
"""
Shared I2C bus singleton for all I2C devices
Prevents conflicts between PCA9685 servo controller and ADS1115 ADC
"""

import board
import busio
import threading

_i2c_instance = None
_i2c_lock = threading.RLock()  # Reentrant lock for nested calls


def get_i2c_bus():
    """
    Get the shared I2C bus instance.

    This singleton ensures all I2C devices (PCA9685, ADS1115, etc.)
    use the same bus instance to prevent conflicts.

    Returns:
        The shared I2C bus instance
    """
    global _i2c_instance
    with _i2c_lock:
        if _i2c_instance is None:
            _i2c_instance = busio.I2C(board.SCL, board.SDA)
        return _i2c_instance


def get_bus_lock():
    """Return the lock for external synchronization"""
    return _i2c_lock
I2C_EOF

echo ""
echo "4. Creating battery monitor service..."
cat > services/power/battery_monitor.py << 'BATTERY_EOF'
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


class BatteryMonitorService:
    """
    Battery voltage monitor using ADS1115 ADC
    Reads from A0 channel with calibrated voltage divider
    """

    # Calibration: empirically determined from actual measurements
    # 15.15V battery = 3.843V at ADC input
    CALIBRATION_FACTOR = 3.942

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

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        self.logger.info("Battery monitoring started")

        while self.running:
            try:
                self._read_voltage()
                self._check_warnings()

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
BATTERY_EOF

echo ""
echo "5. Updating servo_controller.py to use shared I2C bus..."
# Backup original
cp core/hardware/servo_controller.py core/hardware/servo_controller.py.bak

# Update imports
sed -i 's/import busio/from core.hardware.i2c_bus import get_i2c_bus/' core/hardware/servo_controller.py

# Update I2C initialization
sed -i 's/self.i2c = busio.I2C(board.SCL, board.SDA)/self.i2c = get_i2c_bus()/' core/hardware/servo_controller.py

# Add MODE2 fix after PCA9685 initialization
sed -i '/self.pca = PCA9685(self.i2c)/a\            self.pca.reset()\n            time.sleep(0.1)\n            # Ensure MODE2 has correct settings (OUTDRV=1, INVRT=0)\n            self.pca.mode2 = 0x04' core/hardware/servo_controller.py

echo "   Servo controller updated"

echo ""
echo "6. Adding battery API endpoints to server.py..."
if ! grep -q "/battery/status" api/server.py; then
    # Add battery endpoints before telemetry section
    sed -i '/# Telemetry endpoints/i\
# Battery monitoring endpoints\
@app.get("/battery/status")\
async def get_battery_status():\
    """Get battery voltage and status"""\
    try:\
        from services.power.battery_monitor import get_battery_monitor\
        battery = get_battery_monitor()\
        return battery.get_status()\
    except Exception as e:\
        return {"error": str(e), "initialized": False}\
\
@app.get("/battery/voltage")\
async def get_battery_voltage():\
    """Get current battery voltage"""\
    try:\
        from services.power.battery_monitor import get_battery_monitor\
        battery = get_battery_monitor()\
        voltage = battery.get_voltage()\
        return {\
            "voltage": round(voltage, 2),\
            "percentage": battery.get_percentage()\
        }\
    except Exception as e:\
        return {"error": str(e), "voltage": 0.0}\
\
' api/server.py
    echo "   Battery API endpoints added"
else
    echo "   Battery API endpoints already exist"
fi

echo ""
echo "7. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   Cache cleared"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Restart TreatBot to apply changes:"
echo "  sudo systemctl restart treatbot"
echo ""
echo "Test battery monitoring:"
echo "  curl http://localhost:8000/battery/status"
echo ""
echo "Test servos:"
echo "  curl -X POST http://localhost:8000/camera/pantilt -H 'Content-Type: application/json' -d '{\"pan\": 90, \"tilt\": 90}'"
echo ""
