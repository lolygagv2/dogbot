#!/usr/bin/env python3
"""
Soft Power Latch Controller for WIM-Z

Works with Pololu Mini Pushbutton Power Switch (or similar soft-latch circuit).
- On boot: Asserts STAY_ON pin to keep power flowing
- On shutdown: Releases STAY_ON pin so power switch can cut power
- Optional: Detects button press to trigger graceful shutdown

Hardware connections:
- GPIO 26 (STAY_ON_PIN) → Pololu CTRL pin (active HIGH = stay on)
- GPIO 24 (BUTTON_PIN)  → Button sense (optional, directly from button)
- GND → Common ground

Usage:
    sudo systemctl enable soft-power
    sudo systemctl start soft-power
"""

import os
import sys
import time
import signal
import logging
import threading
import atexit

# Add project root to path
sys.path.insert(0, '/home/morgan/dogbot')

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not available - running in simulation mode")

# Configuration
STAY_ON_PIN = 26       # GPIO pin to hold power switch ON (connect to Pololu CTRL)
BUTTON_PIN = 24        # GPIO pin to sense button press (optional)
SHUTDOWN_HOLD_TIME = 3.0  # Seconds button must be held to trigger shutdown
POWER_RELEASE_DELAY = 2.0  # Seconds to wait after shutdown before releasing power

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SoftPower')


class SoftPowerController:
    """
    Controls soft-latch power switch via GPIO.

    Keeps STAY_ON_PIN high during normal operation.
    Releases it on shutdown so external circuit can cut power.
    """

    def __init__(self):
        self.running = False
        self.stay_on_asserted = False
        self.button_press_start = 0
        self.shutdown_triggered = False

        # Initialize GPIO
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # STAY_ON output - keeps power switch latched ON
            GPIO.setup(STAY_ON_PIN, GPIO.OUT)
            GPIO.output(STAY_ON_PIN, GPIO.HIGH)
            self.stay_on_asserted = True
            logger.info(f"STAY_ON pin {STAY_ON_PIN} asserted HIGH - power latch engaged")

            # Button input (optional) - detect button press for shutdown
            GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            logger.info(f"Button sense pin {BUTTON_PIN} configured with pull-up")
        else:
            logger.warning("GPIO not available - running in simulation mode")
            self.stay_on_asserted = True

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("SoftPowerController initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name} - initiating power release sequence")
        self.shutdown_triggered = True
        self.running = False

    def _cleanup(self):
        """Release power latch on exit"""
        if self.stay_on_asserted and GPIO_AVAILABLE:
            logger.warning("=" * 50)
            logger.warning("RELEASING POWER LATCH - Pi shutting down")
            logger.warning("=" * 50)

            # Brief delay to ensure shutdown is committed
            time.sleep(POWER_RELEASE_DELAY)

            # Release the power latch
            GPIO.output(STAY_ON_PIN, GPIO.LOW)
            self.stay_on_asserted = False
            logger.info(f"STAY_ON pin {STAY_ON_PIN} released LOW - power switch will cut power")

            # Small delay to ensure GPIO state is stable
            time.sleep(0.1)

            # Cleanup GPIO
            GPIO.cleanup([STAY_ON_PIN, BUTTON_PIN])
            logger.info("GPIO cleanup complete")

    def _check_button(self):
        """Check if button is pressed and held for shutdown"""
        if not GPIO_AVAILABLE:
            return

        button_pressed = GPIO.input(BUTTON_PIN) == GPIO.LOW  # Active low with pull-up

        if button_pressed:
            if self.button_press_start == 0:
                self.button_press_start = time.time()
                logger.info("Power button pressed - hold for 3 seconds to shutdown")
            else:
                hold_time = time.time() - self.button_press_start
                if hold_time >= SHUTDOWN_HOLD_TIME and not self.shutdown_triggered:
                    logger.warning(f"Power button held for {hold_time:.1f}s - triggering shutdown")
                    self._trigger_shutdown()
        else:
            if self.button_press_start > 0:
                hold_time = time.time() - self.button_press_start
                if hold_time < SHUTDOWN_HOLD_TIME:
                    logger.debug(f"Power button released after {hold_time:.1f}s (need {SHUTDOWN_HOLD_TIME}s)")
            self.button_press_start = 0

    def _trigger_shutdown(self):
        """Trigger system shutdown"""
        self.shutdown_triggered = True

        try:
            # Try to play shutdown audio via the running treatbot service
            import subprocess

            # Play audio announcement
            audio_file = "/home/morgan/dogbot/VOICEMP3/wimz/shutting_down.mp3"
            fallback_audio = "/home/morgan/dogbot/VOICEMP3/wimz/Wimz_lowpower.mp3"

            if os.path.exists(audio_file):
                subprocess.run(['aplay', audio_file], timeout=5, capture_output=True)
            elif os.path.exists(fallback_audio):
                subprocess.run(['aplay', fallback_audio], timeout=5, capture_output=True)

            # Wait for audio
            time.sleep(1.0)

            # Trigger shutdown
            logger.warning("Executing system shutdown...")
            subprocess.run(['sudo', 'poweroff'], check=False)

        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            # Try shutdown anyway
            import subprocess
            subprocess.run(['sudo', 'poweroff'], check=False)

    def run(self):
        """Main loop - keeps power asserted and monitors button"""
        self.running = True
        logger.info("SoftPowerController running - power latch will stay engaged")
        logger.info(f"Press and hold button on GPIO {BUTTON_PIN} for {SHUTDOWN_HOLD_TIME}s to shutdown")

        check_count = 0
        while self.running:
            try:
                # Check button state
                self._check_button()

                # Periodic heartbeat log
                check_count += 1
                if check_count % 600 == 0:  # Every 60 seconds (at 100ms interval)
                    logger.debug("SoftPower heartbeat - power latch engaged")

                time.sleep(0.1)  # 100ms poll interval

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1.0)

        logger.info("SoftPowerController stopped")

    def get_status(self):
        """Get current status"""
        return {
            'stay_on_asserted': self.stay_on_asserted,
            'shutdown_triggered': self.shutdown_triggered,
            'gpio_available': GPIO_AVAILABLE,
            'stay_on_pin': STAY_ON_PIN,
            'button_pin': BUTTON_PIN
        }


def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("WIM-Z Soft Power Controller Starting")
    logger.info("=" * 50)

    controller = SoftPowerController()

    try:
        controller.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Soft power controller exiting")


if __name__ == '__main__':
    main()
