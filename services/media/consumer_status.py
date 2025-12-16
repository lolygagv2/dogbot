#!/usr/bin/env python3
"""
Consumer-friendly status indicators for WIM-Z
Simple LED colors that pet owners can understand at a glance
"""

import time
import logging
from typing import Dict, Any
from services.media.led import get_led_service

logger = logging.getLogger(__name__)

class ConsumerStatus:
    """
    Simple status display system for pet owners

    LED Color Meanings:
    - 🔵 BLUE: WIM-Z is ready and waiting
    - 🟢 GREEN: Good behavior detected / Training success
    - 🟠 ORANGE: Correction needed (barking detected)
    - 🔴 RED: Problem detected (camera issue, treat jam)
    - 💜 PURPLE: Training session active
    - ⚪ WHITE: Setup mode / WiFi connecting
    """

    def __init__(self):
        self.led = get_led_service()
        self.current_status = "startup"

        # Consumer-friendly status definitions
        self.status_patterns = {
            # Normal operation
            "ready": {"color": "blue", "pattern": "steady", "message": "Ready to train"},
            "training": {"color": "purple", "pattern": "pulse", "message": "Training session active"},
            "waiting_for_dog": {"color": "blue", "pattern": "slow_pulse", "message": "Looking for your dog"},

            # Positive feedback
            "good_behavior": {"color": "green", "pattern": "flash", "message": "Great behavior!"},
            "treat_dispensed": {"color": "green", "pattern": "double_flash", "message": "Treat earned!"},
            "quiet_period": {"color": "green", "pattern": "steady", "message": "Nice and quiet"},

            # Corrections
            "bark_detected": {"color": "orange", "pattern": "flash", "message": "Please be quiet"},
            "correction_mode": {"color": "orange", "pattern": "pulse", "message": "Training correction"},

            # Problems
            "camera_issue": {"color": "red", "pattern": "slow_flash", "message": "Camera problem"},
            "treat_jam": {"color": "red", "pattern": "fast_flash", "message": "Treat dispenser stuck"},
            "low_treats": {"color": "red", "pattern": "double_flash", "message": "Running low on treats"},
            "system_error": {"color": "red", "pattern": "steady", "message": "System error"},

            # Setup
            "setup_mode": {"color": "white", "pattern": "pulse", "message": "Setup mode"},
            "wifi_connecting": {"color": "white", "pattern": "flash", "message": "Connecting to WiFi"},
            "calibrating": {"color": "white", "pattern": "slow_pulse", "message": "Calibrating sensors"}
        }

    def show_status(self, status_name: str, duration: float = None):
        """
        Display a consumer-friendly status

        Args:
            status_name: Status to display (from status_patterns)
            duration: How long to show status (None = indefinite)
        """
        if status_name not in self.status_patterns:
            logger.warning(f"Unknown status: {status_name}")
            return False

        pattern = self.status_patterns[status_name]
        self.current_status = status_name

        logger.info(f"Consumer Status: {pattern['message']}")

        if self.led:
            # Show the LED pattern
            if pattern["pattern"] == "steady":
                self.led.set_color(pattern["color"], 0.7)
            elif pattern["pattern"] == "flash":
                self._flash_pattern(pattern["color"], 3, 0.3)
            elif pattern["pattern"] == "double_flash":
                self._double_flash_pattern(pattern["color"], 2)
            elif pattern["pattern"] == "pulse":
                self._pulse_pattern(pattern["color"], duration or 5)
            elif pattern["pattern"] == "slow_pulse":
                self._pulse_pattern(pattern["color"], duration or 10, slow=True)
            elif pattern["pattern"] == "fast_flash":
                self._flash_pattern(pattern["color"], 6, 0.2)
            elif pattern["pattern"] == "slow_flash":
                self._flash_pattern(pattern["color"], 3, 0.8)

        return True

    def _flash_pattern(self, color: str, count: int, interval: float):
        """Flash LED pattern"""
        for _ in range(count):
            if self.led:
                self.led.set_color(color, 0.8)
                time.sleep(interval)
                self.led.set_color("off", 0)
                time.sleep(interval)

    def _double_flash_pattern(self, color: str, count: int):
        """Double flash pattern"""
        for _ in range(count):
            # Two quick flashes
            self.led.set_color(color, 0.8)
            time.sleep(0.15)
            self.led.set_color("off", 0)
            time.sleep(0.15)
            self.led.set_color(color, 0.8)
            time.sleep(0.15)
            self.led.set_color("off", 0)
            time.sleep(0.5)  # Pause between double flashes

    def _pulse_pattern(self, color: str, duration: float, slow: bool = False):
        """Pulsing pattern"""
        start_time = time.time()
        pulse_speed = 2.0 if slow else 1.0

        while time.time() - start_time < duration:
            # Fade up
            for brightness in range(0, 100, 10):
                self.led.set_color(color, brightness / 100.0)
                time.sleep(pulse_speed * 0.05)

            # Fade down
            for brightness in range(100, 0, -10):
                self.led.set_color(color, brightness / 100.0)
                time.sleep(pulse_speed * 0.05)

    def get_current_status(self) -> Dict[str, str]:
        """Get current status for consumer dashboard"""
        if self.current_status in self.status_patterns:
            pattern = self.status_patterns[self.current_status]
            return {
                "status": self.current_status,
                "color": pattern["color"],
                "message": pattern["message"],
                "timestamp": time.time()
            }
        return {"status": "unknown", "color": "red", "message": "Unknown status"}

    def get_status_help(self) -> Dict[str, str]:
        """Get help text for LED colors (for consumer manual)"""
        help_text = {}
        for status, pattern in self.status_patterns.items():
            color_emoji = {
                "blue": "🔵", "green": "🟢", "orange": "🟠",
                "red": "🔴", "purple": "💜", "white": "⚪"
            }.get(pattern["color"], "🔘")

            help_text[pattern["color"]] = f"{color_emoji} {pattern['message']}"

        return help_text


# Global instance
_consumer_status = None

def get_consumer_status() -> ConsumerStatus:
    """Get global consumer status instance"""
    global _consumer_status
    if _consumer_status is None:
        _consumer_status = ConsumerStatus()
    return _consumer_status


if __name__ == "__main__":
    # Test consumer status patterns
    status = get_consumer_status()

    print("🎨 Testing Consumer Status LED Patterns")
    print("Each pattern shows what pet owners will see...")

    test_statuses = [
        "ready", "waiting_for_dog", "good_behavior",
        "bark_detected", "treat_dispensed", "low_treats"
    ]

    for status_name in test_statuses:
        print(f"\n🔄 Testing: {status_name}")
        status.show_status(status_name, 2)
        time.sleep(1)

    print("\n📖 Consumer LED Guide:")
    help_text = status.get_status_help()
    for color, message in help_text.items():
        print(f"  {message}")