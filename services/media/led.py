#!/usr/bin/env python3
"""
LED service using led_controller
Creates patterns and visual feedback
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

from core.bus import get_bus, publish_system_event
from core.state import get_state, SystemMode
from core.hardware.led_controller import LEDController, LEDMode


class LedService:
    """
    LED service using LED controller
    Manages NeoPixel patterns and blue LED status
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.logger = logging.getLogger('LedService')

        # LED controller
        self.led = None
        self.led_initialized = False

        # Blue LED command file monitoring for Xbox controller
        self.last_command_check = 0

        # Pattern definitions (from leds_v3.py)
        self.patterns = {
            'off': self._pattern_off,
            'idle': self._pattern_idle,
            'searching': self._pattern_searching,
            'dog_detected': self._pattern_dog_detected,
            'treat_launching': self._pattern_treat_launching,
            'error': self._pattern_error,
            'charging': self._pattern_charging,
            'rainbow': self._pattern_rainbow,
            'pulse_green': self._pattern_pulse_green,
            'pulse_blue': self._pattern_pulse_blue,
            'spinning_dot': self._pattern_spinning_dot,
            'celebration': self._pattern_celebration,
            'manual_rc': self._pattern_manual_rc,
            'blue_led_on': self._pattern_blue_led_on,
            'blue_led_off': self._pattern_blue_led_off
        }

        # Pattern state
        self.current_pattern = 'off'
        self.pattern_thread = None
        self.pattern_running = False
        self._stop_pattern = threading.Event()

        # Manual override state - when True, ignore automatic mode changes
        self.manual_override_active = False
        self.manual_override_timeout = 0.0

        # Subscribe to system mode changes
        self.bus.subscribe('system', self._on_system_event)

        # Colors (from existing LED controller)
        self.colors = {
            'off': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'dim_white': (30, 30, 30),
            'warm_white': (255, 180, 120)
        }

    def initialize(self) -> bool:
        """Initialize LED controller"""
        try:
            self.led = LEDController()

            if self.led.is_initialized():
                self.led_initialized = True
                self.logger.info("LED system initialized")

                # Set initial pattern
                self.set_pattern('idle')

                self.state.update_hardware(leds_initialized=True)
                return True
            else:
                self.logger.error("LED controller initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"LED initialization error: {e}")
            return False

    def set_pattern(self, pattern_name: str, duration: Optional[float] = None,
                   color: Optional[str] = None, manual_override: bool = False, **kwargs) -> bool:
        """
        Set LED pattern

        Args:
            pattern_name: Name of pattern to run
            duration: Duration to run pattern (None = indefinite)
            color: Color override for pattern
            manual_override: If True, prevents automatic overrides for 30 seconds
            **kwargs: Additional pattern parameters

        Returns:
            bool: True if pattern started successfully
        """
        if not self.led_initialized:
            self.logger.error("LEDs not initialized")
            return False

        if pattern_name not in self.patterns:
            self.logger.warning(f"Unknown pattern: {pattern_name}")
            return False

        # Set manual override if requested
        if manual_override:
            self.manual_override_active = True
            self.manual_override_timeout = time.time() + 30.0  # 30 seconds
            self.logger.info("LED manual override activated for 30 seconds")

        # Stop current pattern
        self._stop_current_pattern()

        try:
            self.current_pattern = pattern_name
            self._stop_pattern.clear()

            # Update state
            self.state.update_hardware(led_pattern=pattern_name)

            # Publish event
            publish_system_event('led_pattern_changed', {
                'pattern': pattern_name,
                'duration': duration,
                'color': color,
                'manual_override': manual_override,
                'kwargs': kwargs
            }, 'led_service')

            # Start pattern thread
            self.pattern_thread = threading.Thread(
                target=self._run_pattern,
                args=(pattern_name, duration, color, kwargs),
                daemon=True,
                name=f"LEDPattern_{pattern_name}"
            )
            self.pattern_thread.start()

            self.logger.info(f"LED pattern started: {pattern_name} (manual: {manual_override})")
            return True

        except Exception as e:
            self.logger.error(f"Pattern start error: {e}")
            return False

    def _stop_current_pattern(self) -> None:
        """Stop current pattern"""
        if self.pattern_running:
            self._stop_pattern.set()
            if self.pattern_thread and self.pattern_thread.is_alive():
                self.pattern_thread.join(timeout=1.0)

    def _run_pattern(self, pattern_name: str, duration: Optional[float],
                    color: Optional[str], kwargs: Dict[str, Any]) -> None:
        """Run pattern in thread"""
        self.pattern_running = True
        start_time = time.time()

        try:
            pattern_func = self.patterns[pattern_name]
            pattern_func(duration, color, **kwargs)

        except Exception as e:
            self.logger.error(f"Pattern execution error: {e}")

        finally:
            self.pattern_running = False
            self.logger.debug(f"Pattern finished: {pattern_name}")

    # Pattern implementations
    def _pattern_off(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Turn off all LEDs (NeoPixels only - blue LED controlled by Xbox X button)"""
        if self.led.pixels:
            self.led.set_solid_color('off')
        # Blue LED controlled separately by Xbox controller X button

    def _pattern_idle(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Idle pattern - dim white (NeoPixels only)"""
        if self.led.pixels:
            self.led.set_solid_color('dim_white')

        # Run continuously and check for blue LED commands
        start_time = time.time()
        while not self._stop_pattern.is_set():
            # Check for blue LED commands from Xbox controller
            self.check_blue_led_commands()

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                break

            time.sleep(0.1)  # Check every 100ms

    def _pattern_searching(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Searching pattern - spinning cyan dot"""
        if not self.led.pixels:
            return

        search_color = color or 'cyan'
        while not self._stop_pattern.is_set():
            for i in range(self.led.neopixel_count):
                if self._stop_pattern.is_set():
                    break

                # Check for blue LED commands from Xbox controller
                self.check_blue_led_commands()

                self.led.pixels.fill(self.colors['off'])
                self.led.pixels[i] = self.colors[search_color]
                self.led.pixels.show()
                time.sleep(0.08)

    def _pattern_dog_detected(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Dog detected - solid green (NeoPixels only)"""
        if self.led.pixels:
            self.led.set_solid_color('green')
        # Blue LED controlled separately by Xbox controller X button

        if duration:
            time.sleep(duration)

    def _pattern_treat_launching(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Treat launching - bright white flash (NeoPixels only)"""
        if self.led.pixels:
            self.led.set_solid_color('white')
        # Blue LED controlled separately by Xbox controller X button

        time.sleep(0.5)

        if self.led.pixels:
            self.led.set_solid_color('off')
        # Blue LED controlled separately by Xbox controller X button

    def _pattern_error(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Error pattern - flashing red"""
        flash_count = 0
        max_flashes = 10 if duration else 999

        while not self._stop_pattern.is_set() and flash_count < max_flashes:
            if self.led.pixels:
                self.led.set_solid_color('red')
            # Blue LED controlled separately by Xbox controller X button
            time.sleep(0.3)

            if self.led.pixels:
                self.led.set_solid_color('off')
            # Blue LED controlled separately by Xbox controller X button
            time.sleep(0.3)

            flash_count += 1

    def _pattern_charging(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Charging pattern - pulsing yellow"""
        while not self._stop_pattern.is_set():
            self._pulse_effect('yellow', 2.0)

    def _pattern_rainbow(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Rainbow pattern - cycling rainbow"""
        if not self.led.pixels:
            return

        start_time = time.time()
        while not self._stop_pattern.is_set():
            if duration and time.time() - start_time > duration:
                break

            for j in range(255):
                if self._stop_pattern.is_set():
                    break

                for i in range(self.led.neopixel_count):
                    pixel_index = (i * 256 // self.led.neopixel_count) + j
                    self.led.pixels[i] = self._wheel(pixel_index & 255)

                self.led.pixels.show()
                time.sleep(0.05)

    def _pattern_pulse_green(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Pulsing green pattern"""
        self._pulse_effect('green', duration or 5.0)

    def _pattern_pulse_blue(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Pulsing blue pattern"""
        self._pulse_effect('blue', duration or 5.0)

    def _pattern_spinning_dot(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Spinning dot pattern"""
        if not self.led.pixels:
            return

        dot_color = color or 'cyan'
        start_time = time.time()

        while not self._stop_pattern.is_set():
            if duration and time.time() - start_time > duration:
                break

            for i in range(self.led.neopixel_count):
                if self._stop_pattern.is_set():
                    break

                self.led.pixels.fill(self.colors['off'])
                self.led.pixels[i] = self.colors[dot_color]
                self.led.pixels.show()
                time.sleep(0.08)

    def _pattern_celebration(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Celebration pattern - rainbow then pulse"""
        # Rainbow for 2 seconds
        self._pattern_rainbow(2.0, None)

        if not self._stop_pattern.is_set():
            # Pulse green for 1 second
            self._pulse_effect('green', 1.0)

        if not self._stop_pattern.is_set():
            # Final bright flash
            if self.led.pixels:
                self.led.set_solid_color('white')
            time.sleep(0.2)
            if self.led.pixels:
                self.led.set_solid_color('off')

    def _pattern_manual_rc(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Manual RC mode - purple base with flashing green (NeoPixels only)"""
        if not self.led.pixels:
            return

        # Blue LED controlled separately by Xbox controller X button

        start_time = time.time()
        while not self._stop_pattern.is_set():
            if duration and time.time() - start_time > duration:
                break

            # Purple base for 1 second
            self.led.pixels.fill(self.colors['purple'])
            self.led.pixels.show()
            time.sleep(0.5)

            if self._stop_pattern.is_set():
                break

            # Flash green briefly
            self.led.pixels.fill(self.colors['green'])
            self.led.pixels.show()
            time.sleep(0.1)

            if self._stop_pattern.is_set():
                break

            # Back to purple
            self.led.pixels.fill(self.colors['purple'])
            self.led.pixels.show()
            time.sleep(0.5)

    def _pulse_effect(self, color_name: str, duration: float) -> None:
        """Pulsing effect for specified duration"""
        if not self.led.pixels:
            return

        base_color = self.colors[color_name]
        steps = 20
        start_time = time.time()

        while not self._stop_pattern.is_set() and time.time() - start_time < duration:
            # Fade in
            for step in range(steps):
                if self._stop_pattern.is_set():
                    return

                brightness = step / steps
                color = tuple(int(c * brightness) for c in base_color)
                self.led.pixels.fill(color)
                self.led.pixels.show()
                time.sleep(0.05)

            # Fade out
            for step in range(steps, 0, -1):
                if self._stop_pattern.is_set():
                    return

                brightness = step / steps
                color = tuple(int(c * brightness) for c in base_color)
                self.led.pixels.fill(color)
                self.led.pixels.show()
                time.sleep(0.05)

    def _wheel(self, pos: int) -> Tuple[int, int, int]:
        """Color wheel function for rainbow effect"""
        if pos < 85:
            return (pos * 3, 255 - pos * 3, 0)
        elif pos < 170:
            pos -= 85
            return (255 - pos * 3, 0, pos * 3)
        else:
            pos -= 170
            return (0, pos * 3, 255 - pos * 3)

    def set_mode(self, mode: LEDMode) -> bool:
        """Set LED mode (compatibility with existing code)"""
        mode_patterns = {
            LEDMode.OFF: 'off',
            LEDMode.IDLE: 'idle',
            LEDMode.SEARCHING: 'searching',
            LEDMode.DOG_DETECTED: 'dog_detected',
            LEDMode.TREAT_LAUNCHING: 'treat_launching',
            LEDMode.ERROR: 'error',
            LEDMode.CHARGING: 'charging',
            LEDMode.MANUAL_RC: 'manual_rc'
        }

        pattern = mode_patterns.get(mode, 'idle')
        return self.set_pattern(pattern)

    def celebration_sequence(self, duration: float = 5.0) -> bool:
        """Run celebration sequence"""
        return self.set_pattern('celebration', duration)

    def get_available_patterns(self) -> List[str]:
        """Get list of available patterns"""
        return list(self.patterns.keys())

    def get_status(self) -> Dict[str, Any]:
        """Get LED service status"""
        return {
            'initialized': self.led_initialized,
            'current_pattern': self.current_pattern,
            'pattern_running': self.pattern_running,
            'available_patterns': len(self.patterns),
            'neopixel_count': self.led.neopixel_count if self.led else 0,
            'blue_led_on': self.led.blue_is_on if self.led else False
        }

    def _pattern_blue_led_on(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Turn blue LED on (Xbox X button)"""
        if self.led:
            success = self.led.blue_on()
            self.logger.info(f"Blue LED turned ON: {success}")

    def _pattern_blue_led_off(self, duration: Optional[float], color: Optional[str], **kwargs) -> None:
        """Turn blue LED off (Xbox X button)"""
        if self.led:
            success = self.led.blue_off()
            self.logger.info(f"Blue LED turned OFF: {success}")

    def _on_system_event(self, event) -> None:
        """Handle system events to update LED patterns"""
        if event.subtype == 'mode_transition':
            # Check if manual override is active
            if self.manual_override_active:
                if time.time() > self.manual_override_timeout:
                    self.manual_override_active = False
                    self.logger.info("LED manual override expired")
                else:
                    self.logger.debug("LED mode change ignored - manual override active")
                    return

            # Automatically update LED pattern when system mode changes
            new_mode = event.data.get('to_mode')
            if new_mode:
                self._update_led_for_system_mode(new_mode)

        elif event.subtype == 'blue_led_control':
            # Handle Xbox controller blue LED commands
            if self.led_initialized and self.led:
                action = event.data.get('action')
                if action == 'on':
                    success = self.led.blue_on()
                    self.logger.warning(f"🔵 Blue LED ON via event bus: {success}")
                elif action == 'off':
                    success = self.led.blue_off()
                    self.logger.warning(f"🔵 Blue LED OFF via event bus: {success}")
            else:
                self.logger.error("Blue LED control failed - LED not initialized")

    def _update_led_for_system_mode(self, system_mode: str) -> None:
        """Update LED pattern based on system mode"""
        mode_to_led_pattern = {
            'IDLE': 'idle',
            'DETECTION': 'searching',
            'VIGILANT': 'searching',
            'MANUAL': 'manual_rc',
            'PHOTOGRAPHY': 'dog_detected',
            'EMERGENCY': 'error',
            'SHUTDOWN': 'off'
        }

        pattern = mode_to_led_pattern.get(system_mode, 'idle')
        self.logger.info(f"System mode changed to {system_mode}, setting LED pattern: {pattern}")
        self.set_pattern(pattern)

    def check_blue_led_commands(self) -> None:
        """Check for blue LED commands from Xbox controller API"""
        try:
            import os
            command_file = '/tmp/treatbot/blue_led_command.txt'

            if os.path.exists(command_file):
                # Check if file was modified since last check
                stat = os.stat(command_file)
                if stat.st_mtime > self.last_command_check:
                    self.last_command_check = stat.st_mtime

                    with open(command_file, 'r') as f:
                        command = f.read().strip()

                    if command == 'ON' and self.led and self.led_initialized:
                        success = self.led.blue_on()
                        self.logger.warning(f"🔵 Blue LED ON executed: {success}")
                    elif command == 'OFF' and self.led and self.led_initialized:
                        success = self.led.blue_off()
                        self.logger.warning(f"🔵 Blue LED OFF executed: {success}")

                    # Clear the command file
                    os.remove(command_file)

        except Exception as e:
            pass  # Ignore errors, this is just a helper

    def cleanup(self) -> None:
        """Clean shutdown"""
        self._stop_current_pattern()
        if self.led:
            self.led.set_solid_color('off')
            # Blue LED controlled separately by Xbox controller X button
        self.logger.info("LED service cleaned up")


# Global LED service instance
_led_instance = None
_led_lock = threading.Lock()

def get_led_service() -> LedService:
    """Get the global LED service instance (singleton)"""
    global _led_instance
    if _led_instance is None:
        with _led_lock:
            if _led_instance is None:
                _led_instance = LedService()
    return _led_instance