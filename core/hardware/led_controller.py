#!/usr/bin/env python3
"""
core/led_controller.py - NeoPixel and Blue LED management
Refactored from proven leds_v3.py with all working patterns and blue LED control
"""

import time
import lgpio
import threading
from enum import Enum

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pins import TreatBotPins
from config.settings import SystemSettings, Colors

class LEDMode(Enum):
    OFF = "off"
    IDLE = "idle"
    SEARCHING = "searching"
    DOG_DETECTED = "dog_detected"
    TREAT_LAUNCHING = "treat_launching"
    ERROR = "error"
    CHARGING = "charging"
    MANUAL_RC = "manual_rc"

class LEDController:
    """LED system management with proven NeoPixel and Blue LED control"""
    
    def __init__(self, neopixel_count=None, neopixel_brightness=None):
        self.pins = TreatBotPins()
        self.settings = SystemSettings()
        
        # Use provided values or defaults from settings
        self.neopixel_count = neopixel_count or self.settings.NEOPIXEL_COUNT
        self.neopixel_brightness = neopixel_brightness or self.settings.NEOPIXEL_BRIGHTNESS
        
        # State management
        self.current_mode = LEDMode.OFF
        self.animation_active = False
        self.animation_thread = None
        self.blue_is_on = False
        
        # Hardware initialization
        self.blue_chip = None
        self.pixels = None
        
        # Blue LED disabled - now handled by API server only
        self.blue_chip = None
        self.blue_is_on = False
        print("Blue LED control moved to API server")

        # NeoPixels disabled - now handled by API server only
        self.pixels = None
        print("NeoPixel control moved to API server")
    
    def _initialize_blue_led(self):
        """Initialize blue LED using proven lgpio method"""
        try:
            self.blue_chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.blue_chip, self.pins.BLUE_LED, lgpio.SET_PULL_NONE)
            lgpio.gpio_write(self.blue_chip, self.pins.BLUE_LED, 0)  # Start OFF
            self.blue_is_on = False
            print(f"Blue LED initialized on GPIO{self.pins.BLUE_LED}")
            return True
            
        except Exception as e:
            print(f"Blue LED initialization failed: {e}")
            self.blue_chip = None
            return False
    
    def _initialize_neopixels(self):
        """Initialize NeoPixel ring using proven method"""
        try:
            import board
            import neopixel
            
            self.pixels = neopixel.NeoPixel(
                board.D12,  # GPIO12 from pins configuration
                self.neopixel_count,
                brightness=self.neopixel_brightness,
                auto_write=False,
                pixel_order=neopixel.GRB
            )
            
            # Start with all LEDs off
            self.pixels.fill(Colors.OFF)
            self.pixels.show()
            
            print(f"NeoPixel ring initialized: {self.neopixel_count} LEDs on GPIO{self.pins.NEOPIXEL}")
            return True
            
        except Exception as e:
            print(f"NeoPixel initialization failed: {e}")
            self.pixels = None
            return False

    def is_initialized(self) -> bool:
        """Check if LED system is properly initialized"""
        return (self.blue_chip is not None) or (self.pixels is not None)

    def blue_on(self):
        """Blue LED now handled by API server"""
        print("Blue LED: API server handles blue LED control")
        return True

    def blue_off(self):
        """Blue LED now handled by API server"""
        print("Blue LED: API server handles blue LED control")
        return True
    
    def set_neopixel_brightness(self, level):
        """Set NeoPixel brightness (0.1 to 1.0)"""
        if not self.pixels:
            return False
            
        try:
            level = max(0.1, min(1.0, level))
            self.pixels.brightness = level
            self.pixels.show()
            print(f"NeoPixel brightness: {level}")
            return True
            
        except Exception as e:
            print(f"Brightness error: {e}")
            return False
    
    def set_solid_color(self, color, stop_animations=True):
        """Set all NeoPixels to solid color"""
        if not self.pixels:
            return False
            
        if stop_animations:
            self.stop_animation()
            time.sleep(0.1)
        
        # Handle color input (string or RGB tuple)
        if isinstance(color, str):
            # Get color from Colors class using getattr
            rgb_color = getattr(Colors, color.upper(), Colors.WHITE)
        else:
            rgb_color = color
            
        try:
            self.pixels.fill(rgb_color)
            self.pixels.show()
            print(f"NeoPixels: solid {color}")
            return True
            
        except Exception as e:
            print(f"NeoPixel color error: {e}")
            return False
    
    def spinning_dot(self, color=Colors.SEARCHING, delay=0.08):
        """Single dot spinning animation (proven pattern)"""
        if not self.pixels:
            return
            
        try:
            while self.animation_active:
                for i in range(self.neopixel_count):
                    if not self.animation_active:
                        break
                    self.pixels.fill(Colors.OFF)
                    self.pixels[i] = color
                    self.pixels.show()
                    time.sleep(delay)
                    
        except Exception as e:
            print(f"Spinning dot animation error: {e}")
    
    def pulse_color(self, color=Colors.DOG_DETECTED, steps=20, delay=0.05):
        """Pulsing effect (proven pattern)"""
        if not self.pixels:
            return
            
        try:
            while self.animation_active:
                # Fade in
                for step in range(steps):
                    if not self.animation_active:
                        break
                    factor = step / (steps - 1)
                    dimmed = tuple(int(c * factor) for c in color)
                    self.pixels.fill(dimmed)
                    self.pixels.show()
                    time.sleep(delay)
                
                # Fade out
                for step in range(steps - 1, -1, -1):
                    if not self.animation_active:
                        break
                    factor = step / (steps - 1)
                    dimmed = tuple(int(c * factor) for c in color)
                    self.pixels.fill(dimmed)
                    self.pixels.show()
                    time.sleep(delay)
                    
        except Exception as e:
            print(f"Pulse animation error: {e}")
    
    def rainbow_cycle(self, delay=0.05):
        """Rainbow animation (proven pattern)"""
        if not self.pixels:
            return
            
        def wheel(pos):
            """Generate rainbow colors"""
            if pos < 85:
                return (pos * 3, 255 - pos * 3, 0)
            elif pos < 170:
                pos -= 85
                return (255 - pos * 3, 0, pos * 3)
            else:
                pos -= 170
                return (0, pos * 3, 255 - pos * 3)
        
        try:
            while self.animation_active:
                for j in range(255):
                    if not self.animation_active:
                        break
                    for i in range(self.neopixel_count):
                        pixel_index = (i * 256 // self.neopixel_count) + j
                        self.pixels[i] = wheel(pixel_index & 255)
                    self.pixels.show()
                    time.sleep(delay)
                    
        except Exception as e:
            print(f"Rainbow animation error: {e}")

    def manual_rc_pattern(self, delay=0.5):
        """Purple base with flashing green pattern for manual RC mode"""
        if not self.pixels:
            return

        try:
            while self.animation_active:
                # Purple base for 1 second
                self.pixels.fill(Colors.PURPLE)
                self.pixels.show()
                time.sleep(delay)

                if not self.animation_active:
                    break

                # Flash green briefly
                self.pixels.fill(Colors.GREEN)
                self.pixels.show()
                time.sleep(0.1)

                if not self.animation_active:
                    break

                # Back to purple
                self.pixels.fill(Colors.PURPLE)
                self.pixels.show()
                time.sleep(delay)

        except Exception as e:
            print(f"Manual RC pattern error: {e}")

    def stop_animation(self):
        """Stop current animation"""
        self.animation_active = False
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=2)
    
    def start_animation(self, animation_func, *args, **kwargs):
        """Start animation in separate thread"""
        self.stop_animation()
        self.animation_active = True
        self.animation_thread = threading.Thread(
            target=animation_func, args=args, kwargs=kwargs
        )
        self.animation_thread.daemon = True
        self.animation_thread.start()
    
    def set_mode(self, mode: LEDMode):
        """Set LED mode with appropriate patterns (proven combinations)"""
        self.current_mode = mode
        print(f"LED Mode: {mode.value}")
        
        if mode == LEDMode.OFF:
            self.stop_animation()
            self.set_solid_color(Colors.OFF)
            self.blue_off()
            
        elif mode == LEDMode.IDLE:
            self.stop_animation()
            self.set_solid_color(Colors.IDLE)
            self.blue_on()
            
        elif mode == LEDMode.SEARCHING:
            self.blue_on()
            self.start_animation(self.spinning_dot, Colors.SEARCHING, 0.08)
            
        elif mode == LEDMode.DOG_DETECTED:
            self.blue_on()
            self.start_animation(self.pulse_color, Colors.DOG_DETECTED, 15, 0.03)
            
        elif mode == LEDMode.TREAT_LAUNCHING:
            self.blue_on()
            self.start_animation(self.rainbow_cycle, 0.05)
            
        elif mode == LEDMode.ERROR:
            self.blue_off()
            self.start_animation(self.pulse_color, Colors.ERROR, 10, 0.1)
            
        elif mode == LEDMode.CHARGING:
            self.blue_on()  # Keep blue on for charging indicator
            self.start_animation(self.pulse_color, Colors.CHARGING, 20, 0.08)

        elif mode == LEDMode.MANUAL_RC:
            self.blue_on()  # Keep blue on for manual mode indicator
            self.start_animation(self.manual_rc_pattern, 0.5)
    
    def get_status(self):
        """Get current LED system status"""
        return {
            'blue_led_on': self.blue_is_on,
            'current_mode': self.current_mode.value,
            'animation_active': self.animation_active,
            'neopixel_brightness': self.pixels.brightness if self.pixels else None,
            'blue_led_initialized': self.blue_chip is not None,
            'neopixels_initialized': self.pixels is not None
        }
    
    def is_initialized(self):
        """Check if LED system is properly initialized"""
        return self.blue_chip is not None and self.pixels is not None
    
    def cleanup(self):
        """Clean shutdown of LED system"""
        print("Cleaning up LED controller...")
        
        # Stop animations
        self.stop_animation()
        
        # Turn off NeoPixels
        if self.pixels:
            try:
                self.pixels.fill(Colors.OFF)
                self.pixels.show()
                print("NeoPixels turned off")
            except Exception as e:
                print(f"NeoPixel cleanup error: {e}")
        
        # Turn off blue LED
        self.blue_off()
        
        # Close GPIO
        if self.blue_chip:
            try:
                lgpio.gpiochip_close(self.blue_chip)
                self.blue_chip = None
                print("Blue LED GPIO closed")
            except Exception as e:
                print(f"Blue LED cleanup error: {e}")

# Test function for individual module testing
def test_leds():
    """Simple test function for LED controller"""
    print("Testing LED Controller...")
    
    leds = LEDController()
    if not leds.is_initialized():
        print("LED initialization incomplete!")
        return
    
    try:
        # Test blue LED
        print("Testing blue LED...")
        leds.blue_on()
        time.sleep(1)
        leds.blue_off()
        time.sleep(1)
        
        # Test solid colors
        print("Testing solid colors...")
        leds.set_solid_color('red')
        time.sleep(1)
        leds.set_solid_color('green')
        time.sleep(1)
        leds.set_solid_color('blue')
        time.sleep(1)
        
        # Test modes
        print("Testing modes...")
        leds.set_mode(LEDMode.IDLE)
        time.sleep(2)
        
        leds.set_mode(LEDMode.SEARCHING)
        time.sleep(3)
        
        leds.set_mode(LEDMode.DOG_DETECTED)
        time.sleep(3)
        
        leds.set_mode(LEDMode.OFF)
        
        print("LED test complete!")
        
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        leds.cleanup()

if __name__ == "__main__":
    test_leds()
