#!/usr/bin/env python3
"""
Raspberry Pi 5 Compatible LED Control System
Uses lgpio instead of RPi.GPIO for Pi 5 compatibility
"""

import time
import lgpio
import threading
from enum import Enum

class LEDMode(Enum):
    OFF = "off"
    IDLE = "idle"
    SEARCHING = "searching"
    DOG_DETECTED = "dog_detected"
    TREAT_LAUNCHING = "treat_launching"
    ERROR = "error"
    CHARGING = "charging"

class Pi5LEDController:
    def __init__(self, blue_led_pin=25):
        """
        Initialize Pi 5 compatible LED system
        Args:
            blue_led_pin: GPIO pin for blue LED tube control
        """
        try:
            # Open GPIO chip (Pi 5 compatible)
            self.gpio_chip = lgpio.gpiochip_open(0)
            self.blue_led_pin = blue_led_pin
            
            # Set blue LED pin as output
            lgpio.gpio_claim_output(self.gpio_chip, self.blue_led_pin, lgpio.SET_PULL_NONE)
            
            print(f"✅ GPIO initialized successfully on Pi 5")
            print(f"🔵 Blue LED control on GPIO{self.blue_led_pin}")
            
        except Exception as e:
            print(f"❌ GPIO initialization failed: {e}")
            self.gpio_chip = None
            
        # Try to initialize NeoPixels
        self.pixels = None
        try:
            import board
            import neopixel
            self.pixels = neopixel.NeoPixel(board.D12, 24, brightness=0.3, auto_write=False)
            print(f"✅ NeoPixel ring initialized: 24 LEDs on GPIO12")
        except Exception as e:
            print(f"❌ NeoPixel initialization failed: {e}")
            print("   Make sure you installed: pip install adafruit-circuitpython-neopixel")
            
        # State management
        self.current_mode = LEDMode.OFF
        self.animation_active = False
        self.animation_thread = None
        self.blue_light_on = False
        
        # Colors (R, G, B)
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
            'pink': (255, 192, 203)
        }

    def blue_light_control(self, state):
        """Control blue LED tube using lgpio"""
        if not self.gpio_chip:
            print("❌ GPIO not initialized")
            return
            
        try:
            lgpio.gpio_write(self.gpio_chip, self.blue_led_pin, 1 if state else 0)
            self.blue_light_on = state
            status = "ON" if state else "OFF"
            print(f"🔵 Blue LED tube: {status}")
        except Exception as e:
            print(f"❌ Blue LED control error: {e}")

    def set_solid_color(self, color):
        """Set all NeoPixels to solid color"""
        if not self.pixels:
            print("⚠️  NeoPixels not available")
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['white'])
            
        try:
            self.pixels.fill(color)
            self.pixels.show()
            print(f"🌈 NeoPixels set to {color}")
        except Exception as e:
            print(f"❌ NeoPixel color error: {e}")

    def rainbow_cycle(self, wait=0.1):
        """Rainbow animation cycle"""
        if not self.pixels:
            return
            
        def wheel(pos):
            """Generate rainbow colors across 0-255 positions"""
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
                    for i in range(len(self.pixels)):
                        pixel_index = (i * 256 // len(self.pixels)) + j
                        self.pixels[i] = wheel(pixel_index & 255)
                    self.pixels.show()
                    time.sleep(wait)
        except Exception as e:
            print(f"❌ Rainbow animation error: {e}")

    def spinning_dot(self, color='blue', wait=0.1):
        """Single dot spinning around the ring"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['blue'])
            
        try:
            while self.animation_active:
                for i in range(len(self.pixels)):
                    if not self.animation_active:
                        break
                    self.pixels.fill((0, 0, 0))  # Clear all
                    self.pixels[i] = color       # Light current pixel
                    self.pixels.show()
                    time.sleep(wait)
        except Exception as e:
            print(f"❌ Spinning dot error: {e}")

    def pulsing_effect(self, color='green', steps=20, wait=0.05):
        """Pulsing brightness effect"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['green'])
            
        try:
            while self.animation_active:
                # Fade in
                for brightness in range(0, steps):
                    if not self.animation_active:
                        break
                    factor = brightness / (steps - 1)
                    dimmed_color = tuple(int(c * factor) for c in color)
                    self.pixels.fill(dimmed_color)
                    self.pixels.show()
                    time.sleep(wait)
                
                # Fade out
                for brightness in range(steps - 1, -1, -1):
                    if not self.animation_active:
                        break
                    factor = brightness / (steps - 1)
                    dimmed_color = tuple(int(c * factor) for c in color)
                    self.pixels.fill(dimmed_color)
                    self.pixels.show()
                    time.sleep(wait)
        except Exception as e:
            print(f"❌ Pulsing effect error: {e}")

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
        """Set LED mode with appropriate animation"""
        self.current_mode = mode
        print(f"🎨 LED Mode: {mode.value}")
        
        if mode == LEDMode.OFF:
            self.stop_animation()
            self.set_solid_color('off')
            self.blue_light_control(False)
            
        elif mode == LEDMode.IDLE:
            self.stop_animation()
            self.set_solid_color('blue')
            self.blue_light_control(True)
            
        elif mode == LEDMode.SEARCHING:
            self.blue_light_control(True)
            self.start_animation(self.spinning_dot, 'cyan', 0.1)
            
        elif mode == LEDMode.DOG_DETECTED:
            self.blue_light_control(True)
            self.start_animation(self.pulsing_effect, 'green', 15, 0.03)
            
        elif mode == LEDMode.TREAT_LAUNCHING:
            self.blue_light_control(True)
            self.start_animation(self.rainbow_cycle, 0.05)
            
        elif mode == LEDMode.ERROR:
            self.blue_light_control(False)
            self.start_animation(self.pulsing_effect, 'red', 10, 0.1)
            
        elif mode == LEDMode.CHARGING:
            self.blue_light_control(False)
            self.start_animation(self.pulsing_effect, 'orange', 20, 0.08)

    def cleanup(self):
        """Clean shutdown"""
        print("🧹 Cleaning up LED system...")
        self.stop_animation()
        if self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
        self.blue_light_control(False)
        if self.gpio_chip:
            lgpio.gpiochip_close(self.gpio_chip)

def test_sequence():
    """Test all LED functions on Pi 5"""
    print("🚀 Starting Pi 5 LED test sequence...")
    
    # Initialize LED system
    leds = Pi5LEDController(blue_led_pin=25)
    
    try:
        # Test 1: Blue light control
        print("\n🔵 Testing blue LED tube...")
        leds.blue_light_control(True)
        time.sleep(2)
        leds.blue_light_control(False)
        time.sleep(1)
        
        # Test 2: Solid colors (if NeoPixels available)
        if leds.pixels:
            print("\n�� Testing solid colors...")
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'white']
            for color in colors:
                print(f"   Color: {color}")
                leds.set_solid_color(color)
                time.sleep(1)
        else:
            print("\n⚠️  Skipping NeoPixel tests - not available")
        
        # Test 3: LED modes
        print("\n🎭 Testing LED modes...")
        modes = [
            (LEDMode.IDLE, 3),
            (LEDMode.SEARCHING, 5),
            (LEDMode.DOG_DETECTED, 4),
            (LEDMode.TREAT_LAUNCHING, 3),
            (LEDMode.ERROR, 3),
            (LEDMode.CHARGING, 3)
        ]
        
        for mode, duration in modes:
            print(f"   Mode: {mode.value}")
            leds.set_mode(mode)
            time.sleep(duration)
        
        # Test 4: Turn everything off
        print("\n⚫ Turning off...")
        leds.set_mode(LEDMode.OFF)
        time.sleep(1)
        
        print("\n✅ Pi 5 LED test sequence complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        leds.cleanup()

if __name__ == "__main__":
    test_sequence()
