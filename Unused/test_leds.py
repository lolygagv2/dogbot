#!/usr/bin/env python3
"""
TreatBot LED Control System
Controls NeoPixel ring and blue LED tube light
GPIO Pin Assignments based on your setup:
- GPIO12: NeoPixel ring data
- GPIO25: Blue LED tube control (via MOSFET/relay)
"""

import time
import board
import neopixel
import RPi.GPIO as GPIO
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

class TreatBotLEDs:
    def __init__(self, num_pixels=24, brightness=0.5):
        """
        Initialize LED system
        Args:
            num_pixels: Number of NeoPixels in ring (adjust based on your hardware)
            brightness: Default brightness (0.0 to 1.0)
        """
        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(25, GPIO.OUT)  # Blue LED tube control
        
        # NeoPixel setup - GPIO12 (Pin 32)
        try:
            self.pixels = neopixel.NeoPixel(board.D12, num_pixels, 
                                          brightness=brightness, auto_write=False)
            self.num_pixels = num_pixels
            print(f"✅ NeoPixel ring initialized: {num_pixels} LEDs on GPIO12")
        except Exception as e:
            print(f"❌ NeoPixel initialization failed: {e}")
            self.pixels = None
            
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
        """Control blue LED tube"""
        try:
            GPIO.output(25, GPIO.HIGH if state else GPIO.LOW)
            self.blue_light_on = state
            status = "ON" if state else "OFF"
            print(f"🔵 Blue LED tube: {status}")
        except Exception as e:
            print(f"❌ Blue LED control error: {e}")

    def set_solid_color(self, color):
        """Set all NeoPixels to solid color"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['white'])
            
        try:
            self.pixels.fill(color)
            self.pixels.show()
        except Exception as e:
            print(f"❌ NeoPixel solid color error: {e}")

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
                    for i in range(self.num_pixels):
                        pixel_index = (i * 256 // self.num_pixels) + j
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
                for i in range(self.num_pixels):
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
        GPIO.cleanup()

def test_sequence():
    """Test all LED functions"""
    print("🚀 Starting LED test sequence...")
    
    # Initialize LED system
    leds = TreatBotLEDs(num_pixels=24, brightness=0.3)
    
    try:
        # Test 1: Blue light control
        print("\n🔵 Testing blue LED tube...")
        leds.blue_light_control(True)
        time.sleep(2)
        leds.blue_light_control(False)
        time.sleep(1)
        
        # Test 2: Solid colors
        print("\n🌈 Testing solid colors...")
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'white']
        for color in colors:
            print(f"   Color: {color}")
            leds.set_solid_color(color)
            time.sleep(1)
        
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
        
        print("\n✅ LED test sequence complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        leds.cleanup()

if __name__ == "__main__":
    test_sequence()
