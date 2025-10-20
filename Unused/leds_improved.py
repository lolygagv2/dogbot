#!/usr/bin/env python3
"""
Improved Pi 5 LED Control System
- 75 NeoPixel LED support with power management
- PWM brightness control for blue LED tube
- Robust error handling and diagnostics
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

class ImprovedLEDController:
    def __init__(self, blue_led_pin=25, neopixel_count=75, neopixel_brightness=0.3):
        """
        Initialize improved LED system for Pi 5
        Args:
            blue_led_pin: GPIO pin for blue LED tube control
            neopixel_count: Total number of NeoPixels (75 for your ring)
            neopixel_brightness: Default brightness (0.0 to 1.0)
        """
        self.neopixel_count = neopixel_count
        self.blue_led_pin = blue_led_pin
        self.blue_brightness = 0  # Current blue LED brightness (0-100)
        
        # Initialize GPIO for Pi 5
        try:
            self.gpio_chip = lgpio.gpiochip_open(0)
            
            # Set up blue LED pin for PWM control
            lgpio.gpio_claim_output(self.gpio_chip, self.blue_led_pin, lgpio.SET_PULL_NONE)
            
            # Create PWM for brightness control (1000 Hz frequency)
            self.pwm_frequency = 1000
            self.pwm_duty = 0  # Start with 0% duty cycle (off)
            
            print(f"✅ GPIO initialized successfully on Pi 5")
            print(f"🔵 Blue LED PWM control on GPIO{self.blue_led_pin}")
            
        except Exception as e:
            print(f"❌ GPIO initialization failed: {e}")
            self.gpio_chip = None
            
        # Initialize NeoPixels with correct count
        self.pixels = None
        try:
            import board
            import neopixel
            
            # Use your actual count of 75 LEDs
            self.pixels = neopixel.NeoPixel(
                board.D12, 
                self.neopixel_count, 
                brightness=neopixel_brightness, 
                auto_write=False,
                pixel_order=neopixel.GRB  # Try GRB if RGB doesn't work
            )
            print(f"✅ NeoPixel ring initialized: {self.neopixel_count} LEDs on GPIO12")
            
            # Test all LEDs briefly to verify count
            self.test_all_pixels()
            
        except Exception as e:
            print(f"❌ NeoPixel initialization failed: {e}")
            print("   Make sure you have sufficient power for 75 LEDs!")
            
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
            'pink': (255, 192, 203),
            'dim_white': (50, 50, 50),
            'warm_white': (255, 180, 120)
        }

    def test_all_pixels(self):
        """Test all 75 NeoPixels to verify they're working"""
        if not self.pixels:
            return
            
        try:
            print(f"🧪 Testing all {self.neopixel_count} NeoPixels...")
            
            # Light up in sections to identify issues
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            section_size = self.neopixel_count // 3
            
            for i, color in enumerate(colors):
                start = i * section_size
                end = min((i + 1) * section_size, self.neopixel_count)
                print(f"   Testing LEDs {start}-{end-1}: {['Red', 'Green', 'Blue'][i]}")
                
                self.pixels.fill((0, 0, 0))  # Clear all
                for j in range(start, end):
                    self.pixels[j] = color
                self.pixels.show()
                time.sleep(1)
            
            # All white briefly
            print("   Testing all LEDs: White")
            self.pixels.fill((50, 50, 50))  # Dim white to save power
            self.pixels.show()
            time.sleep(1)
            
            # Turn off
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
            print("✅ NeoPixel test complete")
            
        except Exception as e:
            print(f"❌ NeoPixel test failed: {e}")

    def set_blue_brightness(self, brightness_percent):
        """
        Set blue LED tube brightness using PWM
        Args:
            brightness_percent: Brightness level 0-100
        """
        if not self.gpio_chip:
            print("❌ GPIO not initialized")
            return
            
        try:
            # Clamp brightness to 0-100
            brightness_percent = max(0, min(100, brightness_percent))
            self.blue_brightness = brightness_percent
            
            if brightness_percent == 0:
                # Turn completely off
                lgpio.gpio_write(self.gpio_chip, self.blue_led_pin, 0)
                self.blue_light_on = False
                print(f"🔵 Blue LED: OFF")
            else:
                # Use PWM for brightness control
                # Convert percentage to duty cycle (0-255 for lgpio)
                duty_cycle = int((brightness_percent / 100) * 255)
                
                # Start PWM
                lgpio.tx_pwm(self.gpio_chip, self.blue_led_pin, self.pwm_frequency, duty_cycle)
                self.blue_light_on = True
                print(f"🔵 Blue LED: {brightness_percent}% brightness")
                
        except Exception as e:
            print(f"❌ Blue LED brightness control error: {e}")

    def blue_light_control(self, state, brightness=100):
        """
        Control blue LED tube with optional brightness
        Args:
            state: True/False for on/off
            brightness: Brightness level 0-100 when turning on
        """
        if state:
            self.set_blue_brightness(brightness)
        else:
            self.set_blue_brightness(0)

    def set_solid_color(self, color, led_range=None):
        """
        Set NeoPixels to solid color
        Args:
            color: Color name or RGB tuple
            led_range: Tuple (start, end) to light specific LEDs, None for all
        """
        if not self.pixels:
            print("⚠️  NeoPixels not available")
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['white'])
            
        try:
            if led_range:
                start, end = led_range
                start = max(0, start)
                end = min(self.neopixel_count, end)
                for i in range(start, end):
                    self.pixels[i] = color
            else:
                self.pixels.fill(color)
                
            self.pixels.show()
            range_text = f"LEDs {led_range[0]}-{led_range[1]}" if led_range else "All LEDs"
            print(f"🌈 {range_text} set to {color}")
        except Exception as e:
            print(f"❌ NeoPixel color error: {e}")

    def progressive_fill(self, color='blue', delay=0.05):
        """Fill LEDs progressively to test connectivity"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['blue'])
            
        try:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
            
            for i in range(self.neopixel_count):
                if not self.animation_active:
                    break
                self.pixels[i] = color
                self.pixels.show()
                time.sleep(delay)
                print(f"🌈 LED {i+1}/{self.neopixel_count} lit")
                
        except Exception as e:
            print(f"❌ Progressive fill error: {e}")

    def rainbow_cycle(self, wait=0.1):
        """Rainbow animation cycle for all 75 LEDs"""
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
                    for i in range(self.neopixel_count):
                        pixel_index = (i * 256 // self.neopixel_count) + j
                        self.pixels[i] = wheel(pixel_index & 255)
                    self.pixels.show()
                    time.sleep(wait)
        except Exception as e:
            print(f"❌ Rainbow animation error: {e}")

    def spinning_dot(self, color='blue', wait=0.1):
        """Single dot spinning around all 75 LEDs"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['blue'])
            
        try:
            while self.animation_active:
                for i in range(self.neopixel_count):
                    if not self.animation_active:
                        break
                    self.pixels.fill((0, 0, 0))  # Clear all
                    self.pixels[i] = color       # Light current pixel
                    self.pixels.show()
                    time.sleep(wait)
        except Exception as e:
            print(f"❌ Spinning dot error: {e}")

    def section_animation(self, colors=['red', 'green', 'blue'], wait=0.5):
        """Animate different sections with different colors"""
        if not self.pixels:
            return
            
        section_size = self.neopixel_count // len(colors)
        
        try:
            while self.animation_active:
                for offset in range(section_size):
                    if not self.animation_active:
                        break
                    self.pixels.fill((0, 0, 0))
                    
                    for i, color_name in enumerate(colors):
                        color = self.colors.get(color_name, self.colors['white'])
                        start = (i * section_size + offset) % self.neopixel_count
                        for j in range(section_size // 3):  # Light 1/3 of section
                            pixel_index = (start + j) % self.neopixel_count
                            self.pixels[pixel_index] = color
                    
                    self.pixels.show()
                    time.sleep(wait)
        except Exception as e:
            print(f"❌ Section animation error: {e}")

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

    def set_mode(self, mode: LEDMode, blue_brightness=80):
        """Set LED mode with appropriate animation and blue brightness"""
        self.current_mode = mode
        print(f"🎨 LED Mode: {mode.value}")
        
        if mode == LEDMode.OFF:
            self.stop_animation()
            self.set_solid_color('off')
            self.blue_light_control(False)
            
        elif mode == LEDMode.IDLE:
            self.stop_animation()
            self.set_solid_color('dim_white')
            self.blue_light_control(True, blue_brightness)
            
        elif mode == LEDMode.SEARCHING:
            self.blue_light_control(True, blue_brightness)
            self.start_animation(self.spinning_dot, 'cyan', 0.08)
            
        elif mode == LEDMode.DOG_DETECTED:
            self.blue_light_control(True, blue_brightness)
            self.start_animation(self.section_animation, ['green', 'yellow', 'green'], 0.3)
            
        elif mode == LEDMode.TREAT_LAUNCHING:
            self.blue_light_control(True, 100)  # Full brightness for treat launch
            self.start_animation(self.rainbow_cycle, 0.05)
            
        elif mode == LEDMode.ERROR:
            self.blue_light_control(False)
            self.start_animation(self.section_animation, ['red', 'off', 'red'], 0.2)
            
        elif mode == LEDMode.CHARGING:
            self.blue_light_control(True, 30)  # Dim for charging
            self.start_animation(self.progressive_fill, 'orange', 0.1)

    def cleanup(self):
        """Clean shutdown"""
        print("🧹 Cleaning up LED system...")
        self.stop_animation()
        if self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
        self.blue_light_control(False)
        if self.gpio_chip:
            lgpio.tx_pwm_stop(self.gpio_chip, self.blue_led_pin)
            lgpio.gpiochip_close(self.gpio_chip)

def interactive_test():
    """Interactive test for brightness and LED count"""
    print("🎮 Interactive LED Control Test")
    print("Commands:")
    print("  blue [0-100]     - Set blue LED brightness")
    print("  neo [color]      - Set NeoPixel color")
    print("  fill             - Progressive fill test")
    print("  mode [mode]      - Set LED mode")
    print("  quit             - Exit")
    
    leds = ImprovedLEDController(blue_led_pin=25, neopixel_count=75, neopixel_brightness=0.4)
    
    try:
        while True:
            cmd = input("\nEnter command: ").strip().lower().split()
            if not cmd:
                continue
                
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'blue':
                brightness = int(cmd[1]) if len(cmd) > 1 else 50
                leds.set_blue_brightness(brightness)
            elif cmd[0] == 'neo':
                color = cmd[1] if len(cmd) > 1 else 'blue'
                leds.set_solid_color(color)
            elif cmd[0] == 'fill':
                color = cmd[1] if len(cmd) > 1 else 'blue'
                leds.stop_animation()
                leds.start_animation(leds.progressive_fill, color, 0.03)
            elif cmd[0] == 'mode':
                mode_name = cmd[1] if len(cmd) > 1 else 'idle'
                try:
                    mode = LEDMode(mode_name)
                    leds.set_mode(mode)
                except ValueError:
                    print(f"Invalid mode. Available: {[m.value for m in LEDMode]}")
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        pass
    finally:
        leds.cleanup()

if __name__ == "__main__":
    interactive_test()
