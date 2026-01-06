#!/usr/bin/env python3
"""
Clean LED Controller - Based on Working Simple Version
Uses reliable digital control + working NeoPixel system
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

class CleanLEDController:
    def __init__(self, blue_led_pin=25, neopixel_count=75, neopixel_brightness=0.3):
        """
        Initialize clean LED system - no PWM conflicts
        """
        self.neopixel_count = neopixel_count
        self.blue_led_pin = blue_led_pin
        
        # Initialize Blue LED with simple digital control
        self.blue_chip = None
        try:
            self.blue_chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.blue_chip, self.blue_led_pin, lgpio.SET_PULL_NONE)
            # Start with LED OFF
            lgpio.gpio_write(self.blue_chip, self.blue_led_pin, 0)
            print(f"✅ Blue LED initialized on GPIO{self.blue_led_pin} - Starting OFF")
        except Exception as e:
            print(f"❌ Blue LED initialization failed: {e}")
            
        # Initialize NeoPixels
        self.pixels = None
        try:
            import board
            import neopixel
            
            self.pixels = neopixel.NeoPixel(
                board.D10,  # GPIO10 (Pin 19)
                self.neopixel_count, 
                brightness=neopixel_brightness, 
                auto_write=False,
                pixel_order=neopixel.GRB
            )
            print(f"✅ NeoPixel ring initialized: {self.neopixel_count} LEDs on GPIO10 (Pin 19)")
            
        except Exception as e:
            print(f"❌ NeoPixel initialization failed: {e}")
            
        # State management
        self.current_mode = LEDMode.OFF
        self.animation_active = False
        self.animation_thread = None
        self.blue_is_on = False
        
        # Colors
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

    def blue_on(self):
        """Turn blue LED on - simple and reliable"""
        if not self.blue_chip:
            return
        try:
            lgpio.gpio_write(self.blue_chip, self.blue_led_pin, 1)
            self.blue_is_on = True
            print("🔵 Blue LED: ON")
        except Exception as e:
            print(f"❌ Blue LED on error: {e}")

    def blue_off(self):
        """Turn blue LED off - simple and reliable"""
        if not self.blue_chip:
            return
        try:
            # Use the proven method from simple test
            lgpio.gpio_write(self.blue_chip, self.blue_led_pin, 0)
            time.sleep(0.05)  # Small delay for reliability
            
            # Double-check with reclaim if needed
            lgpio.gpio_free(self.blue_chip, self.blue_led_pin)
            lgpio.gpio_claim_output(self.blue_chip, self.blue_led_pin, lgpio.SET_PULL_DOWN)
            lgpio.gpio_write(self.blue_chip, self.blue_led_pin, 0)
            
            self.blue_is_on = False
            print("🔵 Blue LED: OFF")
        except Exception as e:
            print(f"❌ Blue LED off error: {e}")

    def set_neopixel_brightness(self, level):
        """Set NeoPixel brightness (0.1 to 1.0)"""
        if not self.pixels:
            return
        try:
            level = max(0.1, min(1.0, level))
            self.pixels.brightness = level
            self.pixels.show()
            print(f"🔆 NeoPixel brightness: {level}")
        except Exception as e:
            print(f"❌ Brightness error: {e}")

    def set_solid_color(self, color, stop_animations=True):
        """Set all NeoPixels to solid color"""
        if not self.pixels:
            return
            
        if stop_animations:
            self.stop_animation()
            time.sleep(0.1)
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['white'])
            
        try:
            self.pixels.fill(color)
            self.pixels.show()
            print(f"🌈 NeoPixels: {color}")
        except Exception as e:
            print(f"❌ NeoPixel color error: {e}")

    def spinning_dot(self, color='cyan', delay=0.08):
        """Single dot spinning animation"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['cyan'])
            
        try:
            while self.animation_active:
                for i in range(self.neopixel_count):
                    if not self.animation_active:
                        break
                    self.pixels.fill((0, 0, 0))
                    self.pixels[i] = color
                    self.pixels.show()
                    time.sleep(delay)
        except Exception as e:
            print(f"❌ Animation error: {e}")

    def rainbow_cycle(self, delay=0.05):
        """Rainbow animation"""
        if not self.pixels:
            return
            
        def wheel(pos):
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
            print(f"❌ Rainbow error: {e}")

    def pulse_color(self, color='green', steps=20, delay=0.05):
        """Pulsing effect"""
        if not self.pixels:
            return
            
        if isinstance(color, str):
            color = self.colors.get(color, self.colors['green'])
            
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
            print(f"❌ Pulse error: {e}")

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
        """Set LED mode with appropriate patterns"""
        self.current_mode = mode
        print(f"🎨 LED Mode: {mode.value}")
        
        if mode == LEDMode.OFF:
            self.stop_animation()
            self.set_solid_color('off')
            self.blue_off()
            
        elif mode == LEDMode.IDLE:
            self.stop_animation()
            self.set_solid_color('dim_white')
            self.blue_on()
            
        elif mode == LEDMode.SEARCHING:
            self.blue_on()
            self.start_animation(self.spinning_dot, 'cyan', 0.08)
            
        elif mode == LEDMode.DOG_DETECTED:
            self.blue_on()
            self.start_animation(self.pulse_color, 'green', 15, 0.03)
            
        elif mode == LEDMode.TREAT_LAUNCHING:
            self.blue_on()
            self.start_animation(self.rainbow_cycle, 0.05)
            
        elif mode == LEDMode.ERROR:
            self.blue_off()
            self.start_animation(self.pulse_color, 'red', 10, 0.1)
            
        elif mode == LEDMode.CHARGING:
            self.blue_on()  # Keep blue on for charging
            self.start_animation(self.pulse_color, 'orange', 20, 0.08)

    def status(self):
        """Show current status"""
        print(f"🔵 Blue LED: {'ON' if self.blue_is_on else 'OFF'}")
        print(f"🌈 Current mode: {self.current_mode.value}")
        print(f"🎬 Animation active: {self.animation_active}")
        if self.pixels:
            print(f"🔆 NeoPixel brightness: {self.pixels.brightness}")

    def cleanup(self):
        """Clean shutdown"""
        print("🧹 Cleaning up LED system...")
        self.stop_animation()
        if self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
        self.blue_off()
        if self.blue_chip:
            lgpio.gpiochip_close(self.blue_chip)

def interactive_test():
    """Clean interactive test"""
    print("🎮 Clean LED Controller Test")
    print("Commands:")
    print("  blue_on          - Turn blue LED on")
    print("  blue_off         - Turn blue LED off")
    print("  neo [color]      - Set NeoPixel color")
    print("  neo_brightness [0.1-1.0] - Set NeoPixel brightness")
    print("  mode [mode]      - Set LED mode")
    print("  stop             - Stop animations")
    print("  status           - Show status")
    print("  quit             - Exit")
    print(f"Available modes: {[m.value for m in LEDMode]}")
    
    leds = CleanLEDController()
    
    try:
        while True:
            cmd = input("\nCommand: ").strip().lower().split()
            if not cmd:
                continue
                
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'blue_on':
                leds.blue_on()
            elif cmd[0] == 'blue_off':
                leds.blue_off()
            elif cmd[0] == 'neo':
                if len(cmd) > 1:
                    color = cmd[1]
                    leds.set_solid_color(color)
                else:
                    leds.set_solid_color('blue')
            elif cmd[0] == 'neo_brightness':
                if len(cmd) > 1:
                    try:
                        level = float(cmd[1])
                        leds.set_neopixel_brightness(level)
                    except ValueError:
                        print("Invalid brightness. Use 0.1 to 1.0")
                else:
                    print("Usage: neo_brightness [0.1-1.0]")
            elif cmd[0] == 'mode':
                if len(cmd) > 1:
                    try:
                        mode = LEDMode(cmd[1])
                        leds.set_mode(mode)
                    except ValueError:
                        print(f"Invalid mode. Available: {[m.value for m in LEDMode]}")
                else:
                    print("Usage: mode [mode_name]")
            elif cmd[0] == 'stop':
                leds.stop_animation()
                print("🛑 Animations stopped")
            elif cmd[0] == 'status':
                leds.status()
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    finally:
        leds.cleanup()

if __name__ == "__main__":
    interactive_test()
