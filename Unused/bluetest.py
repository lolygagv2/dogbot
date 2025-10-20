#!/usr/bin/env python3
"""
Simple Blue LED Test - Debug the blue LED control issue
No complex variable names, just basic on/off control
"""

import lgpio
import time

class SimpleBlueLED:
    def __init__(self, pin=25):
        self.pin = pin
        self.chip = None
        
        try:
            self.chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.chip, self.pin, lgpio.SET_PULL_NONE)
            # Start with LED OFF
            lgpio.gpio_write(self.chip, self.pin, 0)
            print(f"✅ Blue LED controller initialized on GPIO{self.pin}")
            print("🔵 Blue LED: Starting OFF")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
    
    def turn_on(self):
        """Turn LED on (full voltage)"""
        if not self.chip:
            return
        try:
            lgpio.gpio_write(self.chip, self.pin, 1)
            print("🔵 Blue LED: ON")
        except Exception as e:
            print(f"❌ Error turning on: {e}")
    
    def turn_off(self):
        """Turn LED off with aggressive methods"""
        if not self.chip:
            return
        try:
            # Method 1: Simple write
            lgpio.gpio_write(self.chip, self.pin, 0)
            print("🔵 Blue LED: OFF (attempt 1)")
            time.sleep(0.1)
            
            # Method 2: Reclaim and write
            lgpio.gpio_free(self.chip, self.pin)
            lgpio.gpio_claim_output(self.chip, self.pin, lgpio.SET_PULL_DOWN)
            lgpio.gpio_write(self.chip, self.pin, 0)
            print("🔵 Blue LED: OFF (attempt 2 - reclaimed)")
            time.sleep(0.1)
            
            # Method 3: Multiple writes
            for i in range(3):
                lgpio.gpio_write(self.chip, self.pin, 0)
                time.sleep(0.05)
            print("🔵 Blue LED: OFF (attempt 3 - multiple writes)")
            
        except Exception as e:
            print(f"❌ Error turning off: {e}")
    
    def cleanup(self):
        """Clean shutdown"""
        self.turn_off()
        if self.chip:
            lgpio.gpiochip_close(self.chip)
        print("🧹 Cleanup complete")

def interactive_test():
    """Simple interactive test"""
    print("🔵 Simple Blue LED Test")
    print("Commands: on, off, quit")
    
    led = SimpleBlueLED(25)
    
    try:
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'on':
                led.turn_on()
            elif cmd == 'off':
                led.turn_off()
            else:
                print("Unknown command. Use: on, off, quit")
                
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    finally:
        led.cleanup()

if __name__ == "__main__":
    interactive_test()
