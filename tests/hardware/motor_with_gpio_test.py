#!/usr/bin/env python3
"""
Test motors while reading GPIO states to see if encoders respond
"""
import subprocess
import time
import threading
from gpiozero import OutputDevice, PWMOutputDevice

class MotorGPIOTest:
    def __init__(self):
        # Motor control pins (from hardware specs)
        self.left_in1 = OutputDevice(17)  # LEFT motor
        self.left_in2 = OutputDevice(18)
        self.left_ena = PWMOutputDevice(13)

        self.right_in3 = OutputDevice(27)  # RIGHT motor
        self.right_in4 = OutputDevice(22)
        self.right_enb = PWMOutputDevice(19)

        # Encoder pins
        self.encoder_pins = {'L_A': 4, 'L_B': 23, 'R_A': 5, 'R_B': 6}

        self.gpio_running = False
        self.gpio_thread = None

    def read_gpio(self, pin):
        try:
            result = subprocess.run(['gpioget', 'gpiochip0', str(pin)],
                                   capture_output=True, text=True, timeout=0.01)
            return int(result.stdout.strip()) if result.returncode == 0 else -1
        except:
            return -1

    def monitor_gpio(self):
        """Monitor GPIO states while motors run"""
        print("Time | L_A L_B | R_A R_B | Notes")
        print("-" * 40)

        start_time = time.time()
        last_states = None
        change_count = 0

        while self.gpio_running:
            elapsed = time.time() - start_time
            states = {name: self.read_gpio(pin) for name, pin in self.encoder_pins.items()}

            # Check for changes
            if last_states and states != last_states:
                change_count += 1
                note = f"CHANGE #{change_count}"
            else:
                note = ""

            print(f"{elapsed:4.1f} | {states['L_A']:2d} {states['L_B']:2d} | {states['R_A']:2d} {states['R_B']:2d} | {note}")

            last_states = states.copy()
            time.sleep(0.1)

        print(f"\nTotal GPIO changes detected: {change_count}")

    def run_left_motor_test(self):
        print("\n🔴 === LEFT MOTOR TEST (50% PWM, 3 seconds) ===")

        # Start GPIO monitoring
        self.gpio_running = True
        self.gpio_thread = threading.Thread(target=self.monitor_gpio, daemon=True)
        self.gpio_thread.start()

        time.sleep(0.5)  # Let monitoring start

        try:
            print("Starting LEFT motor...")
            # LEFT motor forward at 50%
            self.left_in1.off()
            self.left_in2.on()
            self.left_ena.value = 0.5

            time.sleep(3.0)  # Run for 3 seconds

            print("Stopping LEFT motor...")
            self.left_ena.value = 0
            self.left_in2.off()

        finally:
            self.gpio_running = False
            if self.gpio_thread:
                self.gpio_thread.join(timeout=1.0)

    def run_right_motor_test(self):
        print("\n🔵 === RIGHT MOTOR TEST (50% PWM, 3 seconds) ===")

        # Start GPIO monitoring
        self.gpio_running = True
        self.gpio_thread = threading.Thread(target=self.monitor_gpio, daemon=True)
        self.gpio_thread.start()

        time.sleep(0.5)  # Let monitoring start

        try:
            print("Starting RIGHT motor...")
            # RIGHT motor forward at 50%
            self.right_in3.on()
            self.right_in4.off()
            self.right_enb.value = 0.5

            time.sleep(3.0)  # Run for 3 seconds

            print("Stopping RIGHT motor...")
            self.right_enb.value = 0
            self.right_in3.off()

        finally:
            self.gpio_running = False
            if self.gpio_thread:
                self.gpio_thread.join(timeout=1.0)

    def cleanup(self):
        """Stop all motors"""
        try:
            self.left_ena.value = 0
            self.left_in1.off()
            self.left_in2.off()
            self.right_enb.value = 0
            self.right_in3.off()
            self.right_in4.off()
            print("All motors stopped")
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    print("=== MOTOR + GPIO ENCODER TEST ===")
    print("This will run each motor while monitoring encoder GPIO pins")
    print("If encoders work, you should see GPIO state changes while motors run")
    print()

    tester = MotorGPIOTest()

    try:
        tester.run_left_motor_test()
        time.sleep(1)
        tester.run_right_motor_test()

    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        tester.cleanup()
        print("Test complete")