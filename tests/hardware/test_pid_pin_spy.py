#!/usr/bin/env python3
"""
PID Pin Configuration Spy
Monitors what GPIO pin configurations the PID system actually sends
when it asks for forward/backward motion
"""

import time
import lgpio
from core.hardware.proper_pid_motor_controller import ProperPIDMotorController

# Motor control pins to monitor
LEFT_IN1 = 17
LEFT_IN2 = 18
LEFT_ENA = 13
RIGHT_IN3 = 27
RIGHT_IN4 = 22
RIGHT_ENB = 19

class PIDPinSpy:
    def __init__(self):
        # GPIO for monitoring pin states
        self.gpio_handle = lgpio.gpiochip_open(0)
        print("🕵️ PID Pin Configuration Spy Initialized")
        print(f"   Monitoring: IN1={LEFT_IN1}, IN2={LEFT_IN2}, IN3={RIGHT_IN3}, IN4={RIGHT_IN4}")

    def read_motor_pins(self):
        """Read current state of all motor direction pins"""
        in1 = lgpio.gpio_read(self.gpio_handle, LEFT_IN1)
        in2 = lgpio.gpio_read(self.gpio_handle, LEFT_IN2)
        in3 = lgpio.gpio_read(self.gpio_handle, RIGHT_IN3)
        in4 = lgpio.gpio_read(self.gpio_handle, RIGHT_IN4)
        return in1, in2, in3, in4

    def interpret_pin_config(self, in1, in2, in3, in4):
        """Interpret what motor directions the pin states represent"""
        # Left motor interpretation
        if in1 == 0 and in2 == 0:
            left_dir = "STOP"
        elif in1 == 1 and in2 == 0:
            left_dir = "FORWARD"  # Based on our test results
        elif in1 == 0 and in2 == 1:
            left_dir = "BACKWARD" # Based on our test results
        else:
            left_dir = "BRAKE/ERROR"

        # Right motor interpretation
        if in3 == 0 and in4 == 0:
            right_dir = "STOP"
        elif in3 == 0 and in4 == 1:
            right_dir = "FORWARD"  # Based on our test results
        elif in3 == 1 and in4 == 0:
            right_dir = "BACKWARD" # Based on our test results
        else:
            right_dir = "BRAKE/ERROR"

        return left_dir, right_dir

    def spy_on_pid_system(self):
        """Monitor what the PID system actually sends to motors"""
        print("\n🔍 SPYING ON PID SYSTEM")
        print("=" * 50)
        print("Starting PID controller and monitoring actual GPIO pin states...")

        # Create PID controller
        controller = ProperPIDMotorController()

        try:
            # Start PID controller
            controller.start()
            time.sleep(1)

            # Initial state
            in1, in2, in3, in4 = self.read_motor_pins()
            left_dir, right_dir = self.interpret_pin_config(in1, in2, in3, in4)
            print(f"\nInitial state: IN1={in1} IN2={in2} IN3={in3} IN4={in4}")
            print(f"Interpreted: Left={left_dir}, Right={right_dir}")

            # Test 1: Ask for small forward motion
            print(f"\n🧪 TEST: PID requesting +10 RPM FORWARD for both motors")
            controller.set_motor_rpm(10, 10)

            # Monitor for 3 seconds
            for i in range(6):
                time.sleep(0.5)
                in1, in2, in3, in4 = self.read_motor_pins()
                left_dir, right_dir = self.interpret_pin_config(in1, in2, in3, in4)

                # Get PID status
                status = controller.get_status()
                left_target = status['targets']['ramped_left']
                right_target = status['targets']['ramped_right']
                left_actual = status['actual']['left_rpm']
                right_actual = status['actual']['right_rpm']

                print(f"   {(i+1)*0.5:.1f}s: Pins IN1={in1} IN2={in2} IN3={in3} IN4={in4} | "
                      f"Dirs L={left_dir:8s} R={right_dir:8s} | "
                      f"Target L={left_target:5.1f}R={right_target:5.1f} | "
                      f"Actual L={left_actual:6.1f}R={right_actual:6.1f}")

                # Safety check
                if abs(left_actual) > 50 and left_target > 0 and left_actual < 0:
                    print("   ⚠️ LEFT MOTOR DIRECTION MISMATCH DETECTED!")
                    break

            # Stop motors
            print(f"\n🛑 Stopping motors...")
            controller.set_motor_rpm(0, 0)
            time.sleep(1)

            # Final state
            in1, in2, in3, in4 = self.read_motor_pins()
            left_dir, right_dir = self.interpret_pin_config(in1, in2, in3, in4)
            print(f"Final state: IN1={in1} IN2={in2} IN3={in3} IN4={in4}")
            print(f"Interpreted: Left={left_dir}, Right={right_dir}")

            # Analysis
            print(f"\n📊 ANALYSIS:")
            print(f"When PID requested +10 RPM FORWARD:")
            if left_dir == "FORWARD":
                print(f"   ✅ Left motor: PID correctly sent FORWARD pin configuration")
            elif left_dir == "BACKWARD":
                print(f"   ❌ Left motor: PID sent BACKWARD pin config for FORWARD request!")
            else:
                print(f"   ⚠️ Left motor: Unexpected pin configuration: {left_dir}")

            if right_dir == "FORWARD":
                print(f"   ✅ Right motor: PID correctly sent FORWARD pin configuration")
            elif right_dir == "BACKWARD":
                print(f"   ❌ Right motor: PID sent BACKWARD pin config for FORWARD request!")
            else:
                print(f"   ⚠️ Right motor: Unexpected pin configuration: {right_dir}")

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            controller.set_motor_rpm(0, 0)
            controller.stop()
            controller.cleanup()
            lgpio.gpiochip_close(self.gpio_handle)

def main():
    print("🕵️ PID PIN CONFIGURATION SPY")
    print("=" * 50)
    print("This test monitors the actual GPIO pin states")
    print("that the PID system sends to the motors")
    print("to see if there's a direction mapping bug")
    print("=" * 50)

    spy = PIDPinSpy()
    spy.spy_on_pid_system()

if __name__ == "__main__":
    main()