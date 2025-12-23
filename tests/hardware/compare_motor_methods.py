#!/usr/bin/env python3
"""
Compare motor controller vs direct gpiozero control
Goal: Understand why motor controller gets lower encoder counts
"""

import sys
import time
import logging
from pathlib import Path
from gpiozero import OutputDevice, PWMOutputDevice
import lgpio
import threading

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

# Motor pins (left motor)
MOTOR_IN1 = 17
MOTOR_IN2 = 18
MOTOR_ENA = 13

# Encoder pins (left motor)
ENCODER_A = 4
ENCODER_B = 23

def test_motor_controller_method():
    """Test using MotorControllerPolling class"""
    print("🏭 METHOD 1: MotorControllerPolling class")
    print("=========================================")

    controller = MotorControllerPolling()
    time.sleep(1)  # Let it initialize

    print("Running left motor at 50% for 3 seconds...")
    controller.reset_encoder_counts()
    controller.set_motor_speed('left', 50, 'forward')

    time.sleep(3)
    controller.stop()

    left_count, right_count = controller.get_encoder_counts()
    print(f"Motor Controller Result: {abs(left_count)} counts")

    controller.cleanup()
    return abs(left_count)

def test_direct_gpiozero_method():
    """Test using direct gpiozero control (like our successful test)"""
    print("\n⚡ METHOD 2: Direct gpiozero control")
    print("==================================")

    # Motor setup
    motor_in1 = OutputDevice(MOTOR_IN1)
    motor_in2 = OutputDevice(MOTOR_IN2)
    motor_ena = PWMOutputDevice(MOTOR_ENA)

    # Encoder setup
    gpio_handle = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(gpio_handle, ENCODER_A, lgpio.SET_PULL_UP)
    lgpio.gpio_claim_input(gpio_handle, ENCODER_B, lgpio.SET_PULL_UP)

    # State tracking
    count = 0
    changes = 0
    lock = threading.Lock()
    last_a = lgpio.gpio_read(gpio_handle, ENCODER_A)
    last_b = lgpio.gpio_read(gpio_handle, ENCODER_B)

    def poll_encoder():
        nonlocal count, changes, last_a, last_b
        current_a = lgpio.gpio_read(gpio_handle, ENCODER_A)
        current_b = lgpio.gpio_read(gpio_handle, ENCODER_B)

        with lock:
            if current_a != last_a or current_b != last_b:
                changes += 1

                # Same quadrature decoding as motor controller
                if last_a == 0 and current_a == 1:
                    if current_b == 0:
                        count += 1
                    else:
                        count -= 1
                elif last_a == 1 and current_a == 0:
                    if current_b == 1:
                        count += 1
                    else:
                        count -= 1

            last_a = current_a
            last_b = current_b

    print(f"Initial encoder states: A={last_a}, B={last_b}")
    print("Running left motor at 50% for 3 seconds...")

    # Start motor at 50% PWM (same as motor controller minimum)
    motor_in1.off()  # Forward direction
    motor_in2.on()
    motor_ena.value = 0.5  # 50% PWM

    # Poll encoder for 3 seconds
    start_time = time.time()
    while time.time() - start_time < 3:
        poll_encoder()
        time.sleep(0.0005)  # 2000Hz polling like motor controller

    # Stop motor
    motor_ena.value = 0
    motor_in1.off()
    motor_in2.off()

    with lock:
        final_count = abs(count)
        final_changes = changes

    print(f"Direct Control Result: {final_count} counts, {final_changes} changes")

    # Cleanup
    motor_ena.close()
    motor_in1.close()
    motor_in2.close()
    lgpio.gpiochip_close(gpio_handle)

    return final_count

def main():
    print("🔬 MOTOR CONTROL METHOD COMPARISON")
    print("=================================")
    print("Goal: Find why MotorControllerPolling gets lower counts than direct control")
    print()

    try:
        # Test motor controller method
        controller_counts = test_motor_controller_method()

        # Wait between tests
        time.sleep(2)

        # Test direct method
        direct_counts = test_direct_gpiozero_method()

        # Compare results
        print(f"\n🎯 COMPARISON RESULTS:")
        print(f"Motor Controller: {controller_counts} counts")
        print(f"Direct Control:   {direct_counts} counts")
        print(f"Difference:       {direct_counts - controller_counts} counts")
        print(f"Ratio:           {direct_counts/controller_counts:.1f}x" if controller_counts > 0 else "∞")

        if direct_counts > controller_counts * 2:
            print("\n❌ Motor Controller significantly underperforming")
            print("Possible causes:")
            print("1. PWM value difference (controller uses clamping)")
            print("2. Threading interference")
            print("3. Polling frequency mismatch")
            print("4. Motor direction control difference")
        elif abs(direct_counts - controller_counts) < 10:
            print("\n✅ Both methods performing similarly")
        else:
            print("\n⚠️ Moderate difference - may be normal variation")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()