#!/usr/bin/env python3
"""
Motor Direction and Encoder Test - Post Waterproofing
Tests motor directions and encoder functionality after waterproofing/wire cleanup

This test will help identify:
1. Correct IN1-IN4 pin mappings for actual motor directions
2. Whether encoder wheels now spin freely (no wire interference)
3. If encoder continuity is restored after wire cleanup
"""

import lgpio
import time
import threading
from gpiozero import OutputDevice, PWMOutputDevice

# Motor control pins
LEFT_IN1 = 17
LEFT_IN2 = 18
LEFT_ENA = 13
RIGHT_IN3 = 27
RIGHT_IN4 = 22
RIGHT_ENB = 19

# Encoder pins
LEFT_ENCODER_A = 4
LEFT_ENCODER_B = 23
RIGHT_ENCODER_A = 5
RIGHT_ENCODER_B = 6

class MotorDirectionTester:
    def __init__(self):
        # GPIO setup
        self.gpio_handle = lgpio.gpiochip_open(0)

        # Motor control
        self.left_in1 = OutputDevice(LEFT_IN1)
        self.left_in2 = OutputDevice(LEFT_IN2)
        self.left_ena = PWMOutputDevice(LEFT_ENA)
        self.right_in3 = OutputDevice(RIGHT_IN3)
        self.right_in4 = OutputDevice(RIGHT_IN4)
        self.right_enb = PWMOutputDevice(RIGHT_ENB)

        # Encoder setup
        lgpio.gpio_claim_input(self.gpio_handle, LEFT_ENCODER_A, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, LEFT_ENCODER_B, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, RIGHT_ENCODER_A, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, RIGHT_ENCODER_B, lgpio.SET_PULL_UP)

        # Encoder tracking
        self.left_count = 0
        self.right_count = 0
        self.left_last_a = 0
        self.left_last_b = 0
        self.right_last_a = 0
        self.right_last_b = 0
        self.monitoring = False

        print("🔧 Motor Direction & Encoder Tester Initialized")
        print(f"   Left Motor: IN1={LEFT_IN1}, IN2={LEFT_IN2}, ENA={LEFT_ENA}")
        print(f"   Right Motor: IN3={RIGHT_IN3}, IN4={RIGHT_IN4}, ENB={RIGHT_ENB}")
        print(f"   Left Encoder: A={LEFT_ENCODER_A}, B={LEFT_ENCODER_B}")
        print(f"   Right Encoder: A={RIGHT_ENCODER_A}, B={RIGHT_ENCODER_B}")

    def read_encoder_pins(self):
        """Read current encoder pin states"""
        left_a = lgpio.gpio_read(self.gpio_handle, LEFT_ENCODER_A)
        left_b = lgpio.gpio_read(self.gpio_handle, LEFT_ENCODER_B)
        right_a = lgpio.gpio_read(self.gpio_handle, RIGHT_ENCODER_A)
        right_b = lgpio.gpio_read(self.gpio_handle, RIGHT_ENCODER_B)
        return left_a, left_b, right_a, right_b

    def decode_encoder(self, motor, current_a, current_b, last_a, last_b):
        """Decode quadrature encoder"""
        count_change = 0
        if current_a != last_a or current_b != last_b:
            # A channel rising edge
            if last_a == 0 and current_a == 1:
                count_change = 1 if current_b == 0 else -1
            # A channel falling edge
            elif last_a == 1 and current_a == 0:
                count_change = 1 if current_b == 1 else -1
        return count_change

    def monitor_encoders(self, duration=3.0):
        """Monitor encoder counts during motor test"""
        print(f"🔍 Monitoring encoders for {duration} seconds...")

        # Reset counts and read initial states
        self.left_count = 0
        self.right_count = 0
        left_a, left_b, right_a, right_b = self.read_encoder_pins()
        self.left_last_a, self.left_last_b = left_a, left_b
        self.right_last_a, self.right_last_b = right_a, right_b

        print(f"   Initial states: L_A={left_a} L_B={left_b} | R_A={right_a} R_B={right_b}")

        start_time = time.time()
        self.monitoring = True

        while self.monitoring and (time.time() - start_time) < duration:
            # Read current states
            left_a, left_b, right_a, right_b = self.read_encoder_pins()

            # Decode left encoder
            left_change = self.decode_encoder('left', left_a, left_b,
                                            self.left_last_a, self.left_last_b)
            self.left_count += left_change
            self.left_last_a, self.left_last_b = left_a, left_b

            # Decode right encoder
            right_change = self.decode_encoder('right', right_a, right_b,
                                             self.right_last_a, self.right_last_b)
            self.right_count += right_change
            self.right_last_a, self.right_last_b = right_a, right_b

            # Brief delay
            time.sleep(0.001)  # 1ms polling

        print(f"✅ Monitoring complete: Left={self.left_count} counts, Right={self.right_count} counts")
        return self.left_count, self.right_count

    def stop_all_motors(self):
        """Emergency stop all motors"""
        self.monitoring = False
        self.left_in1.off()
        self.left_in2.off()
        self.left_ena.value = 0
        self.right_in3.off()
        self.right_in4.off()
        self.right_enb.value = 0
        print("🛑 All motors stopped")

    def test_motor_direction(self, motor, direction, pwm=40, duration=3):
        """Test specific motor direction and measure encoder response"""
        print(f"\n🧪 Testing {motor.upper()} motor {direction} at {pwm}% PWM...")

        # Stop all first
        self.stop_all_motors()
        time.sleep(0.5)

        # Start encoder monitoring in separate thread
        encoder_thread = threading.Thread(target=self.monitor_encoders, args=(duration,))
        encoder_thread.start()
        time.sleep(0.1)  # Let monitoring start

        # Apply motor command
        pwm_value = pwm / 100.0

        if motor == 'left':
            if direction == 'forward_config1':
                # Test configuration 1: IN1=0, IN2=1
                self.left_in1.off()
                self.left_in2.on()
                self.left_ena.value = pwm_value
                print(f"   Left Motor: IN1=0, IN2=1, PWM={pwm}%")
            elif direction == 'forward_config2':
                # Test configuration 2: IN1=1, IN2=0
                self.left_in1.on()
                self.left_in2.off()
                self.left_ena.value = pwm_value
                print(f"   Left Motor: IN1=1, IN2=0, PWM={pwm}%")

        elif motor == 'right':
            if direction == 'forward_config1':
                # Test configuration 1: IN3=0, IN4=1
                self.right_in3.off()
                self.right_in4.on()
                self.right_enb.value = pwm_value
                print(f"   Right Motor: IN3=0, IN4=1, PWM={pwm}%")
            elif direction == 'forward_config2':
                # Test configuration 2: IN3=1, IN4=0
                self.right_in3.on()
                self.right_in4.off()
                self.right_enb.value = pwm_value
                print(f"   Right Motor: IN3=1, IN4=0, PWM={pwm}%")

        # Wait for test to complete
        encoder_thread.join()

        # Stop motor
        self.stop_all_motors()

        # Report results
        if motor == 'left':
            encoder_count = self.left_count
        else:
            encoder_count = self.right_count

        print(f"📊 Result: {encoder_count} encoder counts in {duration}s")

        if abs(encoder_count) > 10:
            direction_detected = "FORWARD" if encoder_count > 0 else "BACKWARD"
            print(f"✅ Motor moved {direction_detected} with {abs(encoder_count)} counts")
            return direction_detected, abs(encoder_count)
        else:
            print(f"❌ No significant movement detected ({encoder_count} counts)")
            return "NO_MOVEMENT", 0

def main():
    print("🚀 POST-WATERPROOFING MOTOR & ENCODER TEST")
    print("=" * 60)
    print("This test will help identify:")
    print("1. Correct motor direction pin mappings")
    print("2. Whether encoder wheels spin freely now")
    print("3. If encoder wiring is working properly")
    print("=" * 60)

    tester = MotorDirectionTester()
    results = {}

    try:
        # Test 1: Check initial encoder pin states
        print("\n📍 PHASE 1: Initial Encoder Pin Check")
        left_a, left_b, right_a, right_b = tester.read_encoder_pins()
        print(f"Current encoder states: L_A={left_a} L_B={left_b} | R_A={right_a} R_B={right_b}")

        if right_a == 0 and right_b == 0:
            print("⚠️  Right encoder pins still reading 0,0 - may still have wiring issues")
        else:
            print("✅ Right encoder pins showing valid states!")

        # Test 2: Left motor direction mapping
        print("\n🔍 PHASE 2: Left Motor Direction Mapping")

        # Test left motor config 1
        direction, count = tester.test_motor_direction('left', 'forward_config1')
        results['left_config1'] = (direction, count)

        time.sleep(1)

        # Test left motor config 2
        direction, count = tester.test_motor_direction('left', 'forward_config2')
        results['left_config2'] = (direction, count)

        # Test 3: Right motor direction mapping
        print("\n🔍 PHASE 3: Right Motor Direction Mapping")

        time.sleep(1)

        # Test right motor config 1
        direction, count = tester.test_motor_direction('right', 'forward_config1')
        results['right_config1'] = (direction, count)

        time.sleep(1)

        # Test right motor config 2
        direction, count = tester.test_motor_direction('right', 'forward_config2')
        results['right_config2'] = (direction, count)

        # Analysis and recommendations
        print("\n📋 TEST RESULTS ANALYSIS")
        print("=" * 60)

        print("\nLeft Motor Results:")
        print(f"  Config 1 (IN1=0,IN2=1): {results['left_config1'][0]} - {results['left_config1'][1]} counts")
        print(f"  Config 2 (IN1=1,IN2=0): {results['left_config2'][0]} - {results['left_config2'][1]} counts")

        print("\nRight Motor Results:")
        print(f"  Config 1 (IN3=0,IN4=1): {results['right_config1'][0]} - {results['right_config1'][1]} counts")
        print(f"  Config 2 (IN3=1,IN4=0): {results['right_config2'][0]} - {results['right_config2'][1]} counts")

        print("\n🔧 MOTOR CONTROLLER CONFIGURATION:")

        # Determine best left motor config
        if results['left_config1'][1] > results['left_config2'][1]:
            print("Left Motor - Use Config 1:")
            print("  self.left_in1.off()   # IN1=0 for forward")
            print("  self.left_in2.on()    # IN2=1 for forward")
        else:
            print("Left Motor - Use Config 2:")
            print("  self.left_in1.on()    # IN1=1 for forward")
            print("  self.left_in2.off()   # IN2=0 for forward")

        # Determine best right motor config
        if results['right_config1'][1] > results['right_config2'][1]:
            print("Right Motor - Use Config 1:")
            print("  self.right_in3.off()  # IN3=0 for forward")
            print("  self.right_in4.on()   # IN4=1 for forward")
        else:
            print("Right Motor - Use Config 2:")
            print("  self.right_in3.on()   # IN3=1 for forward")
            print("  self.right_in4.off()  # IN4=0 for forward")

        # Encoder status
        left_working = max(results['left_config1'][1], results['left_config2'][1]) > 10
        right_working = max(results['right_config1'][1], results['right_config2'][1]) > 10

        print(f"\n📈 ENCODER STATUS:")
        print(f"  Left Encoder: {'✅ WORKING' if left_working else '❌ NOT WORKING'}")
        print(f"  Right Encoder: {'✅ WORKING' if right_working else '❌ NOT WORKING'}")

        if right_working:
            print("\n🎉 SUCCESS! Right encoder now working after wire cleanup!")
        else:
            print("\n⚠️  Right encoder still not working - may need further investigation")

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        tester.stop_all_motors()
        lgpio.gpiochip_close(tester.gpio_handle)
        print("\n✅ Test cleanup complete")

if __name__ == "__main__":
    main()