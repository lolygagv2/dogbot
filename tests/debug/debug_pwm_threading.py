#!/usr/bin/env python3
"""
Debug exactly what's wrong with PWM threading
"""
import sys
import os
import time
import threading
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pwm_thread_directly():
    """Test PWM threading in isolation"""
    print("=== PWM THREADING DEBUG ===")

    pwm_running = {}
    pwm_threads = {}

    def run_gpio_command(pin, value):
        try:
            cmd = ['gpioset', 'gpiochip0', f'{pin}={value}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.1)
            if result.returncode == 0:
                print(f"    GPIO{pin}={value} SUCCESS")
                return True
            else:
                print(f"    GPIO{pin}={value} FAILED: {result.stderr}")
                return False
        except Exception as e:
            print(f"    GPIO{pin}={value} ERROR: {e}")
            return False

    def pwm_loop(pin, duty_cycle):
        """Simple PWM loop"""
        frequency = 100
        period = 1.0 / frequency
        on_time = period * (duty_cycle / 100.0)
        off_time = period - on_time

        print(f"PWM Thread GPIO{pin}: duty={duty_cycle}%, on={on_time:.4f}s, off={off_time:.4f}s")

        cycle_count = 0
        while pwm_running.get(pin, False) and cycle_count < 500:  # 5 seconds max
            try:
                # On phase
                success1 = run_gpio_command(pin, 1)
                time.sleep(on_time)

                # Off phase
                if pwm_running.get(pin, False):
                    success2 = run_gpio_command(pin, 0)
                    time.sleep(off_time)

                cycle_count += 1
                if cycle_count % 50 == 0:
                    print(f"    GPIO{pin} completed {cycle_count} cycles")

                if not success1 or not success2:
                    print(f"    GPIO{pin} GPIO COMMANDS FAILING!")
                    break

            except Exception as e:
                print(f"    GPIO{pin} PWM loop error: {e}")
                break

        # Final cleanup
        run_gpio_command(pin, 0)
        print(f"PWM Thread GPIO{pin}: STOPPED after {cycle_count} cycles")

    # Test PWM on right motor enable pin (GPIO19)
    pin = 19
    duty = 50

    print(f"\nStarting PWM thread on GPIO{pin} at {duty}%...")
    pwm_running[pin] = True
    thread = threading.Thread(target=pwm_loop, args=(pin, duty), daemon=False)
    thread.start()
    pwm_threads[pin] = thread

    print("PWM thread started, waiting 3 seconds...")
    time.sleep(3)

    print("Stopping PWM thread...")
    pwm_running[pin] = False
    thread.join(timeout=2)

    if thread.is_alive():
        print("ERROR: Thread did not stop properly!")
    else:
        print("PWM thread stopped successfully")

    print(f"\nFinal GPIO{pin} state:")
    run_gpio_command(pin, 0)

if __name__ == "__main__":
    test_pwm_thread_directly()