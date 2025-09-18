# === Virtual environment patch for Thonny ===
import sys
sys.path.insert(0, '/home/morgan/dogbot/')
import thonny_venv_patch

# === Diagnostic test for PCA9685 and servo control ===
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# Setup I2C
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Channel 0 = first servo output
servo = pca.channels[0]

def pulse_to_duty(pulse_us):
    # pulse_us: 1000 = 1ms, 1500 = 1.5ms, 2000 = 2ms
    duty = int((pulse_us / 20000.0) * 0xFFFF)
    return duty

try:
    print("Moving to center (1500us)...")
    dc = pulse_to_duty(1500)
    print(f"Calculated duty cycle: {dc} / {0xFFFF}")
    servo.duty_cycle = dc
    time.sleep(5)

    print("Sweeping servo...")
    for pulse in [1000, 1500, 2000, 1500]:
        dc = pulse_to_duty(pulse)
        print(f"Pulse: {pulse}us → Duty: {dc}")
        servo.duty_cycle = dc
        time.sleep(1)

    print("Done.")

except KeyboardInterrupt:
    print("Stopping...")
    servo.duty_cycle = 0
