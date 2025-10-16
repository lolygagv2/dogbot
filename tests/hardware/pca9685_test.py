# === Virtual environment patch for Thonny ===
import sys
sys.path.insert(0, '/home/morgan/dogbot/')
import thonny_venv_patch

import time
import board
import busio
from adafruit_pca9685 import PCA9685

# === I2C + PCA Setup ===
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # Standard servo PWM frequency

# === Assign channels ===
pitch = pca.channels[0]  # Camera pitch
pan = pca.channels[1]    # Camera pan
winch = pca.channels[2]  # Continuous rotation

# === Utility ===
def pulse_to_duty(pulse_us):
    return int((pulse_us / 20000.0) * 0xFFFF)

# === Soft reset / unlock all servos ===
def release_servos():
    pitch.duty_cycle = 0
    pan.duty_cycle = 0
    winch.duty_cycle = 0
    print("All servos released")

# === Camera pitch test ===
def test_pitch():
    print("Camera pitch...")
    pitch.duty_cycle = pulse_to_duty(1000)  # Down
    time.sleep(1)
    pitch.duty_cycle = pulse_to_duty(1500)  # Neutral
    time.sleep(1)
    pitch.duty_cycle = pulse_to_duty(2000)  # Max up
    time.sleep(1)
    pitch.duty_cycle = 0  # Release
    print("Pitch done")

# === Camera pan test ===
def test_pan():
    print("Camera pan...")
    pan.duty_cycle = pulse_to_duty(1000)  # Left
    time.sleep(1)
    pan.duty_cycle = pulse_to_duty(1500)  # Center
    time.sleep(1)
    pan.duty_cycle = pulse_to_duty(1800)  # Right
    time.sleep(1)
    pan.duty_cycle = 0  # Release
    print("Pan done")

# === Winch burst test ===
def rotate_winch(direction='forward', duration=0.08):
    if direction == 'forward':
        winch.duty_cycle = pulse_to_duty(1700)
    else:
        winch.duty_cycle = pulse_to_duty(1300)
    time.sleep(duration)
    winch.duty_cycle = 0  # Stop

def test_winch_sequence():
    print("Winch: small bursts forward...")
    for _ in range(2):
        rotate_winch('forward', 0.12)
        time.sleep(0.3)
    print("Winch done")

# === Smooth camera tilt ===
def smooth_pitch_down():
    print("Smooth pitch down...")
    for pulse in range(1600, 1000, -10):
        pitch.duty_cycle = pulse_to_duty(pulse)
        time.sleep(0.02)
    pitch.duty_cycle = 0
    print("Smooth pitch done")

# === MAIN TEST SEQUENCE ===
try:
    release_servos()
    time.sleep(1)

    test_pitch()
    time.sleep(1)

    test_pan()
    time.sleep(1)

    test_winch_sequence()
    time.sleep(1)

    smooth_pitch_down()

    release_servos()

except KeyboardInterrupt:
    release_servos()
    print("Stopped by user.")