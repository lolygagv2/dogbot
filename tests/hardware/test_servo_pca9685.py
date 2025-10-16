from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

print("✅ PCA9685 detected. Sweeping servo on channel 0...")
for angle in range(0, 180, 10):
    kit.servo[0].angle = angle
    time.sleep(0.1)
for angle in range(180, 0, -10):
    kit.servo[0].angle = angle
    time.sleep(0.1)
print("✅ Sweep complete.")