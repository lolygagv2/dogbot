import RPi.GPIO as GPIO
import time

IR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.IN)

print("✅ IR Sensor test running. Waiting for signal...")
try:
    while True:
        if GPIO.input(IR_PIN) == 0:
            print("📡 IR signal detected!")
        time.sleep(0.5)
except KeyboardInterrupt:
    GPIO.cleanup()