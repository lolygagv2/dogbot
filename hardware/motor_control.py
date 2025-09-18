
cat > hardware/motor_control.py << 'EOF'

 Day 1-2: Test existing Devastator motor control
# Using your documented L298N wiring

import RPi.GPIO as GPIO
import time

# Your documented pin assignments
MOTOR_PINS = {
    'IN1': 17,
    'IN2': 18, 
    'IN3': 27,
    'IN4': 22,
    'ENA': 13,
    'ENB': 19
}

# Power relay control (from your docs)
RELAY_PIN = 37  # GPIO37 for DPDT relay

class TreatBotMotors:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        for pin in MOTOR_PINS.values():
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        
        self.pwm_a = GPIO.PWM(MOTOR_PINS['ENA'], 100)
        self.pwm_b = GPIO.PWM(MOTOR_PINS['ENB'], 100)
        self.pwm_a.start(0)
        self.pwm_b.start(0)
    
    def move_forward(self, speed=75, duration=1):
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)
        GPIO.output([MOTOR_PINS['IN1'], MOTOR_PINS['IN3']], GPIO.HIGH)
        GPIO.output([MOTOR_PINS['IN2'], MOTOR_PINS['IN4']], GPIO.LOW)
        time.sleep(duration)
        self.stop()
    
    def rotate_right(self, speed=50, duration=0.5):
        # Tank rotation
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)
        GPIO.output([MOTOR_PINS['IN1'], MOTOR_PINS['IN4']], GPIO.HIGH)
        GPIO.output([MOTOR_PINS['IN2'], MOTOR_PINS['IN3']], GPIO.LOW)
        time.sleep(duration)
        self.stop()
    
    def stop(self):
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
