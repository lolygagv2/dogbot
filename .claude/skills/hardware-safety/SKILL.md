---
name: hardware-safety
description: Hardware safety rules for the WIM-Z robot. Automatically relevant when modifying motor control, servo code, power management, GPIO pins, or any hardware interface code.
---

# WIM-Z Hardware Safety Rules

## CRITICAL — Read Before Any Hardware Code Changes

### Motor Safety
- **Max PWM:** 75% duty cycle (9V effective) — ABSOLUTE MAXIMUM
- **Safe range:** 40-70% duty cycle
- **NEVER use 100% duty cycle** — would supply 12.6V to 6V-rated motors
- **Rate limiting:** 50ms minimum between motor commands
- **Always implement gradual ramp-up/down** — no instant full-speed transitions

### Servo Safety (PCA9685)
- **Channel 0:** Camera Pan (0°-270°, center=140°)
- **Channel 1:** Camera Tilt (20°-200°, horizon=50°)
- **Channel 2:** Treat Carousel — CONTINUOUS ROTATION servo
  - Use `pulse_to_duty(1700)` for dispensing, NOT `kit.servo[2].angle`
  - Duration: 50ms per treat
  - Always use `stop_carousel(gradual=True)` — abrupt stop causes screech

### Power System
- **Battery:** 4S2P Li-ion, 14.8V nominal (12V min, 16.8V max)
- **Pi power:** 5V @ 5A via buck converter — never draw from GPIO 5V pin
- **NeoPixel power:** Separate 5V buck converter — not shared with Pi

### GPIO Rules
- **Pin 10 (GPIO10/MOSI):** NeoPixel signal — do not reassign
- **Pin 25 (GPIO25):** Blue LED MOSFET — do not reassign
- **Pins 7,16,23,29,31:** Encoder pins — do not reassign
- **Pins 11,12,13,15,33,35:** Motor control PWM — do not reassign
- **Pins 3,5:** I2C (PCA9685) — do not reassign

### Before Deploying Hardware Changes
1. Review GPIO pin assignments against the mapping in hardware_specs.md
2. Verify PWM duty cycle is within safe range
3. Test servo movements with small increments first
4. Check battery voltage before running motor tests
5. Have physical access to the power switch during testing
