# Encoder Hardware Debug Notes

## Issue Summary
**Date:** December 16, 2025
**Problem:** Both motor encoders return 0 ticks despite motors spinning at 50% PWM

## Pin Assignments (VERIFIED CORRECT)
**Hardware Specs Match Code:**
- **Left Motor:** A1=GPIO4 (Pin 7), B1=GPIO23 (Pin 16)
- **Right Motor:** A2=GPIO5 (Pin 29), B2=GPIO6 (Pin 31)

## Diagnostic Results
**Date:** December 16, 2025 (Updated)
**Test:** Direct PWM motor control (50% duty cycle for 3 seconds each)
**Expected:** ~1000 ticks (660 ticks/rev × ~3 revolutions)

**ACTUAL RESULTS:**
- **LEFT ENCODER:** ✅ **WORKING** - 579 ticks, 720 state changes, 17.5 RPM
- **RIGHT ENCODER:** ❌ **FAILED** - 0 ticks, 0 state changes, 0.0 RPM

**Problem Isolated:** Right motor encoder wiring/power issue on GPIO5/6

## Hardware Debug Steps Required

### 1. Check Encoder Power Supply
```bash
# Test encoder VCC lines with multimeter
# Should be 3.3V or 5V on encoder red wires
```

### 2. Test GPIO Pin Connectivity
```bash
# Test if GPIO pins can read high/low manually
gpioget gpiochip0 4   # Left A
gpioget gpiochip0 23  # Left B
gpioget gpiochip0 5   # Right A
gpioget gpiochip0 6   # Right B
```

### 3. Check Encoder Wiring
**Expected wiring per DFRobot motor specs:**
- **Red:** VCC (3.3V or 5V)
- **Black:** GND
- **Green:** Channel A (to GPIO4/5)
- **Yellow:** Channel B (to GPIO23/6)

### 4. Test Encoder Signal with Oscilloscope/Multimeter
- Manually rotate motor shaft
- Check for voltage changes on encoder outputs
- Should see 0-3.3V square waves

### 5. Verify Motor Encoder Type
- Confirm DFRobot motors have quadrature encoders
- Check if encoders need pull-up resistors
- Verify encoder PPR (pulses per revolution)

## Previous Working Evidence
**From PID logs:** Occasionally saw 1-10 RPM readings, suggesting encoders worked intermittently.
**Problem:** Readings dropped to 0.0 RPM mid-session, indicating loose connections or power issues.

## Current Status
**Motors:** Both working fine (respond to PWM control)
**Left Encoder:** ✅ Working perfectly (579 ticks detected)
**Right Encoder:** ❌ Hardware failure (0 ticks detected)
**PID System:** Can work with left motor only, right motor needs encoder fix

## Next Actions - RIGHT MOTOR ENCODER ONLY
1. **Check RIGHT motor encoder power** - verify 3.3V on red wire (pin 29/31 area)
2. **Check RIGHT encoder wiring** - green wire to GPIO5, yellow wire to GPIO6
3. **Test RIGHT encoder continuity** - multimeter from motor to Pi GPIO pins
4. **Verify RIGHT encoder mechanical coupling** - manually rotate right motor shaft

**Note:** LEFT encoder working perfectly, no action needed

## Diagnostic Script Location
`/home/morgan/dogbot/encoder_diagnostic.py`

**Usage:**
```bash
python encoder_diagnostic.py
```

**Safe Testing:** Script uses direct PWM control with NO PID interference.