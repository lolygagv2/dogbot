# IR Sensor Docking System Guide
*Roomba-Inspired Autonomous Charging*

## 🎯 Overview

Your question: "How can I use IR sensors for docking?"

**Answer:** Use the same proven approach Roomba has used for 20+ years:
- IR beacon on charging dock transmits 360° signal
- Multiple IR receivers on robot triangulate position
- Software guides robot to precise docking alignment

---

## 🔧 Hardware Components

### On the Charging Dock

#### IR Transmitter (Beacon)
**Recommended:** TSAL6200 or similar 940nm IR LED

**Circuit:**
```
                    +12V
                     |
                    [R] 10Ω (current limiting)
                     |
    GPIO ----[NPN]---|< TSAL6200 IR LED
    (38kHz PWM)      |
                    GND
```

**Key Features:**
- **Wavelength:** 940nm (invisible, won't bother dogs/humans)
- **Modulation:** 38kHz carrier (standard IR protocol)
- **Pattern:** Omnidirectional broadcast (360°)
- **Range:** 3-5 meters
- **Power:** 100mA pulses (safe for continuous operation)

**Beacon Pattern (Roomba-style):**
```python
# Transmit pattern: 500ms ON, 250ms OFF, repeat
while True:
    transmit_38khz_pulse(duration=500)  # "Dock here!" signal
    sleep(250)
```

---

### On the Robot (TreatBot)

#### IR Receivers (3-4 units)
**Recommended:** TSOP38238 or TSOP4838 (38kHz demodulator)

**Placement Strategy:**
```
       [Front]
         IR1
          |
   IR2 --[●]-- IR3    (120° apart)
          |
      [Rear IR4]
     (optional)
```

**Why 3-4 receivers?**
- **3 minimum:** Triangulation (left, center, right)
- **4 better:** Full 360° coverage, rear docking support

**Wiring (per receiver):**
```
TSOP38238:
  Pin 1 (OUT) → Raspberry Pi GPIO (e.g., GPIO 22, 23, 24)
  Pin 2 (GND) → GND
  Pin 3 (VCC) → 3.3V or 5V
```

---

## 📡 Communication Protocol

### Signal Encoding (38kHz Carrier)

**Why 38kHz?**
- Standard for consumer IR (TV remotes, etc.)
- TSOP receivers have built-in 38kHz bandpass filter
- Rejects ambient light, sunlight, fluorescent lights

**Transmitter Code (Dock):**
```python
import RPi.GPIO as GPIO
import time

BEACON_PIN = 18  # GPIO 18 (PWM capable)

def setup_beacon():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BEACON_PIN, GPIO.OUT)
    pwm = GPIO.PWM(BEACON_PIN, 38000)  # 38kHz carrier
    return pwm

def transmit_beacon():
    pwm = setup_beacon()
    while True:
        pwm.start(50)  # 50% duty cycle
        time.sleep(0.5)  # ON for 500ms
        pwm.stop()
        time.sleep(0.25)  # OFF for 250ms

# Run continuously
transmit_beacon()
```

**Receiver Code (Robot):**
```python
import RPi.GPIO as GPIO

IR_LEFT = 22    # GPIO 22
IR_CENTER = 23  # GPIO 23
IR_RIGHT = 24   # GPIO 24

def setup_receivers():
    GPIO.setmode(GPIO.BCM)
    for pin in [IR_LEFT, IR_CENTER, IR_RIGHT]:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def read_ir_sensors():
    """
    Returns: (left, center, right)
    True = beacon detected, False = no signal
    """
    return (
        not GPIO.input(IR_LEFT),    # TSOP outputs LOW when signal detected
        not GPIO.input(IR_CENTER),
        not GPIO.input(IR_RIGHT)
    )

# Example usage
setup_receivers()
left, center, right = read_ir_sensors()

if center and not left and not right:
    print("Beacon straight ahead!")
elif left and not right:
    print("Beacon to the left, turn left")
elif right and not left:
    print("Beacon to the right, turn right")
else:
    print("Searching for beacon...")
```

---

## 🧭 Navigation Algorithm

### Phase 1: Search Pattern
**Goal:** Find the beacon from anywhere in the room

```python
def search_for_dock():
    """
    Rotate 360° slowly until beacon detected
    """
    while True:
        left, center, right = read_ir_sensors()
        
        if any([left, center, right]):
            print("Beacon found!")
            return True
        
        # Rotate in place (slow turn)
        motors.rotate_left(speed=30)
        time.sleep(0.1)  # Check every 100ms
        
        # Timeout after 2 full rotations
        if rotations > 2:
            return False  # Beacon not found
```

### Phase 2: Approach
**Goal:** Drive toward beacon while staying aligned

```python
def approach_dock():
    """
    Navigate toward dock using IR feedback
    """
    while True:
        left, center, right = read_ir_sensors()
        distance = get_distance_to_dock()  # From ultrasonic or encoders
        
        if distance < 0.3:  # 30cm - switch to precision mode
            return precision_dock()
        
        # Steering logic (proportional control)
        if center:
            motors.forward(speed=40)  # Drive straight
        elif left and not center:
            motors.turn_left(speed=30)  # Gentle left
        elif right and not center:
            motors.turn_right(speed=30)  # Gentle right
        elif left and right:
            motors.forward(speed=40)  # Between beacons, go forward
        else:
            # Lost signal - stop and re-search
            motors.stop()
            return search_for_dock()
        
        time.sleep(0.05)  # 20Hz update rate
```

### Phase 3: Precision Docking
**Goal:** Final alignment (<5cm accuracy)

```python
def precision_dock():
    """
    Slow, precise final approach
    Uses all 3-4 IR sensors for alignment
    """
    while True:
        left, center, right = read_ir_sensors()
        distance = get_distance_to_dock()
        
        if distance < 0.05 and center:  # 5cm, aligned
            motors.stop()
            time.sleep(0.5)
            
            if verify_charging():  # Check pogo pins connected
                print("✅ Docked successfully!")
                return True
            else:
                # Not quite there, nudge forward
                motors.forward(speed=10)
                time.sleep(0.3)
        
        # Micro-adjustments
        if not center and left:
            motors.rotate_left(speed=10)  # Very slow turn
        elif not center and right:
            motors.rotate_right(speed=10)
        else:
            motors.forward(speed=15)  # Crawl forward
        
        time.sleep(0.1)
```

---

## 🏗️ Physical Dock Design

### Docking Station Requirements

**Structure:**
```
        [Pogo Pins]
         ↓↓↓↓↓
    ╔═══════════════╗
    ║   CHARGING    ║
    ║     DOCK      ║
    ║               ║
    ║  [IR Beacon]  ║  ← 360° IR LED
    ║               ║
    ╚═══════════════╝
       ↑ ↑ ↑
    Guide rails (optional)
```

**Key Features:**
1. **IR Beacon:** Mounted 10-15cm above ground (dog-eye level)
2. **Pogo Pins:** Spring-loaded charging contacts (5-6 pins for redundancy)
3. **Guide Rails:** Optional funnel to help final alignment
4. **Visual Markers:** Reflective tape for backup optical navigation

**Pogo Pin Layout:**
```
    + + GND GND - -
    [=============]  ← Charging contact strip
    
    + : Positive (16.8V charging)
    - : Negative (GND)
    GND: Extra ground pins for safety
```

---

## 🔍 Troubleshooting Common Issues

### Issue 1: Robot Can't Find Beacon
**Causes:**
- Beacon not transmitting (check GPIO output)
- IR receivers blocked (clean lenses)
- Ambient IR interference (fluorescent lights)

**Solutions:**
- Add beacon power LED indicator
- Shield receivers from side/top light
- Increase beacon pulse power

### Issue 2: Docking Accuracy Poor (<10cm)
**Causes:**
- Only using center receiver (no triangulation)
- Motors not precisely controlled (PWM resolution)
- Surface too slippery (wheels slip)

**Solutions:**
- Use all 3+ receivers for alignment
- Add encoders for precise movement
- Add rubber mat under dock

### Issue 3: False Docking Success
**Causes:**
- Not verifying charging connection
- IR signal bounces off walls (multipath)

**Solutions:**
- Check voltage on pogo pins (16.8V = connected)
- Add timeout: if no charge after 5s, retry
- Use directional IR receivers (narrow FOV)

---

## 📊 Performance Expectations

### Success Rates (Industry Standard)
- **Search & Find:** >95% (if beacon in range)
- **Approach:** >90% (with 3+ IR sensors)
- **Precision Dock:** >85% (first attempt)
- **Overall Success:** >80% (full autonomous cycle)

### Typical Timing
- **Search:** 5-30 seconds (depending on starting position)
- **Approach:** 10-20 seconds (from 3m)
- **Precision:** 5-10 seconds
- **Total:** 20-60 seconds average

---

## 🚀 Advanced Enhancements

### Multi-Dock Support
- Assign unique IR codes to each dock (like TV remotes)
- Robot selects nearest/preferred dock

### Obstacle Avoidance During Docking
- Combine IR docking with cliff/bumper sensors
- Abort docking if obstacle detected, retry

### Backup Navigation Methods
- **Dead Reckoning:** Record path, reverse it
- **Visual Markers:** ArUco tag on dock
- **WiFi RSSI:** Signal strength triangulation

---

## 📦 Bill of Materials (IR System)

| Component | Quantity | Purpose | Cost (Est.) |
|-----------|----------|---------|-------------|
| TSAL6200 IR LED | 1-2 | Beacon transmitter | $1-2 |
| TSOP38238 Receiver | 3-4 | Robot IR sensors | $3-5 |
| NPN Transistor (2N2222) | 1 | Beacon driver | $0.50 |
| 10Ω Resistor | 1 | Current limiting | $0.10 |
| Pogo Pins (spring) | 6-8 | Charging contacts | $5-10 |
| **Total** | - | - | **$10-18** |

---

## 🎯 Implementation Checklist

### Hardware
- [ ] Mount IR beacon on charging dock (10-15cm height)
- [ ] Install 3-4 IR receivers on robot (120° spacing)
- [ ] Wire beacon to dock power supply (12V)
- [ ] Connect receivers to Raspberry Pi GPIOs
- [ ] Add pogo pins to dock (6-pin layout)
- [ ] Add matching pogo contacts to robot underside

### Software
- [ ] Implement 38kHz PWM beacon transmission
- [ ] Create IR receiver polling function (20Hz)
- [ ] Write search algorithm (360° scan)
- [ ] Implement approach navigation (proportional steering)
- [ ] Add precision docking (micro-adjustments)
- [ ] Verify charging detection (voltage check)

### Testing
- [ ] Test beacon visibility (3m+ range)
- [ ] Calibrate IR receiver sensitivity
- [ ] Test docking from 10 different starting positions
- [ ] Measure success rate (target: >80%)
- [ ] Test with ambient light (day/night)
- [ ] Verify charging connection reliability

---

## 💡 Pro Tips

1. **Start Simple:** Test with manual IR remote first (TV remote)
2. **Debug Visually:** Add LEDs to show IR sensor states
3. **Log Everything:** Record approach paths for analysis
4. **Iterate:** First 5cm accuracy, then refine to 2cm
5. **Safety First:** Emergency stop if stuck for 30+ seconds

---

**Questions? Issues? Check:**
- Roomba hacking forums (years of community knowledge)
- iRobot patents (US6809490, US7155308 - docking systems)
- Arduino IR library documentation
- TreatBot project docs: `docs/navigation/`