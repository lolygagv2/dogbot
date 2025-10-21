# TreatBot Hardware Specifications

## 🔧 SERVO SYSTEM - 3 Servos Total

### ⚠️ CRITICAL: Treat Carousel (Continuous Rotation)
**Servo:** Hacked MG996R (originally 180° position servo)
**Modification:** Converted to continuous rotation servo
**Channel:** 2 (PCA9685)
**Function:** **TREAT DISPENSING** - rotates to dispense treats
**Control Method:** `kit.continuous_servo[2].throttle` (NOT `kit.servo[2].angle`)

**Treat Dispensing Specifications:**
- **Treat Positions:** 6 compartments around carousel
- **Rotation per treat:** 60° (360° ÷ 6 positions)
- **Dispensing method:** Brief rotation to advance to next treat slot
- **Duration:** 0.08 seconds per treat (80ms) - CALIBRATED
- **Pulse Width:** 1700μs (forward rotation) - CALIBRATED
- **Control:** `winch.duty_cycle = pulse_to_duty(1700)` for exactly 1 treat

**Safe Control:**
```python
# ✅ CORRECT - Treat Dispensing (CALIBRATED VALUES)
def dispense_one_treat():
    winch.duty_cycle = pulse_to_duty(1700)  # 1700μs pulse
    time.sleep(0.08)                        # 80ms duration
    controller.stop_carousel(gradual=True)  # Safe stop with ramp-down

# ❌ NEVER DO THIS
controller.set_servo_angle('carousel', angle)  # Will spin continuously!
kit.continuous_servo[2].throttle = 0.0  # Abrupt stop causes screech
```

### 📹 Camera Pan Servo
**Servo:** Standard position servo
**Channel:** 0 (PCA9685)
**Function:** Left/right camera movement
**Range:** -90° to +90°
**Control Method:** `kit.servo[0].angle`

### 📹 Camera Pitch Servo
**Servo:** Standard position servo
**Channel:** 1 (PCA9685)
**Function:** Up/down camera movement
**Range:** -45° to +45°
**Control Method:** `kit.servo[1].angle`

**⚠️ NO LAUNCHER SERVO EXISTS - Carousel handles all treat dispensing**
*Updated: October 2025 - Hardware Clarification*

## 🤖 System Overview

**Platform:** Autonomous Mobile Robot  
**Purpose:** AI-powered dog training and behavior reinforcement  
**Control:** Raspberry Pi 5 with Hailo-8 AI accelerator  

---

## 🧠 Core Computing

### Raspberry Pi 5
- **Model:** 8GB RAM variant
- **OS:** Raspberry Pi OS (64-bit)
- **Power:** 5V @ 5A via buck converter
- **Cooling:** Active heatsink + fan

### Hailo-8 AI Accelerator HAT
- **Performance:** 26 TOPS (Tera Operations Per Second)
- **Interface:** PCIe (via HAT)
- **Models:** YOLOv8s (dog detection + pose estimation)
- **Inference Speed:** 30+ FPS @ 640x640 resolution

---

## 📷 Vision System

### IMX500 AI Camera
- **Sensor:** Sony IMX500 (12.3MP)
- **Resolution:** 4056 x 3040 pixels
- **Interface:** CSI-2 (4-lane)
- **Features:** 
  - On-sensor AI processing
  - Built-in ISP
  - HDR support

### Camera Mount System
- **Type:** 2-axis pan/tilt gimbal
- **Pan Servo:** 180° rotation (left-right)
- **Tilt Servo:** 90° rotation (up-down)
- **Mount Location:** Inside acrylic dome
- **Control:** PCA9685 PWM driver

---

## 🔊 Audio System (REVISED)

### ❌ REMOVED Components
- ReSpeaker 2-Mic HAT (replaced by lapel mic)

### ✅ CURRENT Audio Architecture

#### Input
**Lapel Microphone**
- **Type:** Electret condenser microphone
- **Interface:** Analog 3.5mm jack → Raspberry Pi audio input
- **Use Case:** Bark detection, "quiet" command training
- **Sensitivity:** Adjustable via software (decibel threshold)

#### Processing
**Audio Switching - DPDT Relay System**
- **Relay 1 + 2:** 2x single-channel relays (or 1x dual-channel)
- **Control:** GPIO (1 pin for switching)
- **Function:** Route audio source to speakers
  - **Mode A:** DFPlayer Pro → Speakers (pre-recorded sounds)
  - **Mode B:** Raspberry Pi Audio Out → Speakers (live audio/TTS)

**Switching Logic:**
```
GPIO LOW  → DFPlayer   (NC - Normally Closed)
GPIO HIGH → Pi Audio   (NO - Normally Open)
```

#### Output
**DFPlayer Pro MP3 Module**
- **Storage:** MicroSD card (up to 32GB)
- **Output:** Line-level audio to relay
- **Control:** UART (RX/TX)
- **Files:** Pre-recorded training sounds (good_dog.mp3, etc.)

**Amplifier**
- **Model:** 300W Class D amplifier
- **Power:** Direct from battery (12-16.8V)
- **Output:** 2x channels
- **Volume:** Software-controlled via DFPlayer

**Speakers**
- **Model:** Gikfun 4Ω 3W speakers (x2)
- **Placement:** Left/right sides of robot
- **Mounting:** Under carousel, cutouts in saucer

---

## 🚗 Mobility System

### Devastator Tank Chassis
- **Type:** Tracked vehicle (differential drive)
- **Motors:** 2x DC gear motors with encoders
- **Motor Driver:** L298N H-Bridge
  - **Logic Power:** 5V (from buck converter)
  - **Motor Power:** Direct from battery (12-16.8V)
  - **Max Current:** 2A per channel
- **Speed Control:** PWM (0-100%)
- **Turning:** Differential steering

### Wheels/Tracks
- **Material:** Rubber tracks
- **Ground Clearance:** ~3cm
- **Terrain:** Indoor smooth surfaces (hardwood, tile, carpet)

---

## 🍪 Treat Dispensing System

### Carousel Mechanism
- **Design:** 3D-printed rotating disc
- **Diameter:** 110mm
- **Treat Capacity:** 6-8 treats per load
- **Rotation:** Servo-controlled (MG996R)
- **Dispensing:** Gravity-fed through slot

### Servo Motor (Treat Dispenser)
- **Model:** MG996R or similar (high-torque)
- **Torque:** 10+ kg·cm
- **Control:** PCA9685 PWM driver
- **Rotation:** 60° per treat dispense

---

## 💡 Lighting System

### NeoPixels (WS2812B)
- **Count:** 16-24 LEDs (ring or strip)
- **Location:** Around base or under dome
- **Control:** GPIO (SPI via RPi.GPIO)
- **Modes:** 
  - Status indicator (idle/active/training)
  - Celebration animations (treat dispensed)
  - Low battery warning (red flash)

### Blue LED Tube
- **Type:** EL wire or LED strip (blue)
- **Length:** ~50cm
- **Location:** Accent lighting around chassis
- **Control:** On/off via relay or transistor
- **Purpose:** Visual appeal, dog attention

---

## 🔌 Power System

### Battery
- **Type:** 4S2P 21700 Li-ion pack
- **Nominal Voltage:** 14.8V (4x 3.7V cells in series)
- **Capacity:** 6000mAh (2x 3000mAh in parallel)
- **Max Voltage:** 16.8V (fully charged)
- **Min Voltage:** 12.0V (cutoff)
- **BMS:** HX-4S-F30A or Daly Smart BMS
  - Over-discharge protection
  - Over-charge protection
  - Cell balancing

### Power Distribution
**Buck Converter 1:**
- **Input:** 12-16.8V (battery)
- **Output:** 5V @ 5A
- **Load:** Raspberry Pi 5

**Buck Converter 2:**
- **Input:** 12-16.8V (battery)
- **Output:** 5V @ 3A
- **Load:** DFPlayer Pro, L298N logic

**Buck Converter 3:**
- **Input:** 12-16.8V (battery)
- **Output:** 6V @ 2A
- **Load:** PCA9685, Servo motors (3x)

**Amplifier:**
- **Direct Battery:** 12-16.8V
- **Protection:** Fuse (5A)

### Charging System
**Method 1: Manual (Current)**
- **Connector:** XT60
- **Charger:** External balance charger
- **Time:** ~2 hours (6000mAh @ 3A)

**Method 2: Docking (Planned)**
- **Connector:** Pogo pins (magnetic)
- **Charger:** Onboard CC/CV module (16.8V @ 1-2A)
- **Docking:** IR-guided autonomous return-to-base

---

## 🛡️ Sensors (Current + Planned)

### Active Sensors
- [x] **IMX500 Camera** (vision, pose detection)
- [x] **Lapel Microphone** (audio input, bark detection)
- [x] **Motor Encoders** (odometry, navigation)

### Planned Sensors
- [ ] **IR Transmitters/Receivers** (Roomba-style docking)
  - **Dock:** 1x IR transmitter (360° beacon)
  - **Robot:** 3-4x IR receivers (TSOP38238 or similar)
  - **Protocol:** 38kHz modulated signal
  
- [ ] **Bumper Sensors** (collision detection)
  - **Type:** Mechanical switches (Roomba-style)
  - **Location:** Front/rear bumper
  - **Function:** Emergency stop on impact

- [ ] **Cliff Sensors** (edge detection)
  - **Type:** IR reflective sensors (Sharp GP2Y0A21YK0F)
  - **Location:** Underside, 4 corners
  - **Function:** Prevent falls (stairs, ledges)

- [ ] **Ultrasonic Sensors** (optional, obstacle avoidance)
  - **Model:** HC-SR04 or similar
  - **Range:** 2cm - 400cm
  - **Location:** Front (180° coverage)
  - **Function:** Medium-range obstacle detection

---

## 🎮 Control Interfaces

### Primary: WiFi (Web Dashboard)
- **Protocol:** HTTP/WebSocket
- **Port:** 8000 (API), 8080 (dashboard)
- **Range:** ~30m indoor
- **Features:**
  - Live camera stream
  - Mission control
  - Battery monitoring
  - Manual driving

### Secondary: Bluetooth (Optional)
- **Protocol:** Bluetooth Classic or BLE
- **Use Case:** Gamepad controller (Xbox/PS4)
- **Priority:** Overrides web interface when connected
- **Fallback:** If WiFi unavailable

---

## 📐 Physical Specifications

### Dimensions (Approximate)
- **Length:** 35cm
- **Width:** 28cm
- **Height:** 25cm (with dome)
- **Weight:** 2.5kg (fully loaded)

### Clearance
- **Ground:** 3cm
- **Track Width:** 20cm (stable turning)

### Payload
- **Treats:** ~100g (6-8 treats)
- **Electronics:** 1.5kg
- **Battery:** 0.5kg

---

## 🔧 Peripheral Modules

### PCA9685 PWM Driver
- **Channels:** 16 (using 3 for servos)
- **Interface:** I2C
- **Address:** 0x40 (default)
- **Frequency:** 50Hz (servo standard)
- **Control:** 
  - Channel 0: Treat dispenser servo
  - Channel 1: Camera pan servo
  - Channel 2: Camera tilt servo

### Relay Modules
- **Type:** 2x single-channel DPDT or 1x dual-channel
- **Voltage:** 5V coil
- **Current:** <100mA per coil
- **Control:** 1x GPIO pin (audio switching)

---

## 🌡️ Operating Conditions

### Temperature
- **Operating:** 0°C to 40°C
- **Storage:** -10°C to 50°C
- **Critical:** Pi 5 thermal throttle at 80°C (active cooling enabled)

### Humidity
- **Operating:** 20% to 80% RH (non-condensing)
- **Storage:** 10% to 90% RH

### Terrain
- **Suitable:** Hardwood, tile, short carpet (<2cm pile)
- **Unsuitable:** Thick carpet, outdoor (no waterproofing)

---

## ⚡ Power Consumption (Estimated)

| Component | Idle | Active | Peak |
|-----------|------|--------|------|
| Raspberry Pi 5 | 2.5W | 5W | 8W |
| Hailo-8 HAT | 1W | 3W | 5W |
| Camera | 0.5W | 1W | 1.5W |
| Motors | 0W | 12W | 24W |
| Servos (3x) | 0W | 3W | 6W |
| Amplifier + Speakers | 1W | 10W | 50W |
| LEDs | 0.5W | 2W | 5W |
| **Total** | **5.5W** | **36W** | **99.5W** |

**Battery Life:**
- **Idle:** ~18 hours (6000mAh @ 5.5W)
- **Active:** ~4-5 hours (6000mAh @ 36W)
- **Peak:** ~1 hour (6000mAh @ 100W, brief bursts)

---

## 🔄 Upgrade Path (Future)

### Hardware Enhancements
- [ ] GPS module (outdoor navigation)
- [ ] LiDAR sensor (SLAM mapping)
- [ ] Larger battery (10,000mAh, 4S3P)
- [ ] Wireless charging (Qi coil in dock)
- [ ] OLED display (status screen)

### Software Integrations
- [ ] ROS2 (Robot Operating System)
- [ ] Real-time video streaming (RTSP)
- [ ] Voice assistant (Alexa/Google integration)
- [ ] Cloud logging (AWS/Firebase)

---

## 📦 Bill of Materials (Updated)

### Electronics
- [x] Raspberry Pi 5 (8GB)
- [x] Hailo-8 HAT (26 TOPS)
- [x] IMX500 Camera
- [x] DFPlayer Pro
- [x] 300W Amplifier
- [x] L298N Motor Driver
- [x] PCA9685 PWM Driver
- [x] 2x DPDT Relay Modules (audio switching)
- [x] Lapel Microphone (electret)
- [ ] IR Transmitter/Receivers (planned)
- [ ] Bumper Sensors (planned)
- [ ] Cliff Sensors (planned)

### Mechanical
- [x] Devastator Tank Chassis
- [x] 3D-printed carousel
- [x] Acrylic dome
- [x] Servo motors (3x)

### Power
- [x] 4S2P 21700 battery pack
- [x] BMS (HX-4S-F30A)
- [x] Buck converters (3x)
- [x] XT60 connector
- [ ] Pogo pins (docking, planned)

### Wiring
- [x] Dupont jumper wires
- [x] Power distribution board
- [x] Terminal blocks
- [x] Heat shrink tubing
- [x] 18AWG wire (power)
- [x] 22AWG wire (signal)

---

**Last Hardware Update:** Audio system revised (lapel mic + relay switching)  
**Next Update:** IR sensors + bumper sensors installation