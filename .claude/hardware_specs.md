# TreatBot Hardware Specifications

## 🔧 SERVO SYSTEM - 3 Servos Total

### ⚠️ CRITICAL: Treat Carousel (NEMA 17 Stepper)
**Motor:** NEMA 17 Stepper Motor with TMC2209 driver
**Features:** Anti-jam detection and recovery
**Function:** **TREAT DISPENSING** - rotates to dispense treats
**Control Method:** Step/direction via TMC2209 UART

**Treat Dispensing Specifications:**
- **Treat Positions:** 11 compartments around 4 carousels = 44 treat slots
- **Rotation per treat:** 32.7° (360° ÷ 11 positions)
- **Dispensing method:** Precise stepper rotation to advance to next treat slot
- **Anti-jam:** Left stick push on Xbox controller triggers anti-jam sequence
- **Control:** TMC2209 UART via GPIO (see pin mapping)

**Control via TMC2209:**
- **STEP Pin:** GPIO12
- **DIR Pin:** GPIO16  
- **EN Pin:** GPIO24
- **UART:** GPIO14 (TX) / GPIO15 (RX) with 1k resistor

See `docs/TMC2209_UART_SETUP.md` for full configuration.

### 📹 Camera Pan Servo (MG996R)
**Servo:** MG996R high-torque servo (extended range)
**Channel:** 0 (PCA9685)
**Function:** Left/right camera movement
**Range:** 0° to 270° (270° total range)
**TRUE CENTER:** 140° (physical center position)
**Control Method:** `servo_controller.set_camera_pan(degrees)`

**Pan Mapping:**
- **Right Extreme:** 0° (rightmost view)
- **True Center:** 140° (straight ahead, horizon level)
- **Left Extreme:** 270° (leftmost view)
- **Full Range:** 270° of rotation available

### 📹 Camera Pitch/Tilt Servo (MG996R)
**Servo:** MG996R high-torque servo (extended range)
**Channel:** 1 (PCA9685)
**Function:** Up/down camera movement
**Range:** 20° to 200° (estimated full range)
**TRUE CENTER:** 50° (horizon level)
**Control Method:** `servo_controller.set_camera_pitch(degrees)`

**Tilt Mapping:**
- **Down Extreme:** ~20° (looking down)
- **True Horizon:** 50° (level with horizon)
- **Up Extreme:** ~200° (looking up)
- **Working Range:** 180° of tilt available

**⚠️ CRITICAL - Camera Gimbal Calibration Completed (Dec 2025)**
**TRUE CENTER POSITION: Pan=140°, Tilt=50°**
*This is the physically calibrated center - horizon level and centered left/right*

**⚠️ NO LAUNCHER SERVO EXISTS - Carousel handles all treat dispensing**
*Updated: October 2025 - Hardware Clarification*

## 🤖 System Overview

**Platform:** Autonomous Mobile Robot  
**Purpose:** AI-powered dog training and behavior reinforcement  
**Control:** Raspberry Pi 5 with Hailo-8 AI accelerator  

---

## 🧠 Core Computing

### Raspberry Pi 5
- **Model:** 8GB RAM variant (4GB also supported)
- **OS:** Raspberry Pi OS (64-bit)
- **Power:** 5V @ 5A via buck converter
- **Cooling:** Active heatsink + fan

### GPIO Pin Mapping (Current Build)
```
Pin # | Assignment
------|-------------------------------------------------
  1   | 3.3V → PCA9685 LOGIC V+
  2   | 5V (unused here, available for 5V power rail)
  3   | SDA1 → PCA9685 SDA
  4   | 5V → 5V supply
  5   | SCL1 → PCA9685 SCL
  6   | GND → PCA9685 GND
  7   | GPIO4 → Encoder A1 (Motor 1 GREEN WIRE LEFT)
  8   | GPIO14 (TXD) → TMC2209 UART (1k resistor wire)
  9   | GND → battery analyzer (black wire)
 10   | GPIO15 (RXD) → TMC2209 UART (split from TMC2209 Pin 4)
 11   | GPIO17 → PWM IN1
 12   | GPIO18 → PWM IN2
 13   | GPIO27 → PWM IN3 
 14   | GND 
 15   | GPIO22 → PWM IN4
 16   | GPIO23 → ENCODER B1 (Motor 1 YELLOW WIRE LEFT)
 17   | 3.3V → sensor/encoder VCC rail
 18   | GPIO24 → TMC2209 EN (enable)
 19   | GPIO10 (MOSI) → NeoPixel signal (black wire base)
 20   | GND
 21   | GPIO9 (MISO)
 22   | GPIO25 → Blue LED (MOSFET controlled)
 23   | GPIO11 (SCLK) 
 24   | GPIO8 (CE0)
 25   | GND
 26   | GPIO7 (CE1)
 27   | ID_SD (reserved)
 28   | ID_SC (reserved)
 29   | GPIO5 → ENCODER A2 (Motor 2 GREEN WIRE RIGHT)
 30   | GND
 31   | GPIO6 → ENCODER B2 (Motor 2 YELLOW WIRE RIGHT)
 32   | GPIO12 → TMC2209 STEP pin
 33   | GPIO13 → PWM ENA
 34   | GND  
 35   | GPIO19 → PWM ENB
 36   | GPIO16 → TMC2209 DIR pin
 37   | GPIO26 
 38   | GPIO20
 39   | GND
 40   | GPIO21

Motor Mapping:
- LEFT MOTOR = MOTOR A (OUTPUT 1/2)
- RIGHT MOTOR = MOTOR B (OUTPUT 3/4)

Key Notes:
- SDA/SCL (Pins 3 & 5) used by PCA9685
- Pin 19 (GPIO10) → NeoPixel: 330Ω resistor near Pi, 1000µF cap at strip input
- All GND pins tied to central GND rail
```

### Hailo-8 AI Accelerator HAT
- **Performance:** 26 TOPS (Tera Operations Per Second)
- **Interface:** PCIe (via HAT)
- **Models:** YOLOv8s (dog detection + pose estimation)
- **Inference Speed:** 30+ FPS @ 640x640 resolution

---

## 📷 Vision System

### Camera Options
**Option A: Sony IMX500 AI Camera**
- **Sensor:** Sony IMX500 (12.3MP)
- **Resolution:** 4056 x 3040 pixels
- **Features:** On-sensor AI processing, built-in ISP, HDR

**Option B: Raspberry Pi Camera Module 3 Wide**
- **Sensor:** Sony IMX708 (11.9MP)
- **Resolution:** 4608 x 2592 pixels
- **Features:** Wide angle lens, autofocus, HDR

**Common:**
- **Interface:** CSI-2 (4-lane)

### Camera Boot Config (Required for Swaps)
**Location:** `/boot/firmware/config.txt`

```
camera_auto_detect=1
#dtoverlay=imx500   # MUST be commented out
```

**Why:** Hardcoded `dtoverlay=imx500` forces the IMX500 driver to claim i2c 0x1a, blocking IMX708. With `camera_auto_detect=1` alone, libcamera auto-detects whichever sensor is present.

**Fleet status:**
- treatbot2: Fixed
- treatbot1, 3-5: Need this edit before camera swap

### Camera Mount System
- **Type:** 2-axis pan/tilt gimbal
- **Pan Servo:** 180° rotation (left-right)
- **Tilt Servo:** 90° rotation (up-down)
- **Mount Location:** Inside acrylic dome
- **Control:** PCA9685 PWM driver

---

## 🔊 Audio System (USB-BASED)

### ✅ CURRENT Audio Architecture - Unified USB Solution

#### Input & Output
**Ugreen USB Audio Adapter**
- **Type:** USB audio interface with microphone input and speaker output
- **Interface:** USB 2.0 → Raspberry Pi 5
- **Input:** Conference-style microphone (2.5" circular disc, 1cm height) for bark detection and voice commands
- **Output:** Speaker/headphone jack for audio playback (4Ω 5W speakers)
- **Power:** USB bus powered (no external power needed)
- **Control:** Standard Linux ALSA audio interface
- **✅ STATUS:** Simplified single-device solution - no switching needed

#### Audio Files
**Local VOICEMP3 Storage**
- **Location:** `/home/morgan/dogbot/VOICEMP3/`
- **Structure:**
  - `/VOICEMP3/talks/` - Training commands and dog names
  - `/VOICEMP3/songs/` - Background music and system sounds
- **Format:** MP3 files for space efficiency
- **Playback:** Direct Linux audio playback (aplay, pygame, etc.)
- **No SD Card:** Files stored on Pi's storage

#### VOICEMP3 File Organization
**Training Commands (`/VOICEMP3/talks/`):**
- `elsa.mp3`, `bezik.mp3` - Individual dog names
- `good.mp3`, `treat.mp3` - Reward sounds
- `sit.mp3`, ` down.mp3`, `stay.mp3` - Training commands
- `quiet.mp3`, `no.mp3` - Correction commands
- `scooby_intro.mp3` - Fun character sounds

**Background Audio (`/VOICEMP3/songs/`):**
- Various music and system sounds (many files)

#### Removed Components
- ❌ **DFPlayer Pro MP3 Module** - Replaced by USB solution
- ❌ **Audio Relay Switching** - No longer needed
- ❌ **300W Class D Amplifier** - USB adapter has built-in audio output
- ❌ **External Speakers** - Audio through USB adapter output

---

## 🚗 Mobility System

### Devastator Tank Chassis
- **Type:** Tracked vehicle (differential drive)
- **Motors:** 2x DFRobot Metal DC Geared Motor w/Encoder
  - **Model:** 6V 210RPM 10Kg.cm (upgraded from FIT0186)
  - **Rated Voltage:** 6V
  - **Speed:** 210 RPM @ 6V (vs 133 RPM previous)
  - **Torque:** 10 kg·cm (vs 4.5 kg·cm previous)
  - **Encoders:** Built-in quadrature encoders
  - **✅ STATUS:** Working with error-free operation, 50ms rate limiting applied
  - **✅ COMPLETE:** PWM/control issues resolved, safety fixes implemented
- **Motor Driver:** L298N H-Bridge
  - **Logic Power:** 5V (from buck converter)
  - **Motor Power:** Direct from battery (12-16.8V)
  - **Voltage Drop:** ~1.4V typical
  - **Max Current:** 2A per channel
  - **Effective Motor Voltage:** 12.6V (14V - 1.4V drop)
- **Speed Control:** PWM on enable pins
  - **Safe PWM Range:** 40-70% duty cycle 
  - **Maximum PWM:** 75% duty cycle (9V effective - absolute max)
  - **⚠️ NEVER use 100% duty cycle (would supply 12.6V to 6V motors)**
- **Turning:** Differential steering

### Wheels/Tracks
- **Material:** Rubber tracks
- **Ground Clearance:** ~3cm
- **Terrain:** Indoor smooth surfaces (hardwood, tile, carpet)

---

## 🍪 Treat Dispensing System

### Carousel Mechanism
- **Design:** 3D-printed rotating disc (4 stacked carousels)
- **Diameter:** 110mm
- **Treat Capacity:** 44 treats (11 compartments × 4 carousels)
- **Rotation:** NEMA 17 stepper with TMC2209 driver
- **Dispensing:** Gravity-fed through slot

### Stepper Motor (Treat Dispenser)
- **Model:** NEMA 17 stepper motor
- **Driver:** TMC2209 (UART mode)
- **Features:** Anti-jam detection and recovery
- **Rotation:** 32.7° per treat dispense (360° ÷ 11)

---

## 💡 Lighting System

### NeoPixels 
- **Count:** 155 LEDs (ring or strip)
- **Location:** Around base or under dome
- **Control:** GPIO (SPI via RPi.GPIO)
- **Modes:** 
  - Status indicator (idle/active/training)
  - Celebration animations (treat dispensed)
  - Low battery warning (red flash)

### Blue LED Tube
- **Type:** LED spot light with fiber optic tube (blue)
- **Length:** ~50cm
- **Location:** Accent lighting around chassis
- **Control:** On/off via relay or transistor
- **Purpose:** Visual appeal, dog attention

---

## 🔌 Power System

### Battery
- **Type:** 4S2P 21700 Li-ion pack
- **Nominal Voltage:** 14.8V (4x 3.6V cells in series)
- **Capacity:** 8900mAh (2x 4400mAh in parallel)
- **Max Voltage:** 16.8V (fully charged)
- **Min Voltage:** 12.0V (cutoff)
- **BMS:** HX-4S-F30A or 4 Cell standard BMS (two packs of 4 batteries)
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
- **Load:** NeoPixel LED, L298N logic

**Buck Converter 3:**
- **Input:** 12-16.8V (battery)
- **Output:** 6V @ 2A
- **Load:** PCA9685, Servo motors (3x)

**Buck Converter 4:**
- **Input:** 12-16.8V (battery)
- **Output:** 12V @ 2A
- **Load:** Amplifier, LED BLue Tube Light


**Amplifier:**
- **Buck Convert 4
- **Protection:** Fuse (5A)

### Charging System
**Method 1: Manual (Current)**
- **Connector:** XT30
- **Charger:** External balance charger
- **Time:** ~2 hours (8500mAh @ 4A)

**Method 2: Docking (Current)**
- **Connector:** Roomba Style Tabs (magnetic)
- **Charger:** Onboard CC/CV module (16.8V @ 1-2A)
- **Docking:** IR-guided autonomous return-to-base optional but disabled

---

## 🛡️ Sensors (Current + Planned)

### Active Sensors
- [x] **IMX500 Camera** (vision, pose detection)
- [x] **Conference Microphone** (audio input, bark detection)
- [x] **Motor Encoders** (odometry, navigation)

### Recently Added (OFFLINE - Power Issues)
- [🔧] **IR Sensors** (Roomba-style docking)
  - **Location:** Rear Right, Rear Center, Rear Left
  - **STATUS:** Hardware installed but disconnected pending power debug

- [x] **Charging Pads** (Roomba-style charging)
  - **Design:** Bare metal plates wired directly to P+/P-
  - **STATUS:** Hardware installed and working
  
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
- **Protocol:** HTTP/WebSocket/WebRTC
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
- **Channels:** 16 (using 2 for servos)
- **Interface:** I2C
- **Address:** 0x40 (default)
- **Frequency:** 50Hz (servo standard)
- **Channel Mapping:**
  - Channel 0: Camera pan servo
  - Channel 1: Camera tilt servo
  - Channel 2+: Not in use (treat dispenser now uses TMC2209 stepper)

---

## 🌡️ Operating Conditions

### Temperature
- **Operating:** 0°C to 80°C
- **Storage:** -10°C to 50°C
- **Critical:** Pi 5 thermal throttle at 80°C (active cooling enabled)

### Humidity
- **Operating:** 20% to 80% RH (non-condensing)
- **Storage:** 10% to 90% RH

### Terrain
- **Suitable:** All terrain except thick carpet
- **Unsuitable:** Thick carpet, extensive outdoors (limited waterproofing)

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
- **Idle:** ~21.5 hours (8000mAh @ 5.5W)
- **Active:** ~5-6 hours (8000mAh @ 36W)
- **Peak:** ~1.5 hour (8000mAh @ 100W, brief bursts)

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
- [x] 50W Amplifier
- [x] L298N Motor Driver
- [x] PCA9685 PWM Driver
- [x] Conference Microphone (electret)
- [ ] IR Transmitter/Receivers (not deployed)
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
- [x] Pogo pins (docking, planned)

### Wiring
- [x] Dupont jumper wires
- [x] Power distribution board
- [x] Terminal blocks
- [x] Heat shrink tubing
- [x] 18AWG wire (power)
- [x] 22AWG wire (signal)

---

**Last Hardware Update:** April 17, 2026 - Stepper motor, GPIO mapping, camera options
**Software Status:** Build 83 complete (April 2026) - All core systems validated
**Next Phase:** Manufacturing prep and per-unit calibration