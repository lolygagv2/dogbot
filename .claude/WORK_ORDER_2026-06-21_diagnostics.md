# WIM-Z Technical Work Order: Diagnostics, Optimization, and Idiot-Proofing

Owner: Morgan Hill
Date: June 21, 2026
Status: Hand directly to Claude Code on the robot. Execute in order.
Context: User feedback from deployed testers revealed that setup friction (not AI quality) is the biggest barrier to adoption. A FANG software engineer could not figure out WiFi-to-AP mode switching. Boot takes 90 seconds. These are product-killing failures that outweigh every AI feature behind them.

> SAVED 2026-06-27 by Claude Code session. Work paused before execution (out of time).
> Immediate next step requested: **Section 1 + 2 audit (H.264/FPS streaming)**. See
> memory `project_streaming_audit_pending.md` for status and the staged plan.

---

## 1. FPS AND PIPELINE DIAGNOSTIC

Goal: Determine whether the current 30 FPS at 640x640 (YOLOv8n on Hailo-8) is the hardware ceiling or an artificial bottleneck from pipeline inefficiency. The Hailo-8 should run YOLOv8n at 120-150 FPS at 640. We are getting 30. Something is consuming 70-80% of available throughput. Find it.

### 1A. Baseline isolation tests

Run each test for 60 seconds, log average FPS, peak FPS, and per-core CPU utilization. Kill all non-essential services before each test. The point is to isolate each stage of the pipeline and measure its cost independently.

```bash
# PREP: Kill everything that competes for CPU
sudo systemctl stop treatbot  # or whatever the main service is called
# Identify and stop any streaming, audio, web server processes

# TEST 1: Pure Hailo inference, no camera, no stream, no display
# Feed a static 640x640 image in a loop to the Hailo
# This measures raw Hailo throughput with zero CPU pipeline overhead
# Expected: 120-150 FPS for YOLOv8n
# Write a minimal Python script:
python3 << 'EOF'
import time
import numpy as np
from hailo_platform import HEF, VDevice, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams

hef = HEF("/home/morgan/dogbot/ai/models/dogdetector_14.hef")
target = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=target)
network_group = target.configure(hef, configure_params)[0]
network_group_params = network_group.create_params()

input_vstream_info = hef.get_input_vstream_infos()
output_vstream_info = hef.get_output_vstream_infos()
input_shape = input_vstream_info[0].shape
print(f"Model input shape: {input_shape}")

# Create a dummy frame matching the model input
dummy_frame = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)

input_params = InputVStreamParams.make(network_group, quantized=False)
output_params = OutputVStreamParams.make(network_group, quantized=False)

frames = 0
start = time.time()
with InferVStreams(network_group, input_params, output_params) as pipeline:
    while time.time() - start < 60:
        result = pipeline.infer({input_vstream_info[0].name: np.expand_dims(dummy_frame, axis=0)})
        frames += 1
elapsed = time.time() - start
print(f"Pure Hailo inference: {frames/elapsed:.1f} FPS over {elapsed:.1f}s")
EOF

# TEST 2: Camera capture only, no inference, no stream
# This measures raw camera + ISP cost
# Expected: 30-60 FPS at 640x640 depending on ISP config
python3 << 'EOF'
import time
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
cam.configure(config)
cam.start()
time.sleep(2)  # let auto-exposure settle

frames = 0
start = time.time()
while time.time() - start < 60:
    frame = cam.capture_array()
    frames += 1
elapsed = time.time() - start
print(f"Camera capture only (640x480): {frames/elapsed:.1f} FPS")
cam.stop()
EOF

# TEST 3: Camera capture at 1080p, no inference, no stream
python3 << 'EOF'
import time
from picamera2 import Picamera2

cam = Picamera2()
config = cam.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})
cam.configure(config)
cam.start()
time.sleep(2)

frames = 0
start = time.time()
while time.time() - start < 60:
    frame = cam.capture_array()
    frames += 1
elapsed = time.time() - start
print(f"Camera capture only (1080p): {frames/elapsed:.1f} FPS")
cam.stop()
EOF

# TEST 4: Camera 640 + Hailo inference, NO streaming, NO audio, NO motors
# This measures the actual vision pipeline without the stream encode tax
# Compare this to TEST 1 and TEST 2 individually
# The gap between (TEST1 + TEST2 combined theoretical) and TEST4 actual = pipeline overhead

# TEST 5: Camera 640 + Hailo inference + H.264 streaming encode
# This is what currently runs. Compare to TEST 4.
# The gap between TEST 4 and TEST 5 = encode tax

# TEST 6: Camera 1080 + Hailo inference (resize to 640 for model), NO streaming
# This tests whether higher-res capture + downscale kills FPS even without encode
```

### 1B. CPU profiling during normal operation

While the full system is running normally (all services, streaming, audio):

```bash
# Snapshot of what is eating CPU
ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -20

# Continuous monitoring (run in separate SSH session during a 5-min session)
top -b -d 1 -n 300 | grep -E "python|ffmpeg|gst|x264|libcamera|hailo" > /tmp/cpu_profile.log

# Check for thermal throttling
vcgencmd measure_temp
vcgencmd measure_clock arm
vcgencmd get_throttled  # 0x0 = no throttling, anything else = problem

# Check if software H.264 encode is running
ps aux | grep -E "x264|ffmpeg|gst.*x264|gst.*h264"

# Memory pressure
free -m
cat /proc/meminfo | grep -E "MemFree|MemAvail|Buffers|Cached"
```

### 1C. What to report

```
TEST 1 (pure Hailo):         ___ FPS   CPU: ___% per core
TEST 2 (camera 640 only):    ___ FPS   CPU: ___% per core
TEST 3 (camera 1080 only):   ___ FPS   CPU: ___% per core
TEST 4 (cam640 + Hailo):     ___ FPS   CPU: ___% per core
TEST 5 (cam640 + Hailo + stream): ___ FPS  CPU: ___% per core
TEST 6 (cam1080 + Hailo 640, no stream): ___ FPS  CPU: ___% per core
Thermal throttled: yes/no
Software H.264 process found: yes/no (process name)
Top 5 CPU consumers during normal operation: (list)
```

Diagnosis falls out of the gaps:
- TEST 1 vs TEST 4 gap = camera + pre-processing overhead
- TEST 4 vs TEST 5 gap = encode tax (the prime suspect)
- TEST 2 vs TEST 3 gap = resolution scaling cost on ISP/CPU
- TEST 4 vs current 30 FPS = overhead from other services (audio, motors, web server, Python GIL)

### 1D. Known suspects to check in the code

Open `core/ai_controller_3stage_fixed.py` and `core/vision/camera_manager.py`:
- Frame format: YUV420 (efficient) vs RGB888 (3x data). Unnecessary color conversions?
- numpy copies: `frame.copy()`, `np.array(frame)` — ~1.2MB each at 640x640x3, ~6MB at 1080p.
- Synchronous blocking: is Hailo inference blocking the main loop? Camera waiting on previous inference? Should be decoupled (camera fills buffer, inference reads async).
- GIL contention: capture, inference dispatch, post-processing, encode all in one process = GIL serializes CPU-bound work.
- Resolution mismatch: capturing >640 then resizing in Python (slow) vs ISP (fast)?

---

## 2. ENCODING STREAM FIX

"For sure" fix regardless of diagnosis. Stream and inference pipelines must be decoupled.

### 2A. Current problem
Stream resolution locked to inference resolution. 640 -> grainy stream. 1080 -> CPU dies.

### 2B. Target architecture
```
Camera native res
    |
    +--> ISP downscale to 640x640 --> Hailo --> detections
    |                                              |
    +--> Encode at 720p/1080p ----> Stream <-------+ (overlay annotations)
          (MJPEG if H.264 too heavy)
```
Picamera2 multi-stream: `main` for encode, `lores` for inference:
```python
config = cam.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},   # owner stream
    lores={"size": (640, 480), "format": "RGB888"},    # inference feed
    encode="main"
)
# inference: cam.capture_array("lores")  ; stream: cam.capture_array("main")
```

### 2C. MJPEG vs H.264
Pi 5 has NO hardware H.264 encoder. Software H.264 at 1080p may eat 2-3 cores. MJPEG is per-frame, lighter, no inter-frame dependency. Test both:
```bash
libcamera-vid --codec mjpeg --width 1280 --height 720 -t 30000 -o test.mjpeg &  # monitor CPU
libcamera-vid --codec h264  --width 1280 --height 720 -t 30000 -o test.h264 &   # monitor CPU
```
If MJPEG is 20-30% less CPU, use it. Savings go to inference/service headroom.

### 2D. Adaptive bitrate indicator (app-side)
Signal-strength icon from frame latency/drops; "Weak connection" text. Reframes grain as "bad signal" not "bad product."

---

## 3. BOOT TIME AND NETWORK OVERHAUL

Hard specs:
- Boot to LEDs/sound (sign of life): under 5s
- Boot to camera live on phone: under 30s
- WiFi connect or AP fallback: under 15s total
- Zero user-facing mode selection (never choose WiFi vs AP)
- Zero state where device is on but unreachable

### 3A. Boot sequence overhaul (current ~90s)
```bash
systemd-analyze
systemd-analyze blame | head -20
systemd-analyze critical-chain treatbot.service
```
- **Sign of life <5s:** early `sysinit.target` systemd service (`wimz-alive.service`) -> NeoPixel breathing + startup chime, before Python/AI/network. Bash or C, not full Python stack.
- **Camera live <30s:** start camera + stream BEFORE AI pipeline; AI inference attaches to already-running camera as stage 2. User sees dog before AI overlays.
- **Defer non-critical:** audio detection, treat calibration, mission engine, social, telemetry -> start after camera live via `After=wimz-camera.service`.

### 3B. Kill WiFi/AP mode concept (current: 5-min failover, user must understand modes)
Target: always reachable, credentials once, invisible.
```
on_boot:
  start AP mode immediately (always-available fallback)
  if stored_wifi_credentials:
    attempt wifi (timeout 10s)
    if connect: route via WiFi, status="wifi_connected"
    else: status="ap_mode"; retry wifi every 60s
  else: status="ap_mode_first_boot"  # app prompts for creds
  # USER NEVER CHOOSES
```
NetworkManager priority profiles (AP autoconnect + home-wifi higher priority). Pi 5 has ONE radio (wlan0) — AP+client simultaneously is flaky; cleaner: AP always available, WiFi in background, hand off, AP reactivates in seconds if WiFi drops. App tries both endpoints (WiFi IP + 192.168.4.1), uses whichever responds.

### 3C. First-boot credential flow
Boot -> AP -> LEDs pulse -> app auto-discovers via BLE/mDNS -> "Found your WIM-Z!" camera live -> WiFi picker -> connect -> feed continues. No IP, no mode, no manual. <2 min.

---

## 4. IDIOT-PROOFING REVIEW

Assume every interaction attempted incorrectly, wrong order, no instructions.

### 4A. Xbox/Gamepad
- Works instant on power; BT auto-reconnect; app shows exact button to hold if pairing.
- Every button does something or nothing (no-op unmapped); worst case "nothing happened".
- Controller disconnect -> robot STOPS immediately (failsafe); app shows "reconnecting...".
- Visual controller map on first use.
- Review `gamepad.py` checklist: controller on before/after robot; BT drop mid-drive; all buttons at once; trigger held 60s; two controllers; deadzone 0.30 drift across controllers; battery dies while driving (stop vs last command?).

### 4B. Failure-mode audit (abridged — see full checklist in original)
Power (unplug during boot/FS corruption; overcharge; low-voltage graceful shutdown; wrong charger). App (reinstall re-discovery; two phones; phone sleep mid-drive -> robot stops; force-quit mid-mission; captive portal; cellular phone + WiFi robot via relay). Dog (knock-over orientation stop; chew charging cable current-limit; paw power button recessed/long-press; sit-on-top stall stop; treats empty detect+notify; wrong treat size jam recovery). Network (WiFi pw change -> AP fallback; router reboot reconnect; friend's house new WiFi; multiple robots unique mDNS; 5GHz-only AP band). Audio (no voice loaded -> default + prompt; noisy recording warn; dog barks during record; volume too loud -> slider, default low).

### 4C. Meta-rule
"Can an idiot who never saw this device, read no instructions, tries every wrong thing, get value in <2 min?" If no, not done. Every error message says what to DO not what went wrong ("Tap here to reconnect"). Hide dev-only options (AP mode, resolution, servo cal, debug overlays) behind a dev menu (tap serial 7x).

---

## Execution priority
1. Run diagnostics (Section 1) — ~30 min; determines if Section 2 is a fix or confirmed non-issue.
2. Implement stream decoupling (Section 2) — correct regardless.
3. Boot overhaul (3A) — sign-of-life 5s, camera 30s.
4. Kill WiFi/AP switching (3B).
5. Idiot-proof audit (Section 4) -> punch list.

Items 1-2 = one session. Items 3-4 = second session (systemd + NetworkManager). Item 5 = review pass.

## Changelog
- v0.1 (June 21, 2026): Initial work order.
