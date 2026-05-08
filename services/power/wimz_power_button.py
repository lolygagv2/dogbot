#!/usr/bin/env python3
"""
WIMZ Power Button Watcher

Manages soft power shutdown via Pololu Mini Pushbutton Power Switch.

Hardware:
- GPIO21 (Pin 40) → relay module IN (isolates button from Pololu latch)
- GPIO20 (Pin 38) → button press sense (10kΩ external pull-down to GND)
- GPIO26 (Pin 37) → Pololu OFF kill pulse (handled by shutdown hook)

Relay Polarity (CRITICAL - edit per robot):
- Robot 1 (Teyleten optocoupler, active HIGH): active_high=True
- Robots 2-5 (5V SRD module, active LOW): active_high=False

Behavior:
1. On startup: Engages relay to isolate button from Pololu latch
2. Waits for GPIO20 to settle LOW
3. Detects sustained 100ms press (debounce against electrical noise)
4. Triggers graceful shutdown; systemd hook fires GPIO26 kill pulse

Installation:
    sudo cp services/power/wimz_power_button.py /usr/local/bin/
    sudo chmod +x /usr/local/bin/wimz_power_button.py
    sudo cp systemd/wimz-power-button.service /etc/systemd/system/
    sudo cp systemd/wimz-killpulse /lib/systemd/system-shutdown/
    sudo chmod +x /lib/systemd/system-shutdown/wimz-killpulse
    sudo systemctl daemon-reload
    sudo systemctl enable wimz-power-button.service
    sudo systemctl start wimz-power-button.service
"""
from gpiozero import OutputDevice, DigitalInputDevice
from subprocess import call
import time
import sys

# ===== EDIT THIS LINE PER ROBOT =====
# Robot 1 (Teyleten active-HIGH): active_high=True
# Robots 2-5 (SRD active-LOW): active_high=False
RELAY_ACTIVE_HIGH = False  # <-- Change per robot
# ====================================

relay = OutputDevice(21, active_high=RELAY_ACTIVE_HIGH, initial_value=True)

print("WIMZ: Relay engaged on GPIO21 — button isolated from Pololu", flush=True)

button_input = DigitalInputDevice(20, pull_up=False)

print("WIMZ: Waiting for GPIO20 to settle LOW...", flush=True)
settle_timeout = 5.0
settle_start = time.time()
while button_input.is_active:
    if time.time() - settle_start > settle_timeout:
        print("WIMZ: ERROR — GPIO20 never settled LOW after 5s. Check wiring.", flush=True)
        sys.exit(1)
    time.sleep(0.05)

print("WIMZ: GPIO20 settled LOW. Arming press detection (100ms debounce).", flush=True)

DEBOUNCE_S = 0.1
POLL_S = 0.01

while True:
    button_input.wait_for_active()
    high_start = time.time()
    while button_input.is_active:
        if time.time() - high_start >= DEBOUNCE_S:
            print("WIMZ: Sustained press detected — initiating graceful shutdown", flush=True)
            call(["sudo", "shutdown", "-h", "now"])
            sys.exit(0)
        time.sleep(POLL_S)
    print("WIMZ: Transient on GPIO20 ignored", flush=True)
