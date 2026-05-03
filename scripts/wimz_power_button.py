#!/usr/bin/env python3
"""
WIMZ Power Button Watcher
Handles boot-time HIGH state on GPIO20 by engaging relay first,
waiting for GPIO20 to settle LOW, then arming press detection.
"""
from gpiozero import OutputDevice, DigitalInputDevice
from subprocess import call
import time
import sys

# Step 1: Engage relay immediately to isolate button from Pololu pad B
# GPIO21 HIGH = relay coil energized = NC contact open = button isolated
relay = OutputDevice(21, active_high=True, initial_value=True)
print("WIMZ: Relay engaged on GPIO21 — button isolated from Pololu", flush=True)

# Step 2: Initialize GPIO20 input with internal pull-down
button_input = DigitalInputDevice(20, pull_up=False)

# Step 3: Wait for GPIO20 to settle LOW after relay engagement
# The pre-engagement HIGH state from pad B propagation needs to drain
# through the 10k pull-down. This typically takes only a few ms,
# but we give it up to 5 seconds before declaring a wiring problem.
print("WIMZ: Waiting for GPIO20 to settle LOW...", flush=True)
settle_timeout = 5.0
settle_start = time.time()
while button_input.is_active:
    if time.time() - settle_start > settle_timeout:
        print("WIMZ: ERROR — GPIO20 never settled LOW after 5s. Check wiring.", flush=True)
        sys.exit(1)
    time.sleep(0.05)

print("WIMZ: GPIO20 settled LOW. Arming press detection.", flush=True)

# Step 4: Wait for an actual button press (LOW → HIGH transition)
button_input.wait_for_active()
print("WIMZ: Button pressed — initiating graceful shutdown", flush=True)

# Step 5: Trigger graceful shutdown. systemd shutdown hook handles
# the GPIO26 kill pulse at end of shutdown to cut Pololu power.
call(["sudo", "shutdown", "-h", "now"])
