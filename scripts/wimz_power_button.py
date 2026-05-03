#!/usr/bin/env python3
"""
WIMZ Power Button Watcher
- Engages relay first to isolate button from Pololu pad B
- Waits for GPIO20 to settle LOW after relay engagement
- Requires sustained HIGH (100ms) on GPIO20 to trigger shutdown (debounce)
- Triggers graceful shutdown; systemd hook handles GPIO26 kill pulse
"""
from gpiozero import OutputDevice, DigitalInputDevice
from subprocess import call
import time
import sys

# Step 1: Engage relay to isolate button from Pololu pad B
relay = OutputDevice(21, active_high=True, initial_value=True)
print("WIMZ: Relay engaged on GPIO21 — button isolated from Pololu", flush=True)

# Step 2: Initialize GPIO20 input with internal pull-down
button_input = DigitalInputDevice(20, pull_up=False)

# Step 3: Wait for GPIO20 to settle LOW
print("WIMZ: Waiting for GPIO20 to settle LOW...", flush=True)
settle_timeout = 5.0
settle_start = time.time()
while button_input.is_active:
    if time.time() - settle_start > settle_timeout:
        print("WIMZ: ERROR — GPIO20 never settled LOW after 5s. Check wiring.", flush=True)
        sys.exit(1)
    time.sleep(0.05)

print("WIMZ: GPIO20 settled LOW. Arming press detection (debounced).", flush=True)

# Step 4: Detect a sustained press, not a noise transient
# Wait for HIGH, then verify it stays HIGH for the debounce window before acting
DEBOUNCE_MS = 100
DEBOUNCE_S = DEBOUNCE_MS / 1000.0
POLL_INTERVAL_S = 0.01

while True:
    button_input.wait_for_active()

    # Verify the HIGH state is sustained
    high_start = time.time()
    while button_input.is_active:
        if time.time() - high_start >= DEBOUNCE_S:
            # Sustained press confirmed
            print(f"WIMZ: Sustained press detected ({DEBOUNCE_MS}ms hold) — initiating graceful shutdown", flush=True)
            call(["sudo", "shutdown", "-h", "now"])
            sys.exit(0)
        time.sleep(POLL_INTERVAL_S)

    # If we got here, GPIO20 went LOW before debounce timer completed.
    # That was a transient. Loop back and wait for next active.
    print("WIMZ: Transient on GPIO20 ignored (released before debounce)", flush=True)
