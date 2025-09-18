#!/usr/bin/env python3
"""
soft_start.py - Modern Bookworm-compatible audio soft start
Using gpiozero library (recommended for Pi OS Bookworm)
"""

from gpiozero import OutputDevice
import subprocess
import time

# GPIO16 (Physical Pin 36) - Audio relay switch
audio_relay = OutputDevice(16, initial_value=False)  # Start LOW (DFPlayer)

def safe_amixer_set(control, value):
    """Safely set audio levels, handling PipeWire or ALSA"""
    try:
        subprocess.run(['amixer', 'set', control, value], 
                      capture_output=True, check=False)
    except Exception as e:
        print(f"Audio control warning: {e}")

# Wait for system to stabilize
print("Waiting for audio system to stabilize...")
time.sleep(3)

# Ensure relay is in DFPlayer position
audio_relay.off()  # off() = LOW = DFPlayer
print("Audio relay set to DFPlayer")

# Gradually unmute with PipeWire-friendly commands
print("Gradually unmuting audio...")
for vol in range(0, 50, 10):
    safe_amixer_set('Master', f'{vol}%')
    time.sleep(0.2)

safe_amixer_set('Master', 'unmute')
print("Audio system initialized safely - no pop!")

# Keep running to hold GPIO state
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Shutting down soft_start...")
    audio_relay.close()