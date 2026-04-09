#!/usr/bin/env python3
"""Raw stepper diagnostic — uses actual dispenser init."""

import sys
sys.path.insert(0, '/home/morgan/dogbot')
import time
import logging
logging.basicConfig(level=logging.INFO)

from services.reward.dispenser import DispenserService

print("=== STEPPER DIAGNOSTIC (full dispenser init) ===\n")

# Create a real dispenser instance with proper initialization
d = DispenserService()

try:
    input("Test 1: 1 slot CW (dispense direction), SLOW. Press Enter...")
    d._enable_motor()
    d._step(d.steps_per_slot, d.CW, delay=0.01)
    time.sleep(0.5)
    d._disable_motor()
    print("  Done.\n")

    input("Test 2: 1 slot CCW (reverse direction), SLOW. Press Enter...")
    d._enable_motor()
    d._step(d.steps_per_slot, d.CCW, delay=0.01)
    time.sleep(0.5)
    d._disable_motor()
    print("  Done.\n")

    input("Test 3: 3 slots CW, motor stays enabled. Press Enter...")
    d._enable_motor()
    for i in range(3):
        print(f"  Slot {i+1}...")
        d._step(d.steps_per_slot, d.CW, delay=0.008)
        time.sleep(1)
    d._disable_motor()
    print("  Done.\n")

    input("Test 4: Full dispense cycle (advance + nudge). Press Enter...")
    d._rotate_carousel()
    print("  Done.\n")

    print("All tests complete!")

except KeyboardInterrupt:
    print("\nAborted")
finally:
    d._disable_motor()
