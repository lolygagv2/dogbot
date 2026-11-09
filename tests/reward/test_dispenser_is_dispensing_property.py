#!/usr/bin/env python3
"""Regression: DispenserService.is_dispensing must be a bool PROPERTY.

/treat/dispense, api/ws.py and the Xbox path read `dispenser.is_dispensing`
without parentheses as the double-tap-to-cancel guard. On 2026-11-09 the
@property decorator was accidentally moved onto the new _refuse() helper:
is_dispensing became a bound method (always truthy), so every owner tap
was "cancelled" instead of dispensed, while _refuse() raised TypeError.

Run: /home/morgan/dogbot/env_new/bin/python tests/reward/test_dispenser_is_dispensing_property.py
Standalone (no pytest, no hardware — only inspects the class).
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.reward.dispenser import DispenserService  # noqa: E402

FAILS = []


def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)


check(isinstance(inspect.getattr_static(DispenserService, 'is_dispensing'), property),
      "is_dispensing is a property (bare attribute read in server/ws/xbox guards)")
check(not isinstance(inspect.getattr_static(DispenserService, '_refuse'), property),
      "_refuse is a plain method (callable with a code and reason)")
check(callable(DispenserService._refuse), "_refuse is callable")

# Exercise the property on a bare instance without touching hardware.
d = DispenserService.__new__(DispenserService)
d._dispense_active = False
check(d.is_dispensing is False, "is_dispensing reads False when idle (not a truthy bound method)")
d._dispense_active = True
check(d.is_dispensing is True, "is_dispensing reads True while a dispense runs")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
