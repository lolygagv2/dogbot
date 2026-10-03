#!/usr/bin/env python3
"""Battery state-of-charge curve (2026-10-03).

The old straight line 12.0-16.8 V reported 62% at 14.98 V and would have put
15.6 V (DMM-verified) at 75%; a 4S Li-ion pack is ~45% / ~68% there.
Standalone: python3 tests/power/test_battery_soc_curve.py
"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# battery_monitor imports hardware libs at module level; stub what we don't have.
for name in ("board", "busio", "adafruit_ads1x15", "adafruit_ads1x15.ads1115",
             "adafruit_ads1x15.analog_in"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["adafruit_ads1x15.analog_in"].AnalogIn = object
sys.modules["adafruit_ads1x15.ads1115"].ADS1115 = object
from services.power.battery_monitor import BatteryMonitorService as B

f = B.voltage_to_percentage

def test_endpoints():
    assert f(16.8) == 100 and f(17.0) == 100
    assert f(13.2) == 0 and f(12.0) == 0 and f(0) == 0

def test_midpoints_follow_li_ion_curve():
    assert f(15.6) == 68, f(15.6)
    assert 40 <= f(14.98) <= 48, f(14.98)   # curve gives 42
    assert f(16.0) == 80 and f(14.4) == 20

def test_monotonic():
    last = -1
    v = 12.5
    while v <= 17.0:
        p = f(v); assert p >= last, (v, p, last); last = p; v += 0.05

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn(); print(f"PASS {name}")
            except AssertionError as e: fails += 1; print(f"FAIL {name}: {e}")
            except Exception as e: fails += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
