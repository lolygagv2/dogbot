#!/usr/bin/env python3
"""ABR controller anti-flap regression (2026-10-03).

Journal 2026-10-02 16:38: tiers flipped low->medium->high->medium->low every
5-15 s on a clean LAN (loss 0%, rtt ~70 ms) because aiortc's REMB estimate sat
latched at 256k-308k — just above REMB_FLOOR — and was read as "constrained".
The Low tier was also 640x480 (4:3), so every flip changed the picture shape.

Standalone (no aiortc, no hardware): drives _evaluate() directly.
    python3 tests/streaming/test_abr_hysteresis.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import services.streaming.adaptive_bitrate as abr
from services.streaming.adaptive_bitrate import AdaptiveBitrateController, TIERS


class FakeTrack:
    def __init__(self): self.res = None
    def set_output_resolution(self, r): self.res = r


class FakePC:
    def getSenders(self): return []


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def make(clock):
    abr.time.monotonic = clock
    c = AdaptiveBitrateController("test", FakePC(), FakeTrack())
    c._media_confirmed = True
    return c


def tick(c, clock, loss=0.0, rtt=0.07, observed=None, n=1):
    for _ in range(n):
        clock.advance(c.LOOP_INTERVAL)
        c._evaluate(loss, rtt, observed)


def test_all_tiers_16_9():
    for t in TIERS:
        assert abs(t.width / t.height - 16 / 9) < 0.01, f"{t.name} is not 16:9"


def test_remb_just_above_floor_does_not_flap():
    clock = Clock(); c = make(clock)
    # Clean LAN, REMB latched a bit above the floor (what the journal showed).
    tick(c, clock, observed=256_000, n=40)   # 100 s
    assert c.current_tier.name == "high", c.current_tier.name
    tick(c, clock, observed=308_000, n=40)
    assert c.current_tier.name == "high", "stepped down on a floored REMB"


def test_single_loss_spike_ignored_but_sustained_loss_steps_down():
    clock = Clock(); c = make(clock)
    tick(c, clock, observed=256_000, n=40)
    assert c.current_tier.name == "high"
    tick(c, clock, loss=0.08)                       # one bad sample
    assert c.current_tier.name == "high", "stepped down on one bad tick"
    tick(c, clock, loss=0.08)                       # second consecutive
    assert c.current_tier.name == "medium"


def test_loss_collapse_steps_down_immediately():
    clock = Clock(); c = make(clock)
    tick(c, clock, observed=256_000, n=40)
    tick(c, clock, loss=1.0)
    assert c.current_tier.name == "medium"


def test_real_remb_constraint_still_steps_down():
    clock = Clock(); c = make(clock)
    tick(c, clock, observed=1_400_000, n=40)
    assert c.current_tier.name == "high"
    # Receiver really can only take ~600k (well above the floor band).
    tick(c, clock, observed=600_000, n=2)
    assert c.current_tier.name == "high", "REMB-only needs 3 ticks"
    tick(c, clock, observed=600_000, n=1)
    assert c.current_tier.name == "medium"


def test_step_up_hold_is_longer_after_a_step_down():
    clock = Clock(); c = make(clock)
    tick(c, clock, observed=256_000, n=40)
    tick(c, clock, loss=0.08, n=2)
    assert c.current_tier.name == "medium"
    tick(c, clock, observed=256_000, n=6)   # 15 s good — old hold would step up
    assert c.current_tier.name == "medium", "stepped back up too soon"
    tick(c, clock, observed=256_000, n=8)   # 35 s total
    assert c.current_tier.name == "high"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
