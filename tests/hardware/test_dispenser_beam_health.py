#!/usr/bin/env python3
"""Regression tests for dispenser beam health (2026-10-20).

Pure-logic tests — no GPIO, no motor. Build a DispenserService without
running __init__ and stub the hardware edges.

Covers:
  1. miss tracker: 3 unconfirmed first rotations in the last 4 -> beam_suspect;
     one confirmed first rotation clears it
  2. check_beam(): active-low receiver reading LOW at rest -> blocked_at_rest
  3. dispense_treat(): cancelled ladder with zero counted does NOT set
     empty_or_jammed (the Xbox-tap bug)
  4. dispense_treat(): suspect beam -> ONE rotation, credited on motor truth,
     confirmed=0, no retry ladder
  5. dispense_treat(): beam blocked at rest -> ONE rotation on motor truth

Run:  python3 tests/hardware/test_dispenser_beam_health.py
"""
import sys
import threading
import types
from collections import deque
from unittest import mock

sys.path.insert(0, '/home/morgan/wimz/current')
sys.path.insert(0, '/home/morgan/dogbot')

import services.reward.dispenser as disp_mod
from services.reward.dispenser import DispenserService


def make_dispenser(beam_enabled=True, level=1):
    d = DispenserService.__new__(DispenserService)
    d.logger = disp_mod.logging.getLogger('test-dispenser')
    d.bus = mock.MagicMock()
    d.state = types.SimpleNamespace(mission=types.SimpleNamespace(name=None))
    d.store = mock.MagicMock()
    d.gpio_chip = 1
    d.uart = None
    d.initialized = True
    d.beam_enabled = beam_enabled
    d.beam_pin = 7
    d.beam_active_low = True
    d.beam_timeout_s = 4.0
    d._beam_callback = None
    d._last_beam_break_mono = 0.0
    d._beam_breaks = deque(maxlen=256)
    d.BEAM_DEBOUNCE_S = 0.02
    d.COUNT_WINDOW_S = 0.0
    d.DISPENSE_CAP = 2
    d.ANTIJAM_CAP = 6
    d.BEAM_HISTORY_N = 4
    d.BEAM_SUSPECT_MISSES = 3
    d._beam_history = deque(maxlen=4)
    d.beam_suspect = False
    d.beam_blocked_at_rest = False
    d.last_beam_check = None
    d.last_beam_calibration = None
    d.dispense_started_mono = 0.0
    d.empty_or_jammed = False
    d.last_unjam_count = 0
    d.last_wimz_dispense_id = None
    d.last_dispense_confirmed = 0
    d.treats_dispensed_today = 0
    d.last_dispense_time = 0.0
    d.last_dispense_mono = 0.0
    d.min_dispense_interval = 0.0
    d.daily_limit = 300
    d.dog_treat_counts = {}
    d.dog_cooldowns = {}
    d.treats_given = 0
    d.treat_capacity = 40
    d.treats_dispensed_session = 0
    d._empty_published = False
    d._dispense_lock = threading.Lock()
    d._refill_active = False
    d._refill_stop_requested = False
    d._dispense_active = False
    d._cancel_dispense = False
    d.rotations = 0
    d.read_beam_level = lambda: level
    d._increment_treat_counter = lambda: setattr(d, 'treats_given', d.treats_given + 1)
    d._check_tmc_diagnostics = lambda *a, **k: None
    d._disable_motor = lambda: None
    d._anti_jam_wiggle_inner = lambda: True

    def rotate():
        d.rotations += 1
        return True
    d._rotate_carousel = rotate
    return d


def test_miss_tracker_sets_and_clears_suspect():
    d = make_dispenser()
    d._record_beam_outcome(False)
    d._record_beam_outcome(False)
    assert not d.beam_suspect
    d._record_beam_outcome(False)
    assert d.beam_suspect, "3 misses in a row must mark the beam suspect"
    d._record_beam_outcome(True)
    assert not d.beam_suspect, "a confirmed first rotation clears suspect"


def test_check_beam_blocked_at_rest():
    d = make_dispenser(level=0)          # active-low: LOW == broken
    r = d.check_beam(samples=3, interval_s=0)
    assert r['verdict'] == 'blocked_at_rest' and r['blocked_at_rest']
    d2 = make_dispenser(level=1)
    assert d2.check_beam(samples=3, interval_s=0)['verdict'] == 'ok'


def _run_dispense(d, counts):
    """counts: beam results per rotation (list of ints)."""
    it = iter(counts)
    d._count_beam_window = lambda fire, cap: (next(it, 0), None, 0)
    with mock.patch.object(disp_mod, 'get_wimz_store') as gw, \
         mock.patch.object(disp_mod, 'publish_reward_event'):
        gw.return_value.log_dispense.return_value = 1
        gw.return_value.get_or_create_dog.return_value = 'dog'
        return d.dispense_treat(dog_id='x', reason='manual')


def test_cancelled_ladder_does_not_flag_empty():
    d = make_dispenser()
    # cancel arrives during the first rotation
    orig = d._rotate_carousel
    def rotate_then_cancel():
        ok = orig()
        d._cancel_dispense = True
        return ok
    d._rotate_carousel = rotate_then_cancel
    ok = _run_dispense(d, [0, 0, 0])
    assert ok is False
    assert d.rotations == 1
    assert d.empty_or_jammed is False, "a cancelled ladder is not evidence of empty"


def test_full_ladder_still_flags_empty():
    d = make_dispenser()
    ok = _run_dispense(d, [0, 0, 0, 0])
    assert ok is False and d.rotations == 3 and d.empty_or_jammed is True


def test_suspect_beam_single_rotation_motor_truth():
    d = make_dispenser()
    d.beam_suspect = True
    ok = _run_dispense(d, [0, 0, 0])
    assert ok is True
    assert d.rotations == 1, "suspect beam must not climb the retry ladder"
    assert d.treats_given == 1 and d.last_dispense_confirmed == 0
    assert d.empty_or_jammed is False


def test_blocked_beam_single_rotation_motor_truth():
    d = make_dispenser(level=0)
    ok = _run_dispense(d, [0, 0, 0])
    assert ok is True and d.rotations == 1 and d.treats_given == 1


def test_healthy_beam_confirms_first_rotation():
    d = make_dispenser()
    ok = _run_dispense(d, [1])
    assert ok and d.rotations == 1 and d.last_dispense_confirmed == 1
    assert list(d._beam_history) == [True]


if __name__ == '__main__':
    # pytest is not installed in env_new — plain runner
    import traceback
    fails = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            try:
                _fn(); print('PASS', _name)
            except Exception:
                fails += 1; print('FAIL', _name); traceback.print_exc()
    print('failures:', fails)
    sys.exit(1 if fails else 0)
