#!/usr/bin/env python3
"""DogTracker duplicate-box regression (2026-10-03).

Overlay showed "Dog" and "Elsa" boxes on the same animal. Root cause: an IoU
re-match rewrote Elsa's id_method 'aruco' -> 'persistence'; every dedupe check
only looked at aruco/color entries, so a fresh anonymous entry was created and
drawn beside her. Also: ArUco tag id 0 was treated as "no tag" (falsy).

Standalone, no hardware:  python3 tests/core/test_dog_tracker_dedupe.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import core.dog_tracker as dt
from core.dog_tracker import DogTracker

CFG = {'dogs': [{'id': 'elsa', 'marker_id': 315}, {'id': 'testvoicedog', 'marker_id': 0}],
       'persistence_seconds': 15}


def tracker():
    t = DogTracker(CFG)
    t._profile_manager = None   # no colour ID in the test
    return t


def visible(t):
    return {v['name'] for v in t.get_tracked_dogs().values()}


def test_tagless_frames_after_aruco_do_not_spawn_a_dog_twin():
    t = tracker(); dt.time.time = lambda: 100.0
    box = [100, 100, 300, 400]
    t.process_frame([{"bbox": box}], [(315, 200, 250)])              # tag seen
    for i in range(10):                                   # tag hidden, dog drifts
        dt.time.time = lambda i=i: 100.5 + i
        b = [100 + 3 * i, 100, 300 + 3 * i, 400]
        t.process_frame([{"bbox": b}], [])
        assert visible(t) == {'elsa'}, (i, visible(t))
    assert t.last_known_positions[315]['id_method'] == 'aruco'


def test_single_dog_rule_then_tagless_keeps_one_box():
    t = tracker(); dt.time.time = lambda: 200.0
    box = [100, 100, 300, 400]
    # Tag visible but OUTSIDE the box (collar below the detection) -> Rule 0
    t.process_frame([{"bbox": box}], [(315, 200, 450)])
    assert visible(t) == {'elsa'}
    dt.time.time = lambda: 200.5
    t.process_frame([{"bbox": [105, 100, 305, 400]}], [])
    assert visible(t) == {'elsa'}, visible(t)


def test_generic_entry_removed_when_identity_arrives_late():
    t = tracker(); dt.time.time = lambda: 300.0
    box = [100, 100, 300, 400]
    t.process_frame([{"bbox": box}], [])                              # anonymous first
    assert visible(t) == {'Dog'}
    dt.time.time = lambda: 300.3
    t.process_frame([{"bbox": box}], [(315, 200, 250)])               # tag appears
    assert visible(t) == {'elsa'}
    assert not any(k < 0 for k in t.last_known_positions), t.last_known_positions.keys()


def test_tag_id_zero_is_a_real_dog():
    t = tracker(); dt.time.time = lambda: 400.0
    box = [100, 100, 300, 400]
    a = t.process_frame([{"bbox": box}], [(0, 200, 250)])
    assert a == {0: 'testvoicedog'}, a
    assert visible(t) == {'testvoicedog'}
    dt.time.time = lambda: 400.5
    t.process_frame([{"bbox": [103, 100, 303, 400]}], [])
    assert visible(t) == {'testvoicedog'}, visible(t)
    t.clear_dog_tracking('testvoicedog')
    assert 0 not in t.last_known_positions


def test_two_real_dogs_still_two_boxes():
    t = tracker(); dt.time.time = lambda: 500.0
    t.process_frame([{"bbox": [0, 0, 200, 300]}, {"bbox": [400, 0, 600, 300]}], [(315, 100, 150), (0, 500, 150)])
    assert visible(t) == {'elsa', 'testvoicedog'}


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
            except Exception as e:
                fails += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
