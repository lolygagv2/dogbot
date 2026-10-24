#!/usr/bin/env python3
"""Behavior label must land on the box that is DRAWN — 2026-10-24.

Morgan (sit tests): "SIT kept flashing on and off rapidly with different
high-90% values". update_dog_behavior() wrote the label to exactly one tracker
entry — the anonymous -(idx+1000) entry if it existed, else the named one —
while get_tracked_dogs() hides an anonymous entry that overlaps a named box.
Whenever both entries existed for the same dog the label went to the hidden
one and the visible box blinked. Now every entry describing the animal (index,
name, bbox overlap) gets the label.

Run: /home/morgan/dogbot/env_new/bin/python tests/core/test_dog_tracker_behavior_label.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import core.dog_tracker as dt
from core.dog_tracker import DogTracker

CFG = {'dogs': [{'id': 'elsa', 'marker_id': 315}], 'persistence_seconds': 15}
BOX = [100, 100, 300, 400]


def tracker():
    t = DogTracker(CFG)
    t._profile_manager = None
    return t


def labels(t):
    return {v['name']: (v['behavior'], round(v['confidence'], 2)) for v in t.get_tracked_dogs().values()}


def test_named_and_anonymous_twin_both_get_label():
    t = tracker(); dt.time.time = lambda: 100.0
    t.process_frame([{"bbox": BOX}], [(315, 200, 250)])          # named entry 315
    # Hand-plant the anonymous twin the old code preferred (identity race)
    t._update_tracking(-1000, BOX, 100.0, confidence=0.5, id_method="unknown")
    t.valid_dog_ids[-1000] = 'dog_0'
    t.update_dog_behavior('elsa', 'sit', 0.97, detection_idx=0, bbox=BOX)
    assert t.last_known_positions[-1000]['behavior'] == 'sit'
    assert t.last_known_positions[315]['behavior'] == 'sit', "named (drawn) entry must carry the label"
    assert labels(t) == {'elsa': ('sit', 0.97)}, labels(t)


def test_stale_name_cache_still_labels_visible_dog_box():
    t = tracker(); dt.time.time = lambda: 100.0
    t.process_frame([{"bbox": BOX}], [])                           # anonymous only
    t.update_dog_behavior('elsa', 'sit', 0.9, detection_idx=0, bbox=BOX)   # cache says elsa, no 315 entry
    assert labels(t) == {'Dog': ('sit', 0.9)}, labels(t)


def test_index_flip_between_frames_keeps_label_on_the_right_box():
    t = tracker(); dt.time.time = lambda: 100.0
    other = [500, 100, 640, 300]
    t.process_frame([{"bbox": BOX}, {"bbox": other}], [])          # idx0=BOX, idx1=other
    # Classifier result stamped with idx 1 but it is the BOX animal (order swapped this frame)
    t.update_dog_behavior(None, 'sit', 0.95, detection_idx=1, bbox=BOX)
    by_box = {tuple(v['bbox']): v['behavior'] for v in t.get_tracked_dogs().values()}
    assert by_box[tuple(BOX)] == 'sit', by_box
    # the far box got it too via index (legacy) — acceptable; what matters is the real one is labelled


def test_label_blanks_after_two_seconds_of_silence():
    t = tracker(); dt.time.time = lambda: 100.0
    t.process_frame([{"bbox": BOX}], [(315, 200, 250)])
    t.update_dog_behavior('elsa', 'sit', 0.97, detection_idx=0, bbox=BOX)
    dt.time.time = lambda: 101.5
    assert labels(t)['elsa'][0] == 'sit'
    dt.time.time = lambda: 102.5
    assert labels(t)['elsa'][0] == '', "stale label must still blank (2026-07-25 rule)"


def test_no_bbox_argument_is_backward_compatible():
    t = tracker(); dt.time.time = lambda: 100.0
    t.process_frame([{"bbox": BOX}], [(315, 200, 250)])
    t.update_dog_behavior('elsa', 'lie', 0.8, detection_idx=0)
    assert labels(t) == {'elsa': ('lie', 0.8)}, labels(t)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
