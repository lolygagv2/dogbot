#!/usr/bin/env python3
"""Bark emotion classifier must run on LiteRT, not full TensorFlow — 2026-10-24.

tb4 lag RCA: the first COACH entry imported full TensorFlow (+505 MB RSS,
10-20 s) inside the robot process just to run the 7-class bark model, pushing
the 4 GB Pi into swap and the pose pipeline to 2-5 FPS during the first sit
attempts. ai_edge_litert runs the same model in ~+25 MB.

Run: /home/morgan/dogbot/env_new/bin/python tests/audio/test_bark_classifier_runtime.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def rss_mb():
    return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') // 2**20


FAILS = []
def check(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c: FAILS.append(m)


before = rss_mb()
import ai.bark_classifier as bc  # noqa: E402
after_import = rss_mb()
check(bc.TFLITE_RUNTIME_NAME == 'ai_edge_litert', f"runtime is ai_edge_litert (got {bc.TFLITE_RUNTIME_NAME})")
check('tensorflow' not in sys.modules, "full tensorflow NOT imported")
check(after_import - before < 120, f"import cost {after_import - before} MB (< 120)")

clf = bc.BarkClassifier(
    model_path='/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
    emotion_mapping_path='/home/morgan/dogbot/ai/models/emotion_mapping.json',
    sample_rate=22050, duration=3.0, n_mels=128)
after_model = rss_mb()
check(getattr(clf, 'interpreter', None) is not None, "model loaded by the LiteRT interpreter")
check(after_model - before < 200, f"import+model RSS {after_model - before} MB (< 200; TensorFlow was ~505)")

# One real inference on silence through the public API (whatever its name is)
audio = np.zeros(int(22050 * 3.0), dtype=np.float32)
fn = None
for name in ('classify', 'predict', 'classify_audio', 'predict_emotion', 'process'):
    if hasattr(clf, name):
        fn = getattr(clf, name); break
if fn is None:
    check(False, "no public classify/predict method found")
else:
    try:
        out = fn(audio)
        check(out is not None, f"{fn.__name__}() on 3 s silence returned {type(out).__name__}")
    except Exception as e:
        check(False, f"{fn.__name__}() raised {type(e).__name__}: {e}")

print(); print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
