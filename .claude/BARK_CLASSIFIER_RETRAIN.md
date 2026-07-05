# Bark Classifier Fine-Tune — Handoff Spec

**Goal:** Fix false bark detections (human speech / TV / music / the robot's own
audio get classified as barks). Root cause: the current model's `notbark` class
was never trained on those sounds, so it outputs `notbark ≈ 0.00` for speech and
confidently labels it a bark emotion. Every cheap signal feature (duration,
envelope, spectral ratio) overlaps between speech and barks — only the model can
separate them, and it just needs a much stronger `notbark` set.

**PREREQUISITE — you need the trainable SOURCE model, not the .tflite.**
The robot only has `dog_bark_classifier.tflite`, which is inference-only and
CANNOT be fine-tuned. To modify the model you must locate the original
`.h5` / `.keras` / SavedModel it was exported from (desktop / cloud / Colab where
you first trained it). If that source is gone, fine-tuning is impossible and you
must rebuild from scratch (there is no reliable .tflite→trainable path).

## Plan: fine-tune the existing model to strengthen `notbark`
1. Load the source model (`tf.keras.models.load_model(...)`).
2. Freeze the convolutional base; keep the final dense/classifier layers trainable
   (transfer learning — preserves the existing emotion learning).
3. Add a large, diverse `notbark` set (see Dataset) to the training data and
   fine-tune a few epochs at a low learning rate (e.g. 1e-4).
4. Validate `notbark` recall specifically on held-out HUMAN SPEECH — that's the
   metric that matters. It should reject speech with `notbark > 0.5`.
5. Re-export to `.tflite` and install on the robot (see below).

## Exact model contract (must match, or the robot pipeline breaks)

The robot preprocesses audio in `ai/bark_classifier.py: preprocess_audio()`.
Train with the SAME preprocessing so your new model drops in:

- Sample rate: **22050 Hz**, mono
- Clip length: **3.0 s** (pad/truncate to 66150 samples)
- Features: `librosa.feature.melspectrogram(y, sr=22050, n_mels=128)`
  → `librosa.power_to_db(ref=np.max)` → normalize to [0,1]
- Input tensor shape: **(1, 128, 130, 1)**  (batch, n_mels, time, channel), float32
- Output: **softmax over 7 classes**, index order MUST equal
  `ai/models/emotion_mapping.json`:
  `{"aggressive":0,"alert":1,"anxious":2,"attention":3,"notbark":4,"playful":5,"scared":6}`

The robot vetoes a detection when `notbark > 0.5` (`core/audio/bark_detector.py`).
So the single most important improvement is: **make `notbark` fire reliably on
non-bark sound.** Keep the same 7 classes and mapping.

(Simpler alternative if you don't care about emotions: train a 2-class
`bark` / `notbark` model. If you do this, tell me and I'll adjust the robot code —
the downstream expects the 7-class softmax above.)

## Dataset to build

**Positives (barks):** 113 real dog-bark WAVs already on the robot at
`audio/dogbarktest/` (16 kHz — resample to 22050). Add more from public sets:
ESC-50 (`dog` class), UrbanSound8K (`dog_bark`), FSD50K.
For emotion labels, use whatever the original set used, or lean on public
bark-emotion datasets; if emotion labels are unavailable, put all barks in the
bark-emotion classes as best you can — the critical fix is the `notbark` set.

**Negatives (`notbark`) — this is the whole point, make it BIG and diverse:**
- Human speech (many voices, conversational, near + far) — this is the #1 gap
- TV / dialogue / news, music (several genres)
- Household noise, silence/ambient, claps, door slams, footsteps
- **The robot's OWN audio** — record/ingest these files, they self-triggered
  detections: `VOICEMP3/talks/**/quiet.mp3`, `VOICEMP3/songs/**` (calming music),
  and any `good_dog` reward clips. Capturing them through the actual mic is best;
  I can collect those on the robot if you want (see below).

Aim for at least a few thousand 3-s clips total, roughly balanced, with `notbark`
being the largest single class. Augment (noise, gain, time-shift) for robustness.

## Rebuild-from-scratch fallback (ONLY if the source model is lost)

Standard small CNN on the mel-spectrograms above (Conv→Pool blocks → Dense →
softmax(7)). Split train/val/test, watch that `notbark` recall on held-out
**human speech** is high (that's the metric that matters). Then convert:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("dog_bark_classifier.tflite","wb").write(tflite_model)
```

## Install back on the robot

1. Copy your new file to `ai/models/dog_bark_classifier.tflite`
   (keep a backup of the old one first: `cp ai/models/dog_bark_classifier.tflite ai/models/dog_bark_classifier.tflite.bak`)
2. Ensure `ai/models/emotion_mapping.json` matches your class order (7-class: leave as-is).
3. Restart: `sudo systemctl restart treatbot.service`
4. Validate: put robot in Silent Guardian (`POST /mode/set {"mode":"silent_guardian"}`),
   talk near it, and confirm logs show `ML veto: notbark=... — rejecting`
   instead of `Bark detected`.

## Robot side is ALREADY set up (done 2026-07-05)
- The `notbark` veto is now **synchronous** (`core/audio/bark_detector.py`): the
  detector classifies BEFORE emitting a bark, and if `notbark > 0.5` it rejects
  the sound outright — SG never reacts. So your only job is to make `notbark`
  fire correctly; the plumbing to act on it is in place.
- It **fails open**: if the model is missing / not loaded / errors, barks still
  emit (we never drop a real bark over a classifier problem).
- Just drop the new `.tflite` in `ai/models/`, keep the 7-class order in
  `emotion_mapping.json`, and restart. No other code changes needed.
- Temporary tuning instrumentation (`BARK_TUNE`, `BARK_ENV`) has been removed.

## Notes
- If input shape or class order differs from the contract above, the robot code
  in `ai/bark_classifier.py` must be updated to match — flag it.
