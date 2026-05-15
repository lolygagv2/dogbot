# Behavior LSTM Training Guide

## Overview

The behavior LSTM takes 16-frame windows of dog keypoints and classifies into: `stand`, `sit`, `lie`, `spin`.

**Train on:** Morgan's Blackwell PC (CUDA)  
**Deploy to:** Raspberry Pi 5 (TorchScript `.ts` file)

---

## Prerequisites

```bash
# On training PC
pip install torch numpy scikit-learn
```

---

## Step 1: Transfer Captured Data

From the Pi, copy sequences to your training PC:

```bash
# On training PC
scp -r morgan@treatbot1.local:~/dogbot/ai/sequences_imx500 ./sequences_imx500
```

Or if you have multiple sequence folders:
```bash
scp -r morgan@treatbot1.local:~/dogbot/ai/sequences_* ./
```

---

## Step 2: Check Data Balance

Before training, verify you have enough samples per class:

```bash
# Count files per behavior
for f in sequences_imx500/*.npz; do basename "$f" | cut -d'_' -f2; done | sort | uniq -c
```

**Recommended minimums:**
- 30+ files per behavior for decent results
- 50+ files per behavior for good results
- 100+ files per behavior for robust results

**Current capture (2026-04-27):**
- sit: 65, stand: 48, lie: 36, spin: 25

**Note:** "speak" was removed from visual behaviors — it's handled by the bark detector (audio), not pose classification.

---

## Step 3: Train the Model

### Basic Training

```bash
python3 scripts/train_behavior_lstm.py \
    --data sequences_imx500/ \
    --out behavior_imx500.pt \
    --out-ts behavior_imx500.ts
```

### With Multiple Data Sources

```bash
python3 scripts/train_behavior_lstm.py \
    --data sequences_imx500/ sequences_v8/ \
    --out behavior_shared.pt \
    --out-ts behavior_shared.ts
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | One or more sequence directories |
| `--epochs` | 80 | Max training epochs |
| `--bs` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--augment` | medium | Augmentation: none/light/medium/heavy |
| `--patience` | 12 | Early stopping patience |
| `--out` | behavior_head.pt | PyTorch checkpoint |
| `--out-ts` | behavior_head.ts | TorchScript for Pi |

### Augmentation Profiles

| Profile | Scale Jitter | Translation | Flip | Dropout |
|---------|--------------|-------------|------|---------|
| none | 0% | 0% | 0% | 0% |
| light | 5% | 2% | 50% | 5% |
| medium | 10% | 5% | 50% | 10% |
| heavy | 20% | 10% | 50% | 20% |

---

## Step 4: Evaluate Results

Watch the training output:

```
epoch   1  train 1.423/0.312  val 1.156/0.445
epoch   2  train 0.987/0.521  val 0.834/0.589
...
epoch  45  train 0.156/0.942  val 0.234/0.891
[train] early stop at epoch 45
[train] best val_acc=0.891
```

**Target accuracy:**
- 70%+ = usable
- 80%+ = good
- 90%+ = excellent

If accuracy is low:
1. Check class balance (need more of underrepresented behaviors)
2. Try `--augment heavy` for small datasets
3. Capture more varied sequences (different angles, lighting, dogs)

---

## Step 5: Deploy to Pi

Copy the TorchScript model to the Pi:

```bash
# From training PC
scp behavior_imx500.ts morgan@treatbot1.local:~/dogbot/ai/models/behavior_lstm.ts
```

Then restart the service:

```bash
# On Pi
sudo systemctl restart treatbot
```

---

## Capture More Data

If you need more sequences:

```bash
# On Pi - interactive capture
python3 scripts/capture_behavior_sequences.py \
    --behavior sit \
    --dog elsa \
    --condition daylight \
    --location kitchen
```

Press SPACE to mark good frames, Q to quit.

---

## Troubleshooting

### "No module named sklearn"
```bash
pip install scikit-learn
```

### Low accuracy on one class
Capture more sequences for that behavior. The model struggles with underrepresented classes.

### Model works in training but not on Pi
- Ensure PyTorch versions are compatible
- Check that the `.ts` file transferred correctly (compare file sizes)
- Verify the Pi code loads from the correct path

### Adding new behaviors
Edit `train_behavior_lstm.py` BEHAVIORS list, then capture sequences with matching labels:
```python
BEHAVIORS = ["stand", "sit", "lie", "spin", "newbehavior"]
```

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/capture_behavior_sequences.py` | Capture labeled sequences on Pi |
| `scripts/train_behavior_lstm.py` | Train LSTM on GPU PC |
| `ai/sequences_imx500/*.npz` | Captured IMX500 sequences |
| `ai/sequences_v8/*.npz` | Captured YOLOv8 sequences (if any) |
| `ai/models/behavior_lstm.ts` | Deployed TorchScript model |
