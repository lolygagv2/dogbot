#!/usr/bin/env python3
"""
Capture pose-keypoint sequences for behavior LSTM training.

Reuses the live AI controller's pose pipeline so saved keypoints are
identical in format to inference-time data. Saves .npz files matching
the format expected by scripts/train_behavior_lstm.py:
    kpts:  [L, 24, 3] in [-0.5, 0.5] range (bbox-normalized, centered)
    label: int (0..N-1)

Usage (interactive):
    python3 scripts/capture_behavior_sequences.py --out ai/sequences_imx500/

Single-key commands during capture:
    s = sit, t = stand, l = lie, p = spin, k = speak
    space = stop & save current sequence
    d = discard current sequence
    q = quit
"""
import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

BEHAVIORS = ["stand", "sit", "lie", "spin", "speak"]
KEY_TO_LABEL = {"t": 0, "s": 1, "l": 2, "p": 3, "k": 4}
MIN_FRAMES = 16  # match T=16 windowing in train_behavior_lstm.py


def normalize_keypoints(kpts_xyc: np.ndarray, bbox) -> np.ndarray:
    """Normalize keypoints to [-0.5, 0.5] centered on bbox.

    kpts_xyc: (24, 3) raw [x, y, conf] in pixel coords
    bbox: object with .x1 .y1 .x2 .y2
    Returns: (24, 3) with x,y in [-0.5, 0.5], conf preserved
    """
    out = kpts_xyc.copy().astype(np.float32)
    w = max(bbox.x2 - bbox.x1, 1e-6)
    h = max(bbox.y2 - bbox.y1, 1e-6)
    out[:, 0] = ((kpts_xyc[:, 0] - bbox.x1) / w) - 0.5
    out[:, 1] = ((kpts_xyc[:, 1] - bbox.y1) / h) - 0.5
    return out


class _RawTerm:
    """Non-blocking single-key reader for terminal."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self) -> Optional[str]:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for .npz sequences")
    ap.add_argument("--dog", default="unknown", help="Dog name/id for filename")
    ap.add_argument("--condition", default="default", help="Lighting/scene tag for filename")
    ap.add_argument("--session", default=None, help="Shared session id (use same value on both robots for paired captures)")
    ap.add_argument("--res", default="640x640", help="Capture resolution WxH")
    args = ap.parse_args()
    session_id = args.session or str(int(time.time()))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from picamera2 import Picamera2
    from core.ai_controller_3stage_fixed import AI3StageControllerFixed
    from services.perception.camera_detect import detect_camera_type

    cam_type = detect_camera_type()
    print(f"[capture] camera detected: {cam_type}")
    print(f"[capture] output dir:      {out_dir}")
    print(f"[capture] dog:             {args.dog}")
    print(f"[capture] condition:       {args.condition}")

    w, h = (int(x) for x in args.res.split("x"))
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (w, h), "format": "RGB888"}))
    cam.start()
    time.sleep(0.5)

    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("[capture] ERROR: AI controller failed to initialize", file=sys.stderr)
        cam.stop()
        sys.exit(1)
    print("[capture] AI controller ready")
    print("[capture] keys: t=stand s=sit l=lie p=spin k=speak  space=save d=discard q=quit")

    recording = False
    label: Optional[int] = None
    buf: List[np.ndarray] = []
    frames_seen = 0
    seq_counter = 0

    with _RawTerm() as term:
        try:
            while True:
                key = term.getch()
                if key:
                    if key == "q":
                        print("\n[capture] quit")
                        break
                    if key in KEY_TO_LABEL:
                        if recording:
                            print(f"\n[capture] already recording {BEHAVIORS[label]}; press space to save first")
                        else:
                            label = KEY_TO_LABEL[key]
                            buf = []
                            recording = True
                            print(f"\n[capture] RECORDING {BEHAVIORS[label]} ...")
                    elif key == " " and recording:
                        if len(buf) >= MIN_FRAMES:
                            seq_counter += 1
                            kpts_arr = np.stack(buf, axis=0).astype(np.float32)
                            fname = f"{cam_type}_{BEHAVIORS[label]}_{args.dog}_{args.condition}_{session_id}_{seq_counter}.npz"
                            np.savez(out_dir / fname, kpts=kpts_arr, label=np.int64(label))
                            print(f"[capture] saved {fname}  frames={len(buf)}")
                        else:
                            print(f"[capture] discarded — only {len(buf)} frames (need >= {MIN_FRAMES})")
                        recording = False
                        buf = []
                        label = None
                    elif key == "d" and recording:
                        print(f"[capture] discarded sequence ({len(buf)} frames)")
                        recording = False
                        buf = []
                        label = None

                frame = cam.capture_array()
                frames_seen += 1
                _, poses, _ = ai.process_frame(frame, skip_behavior=True)

                if poses:
                    pose = poses[0]  # capture first dog only
                    norm = normalize_keypoints(pose.keypoints, pose.detection)
                    if recording:
                        buf.append(norm)
                    status = f"DOG  kpts_visible={int(np.sum(pose.keypoints[:,2]>0.25))}/24"
                else:
                    status = "no dog"

                tag = f"REC[{BEHAVIORS[label]}] frames={len(buf)}" if recording else "idle"
                sys.stdout.write(f"\r[capture] {tag:<28} | {status:<32} | seen={frames_seen}   ")
                sys.stdout.flush()

        finally:
            cam.stop()
            print(f"\n[capture] done. saved {seq_counter} sequence(s) to {out_dir}")


if __name__ == "__main__":
    main()
