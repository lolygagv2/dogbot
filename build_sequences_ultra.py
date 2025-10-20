# build_sequences_ultra.py  — self-contained
import os, csv, numpy as np
from ultralytics import YOLO

# EDIT to match your clips.csv labels exactly (order matters for your later training)
BEHAVIORS = ["stand","sit","lie","cross","spin"]

def to_np(x):
    try: return x.detach().cpu().numpy()
    except AttributeError: return np.asarray(x)

def build_sequence(video_path, label_id, model, imgsz, conf, vid_stride,
                   min_frames, max_frames, min_kpts, min_kpt_conf, gap_tol):
    kseq=[]; last=None; gaps=0
    for r in model.predict(source=video_path, stream=True, imgsz=imgsz,
                           conf=conf, vid_stride=vid_stride, verbose=False):
        if len(r)==0 or r.boxes is None or r.keypoints is None:
            # allow filling small gaps with last valid frame
            if last is not None and gaps < gap_tol:
                kseq.append(last); gaps += 1
            continue
        i = int(r.boxes.conf.argmax())
        kxy = r.keypoints.xy[i]      # [K,2] torch or np
        kcf = r.keypoints.conf[i]    # [K]
        K = int(kxy.shape[0])

        # pad/truncate to 24 dog keypoints
        if K < 24:
            pad_xy = np.zeros((24,2), np.float32)
            pad_cf = np.zeros(24,       np.float32)
            vxy, vcf = to_np(kxy), to_np(kcf)
            pad_xy[:K], pad_cf[:K] = vxy, vcf
            kxy, kcf = pad_xy, pad_cf
        elif K > 24:
            # non-dog head (e.g., human-17) → try to fill gap
            if last is not None and gaps < gap_tol:
                kseq.append(last); gaps += 1
            continue

        # quality gate
        kxy, kcf = to_np(kxy), to_np(kcf)
        vis = (kcf > 0)
        if vis.sum() < min_kpts or (kcf[vis].mean() if vis.any() else 0.0) < min_kpt_conf:
            if last is not None and gaps < gap_tol:
                kseq.append(last); gaps += 1
            continue

        x1,y1,x2,y2 = to_np(r.boxes.xyxy[i])
        w=max(x2-x1,1e-6); h=max(y2-y1,1e-6)
        nk = np.stack([
            np.clip((kxy[:,0]-x1)/w, 0, 1),
            np.clip((kxy[:,1]-y1)/h, 0, 1),
            kcf.astype(np.float32)
        ], axis=1)  # [24,3]

        last = nk; gaps = 0
        kseq.append(nk)
        if max_frames and len(kseq) >= max_frames: break

    if len(kseq) < min_frames:
        return None
    return {"kpts": np.asarray(kseq, np.float32), "label": int(label_id)}

if __name__=="__main__":
    import argparse, pathlib
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--pt",    required=True)     # 24-kpt dog pose .pt
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--conf",  type=float, default=0.003)
    ap.add_argument("--vid_stride", type=int, default=1)
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--min_kpts", type=int, default=18)      # >= 18 visible
    ap.add_argument("--min_kpt_conf", type=float, default=0.20)
    ap.add_argument("--gap_tol", type=int, default=3)        # repeat last up to N frames
    args=ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.pt); assert model.task=="pose", "Not a pose model"
    saved=skipped=0
    with open(args.clips, newline='') as f:
        for row in csv.reader(f):
            if not row or len(row)<2: continue
            vid, lbl = row[0].strip(), row[1].strip()
            try:
                lid = BEHAVIORS.index(lbl)
            except ValueError:
                raise SystemExit(f"Label '{lbl}' not in BEHAVIORS {BEHAVIORS}")
            seq = build_sequence(
                vid, lid, model,
                imgsz=args.imgsz, conf=args.conf, vid_stride=args.vid_stride,
                min_frames=args.min_frames, max_frames=(args.max_frames or None),
                min_kpts=args.min_kpts, min_kpt_conf=args.min_kpt_conf, gap_tol=args.gap_tol
            )
            stem = pathlib.Path(vid).stem
            if seq is None:
                skipped+=1; print(f"skip {stem}: <{args.min_frames} usable frames")
            else:
                np.savez_compressed(os.path.join(args.out, f"{stem}.npz"), **seq)
                saved+=1; print("saved", stem, seq["kpts"].shape, lbl)
    print(f"done: saved {saved}, skipped {skipped}")
