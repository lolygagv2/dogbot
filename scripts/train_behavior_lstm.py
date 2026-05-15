"""
Train the behavior LSTM head on captured keypoint sequences.

Inputs: one or more directories of .npz files containing
    kpts:  [L, 24, 3]  (in [-0.5, 0.5] bbox-normalized, centered)
    label: int (0..N-1) matching BEHAVIORS index

Outputs:
    --out           PyTorch checkpoint (.pt) for resuming training
    --out-ts        TorchScript model (.ts) for inference on the Pi

Run on Morgan's Blackwell PC (CUDA accelerated).

Example:
    python3 scripts/train_behavior_lstm.py \\
        --data ai/sequences_imx500/ ai/sequences_imx708/ \\
        --out behavior_shared.pt \\
        --out-ts behavior_shared.ts \\
        --augment medium
"""
import os, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

BEHAVIORS = ["stand", "sit", "lie", "spin"]
T = 16  # window length


def windows(x, t=T):
    # x: [L,24,3] -> list of [t,48]
    x = x[..., :2].reshape(len(x), -1)  # drop conf, flatten 24x2=48
    out = []
    if len(x) < t:
        return out
    for i in range(0, len(x) - t + 1):
        out.append(x[i:i + t])
    return out


def load_sequences(folders):
    """Load .npz files from one or more folders."""
    if isinstance(folders, str):
        folders = [folders]
    X, y = [], []
    for folder in folders:
        files = glob.glob(os.path.join(folder, "*.npz"))
        print(f"  {folder}: {len(files)} files")
        for p in files:
            z = np.load(p)
            xs = windows(z["kpts"])
            for w in xs:
                X.append(w)
                y.append(int(z["label"]))
    X = np.array(X, dtype=np.float32)  # [N,T,48]
    y = np.array(y, dtype=np.int64)
    return X, y


# Augmentation profiles: (scale_jitter, trans_jitter, flip_p, dropout_p)
AUG_PROFILES = {
    "none":   (0.00, 0.00, 0.0, 0.00),
    "light":  (0.05, 0.02, 0.5, 0.05),
    "medium": (0.10, 0.05, 0.5, 0.10),
    "heavy":  (0.20, 0.10, 0.5, 0.20),
}


def augment_batch(xb: torch.Tensor, profile: str) -> torch.Tensor:
    """Apply random augmentations per-sample.

    xb shape: [B, T, 48] where 48 = 24 keypoints * (x, y), already in [-0.5, 0.5].
    Augmentations are applied to (x, y) coords; flip negates x.
    """
    if profile == "none":
        return xb
    sj, tj, fp, dp = AUG_PROFILES[profile]
    B, Tw, D = xb.shape
    out = xb.clone()
    coords = out.view(B, Tw, 24, 2)

    if sj > 0:
        scale = 1.0 + (torch.rand(B, 1, 1, 1, device=xb.device) * 2 - 1) * sj
        coords *= scale
    if tj > 0:
        trans = (torch.rand(B, 1, 1, 2, device=xb.device) * 2 - 1) * tj
        coords += trans
    if fp > 0:
        flip = (torch.rand(B, device=xb.device) < fp).view(B, 1, 1, 1).float()
        coords[..., 0] = coords[..., 0] * (1 - 2 * flip.squeeze(-1))
    if dp > 0:
        keep = (torch.rand(B, 1, 24, 1, device=xb.device) > dp).float()
        coords *= keep
    return coords.view(B, Tw, D)


class Head(nn.Module):
    def __init__(self, in_dim=48, hid=64, num_classes=len(BEHAVIORS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hid, num_classes)

    def forward(self, x):          # x: [B,T,48]
        o, _ = self.lstm(x)        # [B,T,H]
        h = o[:, -1]               # last step
        return self.fc(h)          # [B,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="One or more sequence directories")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="behavior_head.pt", help="PyTorch checkpoint output")
    ap.add_argument("--out-ts", default="behavior_head.ts", help="TorchScript output for inference")
    ap.add_argument("--augment", choices=list(AUG_PROFILES), default="medium")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[train] loading data from {len(args.data)} dir(s)")
    X, y = load_sequences(args.data)
    print(f"[train] total windows: {len(X)} | classes seen: {sorted(set(y.tolist()))}")
    print(f"[train] BEHAVIORS: {BEHAVIORS}")
    print(f"[train] augment profile: {args.augment} -> {AUG_PROFILES[args.augment]}")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}")

    model = Head().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def run_epoch(Xd, yd, train=True):
        model.train(train)
        tot = 0
        correct = 0
        n = 0
        for i in range(0, len(Xd), args.bs):
            xb = torch.tensor(Xd[i:i + args.bs]).to(device)
            yb = torch.tensor(yd[i:i + args.bs]).to(device)
            if train:
                xb = augment_batch(xb, args.augment)
                opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            if train:
                loss.backward()
                opt.step()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            tot += loss.item() * len(xb)
            n += len(xb)
        return tot / n, correct / n

    best = 0.0
    bad = 0
    for e in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(Xtr, ytr, True)
        va_loss, va_acc = run_epoch(Xva, yva, False)
        print(f"epoch {e:3d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            bad = 0
            torch.save({"state_dict": model.state_dict(), "behaviors": BEHAVIORS, "T": T}, args.out)
            # Export TorchScript for inference (must be CPU for Pi deployment)
            model.eval()
            model.cpu()
            example = torch.zeros(1, T, 48)  # CPU tensor
            traced = torch.jit.trace(model, example)
            traced.save(args.out_ts)
            model.to(device)  # Move back for continued training
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop at epoch {e}")
                break

    print(f"[train] best val_acc={best:.3f}")
    print(f"[train] saved checkpoint: {args.out}")
    print(f"[train] saved TorchScript: {args.out_ts}")


if __name__ == "__main__":
    main()
