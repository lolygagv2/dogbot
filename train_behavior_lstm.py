import os, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split

BEHAVIORS = ["stand","sit","lie","cross","spin"]
T = 16  # window length

def windows(x, t=T):
    # x: [L,24,3] -> list of [t,48]
    x = x[...,:2].reshape(len(x), -1)  # drop conf, flatten 24x2=48
    out=[]
    if len(x) < t: return out
    for i in range(0, len(x)-t+1):
        out.append(x[i:i+t])
    return out

def load_sequences(folder):
    X, y = [], []
    for p in glob.glob(os.path.join(folder, "*.npz")):
        z = np.load(p)
        xs = windows(z["kpts"])
        for w in xs:
            X.append(w)
            y.append(int(z["label"]))
    X = np.array(X, dtype=np.float32)  # [N,T,48]
    y = np.array(y, dtype=np.int64)
    return X, y

class Head(nn.Module):
    def __init__(self, in_dim=48, hid=64, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hid, num_classes)
    def forward(self, x):          # x: [B,T,48]
        o,_ = self.lstm(x)         # [B,T,H]
        h = o[:,-1]                # last step
        return self.fc(h)          # [B,C]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)    # sequences/*.npz
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="behavior_head.pt")
    args = ap.parse_args()

    X, y = load_sequences(args.data)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Head().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    def run_epoch(Xd, yd, train=True):
        model.train(train)
        tot=0; correct=0; n=0
        for i in range(0, len(Xd), args.bs):
            xb = torch.tensor(Xd[i:i+args.bs]).to(device)
            yb = torch.tensor(yd[i:i+args.bs]).to(device)
            if train:
                opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            if train:
                loss.backward(); opt.step()
            pred = logits.argmax(1)
            correct += (pred==yb).sum().item()
            tot += loss.item()*len(xb); n += len(xb)
        return tot/n, correct/n

    best = 0.0; patience=5; bad=0
    for e in range(1, args.epochs+1):
        tr_loss, tr_acc = run_epoch(Xtr, ytr, True)
        va_loss, va_acc = run_epoch(Xva, yva, False)
        print(f"epoch {e}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
        if va_acc > best:
            best = va_acc; bad=0
            torch.save({"state_dict": model.state_dict(), "behaviors": BEHAVIORS, "T": T}, args.out)
        else:
            bad += 1
            if bad >= patience: break
    print("saved best to", args.out, "val_acc=", best)
