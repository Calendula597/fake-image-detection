from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from fake_image_detection.feature_extractor import load_features
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv
from fake_image_detection.stacking import rank01

CR = resolve_code_root()
PRED = CR / "outputs" / "predictions"
PRED.mkdir(parents=True, exist_ok=True)


def train_head(Xtr, ytr, wtr, device, epochs=80, hidden=256, lr=0.001, wd=0.0001):
    in_dim = Xtr.shape[1]
    if hidden and hidden > 0:
        head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
    else:
        head = nn.Linear(in_dim, 1)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    X = torch.from_numpy(Xtr).to(device)
    y = torch.from_numpy(ytr).to(device)
    w = torch.from_numpy(wtr).to(device)
    n = len(X)
    bs = 256
    head.train()
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            logit = head(X[b]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y[b], reduction="none")
            loss = (loss * w[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict_head(head, X, device, batch=4096):
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        p = torch.sigmoid(head(xb).squeeze(-1)).cpu().numpy()
        out.append(p)
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True, help="feature tag, e.g. commfor_cf384 or clipH")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--mode", default="crop", choices=["crop", "resize"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEAT = CR / "outputs" / "features"

    fs = load_features(FEAT, args.feature, args.size, args.mode, "sample")
    ft = load_features(FEAT, args.feature, args.size, args.mode, "test")
    if fs is None or ft is None:
        raise FileNotFoundError(f"features not found for {args.feature}_{args.size}_{args.mode}")

    lab = pd.read_csv(sample_csv())
    ylab = lab["label"].values.astype(np.float32)
    ids_test = pd.read_csv(test_csv())["id"].tolist()

    # 归一化
    fs_n = fs / (np.linalg.norm(fs, axis=1, keepdims=True) + 1e-8)
    ft_n = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-8)

    head = train_head(fs_n, ylab, np.ones(len(fs_n), np.float32), device, epochs=args.epochs, hidden=args.hidden)
    cur = predict_head(head, ft_n, device)
    pa = predict_head(head, fs_n, device)
    auc = roc_auc_score(ylab, pa)
    print(f"[round0] trainAUC={auc:.4f}")
    sign = 1.0 if auc >= 0.5 else -1.0
    if sign < 0:
        cur = 1.0 - cur

    thresholds = [(0.95, 0.05), (0.92, 0.08), (0.9, 0.1), (0.85, 0.15)]
    for rnd in range(1, args.rounds + 1):
        p_hi, p_lo = thresholds[min(rnd - 1, len(thresholds) - 1)]
        conf = (cur >= p_hi) | (cur <= p_lo)
        plabel = (cur >= p_hi).astype(np.float32)
        Xtr = np.concatenate([fs_n, ft_n[conf]], 0)
        ytr = np.concatenate([ylab, plabel[conf]], 0)
        wtr = np.concatenate(
            [np.ones(len(fs_n), np.float32), np.ones(int(conf.sum()), np.float32) * 0.5], 0
        )
        head = train_head(Xtr, ytr, wtr, device, epochs=args.epochs, hidden=args.hidden)
        cur = predict_head(head, ft_n, device)
        if sign < 0:
            cur = 1.0 - cur
        pa = predict_head(head, fs_n, device)
        if sign < 0:
            pa = 1.0 - pa
        print(f"[round{rnd}] trainAUC={roc_auc_score(ylab, pa):.4f} pseudo={int(conf.sum())}")

    out = args.out or PRED / f"{args.feature}_{args.size}_{args.mode}_st.csv"
    pd.DataFrame({"id": ids_test, "score": cur}).to_csv(out, index=False)
    print(f"wrote {out} mean={cur.mean():.4f}")


if __name__ == "__main__":
    main()
