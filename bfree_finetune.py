from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fake_image_detection.cf_model import load_commfor
from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv

CR = resolve_code_root()
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_tf(size):
    resize = 440 if size == 384 else 256
    return A.Compose([
        A.SmallestMaxSize(max_size=resize), A.RandomCrop(size, size), A.HorizontalFlip(p=0.5),
        A.OneOf([A.ImageCompression(compression_type='jpeg', quality_range=(40, 95), p=1.0),
                 A.GaussNoise(std_range=(0.02, 0.08), p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.5),
        A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(),
    ])


def build_eval_tf(size):
    resize = 440 if size == 384 else 256
    return A.Compose([A.Resize(resize, resize), A.CenterCrop(size, size),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class PairDS(Dataset):
    def __init__(self, items, tf):
        self.items = items  # list of (path, label)
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        try: img = np.array(Image.open(p).convert("RGB"))
        except Exception: img = np.zeros((384, 384, 3), dtype=np.uint8)
        return self.tf(image=img)["image"], float(y), i


def train_epoch(model, loader, opt, scaler, device):
    model.train(); tot, n = 0.0, 0
    for imgs, y, _ in loader:
        imgs = imgs.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(model(imgs).reshape(-1), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, items, tf, device, bs=64):
    model.eval()
    dl = DataLoader(PairDS(items, tf), batch_size=bs, shuffle=False, num_workers=6)
    outs = []
    for imgs, _, _ in dl:
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outs.append(torch.sigmoid(model(imgs).float().reshape(-1)).cpu().numpy())
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--include-old-ai", action="store_true", help="把老AI也作为训练负类")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    data_root = resolve_data_root()
    lab = pd.read_csv(sample_csv())
    real_ids = lab[lab.label == 0]["id"].tolist()
    ai_ids = lab[lab.label == 1]["id"].tolist()
    real_items = [(data_root / "data" / "image_sample_data" / sid, 0) for sid in real_ids]
    fake_items = [(CR / "outputs" / "bfree_fakes" / sid, 1) for sid in real_ids]
    if args.include_old_ai:
        fake_items += [(data_root / "data" / "image_sample_data" / sid, 1) for sid in ai_ids]
    items = real_items + fake_items
    y_all = np.array([y for _, y in items])

    # 跨生成器评估集：原始 1000（真 vs 老AI）
    eval_items = [(data_root / "data" / "image_sample_data" / sid, 0) for sid in real_ids] + \
                 [(data_root / "data" / "image_sample_data" / sid, 1) for sid in ai_ids]
    y_eval = np.array([y for _, y in eval_items])

    skf = StratifiedKFold(args.n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y_all))
    eval_probs = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(items, y_all)):
        model, size = load_commfor("weights/commfor/commfor-model-384", device)
        for p in model.parameters(): p.requires_grad = True
        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if "head" in n], "lr": args.lr * 10},
        ], weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        tr_items = [items[i] for i in tr_idx]; va_items = [items[i] for i in va_idx]
        dl = DataLoader(PairDS(tr_items, build_train_tf(size)), batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)
        best_auc, best_state = -1.0, None
        for ep in range(args.epochs):
            loss = train_epoch(model, dl, opt, scaler, device)
            pv = predict(model, va_items, build_eval_tf(size), device)
            auc = roc_auc_score([y for _, y in va_items], pv)
            if auc > best_auc:
                best_auc, best_state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state: model.load_state_dict(best_state)
        oof[va_idx] = predict(model, va_items, build_eval_tf(size), device)
        # 跨生成器评估
        pe = predict(model, eval_items, build_eval_tf(size), device)
        eval_probs.append(pe)
        print(f"fold {fi}: train-auc={best_auc:.4f} cross-gen(old-AI)={roc_auc_score(y_eval, pe):.4f}")
    eval_mean = np.mean(eval_probs, 0)
    print(f"=== B-Free finetune: in-dist OOF={roc_auc_score(y_all, oof):.4f}  cross-gen(old-AI)={roc_auc_score(y_eval, eval_mean):.4f} ===")


if __name__ == "__main__":
    main()
