from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fake_image_detection.cf_model import load_commfor
from fake_image_detection.muon import SingleDeviceMuonWithAuxAdam
from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()
DATA_ROOT = resolve_data_root()
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_tf(size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    resize = 440 if size == 384 else 256
    return A.Compose([
        A.SmallestMaxSize(max_size=resize),
        A.RandomCrop(size, size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ImageCompression(compression_type='jpeg', quality_range=(40, 95), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),
        A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_eval_tf(size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    resize = 440 if size == 384 else 256
    return A.Compose([
        A.Resize(resize, resize),
        A.CenterCrop(size, size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class ImgDS(Dataset):
    def __init__(self, ids, labels, rel_root, tf):
        self.ids = list(ids)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.rel_root = rel_root
        self.tf = tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        try:
            img = np.array(Image.open(DATA_ROOT / self.rel_root / sid).convert("RGB"))
        except Exception:
            img = np.zeros((384, 384, 3), dtype=np.uint8)
        x = self.tf(image=img)["image"]
        if self.labels is None:
            return x, sid
        return x, float(self.labels[i]), sid


class TimmDetector(nn.Module):
    """timm 骨干 + 二分类头，输出单个 logit。"""

    def __init__(self, name, num_classes=1, head_dropout=0.1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(head_dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        feat = self.backbone(x)


class CLIPDetector(nn.Module):
    """open_clip 骨干 + 二分类头，输出单个 logit。"""

    def __init__(self, arch, ckpt=None, num_classes=1, head_dropout=0.1):
        super().__init__()
        import sys
        sys.path.insert(0, str(CR))
        from fake_image_detection.clip_model import load_openclip
        self.backbone = load_openclip(arch, ckpt, torch.device("cpu"))
        feat_dim = self.backbone.visual.output_dim if hasattr(self.backbone, "visual") else 1024
        self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(head_dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        feat = self.backbone.encode_image(x)
        return self.head(feat).reshape(-1)


def load_backbone(name, device):
    """返回 (model, size, mean, std)。model 输出单个 logit。"""
    if name == "cf384":
        model, size = load_commfor("weights/commfor/commfor-model-384", device)
        for p in model.parameters():
            p.requires_grad = True
        return model, size, IMAGENET_MEAN, IMAGENET_STD
    if name == "cf224":
        model, size = load_commfor("weights/commfor/commfor-model-224", device)
        for p in model.parameters():
            p.requires_grad = True
        return model, size, IMAGENET_MEAN, IMAGENET_STD
    if name == "dinoL":
        return TimmDetector("vit_large_patch16_dinov3").to(device), 224, IMAGENET_MEAN, IMAGENET_STD
    if name == "dinoB":
        return TimmDetector("vit_base_patch16_dinov3").to(device), 224, IMAGENET_MEAN, IMAGENET_STD
    if name == "clipH":
        ckpt = CR / "weights" / "clip-vit-h-14" / "open_clip_pytorch_model.bin"
        return CLIPDetector("ViT-H-14", str(ckpt)).to(device), 224, CLIP_MEAN, CLIP_STD
    if name == "clipBigG":
        ckpt = CR / "weights" / "clip-vit-bigg-14" / "open_clip_pytorch_model.bin"
        return CLIPDetector("ViT-bigG-14", str(ckpt)).to(device), 224, CLIP_MEAN, CLIP_STD
    raise ValueError(f"unknown backbone {name}")


def make_optimizer(model, lr_backbone, lr_head, optimizer="muon"):
    hidden, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in n:
            other.append(p)  # head 用 AdamW，较大 lr
        elif p.ndim >= 2 and "pos_embed" not in n and "cls_token" not in n:
            hidden.append(p)  # 隐藏矩阵权重用 Muon
        else:
            other.append(p)  # embedding/norm/bias 用 AdamW
    if optimizer == "adamw":
        return torch.optim.AdamW([
            {"params": hidden, "lr": lr_backbone, "weight_decay": 0.01},
            {"params": other, "lr": lr_head, "weight_decay": 0.01},
        ])
    groups = [
        dict(params=hidden, use_muon=True, lr=lr_backbone, weight_decay=0.01),
        dict(params=other, use_muon=False, lr=lr_head, betas=(0.9, 0.95), weight_decay=0.01),
    ]
    return SingleDeviceMuonWithAuxAdam(groups)


def train_one_epoch(model, loader, opt, device, scaler):
    model.train()
    total, n = 0.0, 0
    for imgs, y, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logit = model(imgs).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, ids, rel_root, tf, device, batch_size=64, tta=False):
    model.eval()
    ds = ImgDS(ids, None, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    outs = []
    for imgs, _ in dl:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            p = torch.sigmoid(model(imgs).float().reshape(-1))
            if tta:
                p2 = torch.sigmoid(model(imgs.flip(-1)).float().reshape(-1))
                p = (p + p2) / 2
        outs.append(p.cpu().numpy())
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cf384")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=0.01)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--fold", type=int, default=-1, help="-1 = all folds")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"ft_{args.backbone}"
    lab = pd.read_csv(sample_csv())
    y_all = lab["label"].astype(int).to_numpy()
    ids_all = lab["id"].tolist()
    test_ids = pd.read_csv(test_csv())["id"].tolist()

    skf = StratifiedKFold(args.n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(ids_all, y_all))
    run_folds = range(args.n_splits) if args.fold == -1 else [args.fold]

    oof = np.zeros(len(y_all), dtype=np.float64)
    test_preds = []
    for fi in run_folds:
        tr_idx, va_idx = folds[fi]
        ids_tr = [ids_all[i] for i in tr_idx]
        y_tr = y_all[tr_idx]
        ids_va = [ids_all[i] for i in va_idx]
        y_va = y_all[va_idx]
        print(f"=== fold {fi}: train={len(ids_tr)} val={len(ids_va)} ===")

        model, size, mean, std = load_backbone(args.backbone, device)
        opt = make_optimizer(model, args.lr_backbone, args.lr_head, args.optimizer)
        scaler = torch.cuda.amp.GradScaler()
        tf_tr = build_train_tf(size, mean, std)
        tf_ev = build_eval_tf(size, mean, std)
        dl_tr = DataLoader(ImgDS(ids_tr, y_tr, "data/image_sample_data", tf_tr),
                           batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)

        best_auc, best_state = -1.0, None
        for ep in range(1, args.epochs + 1):
            loss = train_one_epoch(model, dl_tr, opt, device, scaler)
            p_va = predict(model, ids_va, "data/image_sample_data", tf_ev, device)
            auc = roc_auc_score(y_va, p_va)
            print(f"  ep{ep}/{args.epochs} loss={loss:.4f} val_auc={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"  fold {fi} best val_auc={best_auc:.4f}")

        # OOF 预测（含 TTA）
        oof[va_idx] = predict(model, ids_va, "data/image_sample_data", tf_ev, device, tta=args.tta)
        # 测试预测
        test_preds.append(predict(model, test_ids, "data/image_test", tf_ev, device, tta=args.tta))
        # 保存 checkpoint
        ck = CR / "outputs" / "checkpoints" / tag / f"fold{fi}.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "size": size, "val_auc": best_auc}, ck)

    overall_auc = roc_auc_score(y_all, oof)
    print(f"=== OOF AUC = {overall_auc:.4f} ===")
    oof_df = pd.DataFrame({"id": ids_all, "prob": oof, "label": y_all})
    oof_df.to_csv(CR / "outputs" / "oof" / f"{tag}_oof.csv", index=False)

    test_mean = np.mean(test_preds, axis=0)
    out = pd.DataFrame({"id": test_ids, "score": test_mean})
    out_path = CR / "outputs" / "predictions" / f"{tag}_test.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"test predictions (mean of {len(test_preds)} folds) -> {out_path} mean={test_mean.mean():.4f}")


if __name__ == "__main__":
    main()
