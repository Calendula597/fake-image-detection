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
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from fake_image_detection.feature_extractor import ImageDataset, IMAGENET_MEAN, IMAGENET_STD, save_features
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv

CR = resolve_code_root()
FEAT_DIR = CR / "outputs" / "features"


class DRCTConvNeXt(nn.Module):
    def __init__(self, ckpt: str, embedding_size: int = 1024, num_classes: int = 2):
        super().__init__()
        import timm
        net = timm.create_model("convnext_base_in22k", pretrained=False)
        n = net.head.fc.in_features
        net.head.fc = nn.Linear(n, embedding_size)
        self.model = net
        self.fc = nn.Linear(embedding_size, num_classes)
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[DRCT] loaded {ckpt} | missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x, return_feature=False):
        feat = self.model(x)
        y = self.fc(feat)
        if return_feature:
            return y, feat
        return y


def build_transform(size=224, is_crop=True):
    if is_crop:
        ops = [
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=0),
            A.CenterCrop(size, size),
        ]
    else:
        ops = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=0),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


@torch.no_grad()
def extract(device, ids, rel_root, model, tf, batch_size=96, num_workers=8):
    model = model.to(device).eval()
    ds = ImageDataset(ids, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, logits = [], []
    for imgs, _ in tqdm(dl, desc=f"drct-feat {rel_root}"):
        imgs = imgs.to(device, non_blocking=True)
        y, f = model(imgs, return_feature=True)
        feats.append(f.float().cpu().numpy())
        logits.append(y.float().cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(logits, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-tag", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DRCTConvNeXt(args.ckpt)
    splits = [("test", pd.read_csv(test_csv()), "data/image_test")] if args.test_only else [
        ("sample", pd.read_csv(sample_csv()), "data/image_sample_data"),
        ("test", pd.read_csv(test_csv()), "data/image_test"),
    ]
    for crop in (True, False):
        mode = "crop" if crop else "resize"
        tf = build_transform(args.size, crop)
        for split, df, rel in splits:
            f, lg = extract(device, df["id"].tolist(), rel, model, tf, args.batch_size)
            save_features(FEAT_DIR, f"drct_{args.ckpt_tag}", args.size, mode, split, f)
            save_features(FEAT_DIR, f"drct_{args.ckpt_tag}", args.size, f"{mode}_logits", split, lg)
            print(f"saved drct_{args.ckpt_tag}_{args.size}_{mode}_{split} {f.shape}")


if __name__ == "__main__":
    main()
