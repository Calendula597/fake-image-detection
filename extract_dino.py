from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fake_image_detection.feature_extractor import ImageDataset, build_clip_transform, save_features
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv

CR = resolve_code_root()
FEAT_DIR = CR / "outputs" / "features"


@torch.no_grad()
def extract(model, ids, rel_root, size, mode, device, batch_size=48, num_workers=8):
    tf = build_clip_transform(size, mode)  # torchvision，ImageNet mean/std（DINO 兼容）
    ds = ImageDataset(ids, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    for imgs, _ in tqdm(dl, desc=f"dino {rel_root}@{size}/{mode}"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            ft = model.forward_features(imgs)
            pooled = model.forward_head(ft, pre_logits=True)
        pooled = F.normalize(pooled.float(), dim=-1)
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch16_dinov3")
    ap.add_argument("--tag", default="dinoL")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import timm
    model = timm.create_model(args.model, pretrained=True, num_classes=0).to(device).eval()
    print(f"loaded {args.model} num_features={model.num_features}")

    splits = [("test", pd.read_csv(test_csv()), "data/image_test")] if args.test_only else [
        ("sample", pd.read_csv(sample_csv()), "data/image_sample_data"),
        ("test", pd.read_csv(test_csv()), "data/image_test"),
    ]
    for mode in ("crop", "resize"):
        for split, df, rel in splits:
            f = extract(model, df["id"].tolist(), rel, args.size, mode, device, args.batch_size)
            save_features(FEAT_DIR, args.tag, args.size, mode, split, f)
            print(f"saved {args.tag}_{args.size}_{mode}_{split} {f.shape}")


if __name__ == "__main__":
    main()
