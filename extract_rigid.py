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

from fake_image_detection.feature_extractor import ImageDataset, build_clip_transform
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv

CR = resolve_code_root()


@torch.no_grad()
def extract_rigid(model, ids, rel_root, size, device, noise_std=0.1, batch_size=48, num_workers=8):
    """RIGID: 真实图对噪声扰动的 DINO 特征相似度高于 AI 图。
    返回每张图：原图与加噪图特征的余弦相似度（越高越像真实图）。"""
    tf = build_clip_transform(size, "crop")
    ds = ImageDataset(ids, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    sims = []
    for imgs, _ in tqdm(dl, desc=f"rigid {rel_root}"):
        imgs = imgs.to(device, non_blocking=True)
        noise = torch.randn_like(imgs) * noise_std
        noisy = (imgs + noise).clamp(-3.0, 3.0)  # normalize 后的合理范围
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            f1 = model.forward_head(model.forward_features(imgs), pre_logits=True)
            f2 = model.forward_head(model.forward_features(noisy), pre_logits=True)
        f1 = F.normalize(f1.float(), dim=-1)
        f2 = F.normalize(f2.float(), dim=-1)
        sim = (f1 * f2).sum(dim=-1)  # 余弦相似度
        sims.append(sim.cpu().numpy())
    return np.concatenate(sims, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch16_dinov3")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out-prefix", default="rigid_dinoL")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import timm
    model = timm.create_model(args.model, pretrained=True, num_classes=0).to(device).eval()

    FEAT = CR / "outputs" / "features"
    FEAT.mkdir(parents=True, exist_ok=True)
    for split, csv, rel in [("sample", sample_csv(), "data/image_sample_data"), ("test", test_csv(), "data/image_test")]:
        df = pd.read_csv(csv)
        sim = extract_rigid(model, df["id"].tolist(), rel, args.size, device, args.noise_std, args.batch_size)
        np.save(FEAT / f"{args.out_prefix}_{args.size}_{split}.npy", sim.astype(np.float32))
        print(f"saved {args.out_prefix}_{args.size}_{split} {sim.shape} mean={sim.mean():.4f}")


if __name__ == "__main__":
    main()
