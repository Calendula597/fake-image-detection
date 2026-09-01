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
def _forward(model, x):
    ft = model.forward_features(x)
    return model.forward_head(ft, pre_logits=True)


@torch.no_grad()
def extract_wepe(model, ids, rel_root, size, device, noise_std, batch_size=48, num_workers=8, seed=0):
    """WePe: 扰动模型权重，真图特征稳定、生成图特征漂移。
    score = cos_sim(干净模型特征, 扰动模型特征)，越高越像真实图。"""
    tf = build_clip_transform(size, "crop")
    ds = ImageDataset(ids, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 备份原权重并生成扰动
    orig = {k: v.detach().clone() for k, v in model.state_dict().items()}
    g = torch.Generator(device="cpu").manual_seed(seed)
    perturbed = {}
    for k, v in orig.items():
        if v.dtype.is_floating_point and v.ndim >= 2:
            scale = (v.norm() / (v.numel() ** 0.5)).item() * noise_std
            noise = torch.randn(v.shape, generator=g).to(v.device, v.dtype) * scale
            perturbed[k] = v + noise
        else:
            perturbed[k] = v

    sims = []
    for imgs, _ in tqdm(dl, desc=f"wepe {rel_root}"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            f_clean = _forward(model, imgs).float()
        # 加载扰动权重
        model.load_state_dict(perturbed)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            f_pert = _forward(model, imgs).float()
        # 恢复原权重
        model.load_state_dict(orig)
        f_clean = F.normalize(f_clean, dim=-1)
        f_pert = F.normalize(f_pert, dim=-1)
        sims.append((f_clean * f_pert).sum(-1).cpu().numpy())
    return np.concatenate(sims, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch16_dinov3")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out-prefix", default="wepe_dinoL")
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import timm
    model = timm.create_model(args.model, pretrained=True, num_classes=0).to(device).eval()

    FEAT = CR / "outputs" / "features"
    FEAT.mkdir(parents=True, exist_ok=True)
    splits = [("sample", sample_csv(), "data/image_sample_data")]
    if not args.sample_only:
        splits.append(("test", test_csv(), "data/image_test"))
    for split, csv, rel in splits:
        df = pd.read_csv(csv)
        sim = extract_wepe(model, df["id"].tolist(), rel, args.size, device, args.noise_std, args.batch_size)
        np.save(FEAT / f"{args.out_prefix}_{args.size}_{split}.npy", sim.astype(np.float32))
        print(f"saved {args.out_prefix}_{args.size}_{split} {sim.shape} mean={sim.mean():.4f}")


if __name__ == "__main__":
    main()
