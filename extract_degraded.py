from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv
from fake_image_detection.feature_extractor import build_cf_transform, build_clip_transform, IMAGENET_MEAN, IMAGENET_STD

CR = resolve_code_root()
DATA_ROOT = resolve_data_root()
FEAT = CR / "outputs" / "features"


_DEG = A.OneOf([
    A.ImageCompression(compression_type='jpeg', quality_range=(30, 90), p=1.0),
    A.Downscale(scale_range=(0.3, 0.9), p=1.0),
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.GaussNoise(std_range=(0.02, 0.10), p=1.0),
], p=1.0)


class DegDS(Dataset):
    """先对图像施加随机退化（模拟对抗测试），再做 backbone 预处理（兼容 albumentations/torchvision）。"""

    def __init__(self, ids, rel_root, prep_tf):
        self.ids = list(ids)
        self.rel_root = rel_root
        self.prep_tf = prep_tf
        self._is_albu = isinstance(prep_tf, A.Compose)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        try:
            pil = Image.open(DATA_ROOT / self.rel_root / sid).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (384, 384))
        img = _DEG(image=np.array(pil))["image"]  # 先退化
        if self._is_albu:
            out = self.prep_tf(image=img)["image"]
        else:
            out = self.prep_tf(Image.fromarray(img))
        return out, sid


@torch.no_grad()
def extract_cf(model, ids, rel, size, is_crop, device, bs=64):
    prep = build_cf_transform(size, is_crop)
    dl = DataLoader(DegDS(ids, rel, prep), batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    feats = []
    for imgs, _ in tqdm(dl, desc=f"cf-deg {rel}@{size}"):
        imgs = imgs.to(device, non_blocking=True)
        ft = model.vit.forward_features(imgs)
        pooled = model.vit.forward_head(ft, pre_logits=True)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, 0)


@torch.no_grad()
def extract_clip(model, ids, rel, size, device, bs=48):
    prep = build_clip_transform(size, "crop")
    dl = DataLoader(DegDS(ids, rel, prep), batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    feats = []
    for imgs, _ in tqdm(dl, desc=f"clip-deg {rel}@{size}"):
        imgs = imgs.to(device, non_blocking=True)
        f = F.normalize(model.encode_image(imgs).float(), dim=-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0)


@torch.no_grad()
def extract_dino(model, ids, rel, size, device, bs=48):
    prep = build_clip_transform(size, "crop")
    dl = DataLoader(DegDS(ids, rel, prep), batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    feats = []
    for imgs, _ in tqdm(dl, desc=f"dino-deg {rel}@{size}"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            ft = model.forward_features(imgs)
            pooled = model.forward_head(ft, pre_logits=True)
        feats.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["cf384", "cf224", "clipH", "clipBigG", "clipH378", "dinoL", "dinoB"])
    ap.add_argument("--n-rounds", type=int, default=1, help="退化轮数（多轮=更多退化训练样本）")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lab = pd.read_csv(sample_csv())
    ids = lab["id"].tolist()

    name = args.backbone
    if name == "cf384":
        from fake_image_detection.cf_model import load_commfor
        model, size = load_commfor("weights/commfor/commfor-model-384", device)
        tag, size = "commfor_cf384", 384
        for r in range(args.n_rounds):
            torch.manual_seed(r); np.random.seed(r)
            f = extract_cf(model, ids, "data/image_sample_data", size, True, device)
            np.save(FEAT / f"{tag}_{size}_crop_deg{r}_sample.npy", f)
            print(f"saved {tag}_{size}_crop_deg{r}_sample {f.shape}")
    elif name == "cf224":
        from fake_image_detection.cf_model import load_commfor
        model, size = load_commfor("weights/commfor/commfor-model-224", device)
        tag, size = "commfor_cf224", 224
        for r in range(args.n_rounds):
            torch.manual_seed(r); np.random.seed(r)
            f = extract_cf(model, ids, "data/image_sample_data", size, True, device)
            np.save(FEAT / f"{tag}_{size}_crop_deg{r}_sample.npy", f)
            print(f"saved {tag}_{size}_crop_deg{r}_sample {f.shape}")
    elif name in ("clipH", "clipBigG", "clipH378"):
        from fake_image_detection.clip_model import load_openclip
        arch, ckpt, tag, size = {
            "clipH": ("ViT-H-14", "weights/clip-vit-h-14/open_clip_pytorch_model.bin", "clipH", 224),
            "clipBigG": ("ViT-bigG-14", "weights/clip-vit-bigg-14/open_clip_pytorch_model.bin", "clipBigG", 224),
            "clipH378": ("ViT-H-14-378-quickgelu", "weights/clip-vit-h14-378/open_clip_pytorch_model.bin", "clipH378", 378),
        }[name]
        model = load_openclip(arch, ckpt, device)
        for r in range(args.n_rounds):
            torch.manual_seed(r); np.random.seed(r)
            f = extract_clip(model, ids, "data/image_sample_data", size, device)
            np.save(FEAT / f"{tag}_{size}_crop_deg{r}_sample.npy", f)
            print(f"saved {tag}_{size}_crop_deg{r}_sample {f.shape}")
    elif name in ("dinoL", "dinoB"):
        import timm
        mname, tag = {"dinoL": ("vit_large_patch16_dinov3", "dinoL"), "dinoB": ("vit_base_patch16_dinov3", "dinoB")}[name]
        model = timm.create_model(mname, pretrained=True, num_classes=0).to(device).eval()
        for r in range(args.n_rounds):
            torch.manual_seed(r); np.random.seed(r)
            f = extract_dino(model, ids, "data/image_sample_data", 224, device)
            np.save(FEAT / f"{tag}_224_crop_deg{r}_sample.npy", f)
            print(f"saved {tag}_224_crop_deg{r}_sample {f.shape}")


if __name__ == "__main__":
    main()
