from __future__ import annotations
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .paths import resolve_data_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageDataset(Dataset):
    """同时兼容 albumentations（image= 关键字，返回 dict）和 torchvision（位置参数，返回 tensor）。"""

    def __init__(self, ids, rel_root: str, transform: Callable):
        self.ids = list(ids)
        self.rel_root = rel_root
        self.transform = transform
        self.data_root = resolve_data_root()
        self._is_albu = isinstance(transform, A.Compose)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        try:
            pil = Image.open(self.data_root / self.rel_root / sid).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (384, 384))
        if self._is_albu:
            img = np.array(pil)
            out = self.transform(image=img)["image"]
        else:
            out = self.transform(pil)
        return out, sid


def build_cf_transform(size: int = 384, is_crop: bool = True):
    """Community Forensics 预处理。"""
    if size == 384:
        resize_size = 440
    elif size == 224:
        resize_size = 256
    else:
        resize_size = int(round(size * 440 / 384))
    if is_crop:
        ops = [A.Resize(resize_size, resize_size), A.CenterCrop(size, size)]
    else:
        ops = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=0),
        ]
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def build_clip_transform(size: int = 224, mode: Literal["crop", "resize"] = "crop"):
    """CLIP/OpenCLIP 预处理。"""
    from torchvision import transforms as T

    if mode == "crop":
        return T.Compose(
            [
                T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


@torch.no_grad()
def extract_cf_features(
    model,
    ids,
    rel_root: str,
    size: int,
    is_crop: bool,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取 Community Forensics 特征和 logits。"""
    tf = build_cf_transform(size, is_crop)
    ds = ImageDataset(ids, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, logits = [], []
    for imgs, _ in tqdm(dl, desc=f"cf-feat {rel_root}@{size}"):
        imgs = imgs.to(device, non_blocking=True)
        feat_tokens = model.vit.forward_features(imgs)
        pooled = model.vit.forward_head(feat_tokens, pre_logits=True)
        logit = model.vit.head(pooled)
        feats.append(pooled.float().cpu().numpy())
        logits.append(logit.float().cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(logits, 0)


@torch.no_grad()
def extract_clip_features(
    model,
    ids,
    rel_root: str,
    size: int,
    mode: Literal["crop", "resize"],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
) -> np.ndarray:
    """提取 CLIP/OpenCLIP 图像特征。"""
    import torch.nn.functional as F

    tf = build_clip_transform(size, mode)
    ds = ImageDataset(ids, rel_root, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    for imgs, _ in tqdm(dl, desc=f"clip-feat {rel_root}@{size}/{mode}"):
        imgs = imgs.to(device, non_blocking=True)
        f = F.normalize(model.encode_image(imgs).float(), dim=-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0)


def save_features(feat_dir: Path, tag: str, size: int, mode: str, split: str, arr: np.ndarray):
    feat_dir.mkdir(parents=True, exist_ok=True)
    path = feat_dir / f"{tag}_{size}_{mode}_{split}.npy"
    np.save(path, arr)
    return path


def load_features(feat_dir: Path, tag: str, size: int, mode: str, split: str) -> Optional[np.ndarray]:
    path = feat_dir / f"{tag}_{size}_{mode}_{split}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return None
