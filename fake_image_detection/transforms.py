from __future__ import annotations
import random
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np


class _RandomJPEG(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.3, quality: Tuple[int, int] = (50, 95), always_apply=False):
        super().__init__(p=p, always_apply=always_apply)
        self.quality = quality

    def apply(self, img, **params):
        q = random.randint(self.quality[0], self.quality[1])
        enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(enc[1], cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


class _RandomResize(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.3, scale=(0.5, 1.0), always_apply=False):
        super().__init__(p=p, always_apply=always_apply)
        self.scale = scale

    def apply(self, img, **params):
        h, w = img.shape[:2]
        s = random.uniform(self.scale[0], self.scale[1])
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return back


def build_transforms(cfg: Dict, train: bool):
    d = cfg.get("data", {})
    aug = cfg.get("aug", {})
    size = int(d.get("image_size", 384))
    mean = aug.get("normalize_mean", [0.485, 0.456, 0.406])
    std = aug.get("normalize_std", [0.229, 0.224, 0.225])
    ops: List = []
    if train:
        if aug.get("hflip", True):
            ops.append(A.HorizontalFlip(p=0.5))
        scale = aug.get("random_crop_scale", [0.8, 1.0])
        ops.append(
            A.RandomResizedCrop(size=(size, size), scale=tuple(scale), ratio=(0.85, 1.2), p=1.0)
        )
        if aug.get("color_jitter", 0.0) > 0:
            ops.append(
                A.ColorJitter(
                    brightness=aug["color_jitter"],
                    contrast=aug["color_jitter"],
                    p=0.5,
                )
            )
        if aug.get("jpeg_prob", 0.0) > 0:
            ops.append(_RandomJPEG(p=aug["jpeg_prob"], quality=tuple(aug.get("jpeg_quality", [50, 95]))))
        if aug.get("resize_prob", 0.0) > 0:
            ops.append(_RandomResize(p=aug["resize_prob"], scale=(0.5, 1.0)))
        if aug.get("blur_prob", 0.0) > 0:
            ops.append(A.GaussianBlur(p=aug["blur_prob"], blur_limits=(3, 5)))
    else:
        ops.append(A.Resize(size, size))
    ops.append(A.Normalize(mean=mean, std=std))
    return A.Compose(ops)
