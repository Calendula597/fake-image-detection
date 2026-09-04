"""提取复赛测试集的水平翻转特征（TTA 用）。

对 4 个骨干（cf384/clipH378/clipBigG/dinoL）的 crop 特征，先 hflip 再过同样的预处理，
存为 outputs/features/<feat_tag>_<size>_crop_testflip.npy，供 stack.py 的 fluxaug 头做 TTA 平均。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from fake_image_detection.paths import resolve_code_root, resolve_data_root, test_csv
from fake_image_detection.feature_extractor import build_cf_transform, build_clip_transform
from extract_aug import load_backbone
import pandas as pd

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"

BACKBONES = [
    ("cf384", "commfor_cf384", 384),
    ("clipH378", "clipH378", 378),
    ("clipBigG", "clipBigG", 224),
    ("dinoL", "dinoL", 224),
]


class FlipDataset(Dataset):
    """读图 → hflip → transform（兼容 albumentations / torchvision）。"""

    def __init__(self, paths, tf, is_albu):
        self.paths = paths
        self.tf = tf
        self.is_albu = is_albu

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            pil = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (384, 384))
        pil = ImageOps.mirror(pil)
        if self.is_albu:
            return self.tf(image=np.array(pil))["image"], str(self.paths[i])
        return self.tf(pil), str(self.paths[i])


@torch.no_grad()
def main():
    import torch.nn.functional as F
    import albumentations as A

    device = torch.device("cuda:0")
    data = resolve_data_root()
    ids = pd.read_csv(test_csv())["id"].tolist()
    paths = [str(data / "data" / "image_test" / i) for i in ids]

    for tag, feat_tag, size in BACKBONES:
        out = FEAT / f"{feat_tag}_{size}_crop_testflip.npy"
        if out.exists():
            print(f"{tag}: exists, skip")
            continue
        model, tf, kind = load_backbone(tag, device)
        is_albu = isinstance(tf, A.Compose)
        dl = DataLoader(FlipDataset(paths, tf, is_albu), batch_size=64, shuffle=False, num_workers=8)
        feats = []
        for imgs, _ in dl:
            imgs = imgs.to(device)
            if kind == "cf":
                ft = model.vit.forward_features(imgs)
                feats.append(model.vit.forward_head(ft, pre_logits=True).float().cpu().numpy())
            else:
                feats.append(F.normalize(model.encode_image(imgs).float(), dim=-1).cpu().numpy())
        np.save(out, np.concatenate(feats, 0).astype(np.float32))
        print(f"{tag}: saved {out}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
