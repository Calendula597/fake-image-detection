"""提取 DINOv3 的 MAC 特征（DINO-MAC, CVPRW 2026 NTIRE 深伪冠军思路）：

拼接 [CLS] + 4×[REG] register tokens + patch 均值 [AVG] → 6×dim 维特征，
比单 CLS/池化特征显著更强（冠军队消融：MAC 头贡献最大，集成只值 0.002）。

输出 outputs/features/dinomac{tag}_224_crop_{sample,test}.npy，供 stack.py 作为新成员。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv
from fake_image_detection.feature_extractor import ImageDataset, build_cf_transform

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"

MODELS = {
    "L": "vit_large_patch16_dinov3.lvd1689m",
    "H": "vit_huge_plus_patch16_dinov3.lvd1689m",
}


@torch.no_grad()
def extract_mac(model, ids, tf, device, bs=64):
    dl = DataLoader(ImageDataset(ids, "", tf), batch_size=bs, shuffle=False, num_workers=8)
    feats = []
    for imgs, _ in dl:
        imgs = imgs.to(device)
        out = model.forward_features(imgs)
        # timm dinov3: forward_features 返回 dict 或 tensor
        if isinstance(out, dict):
            cls = out.get("x_norm_clstoken")
            reg = out.get("x_storage_tokens")  # register tokens
            patch = out.get("x_norm_patchtokens")
            if cls is None:
                # 兼容：直接给完整 token 序列
                tokens = out.get("x", None)
                if tokens is None:
                    raise RuntimeError(f"unexpected forward_features keys: {out.keys()}")
        else:
            tokens = out
            n_reg = getattr(model, "n_storage_tokens", 4)
            cls = tokens[:, 0]
            reg = tokens[:, 1:1 + n_reg]
            patch = tokens[:, 1 + n_reg:]
        avg = patch.mean(dim=1)
        if reg is None or reg.numel() == 0:
            mac = torch.cat([cls, avg], dim=-1)
        else:
            mac = torch.cat([cls, reg.flatten(1), avg], dim=-1)
        feats.append(mac.float().cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["L", "H"])
    args = ap.parse_args()
    import timm

    device = torch.device("cuda:0")
    data = resolve_data_root()
    lab = pd.read_csv(sample_csv())
    te = pd.read_csv(test_csv())
    train_ids = [str(data / "data" / "image_sample_data" / i) for i in lab["id"]]
    test_ids = [str(data / "data" / "image_test" / i) for i in te["id"]]
    tf = build_cf_transform(224, True)

    for tag in args.tags:
        out_s = FEAT / f"dinomac{tag}_224_crop_sample.npy"
        out_t = FEAT / f"dinomac{tag}_224_crop_test.npy"
        if out_s.exists() and out_t.exists():
            print(f"{tag}: exists, skip")
            continue
        model = timm.create_model(MODELS[tag], pretrained=True).to(device).eval()
        n_reg = getattr(model, "n_storage_tokens", None)
        print(f"{tag}: n_storage_tokens={n_reg}", flush=True)
        if not out_s.exists():
            np.save(out_s, extract_mac(model, train_ids, tf, device).astype(np.float32))
            print(f"saved {out_s}", flush=True)
        if not out_t.exists():
            np.save(out_t, extract_mac(model, test_ids, tf, device).astype(np.float32))
            print(f"saved {out_t}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
