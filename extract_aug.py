"""提取扩增训练池特征：COCO 真实图（label 0）+ FLUX 假图（label 1）。

对每个强骨干提取特征存为 outputs/features/aug_<tag>_<split>.npy（split=coco/flux），
供 stack.py 的 aug 头成员使用：在 [1000 训练图 + COCO 真 + FLUX 假] 上训练检测头，
对齐测试集的新生代流匹配生成器分布（NTIRE 2026 冠军策略的本地版）。

可重复运行：已提取的 image id 会跳过（按 ids csv 增量）。
"""
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

from fake_image_detection.paths import resolve_code_root
from fake_image_detection.feature_extractor import ImageDataset, build_cf_transform, build_clip_transform

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"

BACKBONES = ["cf384", "clipH378", "clipBigG", "dinoL"]


@torch.no_grad()
def extract(model, ids, tf, kind, device, bs=32):
    dl = DataLoader(ImageDataset(ids, "", tf), batch_size=bs, shuffle=False, num_workers=4)
    out = []
    for imgs, _ in dl:
        imgs = imgs.to(device)
        if kind == "cf":
            ft = model.vit.forward_features(imgs)
            out.append(model.vit.forward_head(ft, pre_logits=True).float().cpu().numpy())
        else:
            f = model.encode_image(imgs)
            out.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 0), dtype=np.float32)


def load_backbone(tag, device):
    if tag == "cf384":
        from fake_image_detection.cf_model import load_commfor
        model, size = load_commfor("weights/commfor/commfor-model-384", device)
        return model, build_cf_transform(384, True), "cf"
    if tag == "clipH378":
        from fake_image_detection.clip_model import load_openclip
        model = load_openclip("ViT-H-14-378-quickgelu", "weights/clip-vit-h14-378/open_clip_pytorch_model.bin", device)
        return model, build_clip_transform(378, "crop"), "clip"
    if tag == "clipBigG":
        from fake_image_detection.clip_model import load_openclip
        model = load_openclip("ViT-bigG-14", "weights/clip-vit-bigg-14/open_clip_pytorch_model.bin", device)
        return model, build_clip_transform(224, "crop"), "clip"
    if tag == "dinoL":
        import timm
        model = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True).to(device).eval()

        class _W:  # 包装成 cf 接口（forward_features + forward_head pre_logits）
            vit = model
        tf = build_cf_transform(224, True)
        return _W(), tf, "cf"
    raise ValueError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--real-dir", default="data_real/coco_val")
    ap.add_argument("--ai-dir", default="data_real/coco_val_ai_flux")
    ap.add_argument("--all-genimage", action="store_true",
                    help="处理 data_real/genimage_* 全部目录 + coco + flux")
    args = ap.parse_args()
    device = torch.device("cuda:0")

    if args.all_genimage:
        splits = {"coco": sorted((CR / args.real_dir).glob("*.jpg"))}
        for d in sorted((CR / "data_real").glob("*")):
            if d.is_dir() and d.name != args.real_dir.split("/")[-1] and list(d.glob("*.jpg")):
                splits[d.name.replace("coco_val_ai_", "").replace("genimage_", "")] = sorted(d.glob("*.jpg"))
    else:
        splits = {
            "coco": sorted((CR / args.real_dir).glob("*.jpg")),
            "flux": sorted((CR / args.ai_dir).glob("*.jpg")),
        }
    for tag in args.backbones:
        model, tf, kind = load_backbone(tag, device)
        for split, files in splits.items():
            ids_path = FEAT / f"aug_{tag}_{split}_ids.csv"
            feat_path = FEAT / f"aug_{tag}_{split}.npy"
            done_ids = set()
            old = np.zeros((0, 0), dtype=np.float32)
            if ids_path.exists() and feat_path.exists():
                done_ids = set(pd.read_csv(ids_path)["id"])
                old = np.load(feat_path)
            todo = [str(f) for f in files if f.name not in done_ids]
            print(f"{tag}/{split}: {len(done_ids)} done, {len(todo)} todo", flush=True)
            if todo:
                new = extract(model, todo, tf, kind, device)
                feats = np.concatenate([old, new], 0) if old.size else new
                ids = list(done_ids) + [Path(p).name for p in todo]
                # 保持顺序：旧 id 顺序无法从 set 恢复，改为重存（旧 ids 文件 + 新）
                old_ids = pd.read_csv(ids_path)["id"].tolist() if ids_path.exists() else []
                ids = old_ids + [Path(p).name for p in todo]
                np.save(feat_path, feats)
                pd.Series(ids, name="id").to_csv(ids_path, index=False)
        del model
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
