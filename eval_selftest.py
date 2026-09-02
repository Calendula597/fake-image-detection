from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv
from fake_image_detection.feature_extractor import ImageDataset, build_cf_transform, build_clip_transform, IMAGENET_MEAN, IMAGENET_STD

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"


def load_labels(csv_paths):
    ys, ids = [], []
    for p, lab in csv_paths:
        for f in sorted(Path(p).glob("*.jpg")):
            ys.append(lab); ids.append(str(f))
    return np.array(ids), np.array(ys)


@torch.no_grad()
def extract(backbone, model, ids, device, size, kind):
    if kind == "cf":
        tf = build_cf_transform(size, True)
        dl = DataLoader(ImageDataset(ids, "", tf), batch_size=64, shuffle=False, num_workers=8)
        feats = []
        for imgs, _ in tqdm(dl, desc=f"cf {backbone}", leave=False):
            imgs = imgs.to(device)
            ft = model.vit.forward_features(imgs)
            pooled = model.vit.forward_head(ft, pre_logits=True)
            feats.append(pooled.float().cpu().numpy())
        return np.concatenate(feats, 0)
    else:
        tf = build_clip_transform(size, "crop")
        dl = DataLoader(ImageDataset(ids, "", tf), batch_size=48, shuffle=False, num_workers=8)
        feats = []
        for imgs, _ in tqdm(dl, desc=f"clip {backbone}", leave=False):
            imgs = imgs.to(device)
            f = model.encode_image(imgs).float() if kind == "clip" else model.forward_head(model.forward_features(imgs), pre_logits=True).float()
            feats.append(F.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(feats, 0)


class PathImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths = paths; self.tf = tf; self._is_albu = isinstance(tf, type(build_cf_transform(224, True)))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            img = np.array(Image.open(self.paths[i]).convert("RGB"))
        except Exception:
            img = np.zeros((384, 384, 3), dtype=np.uint8)
        if self._is_albu:
            return self.tf(image=img)["image"], i
        return self.tf(Image.fromarray(img)), i


def extract_paths(backbone, model, paths, device, size, kind):
    if kind == "cf":
        tf = build_cf_transform(size, True)
    else:
        tf = build_clip_transform(size, "crop")
    dl = DataLoader(PathImageDataset(paths, tf), batch_size=48, shuffle=False, num_workers=8)
    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(dl, desc=f"{backbone} selftest", leave=False):
            imgs = imgs.to(device)
            if kind == "cf":
                ft = model.vit.forward_features(imgs); f = model.vit.forward_head(ft, pre_logits=True)
            elif kind == "clip":
                f = model.encode_image(imgs)
            else:
                f = model.forward_head(model.forward_features(imgs), pre_logits=True)
            feats.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cf384", choices=["cf384", "clipH378", "clipBigG", "dinoL"])
    ap.add_argument("--real-dir", default="data_real/coco_val")
    args = ap.parse_args()
    device = torch.device("cuda:0")

    real_ids, y_real = load_labels([(CR / args.real_dir, 0)])
    ai_dirs = sorted([d for d in (CR / "data_real").glob(args.real_dir.split("/")[-1] + "_ai_*") if d.is_dir()])
    ai_ids, y_ai = [], []
    for d in ai_dirs:
        for f in sorted(d.glob("*.jpg")):
            ai_ids.append(str(f)); y_ai.append(1)
    ids = np.concatenate([real_ids, np.array(ai_ids)])
    y = np.concatenate([y_real, np.array(y_ai)])
    print(f"self-test set: {len(real_ids)} real + {len(ai_ids)} AI from {len(ai_dirs)} generators")

    # 加载 backbone
    from fake_image_detection.cf_model import load_commfor
    from fake_image_detection.clip_model import load_openclip
    import timm
    if args.backbone == "cf384":
        model, size = load_commfor("weights/commfor/commfor-model-384", device); kind, tag = "cf", "commfor_cf384"
    elif args.backbone == "clipH378":
        model = load_openclip("ViT-H-14-378-quickgelu", "weights/clip-vit-h14-378/open_clip_pytorch_model.bin", device); size, kind, tag = 378, "clip", "clipH378"
    elif args.backbone == "clipBigG":
        model = load_openclip("ViT-bigG-14", "weights/clip-vit-bigg-14/open_clip_pytorch_model.bin", device); size, kind, tag = 224, "clip", "clipBigG"
    else:
        model = timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0).to(device).eval(); size, kind, tag = 224, "dino", "dinoL"

    # 训练头：用比赛 1000 特征
    Xs_comp = np.load(FEAT / f"{tag}_{size}_crop_sample.npy").astype(np.float32)
    y_comp = pd.read_csv(sample_csv())["label"].astype(int).to_numpy()
    Xn = Xs_comp / (np.linalg.norm(Xs_comp, axis=1, keepdims=True) + 1e-8)
    clf = LogisticRegression(C=1.0, max_iter=4000).fit(Xn, y_comp)

    # 在自建测试集上提取特征并评估
    Xtest = extract_paths(args.backbone, model, ids.tolist(), device, size, kind)
    Xtest_n = Xtest / (np.linalg.norm(Xtest, axis=1, keepdims=True) + 1e-8)
    p = clf.predict_proba(Xtest_n)[:, 1]
    print(f"[{args.backbone}] self-test AUC (cross-gen to unseen SD) = {roc_auc_score(y, p):.4f}")
    # 分生成器
    off = len(real_ids)
    for i, d in enumerate(ai_dirs):
        per = p[off:off + 600]
        yy = y[off:off + 600]
        # 与该生成器配对的 real 部分
        pr = np.concatenate([p[:len(real_ids)], per])
        yr = np.concatenate([y[:len(real_ids)], yy])
        print(f"  gen {d.name}: AUC = {roc_auc_score(yr, pr):.4f}")
        off += 600


if __name__ == "__main__":
    main()
