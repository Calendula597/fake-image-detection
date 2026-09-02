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

from fake_image_detection.paths import resolve_code_root, sample_csv
from fake_image_detection.feature_extractor import build_cf_transform, build_clip_transform
from fake_image_detection.stacking import rank01

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, tf, is_albu):
        self.paths = paths; self.tf = tf; self.is_albu = is_albu
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: img = np.array(Image.open(self.paths[i]).convert("RGB"))
        except Exception: img = np.zeros((384, 384, 3), dtype=np.uint8)
        if self.is_albu: return self.tf(image=img)["image"], i
        return self.tf(Image.fromarray(img)), i


def load_backbone(name, device):
    from fake_image_detection.cf_model import load_commfor
    from fake_image_detection.clip_model import load_openclip
    import timm
    if name == "cf384":
        m, size = load_commfor("weights/commfor/commfor-model-384", device); return m, size, "cf", "commfor_cf384"
    if name == "clipH378":
        return load_openclip("ViT-H-14-378-quickgelu", "weights/clip-vit-h14-378/open_clip_pytorch_model.bin", device), 378, "clip", "clipH378"
    if name == "clipBigG":
        return load_openclip("ViT-bigG-14", "weights/clip-vit-bigg-14/open_clip_pytorch_model.bin", device), 224, "clip", "clipBigG"
    if name == "dinoL":
        return timm.create_model("vit_large_patch16_dinov3", pretrained=True, num_classes=0).to(device).eval(), 224, "dino", "dinoL"


@torch.no_grad()
def extract(name, model, kind, paths, size, device):
    if kind == "cf":
        tf, is_albu = build_cf_transform(size, True), True
    else:
        tf, is_albu = build_clip_transform(size, "crop"), False
    dl = DataLoader(DS(paths, tf, is_albu), batch_size=48, shuffle=False, num_workers=8)
    feats = []
    for imgs, _ in tqdm(dl, desc=name, leave=False):
        imgs = imgs.to(device)
        if kind == "cf":
            f = model.vit.forward_head(model.vit.forward_features(imgs), pre_logits=True)
        elif kind == "clip":
            f = model.encode_image(imgs)
        else:
            f = model.forward_head(model.forward_features(imgs), pre_logits=True)
        feats.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    device = torch.device("cuda:0")
    real_paths = sorted((CR / "data_real/coco_val").glob("*.jpg"))
    ai_paths = sorted((CR / "data_real/coco_val_ai_sd14").glob("*.jpg"))
    paths = [str(p) for p in real_paths] + [str(p) for p in ai_paths]
    y = np.concatenate([np.zeros(len(real_paths)), np.ones(len(ai_paths))])
    y_comp = pd.read_csv(sample_csv())["label"].astype(int).to_numpy()
    print(f"self-test: {len(real_paths)} real + {len(ai_paths)} AI")

    names = ["cf384", "clipH378", "clipBigG", "dinoL"]
    scores = {}
    for name in names:
        model, size, kind, tag = load_backbone(name, device)
        Xs = np.load(FEAT / f"{tag}_{size}_crop_sample.npy").astype(np.float32)
        Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
        clf = LogisticRegression(C=1.0, max_iter=4000).fit(Xn, y_comp)
        Xt = extract(name, model, kind, paths, size, device)
        Xt_n = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
        p = clf.predict_proba(Xt_n)[:, 1]
        scores[name] = p
        print(f"[{name}] single self-test AUC = {roc_auc_score(y, p):.4f}")
        del model
        torch.cuda.empty_cache()

    # 堆叠组合（rank 平均）
    stack = np.mean([rank01(scores[n]) for n in names], axis=0)
    print(f"[stack rank-avg ALL] AUC = {roc_auc_score(y, stack):.4f}")
    # 排除 CF384（SD 调优，有偏）
    stack_nocf = np.mean([rank01(scores[n]) for n in names if n != "cf384"], axis=0)
    print(f"[stack rank-avg no-CF384] AUC = {roc_auc_score(y, stack_nocf):.4f}")


if __name__ == "__main__":
    main()
