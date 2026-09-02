from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fake_image_detection.cf_model import load_commfor
from fake_image_detection.feature_extractor import ImageDataset, build_cf_transform
from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv

CR = resolve_code_root()


@torch.no_grad()
def extract(model, paths, tf, device, bs=64):
    from fake_image_detection.feature_extractor import ImageDataset as _ID
    class DS(_ID):
        def __init__(self, paths, tf):
            self.paths = paths; self.tf = tf; self.data_root = Path("/")
            self._is_albu = True
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            try: img = np.array(Image.open(self.paths[i]).convert("RGB"))
            except Exception: img = np.zeros((384, 384, 3), dtype=np.uint8)
            return self.tf(image=img)["image"], i
    dl = DataLoader(DS(paths, tf), batch_size=bs, shuffle=False, num_workers=8)
    feats = []
    for imgs, _ in tqdm(dl, desc="cf-feat"):
        imgs = imgs.to(device)
        ft = model.vit.forward_features(imgs)
        pooled = model.vit.forward_head(ft, pre_logits=True)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, size = load_commfor("weights/commfor/commfor-model-384", device)
    tf = build_cf_transform(size, True)
    data_root = resolve_data_root()
    lab = pd.read_csv(sample_csv())
    real_ids = lab[lab.label == 0]["id"].tolist()
    ai_ids = lab[lab.label == 1]["id"].tolist()

    real_paths = [data_root / "data" / "image_sample_data" / sid for sid in real_ids]
    fake_paths = [CR / "outputs" / "bfree_fakes" / sid for sid in real_ids]
    ai_paths = [data_root / "data" / "image_sample_data" / sid for sid in ai_ids]

    print("extracting real / bfree-fake / old-AI features ...")
    Xr = extract(model, real_paths, tf, device)
    Xf = extract(model, fake_paths, tf, device)
    Xa = extract(model, ai_paths, tf, device)

    # 训练：real(0) vs bfree-fake(1)
    Xtr = np.concatenate([Xr, Xf], 0)
    ytr = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xf))], 0)
    Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    clf = LogisticRegression(C=1.0, max_iter=4000).fit(Xn, ytr)

    # 评估1: 训练集内（real vs bfree-fake）
    p_tr = clf.predict_proba(Xn)[:, 1]
    print(f"[in-dist] real vs bfree-fake AUC = {roc_auc_score(ytr, p_tr):.4f}")

    # 评估2: 跨生成器泛化（real vs old-AI）
    Xte = np.concatenate([Xr, Xa], 0)
    yte = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xa))], 0)
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    p_te = clf.predict_proba(Xte_n)[:, 1]
    print(f"[cross-gen] real vs old-AI AUC = {roc_auc_score(yte, p_te):.4f}  (泛化到未见生成器)")
    np.save(CR / "outputs" / "features" / "bfree_clf.npy", np.array([0]))
    import joblib
    joblib.dump(clf, CR / "outputs" / "features" / "bfree_clf.joblib")
    print("saved bfree_clf.joblib")


if __name__ == "__main__":
    main()
