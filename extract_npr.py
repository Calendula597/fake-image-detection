"""Extract NPR (CVPR 2024) features and scores for sample/test sets.

Model: external/npr (github.com/chuangchuangtan/NPR-DeepfakeDetection),
weights external/npr/NPR.pth (shipped in the repo, DataParallel state dict
under key 'model'). Architecture is a truncated ResNet-50 whose fc1 input is
512-d; we capture that pre-fc feature plus the sigmoid score.

Preprocessing follows the official test protocol (no_resize=False,
no_crop=True): Resize((256, 256)) -> ToTensor -> ImageNet normalize.

Outputs:
  outputs/features/npr_256_sample.npy / npr_256_test.npy   (N, 512) float32
  outputs/predictions/npr.csv                              (id, score) on test
Prints OOF AUC (oof_lr / oof_mlp from fake_image_detection.stacking).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "external" / "npr"))

from fake_image_detection.feature_extractor import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageDataset,
)
from fake_image_detection.paths import (  # noqa: E402
    resolve_code_root,
    sample_csv,
    test_csv,
)
from fake_image_detection.stacking import oof_lr, oof_mlp  # noqa: E402

NPR_SIZE = 256
WEIGHTS = CODE_ROOT / "external" / "npr" / "NPR.pth"
FEAT_DIR = CODE_ROOT / "outputs" / "features"
PRED_DIR = CODE_ROOT / "outputs" / "predictions"


def build_model(device: torch.device):
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    sd = torch.load(WEIGHTS, map_location="cpu")["model"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def build_transform():
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize((NPR_SIZE, NPR_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def extract(model, ids, rel_root, device, batch_size=128, num_workers=8):
    ds = ImageDataset(ids, rel_root, build_transform())
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    feats, scores = [], []
    for imgs, _ in tqdm(dl, desc=f"npr {rel_root}"):
        imgs = imgs.to(device, non_blocking=True)
        feat = {}

        def hook(_m, _inp, out):
            feat["v"] = out

        h = model.avgpool.register_forward_hook(hook)
        logit = model(imgs)
        h.remove()
        feats.append(feat["v"].flatten(1).float().cpu().numpy())
        scores.append(logit.sigmoid().flatten().float().cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(scores, 0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(sample_csv())
    test_df = pd.read_csv(test_csv())
    model = build_model(device)

    # ---- sample split: features + OOF AUC ----
    s_feat_path = FEAT_DIR / f"npr_{NPR_SIZE}_sample.npy"
    if s_feat_path.exists():
        s_feats = np.load(s_feat_path)
        print(f"loaded cached {s_feat_path}")
    else:
        s_feats, s_scores = extract(model, sample_df["id"].tolist(),
                                    "data/image_sample_data", device)
        np.save(s_feat_path, s_feats.astype(np.float32))
        np.save(FEAT_DIR / f"npr_{NPR_SIZE}_sample_scores.npy",
                s_scores.astype(np.float32))
        print(f"saved {s_feat_path} {s_feats.shape}")

    y = sample_df["label"].to_numpy()
    from sklearn.metrics import roc_auc_score

    if (FEAT_DIR / f"npr_{NPR_SIZE}_sample_scores.npy").exists():
        s_scores = np.load(FEAT_DIR / f"npr_{NPR_SIZE}_sample_scores.npy")
        print(f"NPR raw-score AUC (sample): {roc_auc_score(y, s_scores):.5f}")
    oof1 = oof_lr(s_feats.astype(np.float64), y)
    print(f"NPR oof_lr  AUC: {roc_auc_score(y, oof1):.5f}")
    oof2 = oof_mlp(s_feats.astype(np.float32), y)
    print(f"NPR oof_mlp AUC: {roc_auc_score(y, oof2):.5f}")

    # ---- test split: features + prediction csv ----
    t_feat_path = FEAT_DIR / f"npr_{NPR_SIZE}_test.npy"
    if t_feat_path.exists():
        t_feats = np.load(t_feat_path)
        print(f"loaded cached {t_feat_path}")
        t_scores = None
    else:
        t_feats, t_scores = extract(model, test_df["id"].tolist(),
                                    "data/image_test", device)
        np.save(t_feat_path, t_feats.astype(np.float32))
        print(f"saved {t_feat_path} {t_feats.shape}")
    if t_scores is None:
        # recompute scores only (cheap) if features were cached
        _, t_scores = extract(model, test_df["id"].tolist(),
                              "data/image_test", device)
    pred = pd.DataFrame({"id": test_df["id"], "score": t_scores})
    pred_path = PRED_DIR / "npr.csv"
    pred.to_csv(pred_path, index=False)
    print(f"saved {pred_path} ({len(pred)} rows)")


if __name__ == "__main__":
    main()
