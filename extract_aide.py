"""Extract AIDE (ICLR 2025) hybrid features and scores for sample/test sets.

Model: external/aide (github.com/shilinyan99/AIDE), code unmodified.
Checkpoint: AIDE-main/model_epoch_best.pth from the AIGIBench reproducibility
benchmark (github.com/HorizonTEL/AIGIBench), re-hosted on HuggingFace at
HorizonTEL/AIGIBench (downloaded via hf-mirror.com because the official
Google Drive Model Zoo is unreachable from this machine). AIGIBench's
detector_codes/AIDE-main/models/AIDE.py is byte-identical to the official
repo, and the checkpoint is a fully trained AIDE model (their Setting-II).

Architecture (external/aide/models/AIDE.py):
  - Low-level artifact branch: image -> DCT_base_Rec_Module picks the two
    32x32 patches with min/max DCT-band energy (4 patches: min, max, 2nd-min,
    2nd-max) -> each resized to 256x256, ImageNet-normalized -> 30 SRM
    high-pass filters (HPF) -> two ResNet-50 (model_min/model_max, 30-channel
    input) -> 2048-d each -> x_1 = mean of the 4 features (2048-d).
  - Semantic branch: whole image resized to 256x256, ImageNet-normalized ->
    re-normalized to CLIP stats inside forward -> frozen OpenCLIP
    ConvNeXt-XXLarge visual trunk -> 3072-d avg-pooled -> trained linear
    convnext_proj -> x_0 (256-d).
  - Head: MLP(2304 -> 1024 -> 2) on concat([x_0, x_1]).

Preprocessing follows the official TEST protocol (data/datasets.py
TestDataset): ToTensor only (no blur/JPEG perturbation), DCT patch selection
at native resolution, then Resize([256, 256]) + ImageNet normalize for all
five tensors.

Outputs:
  outputs/features/aide_256_{sample,test}.npy  (N, 2304) float16
      concat of semantic-branch x_0 (256-d) and low-level branch x_1 (2048-d)
  outputs/features/aide_prob_{sample,test}.npy (N,) float32
      softmax probability of class 1 (AI-generated) from the trained head
  outputs/predictions/aide.csv                 (id, score) on test
Prints raw-prob AUC and OOF AUC (oof_lr / oof_mlp from
fake_image_detection.stacking) of the hybrid feature on the sample set.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
AIDE_ROOT = CODE_ROOT / "external" / "aide"
sys.path.insert(0, str(AIDE_ROOT))

from fake_image_detection.paths import (  # noqa: E402
    resolve_data_root,
    sample_csv,
    test_csv,
)
from fake_image_detection.stacking import oof_lr, oof_mlp  # noqa: E402

AIDE_SIZE = 256
CKPT_PATH = AIDE_ROOT / "pretrained_ckpts" / "aide_aigibench_model_epoch_best.pth"
FEAT_DIR = CODE_ROOT / "outputs" / "features"
PRED_DIR = CODE_ROOT / "outputs" / "predictions"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEAT_DIM = 256 + 2048  # x_0 (semantic proj) + x_1 (low-level resnet mean)


def load_aide_model(device: torch.device):
    """Build AIDE_Model from the unmodified official repo and load the ckpt."""
    # models/AIDE.py imports openai `clip` but never uses it; stub it out.
    if "clip" not in sys.modules:
        try:
            import clip  # noqa: F401
        except ImportError:
            sys.modules["clip"] = types.ModuleType("clip")

    from models.AIDE import AIDE_Model

    model = AIDE_Model(resnet_path=None, convnext_path=None)
    import argparse

    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # strip DDP prefix if present
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("openclip_convnext_xxl.head")]
    print(f"ckpt loaded: {len(sd)} keys, missing={len(real_missing)}, "
          f"unexpected={len(unexpected)}")
    if real_missing:
        print("  missing (excl. convnext head, replaced by Identity):", real_missing[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])
    assert not real_missing, "checkpoint does not cover the full model"
    return model.to(device).eval()


class AIDEDataset(Dataset):
    """Official AIDE test preprocessing; returns a (5, 3, 256, 256) tensor."""

    def __init__(self, ids, rel_root: str):
        from torchvision import transforms
        from data.dct import DCT_base_Rec_Module

        self.ids = list(ids)
        self.rel_root = rel_root
        self.data_root = resolve_data_root()
        self.dct = DCT_base_Rec_Module()
        self.to_tensor = transforms.ToTensor()
        self.final = transforms.Compose(
            [
                transforms.Resize([AIDE_SIZE, AIDE_SIZE]),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image

        sid = self.ids[i]
        try:
            pil = Image.open(self.data_root / self.rel_root / sid).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (AIDE_SIZE, AIDE_SIZE))
        image = self.to_tensor(pil)  # [3, H, W] in [0, 1]
        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except Exception:
            z = torch.zeros(3, 32, 32)
            x_minmin = x_maxmax = x_minmin1 = x_maxmax1 = z
        x_0 = self.final(image)
        out = torch.stack(
            [
                self.final(x_minmin),
                self.final(x_maxmax),
                self.final(x_minmin1),
                self.final(x_maxmax1),
                x_0,
            ],
            dim=0,
        )
        return out, sid


@torch.no_grad()
def aide_forward_features(model, x: torch.Tensor):
    """Replicates models/AIDE.py::AIDE_Model.forward, also returning features."""
    x_minmin = model.hpf(x[:, 0])
    x_maxmax = model.hpf(x[:, 1])
    x_minmin1 = model.hpf(x[:, 2])
    x_maxmax1 = model.hpf(x[:, 3])
    tokens = x[:, 4]

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tokens.device).view(3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tokens.device).view(3, 1, 1)
    dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=tokens.device).view(3, 1, 1)
    dinov2_std = torch.tensor([0.229, 0.224, 0.225], device=tokens.device).view(3, 1, 1)

    local_feats = model.openclip_convnext_xxl(
        tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
    )
    local_feats = model.avgpool(local_feats).view(tokens.size(0), -1)
    x_0 = model.convnext_proj(local_feats)  # (B, 256)

    x_min = model.model_min(x_minmin)
    x_max = model.model_max(x_maxmax)
    x_min1 = model.model_min(x_minmin1)
    x_max1 = model.model_max(x_maxmax1)
    x_1 = (x_min + x_max + x_min1 + x_max1) / 4  # (B, 2048)

    logits = model.fc(torch.cat([x_0, x_1], dim=1))  # (B, 2)
    return x_0, x_1, logits


@torch.no_grad()
def extract(model, ids, rel_root, device, batch_size=16, num_workers=8):
    ds = AIDEDataset(ids, rel_root)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    feats, probs = [], []
    for x, _ in tqdm(dl, desc=f"aide {rel_root}"):
        x = x.to(device, non_blocking=True)
        x_0, x_1, logits = aide_forward_features(model, x)
        feat = torch.cat([x_0, x_1], dim=1)
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        feats.append(feat.float().cpu().numpy().astype(np.float16))
        probs.append(prob.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, 0), np.concatenate(probs, 0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(sample_csv())
    test_df = pd.read_csv(test_csv())
    y = sample_df["label"].to_numpy()

    tag = f"aide_{AIDE_SIZE}"
    feats, probs = {}, {}
    model = None
    for split, df, rel_root in (
        ("sample", sample_df, "data/image_sample_data"),
        ("test", test_df, "data/image_test"),
    ):
        fpath = FEAT_DIR / f"{tag}_{split}.npy"
        ppath = FEAT_DIR / f"aide_prob_{split}.npy"
        if fpath.exists() and ppath.exists():
            feats[split] = np.load(fpath)
            probs[split] = np.load(ppath)
            print(f"loaded cached {fpath} {feats[split].shape} and {ppath}")
        else:
            if model is None:
                model = load_aide_model(device)
            f, p = extract(model, df["id"].tolist(), rel_root, device)
            np.save(fpath, f)
            np.save(ppath, p)
            feats[split], probs[split] = f, p
            print(f"saved {fpath} {f.shape} {f.dtype}; {ppath} {p.shape}")
    del model
    torch.cuda.empty_cache()

    from sklearn.metrics import roc_auc_score

    print(f"AIDE raw-prob AUC (sample): {roc_auc_score(y, probs['sample']):.5f}")

    pred = pd.DataFrame({"id": test_df["id"], "score": probs["test"]})
    pred_path = PRED_DIR / "aide.csv"
    pred.to_csv(pred_path, index=False)
    print(f"saved {pred_path} ({len(pred)} rows)")

    X = feats["sample"].astype(np.float32)
    oof1 = oof_lr(X.astype(np.float64), y)
    print(f"AIDE hybrid oof_lr  AUC: {roc_auc_score(y, oof1):.5f}")
    oof2 = oof_mlp(X, y)
    print(f"AIDE hybrid oof_mlp AUC: {roc_auc_score(y, oof2):.5f}")


if __name__ == "__main__":
    main()
