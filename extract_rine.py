"""Extract RINE (ECCV 2024) features and scores for sample/test sets.

Model: external/rine (github.com/mever-team/rine), official trainable
checkpoints shipped in external/rine/ckpt/model_{1class,2class,4class,ldm}_trainable.pth.

RINE hooks the CLS token at the input of every Transformer block's MLP
(the output of each `ln_2`) of a CLIP ViT-L/14 image encoder (24 blocks,
1024-d each), projects each block token, re-weights blocks with a learned
softmax(alpha) (TIE), sums, and classifies. Only alpha/proj1/proj2/head are
trained; the CLIP backbone is frozen.

Backbone: open_clip `ViT-L-14` pretrained `openai` — same converted OpenAI
weights (quick_gelu) as the official `clip.load("ViT-L/14")`. Following the
official repo (openai `clip` casts the model to fp16 on CUDA), the backbone
runs in fp16; the concatenated CLS feature g is cast back to fp32.

Preprocessing follows the official test protocol: CenterCrop(224) at native
resolution (torchvision pads smaller images with 0) -> ToTensor -> CLIP
normalize. No resize.

Outputs:
  outputs/features/rine_24_224_crop_{sample,test}.npy   (N, 24*1024) float16
      concatenated intermediate CLS tokens (pre-projection feature g)
  outputs/features/rine_prob_{1class,2class,4class,ldm}_{sample,test}.npy
      sigmoid probability of the official projection head per checkpoint
  outputs/predictions/rine.csv                          (id, score) on test,
      score = 4class checkpoint probability
Prints raw-score AUC per checkpoint and OOF AUC (oof_lr / oof_mlp from
fake_image_detection.stacking) of the concatenated feature on the sample set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))

from fake_image_detection.feature_extractor import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    ImageDataset,
)
from fake_image_detection.paths import sample_csv, test_csv  # noqa: E402
from fake_image_detection.stacking import oof_lr, oof_mlp  # noqa: E402

RINE_SIZE = 224
N_LAYERS = 24  # ViT-L/14 transformer blocks = number of ln_2 hooks
EMB_DIM = 1024
CKPT_DIR = CODE_ROOT / "external" / "rine" / "ckpt"
FEAT_DIR = CODE_ROOT / "outputs" / "features"
PRED_DIR = CODE_ROOT / "outputs" / "predictions"
# official configs from external/rine/src/utils.py::get_our_trained_model
CKPT_CONFIGS = {
    "1class": dict(nproj=4, proj_dim=1024),
    "2class": dict(nproj=4, proj_dim=128),
    "4class": dict(nproj=2, proj_dim=1024),
    "ldm": dict(nproj=4, proj_dim=1024),
}
PRIMARY_CKPT = "4class"


class RINEProjection(nn.Module):
    """Trainable part of external/rine/src/models.py::Model (same module names
    so the official checkpoints load with strict=True)."""

    def __init__(self, nproj: int, proj_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn([1, N_LAYERS, proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(EMB_DIM if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend([nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout()])
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """g: (N, N_LAYERS, EMB_DIM) -> logit (N,)"""
        h = self.proj1(g.float())
        z = torch.softmax(self.alpha, dim=1) * h
        z = torch.sum(z, dim=1)
        z = self.proj2(z)
        return self.head(z).squeeze(-1)


def load_backbone(device: torch.device):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device="cpu"
    )
    for p in model.parameters():
        p.requires_grad = False
    # official repo runs openai CLIP in fp16 on CUDA
    return model.to(device).half().eval()


def build_transform():
    from torchvision import transforms as T

    return T.Compose(
        [
            T.CenterCrop(RINE_SIZE),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


@torch.no_grad()
def extract_cls_concat(model, ids, rel_root, device, batch_size=128, num_workers=8):
    """Concatenated intermediate CLS tokens g: (N, N_LAYERS, EMB_DIM) float16."""
    hooks_out = []
    handles = [
        module.register_forward_hook(lambda _m, _i, out: hooks_out.append(out))
        for name, module in model.visual.named_modules()
        if "ln_2" in name
    ]
    assert len(handles) == N_LAYERS, f"expected {N_LAYERS} ln_2 hooks, got {len(handles)}"

    ds = ImageDataset(ids, rel_root, build_transform())
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    feats = []
    try:
        for imgs, _ in tqdm(dl, desc=f"rine-cls {rel_root}"):
            imgs = imgs.to(device, non_blocking=True).half()
            hooks_out.clear()
            model.encode_image(imgs)
            # each hook output is (seq, batch, dim) = (257, N, 1024); token 0 = CLS
            g = torch.stack([h[0] for h in hooks_out], dim=1)  # (N, 24, 1024)
            feats.append(g.float().cpu().numpy().astype(np.float16))
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(feats, 0)


@torch.no_grad()
def apply_projection(g: np.ndarray, setting: str, device: torch.device, chunk: int = 4096):
    proj = RINEProjection(**CKPT_CONFIGS[setting])
    sd = torch.load(CKPT_DIR / f"model_{setting}_trainable.pth", map_location="cpu")
    proj.load_state_dict(sd, strict=True)
    proj = proj.to(device).eval()
    probs = []
    for s in range(0, len(g), chunk):
        gb = torch.from_numpy(g[s : s + chunk]).to(device)
        probs.append(torch.sigmoid(proj(gb)).float().cpu().numpy())
    return np.concatenate(probs, 0)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(sample_csv())
    test_df = pd.read_csv(test_csv())
    y = sample_df["label"].to_numpy()

    tag = f"rine_{N_LAYERS}_{RINE_SIZE}_crop"
    feats = {}
    model = None
    for split, df, rel_root in (
        ("sample", sample_df, "data/image_sample_data"),
        ("test", test_df, "data/image_test"),
    ):
        path = FEAT_DIR / f"{tag}_{split}.npy"
        if path.exists():
            feats[split] = np.load(path)
            print(f"loaded cached {path} {feats[split].shape}")
        else:
            if model is None:
                model = load_backbone(device)
            g = extract_cls_concat(model, df["id"].tolist(), rel_root, device)
            np.save(path, g)
            feats[split] = g
            print(f"saved {path} {g.shape} {g.dtype}")
    del model
    torch.cuda.empty_cache()

    from sklearn.metrics import roc_auc_score

    # ---- official projection heads: probabilities per checkpoint ----
    test_probs = {}
    for setting in CKPT_CONFIGS:
        for split in ("sample", "test"):
            ppath = FEAT_DIR / f"rine_prob_{setting}_{split}.npy"
            if ppath.exists():
                prob = np.load(ppath)
                print(f"loaded cached {ppath}")
            else:
                prob = apply_projection(
                    feats[split].reshape(len(feats[split]), N_LAYERS, EMB_DIM)
                    .astype(np.float32),
                    setting,
                    device,
                ).astype(np.float32)
                np.save(ppath, prob)
                print(f"saved {ppath} {prob.shape}")
            if split == "sample":
                print(f"RINE {setting} raw-prob AUC (sample): {roc_auc_score(y, prob):.5f}")
            else:
                test_probs[setting] = prob

    pred = pd.DataFrame({"id": test_df["id"], "score": test_probs[PRIMARY_CKPT]})
    pred_path = PRED_DIR / "rine.csv"
    pred.to_csv(pred_path, index=False)
    print(f"saved {pred_path} ({len(pred)} rows, score={PRIMARY_CKPT} prob)")

    # ---- OOF AUC of the concatenated feature on sample ----
    g_sample = feats["sample"].reshape(len(sample_df), -1)
    oof1 = oof_lr(g_sample.astype(np.float64), y)
    print(f"RINE concat oof_lr  AUC: {roc_auc_score(y, oof1):.5f}")
    oof2 = oof_mlp(g_sample.astype(np.float32), y)
    print(f"RINE concat oof_mlp AUC: {roc_auc_score(y, oof2):.5f}")


if __name__ == "__main__":
    main()
