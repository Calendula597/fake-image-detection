"""Extract FSD (Forensic Self-Descriptions, CVPR 2025) features.

Pipeline (official repo: external/fsd, weights: external/fsd/weights):
  1. Grayscale image -> FRE constrained convolution -> 8-channel forensic residuals
  2. Resize so min side = 1024, center-crop to <=1024, 3-scale patch decomposition
  3. Constrained least squares (KKT) -> 960-dim FSD descriptor
  4. Learned residual transforms (FeatureTransform MLPs)
  5. Real-only GMM (K=5, tied cov) log-likelihood -> z-score
     (more negative z-score = more likely AI-generated)

Speed note: the official pipeline accumulates X^T X / X^T y in float64, which is
~2.4 s/img on RTX 4090 (FP64-poor). We accumulate in float32 and only solve the
KKT system in float64. Verified on sample images: z-score Pearson corr = 0.99999,
max |dz| = 0.37 vs train_std = 706 -> negligible. ~22x faster (~0.11 s/img).

Outputs (under outputs/):
  features/fsd_zscore_{sample,test}.npy   raw z-scores (negative = AI)
  features/fsd_desc_{sample,test}.npy     960-dim transformed descriptors
  predictions/fsd.csv                     id,score (score = -zscore, higher = AI)

Usage:
  cd /root/autodl-tmp/fake-image-detection && source .venv/bin/activate
  python extract_fsd.py [--split sample|test|both] [--dtype fp32|fp64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT / "external" / "fsd"))

from fsd.fre import FRE  # noqa: E402
from fsd.gmm import load_gmm  # noqa: E402
from fsd.projection import apply_projections, load_transforms  # noqa: E402

from fake_image_detection.paths import (  # noqa: E402
    ensure_output_dirs,
    resolve_code_root,
    sample_csv,
    sample_image_dir,
    test_csv,
    test_image_dir,
)

WEIGHTS_DIR = CODE_ROOT / "external" / "fsd" / "weights"


class FSDPipeline:
    """FRE + multi-scale constrained least squares + transforms + GMM."""

    def __init__(self, weights_dir: Path, device: str = "cuda", dtype: str = "fp32"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32 if dtype == "fp32" else torch.float64
        if self.device.type != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        with open(weights_dir / "config.json") as f:
            self.config = json.load(f)
        self.fre = FRE.from_pretrained(weights_dir / self.config["fre"]["weights_file"], device=device)
        self.gmm = load_gmm(weights_dir / self.config["gmm"]["weights_file"], device=device)
        self.projections = load_transforms(weights_dir / self.config["transforms"]["weights_file"], device=device)
        self.train_mean = self.config["scoring"]["train_mean"]
        self.train_std = self.config["scoring"]["train_std"]

        fsd_cfg = self.config["fsd"]
        self.kernel_size = fsd_cfg["kernel_size"]
        self.num_scales = fsd_cfg["num_scales"]
        self.max_size = fsd_cfg["max_size"]

    @torch.no_grad()
    def compute_descriptor(self, image_path: Path) -> torch.Tensor:
        """960-dim raw FSD vector (float64, CPU). Mirrors fsd.compute_fsd."""
        device = self.device
        dtype = self.dtype
        img = Image.open(image_path).convert("L")
        image_t = torch.from_numpy(np.array(img)).to(dtype).unsqueeze(0)  # (1, H, W), 0-255

        K = self.fre.conv.out_channels
        border = self.fre.conv.kernel_size // 2
        B = self.kernel_size
        center = B // 2
        mask = torch.ones(B, B, dtype=torch.bool, device=device)
        mask[center, center] = False
        n_feat = int(mask.sum().item())  # B^2 - 1 = 120
        tot = K * n_feat  # 960

        # Forensic residuals, valid region only
        res = self.fre(image_t.to(device))
        res = res[:, border:-border, border:-border]

        # resize so that min side = max_size, then center-crop to max_size
        h, w = res.shape[-2:]
        sf = self.max_size / min(h, w)
        res = F.interpolate(
            res[None], size=(round(h * sf), round(w * sf)),
            mode="bilinear", antialias=False, align_corners=False,
        )[0]
        h, w = res.shape[-2:]
        ch, cw = min(self.max_size, h), min(self.max_size, w)
        sh, sw = (h - ch) // 2, (w - cw) // 2
        res = res[:, sh:sh + ch, sw:sw + cw]

        # Multi-scale patches -> accumulate X^T X / X^T y (in self.dtype)
        XTX = torch.zeros(tot, tot, dtype=dtype, device=device)
        XTy = torch.zeros(tot, dtype=dtype, device=device)
        for l in range(self.num_scales):
            sc = F.interpolate(
                res[None], scale_factor=1 / 2 ** l,
                mode="bilinear", antialias=False, align_corners=False,
            )
            sc = F.pad(sc, (B // 2, B // 2, B // 2, B // 2), mode="reflect")
            un = sc[0].unfold(1, B, 1).unfold(2, B, 1).reshape(K, -1, B, B).permute(1, 0, 2, 3)
            for i in range(0, un.shape[0], 16384):
                chunk = un[i : i + 16384]
                x = chunk[:, :, mask].reshape(-1, tot)
                y = chunk[:, :, center, center].sum(dim=1)
                XTX += x.T @ x
                XTy += x.T @ y

        # KKT solve in float64 for numerical stability
        XTX = XTX.double()
        XTy = XTy.double()
        XTX += 1e-5 * torch.eye(tot, device=device, dtype=torch.float64)
        A = torch.zeros(K, tot, device=device, dtype=torch.float64)
        for k in range(K):
            A[k, k * n_feat : (k + 1) * n_feat] = 1
        b = torch.ones(K, device=device, dtype=torch.float64)
        LHS = torch.cat(
            [torch.cat([XTX, A.T], 1), torch.cat([A, torch.zeros(K, K, device=device, dtype=torch.float64)], 1)], 0
        )
        sol = torch.linalg.solve(LHS, torch.cat([XTy, b]))
        return sol[:tot].cpu()

    @torch.no_grad()
    def score_descriptors(self, descs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Apply transforms and score with GMM.

        Args:
            descs: (N, 960) float64 tensor of raw FSD vectors.

        Returns:
            (transformed descriptors (N, 960) float64 numpy, z-scores (N,))
        """
        out_desc, out_z = [], []
        for s in range(0, descs.shape[0], 256):
            v = descs[s : s + 256].to(self.device)
            v = apply_projections(v, self.projections)
            raw = self.gmm.score_samples(v)
            z = (raw - self.train_mean) / self.train_std
            out_desc.append(v.cpu().numpy())
            out_z.append(z.cpu().numpy())
        return np.concatenate(out_desc, 0), np.concatenate(out_z, 0)


def extract_split(pipeline: FSDPipeline, ids: list[str], image_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw 960-dim descriptors for a split. Returns (ids_kept, descs)."""
    t0 = time.time()
    descs = torch.empty(len(ids), 960, dtype=torch.float64)
    kept = []
    for i, sid in enumerate(ids):
        p = image_dir / sid
        try:
            descs[i] = pipeline.compute_descriptor(p)
            kept.append(sid)
        except Exception as e:
            print(f"[warn] {sid}: {e}", file=sys.stderr)
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"  {tag}: {i + 1}/{len(ids)} ({dt / (i + 1):.3f}s/img, eta {(len(ids) - i - 1) * dt / (i + 1) / 60:.1f}min)", flush=True)
    descs = descs[: len(kept)]
    print(f"  {tag}: done {len(kept)}/{len(ids)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return np.array(kept), descs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["sample", "test", "both"], default="both")
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    code_root = resolve_code_root()
    ensure_output_dirs()
    feat_dir = code_root / "outputs" / "features"
    pred_dir = code_root / "outputs" / "predictions"
    feat_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FSDPipeline(WEIGHTS_DIR, device=args.device, dtype=args.dtype)
    print(f"FSD pipeline loaded (device={pipeline.device}, dtype={args.dtype}, "
          f"train_mean={pipeline.train_mean:.2f}, train_std={pipeline.train_std:.2f})")

    results = {}
    for split in (["sample", "test"] if args.split == "both" else [args.split]):
        if split == "sample":
            df = pd.read_csv(sample_csv())
            image_dir = sample_image_dir()
        else:
            df = pd.read_csv(test_csv())
            image_dir = test_image_dir()
        ids = df["id"].astype(str).tolist()

        kept, descs = extract_split(pipeline, ids, image_dir, split)
        tdesc, zscores = pipeline.score_descriptors(descs)

        np.save(feat_dir / f"fsd_zscore_{split}.npy", zscores.astype(np.float64))
        np.save(feat_dir / f"fsd_desc_{split}.npy", tdesc.astype(np.float32))
        pd.DataFrame({"id": kept}).to_csv(feat_dir / f"fsd_ids_{split}.csv", index=False)
        print(f"saved fsd_zscore_{split}.npy {zscores.shape}, fsd_desc_{split}.npy {tdesc.shape}")
        results[split] = (kept, zscores, tdesc, df)

    # ---- Evaluation on the labeled sample split ----
    if "sample" not in results:
        return
    from sklearn.metrics import roc_auc_score

    kept, zscores, tdesc, df = results["sample"]
    y = df.set_index("id").loc[kept, "label"].to_numpy()
    print(f"\nsample split: n={len(y)}, pos(AI)={int(y.sum())}, neg(real)={int((1 - y).sum())}")

    # FSD convention: more negative z = more AI-like -> score = -z
    score = -zscores
    auc = roc_auc_score(y, score)
    direction = "-zscore"
    if auc < 0.5:
        score = zscores
        auc = roc_auc_score(y, score)
        direction = "+zscore (flipped, official convention did NOT hold on this data)"
    print(f"z-score AUC ({direction}): {auc:.4f}")
    print(f"z-score stats: real mean={zscores[y == 0].mean():.3f} std={zscores[y == 0].std():.3f} | "
          f"AI mean={zscores[y == 1].mean():.3f} std={zscores[y == 1].std():.3f}")

    from fake_image_detection.stacking import oof_lr, oof_mlp

    o_lr = oof_lr(tdesc.astype(np.float32), y)
    print(f"desc oof_lr  AUC: {roc_auc_score(y, o_lr):.4f}")
    o_mlp = oof_mlp(tdesc.astype(np.float32), y)
    print(f"desc oof_mlp AUC: {roc_auc_score(y, o_mlp):.4f}")

    # Prediction CSV (score: higher = more AI-like), aligned with test_csv id order
    if "test" in results:
        kept_t, z_t, _, df_t = results["test"]
        score_t = -z_t if direction == "-zscore" else z_t
        pred = pd.DataFrame({"id": kept_t, "score": score_t})
        pred = df_t[["id"]].merge(pred, on="id", how="left")
        out = pred_dir / "fsd.csv"
        pred.to_csv(out, index=False)
        print(f"saved {out} ({len(pred)} rows, NaN={int(pred['score'].isna().sum())})")


if __name__ == "__main__":
    main()
