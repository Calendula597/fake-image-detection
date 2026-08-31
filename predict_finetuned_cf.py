from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from fake_image_detection.cf_model import ViTClassifier
from fake_image_detection.feature_extractor import ImageDataset, build_cf_transform
from fake_image_detection.paths import resolve_code_root, test_csv
from torch.utils.data import DataLoader
from tqdm import tqdm

CR = resolve_code_root()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/Detector/image/outputs/checkpoints/commfor_ft384/best.pt")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="outputs/predictions/ft384_sem.csv")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    model = ViTClassifier(model_size="small", input_size=args.size, freeze_backbone=False, device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt} missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    test = pd.read_csv(test_csv())
    ids = test["id"].tolist()
    tf = build_cf_transform(args.size, is_crop=True)
    ds = ImageDataset(ids, "data/image_test", tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # TTA: 原图 + 水平翻转
    probs = []
    for imgs, _ in tqdm(dl, desc="ft384-tta"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            p1 = torch.sigmoid(model(imgs).float().reshape(-1))
            p2 = torch.sigmoid(model(imgs.flip(-1)).float().reshape(-1))
        probs.append(((p1 + p2) / 2).cpu().numpy())
    probs = np.concatenate(probs, 0).reshape(-1)

    out = pd.DataFrame({"id": ids, "score": probs})
    out_path = CR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} mean={probs.mean():.4f}")


if __name__ == "__main__":
    main()
