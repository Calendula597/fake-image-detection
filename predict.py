from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fake_image_detection.checkpoints import load_checkpoint
from fake_image_detection.config import cli_config_args, load_config, parse_overrides
from fake_image_detection.dataset import DetectorDataset, collate_fn
from fake_image_detection.logger import get_logger
from fake_image_detection.model import build_model
from fake_image_detection.paths import resolve_code_root, resolve_data_root, test_csv
from fake_image_detection.train_engine import _autocast
from fake_image_detection.transforms import build_transforms

log = get_logger("predict")


@torch.no_grad()
def batched_tta(model, loader, device, precision: str = "bf16", hflip: bool = True):
    probs_list = []
    sids = []
    flip_probs = [] if hflip else None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with _autocast(precision):
            p = torch.sigmoid(model(images)["logits"].float())
        probs_list.append(p.cpu().numpy())
        if hflip:
            fp = torch.sigmoid(model(images.flip(-1))["logits"].float())
            flip_probs.append(fp.cpu().numpy())
        sids.extend(batch["sample_id"])
    probs = np.concatenate(probs_list)
    if hflip:
        probs = (probs + np.concatenate(flip_probs)) / 2.0
    return probs, sids


def main():
    parser = argparse.ArgumentParser()
    parser = cli_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.override))
    device = torch.device(cfg["hardware"]["device"])
    precision = cfg["training"]["precision"]

    df = pd.read_csv(test_csv())
    df["relative_path"] = "data/image_test/" + df["id"].astype(str)
    df["sample_id"] = df["id"]

    data_root = resolve_data_root()
    tf = build_transforms(cfg, train=False)
    ds = DetectorDataset(df, data_root, tf, return_label=False, default_size=cfg["data"]["image_size"])
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["micro_batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    ck = load_checkpoint(args.checkpoint, model, map_location=device)
    log.info(f"loaded {args.checkpoint} epoch={ck.get('epoch')} best_auc={ck.get('best_auc')}")
    model.eval()

    hflip_tta = args.tta and cfg.get("inference", {}).get("tta", {}).get("hflip", True)
    probs, sids = batched_tta(model, loader, device, precision, hflip=hflip_tta)
    out = pd.DataFrame({"id": sids, "score": probs})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info(f"wrote {len(out)} predictions to {out_path} (tta={hflip_tta})")


if __name__ == "__main__":
    main()
