from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from fake_image_detection.checkpoints import load_checkpoint, save_checkpoint
from fake_image_detection.config import cli_config_args, load_config, parse_overrides
from fake_image_detection.dataset import DetectorDataset, collate_fn
from fake_image_detection.losses import build_loss
from fake_image_detection.logger import get_logger
from fake_image_detection.metrics import compute_metrics
from fake_image_detection.model import build_model
from fake_image_detection.optimizer import build_optimizer
from fake_image_detection.paths import ensure_output_dirs, resolve_code_root, resolve_data_root, sample_csv
from fake_image_detection.scheduler import build_scheduler
from fake_image_detection.seed import set_seed
from fake_image_detection.train_engine import EMA, evaluate, train_one_epoch
from fake_image_detection.transforms import build_transforms

log = get_logger("train")


def load_split(fold: int, n_splits: int, seed: int):
    """直接读取 image_sample_data.csv 并做 StratifiedKFold。"""
    df = pd.read_csv(sample_csv())
    df["sample_id"] = df["id"].astype(str)
    df["relative_path"] = "data/image_sample_data/" + df["id"].astype(str)
    df["split"] = "train"
    if fold == -1:
        return df.copy(), df.iloc[0:0].copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(df, df["label"]))[fold]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser = cli_config_args(parser)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny smoke run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--cpu", action="store_true", help="allow CPU training (slow, for smoke test only)")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    cfg = load_config(args.config, overrides)
    set_seed(cfg["experiment"]["seed"])
    ensure_output_dirs()

    device = torch.device(cfg["hardware"]["device"])
    if not torch.cuda.is_available():
        if args.cpu:
            log.warning("CUDA not available, running on CPU (very slow)")
            device = torch.device("cpu")
            cfg["hardware"]["device"] = "cpu"
        else:
            raise RuntimeError("CUDA not available. Training requires a GPU. Use --cpu to force CPU.")

    precision = cfg["training"]["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        log.warning("BF16 not supported, falling back to FP16")
        precision = "fp16"
        cfg["training"]["precision"] = precision

    data_root = resolve_data_root()
    log.info(f"data_root={data_root} device={device} precision={precision}")

    n_splits = cfg.get("folds", {}).get("n_splits", 5)
    seed = cfg["experiment"]["seed"]
    train_df, val_df = load_split(args.fold, n_splits, seed)

    if args.smoke:
        n = min(64, len(train_df))
        train_df = train_df.head(n)
        val_df = val_df.head(min(32, len(val_df)))

    log.info(f"train={len(train_df)} val={len(val_df)}")

    train_tf = build_transforms(cfg, train=True)
    val_tf = build_transforms(cfg, train=False)
    train_ds = DetectorDataset(train_df, data_root, train_tf, default_size=cfg["data"]["image_size"])
    val_ds = DetectorDataset(val_df, data_root, val_tf, default_size=cfg["data"]["image_size"])

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["micro_batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
        persistent_workers=cfg["data"].get("persistent_workers", False) and cfg["data"].get("num_workers", 4) > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["micro_batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    pw = None
    if tcfg.get("pos_weight") == "auto":
        pos = float((train_df["label"] == 1).sum())
        neg = float((train_df["label"] == 0).sum())
        pw = torch.tensor([neg / max(pos, 1)], device=device)
        log.info(f"pos_weight={pw.item():.3f} (pos={pos} neg={neg})")
    elif tcfg.get("pos_weight") not in (None, "auto"):
        pw = torch.tensor([float(tcfg["pos_weight"])], device=device)
    loss_fn = build_loss(cfg, pos_weight=pw)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(len(train_loader) // tcfg.get("gradient_accumulation_steps", 1), 1)
    epochs = int(tcfg.get("epochs", 8))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.999))) if tcfg.get("ema", True) else None
    scaler = torch.cuda.amp.GradScaler() if precision == "fp16" else None

    tag = args.output_tag or cfg["experiment"]["name"]
    ckpt_dir = resolve_code_root() / "outputs" / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_auc = -1.0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = ck.get("epoch", 0) + 1
        best_auc = ck.get("best_auc", -1.0)
        log.info(f"resumed from {args.resume} epoch={start_epoch} best_auc={best_auc}")

    oof_records = []
    for epoch in range(start_epoch, epochs):
        log.info(f"=== epoch {epoch + 1}/{epochs} ===")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            precision,
            tcfg.get("gradient_accumulation_steps", 1),
            clip_grad=tcfg.get("gradient_clip", 0.0),
            ema=ema,
            log_every=tcfg.get("log_every", 20),
            epoch=epoch,
            scaler=scaler,
        )
        log.info(f"train loss={train_metrics['loss']:.4f}")

        if ema is not None:
            ema.copy_to(model)

        val_result = evaluate(model, val_loader, device, precision, loss_fn=loss_fn)
        val_metrics = val_result["metrics"]
        val_auc = val_metrics.get("roc_auc", -1.0)
        log.info(f"val metrics={val_metrics}")

        if ema is not None:
            # 评估后把 EMA 状态也存下来
            ema_state = ema.state_dict()
        else:
            ema_state = None

        is_best = val_auc > best_auc
        if is_best or args.fold == -1:
            best_auc = max(best_auc, val_auc)
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                ema_state,
                epoch,
                best_auc,
                cfg,
            )
            log.info(f"saved best checkpoint (auc={best_auc:.4f})")

        if len(val_df) > 0:
            for sid, p, lab in zip(val_result["sample_id"], val_result["probs"], val_result["labels"]):
                oof_records.append({"sample_id": sid, "prob": float(p), "label": int(lab), "fold": args.fold})

    if oof_records:
        oof_df = pd.DataFrame(oof_records)
        oof_path = resolve_code_root() / "outputs" / "oof" / f"{tag}.csv"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df.to_csv(oof_path, index=False)
        oof_auc = compute_metrics(oof_df["label"], oof_df["prob"])["roc_auc"]
        log.info(f"OOF saved to {oof_path} auc={oof_auc:.4f}")

    log.info("training done")


if __name__ == "__main__":
    main()
