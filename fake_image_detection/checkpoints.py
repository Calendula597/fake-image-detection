from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(path, model, optimizer, scheduler, ema_state, epoch, best_auc, cfg, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": _strip_module(model.state_dict()),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema_state if ema_state is not None else None,
        "epoch": epoch,
        "best_auc": best_auc,
        "cfg": cfg,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def _strip_module(sd):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
