from __future__ import annotations
import contextlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .logger import get_logger
from .metrics import compute_metrics

log = get_logger("engine")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model_sd = model.state_dict()
        for k, v in self.shadow.items():
            model_sd[k].copy_(v)
        model.load_state_dict(model_sd)

    def state_dict(self):
        return dict(self.shadow)


def _autocast(precision: str):
    if precision == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    precision,
    accum_steps,
    clip_grad=0.0,
    ema: Optional[EMA] = None,
    log_every=20,
    epoch=0,
    scaler=None,
):
    model.train()
    optimizer.zero_grad()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True) if "label" in batch else None
        if labels is None:
            continue
        with _autocast(precision):
            out = model(images)
            logits = out["logits"]
            loss = loss_fn(logits, labels)
            if "weight" in batch:
                w = batch["weight"].to(device, non_blocking=True)
                loss = (loss.view(-1) * w).mean() / accum_steps
            else:
                loss = loss.mean() / accum_steps
        if precision == "fp16" and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            if clip_grad and clip_grad > 0:
                if precision == "fp16" and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if precision == "fp16" and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
        total += loss.item() * accum_steps * len(labels)
        n += len(labels)
        if log_every and i % log_every == 0:
            log.info(
                f"epoch {epoch} step {i}/{len(loader)} loss {loss.item() * accum_steps:.4f} "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )
    return {"loss": total / max(n, 1)}


@torch.no_grad()
def evaluate(model, loader, device, precision="bf16", loss_fn=None):
    model.eval()
    probs, labels, sids = [], [], []
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        sids.extend(batch["sample_id"])
        with _autocast(precision):
            out = model(images)
            logits = out["logits"]
            p = torch.sigmoid(logits.float())
        probs.append(p.cpu().numpy())
        if "label" in batch:
            lab = batch["label"].cpu().numpy()
            labels.append(lab)
            if loss_fn is not None:
                ll = loss_fn(logits, batch["label"].to(device)).mean().item()
                total_loss += ll * len(lab)
                n += len(lab)
    if not probs:
        return {"probs": np.array([]), "labels": None, "sample_id": [], "metrics": {}}
    probs = np.concatenate(probs)
    metrics = {}
    if labels:
        labels = np.concatenate(labels)
        metrics = compute_metrics(labels, probs)
        if n:
            metrics["val_loss"] = total_loss / n
    return {"probs": probs, "labels": labels if labels is not None else None, "sample_id": sids, "metrics": metrics}
