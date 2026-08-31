from __future__ import annotations
import math

import torch


def build_scheduler(optimizer, cfg, steps_per_epoch: int, epochs: int):
    t = cfg.get("training", {})
    warmup_ratio = float(t.get("warmup_ratio", 0.05))
    min_lr = float(t.get("min_lr", 1e-6))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lambda_with_min(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        # 保证最终 lr 不低于 min_lr / base_lr 的比例
        return cos * (1 - min_lr / max(min(base_lrs), 1e-12)) + min_lr / max(min(base_lrs), 1e-12)

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_with_min)
    sched._base_lrs = base_lrs
    sched._min_lr = min_lr
    sched._warmup_steps = warmup_steps
    sched._total_steps = total_steps
    return sched
