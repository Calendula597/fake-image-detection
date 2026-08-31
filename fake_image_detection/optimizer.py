from __future__ import annotations

import torch
import torch.nn as nn


def _is_no_decay(name: str) -> bool:
    return name.endswith("bias") or "norm" in name.lower() or name.endswith("gamma") or name.endswith("beta")


def param_groups_layer_decay(model, backbone_lr, head_lr, weight_decay, no_decay_bn_bias=True, layer_decay=0.9):
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = not (n.startswith("backbone.") or n.startswith("model."))
        nd = no_decay_bn_bias and _is_no_decay(n)
        if is_head:
            (head_no_decay if nd else head_decay).append(p)
        else:
            (backbone_no_decay if nd else backbone_decay).append(p)
    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_nd"})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "name": "head"})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "head_nd"})
    return groups


def build_optimizer(model, cfg):
    t = cfg.get("training", {})
    backbone_lr = float(t.get("backbone_lr", 2e-5))
    head_lr = float(t.get("head_lr", 0.0001))
    wd = float(t.get("weight_decay", 0.05))
    layer_decay = float(t.get("layer_decay", 0.9))
    no_decay_bn_bias = bool(t.get("no_decay_bn_bias", True))
    groups = param_groups_layer_decay(model, backbone_lr, head_lr, wd, no_decay_bn_bias, layer_decay)
    name = t.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=head_lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=head_lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=head_lr, momentum=0.9)
    raise ValueError(f"unknown optimizer {name}")
