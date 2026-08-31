from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        t = targets
        if self.label_smoothing > 0:
            t = t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight, reduction="none")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        return (1 - pt) ** self.gamma * bce


def build_loss(cfg, pos_weight=None):
    name = cfg.get("training", {}).get("loss", "bce")
    ls = cfg.get("training", {}).get("label_smoothing", 0.0)
    pw = pos_weight
    if name == "focal":
        return FocalLoss(pos_weight=pw)
    return BCEWithLogitsLoss(pos_weight=pw, label_smoothing=ls)
