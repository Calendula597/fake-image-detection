from __future__ import annotations

import timm
import torch
import torch.nn as nn


class DetectorModel(nn.Module):
    def forward(self, images, **kwargs):
        raise NotImplementedError


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float().reshape(-1))


class TimmDetector(DetectorModel):
    def __init__(
        self,
        name: str = "convnext_large.fb_in22k_ft_in1k",
        pretrained: bool = True,
        num_classes: int = 1,
        drop_path: float = 0.2,
        head_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(head_dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes
        self._name = name
        if gradient_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(enable=True)
            except Exception:
                pass

    def forward(self, images, **kwargs):
        feats = self.backbone(images)
        logits = self.head(feats).reshape(-1)
        return {"logits": logits, "features": feats}


def build_model(cfg) -> DetectorModel:
    m = cfg.get("model", {})
    name = m.get("name", "convnext_large")
    if name in ("convnext_large", "convnext_large.fb_in22k_ft_in1k"):
        name = "convnext_large.fb_in22k_ft_in1k"
    return TimmDetector(
        name=name,
        pretrained=m.get("pretrained", True),
        num_classes=m.get("num_classes", 1),
        drop_path=m.get("drop_path", 0.2),
        head_dropout=m.get("head_dropout", 0.1),
        gradient_checkpointing=m.get("gradient_checkpointing", False),
    )
