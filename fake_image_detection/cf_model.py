from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    """Community Forensics ViT 分类器（与官方实现保持一致）。"""

    def __init__(
        self,
        model_size: str = "small",
        input_size: int = 384,
        patch_size: int = 16,
        freeze_backbone: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        import timm

        self.device = device

        if model_size == "small":
            if input_size == 224:
                if patch_size == 32:
                    name = "vit_small_patch32_224.augreg_in21k_ft_in1k"
                else:
                    name = "vit_small_patch16_224.augreg_in21k_ft_in1k"
            else:
                if patch_size == 32:
                    name = "vit_small_patch32_384.augreg_in21k_ft_in1k"
                else:
                    name = "vit_small_patch16_384.augreg_in21k_ft_in1k"
        elif model_size == "tiny":
            assert patch_size == 16, "Only patch size 16 is available for ViT-Ti"
            if input_size == 224:
                name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
            else:
                name = "vit_tiny_patch16_384.augreg_in21k_ft_in1k"
        else:
            name = f"vit_{model_size}_patch{patch_size}_{input_size}"

        self.vit = timm.create_model(name, pretrained=False).to(device)
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        feat_dim = self.vit.num_features
        self.vit.head = nn.Linear(in_features=feat_dim, out_features=1, bias=True, device=device)

    def forward(self, x):
        return self.vit(x)


def load_commfor(repo_path: str | Path, device: torch.device) -> Tuple[nn.Module, int]:
    """从本地目录加载 Community Forensics 权重。"""
    from safetensors.torch import load_file

    repo_path = Path(repo_path)
    cfg_path = repo_path / "config.json"
    weight_path = repo_path / "model.safetensors"
    if not cfg_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"missing config/weights under {repo_path}")

    cfg = json.loads(cfg_path.read_text())
    model = ViTClassifier(
        model_size=cfg.get("model_size", "small"),
        input_size=int(cfg.get("input_size", 384)),
        patch_size=int(cfg.get("patch_size", 16)),
        freeze_backbone=True,
        device="cpu",
    )
    sd = load_file(str(weight_path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[commfor] loaded {weight_path} | missing={len(missing)} unexpected={len(unexpected)} | "
        f"size={cfg.get('input_size')} model={cfg.get('model_size')}"
    )
    return model.to(device).eval(), int(cfg.get("input_size", 384))
