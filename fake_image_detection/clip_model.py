from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def load_openclip(
    arch: str,
    ckpt_path: Optional[str | Path] = None,
    device: torch.device = torch.device("cpu"),
    pretrained: str = "dfn5b",
) -> nn.Module:
    """加载 OpenCLIP 模型。"""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=None, device="cpu")
    if ckpt_path and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[openclip] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[openclip] no local ckpt; trying pretrained={pretrained} for {arch}")
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="cpu")
    return model.to(device).eval()


def find_local_ckpt(code_root: Path, candidates: list[str]) -> Optional[Path]:
    """在 weights/ 下按候选名查找 open_clip_pytorch_model.bin。"""
    for name in candidates:
        p = code_root / "weights" / name / "open_clip_pytorch_model.bin"
        if p.exists() and p.stat().st_size > 100_000_000:
            return p
    return None
