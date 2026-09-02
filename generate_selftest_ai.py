from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from fake_image_detection.paths import resolve_code_root

CR = resolve_code_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="data_real/coco_val")
    ap.add_argument("--model", default="sd14", choices=["sd14", "sd15"])
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from diffusers import StableDiffusionImg2ImgPipeline
    repo = {"sd14": "CompVis/stable-diffusion-v1-4", "sd15": "runwayml/stable-diffusion-v1-5"}[args.model]
    tag = args.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo, dtype=torch.bfloat16, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    real_dir = CR / args.real_dir
    out_dir = real_dir.parent / (real_dir.name + f"_ai_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(real_dir.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    with torch.no_grad():
        for p in tqdm(imgs, desc=f"gen {tag} ai"):
            try:
                img = Image.open(p).convert("RGB").resize((512, 512))
            except Exception:
                continue
            out = pipe(prompt="a photo", image=img, strength=args.strength, num_inference_steps=args.steps, guidance_scale=2.0).images[0]
            out.save(out_dir / p.name, quality=95)
    print(f"generated {len(list(out_dir.glob('*.jpg')))} AI images ({tag}) -> {out_dir}")


if __name__ == "__main__":
    main()
