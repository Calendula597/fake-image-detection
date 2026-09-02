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

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv

CR = resolve_code_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strength", type=float, default=0.55, help="img2img 去噪强度（越大越偏离原图）")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out-dir", default="outputs/bfree_fakes")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from diffusers import StableDiffusionImg2ImgPipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", dtype=torch.bfloat16, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    data_root = resolve_data_root()
    lab = pd.read_csv(sample_csv())
    real_ids = lab[lab.label == 0]["id"].tolist()
    if args.limit:
        real_ids = real_ids[: args.limit]

    out_dir = CR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for sid in tqdm(real_ids, desc="gen bfree fakes"):
            try:
                img = Image.open(data_root / "data" / "image_sample_data" / sid).convert("RGB").resize((args.size, args.size))
            except Exception:
                continue
            out = pipe(
                prompt="a photo",
                image=img,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            ).images[0]
            out.save(out_dir / sid, quality=95)
            manifest.append(sid)
    pd.DataFrame({"id": manifest}).to_csv(out_dir / "manifest.csv", index=False)
    print(f"generated {len(manifest)} debiased fakes -> {out_dir}")


if __name__ == "__main__":
    main()
