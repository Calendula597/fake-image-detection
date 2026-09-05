"""用 Kolors（快手开源，可灵系代理）从 COCO caption 生成假图。

动机：比赛测试集若含可灵/快手系生成器，Kolors 是本地最近代理（与 HunyuanDiT 互补）。
支持 --zh-frac 中文 prompt（复用 hydit_zh_prompts.json）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_image_detection.paths import resolve_code_root
from generate_flux import GEN_SIZES, load_coco_captions

CR = resolve_code_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2028)
    ap.add_argument("--out", default="data_real/kolors")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--zh-frac", type=float, default=0.4)
    args = ap.parse_args()

    from diffusers import KolorsPipeline

    out_dir = CR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    caps = load_coco_captions(args.start + args.n, seed=2027)
    caps = caps.iloc[args.start:].reset_index(drop=True)

    zh_prompts = {}
    zh_path = CR / "data_real" / "hydit_zh_prompts.json"
    if args.zh_frac > 0 and zh_path.exists():
        zh_prompts = json.loads(zh_path.read_text())
        print(f"loaded {len(zh_prompts)} zh prompts")

    def _prompt(global_idx, caption):
        if zh_prompts and random.random() < args.zh_frac:
            zh = zh_prompts.get(str(global_idx))
            if zh:
                return zh
        return caption

    pipe = KolorsPipeline.from_pretrained(
        "Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    manifest = []
    for i, row in caps.iterrows():
        name = f"kolors_{args.start + i:05d}.jpg"
        out_path = out_dir / name
        if out_path.exists():
            continue
        w, h = GEN_SIZES[(args.start + i) % len(GEN_SIZES)]
        w, h = max(256, round(w / 64) * 64), max(256, round(h / 64) * 64)  # SDXL 系 64 对齐
        gen = torch.Generator(device="cuda").manual_seed(args.seed + args.start + i)
        img = pipe(
            prompt=_prompt(args.start + i, row["caption"]),
            width=w,
            height=h,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
            generator=gen,
        ).images[0]
        q = random.randint(75, 100)
        img.convert("RGB").save(out_path, "JPEG", quality=q)
        manifest.append({"id": name, "caption": row["caption"], "w": w, "h": h, "q": q})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(caps)}", flush=True)
            pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"done -> {out_dir} ({len(manifest)} images)")


if __name__ == "__main__":
    main()
