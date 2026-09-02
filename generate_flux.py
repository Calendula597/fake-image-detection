"""用 FLUX.1-schnell（流匹配，赛事官方点名的架构）从 COCO caption 生成假图。

用途：
1. 自建测试集 v2：COCO 真实图 vs FLUX 假图，比 SD1.4 更接近比赛测试集的"新生成器"分布。
2. 训练池扩增：把 FLUX 假图当额外 AI 样本，对齐测试集的跨生成器分布（NTIRE 冠军策略）。

权重（GGUF 量化，fit 24G 显存）：
- weights/flux/flux1-schnell-Q8_0.gguf          (transformer, ~12.4GB)
- weights/flux/t5-v1_1-xxl-encoder-Q4_K_M.gguf  (T5 文本编码器, ~2.9GB)
- weights/flux/clip_l.safetensors               (CLIP 文本编码器)
- weights/flux/ae.safetensors                   (VAE)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_image_detection.paths import resolve_code_root, resolve_data_root

CR = resolve_code_root()
FLUX_DIR = CR / "weights" / "flux"

# 比赛训练集 AI 图的典型尺寸（按训练集分布采样）
GEN_SIZES = [
    (1024, 1024), (768, 768), (1280, 720), (1024, 576), (576, 1024),
    (512, 512), (768, 432), (1024, 768), (768, 1024), (720, 1280),
]


def load_pipeline(no_offload: bool = False):
    from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    from transformers import CLIPTextModel, T5EncoderModel

    gguf_q8 = FLUX_DIR / "flux1-schnell-Q8_0.gguf"
    t5_gguf = FLUX_DIR / "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
    mirror = "Niansuh/FLUX.1-schnell"  # 非 gated diffusers 镜像
    transformer = FluxTransformer2DModel.from_single_file(
        str(gguf_q8),
        config=mirror,
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        str(FLUX_DIR),  # 本地目录，config 从 GGUF 元数据读取
        gguf_file=t5_gguf.name,
        torch_dtype=torch.bfloat16,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        mirror, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKL.from_pretrained(
        mirror, subfolder="vae", torch_dtype=torch.bfloat16
    )
    from diffusers import FlowMatchEulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    import json
    sched_cfg = hf_hub_download(mirror, "scheduler/config.json")
    with open(sched_cfg) as f:
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(json.load(f))
    pipe = FluxPipeline.from_pretrained(
        mirror,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    if no_offload:
        pipe.to("cuda")  # Q8 transformer ~13G + T5 Q4 ~3G，24G 显存可全放 GPU，速度快数倍
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def load_coco_captions(n: int, seed: int = 0) -> pd.DataFrame:
    """流式拉取 COCO captions（不下载图片，几 MB 即可）。prompt 无需与真实图配对。"""
    from datasets import load_dataset

    ds = load_dataset("jxie/coco_captions", split="train", streaming=True)
    rows, seen = [], set()
    for ex in ds:
        cap = ex.get("caption") or ""
        iid = ex.get("image_id", len(rows))
        if cap and iid not in seen:
            seen.add(iid)
            rows.append({"image_id": iid, "caption": cap})
        if len(rows) >= n * 2:
            break
    df = pd.DataFrame(rows)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data_real/coco_val_ai_flux")
    ap.add_argument("--jpeg-quality", type=int, default=0, help=">0 则按该质量重编码；0=随机 75-100")
    ap.add_argument("--start", type=int, default=0, help="跳过前 start 条（断点续跑）")
    ap.add_argument("--no-offload", action="store_true", help="全部放 GPU（显存够时显著提速）")
    args = ap.parse_args()

    out_dir = CR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    caps = load_coco_captions(args.start + args.n, seed=args.seed)
    caps = caps.iloc[args.start:].reset_index(drop=True)

    pipe = load_pipeline(no_offload=args.no_offload)
    manifest = []
    for i, row in caps.iterrows():
        name = f"flux_{args.start + i:05d}.jpg"
        out_path = out_dir / name
        if out_path.exists():
            continue
        w, h = GEN_SIZES[(args.start + i) % len(GEN_SIZES)]
        gen = torch.Generator(device="cuda").manual_seed(args.seed + args.start + i)
        img = pipe(
            prompt=row["caption"],
            width=w,
            height=h,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=gen,
            max_sequence_length=256,
        ).images[0]
        q = args.jpeg_quality or random.randint(75, 100)
        img.convert("RGB").save(out_path, "JPEG", quality=q)
        manifest.append({"id": name, "caption": row["caption"], "w": w, "h": h, "q": q})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(caps)}", flush=True)
            pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    print(f"done -> {out_dir} ({len(manifest)} images)")


if __name__ == "__main__":
    main()
