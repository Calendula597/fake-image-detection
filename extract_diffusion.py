from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()


@torch.no_grad()
def noise_pred_error(pipe, img_tensor, timestep, device, null_emb):
    """单步噪声预测误差：zt = sqrt(a)z0 + sqrt(1-a)eps，UNet 预测 eps，返回 ||eps - eps_hat||。
    AI 图（扩散分布内）误差应更低。"""
    scheduler = pipe.scheduler
    z0 = pipe.vae.encode(img_tensor).latent_dist.mean
    eps = torch.randn_like(z0)
    t = torch.tensor([timestep], device=device).long()
    zt = scheduler.add_noise(z0, eps, t)
    emb = null_emb.repeat(img_tensor.shape[0], 1, 1)
    eps_hat = pipe.unet(zt, t, encoder_hidden_states=emb).sample
    err = F.mse_loss(eps_hat, eps, reduction="none").mean(dim=(1, 2, 3))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--timestep", type=int, default=250)
    ap.add_argument("--n-noise", type=int, default=4, help="重复加噪次数取平均")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-prefix", default="diff")
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    # 空 prompt 的 null 文本嵌入（UNet 需要的条件）
    with torch.no_grad():
        tok = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        null_emb = pipe.text_encoder(tok.input_ids.to(device))[0]

    data_root = resolve_data_root()
    FEAT = CR / "outputs" / "features"
    FEAT.mkdir(parents=True, exist_ok=True)

    from torchvision import transforms as T
    tf = T.Compose([
        T.Resize((args.size, args.size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    splits = [("sample", sample_csv(), "data/image_sample_data")]
    if not args.sample_only:
        splits.append(("test", test_csv(), "data/image_test"))

    for split, csv, rel in splits:
        df = pd.read_csv(csv)
        ids = df["id"].tolist()
        errs = np.zeros(len(ids), dtype=np.float64)
        t0 = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(ids), args.batch_size), desc=f"diff {split}"):
                batch_ids = ids[i:i + args.batch_size]
                imgs = []
                for sid in batch_ids:
                    try:
                        im = Image.open(data_root / rel / sid).convert("RGB")
                    except Exception:
                        im = Image.new("RGB", (args.size, args.size))
                    imgs.append(tf(im))
                img_tensor = torch.stack(imgs).to(device, pipe.vae.dtype)
                acc = torch.zeros(len(batch_ids), device=device)
                for _ in range(args.n_noise):
                    acc += noise_pred_error(pipe, img_tensor, args.timestep, device, null_emb)
                errs[i:i + len(batch_ids)] = (acc / args.n_noise).float().cpu().numpy()
        np.save(FEAT / f"{args.out_prefix}_err_{split}.npy", errs.astype(np.float32))
        print(f"saved {args.out_prefix}_err_{split} {errs.shape} mean={errs.mean():.4f} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
