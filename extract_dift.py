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
from torchvision import transforms as T
from tqdm import tqdm

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--timestep", type=int, default=250)
    ap.add_argument("--block", default="mid", choices=["mid", "up2", "down0"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-prefix", default="dift")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    tok = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        null_emb = pipe.text_encoder(tok.input_ids.to(device))[0]

    feats = {}
    def hook(name):
        def h(module, inp, out):
            feats[name] = out
        return h
    if args.block == "mid":
        pipe.unet.mid_block.register_forward_hook(hook("f"))
    elif args.block == "up2":
        pipe.unet.up_blocks[2].register_forward_hook(hook("f"))
    else:
        pipe.unet.down_blocks[0].register_forward_hook(hook("f"))

    tf = T.Compose([
        T.Resize((args.size, args.size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    data_root = resolve_data_root()
    FEAT = CR / "outputs" / "features"
    FEAT.mkdir(parents=True, exist_ok=True)

    for split, csv, rel in [("sample", sample_csv(), "data/image_sample_data"), ("test", test_csv(), "data/image_test")]:
        df = pd.read_csv(csv)
        ids = df["id"].tolist()
        allf = []
        t0 = time.time()
        with torch.no_grad():
            for i in tqdm(range(0, len(ids), args.batch_size), desc=f"dift {split}"):
                imgs = []
                for sid in ids[i:i + args.batch_size]:
                    try:
                        im = Image.open(data_root / rel / sid).convert("RGB")
                    except Exception:
                        im = Image.new("RGB", (args.size, args.size))
                    imgs.append(tf(im))
                img_tensor = torch.stack(imgs).to(device, pipe.vae.dtype)
                z0 = pipe.vae.encode(img_tensor).latent_dist.mean
                t = torch.tensor([args.timestep], device=device).long()
                emb = null_emb.repeat(z0.shape[0], 1, 1)
                _ = pipe.unet(z0, t, encoder_hidden_states=emb).sample
                fmap = feats["f"]
                f = F.adaptive_avg_pool2d(fmap.float(), 1).reshape(fmap.shape[0], -1)
                allf.append(f.cpu().numpy())
        X = np.concatenate(allf, 0).astype(np.float32)
        np.save(FEAT / f"{args.out_prefix}_{args.block}_{split}.npy", X)
        print(f"saved {args.out_prefix}_{args.block}_{split} {X.shape} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
