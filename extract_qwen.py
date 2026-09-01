from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="materials/Qwen3.5-0.8B")
    ap.add_argument("--out-prefix", default="qwen_zs")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=device
    ).eval()
    tok_ai = proc.tokenizer.encode(" AI", add_special_tokens=False)[0]
    tok_real = proc.tokenizer.encode(" Real", add_special_tokens=False)[0]
    data_root = resolve_data_root()

    FEAT = CR / "outputs" / "features"
    FEAT.mkdir(parents=True, exist_ok=True)

    for split, csv, rel in [("sample", sample_csv(), "data/image_sample_data"), ("test", test_csv(), "data/image_test")]:
        df = pd.read_csv(csv)
        ids = df["id"].tolist()
        scores = np.zeros(len(ids), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i, sid in enumerate(ids):
                try:
                    img = Image.open(data_root / rel / sid).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224))
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Is this image AI-generated or a real photograph? Answer with one word: AI or Real."},
                ]}]
                inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
                logits = model(**inp).logits[0, -1]
                scores[i] = 1 / (1 + math.exp(-(logits[tok_ai].item() - logits[tok_real].item())))
                if (i + 1) % 2000 == 0:
                    print(f"{split} {i+1}/{len(ids)} {(time.time()-t0)/(i+1):.2f}s/img", flush=True)
        np.save(FEAT / f"{args.out_prefix}_{split}.npy", scores)
        print(f"saved {args.out_prefix}_{split} {scores.shape} mean={scores.mean():.4f} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
