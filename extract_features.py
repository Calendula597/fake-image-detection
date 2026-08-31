from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from fake_image_detection.cf_model import load_commfor
from fake_image_detection.clip_model import find_local_ckpt, load_openclip
from fake_image_detection.feature_extractor import extract_cf_features, extract_clip_features, save_features
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv

CR = resolve_code_root()
FEAT_DIR = CR / "outputs" / "features"


def extract_commfor(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    repo = CR / args.repo
    model, cfg_size = load_commfor(repo, device)
    size = args.size or cfg_size
    tag = args.tag or Path(args.repo).name.replace("commfor-model-", "cf")
    is_crop = not args.no_crop

    lab = pd.read_csv(sample_csv())
    test = pd.read_csv(test_csv())

    for split, df, rel in [("sample", lab, "data/image_sample_data"), ("test", test, "data/image_test")]:
        feats, logits = extract_cf_features(
            model, df["id"].tolist(), rel, size, is_crop, device, batch_size=args.batch_size
        )
        mode = "crop" if is_crop else "resize"
        save_features(FEAT_DIR, f"commfor_{tag}", size, mode, split, feats)
        save_features(FEAT_DIR, f"commfor_{tag}", size, f"{mode}_logits", split, logits)
        print(f"saved commfor_{tag}_{size}_{mode}_{split} {feats.shape}")


def extract_clip(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt
    if not ckpt:
        ckpt_path = find_local_ckpt(CR, [args.arch_dir])
        if ckpt_path:
            ckpt = str(ckpt_path)
    model = load_openclip(args.arch, ckpt, device, pretrained=args.pretrained_tag)

    lab = pd.read_csv(sample_csv())
    test = pd.read_csv(test_csv())

    for mode in ("crop", "resize"):
        for split, df, rel in [("sample", lab, "data/image_sample_data"), ("test", test, "data/image_test")]:
            feats = extract_clip_features(
                model, df["id"].tolist(), rel, args.size, mode, device, batch_size=args.batch_size
            )
            save_features(FEAT_DIR, args.tag, args.size, mode, split, feats)
            print(f"saved {args.tag}_{args.size}_{mode}_{split} {feats.shape}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cf = sub.add_parser("commfor")
    p_cf.add_argument("--repo", default="weights/commfor/commfor-model-384")
    p_cf.add_argument("--tag", default=None)
    p_cf.add_argument("--size", type=int, default=0)
    p_cf.add_argument("--no-crop", action="store_true")
    p_cf.add_argument("--batch-size", type=int, default=64)

    p_clip = sub.add_parser("clip")
    p_clip.add_argument("--arch", default="ViT-H-14")
    p_clip.add_argument("--arch-dir", default="clip-vit-h-14")
    p_clip.add_argument("--ckpt", default="")
    p_clip.add_argument("--tag", default="clipH")
    p_clip.add_argument("--size", type=int, default=224)
    p_clip.add_argument("--batch-size", type=int, default=48)
    p_clip.add_argument("--pretrained-tag", default="dfn5b")

    args = ap.parse_args()
    if args.cmd == "commfor":
        extract_commfor(args)
    elif args.cmd == "clip":
        extract_clip(args)


if __name__ == "__main__":
    main()
