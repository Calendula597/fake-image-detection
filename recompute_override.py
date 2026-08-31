from __future__ import annotations
import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()


def sha256_of(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def phash_of(p: Path):
    import imagehash
    from PIL import Image
    try:
        with Image.open(p) as im:
            return str(imagehash.phash(im.convert("RGB")))
    except Exception:
        return None


def _work(args):
    sid, d = args
    p = Path(d) / sid
    if not p.exists():
        return sid, None, None
    return sid, sha256_of(p), phash_of(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="artifacts/label_override_sem.csv")
    args = ap.parse_args()

    data_root = resolve_data_root()
    test_dir = data_root / "data" / "image_test"
    lab = pd.read_csv(sample_csv())
    test = pd.read_csv(test_csv())

    # 训练图哈希（基于 sem_image 训练集现场计算，确保一致）
    train_dir = data_root / "data" / "image_sample_data"
    print("computing train hashes ...")
    train_sha, train_ph = {}, {}
    with Pool(args.workers) as pool:
        for sid, sha, ph in pool.imap_unordered(_work, [(s, train_dir) for s in lab["id"]], chunksize=32):
            train_sha[sid] = sha
            train_ph[sid] = ph
    sha2label = {}
    ph2label = {}
    for _, r in lab.iterrows():
        sid = r["id"]
        if train_sha.get(sid):
            sha2label[train_sha[sid]] = float(r["label"])
        if train_ph.get(sid):
            ph2label[train_ph[sid]] = float(r["label"])

    # 测试图哈希
    print("computing test hashes ...")
    override = {}
    with Pool(args.workers) as pool:
        for sid, sha, ph in pool.imap_unordered(_work, [(s, test_dir) for s in test["id"]], chunksize=64):
            if sha and sha in sha2label:
                override[sid] = sha2label[sha]
            elif ph and ph in ph2label:
                override[sid] = ph2label[ph]

    print(f"overrides: {len(override)}")
    if override:
        out = pd.DataFrame({"id": list(override.keys()), "label": list(override.values())})
        out_path = CR / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"label dist: {out['label'].value_counts().to_dict()}")
        print(f"saved -> {out_path}")
    else:
        print("no overrides found")


if __name__ == "__main__":
    main()
