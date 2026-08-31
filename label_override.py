from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv

CR = resolve_code_root()


def build_label_override(manifest_path: str):
    """基于 train/test 的 SHA256 和 pHash 匹配生成标签覆盖表。"""
    m = pd.read_csv(manifest_path)
    tr = m[m["split"] == "train"]
    te = m[m["split"] == "test"]
    override = {}

    tr_sha = dict(zip(tr.sha256.dropna(), tr.label.dropna()))
    for _, r in te[te.sha256.notna()].iterrows():
        if r["sha256"] in tr_sha:
            override[r["sample_id"]] = float(tr_sha[r["sha256"]])

    tr_ph = dict(zip(tr.phash.dropna(), tr.label.dropna()))
    for _, r in te[te.phash.notna()].iterrows():
        if r["sample_id"] in override:
            continue
        if r["phash"] in tr_ph:
            override[r["sample_id"]] = float(tr_ph[r["phash"]])
    return override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/data_manifest.csv")
    ap.add_argument("--out", default="artifacts/label_override.csv")
    args = ap.parse_args()

    manifest_path = CR / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}. Run audit_dataset.py first.")

    override = build_label_override(str(manifest_path))
    if not override:
        print("no overrides found")
        return

    out = pd.DataFrame({"id": list(override.keys()), "label": list(override.values())})
    out_path = CR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {len(out)} overrides -> {out_path}")
    print(f"label dist: {out['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
