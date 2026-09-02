"""SRM/Bayar 高通残差取证特征（NTIRE 2026 RAPID 思路，与 NPR 不同的取证线索）。

对每张图：
1. 灰度化后用 3 个经典 SRM 5x5 高通核 + 2 个 Bayar 3x3 约束核提取残差图。
2. 残差图 resize 到 64x64，统计 + 直方图特征。
3. 输出 train/test 特征矩阵 (.npy)，供 stack.py 作为新成员。

CPU 多进程即可，无需 GPU。
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"

# 经典 SRM 高通核 (5x5)
SRM_KERNELS = [
    # KB 3x3 中心
    np.array([[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    # 一阶边缘水平
    np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0,
    # 二阶对角
    np.array([[0, 0, 0, 0, 0],
              [0, -1, 0, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 1, 0, -1, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0,
    # SQUARE5x5
    np.array([[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
]

# Bayar 约束核 (3x3, 中心 -1, 其余和为 1)
BAYAR_KERNELS = [
    np.array([[-1, 2, -1],
              [2, -4, 2],
              [-1, 2, -1]], dtype=np.float32) / 4.0,
    np.array([[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]], dtype=np.float32) / 4.0,
]

KERNELS = SRM_KERNELS + BAYAR_KERNELS
RES = 64  # 残差图 resize 尺寸


def _conv2d_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """灰度图 valid 卷积（用 stride trick，避免 scipy 依赖）。"""
    kh, kw = k.shape
    ih, iw = img.shape
    if ih < kh or iw < kw:
        img = np.asarray(Image.fromarray(img).resize((max(iw, kw), max(ih, kh))))
        ih, iw = img.shape
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(img, (kh, kw))
    return np.tensordot(win, k, axes=((2, 3), (0, 1))).astype(np.float32)


def extract_one(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    # 统一工作分辨率，避免大图主导统计
    im = im.resize((256, 256))
    g = np.asarray(im, dtype=np.float32) / 255.0
    feats = []
    for k in KERNELS:
        r = _conv2d_valid(g, k)
        ra = np.abs(r)
        feats += [
            float(r.mean()), float(r.std()),
            float(ra.mean()), float(np.percentile(ra, 90)), float(np.percentile(ra, 99)),
            float((ra > 0.05).mean()), float((ra > 0.15).mean()),
        ]
        small = np.asarray(Image.fromarray(r).resize((RES, RES), Image.BILINEAR), dtype=np.float32)
        feats += [float(small.mean()), float(small.std())]
        # 残差能量直方图（8 bin）
        hist, _ = np.histogram(np.clip(ra, 0, 0.4), bins=8, range=(0, 0.4), density=True)
        feats += hist.tolist()
    return np.array(feats, dtype=np.float32)


def _work(args):
    sid, d = args
    try:
        return extract_one(str(Path(d) / sid))
    except Exception:
        return None


def extract_split(csv_path, image_dir, desc, workers=16):
    src = pd.read_csv(csv_path)
    feats, keep_ids = [], []
    with Pool(workers) as pool:
        for i, f in enumerate(pool.imap(_work, [(s, image_dir) for s in src["id"]], chunksize=32)):
            if f is not None:
                feats.append(f)
                keep_ids.append(src["id"].iloc[i])
            if i % 2000 == 0:
                print(f"{desc}: {i}/{len(src)}", flush=True)
    return np.stack(feats), keep_ids


def main():
    FEAT.mkdir(parents=True, exist_ok=True)
    data = resolve_data_root()
    Xs, ids_s = extract_split(sample_csv(), data / "data" / "image_sample_data", "train")
    Xt, ids_t = extract_split(test_csv(), data / "data" / "image_test", "test")
    np.save(FEAT / "srm_sample.npy", Xs)
    np.save(FEAT / "srm_test.npy", Xt)
    pd.Series(ids_s).to_csv(FEAT / "srm_sample_ids.csv", index=False)
    pd.Series(ids_t).to_csv(FEAT / "srm_test_ids.csv", index=False)
    print(f"saved {Xs.shape} / {Xt.shape} -> {FEAT}")


if __name__ == "__main__":
    main()
