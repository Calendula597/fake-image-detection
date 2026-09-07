"""生成多轮转发退化变体（deg2）：2-3 轮 (缩放回采 + JPEG q25-65 重编码 + 偶发模糊)。

动机（09-07 实测）：多轮退化让 cf384/clipH378 头崩盘到 0.76-0.78，
比赛"社交平台多层转发"正是这个强度。单轮 deg 池（data_real_deg）太温和。

输入 data_real/ 全部 split（holdout_* 除外），输出 data_real_deg2/deg2_<split>/。
"""
from __future__ import annotations

import io
import random
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_image_detection.paths import resolve_code_root

CR = resolve_code_root()
SRC = CR / "data_real"
DST = CR / "data_real_deg2"

SKIP = {"holdout_midjourney", "holdout_adm"}


def degrade(args):
    src_path, dst_path, seed = args
    if dst_path.exists():
        return 0
    rng = random.Random(seed)
    try:
        im = Image.open(src_path).convert("RGB")
    except Exception:
        return 0
    for _ in range(rng.choice([2, 3])):
        w, h = im.size
        s = rng.choice([0.4, 0.5, 0.6])
        im = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if rng.random() < 0.5:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(25, 65))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst_path, "JPEG", quality=90)
    return 1


def main():
    jobs = []
    for d in sorted(SRC.glob("*")):
        if not d.is_dir() or d.name in SKIP or not list(d.glob("*.jpg")):
            continue
        out_dir = DST / f"deg2_{d.name.replace('coco_val_ai_', '').replace('genimage_', '')}"
        if d.name == "coco_val":
            out_dir = DST / "deg2_coco"
        for f in sorted(d.glob("*.jpg")):
            jobs.append((f, out_dir / f.name, hash((d.name, f.name)) & 0x7FFFFFFF))
    print(f"{len(jobs)} jobs")
    with Pool(16) as pool:
        done = sum(pool.imap_unordered(degrade, jobs, chunksize=32))
    print(f"generated {done}")


if __name__ == "__main__":
    main()
