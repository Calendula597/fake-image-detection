from __future__ import annotations
import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin

from fake_image_detection.paths import resolve_code_root, resolve_data_root, sample_csv, test_csv

CR = resolve_code_root()


def _stable_int(value) -> int:
    raw = json.dumps(value, sort_keys=True, default=str).encode()
    return int(hashlib.sha1(raw).hexdigest()[:12], 16)


def extract_one(path: Path) -> dict:
    size = path.stat().st_size
    with Image.open(path) as im:
        width, height = im.size
        info = im.info
        qtables = im.quantization or {}
        exif = im.getexif()
        row = {
            "file_size": size,
            "log_file_size": np.log1p(size),
            "width": width,
            "height": height,
            "pixels": width * height,
            "aspect": width / max(height, 1),
            "bytes_per_pixel": size / max(width * height, 1),
            "sampling": JpegImagePlugin.get_sampling(im),
            "progressive": float(bool(info.get("progressive") or info.get("progression"))),
            "jfif": float(info.get("jfif", -1)),
            "jfif_unit": float(info.get("jfif_unit", -1)),
            "jfif_dx": float((info.get("jfif_density") or (-1, -1))[0]),
            "jfif_dy": float((info.get("jfif_density") or (-1, -1))[1]),
            "dpi_x": float((info.get("dpi") or (-1, -1))[0]),
            "dpi_y": float((info.get("dpi") or (-1, -1))[1]),
            "exif_len": len(exif),
            "icc_len": len(info.get("icc_profile") or b""),
            "comment_len": len(info.get("comment") or b""),
            "qtables_count": len(qtables),
            "qtable_fingerprint": _stable_int(qtables) % 1000003,
            "exif_fingerprint": _stable_int(dict(exif)) % 1000003,
        }
        for table_id in range(4):
            table = list(qtables.get(table_id, []))
            table = (table + [-1] * 64)[:64]
            for idx, value in enumerate(table):
                row[f"qt{table_id}_{idx}"] = float(value)
            valid = [v for v in table if v >= 0]
            row[f"qt{table_id}_sum"] = float(sum(valid)) if valid else -1
            row[f"qt{table_id}_mean"] = float(np.mean(valid)) if valid else -1
        arr = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        for channel in range(3):
            x = arr[:, :, channel]
            row[f"mean_c{channel}"] = float(x.mean())
            row[f"std_c{channel}"] = float(x.std())
            row[f"sat_low_c{channel}"] = float((x < 0.02).mean())
            row[f"sat_high_c{channel}"] = float((x > 0.98).mean())
        gray = arr.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        row["edge_mean"] = float((dx.mean() + dy.mean()) / 2)
        row["edge_std"] = float((dx.std() + dy.std()) / 2)
        row["block8_h"] = float(np.abs(gray[:, 7:-1:8] - gray[:, 8::8]).mean())
        row["block8_v"] = float(np.abs(gray[7:-1:8, :] - gray[8::8, :]).mean())
        return row


def _work(args):
    sid, d = args
    try:
        row = {"id": sid}
        row.update(extract_one(Path(d) / sid))
        return row
    except Exception:
        return {"id": sid}


def extract_split(csv_path, image_dir, desc, workers):
    src = pd.read_csv(csv_path)
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap(_work, [(s, image_dir) for s in src["id"]], chunksize=64)):
            rows.append(row)
            if i % 2000 == 0:
                print(f"{desc}: {i}/{len(src)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="artifacts/metadata_fingerprints_sem.csv")
    args = ap.parse_args()

    data = resolve_data_root()
    train = extract_split(sample_csv(), data / "data" / "image_sample_data", "train", args.workers)
    train["split"] = "train"
    test = extract_split(test_csv(), data / "data" / "image_test", "test", args.workers)
    test["split"] = "test"
    out = pd.concat([train, test], ignore_index=True)
    out_path = CR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
