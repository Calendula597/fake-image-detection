from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from fake_image_detection.paths import submission_example_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sample-submission", default=None)
    args = ap.parse_args()

    sub = pd.read_csv(args.submission)
    ss_path = Path(args.sample_submission) if args.sample_submission else submission_example_csv()
    ss = pd.read_csv(ss_path)

    errors = []
    if list(sub.columns) != ["id", "score"]:
        errors.append(f"columns must be [id, score], got {sub.columns.tolist()}")
    if len(sub) != len(ss):
        errors.append(f"row count {len(sub)} != expected {len(ss)}")
    if not sub["id"].is_unique:
        errors.append("id column has duplicates")
    if sub["score"].isna().any():
        errors.append(f"{int(sub['score'].isna().sum())} NaN scores")
    if (sub["score"] < 0).any() or (sub["score"] > 1).any():
        errors.append("scores out of [0,1]")
    if set(sub["id"]) != set(ss["id"]):
        missing = set(ss["id"]) - set(sub["id"])
        extra = set(sub["id"]) - set(ss["id"])
        if missing:
            errors.append(f"{len(missing)} missing ids")
        if extra:
            errors.append(f"{len(extra)} extra ids")
    order_ok = list(sub["id"]) == list(ss["id"])

    print(f"submission: {args.submission}")
    print(f"rows: {len(sub)}  expected: {len(ss)}")
    print(f"columns: {sub.columns.tolist()}")
    print(f"score range: [{sub['score'].min():.4f}, {sub['score'].max():.4f}]  mean={sub['score'].mean():.4f}")
    print(f"id order matches sample: {order_ok}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("VALID ✓")


if __name__ == "__main__":
    main()
