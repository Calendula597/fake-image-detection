from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from fake_image_detection.paths import resolve_code_root, submission_example_csv


def build_submission(
    predictions: pd.DataFrame,
    out_path: str,
    sample_submission: Optional[str] = None,
    clip: bool = True,
) -> Path:
    if "probability" in predictions.columns and "score" not in predictions.columns:
        predictions = predictions.rename(columns={"probability": "score"})
    if "id" not in predictions.columns or "score" not in predictions.columns:
        raise ValueError("predictions must have 'id' and 'score' columns")

    ss_path = Path(sample_submission) if sample_submission else submission_example_csv()
    if not ss_path.exists():
        raise FileNotFoundError(f"sample submission not found: {ss_path}")
    ss = pd.read_csv(ss_path)

    pred = predictions.set_index("id")["score"]
    out = ss[["id"]].copy()
    out["score"] = out["id"].map(pred)
    if out["score"].isna().any():
        missing = int(out["score"].isna().sum())
        raise ValueError(f"{missing} test ids missing from predictions")
    if clip:
        out["score"] = out["score"].clip(0.0, 1.0)
    assert out["score"].notna().all(), "NaN scores present"
    assert out["id"].is_unique, "duplicate ids"
    assert list(out.columns) == ["id", "score"], f"bad columns {out.columns.tolist()}"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    sha_path.write_text(f"{h}  {out_path.name}\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="path to predictions CSV (id,score)")
    ap.add_argument("--out", default="outputs/submissions/submission.csv")
    ap.add_argument("--sample-submission", default=None)
    ap.add_argument("--override-csv", default=None, help="optional id,label CSV for near-duplicate override")
    args = ap.parse_args()

    cr = resolve_code_root()
    pred = pd.read_csv(args.predictions)

    if args.override_csv:
        ov_path = Path(args.override_csv)
        if not ov_path.is_absolute():
            ov_path = cr / ov_path
        if ov_path.exists():
            extra = pd.read_csv(ov_path)
            pred = pred.set_index("id")
            for _, r in extra.iterrows():
                sid = str(r["id"])
                if sid in pred.index:
                    pred.at[sid, "score"] = float(r["label"])
            pred = pred.reset_index()
            print(f"applied {len(extra)} label overrides")

    out_path = cr / args.out
    build_submission(pred, str(out_path), sample_submission=args.sample_submission)
    print(f"submission saved -> {out_path}")


if __name__ == "__main__":
    main()
