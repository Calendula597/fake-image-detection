import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_image_detection.paths import resolve_data_root
from submit import build_submission


def test_build_submission():
    root = resolve_data_root()
    ss = pd.read_csv(root / "image_submission_example.csv")
    pred = pd.DataFrame({"id": ss["id"], "score": 0.5})
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.csv"
        build_submission(pred, str(out), sample_submission=str(root / "image_submission_example.csv"))
        sub = pd.read_csv(out)
        assert list(sub.columns) == ["id", "score"]
        assert len(sub) == len(ss)
        assert list(sub["id"]) == list(ss["id"])
    print("test_build_submission passed")


if __name__ == "__main__":
    test_build_submission()
