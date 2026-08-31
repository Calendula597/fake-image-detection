import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fake_image_detection.dataset import DetectorDataset
from fake_image_detection.paths import resolve_data_root


def test_dataset_loads():
    root = resolve_data_root()
    df = pd.read_csv(root / "image_sample_data.csv")
    df["sample_id"] = df["id"].astype(str)
    df["relative_path"] = "data/image_sample_data/" + df["id"].astype(str)
    ds = DetectorDataset(df, root, transform=None)
    sample = ds[0]
    assert "image" in sample
    assert "sample_id" in sample
    assert "label" in sample
    print("test_dataset_loads passed")


if __name__ == "__main__":
    test_dataset_loads()
