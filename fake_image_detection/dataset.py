from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io_utils import load_image


class DetectorDataset(Dataset):
    def __init__(
        self,
        manifest,
        data_root,
        transform: Optional[Callable] = None,
        return_label: bool = True,
        default_size: int = 384,
    ):
        if isinstance(manifest, pd.DataFrame):
            self.df = manifest.reset_index(drop=True)
        else:
            self.df = pd.read_csv(manifest).reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.return_label = return_label
        self.default_size = default_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row["relative_path"]
        path = self.data_root / rel
        img = load_image(path, mode="RGB")
        if img is None:
            img = np.zeros((self.default_size, self.default_size, 3), dtype=np.uint8)
            valid = False
        else:
            valid = True
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        sample = {"image": img, "sample_id": str(row["sample_id"]), "valid": valid, "idx": idx}
        if self.return_label and "label" in row and (not pd.isna(row["label"])):
            sample["label"] = torch.tensor(float(row["label"]), dtype=torch.float32)
        if "weight" in row and (not pd.isna(row.get("weight"))):
            sample["weight"] = torch.tensor(float(row["weight"]), dtype=torch.float32)
        return sample


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    sample_ids = [b["sample_id"] for b in batch]
    valid = torch.tensor([b["valid"] for b in batch], dtype=torch.bool)
    idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    out = {"image": images, "sample_id": sample_ids, "valid": valid, "idx": idx}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    if "weight" in batch[0]:
        out["weight"] = torch.stack([b["weight"] for b in batch], dim=0)
    return out
