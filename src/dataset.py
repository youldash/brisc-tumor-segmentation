"""PyTorch Dataset for BRISC 2025 segmentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_index import load_pair


class BriscSegDataset(Dataset):
    """Returns (image [3,H,W] float32, mask [1,H,W] float32 in {0,1})."""

    def __init__(self, df: pd.DataFrame, data_root: str | Path, transform=None, return_meta: bool = False):
        self.df = df.reset_index(drop=True).fillna({"mask": ""})
        self.root = Path(data_root)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image, mask = load_pair(self.root, row)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)

        if self.return_meta:
            meta = {"stem": row["stem"], "tumor_type": row["tumor_type"], "plane": row["plane"]}
            return image_t, mask_t, meta
        return image_t, mask_t
