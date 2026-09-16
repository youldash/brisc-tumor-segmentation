"""Index BRISC 2025 and write fixed train/val/test split CSVs.

Usage (from the project root):
    python -m src.prepare_data                       # uses data/raw
    python -m src.prepare_data --data-root /path/to/unzipped/brisc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data_index import build_healthy_index, build_index, read_mask, split_train_val
from src.utils import resolve_path, save_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out", default="data/splits")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inspect-n", type=int, default=50, help="images to inspect for size/mask checks")
    return p.parse_args()


def inspect_sample(df: pd.DataFrame, root: Path, n: int, seed: int) -> dict:
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    shapes, mask_values, empty = set(), set(), 0
    for _, row in sample.iterrows():
        img = cv2.imread(str(root / row["image"]), cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        shapes.add(tuple(img.shape))
        mask_values.update(np.unique(raw_mask).tolist()[:10])
        empty += int(read_mask(root / row["mask"]).sum() == 0)
    return {
        "inspected": len(sample),
        "image_shapes": sorted(str(s) for s in shapes),
        "raw_mask_values_seen": sorted(mask_values)[:20],
        "empty_masks_in_sample": empty,
    }


def main():
    args = parse_args()
    root = resolve_path(args.data_root)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full_train = build_index(root, "train")
    test = build_index(root, "test")
    train, val, strat = split_train_val(full_train, args.val_fraction, args.seed)
    train["split"], val["split"], test["split"] = "train", "val", "test"

    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(out / f"{name}.csv", index=False)

    # Healthy slices (classification task) -> false-positive evaluation / optional extra training data
    healthy = {}
    for split in ("train", "test"):
        h = build_healthy_index(root, split)
        h["split"] = f"healthy_{split}"
        h.to_csv(out / f"healthy_{split}.csv", index=False)
        healthy[split] = len(h)

    overlap = set(train["stem"]) & set(test["stem"])
    all_df = pd.concat([train, val, test], ignore_index=True)

    print(f"\nData root : {root}")
    print(f"Stratified: {strat}")
    print(f"Sizes     : train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Healthy   : train={healthy['train']}  test={healthy['test']}  (empty masks, from classification task)")
    if overlap:
        print(f"WARNING: {len(overlap)} file stems appear in both train and test!")
    print("\nImages per tumour type:")
    print(pd.crosstab(all_df["tumor_type"], all_df["split"], margins=True))
    print("\nImages per imaging plane:")
    print(pd.crosstab(all_df["plane"], all_df["split"], margins=True))

    checks = inspect_sample(full_train, root, args.inspect_n, args.seed)
    print("\nSample checks:", checks)

    save_json({
        "data_root": str(root),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "stratified_by": strat,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "healthy_sizes": healthy,
        "train_test_stem_overlap": len(overlap),
        "by_tumor_type": pd.crosstab(all_df["tumor_type"], all_df["split"]).to_dict(),
        "by_plane": pd.crosstab(all_df["plane"], all_df["split"]).to_dict(),
        "sample_checks": checks,
    }, out / "summary.json")
    print(f"\nSplit CSVs and summary.json written to {out}")


if __name__ == "__main__":
    main()
