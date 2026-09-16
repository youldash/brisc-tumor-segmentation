"""Create a tiny synthetic dataset that mimics the BRISC folder layout.

It is only used to smoke-test the pipeline end-to-end before the real data
is downloaded or before a long training run:

    python scripts/make_synthetic_data.py
    python -m src.prepare_data --data-root data/synthetic --out data/splits_synthetic
    python -m src.train --config configs/unet_scratch.yaml --smoke \
        --data-root data/synthetic --splits-dir data/splits_synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TUMORS = ["gl", "me", "pi"]  # the real segmentation task has tumour images only
PLANES = ["ax", "co", "sa"]


def make_pair(rng: np.random.Generator, size: int, tumor: str):
    img = np.zeros((size, size), np.uint8)
    c = size // 2
    cv2.ellipse(img, (c, c), (int(size * 0.38), int(size * 0.45)), 0, 0, 360, 90, -1)
    img = cv2.add(img, rng.integers(0, 40, img.shape, dtype=np.uint8))
    mask = np.zeros_like(img)
    if tumor != "no":
        center = (int(rng.integers(size * 0.3, size * 0.7)), int(rng.integers(size * 0.3, size * 0.7)))
        axes = (int(rng.integers(size * 0.04, size * 0.12)), int(rng.integers(size * 0.04, size * 0.12)))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(img, center, axes, angle, 0, 360, int(rng.integers(170, 240)), -1)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic/brisc2025/segmentation_task")
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            tumor, plane = TUMORS[i % len(TUMORS)], PLANES[(i // len(TUMORS)) % len(PLANES)]
            img, mask = make_pair(rng, args.size, tumor)
            stem = f"brisc2025_{split}_{i + 1:05d}_{tumor}_{plane}_t1"
            cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), img)
            cv2.imwrite(str(out / split / "masks" / f"{stem}.png"), mask)
        # healthy slices live in the classification task, as in the real archive
        healthy_dir = out.parent / "classification_task" / split / "no_tumor"
        healthy_dir.mkdir(parents=True, exist_ok=True)
        for i in range(max(4, n // 6)):
            img, _ = make_pair(rng, args.size, "no")
            cv2.imwrite(str(healthy_dir / f"brisc2025_{split}_{i + 1:05d}_no_{PLANES[i % 3]}_t1.jpg"), img)
    print(f"Synthetic data written to {out.parent}")


if __name__ == "__main__":
    main()
