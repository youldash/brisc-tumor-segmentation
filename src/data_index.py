"""Dataset discovery and indexing for BRISC 2025 (no PyTorch dependency).

The Kaggle archive layout is discovered automatically: we look for sibling
``images/`` and ``masks/`` folders whose parent is named after the split
(``train`` / ``test``), preferring paths that contain "segmentation".
Image/mask pairs are matched by file stem. Tumour type and imaging plane are
parsed from filename tokens (e.g. ``..._gl_ax_...`` -> glioma, axial) and
fall back to "unknown" if the naming differs.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TUMOR_CODES = {
    "gl": "glioma", "glioma": "glioma",
    "me": "meningioma", "meningioma": "meningioma",
    "pi": "pituitary", "pituitary": "pituitary",
    "no": "no_tumor", "nt": "no_tumor", "notumor": "no_tumor",
}
PLANE_CODES = {
    "ax": "axial", "axial": "axial",
    "co": "coronal", "coronal": "coronal",
    "sa": "sagittal", "sagittal": "sagittal",
}


def parse_stem(stem: str) -> tuple[str, str]:
    tokens = re.split(r"[_\-\s.]+", stem.lower())
    tumor = next((TUMOR_CODES[t] for t in tokens if t in TUMOR_CODES), "unknown")
    plane = next((PLANE_CODES[t] for t in tokens if t in PLANE_CODES), "unknown")
    return tumor, plane


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_split_dirs(root: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(root)
    candidates = []
    for img_dir in root.rglob("*"):
        if not img_dir.is_dir() or img_dir.name.lower() != "images":
            continue
        mask_dir = img_dir.parent / "masks"
        if mask_dir.is_dir() and img_dir.parent.name.lower() == split.lower():
            candidates.append((img_dir, mask_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No '<{split}>/images' + '<{split}>/masks' folders found under {root}. "
            "Check that the Kaggle archive was unzipped there (see scripts/download_data.sh)."
        )
    candidates.sort(key=lambda c: ("segment" not in str(c[0]).lower(), len(c[0].parts)))
    if len(candidates) > 1:
        print(f"[data_index] Several candidate '{split}' folders found; using {candidates[0][0].parent}")
    return candidates[0]


def build_index(root: str | Path, split: str) -> pd.DataFrame:
    """Return one row per image/mask pair. Paths are stored relative to ``root``."""
    root = Path(root)
    img_dir, mask_dir = find_split_dirs(root, split)
    masks: dict[str, Path] = {}
    for m in _image_files(mask_dir):
        if m.stem in masks:
            print(f"[data_index] Duplicate mask stem ignored: {m.name}")
            continue
        masks[m.stem] = m

    rows, missing = [], []
    for img in _image_files(img_dir):
        mask = masks.get(img.stem)
        if mask is None:
            missing.append(img.name)
            continue
        tumor, plane = parse_stem(img.stem)
        rows.append({
            "image": img.relative_to(root).as_posix(),
            "mask": mask.relative_to(root).as_posix(),
            "stem": img.stem,
            "tumor_type": tumor,
            "plane": plane,
            "source_split": split,
        })
    if missing:
        print(f"[data_index] {len(missing)} '{split}' images have no mask and were skipped "
              f"(e.g. {missing[:3]})")
    if not rows:
        raise RuntimeError(f"No image/mask pairs found for split '{split}' in {img_dir}")
    return pd.DataFrame(rows)


def build_healthy_index(root: str | Path, split: str) -> pd.DataFrame:
    """Index healthy slices (``.../<split>/no_tumor/``, from the classification task).

    The segmentation task contains tumour images only, so these slices are used to
    measure false positives. Their ground-truth mask is empty (``mask`` column is blank).
    """
    root = Path(root)
    dirs = [d for d in root.rglob("*")
            if d.is_dir() and d.name.lower() in {"no_tumor", "notumor", "no_tumour"}
            and d.parent.name.lower() == split.lower()]
    if not dirs:
        return pd.DataFrame(columns=["image", "mask", "stem", "tumor_type", "plane", "source_split"])
    dirs.sort(key=lambda d: len(d.parts))
    rows = []
    for img in _image_files(dirs[0]):
        _, plane = parse_stem(img.stem)
        rows.append({
            "image": img.relative_to(root).as_posix(),
            "mask": "",
            "stem": img.stem,
            "tumor_type": "no_tumor",
            "plane": plane,
            "source_split": split,
        })
    return pd.DataFrame(rows)


def split_train_val(df: pd.DataFrame, val_fraction: float, seed: int):
    """Stratified split by tumour type x plane, with graceful fallbacks."""
    from sklearn.model_selection import train_test_split

    options = [
        ("tumor_type x plane", df["tumor_type"] + "|" + df["plane"]),
        ("tumor_type", df["tumor_type"]),
        ("none", None),
    ]
    for name, strata in options:
        if strata is not None and strata.value_counts().min() < 2:
            continue
        try:
            tr, va = train_test_split(df, test_size=val_fraction, random_state=seed, stratify=strata)
            return tr.reset_index(drop=True), va.reset_index(drop=True), name
        except ValueError:
            continue
    raise RuntimeError("Could not split the training data.")


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Read an MRI slice as uint8 grayscale and replicate it to 3 channels (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask and binarise it to {0, 1} (handles 0/1 and 0/255 encodings)."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    threshold = 0 if m.max() <= 1 else 127
    return (m > threshold).astype(np.uint8)


def has_mask(mask_value) -> bool:
    return isinstance(mask_value, str) and mask_value.strip() != ""


def load_pair(root: str | Path, row) -> tuple[np.ndarray, np.ndarray]:
    """Load (image RGB uint8, mask {0,1} uint8) for an index row; blank mask -> all zeros."""
    root = Path(root)
    image = read_image_rgb(root / row["image"])
    if has_mask(row["mask"]):
        mask = read_mask(root / row["mask"])
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros(image.shape[:2], np.uint8)
    return image, mask


def load_train_val_frames(splits_dir: str | Path, include_healthy: bool = False,
                          healthy_val_fraction: float = 0.15, seed: int = 42):
    """Read train/val CSVs; optionally add healthy slices (empty masks), split with the same seed."""
    splits_dir = Path(splits_dir)
    for name in ("train.csv", "val.csv"):
        if not (splits_dir / name).exists():
            raise FileNotFoundError(f"{splits_dir / name} not found. Run `python -m src.prepare_data` first.")
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    if include_healthy:
        path = splits_dir / "healthy_train.csv"
        healthy = pd.read_csv(path).fillna({"mask": ""}) if path.exists() else pd.DataFrame()
        if healthy.empty:
            raise FileNotFoundError(f"{path} is missing or empty. Re-run `python -m src.prepare_data`.")
        h_train, h_val, _ = split_train_val(healthy, healthy_val_fraction, seed)
        h_train["split"], h_val["split"] = "healthy_train", "healthy_val"
        train_df = pd.concat([train_df, h_train], ignore_index=True)
        val_df = pd.concat([val_df, h_val], ignore_index=True)
    return train_df.fillna({"mask": ""}), val_df.fillna({"mask": ""})
