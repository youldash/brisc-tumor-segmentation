"""Segmentation metrics (NumPy) based on per-image confusion counts.

Convention for images whose ground truth AND prediction are both empty
(e.g. healthy slices): Dice = IoU = 1.
"""
from __future__ import annotations

import numpy as np


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred/target: boolean arrays (N, H, W). Returns int array (N, 4) = TP, FP, FN, TN."""
    pred = pred.reshape(pred.shape[0], -1).astype(bool)
    target = target.reshape(target.shape[0], -1).astype(bool)
    tp = (pred & target).sum(1)
    fp = (pred & ~target).sum(1)
    fn = (~pred & target).sum(1)
    tn = (~pred & ~target).sum(1)
    return np.stack([tp, fp, fn, tn], axis=1).astype(np.int64)


def per_image_scores(counts: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, fn, _ = counts.T.astype(np.float64)
    empty = (tp + fp + fn) == 0
    dice = np.where(empty, 1.0, 2 * tp / np.maximum(2 * tp + fp + fn, 1))
    iou = np.where(empty, 1.0, tp / np.maximum(tp + fp + fn, 1))
    precision = np.where(tp + fp == 0, np.where(fn == 0, 1.0, 0.0), tp / np.maximum(tp + fp, 1))
    recall = np.where(tp + fn == 0, 1.0, tp / np.maximum(tp + fn, 1))
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def summarize(counts: np.ndarray) -> dict[str, float]:
    """Mean per-image scores plus dataset-level (pixel-pooled) Dice/IoU."""
    scores = per_image_scores(counts)
    tp, fp, fn, _ = counts.sum(0).astype(np.float64)
    out = {k: float(v.mean()) for k, v in scores.items()}
    out["global_dice"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    out["global_iou"] = float(tp / max(tp + fp + fn, 1))
    return out


def validation_summary(counts: np.ndarray) -> dict[str, float]:
    """Metrics over all images, plus tumour-only Dice and the healthy-slice false-positive rate.

    Images whose ground-truth mask is empty are treated as healthy slices.
    """
    out = summarize(counts)
    has_tumor = (counts[:, 0] + counts[:, 2]) > 0
    out["dice_tumor"] = summarize(counts[has_tumor])["dice"] if has_tumor.any() else float("nan")
    out["healthy_fp_rate"] = float((counts[~has_tumor, 1] > 0).mean()) if (~has_tumor).any() else float("nan")
    out["n_tumor"], out["n_healthy"] = int(has_tumor.sum()), int((~has_tumor).sum())
    return out
