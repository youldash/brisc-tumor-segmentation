"""Evaluation helpers that do not need PyTorch: post-processing, grouped tables, figures."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.metrics import summarize

# Error-map colours (RGB)
TP_COLOR = (40, 200, 70)    # correctly predicted tumour
FP_COLOR = (230, 50, 50)    # predicted, but not tumour
FN_COLOR = (60, 130, 255)   # tumour that was missed


def component_areas(mask: np.ndarray) -> np.ndarray:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0, dtype=np.int64)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels."""
    if min_area <= 0 or not mask.any():
        return mask.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def add_size_buckets(df: pd.DataFrame, col: str = "gt_area_frac") -> tuple[pd.DataFrame, list[float]]:
    """Tertiles of tumour size (small / medium / large), computed on this split."""
    df = df.copy()
    df["size_bucket"], edges = pd.qcut(df[col], 3, labels=["small", "medium", "large"],
                                       retbins=True, duplicates="drop")
    return df, [float(e) for e in edges]


def overall_metrics(df: pd.DataFrame) -> dict:
    counts = df[["tp", "fp", "fn", "tn"]].to_numpy()
    out = summarize(counts)
    out["dice_median"] = float(df["dice"].median())
    out["dice_std"] = float(df["dice"].std(ddof=0))
    out["n"] = int(len(df))
    return out


def grouped_metrics(df: pd.DataFrame, by: str) -> dict:
    return {str(k): overall_metrics(g) for k, g in df.groupby(by, observed=True)}


def healthy_metrics(df: pd.DataFrame, area_thresholds=(1, 50, 200)) -> dict:
    """False-positive behaviour on slices without a tumour (areas in native-resolution pixels)."""
    if df.empty:
        return {"n": 0}
    out = {"n": int(len(df)),
           "mean_fp_area_frac": float(df["pred_area_frac"].mean()),
           "specificity_pixel": float(df["tn"].sum() / max(df["tn"].sum() + df["fp"].sum(), 1))}
    for t in area_thresholds:
        out[f"fp_rate_largest_ge_{t}px"] = float((df["largest_component_px"] >= t).mean())
    return out


def metrics_table(groups: dict, first_col: str) -> pd.DataFrame:
    rows = [{first_col: k, "n": v["n"], "Dice (mean)": v["dice"], "Dice (median)": v["dice_median"],
             "IoU": v["iou"], "Precision": v["precision"], "Recall": v["recall"]}
            for k, v in groups.items()]
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal Markdown table writer (avoids the optional 'tabulate' dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def contour_overlay(image: np.ndarray, mask: np.ndarray, color=(255, 215, 0), thickness: int = 2) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def fill_overlay(image: np.ndarray, mask: np.ndarray, color=(230, 50, 50), alpha: float = 0.4) -> np.ndarray:
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def error_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Green = true positive, red = false positive, blue = false negative."""
    out = image.astype(np.float32).copy()
    pred, gt = pred.astype(bool), gt.astype(bool)
    for m, c in [(pred & gt, TP_COLOR), (pred & ~gt, FP_COLOR), (~pred & gt, FN_COLOR)]:
        out[m] = (1 - alpha) * out[m] + alpha * np.array(c, np.float32)
    return out.astype(np.uint8)


def plot_gallery(items: list[dict], title: str, path: str | Path) -> None:
    """items: dicts with image, gt, pred, prob, caption. One row per item, four panels."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not items:
        return
    fig, axes = plt.subplots(len(items), 4, figsize=(13, 3.3 * len(items)), squeeze=False)
    heads = ["MRI slice", "Ground truth", "Prediction (probability)", "Errors"]
    for i, it in enumerate(items):
        panels = [it["image"], fill_overlay(it["image"], it["gt"], (255, 200, 0)),
                  None, error_overlay(it["image"], it["pred"], it["gt"])]
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j == 2:
                ax.imshow(it["image"])
                ax.imshow(it["prob"], cmap="magma", alpha=0.55, vmin=0, vmax=1)
            else:
                ax.imshow(panels[j])
            if i == 0:
                ax.set_title(heads[j], fontsize=11)
        axes[i, 0].text(0.02, 0.98, it["caption"], transform=axes[i, 0].transAxes, va="top",
                        fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.6, pad=2))
    legend = [Patch(color=np.array(c) / 255, label=l) for c, l in
              [(TP_COLOR, "true positive"), (FP_COLOR, "false positive"), (FN_COLOR, "missed tumour")]]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
