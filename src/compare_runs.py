"""Compare several evaluated runs and write README-ready tables and figures.

Usage (from the project root, after training + `src.evaluate` for each run):
    python -m src.compare_runs --runs runs/unet_scratch runs/unet_resnet34 runs/unet_effb0

Writes to assets/: model_comparison.md/.csv, training_curves.png, dice_by_tumor_type.png,
dice_by_plane.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.analysis import to_markdown  # noqa: E402
from src.utils import resolve_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--eval-name", default="eval", help="evaluation sub-folder to read in each run")
    p.add_argument("--out", default="assets")
    return p.parse_args()


def label_for(model_cfg: dict, include_healthy: bool = False) -> str:
    suffix = " + healthy slices" if include_healthy else ""
    if model_cfg["name"] == "unet_scratch":
        return "U-Net (scratch)" + suffix
    arch = {"smp_unet": "U-Net", "smp_unetplusplus": "U-Net++", "smp_fpn": "FPN"}.get(model_cfg["name"],
                                                                                       model_cfg["name"])
    enc = model_cfg["encoder_name"].replace("tu-", "")
    return f"{arch} + {enc}" + (" (pretrained)" if model_cfg.get("encoder_weights") else "") + suffix


def load_run(run_dir: Path, eval_name: str) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg_path = run_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    metrics_path = run_dir / eval_name / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} missing — run `python -m src.evaluate --run {run_dir}` first")
    return {"dir": run_dir, "summary": summary, "metrics": json.loads(metrics_path.read_text()),
            "history": pd.read_csv(run_dir / "history.csv"),
            "label": label_for(summary["model"], bool(cfg.get("data", {}).get("include_healthy")))}


def grouped_bar(runs: list[dict], key: str, title: str, path: Path) -> None:
    groups = sorted({g for r in runs for g in r["metrics"][key]})
    width = 0.8 / len(runs)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(1.8 * len(groups) + 3, 4))
    for i, r in enumerate(runs):
        vals = [r["metrics"][key].get(g, {}).get("dice", np.nan) for g in groups]
        bars = ax.bar(x + (i - (len(runs) - 1) / 2) * width, vals, width, label=r["label"])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, groups)
    ax.set_ylabel("mean Dice (test)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(resolve_path(r), args.eval_name) for r in args.runs]

    rows = []
    for r in runs:
        m, s = r["metrics"], r["summary"]
        rows.append({
            "Model": r["label"],
            "Params (M)": s["parameters"] / 1e6,
            "Best epoch": s["best_epoch"],
            "Val Dice (tumour)": s.get("best_val_metrics", {}).get("dice_tumor", s["best_val_dice"]),
            "Test Dice": m["overall"]["dice"],
            "Test IoU": m["overall"]["iou"],
            "Precision": m["overall"]["precision"],
            "Recall": m["overall"]["recall"],
            "Healthy FP rate (≥50 px)": m["healthy"].get("fp_rate_largest_ge_50px", float("nan")),
            "ms / image": m["speed"]["ms_per_image_forward"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "model_comparison.csv", index=False)
    md = to_markdown(table)
    (out / "model_comparison.md").write_text(md + "\n")
    print(md)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for r in runs:
        h = r["history"]
        line, = axes[0].plot(h["epoch"], h["train_loss"], label=f"{r['label']} · train")
        axes[0].plot(h["epoch"], h["val_loss"], ls="--", color=line.get_color(), label=f"{r['label']} · val")
        col = "val_dice_tumor" if "val_dice_tumor" in h and h["val_dice_tumor"].notna().any() else "val_dice"
        axes[1].plot(h["epoch"], h[col], label=r["label"])
    axes[0].set_title("Loss (BCE + Dice)")
    axes[1].set_title("Validation Dice (tumour slices)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    grouped_bar(runs, "by_tumor_type", "Test Dice by tumour type", out / "dice_by_tumor_type.png")
    grouped_bar(runs, "by_plane", "Test Dice by imaging plane", out / "dice_by_plane.png")
    print(f"\nFigures and tables written to {out}")


if __name__ == "__main__":
    main()
