"""Collect all results of the final model and write them into README.md.

Usage (from the project root, after evaluation / experiments / deployment benchmark):
    python -m src.report

* copies galleries and figures to assets/
* draws assets/training_curves.png and assets/results_breakdown.png
* writes assets/RESULTS.md and replaces the block between
  <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md
Missing inputs are skipped, so it can be re-run at any stage.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import to_markdown  # noqa: E402
from src.utils import PROJECT_ROOT, resolve_path  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
EXPERIMENTS = [
    ("eval", "Baseline (threshold 0.5)"),
    ("eval_min200", "+ remove regions < 200 px"),
    ("eval_tta", "+ flip TTA"),
    ("eval_tta_min200", "+ flip TTA + remove < 200 px"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/unet_resnet34")
    p.add_argument("--assets", default="assets")
    p.add_argument("--readme", default="README.md")
    return p.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f} %"


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def group_table(groups: dict, name: str) -> str:
    df = pd.DataFrame([{name: k, "n": v["n"], "Dice (mean)": v["dice"], "Dice (median)": v["dice_median"],
                        "IoU": v["iou"], "Precision": v["precision"], "Recall": v["recall"]}
                       for k, v in groups.items()])
    return to_markdown(df)


def plot_training(history: pd.DataFrame, best_epoch: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_yscale("log"); axes[0].set_title("Loss (BCE + Dice)")
    axes[1].plot(history["epoch"], history["val_dice"], label="Dice")
    axes[1].plot(history["epoch"], history["val_iou"], label="IoU")
    axes[1].set_title("Validation overlap (256 px)")
    axes[2].plot(history["epoch"], history["lr"], color="tab:purple", label="learning rate (decoder)")
    ax2 = axes[2].twinx()
    ax2.bar(history["epoch"], history["epoch_time_s"], alpha=0.25, color="grey", label="epoch time (s)")
    ax2.set_ylabel("seconds per epoch"); ax2.grid(False)
    axes[2].set_zorder(ax2.get_zorder() + 1); axes[2].patch.set_visible(False)
    axes[2].set_yscale("log"); axes[2].set_title("Schedule and epoch duration")
    for ax in axes:
        ax.axvline(best_epoch, color="green", ls=":", lw=1.2, label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_breakdown(metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key, title in zip(axes, ["by_tumor_type", "by_plane", "by_tumor_size"],
                              ["Tumour type", "Imaging plane", "Tumour size (tertiles)"]):
        groups = metrics[key]
        names = list(groups)
        x = np.arange(len(names))
        mean = [groups[g]["dice"] for g in names]
        med = [groups[g]["dice_median"] for g in names]
        b1 = ax.bar(x - 0.2, mean, 0.4, label="mean Dice")
        b2 = ax.bar(x + 0.2, med, 0.4, label="median Dice")
        ax.bar_label(b1, fmt="%.3f", fontsize=8)
        ax.bar_label(b2, fmt="%.3f", fontsize=8)
        ax.set_xticks(x, [f"{g}\n(n={groups[g]['n']})" for g in names])
        ax.set_ylim(0.5, 1.02); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left")
    fig.suptitle("Test-set Dice of U-Net + ResNet34 (860 slices, native resolution)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir, assets = resolve_path(args.run), resolve_path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    readme = resolve_path(args.readme)
    parts, findings = [], []

    summary = read_json(run_dir / "summary.json")
    history = pd.read_csv(run_dir / "history.csv") if (run_dir / "history.csv").exists() else None
    metrics = read_json(run_dir / "eval" / "metrics.json")
    split_summary = read_json(resolve_path("data/splits/summary.json"))

    # ---------------- training ----------------
    if summary and history is not None:
        plot_training(history, summary["best_epoch"], assets / "training_curves.png")
        train_min = history["epoch_time_s"].sum() / 60
        parts += [
            "### Training", "",
            to_markdown(pd.DataFrame([{
                "Parameters": f"{summary['parameters'] / 1e6:.1f} M", "Epochs": len(history),
                "Best epoch": summary["best_epoch"], "Best val Dice": summary["best_val_dice"],
                "Val IoU @best": float(history.loc[history["val_dice"].idxmax(), "val_iou"]),
                "Training time": f"{train_min:.0f} min ({summary['device']})",
            }])), "",
            "![Training curves](assets/training_curves.png)", "",
        ]

    # ---------------- test results ----------------
    if metrics:
        o = metrics["overall"]
        plot_breakdown(metrics, assets / "results_breakdown.png")
        parts += [
            "### Test set (860 tumour slices, never used for training or model selection)", "",
            to_markdown(pd.DataFrame([{"Dice (mean)": o["dice"], "Dice (median)": o["dice_median"], "IoU": o["iou"],
                                       "Precision": o["precision"], "Recall": o["recall"],
                                       "Global Dice": o["global_dice"]}])), "",
            "*Global Dice pools all pixels of the test set; the other scores are averaged per slice.*", "",
            f"Model speed: {metrics['speed']['ms_per_image_forward']:.1f} ms per slice "
            f"({metrics['speed']['device']}, batch {metrics['speed'].get('batch_size', '—')}).", "",
            "![Dice breakdown](assets/results_breakdown.png)", "",
            "<details><summary>Breakdown tables</summary>", "",
            group_table(metrics["by_tumor_type"], "Tumour type"), "",
            group_table(metrics["by_plane"], "Plane"), "",
            group_table(metrics["by_tumor_size"], "Size"), "",
            "Size tertile edges (fraction of slice area): "
            + ", ".join(f"{e:.4f}" for e in metrics["size_bucket_edges_area_frac"]), "",
            "</details>", "",
        ]
        types = metrics["by_tumor_type"]
        worst_t, best_t = min(types, key=lambda k: types[k]["dice"]), max(types, key=lambda k: types[k]["dice"])
        if summary:
            gap = summary["best_val_dice"] - o["dice"]
            verdict = ("close to the best validation Dice ({:.3f}) → the checkpoint generalises"
                       if gap < 0.03 else "clearly below the best validation Dice ({:.3f}) → check for over-fitting")
            first = (f"Mean test Dice **{o['dice']:.3f}** (median {o['dice_median']:.3f}), "
                     + verdict.format(summary["best_val_dice"])
                     + ("; the median above the mean shows that a minority of hard slices pulls the average down."
                        if o["dice_median"] - o["dice"] > 0.02 else "."))
        else:
            first = f"Mean test Dice **{o['dice']:.3f}**."
        findings += [
            first,
            f"**{worst_t.capitalize()}** is the hardest class (Dice {types[worst_t]['dice']:.3f}) and "
            f"**{best_t}** the easiest ({types[best_t]['dice']:.3f}).",
            f"Small tumours are harder (Dice {metrics['by_tumor_size']['small']['dice']:.3f}) than large ones "
            f"({metrics['by_tumor_size']['large']['dice']:.3f}).",
        ]

        h = metrics.get("healthy", {})
        if h.get("n"):
            parts += [
                "### Healthy slices (false positives)", "",
                f"The segmentation split contains tumour slices only, so the model was never shown a healthy brain. "
                f"On the {h['n']} healthy test slices (from the classification split) it predicts a region of:", "",
                to_markdown(pd.DataFrame([{"≥ 1 px": pct(h["fp_rate_largest_ge_1px"]),
                                           "≥ 50 px": pct(h["fp_rate_largest_ge_50px"]),
                                           "≥ 200 px": pct(h["fp_rate_largest_ge_200px"])}])), "",
            ]
            findings.append(f"On healthy slices the model still predicts a tumour region (≥ 50 px) in "
                            f"**{pct(h['fp_rate_largest_ge_50px'])}** of cases — it has learned that every slice "
                            "contains a tumour. This is the main limitation (see Future Improvements).")

    # ---------------- improvement experiments ----------------
    rows = []
    for folder, name in EXPERIMENTS:
        m = read_json(run_dir / folder / "metrics.json")
        if m:
            rows.append((name, m))
    for thr_dir in sorted(run_dir.glob("eval_thr*")):
        m = read_json(thr_dir / "metrics.json")
        if m:
            rows.append((f"threshold {m['settings']['threshold']:.2f} (tuned on validation)", m))
    if len(rows) > 1:
        exp = pd.DataFrame([{
            "Variant": n, "Dice (mean)": m["overall"]["dice"], "Dice (median)": m["overall"]["dice_median"],
            "Precision": m["overall"]["precision"], "Recall": m["overall"]["recall"],
            "Healthy FP ≥ 50 px": pct(m["healthy"].get("fp_rate_largest_ge_50px")),
            "ms / slice": f"{m['speed']['ms_per_image_forward']:.1f}",
        } for n, m in rows])
        (assets / "postprocessing.md").write_text(to_markdown(exp) + "\n")
        parts += ["### Improvement experiments (inference-time)", "",
                  "Post-processing and test-time augmentation (TTA: average with the horizontally flipped slice) "
                  "applied to the same trained model.", "", to_markdown(exp), ""]
        base = rows[0][1]
        best_name, best_m = max(rows, key=lambda r: r[1]["overall"]["dice"])
        findings.append(
            f"Inference-time tricks change test Dice by at most "
            f"{max(abs(m['overall']['dice'] - base['overall']['dice']) for _, m in rows):.3f} "
            f"(best: {best_name}, {best_m['overall']['dice']:.3f}); healthy-slice false positives stay between "
            f"{pct(min(m['healthy'].get('fp_rate_largest_ge_50px', 1) for _, m in rows))} and "
            f"{pct(max(m['healthy'].get('fp_rate_largest_ge_50px', 0) for _, m in rows))} → they cannot replace "
            "training on healthy slices.")
    sweep_path = run_dir / "threshold_sweep_val.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(sweep["threshold"], sweep["dice_tumor"], marker=".", label="val Dice (tumour slices)")
        ax.plot(sweep["threshold"], sweep["dice"], marker=".", label="val Dice (tumour + healthy)")
        ax2 = ax.twinx()
        ax2.plot(sweep["threshold"], 100 * sweep["healthy_fp_rate"], color="tab:red", ls="--",
                 label="healthy slices with FP (%)")
        ax2.set_ylabel("healthy slices with any FP (%)"); ax2.grid(False)
        ax.axvline(0.5, color="grey", ls=":")
        ax.set_xlabel("probability threshold"); ax.set_ylabel("Dice"); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
        ax.set_title("Threshold sweep on the validation set")
        fig.tight_layout(); fig.savefig(assets / "threshold_sweep.png", dpi=150); plt.close(fig)
        parts += ["![Threshold sweep](assets/threshold_sweep.png)", ""]

    # ---------------- qualitative ----------------
    galleries = [("gallery_best", "Successful predictions (highest Dice)"),
                 ("gallery_typical", "Typical predictions (near median Dice)"),
                 ("gallery_worst", "Failure cases (lowest Dice)"),
                 ("gallery_healthy_fp", "Failure cases on healthy slices (false positives)")]
    shown = []
    for g, title in galleries:
        src = run_dir / "eval" / f"{g}.png"
        if src.exists():
            shutil.copy(src, assets / f"{g}.png")
            shown += [f"**{title}**", "", f"![{title}](assets/{g}.png)", ""]
    if shown:
        parts += ["### Qualitative results", "",
                  "Columns: MRI slice · expert mask · predicted probability · error map "
                  "(green = correct, red = false positive, blue = missed tumour).", "", *shown]

    # ---------------- deployment ----------------
    bench = read_json(assets / "deployment_benchmark.json")
    card = read_json(resolve_path("models/model_card.json"))
    if bench:
        parts += ["### Deployment benchmark (ONNX Runtime)", "",
                  f"{bench['n_test']} test + {bench['n_healthy']} healthy slices, batch size 1 for ONNX, "
                  f"threshold {bench['threshold']}.", "",
                  (assets / "deployment_benchmark.md").read_text().strip(), ""]
        onnx_rows = {r["variant"]: r for r in bench["results"]}
        fp32 = next((r for k, r in onnx_rows.items() if "FP32 (CPU)" in k), None)
        int8 = next((r for k, r in onnx_rows.items() if "INT8" in k), None)
        if fp32 and int8:
            findings.append(
                f"INT8 quantization shrinks the model {fp32['size_mb'] / int8['size_mb']:.1f}× "
                f"({fp32['size_mb']:.1f} → {int8['size_mb']:.1f} MB) with a test-Dice change of "
                f"{int8['test_dice'] - fp32['test_dice']:+.4f}; CPU time per slice "
                f"{fp32['model_ms_per_image']:.1f} → {int8['model_ms_per_image']:.1f} ms"
                + (" (faster)." if int8["model_ms_per_image"] < fp32["model_ms_per_image"]
                   else " (no speed-up on this CPU: the gain is mainly size/memory)."))
    if card and card.get("parity_check"):
        pc = card["parity_check"]
        parts += [f"ONNX export verified against PyTorch on {pc['n_images']} test slices: max probability "
                  f"difference {pc['max_abs_prob_diff']:.1e}, mask agreement Dice {pc['mask_agreement_dice']:.4f}.", ""]
    if (assets / "app_screenshot.png").exists():
        parts += ["![Gradio demo](assets/app_screenshot.png)", ""]

    if split_summary:
        checks = split_summary.get("sample_checks", {})
        shapes = checks.get("image_shapes", [])
        values = checks.get("raw_mask_values_seen", [])
        notes = []
        if len(shapes) > 1:
            notes.append("image sizes are not uniform (" + ", ".join(shapes) + ")")
        if any(0 < v < 255 for v in values):
            notes.append("masks contain compression-like grey values")
        if notes:
            findings.append("Data check: " + " and ".join(notes) + "; the pipeline resizes every slice, binarises "
                            "masks at 127 and scores predictions at each slice's native size.")

    if findings:
        parts += ["### Key findings", "", *[f"- {f}" for f in findings], ""]

    block = "\n".join(parts).strip()
    (assets / "RESULTS.md").write_text("# Results\n\n" + block + "\n")
    if readme.exists():
        text = readme.read_text()
        if START in text and END in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                          lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
            readme.write_text(text)
            print(f"README results section updated ({len(block.splitlines())} lines)")
        else:
            print("README has no results markers; see assets/RESULTS.md")
    print(f"Assets written to {assets.relative_to(PROJECT_ROOT) if assets.is_relative_to(PROJECT_ROOT) else assets}")


if __name__ == "__main__":
    main()
